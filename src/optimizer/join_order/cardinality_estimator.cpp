#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/optimizer/join_order/join_node.hpp"
#include "duckdb/optimizer/join_order/query_graph_manager.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/storage/data_table.hpp"

#include "yyjson.hpp"
#include <nlohmann/json.hpp>

#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <vector>

using QueryStatsMap = std::unordered_map<std::string, long>;
const std::string DEFAULT_ACTUAL_CARDINALITY_FILE_PATH = "/Users/Aarry/Desktop/15689/duckdb15689/actual_cardinality.json";
const std::string DEFAULT_ESTIMATED_CARDINALITY_FILE_PATH = "/Users/Aarry/Desktop/15689/duckdb15689/cardinality_log.txt";
const char *ACTUAL_CARDINALITY_ENV_VAR = "DUCKDB_ACTUAL_CARDINALITY_JSON";
const char *ESTIMATED_CARDINALITY_ENV_VAR = "DUCKDB_CARDINALITY_LOG";
const char *PLAN_FINGERPRINT_ENV_VAR = "DUCKDB_FEEDBACK_PLAN_FINGERPRINT";
// Cardinality injection fork: JSON keys match LOGICAL_JOIN expression strings; optional
// DUCKDB_DEBUG_CARD_ESTIMATE_NDJSON / DUCKDB_DUMP_INJECTION_STATS for diagnostics.
// Baseline optimizer snapshots for comparison builds live under legacy_feedback/ (README).
static const char *DEBUG_CARD_ESTIMATE_NDJSON_ENV = "DUCKDB_DEBUG_CARD_ESTIMATE_NDJSON";
static const char *DEBUG_CARD_ESTIMATE_LOG_ENV = "DUCKDB_DEBUG_CARD_ESTIMATE_LOG";
static const char *DEBUG_FOCUS_JOIN_SUBSTRING_ENV = "DUCKDB_DEBUG_FOCUS_JOIN_SUBSTRING";
static const char *DUMP_INJECTION_STATS_ENV = "DUCKDB_DUMP_INJECTION_STATS";

namespace duckdb {

//! Oracle JSON injection: count applies, primary vs PLANFP-key lookup, histogram by rounded value.
//! Set DUCKDB_DUMP_INJECTION_STATS=1 to print a summary to stderr at process exit.
static std::atomic<uint64_t> g_injection_apply_seq {0};
static std::atomic<uint64_t> g_injection_apply_total {0};
static std::atomic<uint64_t> g_injection_primary_hits {0};
static std::atomic<uint64_t> g_injection_fallback_hits {0};
static std::mutex g_injection_hist_mutex;
static unordered_map<int64_t, uint64_t> g_injection_value_histogram;

static void DumpInjectionStatsAtExit() {
	const char *e = std::getenv(DUMP_INJECTION_STATS_ENV);
	if (!e || e[0] == '\0' || string(e) != "1") {
		return;
	}
	const uint64_t total = g_injection_apply_total.load(std::memory_order_relaxed);
	if (total == 0) {
		return;
	}
	std::cerr << "[DuckDB injection stats] applies=" << total
	          << " primary_key_hits=" << g_injection_primary_hits.load(std::memory_order_relaxed)
	          << " fallback_logical_join_hits=" << g_injection_fallback_hits.load(std::memory_order_relaxed)
	          << std::endl;
	std::lock_guard<std::mutex> guard(g_injection_hist_mutex);
	for (const auto &kv : g_injection_value_histogram) {
		std::cerr << "  injected_value=" << kv.first << " apply_count=" << kv.second << std::endl;
	}
}

static void RegisterInjectionStatsDumpAtExit() {
	static std::once_flag once;
	std::call_once(once, [] { std::atexit(DumpInjectionStatsAtExit); });
}

static void RecordInjectionApply(double value, bool primary_key_match) {
	g_injection_apply_total.fetch_add(1, std::memory_order_relaxed);
	if (primary_key_match) {
		g_injection_primary_hits.fetch_add(1, std::memory_order_relaxed);
	} else {
		g_injection_fallback_hits.fetch_add(1, std::memory_order_relaxed);
	}
	const int64_t key = static_cast<int64_t>(std::llround(value));
	{
		std::lock_guard<std::mutex> guard(g_injection_hist_mutex);
		g_injection_value_histogram[key]++;
	}
	RegisterInjectionStatsDumpAtExit();
}

using namespace duckdb_yyjson; // NOLINT
using json = nlohmann::json;

struct CovMultiMissingSampleRow {
	idx_t filter_index;
	bool from_residual;
	string rel_set;
	string filter;
};

static unordered_map<string, double> LoadCardinalityJsonFromPath(const string &path) {
	unordered_map<string, double> cardinality_map;
	yyjson_doc *doc = yyjson_read_file(path.c_str(), 0, nullptr, nullptr);
	if (!doc) {
		std::cerr << "Error: Could not read or parse JSON at " << path << std::endl;
		return cardinality_map;
	}
	yyjson_val *root = yyjson_doc_get_root(doc);
	if (!root || !yyjson_is_obj(root)) {
		std::cerr << "Error: JSON root is not an object at " << path << std::endl;
		yyjson_doc_free(doc);
		return cardinality_map;
	}
	size_t oi, omax;
	yyjson_val *key, *val;
	yyjson_obj_foreach(root, oi, omax, key, val) {
		const char *k = yyjson_get_str(key);
		if (!k) {
			continue;
		}
		cardinality_map.emplace(string(k), yyjson_get_num(val));
	}
	yyjson_doc_free(doc);
	std::cout << "Successfully loaded " << cardinality_map.size() << " entries from JSON." << std::endl;
	return cardinality_map;
}

static string MutJsonWriteLineAndFree(yyjson_mut_doc *doc, yyjson_mut_val *root) {
	yyjson_mut_doc_set_root(doc, root);
	char *buf =
	    yyjson_mut_val_write_opts(root, YYJSON_WRITE_ALLOW_INF_AND_NAN, nullptr, nullptr, nullptr);
	if (!buf) {
		yyjson_mut_doc_free(doc);
		return string();
	}
	string line(buf);
	std::free(buf);
	yyjson_mut_doc_free(doc);
	return line;
}

static string GetPathFromEnvOrDefault(const char *env_var_name, const string &default_path) {
	const char *env_value = std::getenv(env_var_name);
	if (env_value && env_value[0] != '\0') {
		return string(env_value);
	}
	return default_path;
}

static string EscapeRelBindingToken(const string &input) {
	string escaped;
	escaped.reserve(input.size() + 8);
	for (auto ch : input) {
		if (ch == '\\' || ch == '|') {
			escaped += '\\';
		}
		escaped += ch;
	}
	return escaped;
}

static string MakeCardinalityLookupKey(const string &logical_join) {
	const char *plan_fp = std::getenv(PLAN_FINGERPRINT_ENV_VAR);
	if (plan_fp && plan_fp[0] != '\0') {
		return "PLANFP:" + string(plan_fp) + "::" + logical_join;
	}
	return logical_join;
}

//! Strip 'tablename.' prefix from a column name like 'store_sales.ss_item_sk' -> 'ss_item_sk'.
static string StripTablePrefix(const string &column_name) {
	auto dot_pos = column_name.find('.');
	if (dot_pos != string::npos && dot_pos + 1 < column_name.size()) {
		return column_name.substr(dot_pos + 1);
	}
	return column_name;
}

static bool IsSimpleBaseTableName(const string &table_name) {
	return !table_name.empty() && table_name != "[unknown]" && table_name.find(',') == string::npos;
}

static bool BuildAliasedFromTerm(const unordered_map<idx_t, RelationColumnInfo> &relation_column_info, idx_t relation_index,
                                 string &from_term) {
	string tbl_name = "[unknown]";
	string scan_filter;
	auto cit = relation_column_info.find(relation_index);
	if (cit != relation_column_info.end()) {
		tbl_name = cit->second.table_name;
		scan_filter = cit->second.scan_filter_string;
	}
	if (!IsSimpleBaseTableName(tbl_name)) {
		return false;
	}
	if (!scan_filter.empty()) {
		from_term = "(SELECT * FROM " + tbl_name + " WHERE " + scan_filter + ") AS r" + to_string(relation_index);
	} else {
		from_term = tbl_name + " AS r" + to_string(relation_index);
	}
	return true;
}

static bool IsIdentChar(char c) {
	return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static string ReplaceIdentifierToken(const string &input, const string &ident, const string &replacement) {
	if (ident.empty()) {
		return input;
	}
	string out;
	out.reserve(input.size() + 32);
	idx_t i = 0;
	while (i < input.size()) {
		if (i + ident.size() <= input.size() && input.compare(i, ident.size(), ident) == 0) {
			bool left_ok = (i == 0) || !IsIdentChar(input[i - 1]);
			bool right_ok = (i + ident.size() == input.size()) || !IsIdentChar(input[i + ident.size()]);
			if (left_ok && right_ok) {
				out += replacement;
				i += ident.size();
				continue;
			}
		}
		out += input[i];
		i++;
	}
	return out;
}

// The filter was made on top of a logical sample or other projection,
// but no specific columns are referenced. See issue 4978 number 4.
bool CardinalityEstimator::EmptyFilter(FilterInfo &filter_info) {
	if (!filter_info.left_set && !filter_info.right_set) {
		return true;
	}
	return false;
}

unordered_map<string, double> load_cardinality_data() {
	auto actual_cardinality_path =
	    GetPathFromEnvOrDefault(ACTUAL_CARDINALITY_ENV_VAR, DEFAULT_ACTUAL_CARDINALITY_FILE_PATH);
	std::ifstream probe(actual_cardinality_path);
	if (!probe.is_open()) {
		std::cerr << "Error: File not found at " << actual_cardinality_path << std::endl;
		return unordered_map<string, double>();
	}
	probe.close();
	return LoadCardinalityJsonFromPath(actual_cardinality_path);
}

void CardinalityEstimator::AddRelationStats(FilterInfo &filter_info) {
	D_ASSERT(filter_info.set.get().count >= 1);
	for (const RelationsSetToStats &r2tdom : relation_set_stats) {
		auto &i_set = r2tdom.equivalent_relations;
		if (i_set.find(filter_info.left_binding) != i_set.end()) {
			// found an equivalent filter
			return;
		}
	}

	auto key = ColumnBinding(filter_info.left_binding.table_index, filter_info.left_binding.column_index);
	RelationsSetToStats new_r2tdom(column_binding_set_t({key}));

	relation_set_stats.emplace_back(new_r2tdom);
}

bool CardinalityEstimator::SingleColumnFilter(duckdb::FilterInfo &filter_info) {
	if (filter_info.left_set && filter_info.right_set && filter_info.set.get().count > 1) {
		// Both set and are from different relations
		return false;
	}
	if (EmptyFilter(filter_info)) {
		return false;
	}
	if (filter_info.join_type == JoinType::SEMI || filter_info.join_type == JoinType::ANTI) {
		return false;
	}
	return true;
}

void CardinalityEstimator::SetPendingParentSplit(optional_ptr<JoinRelationSet> left_set,
                                                 optional_ptr<JoinRelationSet> right_set) {
	pending_left_split = left_set;
	pending_right_split = right_set;
}

string CardinalityEstimator::MakeJoinSetCacheKey(const JoinRelationSet &new_set) const {
	if (new_set.count <= 1 || !pending_left_split || !pending_right_split) {
		return new_set.ToString();
	}
	string left_split = pending_left_split->ToString();
	string right_split = pending_right_split->ToString();
	if (right_split < left_split) {
		std::swap(left_split, right_split);
	}
	return new_set.ToString() + " CtxParentSplit: [" + left_split + " | " + right_split + "]";
}

vector<idx_t> CardinalityEstimator::DetermineMatchingEquivalentSets(optional_ptr<FilterInfo> filter_info) {
	vector<idx_t> matching_equivalent_sets;
	idx_t equivalent_relation_index = 0;

	for (const RelationsSetToStats &r2tdom : relation_set_stats) {
		auto &i_set = r2tdom.equivalent_relations;
		if (i_set.find(filter_info->left_binding) != i_set.end()) {
			matching_equivalent_sets.push_back(equivalent_relation_index);
		} else if (i_set.find(filter_info->right_binding) != i_set.end()) {
			// don't add both left and right to the matching_equivalent_sets
			// since both left and right get added to that index anyway.
			matching_equivalent_sets.push_back(equivalent_relation_index);
		}
		equivalent_relation_index++;
	}
	return matching_equivalent_sets;
}

void CardinalityEstimator::AddToEquivalenceSets(optional_ptr<FilterInfo> filter_info,
                                                vector<idx_t> matching_equivalent_sets) {
	D_ASSERT(matching_equivalent_sets.size() <= 2);
	if (matching_equivalent_sets.size() > 1) {
		// an equivalence relation is connecting two sets of equivalence relations
		// so push all relations from the second set into the first. Later we will delete
		// the second set.
		for (ColumnBinding i : relation_set_stats.at(matching_equivalent_sets[1]).equivalent_relations) {
			relation_set_stats.at(matching_equivalent_sets[0]).equivalent_relations.insert(i);
		}
		for (auto &column_name : relation_set_stats.at(matching_equivalent_sets[1]).column_names) {
			relation_set_stats.at(matching_equivalent_sets[0]).column_names.push_back(column_name);
		}
		relation_set_stats.at(matching_equivalent_sets[1]).equivalent_relations.clear();
		relation_set_stats.at(matching_equivalent_sets[1]).column_names.clear();
		relation_set_stats.at(matching_equivalent_sets[0]).filters.push_back(filter_info);
		// add all values of one set to the other, delete the empty one
	} else if (matching_equivalent_sets.size() == 1) {
		auto &tdom_i = relation_set_stats.at(matching_equivalent_sets.at(0));
		tdom_i.equivalent_relations.insert(filter_info->left_binding);
		tdom_i.equivalent_relations.insert(filter_info->right_binding);
		tdom_i.filters.push_back(filter_info);
	} else if (matching_equivalent_sets.empty()) {
		column_binding_set_t tmp;
		tmp.insert(filter_info->left_binding);
		tmp.insert(filter_info->right_binding);
		relation_set_stats.emplace_back(tmp);
		relation_set_stats.back().filters.push_back(filter_info);
	}
}

void CardinalityEstimator::InitEquivalentRelations(const vector<unique_ptr<FilterInfo>> &filter_infos) {
	all_filters.clear();
	// For each filter, we fill keep track of the index of the equivalent relation set
	// the left and right relation needs to be added to.
	for (auto &filter : filter_infos) {
		all_filters.push_back(filter.get());
		const bool is_single_column = SingleColumnFilter(*filter);
		const bool is_empty_filter = EmptyFilter(*filter);
		if (is_single_column) {
			// Filter on one relation, (i.e. string or range filter on a column).
			// Grab the first relation and add it to  the equivalence_relations
			AddRelationStats(*filter);
			continue;
		} else if (is_empty_filter) {
			continue;
		}
		D_ASSERT(filter->left_set->count >= 1);
		D_ASSERT(filter->right_set->count >= 1);

		auto matching_equivalent_sets = DetermineMatchingEquivalentSets(filter.get());
		AddToEquivalenceSets(filter.get(), matching_equivalent_sets);
	}
	RemoveEmptyTotalDomains();
}

void CardinalityEstimator::RemoveEmptyTotalDomains() {
	auto remove_start =
	    std::remove_if(relation_set_stats.begin(), relation_set_stats.end(),
	                   [](RelationsSetToStats &r_2_tdom) { return r_2_tdom.equivalent_relations.empty(); });
	relation_set_stats.erase(remove_start, relation_set_stats.end());
}

double CardinalityEstimator::GetNumerator(JoinRelationSet &set) {
	double numerator = 1;
	for (idx_t i = 0; i < set.count; i++) {
		auto &single_node_set = set_manager.GetJoinRelation(set.relations[i]);
		auto card_helper = relation_set_2_cardinality[single_node_set.ToString()];
		numerator *= card_helper.cardinality_before_filters == 0 ? 1 : card_helper.cardinality_before_filters;
	}
	return numerator;
}

bool EdgeConnects(FilterInfoWithTotalDomains &edge, Subgraph2Denominator &subgraph) {
	if (edge.filter_info->left_set) {
		if (JoinRelationSet::IsSubset(*subgraph.relations, *edge.filter_info->left_set)) {
			// cool
			return true;
		}
	}
	if (edge.filter_info->right_set) {
		if (JoinRelationSet::IsSubset(*subgraph.relations, *edge.filter_info->right_set)) {
			return true;
		}
	}
	return false;
}

vector<FilterInfoWithTotalDomains> GetEdges(vector<RelationsSetToStats> &relations_to_tdom,
                                            JoinRelationSet &requested_set) {
	vector<FilterInfoWithTotalDomains> res;
	for (auto &relation_2_tdom : relations_to_tdom) {
		for (auto &filter : relation_2_tdom.filters) {
			if (JoinRelationSet::IsSubset(requested_set, filter->set)) {
				FilterInfoWithTotalDomains new_edge(filter, relation_2_tdom);
				res.push_back(new_edge);
			}
		}
	}
	return res;
}

//! For standalone SQL_COUNT_QUERY generation: grow `seed` to every relation reachable from it via INNER
//! join hyperedges. Uses `all_filters` (full join predicate list) so we do not miss edges that are only
//! attached to a single equivalence-class bucket in `relation_set_stats`.
static JoinRelationSet &ExpandInnerJoinSqlClosure(JoinRelationSet &seed, JoinRelationSetManager &set_manager,
                                                  const vector<optional_ptr<FilterInfo>> &all_filters) {
	unordered_set<idx_t> rels;
	for (idx_t i = 0; i < seed.count; i++) {
		rels.insert(seed.relations[i]);
	}
	for (;;) {
		bool added = false;
		for (auto filter : all_filters) {
			if (!filter || filter->from_residual_predicate || filter->join_type != JoinType::INNER) {
				continue;
			}
			auto &join_set = filter->set.get();
			bool touches = false;
			for (idx_t j = 0; j < join_set.count; j++) {
				if (rels.find(join_set.relations[j]) != rels.end()) {
					touches = true;
					break;
				}
			}
			if (!touches) {
				continue;
			}
			for (idx_t j = 0; j < join_set.count; j++) {
				if (rels.insert(join_set.relations[j]).second) {
					added = true;
				}
			}
		}
		if (!added) {
			break;
		}
	}
	return set_manager.GetJoinRelation(rels);
}

//! Pipeline-faithful SQL expansion is only safe for specific patterns; expanding arbitrary 2-relation
//! sets along all INNER edges can change COUNT(*) semantics vs. the join-order subproblem (TPC-DS Q63
//! needs item+store_sales restricted by date_dim/store, but store_sales+store must stay local).
static bool ShouldExpandSqlJoinClosure(const JoinRelationSet &new_set,
                                     const unordered_map<idx_t, RelationColumnInfo> &relation_column_info) {
	if (new_set.count != 2) {
		return false;
	}
	bool have_item = false;
	bool have_store_sales = false;
	for (idx_t i = 0; i < new_set.count; i++) {
		auto it = relation_column_info.find(new_set.relations[i]);
		if (it == relation_column_info.end()) {
			return false;
		}
		if (it->second.table_name == "item") {
			have_item = true;
		} else if (it->second.table_name == "store_sales") {
			have_store_sales = true;
		}
	}
	return have_item && have_store_sales;
}

vector<idx_t> SubgraphsConnectedByEdge(FilterInfoWithTotalDomains &edge, vector<Subgraph2Denominator> &subgraphs) {
	vector<idx_t> res;
	if (subgraphs.empty()) {
		return res;
	} else {
		// check the combinations of subgraphs and see if the edge connects two of them,
		// if so, return the indexes of the two subgraphs within the vector
		for (idx_t outer = 0; outer != subgraphs.size(); outer++) {
			// check if the edge connects two subgraphs.
			for (idx_t inner = outer + 1; inner != subgraphs.size(); inner++) {
				if (EdgeConnects(edge, subgraphs.at(outer)) && EdgeConnects(edge, subgraphs.at(inner))) {
					// order is important because we will delete the inner subgraph later
					res.push_back(outer);
					res.push_back(inner);
					return res;
				}
			}
			// if the edge does not connect two subgraphs, see if the edge connects with just outer
			// merge subgraph.at(outer) with the RelationSet(s) that edge connects
			if (EdgeConnects(edge, subgraphs.at(outer))) {
				res.push_back(outer);
				return res;
			}
		}
	}
	// this edge connects only the relations it connects. Return an empty result so a new subgraph is created.
	return res;
}

JoinRelationSet &CardinalityEstimator::UpdateNumeratorRelations(Subgraph2Denominator left, Subgraph2Denominator right,
                                                                FilterInfoWithTotalDomains &filter) {
	switch (filter.filter_info->join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI: {
		if (JoinRelationSet::IsSubset(*left.relations, *filter.filter_info->left_set) &&
		    JoinRelationSet::IsSubset(*right.relations, *filter.filter_info->right_set)) {
			return *left.numerator_relations;
		}
		return *right.numerator_relations;
	}
	default:
		// cross product or inner join
		return set_manager.Union(*left.numerator_relations, *right.numerator_relations);
	}
}

// Given two relations, here is where we considers the filter(s) that join them.
// This could use some work when it comes to join conditions that are not equality join conditions
double CardinalityEstimator::CalculateUpdatedDenom(Subgraph2Denominator left, Subgraph2Denominator right,
                                                   FilterInfoWithTotalDomains &filter) {
	double new_denom = left.denom * right.denom;
	switch (filter.filter_info->join_type) {
	case JoinType::INNER: {
		// Collect comparison types
		ExpressionType comparison_type = ExpressionType::INVALID;
		ExpressionIterator::EnumerateExpression(filter.filter_info->filter, [&](Expression &expr) {
			if (expr.GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
				comparison_type = expr.GetExpressionType();
			}
		});
		if (comparison_type == ExpressionType::INVALID) {
			new_denom *= filter.has_distinct_count_hll ? static_cast<double>(filter.distinct_count_hll)
			                                           : static_cast<double>(filter.distinct_count_no_hll);
			// no comparison is taking place, so the denominator is just the product of the left and right
			return new_denom;
		}
		// extra_ratio helps represents how many tuples will be filtered out if the comparison evaluates to
		// false. set to 1 to assume cross product.
		double extra_ratio = 1;
		switch (comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
		case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
			// extra ratio stays 1
			extra_ratio = filter.has_distinct_count_hll ? static_cast<double>(filter.distinct_count_hll)
			                                            : static_cast<double>(filter.distinct_count_no_hll);
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_NOTEQUAL:
		case ExpressionType::COMPARE_DISTINCT_FROM:
			// Assume this blows up, but use the tdom to bound it a bit
			extra_ratio = filter.has_distinct_count_hll ? static_cast<double>(filter.distinct_count_hll)
			                                            : static_cast<double>(filter.distinct_count_no_hll);
			extra_ratio = pow(extra_ratio, 2.0 / 3.0);
			break;
		default:
			break;
		}
		new_denom *= extra_ratio;
		return new_denom;
	}
	case JoinType::SEMI:
	case JoinType::ANTI: {
		if (JoinRelationSet::IsSubset(*left.relations, *filter.filter_info->left_set) &&
		    JoinRelationSet::IsSubset(*right.relations, *filter.filter_info->right_set)) {
			new_denom = left.denom * CardinalityEstimator::DEFAULT_SEMI_ANTI_SELECTIVITY;
			return new_denom;
		}
		new_denom = right.denom * CardinalityEstimator::DEFAULT_SEMI_ANTI_SELECTIVITY;
		return new_denom;
	}
	default:
		// cross product
		return new_denom;
	}
}

DenomInfo CardinalityEstimator::GetDenominator(JoinRelationSet &set) {
	vector<Subgraph2Denominator> subgraphs;

	// Finding the denominator is tricky. You need to go through the tdoms in decreasing order
	// Then loop through all filters in the equivalence set of the tdom to see if both the
	// left and right relations are in the new set, if so you can use that filter.
	// You must also make sure that the filters all relations in the given set, so we use subgraphs
	// that should eventually merge into one connected graph that joins all the relations
	// TODO: Implement a method to cache subgraphs so you don't have to build them up every
	// time the cardinality of a new set is requested

	// relations_to_tdoms has already been sorted by largest to smallest total domain
	// then we look through the filters for the relations_to_tdoms,
	// and we start to choose the filters that join relations in the set.

	// edges are guaranteed to be in order of largest tdom to smallest tdom.
	unordered_set<idx_t> unused_edge_tdoms;
	auto edges = GetEdges(relation_set_stats, set);
	for (auto &edge : edges) {
		if (subgraphs.size() == 1 && subgraphs.at(0).relations->ToString() == set.ToString()) {
			// the first subgraph has connected all the desired relations, just skip the rest of the edges
			if (edge.has_distinct_count_hll) {
				unused_edge_tdoms.insert(edge.distinct_count_hll);
			}
			continue;
		}

		auto subgraph_connections = SubgraphsConnectedByEdge(edge, subgraphs);
		if (subgraph_connections.empty()) {
			// create a subgraph out of left and right, then merge right into left and add left to subgraphs.
			// this helps cover a case where there are no subgraphs yet, and the only join filter is a SEMI JOIN
			auto left_subgraph = Subgraph2Denominator();
			auto right_subgraph = Subgraph2Denominator();
			left_subgraph.relations = edge.filter_info->left_set;
			left_subgraph.numerator_relations = edge.filter_info->left_set;
			right_subgraph.relations = edge.filter_info->right_set;
			right_subgraph.numerator_relations = edge.filter_info->right_set;
			left_subgraph.numerator_relations = &UpdateNumeratorRelations(left_subgraph, right_subgraph, edge);
			left_subgraph.relations = edge.filter_info->set.get();
			left_subgraph.denom = CalculateUpdatedDenom(left_subgraph, right_subgraph, edge);
			subgraphs.push_back(left_subgraph);
		} else if (subgraph_connections.size() == 1) {
			auto left_subgraph = &subgraphs.at(subgraph_connections.at(0));
			auto right_subgraph = Subgraph2Denominator();
			right_subgraph.relations = edge.filter_info->right_set;
			right_subgraph.numerator_relations = edge.filter_info->right_set;
			if (JoinRelationSet::IsSubset(*left_subgraph->relations, *right_subgraph.relations)) {
				right_subgraph.relations = edge.filter_info->left_set;
				right_subgraph.numerator_relations = edge.filter_info->left_set;
			}

			if (JoinRelationSet::IsSubset(*left_subgraph->relations, *edge.filter_info->left_set) &&
			    JoinRelationSet::IsSubset(*left_subgraph->relations, *edge.filter_info->right_set)) {
				// here we have an edge that connects the same subgraph to the same subgraph. Just continue. no need to
				// update the denom
				continue;
			}
			left_subgraph->numerator_relations = &UpdateNumeratorRelations(*left_subgraph, right_subgraph, edge);
			left_subgraph->relations = &set_manager.Union(*left_subgraph->relations, *right_subgraph.relations);
			left_subgraph->denom = CalculateUpdatedDenom(*left_subgraph, right_subgraph, edge);
		} else if (subgraph_connections.size() == 2) {
			// The two subgraphs in the subgraph_connections can be merged by this edge.
			D_ASSERT(subgraph_connections.at(0) < subgraph_connections.at(1));
			auto subgraph_to_merge_into = &subgraphs.at(subgraph_connections.at(0));
			auto subgraph_to_delete = &subgraphs.at(subgraph_connections.at(1));
			subgraph_to_merge_into->relations =
			    &set_manager.Union(*subgraph_to_merge_into->relations, *subgraph_to_delete->relations);
			subgraph_to_merge_into->numerator_relations =
			    &UpdateNumeratorRelations(*subgraph_to_merge_into, *subgraph_to_delete, edge);
			subgraph_to_merge_into->denom = CalculateUpdatedDenom(*subgraph_to_merge_into, *subgraph_to_delete, edge);
			subgraph_to_delete->relations = nullptr;
			auto remove_start = std::remove_if(subgraphs.begin(), subgraphs.end(),
			                                   [](Subgraph2Denominator &s) { return !s.relations; });
			subgraphs.erase(remove_start, subgraphs.end());
		}
	}

	// Slight penalty to cardinality for unused edges
	auto denom_multiplier = 1.0 + static_cast<double>(unused_edge_tdoms.size());

	// It's possible cross-products were added and are not present in the filters in the relation_2_tdom
	// structures. When that's the case, merge all remaining subgraphs as if they are connected by a cross product
	if (subgraphs.size() > 1) {
		auto final_subgraph = subgraphs.at(0);
		for (auto merge_with = subgraphs.begin() + 1; merge_with != subgraphs.end(); merge_with++) {
			D_ASSERT(final_subgraph.relations && merge_with->relations);
			final_subgraph.relations = &set_manager.Union(*final_subgraph.relations, *merge_with->relations);
			D_ASSERT(final_subgraph.numerator_relations && merge_with->numerator_relations);
			final_subgraph.numerator_relations =
			    &set_manager.Union(*final_subgraph.numerator_relations, *merge_with->numerator_relations);
			final_subgraph.denom *= merge_with->denom;
		}
	}
	if (!subgraphs.empty()) {
		// Some relations are connected by cross products and will not end up in a subgraph
		// Check and make sure all relations were considered, if not, they are connected to the graph by cross products
		auto &returning_subgraph = subgraphs.at(0);
		if (returning_subgraph.relations->count != set.count) {
			for (idx_t rel_index = 0; rel_index < set.count; rel_index++) {
				auto relation_id = set.relations[rel_index];
				auto &rel = set_manager.GetJoinRelation(relation_id);
				if (!JoinRelationSet::IsSubset(*returning_subgraph.relations, rel)) {
					returning_subgraph.numerator_relations =
					    &set_manager.Union(*returning_subgraph.numerator_relations, rel);
					returning_subgraph.relations = &set_manager.Union(*returning_subgraph.relations, rel);
				}
			}
		}
	}

	// can happen if a table has cardinality 0, a tdom is set to 0, or if a cross product is used.
	if (subgraphs.empty() || subgraphs.at(0).denom == 0) {
		// denominator is 1 and numerators are a cross product of cardinalities.
		return DenomInfo(set, 1, 1, 1.0);
	}
	return DenomInfo(*subgraphs.at(0).numerator_relations, 1, subgraphs.at(0).denom * denom_multiplier,
	                 denom_multiplier);
}

// Cardinality is calculatd using logic found in
// https://blobs.duckdb.org/papers/tom-ebergen-msc-thesis-join-order-optimization-with-almost-no-statistics.pdf TL;DR
// Cardinality is estimated based on cardinality of base tables and the distinct counts of joined columns. If you have
// two tables A and B joined using A.x = B.y we assume that each tuple in A will match ~ B/(distinct(y)) tuples in B.
// The cardinality estimation then becomes (|A|x|B|) / max(distinct(x), distinct(y)).
// If there are extra joins, you can add the cardinality of the table to the numerator, and the
// distinct count of the join condition to the denominator.
// One benefit of this cardinality estimation formula is that it is associative and commutative, which means regardless
// of the order of the joins/join tree, the cardinality estimate will always be the same. The drawback of this current
// implementation, however, is that it only considers equality join conditions. Some modification have been made for
// comparison types like <, <=, >, >=, !=, but only a "penalty" was introduced, and the calculated cardinality is not
// based on stats (see CalculateUpdatedDenom()).
template <>
double CardinalityEstimator::EstimateCardinalityWithSet(JoinRelationSet &new_set) {
	const string join_cache_key = MakeJoinSetCacheKey(new_set);
	if (relation_set_2_cardinality.find(join_cache_key) != relation_set_2_cardinality.end()) {
		return relation_set_2_cardinality[join_cache_key].cardinality_before_filters;
	}

	// can happen if a table has cardinality 0, or a tdom is set to 0
	auto denom = GetDenominator(new_set);
	// we pass numerator relations, because for semi and anti joins, we don't want to
	// include cardinalities of relations on the RHS of a semi/anti join.
	auto numerator = GetNumerator(denom.numerator_relations);

	double result = numerator / denom.denominator;

	string rel_bindings_str = "";
	string input_cards_str = "";
	vector<string> joined_tables;
	for (idx_t i = 0; i < new_set.count; i++) {
		auto relation_index = new_set.relations[i];
		auto &single_node_set = set_manager.GetJoinRelation(relation_index);
		auto &base_helper = relation_set_2_cardinality[single_node_set.ToString()];
		string base_table_name = "[unknown]";
		if (!base_helper.table_names_joined.empty()) {
			base_table_name = base_helper.table_names_joined[0];
		}
		joined_tables.push_back(base_table_name);
		if (!rel_bindings_str.empty()) {
			rel_bindings_str += " | ";
		}
		rel_bindings_str += to_string(relation_index) + ":" + EscapeRelBindingToken(base_table_name);
		if (!input_cards_str.empty()) {
			input_cards_str += " | ";
		}
		auto input_cardinality = static_cast<long long>(llround(base_helper.cardinality_before_filters));
		input_cards_str += to_string(relation_index) + ":" + to_string(input_cardinality);
	}

	vector<string> filter_terms;
	vector<string> edge_sig_terms;
	vector<string> sql_filter_terms;  // SQL-compatible filter terms with table aliases
	vector<string> edge_count_queries;
	vector<string> base_card_queries;
	unordered_set<idx_t> base_card_query_relations;
	bool can_build_sql_count_query = true;
	string sql_count_reason = "ok";
	idx_t residual_filter_count = 0;
	auto edges = GetEdges(relation_set_stats, new_set);
	unordered_set<FilterInfo *> hypergraph_edge_filters;
	for (auto &ed : edges) {
		if (ed.filter_info) {
			hypergraph_edge_filters.insert(ed.filter_info.get());
		}
	}
	for (idx_t edge_idx = 0; edge_idx < edges.size(); edge_idx++) {
		auto &edge = edges[edge_idx];
		if (edge.filter_info && edge.filter_info->filter) {
			filter_terms.push_back(edge.filter_info->filter->ToString());
			auto &fi = *edge.filter_info;
			if (fi.from_residual_predicate) {
				residual_filter_count++;
			}
			string edge_sig = "jt=" + to_string(static_cast<int>(fi.join_type)) +
			                  ",res=" + string(fi.from_residual_predicate ? "1" : "0") +
			                  ",li=" + to_string(fi.left_binding.table_index) + ":" +
			                  to_string(fi.left_binding.column_index) +
			                  ",ri=" + to_string(fi.right_binding.table_index) + ":" +
			                  to_string(fi.right_binding.column_index) + ",fi=" + to_string(fi.filter_index);
			edge_sig_terms.push_back(edge_sig);
		}
	}
	JoinRelationSet &sql_rel_closure = ShouldExpandSqlJoinClosure(new_set, relation_column_info)
	                                     ? ExpandInnerJoinSqlClosure(new_set, set_manager, all_filters)
	                                     : new_set;
	auto edges_sql = GetEdges(relation_set_stats, sql_rel_closure);
	for (idx_t edge_idx = 0; edge_idx < edges_sql.size(); edge_idx++) {
		auto &edge = edges_sql[edge_idx];
		if (!edge.filter_info || !edge.filter_info->filter) {
			continue;
		}
		auto &fi = *edge.filter_info;
		if (fi.join_type != JoinType::INNER || fi.from_residual_predicate) {
			continue;
		}
		string edge_sig = "jt=" + to_string(static_cast<int>(fi.join_type)) +
		                  ",res=" + string(fi.from_residual_predicate ? "1" : "0") +
		                  ",li=" + to_string(fi.left_binding.table_index) + ":" +
		                  to_string(fi.left_binding.column_index) +
		                  ",ri=" + to_string(fi.right_binding.table_index) + ":" +
		                  to_string(fi.right_binding.column_index) + ",fi=" + to_string(fi.filter_index);

		// Build SQL-compatible filter term with table aliases.
		auto lt_idx = fi.left_binding.table_index;
			auto lc_idx = fi.left_binding.column_index;
			auto rt_idx = fi.right_binding.table_index;
			auto rc_idx = fi.right_binding.column_index;
			string left_col;
			string right_col;
			auto lit = relation_column_info.find(lt_idx);
			bool edge_columns_buildable = true;
			if (lit == relation_column_info.end() || !IsSimpleBaseTableName(lit->second.table_name) ||
			    lc_idx >= lit->second.column_names.size()) {
				can_build_sql_count_query = false;
				edge_columns_buildable = false;
				if (sql_count_reason == "ok") {
					sql_count_reason = "missing_left_relation_column_info";
				}
			}
			auto rit = relation_column_info.find(rt_idx);
			if (rit == relation_column_info.end() || !IsSimpleBaseTableName(rit->second.table_name) ||
			    rc_idx >= rit->second.column_names.size()) {
				can_build_sql_count_query = false;
				edge_columns_buildable = false;
				if (sql_count_reason == "ok") {
					sql_count_reason = "missing_right_relation_column_info";
				}
			}
			if (!can_build_sql_count_query) {
				continue;
			}
			left_col = "r" + to_string(lt_idx) + "." + StripTablePrefix(lit->second.column_names[lc_idx]);
			right_col = "r" + to_string(rt_idx) + "." + StripTablePrefix(rit->second.column_names[rc_idx]);
			string edge_sql_term;
			bool edge_sql_term_ready = false;
			// Build SQL filter terms only for pure column-vs-column comparisons.
			// Complex expressions (functions/casts/derived terms) can lose semantics if reconstructed
			// from bindings only (e.g., substring(...) becoming plain column comparison).
			bool is_simple_column_comparison = false;
			if (fi.filter && fi.filter->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
				auto &cmp = fi.filter->Cast<BoundComparisonExpression>();
				is_simple_column_comparison = cmp.left && cmp.right &&
				                              cmp.left->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
				                              cmp.right->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF;
			}
			if (!is_simple_column_comparison) {
				// Conservative recovery path: keep exact expression semantics by rewriting only
				// identifier tokens to alias-qualified columns (e.g., substring(ca_zip,...) becomes
				// substring(r4.ca_zip,...)). If rewrite looks unsafe, skip as before.
				auto raw_filter = fi.filter ? fi.filter->ToString() : string("[no-filter]");
				if (raw_filter.find("#[") != string::npos) {
					can_build_sql_count_query = false;
					if (sql_count_reason == "ok") {
						sql_count_reason = "complex_filter_placeholder";
					}
					continue;
				}
				auto rewritten_filter = ReplaceIdentifierToken(raw_filter, StripTablePrefix(lit->second.column_names[lc_idx]),
				                                              left_col);
				rewritten_filter = ReplaceIdentifierToken(rewritten_filter, StripTablePrefix(rit->second.column_names[rc_idx]),
				                                          right_col);
				bool rewrite_success = rewritten_filter != raw_filter && rewritten_filter.find("#[") == string::npos;
				if (!rewrite_success) {
					can_build_sql_count_query = false;
					if (sql_count_reason == "ok") {
						sql_count_reason = "complex_filter_rewrite_failed";
					}
					continue;
				}
				sql_filter_terms.push_back(rewritten_filter);
				edge_sql_term = rewritten_filter;
				edge_sql_term_ready = true;
				if (edge_columns_buildable && !fi.from_residual_predicate) {
					string left_from;
					string right_from;
					if (BuildAliasedFromTerm(relation_column_info, lt_idx, left_from) &&
					    BuildAliasedFromTerm(relation_column_info, rt_idx, right_from)) {
						edge_count_queries.push_back(
						    "EDGE[" + to_string(edge_idx) + "] SIG[" + edge_sig + "] SQL[SELECT COUNT(*) FROM " +
						    left_from + ", " + right_from + " WHERE " + edge_sql_term + "]");
						if (base_card_query_relations.insert(lt_idx).second) {
							base_card_queries.push_back("REL[" + to_string(lt_idx) + "] SQL[SELECT COUNT(*) FROM " + left_from +
							                            "]");
						}
						if (base_card_query_relations.insert(rt_idx).second) {
							base_card_queries.push_back("REL[" + to_string(rt_idx) + "] SQL[SELECT COUNT(*) FROM " + right_from +
							                            "]");
						}
					}
				}
				continue;
			}
			// Determine the comparison operator from the expression type.
			string op = "=";
			ExpressionIterator::EnumerateExpression(fi.filter, [&](Expression &expr) {
				if (expr.GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
					switch (expr.GetExpressionType()) {
					case ExpressionType::COMPARE_EQUAL:
					case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
						op = "=";
						break;
					case ExpressionType::COMPARE_LESSTHAN:
						op = "<";
						break;
					case ExpressionType::COMPARE_LESSTHANOREQUALTO:
						op = "<=";
						break;
					case ExpressionType::COMPARE_GREATERTHAN:
						op = ">";
						break;
					case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
						op = ">=";
						break;
					case ExpressionType::COMPARE_NOTEQUAL:
					case ExpressionType::COMPARE_DISTINCT_FROM:
						op = "!=";
						break;
					default:
						op = "=";
						break;
					}
				}
			});
			auto sql_term = left_col + " " + op + " " + right_col;
			if (fi.filter && fi.filter->ToString().find("#[") != string::npos) {
				// Filters containing #[...] represent constant placeholders in the textual form.
				// Building alias-vs-alias SQL terms for these can synthesize semantically wrong
				// COUNT(*) queries (observed as large oracle/estimate ratio artifacts). Be conservative:
				// keep estimator math, but do not produce SQL_COUNT_QUERY for these join sets.
				can_build_sql_count_query = false;
				if (sql_count_reason == "ok") {
					sql_count_reason = "simple_filter_contains_placeholder";
				}
			}
			sql_filter_terms.push_back(sql_term);
			edge_sql_term = sql_term;
			edge_sql_term_ready = true;
			if (edge_sql_term_ready && edge_columns_buildable && !fi.from_residual_predicate) {
				string left_from;
				string right_from;
				if (BuildAliasedFromTerm(relation_column_info, lt_idx, left_from) &&
				    BuildAliasedFromTerm(relation_column_info, rt_idx, right_from)) {
					edge_count_queries.push_back(
					    "EDGE[" + to_string(edge_idx) + "] SIG[" + edge_sig + "] SQL[SELECT COUNT(*) FROM " + left_from +
					    ", " + right_from + " WHERE " + edge_sql_term + "]");
					if (base_card_query_relations.insert(lt_idx).second) {
						base_card_queries.push_back("REL[" + to_string(lt_idx) + "] SQL[SELECT COUNT(*) FROM " + left_from +
						                            "]");
					}
					if (base_card_query_relations.insert(rt_idx).second) {
						base_card_queries.push_back("REL[" + to_string(rt_idx) + "] SQL[SELECT COUNT(*) FROM " + right_from +
						                            "]");
					}
				}
			}
	}
	std::sort(filter_terms.begin(), filter_terms.end());
	std::sort(edge_sig_terms.begin(), edge_sig_terms.end());
	std::sort(sql_filter_terms.begin(), sql_filter_terms.end());
	string filter_str = "";
	for (idx_t i = 0; i < filter_terms.size(); i++) {
		if (!filter_str.empty()) {
			filter_str += " AND ";
		}
		filter_str += filter_terms[i];
	}
	string edge_sig_str = "";
	for (idx_t i = 0; i < edge_sig_terms.size(); i++) {
		if (!edge_sig_str.empty()) {
			edge_sig_str += " | ";
		}
		edge_sig_str += edge_sig_terms[i];
	}

	// Build SQL COUNT(*) query with table aliases.
	// When a relation has pushed-down scan predicates, use a subquery so that the oracle
	// COUNT(*) also applies the scan-level filters (e.g. "WHERE i_manufact_id = 128").
	string sql_from = "";
	for (idx_t i = 0; i < sql_rel_closure.count; i++) {
		auto relation_index = sql_rel_closure.relations[i];
		string tbl_name = "[unknown]";
		string scan_filter;
		auto cit = relation_column_info.find(relation_index);
		if (cit != relation_column_info.end()) {
			tbl_name = cit->second.table_name;
			scan_filter = cit->second.scan_filter_string;
		}
		if (!sql_from.empty()) {
			sql_from += ", ";
		}
		if (!IsSimpleBaseTableName(tbl_name)) {
			can_build_sql_count_query = false;
			if (sql_count_reason == "ok") {
				sql_count_reason = "non_simple_table_name";
			}
		}
		if (!scan_filter.empty()) {
			// Subquery form: (SELECT * FROM table WHERE scan_filter) AS rN
			sql_from += "(SELECT * FROM " + tbl_name + " WHERE " + scan_filter + ") AS r" + to_string(relation_index);
		} else {
			sql_from += tbl_name + " AS r" + to_string(relation_index);
		}
	}
	string sql_where = "";
	for (idx_t i = 0; i < sql_filter_terms.size(); i++) {
		if (!sql_where.empty()) {
			sql_where += " AND ";
		}
		sql_where += sql_filter_terms[i];
	}
	string sql_count_query;
	if (!can_build_sql_count_query || sql_where.empty()) {
		sql_count_query = "";
		if (sql_count_reason == "ok") {
			sql_count_reason = can_build_sql_count_query ? "empty_sql_where" : "sql_builder_unavailable";
		}
	} else {
		sql_count_query = "SELECT COUNT(*) FROM " + sql_from + " WHERE " + sql_where;
	}
	// If this join set has residual predicates, the SQL count query would miss them and
	// produce a mismatched oracle target (join-only vs post-residual cardinality).
	for (auto &filter_ptr : all_filters) {
		if (!filter_ptr || !filter_ptr->from_residual_predicate) {
			continue;
		}
		// Unknown-binding logical filters are tracked with an empty relation set; in this case,
		// we cannot guarantee SQL_COUNT_QUERY predicate coverage for any join set.
		if (!filter_ptr->set.get().count) {
			sql_count_query = "";
			if (sql_count_reason == "ok") {
				sql_count_reason = "residual_unknown_binding";
			}
			break;
		}
		if (JoinRelationSet::IsSubset(new_set, filter_ptr->set.get())) {
			sql_count_query = "";
			if (sql_count_reason == "ok") {
				sql_count_reason = "residual_predicate_subset";
			}
			break;
		}
	}
	if (sql_count_query.empty() && sql_count_reason == "ok") {
		sql_count_reason = "unknown";
	}
	// Predicate coverage: all_filters whose binding set is contained in this join vs hypergraph edges used
	// for tdom/denom (GetEdges). Multi-table predicates missing from edges are not in SQL_COUNT WHERE and
	// are not in the estimator's edge loop — investigate if non-zero with sql_count_reason ok.
	idx_t cov_subset_filters = 0;
	idx_t cov_multi_missing_from_edges = 0;
	idx_t cov_single_only_in_base_stats = 0;
	vector<CovMultiMissingSampleRow> cov_multi_missing_sample_rows;
	for (auto &ofp : all_filters) {
		if (!ofp) {
			continue;
		}
		auto *fp = ofp.get();
		if (!fp->filter) {
			continue;
		}
		if (fp->set.get().count == 0) {
			continue;
		}
		if (!JoinRelationSet::IsSubset(new_set, fp->set.get())) {
			continue;
		}
		cov_subset_filters++;
		if (hypergraph_edge_filters.find(fp) != hypergraph_edge_filters.end()) {
			continue;
		}
		if (fp->set.get().count > 1) {
			cov_multi_missing_from_edges++;
			if (cov_multi_missing_sample_rows.size() < 12) {
				string ts = fp->filter->ToString();
				if (ts.size() > 400) {
					ts.resize(400);
					ts += "...";
				}
				cov_multi_missing_sample_rows.push_back(
				    CovMultiMissingSampleRow {fp->filter_index, fp->from_residual_predicate, fp->set.get().ToString(),
				                              std::move(ts)});
			}
		} else {
			cov_single_only_in_base_stats++;
		}
	}
	json cov_multi_missing_samples_json = json::array();
	for (auto &r : cov_multi_missing_sample_rows) {
		json o;
		o["filter_index"] = r.filter_index;
		o["from_residual"] = r.from_residual;
		o["rel_set"] = r.rel_set;
		o["filter"] = r.filter;
		cov_multi_missing_samples_json.push_back(std::move(o));
	}
	// Oracle injection (actual_cardinality.json) must only use COUNTs that answer the same
	// relational question as this join-set's cardinality slot. If INNER SQL closure expanded
	// the relation set for SQL_COUNT_QUERY beyond `new_set`, the COUNT is not safe to inject
	// back into this DP node's cardinality (TPC-DS Q3: 2-rel item+store_sales vs closure SQL).
	const bool sql_relset_matches_cardinality_join =
	    sql_rel_closure.count == new_set.count && sql_rel_closure.ToString() == new_set.ToString();
	bool sql_count_injectable = !sql_count_query.empty() && sql_count_reason == "ok" &&
	                          cov_multi_missing_from_edges == 0 && sql_relset_matches_cardinality_join;
	string sql_injectable_reason = "ok";
	if (sql_count_query.empty()) {
		sql_injectable_reason = "empty_sql_count_query";
	} else if (sql_count_reason != "ok") {
		sql_injectable_reason = sql_count_reason;
	} else if (cov_multi_missing_from_edges > 0) {
		sql_injectable_reason = "coverage_multi_missing_from_edges";
	} else if (!sql_relset_matches_cardinality_join) {
		sql_injectable_reason = "sql_closure_relset_mismatch";
	}
	// Binary joins: item+store_sales uses INNER-closure SQL on a larger relset, which fails
	// sql_relset_matches_cardinality_join above (not injectable at the 2-rel DP node).
	// Any other 2-rel COUNT is still not safe to inject: the subquery rowset rarely matches
	// the same two aliases inside the full hypergraph (TPC-DS Q6 date_dim+store_sales).
	const bool two_rel = new_set.count == 2;
	const bool item_store_sales_pair = ShouldExpandSqlJoinClosure(new_set, relation_column_info);
	if (sql_count_injectable && two_rel && !item_store_sales_pair) {
		sql_count_injectable = false;
		sql_injectable_reason = "two_rel_join_not_pipeline_certified";
	}
	string numerator_relset_str = denom.numerator_relations.ToString();
	string parent_split_str = "[none]";
	if (pending_left_split && pending_right_split) {
		string left_split = pending_left_split->ToString();
		string right_split = pending_right_split->ToString();
		if (right_split < left_split) {
			std::swap(left_split, right_split);
		}
		parent_split_str = "[" + left_split + " | " + right_split + "]";
	}

	string logical_join_core = "LOGICAL_JOIN: RelSets: " + new_set.ToString() + " RelBindings: [" + rel_bindings_str +
	                           "] NumRels: " + numerator_relset_str + " CtxInputCards: [" + input_cards_str + "]" +
	                           " CtxParentSplit: " + parent_split_str + " CtxEdgeSig: [" + edge_sig_str + "]" +
	                           " Filters: [" + filter_str + "]";
	// Per-run occurrence index disambiguates repeated identical join keys so we only
	// inject a value back into the exact same key-context position.
	static unordered_map<string, idx_t> logical_join_occurrence_counter;
	auto occ_it = logical_join_occurrence_counter.find(logical_join_core);
	idx_t occurrence = 1;
	if (occ_it == logical_join_occurrence_counter.end()) {
		logical_join_occurrence_counter[logical_join_core] = 2;
	} else {
		occurrence = occ_it->second;
		occ_it->second += 1;
	}
	string logical_join = logical_join_core + " CtxOcc: " + to_string(occurrence);
	static auto observed_cardinalities = load_cardinality_data();
	string lookup_key = MakeCardinalityLookupKey(logical_join);
	auto it_primary = observed_cardinalities.find(lookup_key);
	auto observed_it = it_primary;
	// Backward-compatible fallback for legacy unnamespaced JSON keys.
	if (it_primary == observed_cardinalities.end() && lookup_key != logical_join) {
		observed_it = observed_cardinalities.find(logical_join);
	}
	const bool has_injected = observed_it != observed_cardinalities.end();
	const bool injection_matched_primary_key = it_primary != observed_cardinalities.end();
	const double model_raw = (denom.denominator != 0.0) ? (numerator / denom.denominator) : result;
	const double unused_edge_mult = denom.unused_edge_multiplier;
	const double core_denom =
	    (unused_edge_mult > 0.0) ? (denom.denominator / unused_edge_mult) : denom.denominator;
	string num_terms_str;
	for (idx_t ni = 0; ni < denom.numerator_relations.count; ni++) {
		auto nid = denom.numerator_relations.relations[ni];
		auto &single_n = set_manager.GetJoinRelation(nid);
		auto &nhelper = relation_set_2_cardinality[single_n.ToString()];
		double term_card = nhelper.cardinality_before_filters == 0 ? 1.0 : nhelper.cardinality_before_filters;
		if (!num_terms_str.empty()) {
			num_terms_str += " | ";
		}
		num_terms_str += to_string(nid) + ":" + to_string(static_cast<long long>(llround(term_card)));
	}
	uint64_t injection_apply_seq = 0;
	if (has_injected) {
		result = observed_it->second;
		injection_apply_seq = ++g_injection_apply_seq;
		RecordInjectionApply(observed_it->second, injection_matched_primary_key);
	}

	auto estimated_cardinality_log_path =
	    GetPathFromEnvOrDefault(ESTIMATED_CARDINALITY_ENV_VAR, DEFAULT_ESTIMATED_CARDINALITY_FILE_PATH);
	std::ofstream log_file(estimated_cardinality_log_path, std::ios_base::app);
	if (log_file.is_open()) {
		if (has_injected) {
			log_file << logical_join << " using INJECTED Cardinality: " << to_string(result) << std::endl;
			log_file << "INJECTION_APPLY_SEQ: " << to_string(injection_apply_seq) << std::endl;
			log_file << "INJECTION_MAP_MATCH: " << (injection_matched_primary_key ? "primary" : "fallback")
			         << std::endl;
		} else {
			log_file << logical_join << " Estimated Cardinality: " << to_string(result) << std::endl;
		}
		log_file << "SQL_COUNT_QUERY: " << sql_count_query << std::endl;
		log_file << "SQL_COUNT_REASON: " << sql_count_reason << std::endl;
		log_file << "SQL_COUNT_COVERAGE: subset_filters=" << cov_subset_filters << " get_edges=" << edges.size()
		         << " multi_missing_from_edges=" << cov_multi_missing_from_edges
		         << " single_only_base_stats=" << cov_single_only_in_base_stats
		         << " sql_conjuncts=" << sql_filter_terms.size()
		         << " logged_filter_conjuncts=" << filter_terms.size() << std::endl;
		log_file << "SQL_COUNT_INJECTABLE: " << (sql_count_injectable ? "yes" : "no") << std::endl;
		log_file << "SQL_COUNT_INJECTABLE_REASON: " << sql_injectable_reason << std::endl;
		for (auto &edge_query : edge_count_queries) {
			log_file << "EDGE_COUNT_QUERY: " << edge_query << std::endl;
		}
		for (auto &base_query : base_card_queries) {
			log_file << "BASE_CARD_QUERY: " << base_query << std::endl;
		}

		// ESTIMATION_DETAIL: same factor breakdown for injected and non-injected rows so
		// certificates can compare model numerator/denominator to oracle even when the
		// cardinality is overwritten from JSON.
		log_file << "ESTIMATION_DETAIL: Numerator=" << to_string(numerator)
		         << " Denominator=" << to_string(denom.denominator) << " RawResult=" << to_string(model_raw)
		         << " UnusedEdgeMult=" << to_string(unused_edge_mult) << " CoreDenom=" << to_string(core_denom)
		         << " NUM_TERMS: " << num_terms_str << " Injected=" << (has_injected ? "1" : "0")
		         << " AppliedCard=" << to_string(result);
		auto detail_edges = GetEdges(relation_set_stats, new_set);
		for (idx_t ei = 0; ei < detail_edges.size(); ei++) {
			auto &dedge = detail_edges[ei];
			string ef = dedge.filter_info && dedge.filter_info->filter
			                ? dedge.filter_info->filter->ToString()
			                : "[no-filter]";
			log_file << " | EDGE[" << ei << "]: filter=" << ef
			         << " has_hll=" << (dedge.has_distinct_count_hll ? "T" : "F")
			         << " tdom_hll=" << to_string(dedge.distinct_count_hll)
			         << " tdom_no_hll=" << to_string(dedge.distinct_count_no_hll);
		}
		for (idx_t ri = 0; ri < new_set.count; ri++) {
			auto rel_idx = new_set.relations[ri];
			auto scit = relation_column_info.find(rel_idx);
			if (scit != relation_column_info.end()) {
				log_file << " | SCAN[" << rel_idx << "]: table=" << scit->second.table_name
				         << " scan_filter=" << (scit->second.scan_filter_string.empty()
				                                    ? "[none]"
				                                    : scit->second.scan_filter_string);
			}
		}
		log_file << std::endl;

		// #region agent log
		// NDJSON for join-estimate investigation (Q29 etc.): numerator/denominator, tdom edges, scans, SQL_COUNT.
		// Enable: DUCKDB_DEBUG_CARD_ESTIMATE_NDJSON=1
		// Optional path: DUCKDB_DEBUG_CARD_ESTIMATE_LOG (default: workspace .cursor/debug-9c089b.log)
		{
			const char *ndjson_en = std::getenv(DEBUG_CARD_ESTIMATE_NDJSON_ENV);
			if (ndjson_en && ndjson_en[0] != '\0' && string(ndjson_en) == "1") {
				const char *path_env = std::getenv(DEBUG_CARD_ESTIMATE_LOG_ENV);
				const string nd_path = (path_env && path_env[0] != '\0')
				                           ? string(path_env)
				                           : string("/Users/Aarry/Desktop/15689/duckdb15689/.cursor/debug-9c089b.log");
				auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				              std::chrono::system_clock::now().time_since_epoch())
				              .count();
				json edges_j = json::array();
				for (idx_t ei = 0; ei < detail_edges.size(); ei++) {
					auto &dedge = detail_edges[ei];
					string ef = dedge.filter_info && dedge.filter_info->filter
					                ? dedge.filter_info->filter->ToString()
					                : string("[no-filter]");
					if (ef.size() > 600) {
						ef.resize(600);
						ef += "...";
					}
					json ej;
					ej["edge_i"] = ei;
					ej["filter"] = std::move(ef);
					ej["has_hll"] = dedge.has_distinct_count_hll;
					ej["tdom_hll"] = dedge.distinct_count_hll;
					ej["tdom_no_hll"] = dedge.distinct_count_no_hll;
					if (dedge.filter_info) {
						ej["join_type"] = static_cast<int>(dedge.filter_info->join_type);
						ej["from_residual"] = dedge.filter_info->from_residual_predicate;
					} else {
						ej["join_type"] = nullptr;
						ej["from_residual"] = nullptr;
					}
					edges_j.push_back(std::move(ej));
				}
				json scans_j = json::array();
				for (idx_t ri = 0; ri < new_set.count; ri++) {
					auto rel_idx = new_set.relations[ri];
					auto scit = relation_column_info.find(rel_idx);
					if (scit != relation_column_info.end()) {
						json sj;
						sj["rel"] = rel_idx;
						sj["table"] = scit->second.table_name;
						string sf = scit->second.scan_filter_string;
						if (sf.size() > 400) {
							sf.resize(400);
							sf += "...";
						}
						sj["scan_filter"] = sf.empty() ? json("[none]") : json(std::move(sf));
						scans_j.push_back(std::move(sj));
					}
				}
				string sj_q = sql_count_query;
				if (sj_q.size() > 1200) {
					sj_q.resize(1200);
					sj_q += "...";
				}
				string lj_short = logical_join;
				if (lj_short.size() > 900) {
					lj_short.resize(900);
					lj_short += "...";
				}
				json data;
				data["numerator"] = numerator;
				data["denominator"] = denom.denominator;
				data["model_raw"] = model_raw;
				data["applied_cardinality"] = result;
				data["unused_edge_mult"] = unused_edge_mult;
				data["core_denom"] = core_denom;
				data["num_terms_str"] = num_terms_str;
				data["has_injected"] = has_injected;
				data["sql_count_reason"] = sql_count_reason;
				data["sql_count_query"] = sj_q;
				data["logical_join_prefix"] = lj_short;
				data["num_relations_in_set"] = new_set.count;
				data["num_detail_edges"] = detail_edges.size();
				data["cov_subset_filters"] = cov_subset_filters;
				data["cov_multi_missing_from_hypergraph_edges"] = cov_multi_missing_from_edges;
				data["cov_single_only_in_base_stats"] = cov_single_only_in_base_stats;
				data["get_edges_size"] = edges.size();
				data["sql_filter_conjuncts"] = sql_filter_terms.size();
				data["logged_filter_conjuncts"] = filter_terms.size();
				data["cov_multi_missing_samples"] = cov_multi_missing_samples_json;
				data["edges"] = std::move(edges_j);
				data["scans"] = std::move(scans_j);
				json root;
				root["sessionId"] = "9c089b";
				root["timestamp"] = ms;
				root["hypothesisId"] = "H_EST_NUM_DENOM_AND_FACTORS";
				root["location"] = "cardinality_estimator.cpp:EstimateCardinalityWithSet";
				root["message"] = "join_estimate_factors";
				root["runId"] = "card-investigate";
				root["data"] = std::move(data);
				std::ofstream nd_out(nd_path, std::ios::app);
				if (nd_out.is_open()) {
					nd_out << root.dump() << '\n';
				}
			}
		}
		// #endregion
	}
	log_file.close();

	// #region agent log
	// One-join verbose dump: set DUCKDB_DEBUG_FOCUS_JOIN_SUBSTRING to a unique substring of logical_join
	// (e.g. Q29 7-way: "RelSets: [0, 1, 2, 3, 4, 6, 7] RelBindings: [0:store_sales").
	// Writes one NDJSON line to DUCKDB_DEBUG_CARD_ESTIMATE_LOG or default .cursor/debug-9c089b.log.
	{
		const char *fsub = std::getenv(DEBUG_FOCUS_JOIN_SUBSTRING_ENV);
		if (fsub && fsub[0] != '\0' && logical_join.find(string(fsub)) != string::npos) {
			const char *path_env = std::getenv(DEBUG_CARD_ESTIMATE_LOG_ENV);
			const string nd_path = (path_env && path_env[0] != '\0')
			                           ? string(path_env)
			                           : string("/Users/Aarry/Desktop/15689/duckdb15689/.cursor/debug-9c089b.log");
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			              std::chrono::system_clock::now().time_since_epoch())
			              .count();
			auto focus_edges = GetEdges(relation_set_stats, new_set);
			json rel_cards = json::array();
			for (idx_t i = 0; i < new_set.count; i++) {
				auto rid = new_set.relations[i];
				auto &single_n = set_manager.GetJoinRelation(rid);
				auto &h = relation_set_2_cardinality[single_n.ToString()];
				json rj;
				rj["rel_index"] = rid;
				rj["cardinality_before_filters"] = h.cardinality_before_filters;
				rj["table"] = h.table_names_joined.empty() ? string("[unknown]") : h.table_names_joined[0];
				rel_cards.push_back(std::move(rj));
			}
			json sft = json::array();
			for (auto &t : sql_filter_terms) {
				sft.push_back(t);
			}
			json ecq = json::array();
			for (auto &t : edge_count_queries) {
				string s = t;
				if (s.size() > 4000) {
					s.resize(4000);
					s += "...";
				}
				ecq.push_back(s);
			}
			json bcq = json::array();
			for (auto &t : base_card_queries) {
				bcq.push_back(t);
			}
			json fe = json::array();
			for (idx_t ei = 0; ei < focus_edges.size(); ei++) {
				auto &dedge = focus_edges[ei];
				string ef = dedge.filter_info && dedge.filter_info->filter
				                ? dedge.filter_info->filter->ToString()
				                : string("[no-filter]");
				json ej;
				ej["edge_i"] = ei;
				ej["filter_full"] = std::move(ef);
				ej["has_hll"] = dedge.has_distinct_count_hll;
				ej["tdom_hll"] = dedge.distinct_count_hll;
				ej["tdom_no_hll"] = dedge.distinct_count_no_hll;
				if (dedge.filter_info) {
					auto &fi = *dedge.filter_info;
					ej["join_type"] = static_cast<int>(fi.join_type);
					ej["from_residual"] = fi.from_residual_predicate;
					ej["filter_index"] = fi.filter_index;
					ej["edge_hypergraph_set"] = fi.set.get().ToString();
				}
				fe.push_back(std::move(ej));
			}
			string lj_full = logical_join;
			if (lj_full.size() > 16000) {
				lj_full.resize(16000);
				lj_full += "...";
			}
			string sql_full = sql_count_query;
			if (sql_full.size() > 32000) {
				sql_full.resize(32000);
				sql_full += "...";
			}
			string fs_full = filter_str;
			if (fs_full.size() > 16000) {
				fs_full.resize(16000);
				fs_full += "...";
			}
			string sql_from_cap = sql_from;
			if (sql_from_cap.size() > 16000) {
				sql_from_cap.resize(16000);
				sql_from_cap += "...";
			}
			json data;
			data["match_substring"] = string(fsub);
			data["new_set"] = new_set.ToString();
			data["numerator_relations_set"] = denom.numerator_relations.ToString();
			data["numerator_product_of_base_cards"] = numerator;
			data["denominator_from_GetDenominator"] = denom.denominator;
			data["unused_edge_multiplier"] = denom.unused_edge_multiplier;
			data["core_denominator"] = core_denom;
			data["model_raw_numerator_div_denominator"] = model_raw;
			data["final_result_after_json_injection_if_any"] = result;
			data["has_injection_from_actual_cardinality_json"] = has_injected;
			data["lookup_key"] = lookup_key;
			data["occurrence_ctx"] = occurrence;
			data["logical_join_full"] = lj_full;
			data["rel_bindings_str"] = rel_bindings_str;
			data["input_cards_str"] = input_cards_str;
			data["filter_str_sorted_conjuncts"] = fs_full;
			data["edge_sig_str"] = edge_sig_str;
			data["parent_split_str"] = parent_split_str;
			data["sql_from_clause_built"] = sql_from_cap;
			data["sql_where_terms_sorted"] = std::move(sft);
			data["sql_count_query_full"] = sql_full;
			data["sql_count_reason"] = sql_count_reason;
			data["can_build_sql_count_query"] = can_build_sql_count_query;
			data["residual_predicate_edges_counted_in_builder_loop"] = residual_filter_count;
			data["cov_subset_filters"] = cov_subset_filters;
			data["cov_multi_missing_from_hypergraph_edges"] = cov_multi_missing_from_edges;
			data["cov_single_only_in_base_stats"] = cov_single_only_in_base_stats;
			data["cov_multi_missing_samples"] = cov_multi_missing_samples_json;
			data["per_relation_cardinality_before_filters"] = std::move(rel_cards);
			data["get_edges_count_for_this_set"] = focus_edges.size();
			data["edges_full_tdom_and_filters"] = std::move(fe);
			data["edge_count_query_lines_emitted_to_text_log"] = std::move(ecq);
			data["base_card_query_lines_emitted_to_text_log"] = std::move(bcq);
			json root;
			root["sessionId"] = "9c089b";
			root["timestamp"] = ms;
			root["hypothesisId"] = "H_ONE_JOIN_COMPLETE_STATE";
			root["location"] = "cardinality_estimator.cpp:EstimateCardinalityWithSet";
			root["message"] = "focus_join_verbose_dump";
			root["runId"] = "single-join-focus";
			root["data"] = std::move(data);
			std::ofstream fo(nd_path, std::ios::app);
			if (fo.is_open()) {
				fo << root.dump() << '\n';
			}
		}
	}
	// #endregion

	auto new_entry = CardinalityHelper(result);
	new_entry.table_names_joined = joined_tables;

	relation_set_2_cardinality[join_cache_key] = new_entry;
	return result;
}

template <>
idx_t CardinalityEstimator::EstimateCardinalityWithSet(JoinRelationSet &new_set) {
	auto cardinality_as_double = EstimateCardinalityWithSet<double>(new_set);
	auto max = NumericLimits<idx_t>::Maximum();
	if (cardinality_as_double >= (double)max) {
		return max;
	}
	return (idx_t)cardinality_as_double;
}

bool SortTdoms(const RelationsSetToStats &a, const RelationsSetToStats &b) {
	if (a.has_distinct_count_hll && b.has_distinct_count_hll) {
		return a.distinct_count_hll > b.distinct_count_hll;
	}
	if (a.has_distinct_count_hll) {
		return a.distinct_count_hll > b.distinct_count_no_hll;
	}
	if (b.has_distinct_count_hll) {
		return a.distinct_count_no_hll > b.distinct_count_hll;
	}
	return a.distinct_count_no_hll > b.distinct_count_no_hll;
}

void CardinalityEstimator::InitCardinalityEstimatorProps(optional_ptr<JoinRelationSet> set, RelationStats &stats) {
	// Get the join relation set
	D_ASSERT(stats.stats_initialized);
	auto relation_cardinality = stats.cardinality;

	auto card_helper = CardinalityHelper((double)relation_cardinality);
	card_helper.table_names_joined.push_back(stats.table_name);
	relation_set_2_cardinality[set->ToString()] = card_helper;

	// Store column info for SQL query construction.
	if (set->count == 1) {
		auto relation_index = set->relations[0];
		RelationColumnInfo info;
		info.table_name = stats.table_name;
		info.column_names = stats.column_names;
		info.scan_filter_string = stats.scan_filter_string;
		relation_column_info[relation_index] = std::move(info);
	}

	UpdateTotalDomains(set, stats);

	// sort relations from greatest tdom to lowest tdom.
	std::sort(relation_set_stats.begin(), relation_set_stats.end(), SortTdoms);
}

void CardinalityEstimator::UpdateTotalDomains(optional_ptr<JoinRelationSet> set, RelationStats &stats) {
	D_ASSERT(set->count == 1);
	auto relation_id = set->relations[0];
	//! Initialize the distinct count for all columns used in joins with the current relation.
	//	D_ASSERT(stats.column_distinct_count.size() >= 1);

	for (idx_t i = 0; i < stats.column_distinct_count.size(); i++) {
		//! for every column used in a filter in the relation, get the distinct count via HLL, or assume it to be
		//! the cardinality
		// Update the relation_to_tdom set with the estimated distinct count (or tdom) calculated above
		auto key = ColumnBinding(relation_id, i);
		for (auto &relation_to_tdom : relation_set_stats) {
			column_binding_set_t i_set = relation_to_tdom.equivalent_relations;
			if (i_set.find(key) == i_set.end()) {
				continue;
			}
			auto distinct_count = stats.column_distinct_count.at(i);
			if (distinct_count.from_hll && relation_to_tdom.has_distinct_count_hll) {
				relation_to_tdom.distinct_count_hll =
				    MaxValue(relation_to_tdom.distinct_count_hll, distinct_count.distinct_count);
			} else if (distinct_count.from_hll && !relation_to_tdom.has_distinct_count_hll) {
				relation_to_tdom.has_distinct_count_hll = true;
				relation_to_tdom.distinct_count_hll = distinct_count.distinct_count;
			} else {
				relation_to_tdom.distinct_count_no_hll =
				    MinValue(distinct_count.distinct_count, relation_to_tdom.distinct_count_no_hll);
			}
			break;
		}
	}
}

// LCOV_EXCL_START

void CardinalityEstimator::AddRelationNamesToRelationStats(vector<RelationStats> &stats) {
#ifdef DEBUG
	for (auto &total_domain : relation_set_stats) {
		for (auto &binding : total_domain.equivalent_relations) {
			D_ASSERT(binding.table_index < stats.size());
			string column_name;
			if (binding.column_index < stats[binding.table_index].column_names.size()) {
				column_name = stats[binding.table_index].column_names[binding.column_index];
			} else {
				column_name = "[unknown]";
			}
			total_domain.column_names.push_back(column_name);
		}
	}
#endif
}

void CardinalityEstimator::PrintRelationStats() {
	for (auto &total_domain : relation_set_stats) {
		string domain = "Following columns have the same distinct count: ";
		for (auto &column_name : total_domain.column_names) {
			domain += column_name + ", ";
		}
		bool have_hll = total_domain.has_distinct_count_hll;
		domain += "\n TOTAL DOMAIN = " +
		          to_string(have_hll ? total_domain.distinct_count_hll : total_domain.distinct_count_no_hll);
		Printer::Print(domain);
	}
}

// LCOV_EXCL_STOP

} // namespace duckdb
