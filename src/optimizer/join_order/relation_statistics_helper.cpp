#include "duckdb/optimizer/join_order/relation_statistics_helper.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

#include <chrono>
#include <fstream>
#include <math.h>
#include <limits>

namespace duckdb {

static bool StatsExpressionTreeContainsIsNull(const Expression &expr) {
	if (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL) {
		return true;
	}
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COMPARISON) {
		auto &cmp = expr.Cast<BoundComparisonExpression>();
		if (cmp.GetExpressionType() == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			auto side_is_null_const = [](const unique_ptr<Expression> &side) {
				return side && side->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT &&
				       side->Cast<BoundConstantExpression>().value.IsNull();
			};
			if (side_is_null_const(cmp.left) || side_is_null_const(cmp.right)) {
				return true;
			}
		}
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && StatsExpressionTreeContainsIsNull(child)) {
			found = true;
		}
	});
	return found;
}

static bool ContainsEqualityPredicate(const TableFilter &filter) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &comparison_filter = filter.Cast<ConstantFilter>();
		return comparison_filter.comparison_type == ExpressionType::COMPARE_EQUAL;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		for (auto &child_filter : and_filter.child_filters) {
			if (ContainsEqualityPredicate(*child_filter)) {
				return true;
			}
		}
		return false;
	}
	default:
		return false;
	}
}

static bool ContainsIsNullTableFilter(const TableFilter &filter) {
	switch (filter.filter_type) {
	case TableFilterType::IS_NULL:
		return true;
	case TableFilterType::OPTIONAL_FILTER: {
		auto &opt = filter.Cast<OptionalFilter>();
		return opt.child_filter && ContainsIsNullTableFilter(*opt.child_filter);
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		for (auto &child_filter : and_filter.child_filters) {
			if (ContainsIsNullTableFilter(*child_filter)) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::EXPRESSION_FILTER: {
		auto &expr_filter = filter.Cast<ExpressionFilter>();
		return StatsExpressionTreeContainsIsNull(*expr_filter.expr);
	}
	default:
		return false;
	}
}

static ExpressionBinding GetChildColumnBinding(Expression &expr) {
	auto ret = ExpressionBinding();
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_FUNCTION: {
		// TODO: Other expression classes that can have 0 children?
		auto &func = expr.Cast<BoundFunctionExpression>();
		// no children some sort of gen_random_uuid() or equivalent.
		if (func.children.empty()) {
			ret.expression = expr;
			ret.expression_is_constant = true;
			return ret;
		}
		break;
	}
	case ExpressionClass::BOUND_COLUMN_REF: {
		ret.expression = expr;
		auto &new_col_ref = expr.Cast<BoundColumnRefExpression>();
		ret.child_binding = ColumnBinding(new_col_ref.binding.table_index, new_col_ref.binding.column_index);
		return ret;
	}
	case ExpressionClass::BOUND_LAMBDA_REF:
	case ExpressionClass::BOUND_CONSTANT:
	case ExpressionClass::BOUND_DEFAULT:
	case ExpressionClass::BOUND_PARAMETER:
	case ExpressionClass::BOUND_REF:
		ret.expression = expr;
		ret.expression_is_constant = true;
		return ret;
	default:
		break;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &child) {
		if (ret.FoundColumnRef()) {
			//! Already found a column ref expression
			return;
		}
		auto recursive_result = GetChildColumnBinding(*child);
		if (recursive_result.FoundExpression()) {
			ret = recursive_result;
			return;
		}
	});
	// we didn't find a Bound Column Ref
	return ret;
}

idx_t RelationStatisticsHelper::GetDistinctCount(LogicalGet &get, ClientContext &context,
                                                 const ColumnIndex &column_id) {
	if (!get.function.statistics && !get.function.statistics_extended) {
		return 0;
	}
	unique_ptr<BaseStatistics> column_statistics;
	if (get.function.statistics_extended) {
		TableFunctionGetStatisticsInput input(get.bind_data.get(), column_id);
		column_statistics = get.function.statistics_extended(context, input);
	} else {
		D_ASSERT(get.function.statistics);
		column_statistics = get.function.statistics(context, get.bind_data.get(), column_id.GetPrimaryIndex());
	}
	if (!column_statistics) {
		return 0;
	}
	auto distinct_count = column_statistics->GetDistinctCount();
	return distinct_count;
}

RelationStats RelationStatisticsHelper::ExtractGetStats(LogicalGet &get, ClientContext &context) {
	auto return_stats = RelationStats();

	auto base_table_cardinality = get.EstimateCardinality(context);
	auto cardinality_after_filters = base_table_cardinality;
	unique_ptr<BaseStatistics> column_statistics;

	auto catalog_table = get.GetTable();
	auto name = string("some table");
	if (catalog_table) {
		name = catalog_table->name;
		return_stats.table_name = name;
	}

	// first push back basic distinct counts for each column (if we have them).
	auto &column_ids = get.GetColumnIds();
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto column_id = column_ids[i].GetPrimaryIndex();
		auto distinct_count = GetDistinctCount(get, context, column_ids[i]);
		if (distinct_count > 0) {
			auto column_distinct_count = DistinctCount({distinct_count, true});
			return_stats.column_distinct_count.push_back(column_distinct_count);
			return_stats.column_names.push_back(name + "." + get.names.at(column_id));
		} else {
			// treat the cardinality as the distinct count.
			// the cardinality estimator will update these distinct counts based
			// on the extra columns that are joined on.
			auto column_distinct_count = DistinctCount({cardinality_after_filters, false});
			return_stats.column_distinct_count.push_back(column_distinct_count);
			auto column_name = string("column");
			if (column_id < get.names.size()) {
				column_name = get.names.at(column_id);
			}
			return_stats.column_names.push_back(get.GetName() + "." + column_name);
		}
	}

	if (!get.table_filters.filters.empty()) {
		column_statistics = nullptr;
		bool has_non_optional_filters = false;
		bool has_equality_predicate = false;
		for (auto &it : get.table_filters.filters) {
			column_statistics = nullptr;
			if (get.bind_data && (get.function.statistics || get.function.statistics_extended)) {
				if (get.function.statistics_extended) {
					auto column_index = ColumnIndex(it.first);
					TableFunctionGetStatisticsInput input(get.bind_data.get(), column_index);
					column_statistics = get.function.statistics_extended(context, input);
				} else {
					D_ASSERT(get.function.statistics);
					column_statistics = get.function.statistics(context, get.bind_data.get(), it.first);
				}
			}

			if (column_statistics) {
				idx_t cardinality_with_filter =
				    InspectTableFilter(cardinality_after_filters, it.first, *it.second, *column_statistics);
				cardinality_after_filters = MinValue(cardinality_after_filters, cardinality_with_filter);
			} else if (it.first < get.returned_types.size()) {
				// No per-column stats from the table function (common for catalog tables): still run filter
				// inspection so OPTIONAL-wrapped IS NULL, conjunctions, etc. can adjust cardinality.
				auto unknown_stats = BaseStatistics::CreateUnknown(get.returned_types[it.first]);
				idx_t cardinality_with_filter =
				    InspectTableFilter(cardinality_after_filters, it.first, *it.second, unknown_stats);
				cardinality_after_filters = MinValue(cardinality_after_filters, cardinality_with_filter);
			}

			has_equality_predicate = has_equality_predicate || ContainsEqualityPredicate(*it.second) ||
			                         ContainsIsNullTableFilter(*it.second);
			if (it.second->filter_type != TableFilterType::OPTIONAL_FILTER) {
				has_non_optional_filters = true;
			}
			// #region agent log
			{
				auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
				              std::chrono::system_clock::now().time_since_epoch())
				              .count();
				std::ofstream alog("/Users/Aarry/Desktop/15689/duckdb15689/.cursor/debug-9c089b.log",
				                   std::ios::app);
				if (alog) {
					alog << "{\"sessionId\":\"9c089b\",\"hypothesisId\":\"A\",\"location\":"
					        "\"relation_statistics_helper.cpp:ExtractGetStats\","
					     << "\"message\":\"table_filter\",\"data\":{\"table\":\"" << name
					     << "\",\"col\":" << it.first << ",\"filter_type\":" << (int)it.second->filter_type
					     << ",\"expr_has_is_null\":" << (ContainsIsNullTableFilter(*it.second) ? 1 : 0)
					     << ",\"card_after\":" << cardinality_after_filters << ",\"base_card\":"
					     << base_table_cardinality << "},\"timestamp\":" << ts << "}\n";
				}
			}
			// #endregion
		}
		// if the above code didn't find an equality filter (i.e country_code = "[us]")
		// and there are other table filters (i.e cost > 50), use default selectivity.
		bool has_equality_filter = (cardinality_after_filters != base_table_cardinality) || has_equality_predicate;
		if (!has_equality_filter && has_non_optional_filters) {
			cardinality_after_filters = MaxValue<idx_t>(
			    LossyNumericCast<idx_t>(double(base_table_cardinality) * RelationStatisticsHelper::DEFAULT_SELECTIVITY),
			    1U);
		}
		if (base_table_cardinality == 0) {
			cardinality_after_filters = 0;
		}
		// If pushed-down scan filters reduced cardinality, reduce per-column distinct
		// counts proportionally so join tdoms reflect the filtered domain.
		if (base_table_cardinality > 0 && cardinality_after_filters < base_table_cardinality) {
			auto ratio = static_cast<double>(cardinality_after_filters) / static_cast<double>(base_table_cardinality);
			for (auto &dc : return_stats.column_distinct_count) {
				auto scaled = LossyNumericCast<idx_t>(ceil(static_cast<double>(dc.distinct_count) * ratio));
				scaled = MinValue(dc.distinct_count, scaled);
				dc.distinct_count = cardinality_after_filters == 0 ? 0 : MaxValue<idx_t>(scaled, 1);
			}
		}

		// Build a SQL WHERE-clause fragment from the pushed-down table filters so that
		// callers can construct accurate oracle COUNT(*) subqueries.
		string scan_filter_sql;
		for (auto &it : get.table_filters.filters) {
			if (it.second->filter_type == TableFilterType::OPTIONAL_FILTER) {
				continue; // skip bloom/optional filters
			}
			// Map the pushed-down column index to the actual column name.
			string col_name;
			if (it.first < get.names.size()) {
				col_name = get.names.at(it.first);
			} else {
				continue; // can't resolve name, skip
			}
			if (!scan_filter_sql.empty()) {
				scan_filter_sql += " AND ";
			}
			scan_filter_sql += it.second->ToString(col_name);
		}
		return_stats.scan_filter_string = scan_filter_sql;
	}
	return_stats.cardinality = cardinality_after_filters;
	// update the estimated cardinality of the get as well.
	// This is not updated during plan reconstruction.
	get.estimated_cardinality = cardinality_after_filters;
	get.has_estimated_cardinality = true;
	D_ASSERT(base_table_cardinality >= cardinality_after_filters);
	return_stats.stats_initialized = true;
	return return_stats;
}

RelationStats RelationStatisticsHelper::ExtractDelimGetStats(LogicalDelimGet &delim_get, ClientContext &context) {
	RelationStats stats;
	stats.table_name = delim_get.GetName();
	idx_t card = delim_get.EstimateCardinality(context);
	stats.cardinality = card;
	stats.stats_initialized = true;
	for (auto &binding : delim_get.GetColumnBindings()) {
		stats.column_distinct_count.push_back(DistinctCount({1, false}));
		stats.column_names.push_back("column" + to_string(binding.column_index));
	}
	return stats;
}

RelationStats RelationStatisticsHelper::ExtractProjectionStats(LogicalProjection &proj, RelationStats &child_stats) {
	auto proj_stats = RelationStats();
	proj_stats.cardinality = child_stats.cardinality;
	proj_stats.table_name = child_stats.table_name.empty() ? proj.GetName() : child_stats.table_name;
	for (auto &expr : proj.expressions) {
		proj_stats.column_names.push_back(expr->GetName());
		auto res = GetChildColumnBinding(*expr);
		D_ASSERT(res.FoundExpression());
		if (res.expression_is_constant) {
			proj_stats.column_distinct_count.push_back(DistinctCount({1, true}));
		} else {
			auto column_index = res.child_binding.column_index;
			if (column_index >= child_stats.column_distinct_count.size() && expr->ToString() == "count_star()") {
				// only one value for a count star
				proj_stats.column_distinct_count.push_back(DistinctCount({1, true}));
			} else {
				// TODO: add this back in
				//	D_ASSERT(column_index < stats.column_distinct_count.size());
				if (column_index < child_stats.column_distinct_count.size()) {
					proj_stats.column_distinct_count.push_back(child_stats.column_distinct_count.at(column_index));
				} else {
					proj_stats.column_distinct_count.push_back(DistinctCount({proj_stats.cardinality, false}));
				}
			}
		}
	}
	proj_stats.stats_initialized = true;
	return proj_stats;
}

RelationStats RelationStatisticsHelper::ExtractDummyScanStats(LogicalDummyScan &dummy_scan, ClientContext &context) {
	auto stats = RelationStats();
	idx_t card = dummy_scan.EstimateCardinality(context);
	stats.cardinality = card;
	for (idx_t i = 0; i < dummy_scan.GetColumnBindings().size(); i++) {
		stats.column_distinct_count.push_back(DistinctCount({card, false}));
		stats.column_names.push_back("dummy_scan_column");
	}
	stats.stats_initialized = true;
	stats.table_name = "dummy scan";
	return stats;
}

void RelationStatisticsHelper::CopyRelationStats(RelationStats &to, const RelationStats &from) {
	to.column_distinct_count = from.column_distinct_count;
	to.column_names = from.column_names;
	to.cardinality = from.cardinality;
	to.table_name = from.table_name;
	to.stats_initialized = from.stats_initialized;
	to.scan_filter_string = from.scan_filter_string;
}

RelationStats RelationStatisticsHelper::CombineStatsOfReorderableOperator(vector<ColumnBinding> &bindings,
                                                                          vector<RelationStats> relation_stats) {
	RelationStats stats;
	idx_t max_card = 0;
	for (auto &child_stats : relation_stats) {
		for (idx_t i = 0; i < child_stats.column_distinct_count.size(); i++) {
			stats.column_distinct_count.push_back(child_stats.column_distinct_count.at(i));
			stats.column_names.push_back(child_stats.column_names.at(i));
		}
		if (!child_stats.table_name.empty()) {
			if (!stats.table_name.empty()) {
				stats.table_name += ", ";
			}
			stats.table_name += child_stats.table_name;
		}
		max_card = MaxValue(max_card, child_stats.cardinality);
	}
	stats.stats_initialized = true;
	stats.cardinality = max_card;
	return stats;
}

RelationStats RelationStatisticsHelper::CombineStatsOfNonReorderableOperator(LogicalOperator &op,
                                                                             const vector<RelationStats> &child_stats) {
	RelationStats ret;
	ret.cardinality = 0;

	// default predicted cardinality is the max of all child cardinalities
	vector<idx_t> child_cardinalities;
	for (auto &stats : child_stats) {
		idx_t child_cardinality = stats.stats_initialized ? stats.cardinality : 0;
		ret.cardinality = MaxValue(ret.cardinality, child_cardinality);
		child_cardinalities.push_back(child_cardinality);
	}
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		D_ASSERT(child_stats.size() == 2);
		auto &join = op.Cast<LogicalComparisonJoin>();
		switch (join.join_type) {
		case JoinType::RIGHT_ANTI:
		case JoinType::RIGHT_SEMI:
			ret.cardinality = child_cardinalities[1];
			break;
		case JoinType::ANTI:
		case JoinType::SEMI:
		case JoinType::SINGLE:
		case JoinType::MARK:
			ret.cardinality = child_cardinalities[0];
			break;
		default:
			break;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_UNION: {
		auto &setop = op.Cast<LogicalSetOperation>();
		if (setop.setop_all) {
			// setop returns all records
			ret.cardinality = 0;
			for (auto &child_cardinality : child_cardinalities) {
				ret.cardinality += child_cardinality;
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_INTERSECT: {
		D_ASSERT(child_stats.size() == 2);
		ret.cardinality = MinValue(child_cardinalities[0], child_cardinalities[1]);
		break;
	}
	case LogicalOperatorType::LOGICAL_EXCEPT: {
		D_ASSERT(child_stats.size() == 2);
		ret.cardinality = child_cardinalities[0];
		break;
	}
	default:
		break;
	}

	ret.stats_initialized = true;
	ret.filter_strength = 1;
	ret.table_name = string();
	for (auto &stats : child_stats) {
		if (!stats.table_name.empty()) {
			if (!ret.table_name.empty()) {
				ret.table_name += ", ";
			}
			ret.table_name += stats.table_name;
		}
		// MARK joins are nonreorderable. They won't return initialized stats
		// continue in this case.
		if (!stats.stats_initialized) {
			continue;
		}
		for (auto &distinct_count : stats.column_distinct_count) {
			ret.column_distinct_count.push_back(distinct_count);
		}
		for (auto &column_name : stats.column_names) {
			ret.column_names.push_back(column_name);
		}
	}
	return ret;
}

RelationStats RelationStatisticsHelper::ExtractExpressionGetStats(LogicalExpressionGet &expression_get,
                                                                  ClientContext &context) {
	auto stats = RelationStats();
	idx_t card = expression_get.EstimateCardinality(context);
	stats.cardinality = card;
	for (idx_t i = 0; i < expression_get.GetColumnBindings().size(); i++) {
		stats.column_distinct_count.push_back(DistinctCount({card, false}));
		stats.column_names.push_back("expression_get_column");
	}
	stats.stats_initialized = true;
	stats.table_name = "expression_get";
	return stats;
}

RelationStats RelationStatisticsHelper::ExtractWindowStats(LogicalWindow &window, RelationStats &child_stats) {
	RelationStats stats;
	stats.cardinality = child_stats.cardinality;
	stats.column_distinct_count = child_stats.column_distinct_count;
	stats.column_names = child_stats.column_names;
	stats.table_name = child_stats.table_name.empty() ? window.GetName() : child_stats.table_name;
	stats.stats_initialized = true;
	auto num_child_columns = window.GetColumnBindings().size();

	for (idx_t column_index = child_stats.column_distinct_count.size(); column_index < num_child_columns;
	     column_index++) {
		stats.column_distinct_count.push_back(DistinctCount({child_stats.cardinality, false}));
		stats.column_names.push_back("window");
	}
	return stats;
}

RelationStats RelationStatisticsHelper::ExtractAggregationStats(LogicalAggregate &aggr, RelationStats &child_stats) {
	RelationStats stats;
	// TODO: look at child distinct count to better estimate cardinality.
	stats.cardinality = child_stats.cardinality;
	stats.column_distinct_count = child_stats.column_distinct_count;
	vector<double> distinct_counts;
	for (auto &g_set : aggr.grouping_sets) {
		vector<double> set_distinct_counts;
		for (auto &ind : g_set) {
			if (aggr.groups[ind]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				continue;
			}
			auto bound_col = &aggr.groups[ind]->Cast<BoundColumnRefExpression>();
			auto col_index = bound_col->binding.column_index;
			if (col_index >= child_stats.column_distinct_count.size()) {
				// it is possible the column index of the grouping_set is not in the child stats.
				// this can happen when delim joins are present, since delim scans are not currently
				// reorderable. Meaning they don't add a relation or column_ids that could potentially
				// be grouped by. Hopefully this can be fixed with duckdb-internal#606
				continue;
			}
			double distinct_count = static_cast<double>(child_stats.column_distinct_count[col_index].distinct_count);
			set_distinct_counts.push_back(distinct_count == 0 ? 1 : distinct_count);
		}
		// We use the grouping set with the most group key columns for cardinality estimation
		if (set_distinct_counts.size() > distinct_counts.size()) {
			distinct_counts = std::move(set_distinct_counts);
		}
	}

	double new_card;
	if (distinct_counts.empty()) {
		// We have no good statistics on distinct count.
		// most likely we are running on parquet files. Therefore we divide by 2.
		new_card = static_cast<double>(child_stats.cardinality) / 2.0;
	} else {
		// Multiply distinct counts
		double product = 1;
		for (const auto &distinct_count : distinct_counts) {
			product *= distinct_count;
		}

		// Assume slight correlation for each grouping column
		const auto correction = pow(0.95, static_cast<double>(distinct_counts.size() - 1));
		product *= correction;

		// Estimate using the "Occupancy Problem",
		// where "product" is number of bins, and "child_stats.cardinality" is number of balls
		const auto mult = 1.0 - exp(-static_cast<double>(child_stats.cardinality) / product);
		if (mult == 0) { // Can become 0 with very large estimates due to double imprecision
			new_card = static_cast<double>(child_stats.cardinality);
		} else {
			new_card = product * mult;
		}
		new_card = MinValue(new_card, static_cast<double>(child_stats.cardinality));
	}

	// an ungrouped aggregate has 1 row
	stats.cardinality = aggr.groups.empty() ? 1 : LossyNumericCast<idx_t>(new_card);
	stats.column_names = child_stats.column_names;
	stats.table_name = child_stats.table_name.empty() ? aggr.GetName() : child_stats.table_name;
	stats.stats_initialized = true;
	const auto aggr_column_bindings = aggr.GetColumnBindings();
	auto num_child_columns = aggr_column_bindings.size();

	for (idx_t column_index = 0; column_index < num_child_columns; column_index++) {
		const auto &binding = aggr_column_bindings[column_index];
		if (binding.table_index == aggr.group_index && column_index < distinct_counts.size()) {
			// Group column that we have the HLL of
			stats.column_distinct_count.push_back(
			    DistinctCount({LossyNumericCast<idx_t>(distinct_counts[column_index]), true}));
		} else {
			// Non-group column, or we don't have the HLL
			stats.column_distinct_count.push_back(DistinctCount({child_stats.cardinality, false}));
		}
		stats.column_names.push_back("aggregate");
	}
	return stats;
}

RelationStats RelationStatisticsHelper::ExtractEmptyResultStats(LogicalEmptyResult &empty) {
	RelationStats stats;
	for (idx_t i = 0; i < empty.GetColumnBindings().size(); i++) {
		stats.column_distinct_count.push_back(DistinctCount({0, false}));
		stats.column_names.push_back("empty_result_column");
	}
	stats.stats_initialized = true;
	return stats;
}

idx_t RelationStatisticsHelper::InspectTableFilter(idx_t cardinality, idx_t column_index, const TableFilter &filter,
                                                   BaseStatistics &base_stats) {
	auto cardinality_after_filters = cardinality;
	switch (filter.filter_type) {
	case TableFilterType::CONJUNCTION_AND: {
		auto &and_filter = filter.Cast<ConjunctionAndFilter>();
		// Special-case date range conjunctions so we don't fall back to coarse DEFAULT_SELECTIVITY.
		// This targets patterns like d_date >= a AND d_date <= b that currently regress to 0.2.
		if (base_stats.GetType().id() == LogicalTypeId::DATE && NumericStats::HasMinMax(base_stats)) {
			auto min_day = NumericStats::GetMin<int32_t>(base_stats);
			auto max_day = NumericStats::GetMax<int32_t>(base_stats);
			int32_t lower_day = min_day;
			int32_t upper_day = max_day;
			bool saw_supported_date_predicate = false;
			bool unsupported_child = false;
			for (auto &child_filter : and_filter.child_filters) {
				if (child_filter->filter_type != TableFilterType::CONSTANT_COMPARISON) {
					unsupported_child = true;
					continue;
				}
				auto &comparison_filter = child_filter->Cast<ConstantFilter>();
				if (comparison_filter.constant.IsNull()) {
					unsupported_child = true;
					continue;
				}
				Value constant_val;
				if (!comparison_filter.constant.DefaultTryCastAs(LogicalType::DATE, constant_val, nullptr, false)) {
					unsupported_child = true;
					continue;
				}
				auto c_day = constant_val.GetValueUnsafe<date_t>().days;
				switch (comparison_filter.comparison_type) {
				case ExpressionType::COMPARE_GREATERTHAN:
					lower_day = MaxValue(lower_day, c_day + 1);
					saw_supported_date_predicate = true;
					break;
				case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
					lower_day = MaxValue(lower_day, c_day);
					saw_supported_date_predicate = true;
					break;
				case ExpressionType::COMPARE_LESSTHAN:
					upper_day = MinValue(upper_day, c_day - 1);
					saw_supported_date_predicate = true;
					break;
				case ExpressionType::COMPARE_LESSTHANOREQUALTO:
					upper_day = MinValue(upper_day, c_day);
					saw_supported_date_predicate = true;
					break;
				case ExpressionType::COMPARE_EQUAL:
					lower_day = MaxValue(lower_day, c_day);
					upper_day = MinValue(upper_day, c_day);
					saw_supported_date_predicate = true;
					break;
				default:
					unsupported_child = true;
					break;
				}
			}
			if (saw_supported_date_predicate && !unsupported_child) {
				if (lower_day > upper_day) {
					return 0;
				}
				auto domain = static_cast<double>(max_day) - static_cast<double>(min_day) + 1.0;
				if (domain > 0.0) {
					auto selected = static_cast<double>(upper_day) - static_cast<double>(lower_day) + 1.0;
					selected = MaxValue<double>(0.0, MinValue<double>(selected, domain));
					auto selectivity = selected / domain;
					auto range_cardinality =
					    MaxValue<idx_t>(LossyNumericCast<idx_t>(ceil(static_cast<double>(cardinality) * selectivity)), 1U);
					auto out_cardinality = MinValue(cardinality_after_filters, range_cardinality);
					return out_cardinality;
				}
			}
		}
		// Handle integer range predicates (e.g., d_year >= a AND d_year <= b) with min/max stats.
		if (base_stats.GetType().id() == LogicalTypeId::BIGINT && NumericStats::HasMinMax(base_stats)) {
			auto min_val = NumericStats::GetMin<int64_t>(base_stats);
			auto max_val = NumericStats::GetMax<int64_t>(base_stats);
			int64_t lower_val = min_val;
			int64_t upper_val = max_val;
			bool saw_supported_bigint_predicate = false;
			bool unsupported_child = false;
			for (auto &child_filter : and_filter.child_filters) {
				if (child_filter->filter_type != TableFilterType::CONSTANT_COMPARISON) {
					unsupported_child = true;
					continue;
				}
				auto &comparison_filter = child_filter->Cast<ConstantFilter>();
				if (comparison_filter.constant.IsNull()) {
					unsupported_child = true;
					continue;
				}
				Value constant_val;
				if (!comparison_filter.constant.DefaultTryCastAs(LogicalType::BIGINT, constant_val, nullptr, false)) {
					unsupported_child = true;
					continue;
				}
				auto c_val = constant_val.GetValueUnsafe<int64_t>();
				switch (comparison_filter.comparison_type) {
				case ExpressionType::COMPARE_GREATERTHAN:
					lower_val = MaxValue(lower_val,
					                     c_val < std::numeric_limits<int64_t>::max() ? c_val + 1 : c_val);
					saw_supported_bigint_predicate = true;
					break;
				case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
					lower_val = MaxValue(lower_val, c_val);
					saw_supported_bigint_predicate = true;
					break;
				case ExpressionType::COMPARE_LESSTHAN:
					upper_val = MinValue(upper_val,
					                     c_val > std::numeric_limits<int64_t>::min() ? c_val - 1 : c_val);
					saw_supported_bigint_predicate = true;
					break;
				case ExpressionType::COMPARE_LESSTHANOREQUALTO:
					upper_val = MinValue(upper_val, c_val);
					saw_supported_bigint_predicate = true;
					break;
				case ExpressionType::COMPARE_EQUAL:
					lower_val = MaxValue(lower_val, c_val);
					upper_val = MinValue(upper_val, c_val);
					saw_supported_bigint_predicate = true;
					break;
				default:
					unsupported_child = true;
					break;
				}
			}
			if (saw_supported_bigint_predicate && !unsupported_child) {
				if (lower_val > upper_val) {
					return 0;
				}
				auto domain = static_cast<double>(max_val) - static_cast<double>(min_val) + 1.0;
				if (domain > 0.0) {
					auto selected = static_cast<double>(upper_val) - static_cast<double>(lower_val) + 1.0;
					selected = MaxValue<double>(0.0, MinValue<double>(selected, domain));
					auto selectivity = selected / domain;
					auto range_cardinality =
					    MaxValue<idx_t>(LossyNumericCast<idx_t>(ceil(static_cast<double>(cardinality) * selectivity)), 1U);
					auto out_cardinality = MinValue(cardinality_after_filters, range_cardinality);
					return out_cardinality;
				}
			}
		}
		for (auto &child_filter : and_filter.child_filters) {
			cardinality_after_filters = MinValue(
			    cardinality_after_filters, InspectTableFilter(cardinality, column_index, *child_filter, base_stats));
		}
		return cardinality_after_filters;
	}
	case TableFilterType::IS_NULL: {
		// Mirror IsNullFilter::CheckStatistics: empty result vs all-null vs unknown null fraction.
		if (!base_stats.CanHaveNull()) {
			return 0;
		}
		if (!base_stats.CanHaveNoNull()) {
			return cardinality;
		}
		auto est = LossyNumericCast<idx_t>(ceil(static_cast<double>(cardinality) *
		                                        RelationStatisticsHelper::DEFAULT_IS_NULL_SELECTIVITY));
		return MaxValue<idx_t>(1, MinValue(cardinality, est));
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &opt = filter.Cast<OptionalFilter>();
		if (!opt.child_filter) {
			return cardinality_after_filters;
		}
		return InspectTableFilter(cardinality, column_index, *opt.child_filter, base_stats);
	}
	case TableFilterType::EXPRESSION_FILTER: {
		auto &expr_filter = filter.Cast<ExpressionFilter>();
		if (!StatsExpressionTreeContainsIsNull(*expr_filter.expr)) {
			return cardinality_after_filters;
		}
		if (!base_stats.CanHaveNull()) {
			return 0;
		}
		if (!base_stats.CanHaveNoNull()) {
			return cardinality;
		}
		auto est = LossyNumericCast<idx_t>(ceil(static_cast<double>(cardinality) *
		                                        RelationStatisticsHelper::DEFAULT_IS_NULL_SELECTIVITY));
		return MaxValue<idx_t>(1, MinValue(cardinality, est));
	}
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &comparison_filter = filter.Cast<ConstantFilter>();
		if (comparison_filter.comparison_type != ExpressionType::COMPARE_EQUAL) {
			return cardinality_after_filters;
		}
		auto column_count = base_stats.GetDistinctCount();
		// column_count = 0 when there is no column count (i.e parquet scans)
		if (column_count > 0) {
			auto used_distinct = column_count;
			// For small-domain BIGINT columns, prefer max(HLL distinct, min/max domain size).
			if (base_stats.GetType().id() == LogicalTypeId::BIGINT && NumericStats::HasMinMax(base_stats)) {
				auto min_val = NumericStats::GetMin<int64_t>(base_stats);
				auto max_val = NumericStats::GetMax<int64_t>(base_stats);
				if (max_val >= min_val) {
					auto domain_span = static_cast<uint64_t>(max_val - min_val) + 1;
					if (domain_span <= static_cast<uint64_t>(10000)) {
						auto domain_distinct = LossyNumericCast<idx_t>(domain_span);
						used_distinct = MaxValue<idx_t>(used_distinct, domain_distinct);
					}
				}
			}
			// we want the ceil of cardinality/column_count. We also want to avoid compiler errors
			cardinality_after_filters = (cardinality + used_distinct - 1) / used_distinct;
		}
		return cardinality_after_filters;
	}
	default:
		return cardinality_after_filters;
	}
}

// TODO: Currently only simple AND filters are pushed into table scans.
//  When OR filters are pushed this function can be added
// idx_t RelationStatisticsHelper::InspectConjunctionOR(idx_t cardinality, idx_t column_index, ConjunctionOrFilter
// &filter,
//                                                     BaseStatistics &base_stats) {
//	auto has_equality_filter = false;
//	auto cardinality_after_filters = cardinality;
//	for (auto &child_filter : filter.child_filters) {
//		if (child_filter->filter_type != TableFilterType::CONSTANT_COMPARISON) {
//			continue;
//		}
//		auto &comparison_filter = child_filter->Cast<ConstantFilter>();
//		if (comparison_filter.comparison_type == ExpressionType::COMPARE_EQUAL) {
//			auto column_count = base_stats.GetDistinctCount();
//			auto increment = MaxValue<idx_t>(((cardinality + column_count - 1) / column_count), 1);
//			if (has_equality_filter) {
//				cardinality_after_filters += increment;
//			} else {
//				cardinality_after_filters = increment;
//			}
//			has_equality_filter = true;
//		}
//		if (child_filter->filter_type == TableFilterType::CONJUNCTION_AND) {
//			auto &and_filter = child_filter->Cast<ConjunctionAndFilter>();
//			cardinality_after_filters = RelationStatisticsHelper::InspectConjunctionAND(
//			    cardinality_after_filters, column_index, and_filter, base_stats);
//			continue;
//		}
//	}
//	D_ASSERT(cardinality_after_filters > 0);
//	return cardinality_after_filters;
//}

} // namespace duckdb
