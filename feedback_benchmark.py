"""
TPC-DS Cardinality Feedback Benchmark
======================================
Iteratively runs TPC-DS queries, captures actual join cardinalities from
the physical plan, injects them back into the optimizer via actual_cardinality.json,
and repeats until the physical plan converges (stops changing).

Query SQL is loaded from ``feedback_queries/tpcds/qNN.sql`` (see ``feedback_queries/README.md``).
Uses JSON profiling for structured plan trees.

Environment:
  DUCKDB_FEEDBACK_DB, DUCKDB_FEEDBACK_SF — database path and label for the banner
  DUCKDB_FEEDBACK_MAX_QUERIES — optional **cap on how many queries run** (Q1…QN only).
  Useful for fast smoke tests (e.g. ``MAX_QUERIES=15``); unset runs all 99. Not “MOQ” / minimum
  order quantity — it is only a **max query count** limit.
"""

import ast
import hashlib
import json
import os
import re
import csv
import traceback
from collections import defaultdict
import subprocess
from collections import Counter

# --- Repo paths ---
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
DUCKDB_BIN = os.path.join(REPO_ROOT, "build/release/duckdb")
CARDINALITY_LOG = os.path.join(REPO_ROOT, "cardinality_log.txt")
ACTUAL_CARDINALITY_JSON = os.path.join(REPO_ROOT, "actual_cardinality.json")
PROFILE_OUTPUT = os.path.join(REPO_ROOT, "profile_output.json")
TPCDS_QUERY_DIR = os.path.join(REPO_ROOT, "feedback_queries", "tpcds")

# --- Run configuration ---
DB_FILE = os.environ.get("DUCKDB_FEEDBACK_DB", "/Users/Aarry/Desktop/15689/tpcds_sf10.db").strip()
SCALE_FACTOR = int(os.environ.get("DUCKDB_FEEDBACK_SF", "10"))
TARGET_QUERIES = list(range(1, 100))
_MAX_QUERIES_ENV = os.environ.get("DUCKDB_FEEDBACK_MAX_QUERIES")
if _MAX_QUERIES_ENV:
    _nq = max(1, min(99, int(_MAX_QUERIES_ENV.strip())))
    TARGET_QUERIES = list(range(1, _nq + 1))
MAX_ITERATIONS = 20
MAIN_QUERY_TIMEOUT_SEC = int(os.getenv("DUCKDB_BENCHMARK_MAIN_QUERY_TIMEOUT_SEC", "600"))

# --- Matching / JSON drift thresholds ---
STABLE_PLAN_ABS_DRIFT_TOLERANCE = 5000
STABLE_PLAN_REL_DRIFT_TOLERANCE = 0.005  # 0.5%
LARGE_DELTA_ABS_THRESHOLD = 100000
LARGE_DELTA_REL_THRESHOLD = 0.50
PLAN_FINGERPRINT_ENV_VAR = "DUCKDB_FEEDBACK_PLAN_FINGERPRINT"
PLAN_KEY_PREFIX = "PLANFP:"

JOIN_OPERATOR_NAMES = {
    "HASH_JOIN",
    "NESTED_LOOP_JOIN",
    "PIECEWISE_MERGE_JOIN",
    "CROSS_PRODUCT",
    "POSITIONAL_JOIN",
    "BLOCKWISE_NL_JOIN",
    "IE_JOIN",
    "ASOF_JOIN",
    "DELIM_JOIN",
    "LEFT_DELIM_JOIN",
    "RIGHT_DELIM_JOIN",
}


def base_table_name(table_path):
    """Last component of a qualified table name (schema.table -> table)."""
    return str(table_path).split(".")[-1]


def compute_plan_fingerprint(plan_text):
    """Short hash of plan structure text (for stable-plan checks and namespaced keys)."""
    return hashlib.sha1(plan_text.encode("utf-8")).hexdigest()[:12]


def make_namespaced_expression_key(expression, plan_fingerprint):
    if not plan_fingerprint:
        return expression
    return f"{PLAN_KEY_PREFIX}{plan_fingerprint}::{expression}"


def decode_namespaced_expression_key(key):
    if key.startswith(PLAN_KEY_PREFIX) and "::" in key:
        rest = key[len(PLAN_KEY_PREFIX):]
        plan_fingerprint, expression = rest.split("::", 1)
        return plan_fingerprint, expression
    return None, key


def project_json_for_fingerprint(card_map, plan_fingerprint):
    """
    Convert raw JSON key map into {expression: cardinality} for one fingerprint.
    If fingerprint-specific keys exist for an expression, prefer those.
    Otherwise fall back to legacy unnamespaced keys.
    """
    scoped = {}
    has_fingerprint_value = set()
    for key, value in card_map.items():
        fp, expression = decode_namespaced_expression_key(key)
        if fp == plan_fingerprint:
            scoped[expression] = value
            has_fingerprint_value.add(expression)
        elif fp is None and expression not in has_fingerprint_value:
            scoped.setdefault(expression, value)
    return scoped


def _collect_tables_with_cte(node, cte_lineage):
    tables = set()
    lineage_incomplete = False
    op_name = node.get("operator_name", node.get("name", ""))
    extra = node.get("extra_info", {})

    if op_name == "SEQ_SCAN" and "Table" in extra:
        tables.add(base_table_name(extra["Table"]))
    elif op_name == "CTE_SCAN":
        cte_idx = str(extra.get("CTE Index", "")).strip()
        if cte_idx and cte_idx in cte_lineage and cte_lineage[cte_idx]:
            tables.update(cte_lineage[cte_idx])
        else:
            lineage_incomplete = True

    for child in node.get("children", []):
        child_tables, child_incomplete = _collect_tables_with_cte(child, cte_lineage)
        tables.update(child_tables)
        lineage_incomplete = lineage_incomplete or child_incomplete
    return tables, lineage_incomplete


def build_cte_lineage(root):
    """
    Build map {cte_table_index: set(base_tables)} from CTE definitions.
    We use the first child of a CTE node as the CTE-defining subtree.
    """
    cte_def_nodes = {}

    def collect(node):
        op_name = node.get("operator_name", node.get("name", ""))
        extra = node.get("extra_info", {})
        if op_name == "CTE" and "Table Index" in extra:
            idx = str(extra["Table Index"]).strip()
            if idx and node.get("children"):
                cte_def_nodes[idx] = node["children"][0]
        for child in node.get("children", []):
            collect(child)

    collect(root)
    cte_lineage = {idx: set() for idx in cte_def_nodes}

    # Iteratively resolve CTE dependencies (if a CTE references another CTE).
    for _ in range(len(cte_def_nodes) + 2):
        changed = False
        for idx, def_node in cte_def_nodes.items():
            tables, _ = _collect_tables_with_cte(def_node, cte_lineage)
            if tables != cte_lineage[idx]:
                cte_lineage[idx] = tables
                changed = True
        if not changed:
            break
    return cte_lineage


# ============================================================================
# TPC-DS QUERY FILES (feedback_queries/tpcds/qNN.sql)
# ============================================================================


def load_tpcds_queries(query_nrs):
    """
    Load SQL for each query number from ``feedback_queries/tpcds/q01.sql`` … ``q99.sql``.
    Missing files are skipped (caller may treat as absent query).
    """
    out = {}
    for nr in query_nrs:
        path = os.path.join(TPCDS_QUERY_DIR, f"q{nr:02d}.sql")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            out[nr] = f.read().strip() + "\n"
    return out


# ============================================================================
# FILE MANAGEMENT
# ============================================================================

def clear_cardinality_log():
    """Truncate cardinality_log.txt to empty."""
    with open(CARDINALITY_LOG, "w") as f:
        pass


def clear_actual_cardinality_json():
    """Delete actual_cardinality.json if it exists, so DuckDB starts with no injections."""
    if os.path.exists(ACTUAL_CARDINALITY_JSON):
        os.remove(ACTUAL_CARDINALITY_JSON)


def read_actual_cardinality_json():
    """
    Read the current actual_cardinality.json. Returns a dict {expression: cardinality}.
    Returns empty dict if the file does not exist.
    """
    if not os.path.exists(ACTUAL_CARDINALITY_JSON):
        return {}
    with open(ACTUAL_CARDINALITY_JSON, "r") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)


def write_actual_cardinality_json(data):
    """Write the cardinality map to actual_cardinality.json."""
    with open(ACTUAL_CARDINALITY_JSON, "w") as f:
        json.dump(data, f, indent=2)


def purge_unsafe_expressions_from_json(unsafe_expressions, *, query_nr=None, iteration=None):
    """
    Remove quarantined expressions from actual_cardinality.json so they cannot
    be injected in subsequent iterations.

    Also removes namespaced variants whose decoded expression matches.
    """
    if not unsafe_expressions:
        return 0

    current = read_actual_cardinality_json()
    if not current:
        return 0

    keys_to_delete = []
    for key in list(current.keys()):
        _, decoded_expr = decode_namespaced_expression_key(key)
        if key in unsafe_expressions or decoded_expr in unsafe_expressions:
            keys_to_delete.append(key)

    if not keys_to_delete:
        return 0

    for key in keys_to_delete:
        del current[key]

    write_actual_cardinality_json(current)
    print(
        f"    [QUARANTINE-DELETE] Removed {len(keys_to_delete)} unsafe key(s) "
        f"from actual_cardinality.json."
    )
    return len(keys_to_delete)


# ============================================================================
# QUERY EXECUTION
# ============================================================================

def child_duckdb_env(base_env=None):
    """
    Environment passed to the DuckDB subprocess. Must set DUCKDB_ACTUAL_CARDINALITY_JSON
    and DUCKDB_CARDINALITY_LOG so the forked binary reads/writes the same paths as this
    script (the C++ defaults point elsewhere).
    """
    env = (base_env if base_env is not None else os.environ).copy()
    env["DUCKDB_ACTUAL_CARDINALITY_JSON"] = ACTUAL_CARDINALITY_JSON
    env["DUCKDB_CARDINALITY_LOG"] = CARDINALITY_LOG
    return env


def run_query_with_json_profile(query_sql, plan_fingerprint_hint=None):
    """
    Run a query using DuckDB with JSON profiling enabled.
    Returns the parsed JSON profile dict, or None on failure.
    """
    # Remove any existing profile output
    if os.path.exists(PROFILE_OUTPUT):
        os.remove(PROFILE_OUTPUT)

    full_sql = (
        f"PRAGMA enable_profiling = 'json';\n"
        f"PRAGMA profiling_mode = 'detailed';\n"
        f"PRAGMA profiling_output = '{PROFILE_OUTPUT}';\n"
        f"PRAGMA enable_progress_bar = false;\n"
        + query_sql + "\n"
    )

    run_env = os.environ.copy()
    if plan_fingerprint_hint:
        run_env[PLAN_FINGERPRINT_ENV_VAR] = plan_fingerprint_hint
    else:
        run_env.pop(PLAN_FINGERPRINT_ENV_VAR, None)
    run_env = child_duckdb_env(run_env)

    try:
        proc = subprocess.run(
            [DUCKDB_BIN, DB_FILE, "-c", full_sql],
            capture_output=True, text=True, env=run_env, timeout=MAIN_QUERY_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Profile run TIMEOUT after {MAIN_QUERY_TIMEOUT_SEC}s")
        return None

    if proc.returncode != 0:
        print(f"  [ERROR] DuckDB returned non-zero: {proc.stderr[:500]}")
        return None

    if not os.path.exists(PROFILE_OUTPUT):
        print(f"  [ERROR] Profile output file not created")
        return None

    with open(PROFILE_OUTPUT, "r") as f:
        profile = json.load(f)

    os.remove(PROFILE_OUTPUT)
    return profile


# ============================================================================
# PLAN PARSING (JSON Profile)
# ============================================================================

def get_descendant_tables(node, cte_lineage):
    """
    Collect all table names from SEQ_SCAN nodes in the subtree rooted at `node`.
    Returns a set of table base names (stripped of schema prefix).
    """
    tables, _ = _collect_tables_with_cte(node, cte_lineage)
    return tables


def _node_context_signature(node):
    op_name = node.get("operator_name", node.get("name", ""))
    extra = node.get("extra_info", {}) or {}
    keys = ["Join Type", "Conditions", "Table", "Filters", "Estimated Cardinality"]
    parts = [f"op={op_name}"]
    for k in keys:
        if k in extra:
            parts.append(f"{k}={extra.get(k)}")
    return " | ".join(parts)


def _collect_subtree_scan_signatures(node):
    out = []
    op_name = node.get("operator_name", node.get("name", ""))
    extra = node.get("extra_info", {}) or {}
    if op_name == "SEQ_SCAN":
        table = base_table_name(str(extra.get("Table", "")))
        filters = str(extra.get("Filters", ""))
        projections = str(extra.get("Projections", ""))
        out.append(f"SEQ_SCAN table={table} filters={filters} projections={projections}")
    for child in node.get("children", []):
        out.extend(_collect_subtree_scan_signatures(child))
    return out


def _collect_subtree_operator_signatures(node):
    out = []
    op_name = node.get("operator_name", node.get("name", ""))
    extra = node.get("extra_info", {}) or {}
    keys = sorted(extra.keys())
    kv = []
    for k in keys:
        v = extra.get(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            kv.append(f"{k}={v}")
        else:
            kv.append(f"{k}={str(v)}")
    out.append(f"{op_name} :: " + " | ".join(kv))
    for child in node.get("children", []):
        out.extend(_collect_subtree_operator_signatures(child))
    return out


def extract_join_nodes(node, cte_lineage, path=(), ancestors=None):
    """
    Recursively walk the JSON profile tree and extract all join operator nodes.
    Returns a list of dicts:
        {
            "operator_name": str,
            "actual_cardinality": int,
            "conditions": str,        # raw condition string from extra_info
            "join_type": str,
            "estimated_cardinality": int,
            "descendant_tables": set,  # tables scanned in subtree
        }
    """
    results = []
    if ancestors is None:
        ancestors = []
    op_name = node.get("operator_name", node.get("name", ""))

    if op_name in JOIN_OPERATOR_NAMES:
        extra = node.get("extra_info", {})
        conditions_raw = extra.get("Conditions", "")
        join_type = extra.get("Join Type", "UNKNOWN")
        est_card_str = extra.get("Estimated Cardinality", "0")
        child_context = []
        for child in node.get("children", [])[:2]:
            child_extra = child.get("extra_info", {}) or {}
            child_est = child_extra.get("Estimated Cardinality", "0")
            child_context.append(
                {
                    "operator_name": child.get("operator_name", child.get("name", "")),
                    "actual_cardinality": int(
                        child.get("operator_cardinality", child.get("cardinality", 0))
                    ),
                    "estimated_cardinality": int(child_est)
                    if str(child_est).isdigit()
                    else 0,
                }
            )

        actual_card = node.get("operator_cardinality", node.get("cardinality", 0))
        desc_tables, lineage_incomplete = _collect_tables_with_cte(node, cte_lineage)

        results.append({
            "operator_name": op_name,
            "actual_cardinality": int(actual_card),
            "conditions": conditions_raw.strip() if isinstance(conditions_raw, str) else str(conditions_raw),
            "join_type": join_type,
            "estimated_cardinality": int(est_card_str) if str(est_card_str).isdigit() else 0,
            "descendant_tables": desc_tables,
            "lineage_incomplete": lineage_incomplete,
            "plan_path": list(path),
            "child_context": child_context,
            "ancestor_context": list(ancestors[-8:]),
            "subtree_scan_signatures": sorted(_collect_subtree_scan_signatures(node))[:20],
            "subtree_operator_signatures": _collect_subtree_operator_signatures(node)[:60],
            "subtree_structure_text": get_plan_structure_text(node),
            "subtree_structure_hash": compute_plan_fingerprint(get_plan_structure_text(node)),
        })

    next_ancestors = ancestors + [_node_context_signature(node)]
    for child_idx, child in enumerate(node.get("children", [])):
        results.extend(extract_join_nodes(child, cte_lineage, path + (child_idx,), next_ancestors))

    return results


def get_plan_structure_text(node, depth=0):
    """
    Build a canonical text representation of the physical plan STRUCTURE
    for convergence comparison. Includes operator names, join conditions,
    table names, and filters — but NOT actual cardinalities or timings,
    since those vary between runs even for the same plan shape.
    """
    op_name = node.get("operator_name", node.get("name", "?"))
    extra = node.get("extra_info", {})

    indent = "  " * depth
    lines = [f"{indent}{op_name}"]

    # Include structural extra_info (conditions, filters, tables) but NOT
    # cardinalities or timing
    STRUCTURAL_KEYS = {"Join Type", "Conditions", "Filters", "Table",
                       "Projections", "Limit", "Groups", "Aggregates",
                       "Join Condition"}
    for key in sorted(extra.keys()):
        if key in STRUCTURAL_KEYS:
            lines.append(f"{indent}  {key}: {extra[key]}")

    # Dynamic Filters: include only the bloom-filter (BF) structural signature,
    # NOT the runtime min-max ranges which can vary between runs.
    # e.g. "c_customer_sk IN BF(cs_bill_customer_sk)" → include
    #      "c_customer_sk>=4 AND c_customer_sk<=499987" → exclude
    dyn_filters_raw = str(extra.get("Dynamic Filters", ""))
    if dyn_filters_raw:
        import re as _re
        bf_parts = sorted(_re.findall(r"\w+ IN BF\(\w+\)", dyn_filters_raw))
        if bf_parts:
            lines.append(f"{indent}  BloomFilters: {bf_parts}")

    for child in node.get("children", []):
        lines.extend(get_plan_structure_text(child, depth + 1).splitlines())

    return "\n".join(lines)


# ============================================================================
# CARDINALITY LOG PARSING
# ============================================================================

def parse_cardinality_log():
    """
    Parse cardinality_log.txt. Each line has the form:
        LOGICAL_JOIN: RelSets: [...] RelBindings: [...] NumRels: [...] CtxInputCards: [...] Filters: [...] Estimated Cardinality: N
    or:
        LOGICAL_JOIN: RelSets: [...] RelBindings: [...] Filters: [...] using INJECTED Cardinality: N

    Returns a list of dicts:
        {
            "expression": str,     # the full expression string (key for JSON)
            "tables": list[str],   # base names extracted from rel bindings
            "rel_bindings": list[{"relation_index": int|None, "table_name": str, "raw": str}],
            "filters": list[str],
            "cardinality": float,
            "is_injected": bool,
        }
    """
    if not os.path.exists(CARDINALITY_LOG):
        return []

    with open(CARDINALITY_LOG, "r") as f:
        lines = f.readlines()

    entries = []
    ctx_input_raw_count = 0
    logical_join_line_count = 0
    sample_without_ctx = None
    for line in lines:
        line = line.strip()
        if not line or "LOGICAL_JOIN" not in line:
            continue
        logical_join_line_count += 1

        is_injected = "using INJECTED Cardinality:" in line

        # Extract the expression string (everything before " Estimated" or " using INJECTED")
        if is_injected:
            expr_end = line.find(" using INJECTED Cardinality:")
            card_str = line[expr_end + len(" using INJECTED Cardinality:"):].strip()
        else:
            expr_end = line.find(" Estimated Cardinality:")
            if expr_end == -1:
                continue
            card_str = line[expr_end + len(" Estimated Cardinality:"):].strip()

        expression = line[:expr_end]
        if "CtxInputCards:" in expression:
            ctx_input_raw_count += 1
        elif sample_without_ctx is None:
            sample_without_ctx = expression[-220:] if len(expression) > 220 else expression
        cardinality = float(card_str)

        # Parse relation bindings first (new format), fallback to legacy Tables
        rel_bindings_match = re.search(r"RelBindings: \[(.*?)\]", expression)
        rel_bindings = []
        tables = []
        if rel_bindings_match:
            rel_bindings_raw = rel_bindings_match.group(1)
            raw_bindings = []
            if " | " in rel_bindings_raw:
                # New format: entries are separated by " | ", table names may contain commas.
                raw_bindings = [b.strip() for b in rel_bindings_raw.split(" | ") if b.strip()]
            else:
                # Legacy fallback: comma-separated (ambiguous if table names contain commas).
                raw_bindings = [b.strip() for b in rel_bindings_raw.split(",") if b.strip()]
            for raw_binding in raw_bindings:
                relation_index = None
                table_name = raw_binding
                if ":" in raw_binding:
                    idx_part, table_part = raw_binding.split(":", 1)
                    table_name = (
                        table_part.strip()
                        .replace("\\|", "|")
                        .replace("\\\\", "\\")
                    )
                    try:
                        relation_index = int(idx_part.strip())
                    except ValueError:
                        relation_index = None
                rel_bindings.append({
                    "relation_index": relation_index,
                    "table_name": table_name,
                    "raw": raw_binding,
                })
                # Composite table names from subquery rewrites (e.g. "web_sales, catalog_sales")
                # must be split so the table set matches the profile's individual SEQ_SCANs.
                if table_name:
                    if "," in table_name:
                        for sub in table_name.split(","):
                            sub = sub.strip()
                            if sub:
                                tables.append(sub)
                    else:
                        tables.append(table_name)
        else:
            tables_match = re.search(r"Tables: \[(.*?)\]", expression)
            if tables_match:
                tables = [t.strip() for t in tables_match.group(1).split(",") if t.strip()]

        # Parse filters from the expression
        # Use non-greedy (.*?) for Filters so it stops before CtxScanFilters if present.
        filters_match = re.search(r"Filters: \[(.*?)\](?: CtxScanFilters: \[.*?\])?(?: CtxOcc: \d+)?$", expression)
        filters = []
        if filters_match:
            filters_raw = filters_match.group(1)
            # Split by " AND " at the top level (filters are separated by AND)
            # Each filter is wrapped in parentheses like (a = b)
            filters = split_filter_string(filters_raw)

        entries.append({
            "expression": expression,
            "tables": tables,
            "rel_bindings": rel_bindings,
            "filters": filters,
            "cardinality": cardinality,
            "is_injected": is_injected,
        })

    # #endregion

    return entries


def split_filter_string(filter_str):
    """
    Split a filter string like '(a = b) AND (c = d)' into individual filters.
    Handles nested parentheses correctly.
    Returns list of individual filter strings (with outer parens stripped).
    """
    filters = []
    depth = 0
    current = ""

    for char in filter_str:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
            if depth == 0:
                stripped = current.strip()
                if stripped:
                    # Strip outer parens
                    if stripped.startswith("(") and stripped.endswith(")"):
                        stripped = stripped[1:-1]
                    filters.append(stripped)
                current = ""
        elif depth > 0:
            current += char
        # Skip ' AND ' between top-level filters

    return filters


# ============================================================================
# CONDITION NORMALIZATION & MATCHING
# ============================================================================

def normalize_single_condition(cond):
    """
    Normalize a single condition string like 'a = b' or 'b = a' into a
    canonical frozenset form so that operand order doesn't matter for
    equality conditions.

    Returns a frozenset of the two operands for = conditions,
    or a plain string for non-equality conditions.
    """
    cond = cond.strip()
    # Strip outer parentheses
    while cond.startswith("(") and cond.endswith(")"):
        cond = cond[1:-1].strip()

    # Handle equality conditions: split on ' = '
    if " = " in cond:
        parts = cond.split(" = ", 1)
        return frozenset(p.strip() for p in parts)

    # Handle IS NOT DISTINCT FROM
    if " IS NOT DISTINCT FROM " in cond:
        parts = cond.split(" IS NOT DISTINCT FROM ", 1)
        return frozenset(p.strip() for p in parts)

    # For other conditions, return as-is string
    return cond


def normalize_condition_set(conditions_list):
    """
    Normalize a list of condition strings into a frozenset of normalized conditions.
    This allows order-independent comparison.
    """
    return frozenset(normalize_single_condition(c) for c in conditions_list)


def parse_explain_conditions(conditions_str):
    """
    Parse the conditions string from a JSON profile join node.
    The Conditions field can have multiple conditions separated by newlines or AND.
    Returns a list of individual condition strings.
    """
    if not conditions_str:
        return []
    if isinstance(conditions_str, list):
        parts = []
        for item in conditions_str:
            parts.extend(re.split(r"\n| AND ", str(item)))
        return [p.strip() for p in parts if p.strip()]

    if isinstance(conditions_str, str) and conditions_str.strip().startswith("["):
        try:
            parsed = ast.literal_eval(conditions_str)
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    parts.extend(re.split(r"\n| AND ", str(item)))
                return [p.strip() for p in parts if p.strip()]
        except Exception:
            pass

    # Split by newline or ' AND ' (conditions can be multiline in the JSON)
    parts = re.split(r"\n| AND ", str(conditions_str))
    return [p.strip() for p in parts if p.strip()]


def extract_condition_columns(conditions_str):
    cols = set()
    for cond in parse_explain_conditions(conditions_str):
        for m in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", cond):
            if "_" in m:
                cols.add(m)
    return cols


def extract_dynamic_filter_columns(subtree_operator_signatures):
    cols = set()
    for sig in subtree_operator_signatures or []:
        if "Dynamic Filters=" not in sig:
            continue
        for col in re.findall(r"optional:\s*([a-zA-Z_][a-zA-Z0-9_]*)", sig):
            cols.add(col)
    return cols


def _is_tautology_condition(norm_cond):
    """
    Filter out empty or degenerate normalized conditions. After ``normalize_single_condition``,
    equality becomes a 2-element frozenset; if parsing produces a 1-element frozenset, treat
    it as non-informative and drop it from matching.
    """
    if isinstance(norm_cond, frozenset):
        return len(norm_cond) == 1
    return False


def precompute_join_match_caches(profile_joins, log_entries):
    """
    Normalized condition sets per profile join and per log line, plus table sets per log line.
    Profile rows parse ``conditions`` strings; log rows use parsed ``filters`` lists.
    """
    profile_normalized = []
    for pj in profile_joins:
        conds = parse_explain_conditions(pj["conditions"])
        norm = normalize_condition_set(conds)
        profile_normalized.append(frozenset(c for c in norm if not _is_tautology_condition(c)))

    log_normalized = []
    log_table_sets = []
    for le in log_entries:
        norm = normalize_condition_set(le["filters"])
        log_normalized.append(frozenset(c for c in norm if not _is_tautology_condition(c)))
        log_table_sets.append(set(le["tables"]))
    return profile_normalized, log_normalized, log_table_sets


def _join_match_stage(pnorm, lnorm, p_desc_tables, l_tables):
    """Discrete priority stage + auxiliary deltas for tie-breaking among viable candidates."""
    cond_diff = abs(len(lnorm) - len(pnorm))
    table_delta = abs(len(l_tables) - len(p_desc_tables))
    table_exact = bool(p_desc_tables) and l_tables == p_desc_tables
    cond_exact = lnorm == pnorm
    if cond_exact and table_exact:
        stage = 0
    elif cond_exact:
        stage = 1
    elif table_exact:
        stage = 2
    else:
        stage = 3
    return stage, cond_diff, table_delta


def _log_candidate_context_ok(pnorm, lnorm, p_desc_tables, l_tables, p_lineage_incomplete):
    """Whether normalized conditions + table lineage allow this log line as a candidate."""
    if p_lineage_incomplete:
        if not (lnorm == pnorm or pnorm.issubset(lnorm)):
            return False
    elif lnorm != pnorm:
        return False
    if p_desc_tables and p_lineage_incomplete and not (p_desc_tables & l_tables):
        return False
    if p_desc_tables and not p_lineage_incomplete and l_tables != p_desc_tables:
        return False
    return True


def _candidate_beats_best(
    p_estimated_cardinality,
    cand_est_delta,
    stage,
    table_delta,
    cond_diff,
    expr,
    best_est_delta,
    best_stage,
    best_table_delta,
    best_cond_diff,
    best_expr,
):
    """Lexicographic tie-break (optionally using estimated-cardinality distance first)."""
    use_est = p_estimated_cardinality > 0
    if use_est:
        if cand_est_delta < best_est_delta:
            return True
        if cand_est_delta > best_est_delta:
            return False
        if stage < best_stage:
            return True
        if stage > best_stage:
            return False
        if table_delta < best_table_delta:
            return True
        if table_delta > best_table_delta:
            return False
        if cond_diff < best_cond_diff:
            return True
        if cond_diff > best_cond_diff:
            return False
        return best_expr is None or expr < best_expr
    if stage < best_stage:
        return True
    if stage > best_stage:
        return False
    if table_delta < best_table_delta:
        return True
    if table_delta > best_table_delta:
        return False
    if cond_diff < best_cond_diff:
        return True
    if cond_diff > best_cond_diff:
        return False
    return best_expr is None or expr < best_expr


def match_single_profile_join(
    pidx,
    pj,
    log_entries,
    profile_normalized,
    log_normalized,
    log_table_sets,
    used_log_indices,
):
    """
    Pick the best unused cardinality-log line for one physical join, or None if unresolved.

    Returns:
        (match_dict, None) on success,
        (None, unresolved_dict) if no eligible log line,
        (None, None) if this join has no conditions (cross join) — caller skips.
    """
    pnorm = profile_normalized[pidx]
    if not pnorm:
        return None, None

    p_desc_tables = pj.get("descendant_tables", set())
    actual_card = int(pj["actual_cardinality"])
    p_estimated_cardinality = int(pj.get("estimated_cardinality", 0))
    p_plan_path = pj.get("plan_path", [])
    p_lineage_incomplete = pj.get("lineage_incomplete", False)

    best_lidx = -1
    best_stage = 99
    best_table_delta = float("inf")
    best_cond_diff = float("inf")
    best_est_delta = float("inf")
    best_expr = None
    candidate_count = 0

    for lidx, le in enumerate(log_entries):
        if lidx in used_log_indices:
            continue
        lnorm = log_normalized[lidx]
        if not lnorm:
            continue
        l_tables = log_table_sets[lidx]
        if not _log_candidate_context_ok(pnorm, lnorm, p_desc_tables, l_tables, p_lineage_incomplete):
            continue

        stage, cond_diff, table_delta = _join_match_stage(pnorm, lnorm, p_desc_tables, l_tables)
        candidate_count += 1
        expr = le["expression"]
        cand_card = int(le["cardinality"])
        cand_est_delta = abs(cand_card - p_estimated_cardinality)
        if _candidate_beats_best(
            p_estimated_cardinality,
            cand_est_delta,
            stage,
            table_delta,
            cond_diff,
            expr,
            best_est_delta,
            best_stage,
            best_table_delta,
            best_cond_diff,
            best_expr,
        ):
            best_est_delta = cand_est_delta
            best_stage = stage
            best_table_delta = table_delta
            best_cond_diff = cond_diff
            best_lidx = lidx
            best_expr = expr

    if best_lidx != -1:
        return (
            {
                "expression": log_entries[best_lidx]["expression"],
                "log_index": best_lidx,
                "actual_cardinality": actual_card,
                "profile_join_index": pidx,
                "candidate_count": candidate_count,
                "selected_stage": best_stage,
                "selected_table_delta": best_table_delta,
                "selected_cond_diff": best_cond_diff,
                "selected_abs_delta_to_profile_est": best_est_delta,
                "profile_estimated_cardinality": p_estimated_cardinality,
                "profile_plan_path": list(p_plan_path) if isinstance(p_plan_path, list) else [],
                "lineage_incomplete": p_lineage_incomplete,
                "is_injected": log_entries[best_lidx].get("is_injected", False),
            },
            None,
        )
    return (
        None,
        {
            "profile_join_index": pidx,
            "conditions": pj.get("conditions", ""),
            "descendant_tables": sorted(list(p_desc_tables)),
            "lineage_incomplete": p_lineage_incomplete,
            "actual_cardinality": int(actual_card),
            "estimated_cardinality": int(p_estimated_cardinality),
            "plan_path": list(p_plan_path) if isinstance(p_plan_path, list) else [],
        },
    )


def match_joins(profile_joins, log_entries):
    """
    Match cardinality log entries to profile join nodes using BOTH normalized
    conditions AND descendant table context.

    For each profile join, we find the log entry whose:
    1. Normalized conditions match (profile conditions ⊆ log conditions)
    2. Descendant tables from the physical plan are a subset of the log's
       table set (ensuring the log entry covers the same tables)
    3. Among valid matches, pick the log entry with the smallest table set
       (most specific match)

    Returns:
      - matches: list of dicts with expression, log_index (cardinality_log line),
        actual_cardinality, profile_join_index, etc.
      - unresolved: list of unresolved physical join diagnostics
    """
    matches = []
    unresolved = []
    used_log_indices = set()

    profile_normalized, log_normalized, log_table_sets = precompute_join_match_caches(
        profile_joins, log_entries
    )

    for pidx, pj in enumerate(profile_joins):
        m, u = match_single_profile_join(
            pidx,
            pj,
            log_entries,
            profile_normalized,
            log_normalized,
            log_table_sets,
            used_log_indices,
        )
        if m is not None and u is None:
            used_log_indices.add(m["log_index"])
            matches.append(m)
        elif m is None and u is not None:
            unresolved.append(u)

    return matches, unresolved


# ============================================================================
# CTE / DUPLICATE DETECTION
# ============================================================================

def detect_cte_duplicates(log_entries):
    """
    Detect expressions that appear more than once in the cardinality log.
    These are likely CTEs that are planned once but executed multiple times.
    We should NOT inject cardinalities for these.

    Returns a set of expression strings that are duplicates.
    """
    expr_counts = Counter(entry["expression"] for entry in log_entries)
    return {expr for expr, count in expr_counts.items() if count > 1}


def log_has_injected_line(log_entries, expr):
    """True if any cardinality-log row with this expression is marked injected."""
    return any(e["expression"] == expr and e["is_injected"] for e in log_entries)


# ============================================================================
# INJECTION UPDATE
# ============================================================================

def update_actual_cardinality_json(
    matches,
    cte_expressions,
    *,
    query_nr=None,
    iteration=None,
    plan_stable=None,
    plan_fingerprint=None,
):
    """
    Read current actual_cardinality.json, add new entries from matches,
    and write back. Skips CTE duplicate expressions.

    For each match (expression, actual_cardinality):
    - If expression is a CTE duplicate, skip it with a warning.
    - If expression already exists with SAME cardinality, skip (no change).
    - If expression already exists with DIFFERENT cardinality:
      - [ALARM] if plan_stable is True (same plan shape as last iteration — unexpected).
      - [INFO] if plan_stable is False (plan changed — normal under oscillation).
    - If expression is new, add it.

    plan_stable: True iff physical plan structure equals previous iteration's plan
    (get_plan_structure_text). If False, cardinality drift prints as [INFO] not [ALARM].
    None is treated like True (ALARM on mismatch) for backward compatibility.

    Returns True if any new entries were added, False otherwise.
    """
    current = read_actual_cardinality_json()
    changes_made = False


    for match in matches:
        expression = match["expression"]
        namespaced_key = make_namespaced_expression_key(expression, plan_fingerprint)
        actual_card = match["actual_cardinality"]
        if expression in cte_expressions:
            print(f"    [SKIP-CTE] {expression}")
            continue

        existing_key = None
        if namespaced_key in current:
            existing_key = namespaced_key
        elif expression in current:
            # Legacy fallback for pre-namespaced JSON files.
            existing_key = expression

        if existing_key is not None:
            existing_card = current[existing_key]
            if int(existing_card) != int(actual_card):
                _stable = plan_stable is not False
                abs_delta = abs(int(existing_card) - int(actual_card))
                max_mag = max(abs(int(existing_card)), abs(int(actual_card)), 1)
                rel_delta = float(abs_delta) / float(max_mag)
                is_small_stable_drift = (
                    _stable
                    and (
                        abs_delta <= STABLE_PLAN_ABS_DRIFT_TOLERANCE
                        or rel_delta <= STABLE_PLAN_REL_DRIFT_TOLERANCE
                    )
                )
                if is_small_stable_drift:
                    print(
                        f"    [INFO] Small stable-plan drift (keeping existing JSON value):"
                    )
                    print(f"      Expression: {expression}")
                    print(
                        f"      Existing: {int(existing_card)}, New: {int(actual_card)}, "
                        f"Delta: {abs_delta} ({rel_delta:.6%})"
                    )
                    continue
                if _stable:
                    print(f"    [ALARM] Cardinality mismatch for expression!")
                else:
                    print(
                        f"    [INFO] Cardinality update (plan changed vs last iteration; "
                        f"overwriting JSON):"
                    )
                print(f"      Expression: {expression}")
                print(f"      Existing: {int(existing_card)}, New: {int(actual_card)}")
                # Update to new value (the latest run is most accurate)
                current[namespaced_key] = float(actual_card)
                if existing_key != namespaced_key and existing_key in current:
                    del current[existing_key]
                changes_made = True
            # else: same value, no change needed
        else:
            current[namespaced_key] = float(actual_card)
            changes_made = True
            print(f"    [NEW] {expression} -> {actual_card}")

    if changes_made:
        write_actual_cardinality_json(current)

    return changes_made


# ============================================================================
# VERIFICATION (one function per check; verify_injection orchestrates)
# ============================================================================


def verify_check_1_injected_vs_json(log_entries, active_pre_update_json):
    """Injected log cardinalities should match pre-update JSON values."""
    injected_count = 0
    warns = []
    for entry in log_entries:
        if not entry["is_injected"]:
            continue
        injected_count += 1
        expr = entry["expression"]
        log_val = entry["cardinality"]
        if expr not in active_pre_update_json:
            warns.append(("missing_key", expr, None, None))
            continue
        json_val = active_pre_update_json[expr]
        if abs(json_val - log_val) >= 1.0:
            warns.append(("value_mismatch", expr, json_val, log_val))
    if warns:
        print(
            f"    [VERIFY] Check 1 WARN: {len(warns)} injected log line(s) "
            f"do not align with pre-update JSON (plan/JSON drift or float noise):"
        )
        for kind, expr, jv, lv in warns[:10]:
            ex = expr if len(expr) <= 160 else expr[:160] + "..."
            if kind == "missing_key":
                print(f"      key not in pre-update JSON: {ex}")
            else:
                print(f"      value mismatch: JSON={jv}, Log={lv} — {ex}")
        if len(warns) > 10:
            print(f"      ... and {len(warns) - 10} more")
    else:
        print(
            f"    [VERIFY] Check 1 PASSED: {injected_count} injected log values "
            f"match the pre-update JSON."
        )


def verify_check_2_json_keys_in_log(log_entries, active_pre_update_json):
    """Every JSON key from before this iteration should appear somewhere in the log."""
    log_expressions = {entry["expression"] for entry in log_entries}
    missing = [e for e in active_pre_update_json if e not in log_expressions]
    if missing:
        for expr in missing:
            print(f"    [VERIFY] Check 2 WARN: JSON key not in log: {expr}")
    else:
        print(
            f"    [VERIFY] Check 2 PASSED: All {len(active_pre_update_json)} JSON keys "
            f"found in the cardinality log."
        )


def verify_check_3_known_matches_injected(matches, cte_exprs, active_pre_update_json, log_entries):
    """Previously-known matches should show INJECTED on at least one log line."""
    matched_not_injected = []
    matched_injected = []
    matched_new = []
    for match in matches:
        expr = match["expression"]
        if expr in cte_exprs:
            continue
        if expr not in active_pre_update_json:
            matched_new.append(expr)
            continue
        if log_has_injected_line(log_entries, expr):
            matched_injected.append(expr)
        else:
            matched_not_injected.append(expr)

    if matched_new:
        print(
            f"    [VERIFY] Check 3 INFO: {len(matched_new)} new join(s) from "
            f"plan change (not previously injected, OK)."
        )
    if matched_not_injected:
        print(
            f"    [VERIFY] Check 3 WARN: {len(matched_not_injected)} previously-known "
            f"join(s) have no INJECTED line in this run's cardinality log (plan may have "
            f"changed, or the optimizer did not apply actual_cardinality.json for that key):"
        )
        for expr in matched_not_injected:
            print(f"      NOT INJECTED: {expr}")
    else:
        print(
            f"    [VERIFY] Check 3 PASSED: All {len(matched_injected)} previously-known "
            f"joins (non-CTE) were INJECTED."
        )


def verify_check_4_injected_vs_actual_plan(log_entries, matches, plan_stable):
    """Injected value vs measured cardinality on matched joins (informational if plan moved)."""
    matched_exprs = {m["expression"]: m["actual_cardinality"] for m in matches}
    mismatches = []
    tolerated = []
    for entry in log_entries:
        if not entry["is_injected"] or entry["expression"] not in matched_exprs:
            continue
        expr = entry["expression"]
        injected_val = int(entry["cardinality"])
        actual_val = matched_exprs[expr]
        abs_delta = abs(int(injected_val) - int(actual_val))
        max_mag = max(abs(int(injected_val)), abs(int(actual_val)), 1)
        rel_delta = float(abs_delta) / float(max_mag)
        small_drift = (
            plan_stable is True
            and abs_delta > 0
            and (
                abs_delta <= STABLE_PLAN_ABS_DRIFT_TOLERANCE
                or rel_delta <= STABLE_PLAN_REL_DRIFT_TOLERANCE
            )
        )
        if small_drift:
            tolerated.append((expr, injected_val, actual_val, abs_delta, rel_delta))
        elif injected_val != actual_val:
            mismatches.append((expr, injected_val, actual_val))

    if mismatches:
        print(
            f"    [VERIFY] Check 4 INFO: {len(mismatches)} "
            f"injected != actual (plan may have changed):"
        )
        for expr, inj, act in mismatches:
            print(f"      {expr}")
            print(f"        Injected: {inj}, Actual: {act}")
    else:
        print(
            f"    [VERIFY] Check 4 PASSED: All injected cardinalities match "
            f"actual plan cardinalities."
        )
    if tolerated:
        print(
            f"    [VERIFY] Check 4 INFO: tolerated {len(tolerated)} "
            f"small stable-plan drift(s) (treated as measurement noise)."
        )


def verify_check_5_distinct_log_bindings(matches):
    """Each profile↔log binding should use a distinct log_index when matching."""
    log_indices = [m["log_index"] for m in matches]
    dup_log_idx = [i for i, c in Counter(log_indices).items() if c > 1]
    if dup_log_idx:
        print(
            f"    [VERIFY] Check 5 WARN: duplicate log_index in matches (matcher bug?) "
            f": {dup_log_idx}"
        )
        return
    matched_expr_list = [m["expression"] for m in matches]
    dup_exprs = [expr for expr, c in Counter(matched_expr_list).items() if c > 1]
    if dup_exprs:
        print(
            f"    [VERIFY] Check 5 PASSED: {len(matches)} mappings; {len(dup_exprs)} "
            f"expression string(s) repeated across distinct log line(s) (OK)."
        )
    else:
        print(
            f"    [VERIFY] Check 5 PASSED: {len(matches)} mappings; all expression strings unique."
        )


def verify_check_6_join_coverage(matches, profile_joins):
    """How many physical joins with parseable conditions got a match."""
    joins_with_conditions = sum(
        1 for pj in profile_joins if parse_explain_conditions(pj.get("conditions", ""))
    )
    unmatched = joins_with_conditions - len(matches)
    print(
        f"    [VERIFY] Check 6 INFO: matched {len(matches)}/{joins_with_conditions} "
        f"joins-with-conditions (unmatched={max(unmatched, 0)})."
    )


def verify_check_7_ambiguous_matches(matches):
    """Report matches where candidate_count > 1 (tie-breakers used)."""
    ambiguous = [m for m in matches if int(m.get("candidate_count", 0)) > 1]
    if not ambiguous:
        print("    [VERIFY] Check 7 PASSED: no ambiguous match candidates.")
        return
    stage0_count = sum(1 for m in ambiguous if int(m.get("selected_stage", 99)) == 0)
    non_stage0 = [m for m in ambiguous if int(m.get("selected_stage", 99)) > 0]
    print(
        f"    [VERIFY] Check 7 INFO: {len(ambiguous)} ambiguous match(es); "
        f"stage0={stage0_count}, non_stage0={len(non_stage0)}."
    )
    if not non_stage0:
        return
    print(
        "    [VERIFY] Check 7 INFO: non-stage0 ambiguous selections "
        "(estimate-proxy likely applied):"
    )
    for m in non_stage0[:6]:
        expr = m.get("expression", "")
        expr_tail = expr[-180:] if len(expr) > 180 else expr
        print(
            f"      log_index={m.get('log_index')}, stage={m.get('selected_stage')}, "
            f"est_delta={m.get('selected_abs_delta_to_profile_est')}, "
            f"profile_est={m.get('profile_estimated_cardinality')}, "
            f"path={m.get('profile_plan_path')}: {expr_tail}"
        )

def verify_check_8_unused_json_keys(log_entries, active_pre_update_json):
    """JSON keys that exist but were NOT applied as INJECTED — optimizer didn't visit them."""
    injected_exprs = {e["expression"] for e in log_entries if e["is_injected"]}
    log_exprs = {e["expression"] for e in log_entries}
    unused = []
    absent = []
    for key in active_pre_update_json:
        if key in injected_exprs:
            continue
        if key in log_exprs:
            # Key appeared in log but was NOT injected (should not happen normally)
            unused.append(key)
        else:
            # Key was not visited at all (plan may have changed to a different shape)
            absent.append(key)
    total_json = len(active_pre_update_json)
    n_injected = len(injected_exprs & set(active_pre_update_json))
    if unused:
        print(
            f"    [VERIFY] Check 8 WARN: {len(unused)} JSON key(s) appear in log "
            f"but were NOT injected (optimizer should have applied them):"
        )
        for k in unused[:5]:
            print(f"      {k if len(k) <= 160 else k[:160] + '...'}")
    if absent:
        print(
            f"    [VERIFY] Check 8 INFO: {len(absent)} JSON key(s) not visited "
            f"by optimizer (plan shape changed, key's subplan not reached)."
        )
    if not unused and not absent:
        print(
            f"    [VERIFY] Check 8 PASSED: all {total_json} JSON keys were "
            f"INJECTED ({n_injected} applied)."
        )


def verify_check_9_no_key_regression(pre_update_json, post_update_json):
    """All keys in pre-update JSON must still exist in post-update JSON (monotonic growth)."""
    lost = [k for k in pre_update_json if k not in post_update_json]
    if lost:
        print(
            f"    [VERIFY] Check 9 WARN: {len(lost)} key(s) LOST from JSON "
            f"(regression — quarantine may have removed them):"
        )
        for k in lost[:5]:
            print(f"      LOST: {k if len(k) <= 160 else k[:160] + '...'}")
    else:
        print(
            f"    [VERIFY] Check 9 PASSED: all {len(pre_update_json)} pre-update "
            f"keys preserved in post-update JSON."
        )


def verify_injection(
    log_entries,
    pre_update_json,
    matches,
    profile_joins,
    cte_exprs,
    iteration,
    *,
    plan_stable=None,
    injection_plan_fingerprint=None,
    post_update_json=None,
):
    """
    Run all verification checks for iteration >= 2 (see verify_check_* functions).
    """
    print(f"    [VERIFY] Running verification for iteration {iteration}...")

    active_pre_update_json = project_json_for_fingerprint(
        pre_update_json, injection_plan_fingerprint
    )

    verify_check_1_injected_vs_json(log_entries, active_pre_update_json)
    verify_check_2_json_keys_in_log(log_entries, active_pre_update_json)
    verify_check_3_known_matches_injected(
        matches, cte_exprs, active_pre_update_json, log_entries
    )
    verify_check_4_injected_vs_actual_plan(log_entries, matches, plan_stable)
    verify_check_5_distinct_log_bindings(matches)
    verify_check_6_join_coverage(matches, profile_joins)
    verify_check_7_ambiguous_matches(matches)
    verify_check_8_unused_json_keys(log_entries, active_pre_update_json)
    if post_update_json is not None:
        verify_check_9_no_key_regression(pre_update_json, post_update_json)


# ============================================================================
# PLAN TIMING COMPARISON
# ============================================================================


def time_query_n_runs(query_sql, n_runs=5):
    """Run query n_runs times and return list of wall-clock seconds (no profiling overhead)."""
    import time as _time
    timings = []
    run_env = child_duckdb_env()
    bare_sql = (
        "PRAGMA enable_progress_bar = false;\n" + query_sql + "\n"
    )
    for _ in range(n_runs):
        t0 = _time.monotonic()
        try:
            proc = subprocess.run(
                [DUCKDB_BIN, DB_FILE, "-c", bare_sql],
                capture_output=True, text=True, env=run_env,
                timeout=MAIN_QUERY_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            timings.append(float("inf"))
            continue
        t1 = _time.monotonic()
        if proc.returncode != 0:
            timings.append(float("inf"))
        else:
            timings.append(t1 - t0)
    return timings


def compare_plan_timings(query_sql, query_nr, n_runs=5):
    """Run the query with and without injection, print mean/min for both.

    Call this AFTER the feedback loop converges and the final actual_cardinality.json
    is in place. Temporarily removes the JSON to get vanilla timings, then restores it.
    """
    print(f"\n  [TIMING] Comparing vanilla vs feedback plan for Q{query_nr} ({n_runs} runs each)...")

    # --- Feedback plan (with current JSON) ---
    feedback_times = time_query_n_runs(query_sql, n_runs)

    # --- Vanilla plan (no injection) ---
    saved_json = read_actual_cardinality_json()
    clear_actual_cardinality_json()
    vanilla_times = time_query_n_runs(query_sql, n_runs)
    # Restore
    if saved_json:
        write_actual_cardinality_json(saved_json)

    def _stats(times):
        finite = [t for t in times if t != float("inf")]
        if not finite:
            return float("inf"), float("inf")
        return min(finite), sum(finite) / len(finite)

    v_min, v_mean = _stats(vanilla_times)
    f_min, f_mean = _stats(feedback_times)

    print(f"    Vanilla:  min={v_min:.3f}s  mean={v_mean:.3f}s  runs={vanilla_times}")
    print(f"    Feedback: min={f_min:.3f}s  mean={f_mean:.3f}s  runs={feedback_times}")
    if f_min < float("inf") and v_min < float("inf"):
        speedup = v_min / f_min if f_min > 0 else float("inf")
        print(f"    Speedup (min): {speedup:.2f}x {'(FASTER)' if speedup > 1.0 else '(SLOWER ⚠)'}")
    if f_mean > v_mean * 1.10:
        print(f"    [TIMING-WARN] Feedback plan appears SLOWER than vanilla — "
              f"injections may be counterproductive for Q{query_nr}.")

    return {
        "vanilla_min": v_min, "vanilla_mean": v_mean,
        "feedback_min": f_min, "feedback_mean": f_mean,
    }


# ============================================================================
# MAIN LOOP PER QUERY
# ============================================================================


def _record_match_history(
    expr_match_history,
    matches,
    profile_joins,
    iteration,
    current_plan_fingerprint,
):
    """Append one history record per match for this iteration (safety / drift analysis)."""
    for m in matches:
        expr = m["expression"]
        pidx = m.get("profile_join_index")
        profile_conditions = ""
        profile_desc_tables = []
        profile_plan_path = []
        if isinstance(pidx, int) and 0 <= pidx < len(profile_joins):
            profile_conditions = profile_joins[pidx].get("conditions", "")
            profile_desc_tables = sorted(
                list(profile_joins[pidx].get("descendant_tables", set()))
            )
            profile_plan_path = list(profile_joins[pidx].get("plan_path", []))
        expr_match_history.setdefault(expr, []).append(
            {
                "iteration": iteration,
                "profile_join_index": pidx,
                "log_index": m.get("log_index"),
                "actual_cardinality": int(m.get("actual_cardinality", 0)),
                "plan_fingerprint": current_plan_fingerprint,
                "match_stage": m.get("selected_stage", 99),
                "candidate_count": m.get("candidate_count", 0),
                "is_injected": m.get("is_injected", False),
                "profile_conditions": profile_conditions,
                "profile_descendant_tables": profile_desc_tables,
                "profile_plan_path": profile_plan_path,
                "profile_ancestor_context": (
                    profile_joins[pidx].get("ancestor_context", [])
                    if isinstance(pidx, int) and 0 <= pidx < len(profile_joins)
                    else []
                ),
                "profile_subtree_scan_signatures": (
                    profile_joins[pidx].get("subtree_scan_signatures", [])
                    if isinstance(pidx, int) and 0 <= pidx < len(profile_joins)
                    else []
                ),
                "profile_subtree_operator_signatures": (
                    profile_joins[pidx].get("subtree_operator_signatures", [])
                    if isinstance(pidx, int) and 0 <= pidx < len(profile_joins)
                    else []
                ),
                "profile_subtree_structure_hash": (
                    profile_joins[pidx].get("subtree_structure_hash", "")
                    if isinstance(pidx, int) and 0 <= pidx < len(profile_joins)
                    else ""
                ),
                "candidate_count": int(m.get("candidate_count", 0)),
            }
        )


def _quarantine_ambiguous_matches(matches):
    """Multiple viable log lines for one join — quarantine UNLESS safely resolved.

    When ``candidate_count > 1`` it means the matcher found multiple log entries
    with compatible (conditions, tables).  This is normal for the DP join
    enumerator: it evaluates the same predicate under many input-cardinality
    contexts, each producing a separate log line with a distinct ``CtxOcc``
    suffix that makes the full expression key unique.

    The match is safe to inject when:
      1. The winner is **stage 0** — exact conditions AND exact table set.
      2. The expression key appears **at most once** in the current batch of
         matches (no collision with another physical join mapping to the same key).

    If both conditions hold the match was correctly resolved by the tie-breaker
    and the injection key is unambiguous.  Otherwise we quarantine.
    """
    # Build a frequency map of expression keys across all matches.
    expr_freq = Counter(m["expression"] for m in matches)

    out = set()
    for m in matches:
        cc = int(m.get("candidate_count", 0))
        if cc <= 1:
            continue
        expr = m["expression"]
        stage = int(m.get("selected_stage", 99))
        if stage == 0 and expr_freq[expr] == 1:
            # High-confidence: exact match + unique key → safe to inject.
            print(f"    [INFO-AMBIGUOUS-RESOLVED] stage0 unique key (candidates={cc}): {expr}")
            continue
        print(f"    [ALARM-AMBIGUOUS-MATCH] {expr}")
        out.add(expr)
    return out


def _quarantine_same_key_actual_collisions(matches):
    """Same LOGICAL_JOIN key maps to different measured cardinalities in one batch."""
    out = set()
    batch_actuals = {}
    for m in matches:
        batch_actuals.setdefault(m["expression"], set()).add(
            int(m.get("actual_cardinality", 0))
        )
    for expr, vals in batch_actuals.items():
        if len(vals) <= 1:
            continue
        print(
            f"    [ALARM-CONTEXT-COLLISION] same key has conflicting actuals in batch: "
            f"{sorted(list(vals))} -- {expr}"
        )
        out.add(expr)
    return out


def _quarantine_dynamic_filter_context(matches, profile_joins):
    """Dynamic-filter guard — DISABLED.

    Dynamic filters are runtime bloom/min-max pushdowns installed by a different
    pipeline.  They prune scan rows but do NOT change logical join selectivity.
    The injection key already encodes ``CtxInputCards`` (input cardinalities
    after scan-level filtering), so the measured join cardinality is deterministic
    and reproducible for a given plan structure.  Existing Check 4 (injected vs
    actual) catches any real mismatches on subsequent iterations.

    Previously this function quarantined any match whose subtree contained dynamic-
    filter columns outside the join's own condition columns — roughly ~35 injections
    per run were blocked.  All of those were false positives.
    """
    return set()


def _quarantine_cross_iteration_cardinality_drift(expr_match_history):
    """Same expression saw materially different actuals across iterations ON A STABLE PLAN.

    When the plan changes between iterations, cardinalities naturally differ because
    the join runs in a different physical context — this is expected, not a collision.
    We only quarantine when the plan fingerprint is the same (stable plan) but the
    measured cardinality still drifts beyond noise tolerances.
    """
    out = set()
    for expr, hist in expr_match_history.items():
        if len(hist) < 2:
            continue
        prev = hist[-2]
        curr = hist[-1]
        # If the plan changed, cardinality drift is expected — skip.
        if prev.get("plan_fingerprint") != curr.get("plan_fingerprint"):
            continue
        prev_actual = int(prev.get("actual_cardinality", 0))
        curr_actual = int(curr.get("actual_cardinality", 0))
        if prev_actual == curr_actual:
            continue
            
        # If the previous match was NOT injected (vanilla estimation) 
        # and the current match IS injected, the "drift" is just the script
        # fixing its own guessed mapping from Iteration 1. Do not quarantine.
        prev_injected = prev.get("is_injected", False)
        curr_injected = curr.get("is_injected", False)
        if not prev_injected and curr_injected:
            continue
            
        abs_delta = abs(prev_actual - curr_actual)
        max_mag = max(abs(prev_actual), abs(curr_actual), 1)
        rel_delta = float(abs_delta) / float(max_mag)
        if (
            abs_delta <= STABLE_PLAN_ABS_DRIFT_TOLERANCE
            or rel_delta <= STABLE_PLAN_REL_DRIFT_TOLERANCE
        ):
            continue
        print(
            f"    [ALARM-CONTEXT-COLLISION] same key changed cardinality on STABLE plan: "
            f"prev={prev_actual}, curr={curr_actual} "
            f"(delta={abs_delta}, {rel_delta:.2%}) -- {expr}"
        )
        out.add(expr)
    return out


def collect_quarantine_unsafe_expressions(matches, profile_joins, expr_match_history):
    """Union of all injection-safety quarantine rules for this iteration."""
    newly = set()
    newly |= _quarantine_ambiguous_matches(matches)
    newly |= _quarantine_same_key_actual_collisions(matches)
    newly |= _quarantine_dynamic_filter_context(matches, profile_joins)
    newly |= _quarantine_cross_iteration_cardinality_drift(expr_match_history)
    return newly


def _feedback_query_result(
    iterations,
    converged,
    plan_changed_iterations,
    *,
    error=False,
    oscillation=False,
    cycle_length=None,
    feedback_unique_plans=0,
    final_plan_fingerprint=None,
    n_injected=0,
    timing=None,
):
    """Single schema for run_single_query return dict (avoids drift between branches)."""
    res = {
        "iterations": iterations,
        "converged": converged,
        "plan_changed_iterations": plan_changed_iterations,
        "error": error,
        "feedback_unique_plans": feedback_unique_plans,
        "final_plan_fingerprint": final_plan_fingerprint,
        "n_injected": n_injected,
    }
    if oscillation:
        res["oscillation"] = True
        res["cycle_length"] = cycle_length
    if timing is not None:
        res["timing"] = timing
    return res


def run_single_query(query_nr, query_sql):
    """
    Run the iterative feedback loop for a single TPC-DS query:
    1. Clear injection data and logs
    2. Run query with JSON profiling
    3. Parse joins from profile and cardinality log
    4. Match and inject actual cardinalities
    5. Repeat until plan converges or MAX_ITERATIONS reached

    Returns a dict with:
        - "iterations": int (number of iterations until convergence)
        - "converged": bool
        - "plan_changed_iterations": list of iteration numbers where plan changed
        - "feedback_unique_plans": int (distinct plan structure texts seen)
        - "final_plan_fingerprint": str or None (last successful iteration)
        - "n_injected": int — log lines marked INJECTED on the last completed iteration
    """
    print(f"\n{'='*60}")
    print(f"  Query {query_nr}")
    print(f"{'='*60}")

    # Step 1: Start fresh — clear injection file and log
    clear_actual_cardinality_json()
    clear_cardinality_log()

    seen_plan_structures = []  # Track all seen plan structures for oscillation detection
    prev_plan_text = None
    plan_changed_iterations = []
    expr_match_history = defaultdict(list)
    unsafe_expressions = set()
    unique_plan_texts = set()
    last_plan_fingerprint = None
    feedback_last_n_injected = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  --- Iteration {iteration} ---")

        # Step 2: Fresh cardinality log for this iteration (JSON still accumulates).
        clear_cardinality_log()

        # Step 3: Execute query once per iteration (no global plan fingerprint namespace).
        profile = run_query_with_json_profile(
            query_sql,
            plan_fingerprint_hint=None,
        )
        if profile is None:
            print(f"  [ERROR] Query {query_nr} failed on iteration {iteration}. Skipping.")
            return _feedback_query_result(
                iteration,
                False,
                plan_changed_iterations,
                error=True,
                feedback_unique_plans=len(unique_plan_texts),
                final_plan_fingerprint=last_plan_fingerprint,
                n_injected=feedback_last_n_injected,
            )

        # Step 4: Get plan structure and parse everything FIRST
        root = profile.get("children", [profile])[0] if profile.get("children") else profile
        current_plan_text = get_plan_structure_text(root)
        current_plan_fingerprint = compute_plan_fingerprint(current_plan_text)
        unique_plan_texts.add(current_plan_text)
        last_plan_fingerprint = current_plan_fingerprint

        # Step 5: Parse joins from the JSON profile
        cte_lineage = build_cte_lineage(root)
        profile_joins = extract_join_nodes(root, cte_lineage)
        print(f"  Found {len(profile_joins)} join(s) in physical plan.")

        # Step 6: Parse the cardinality log
        log_entries = parse_cardinality_log()
        feedback_last_n_injected = sum(1 for e in log_entries if e.get("is_injected"))
        print(f"  Found {len(log_entries)} log entries in cardinality_log.txt.")

        # Step 7: Detect CTE duplicates
        cte_exprs = detect_cte_duplicates(log_entries)
        if cte_exprs:
            print(f"  Detected {len(cte_exprs)} CTE/duplicate expression(s).")

        # Step 8: Match log entries to profile joins
        matches, unresolved = match_joins(profile_joins, log_entries)
        print(f"  Matched {len(matches)} expression(s) to actual cardinalities.")
        if unresolved:
            print(f"  [MATCH WARN] Unresolved physical joins: {len(unresolved)}")

        # Step 9: Save pre-update JSON snapshot for verification, then update
        pre_update_json = read_actual_cardinality_json()
        plan_stable = (
            prev_plan_text is not None and current_plan_text == prev_plan_text
        )
        _record_match_history(
            expr_match_history,
            matches,
            profile_joins,
            iteration,
            current_plan_fingerprint,
        )
        newly_unsafe = collect_quarantine_unsafe_expressions(
            matches, profile_joins, expr_match_history
        )
        unsafe_expressions.update(newly_unsafe)
        purge_unsafe_expressions_from_json(
            unsafe_expressions,
            query_nr=query_nr,
            iteration=iteration,
        )
        matches_to_update = [
            m for m in matches if m["expression"] not in unsafe_expressions
        ]
        skipped_unsafe = len(
            [m for m in matches if m["expression"] in unsafe_expressions]
        )
        if skipped_unsafe > 0:
            print(
                f"    [SKIP-UNSAFE] Skipping {skipped_unsafe} match(es) due to "
                f"ambiguity/context collision quarantine."
            )

        changes_made = update_actual_cardinality_json(
            matches_to_update,
            cte_exprs,
            query_nr=query_nr,
            iteration=iteration,
            plan_stable=plan_stable,
            plan_fingerprint=None,
        )

        # Step 10: Verification — runs on EVERY iteration >= 2
        # We verify against the PRE-UPDATE JSON, because that's what DuckDB loaded
        if iteration >= 2:
            post_update_json = read_actual_cardinality_json()
            verify_injection(
                log_entries,
                pre_update_json,
                matches,
                profile_joins,
                cte_exprs,
                iteration,
                plan_stable=plan_stable,
                injection_plan_fingerprint=None,
                post_update_json=post_update_json,
            )
            print(f"  Verification passed.")

        # Step 11: Check convergence
        # The plan has converged ONLY IF the physical plan hasn't changed
        # AND we didn't need to make any changes to the actual_cardinality.json
        if prev_plan_text is not None and current_plan_text == prev_plan_text and not changes_made:
            print(f"  Plan CONVERGED after {iteration} iterations.")
            # Timing comparison: if the plan changed at any point, compare perf
            timing_result = None
            if plan_changed_iterations:
                timing_result = compare_plan_timings(query_sql, query_nr)
            return _feedback_query_result(
                iteration,
                True,
                plan_changed_iterations,
                feedback_unique_plans=len(unique_plan_texts),
                final_plan_fingerprint=last_plan_fingerprint,
                n_injected=feedback_last_n_injected,
                timing=timing_result,
            )

        # Step 11b: Plan change and Oscillation detection
        if prev_plan_text is not None and current_plan_text != prev_plan_text:
            plan_changed_iterations.append(iteration)
            print(f"  Plan CHANGED on iteration {iteration}.")

            # Oscillation detection — if we've seen this exact plan before,
            # the optimizer is cycling between plans. This is a valid stopping point.
            if current_plan_text in seen_plan_structures:
                cycle_start = seen_plan_structures.index(current_plan_text) + 1
                cycle_len = iteration - cycle_start
                print(f"  Plan OSCILLATION detected: cycle of length {cycle_len} "
                      f"(iteration {cycle_start} == iteration {iteration}).")

                # Purge keys that oscillated
                oscillating_keys = set()
                for expr, hist in expr_match_history.items():
                    if len(hist) < 2:
                        continue
                    cards = [int(h.get("actual_cardinality", 0)) for h in hist]
                    if len(set(cards)) > 1:
                        oscillating_keys.add(expr)

                if oscillating_keys:
                    current_json = read_actual_cardinality_json()
                    purged = 0
                    for key in oscillating_keys:
                        if key in current_json:
                            del current_json[key]
                            purged += 1
                    if purged:
                        write_actual_cardinality_json(current_json)
                        print(f"  [OSCILLATION-CLEANUP] Removed {purged} oscillating key(s) from JSON.")
                        for k in sorted(oscillating_keys):
                            cards = [int(h.get("actual_cardinality", 0)) for h in expr_match_history[k]]
                            print(f"    {k[:140]}... cards={cards}")

                return _feedback_query_result(
                    iteration,
                    False,
                    plan_changed_iterations,
                    oscillation=True,
                    cycle_length=cycle_len,
                    feedback_unique_plans=len(unique_plan_texts),
                    final_plan_fingerprint=last_plan_fingerprint,
                    n_injected=feedback_last_n_injected,
                )

        if current_plan_text not in seen_plan_structures:
            seen_plan_structures.append(current_plan_text)
        prev_plan_text = current_plan_text

        if not changes_made and iteration > 1:
            # No new entries added and we already had injections — plan should converge next run
            print(f"  No new cardinality entries. Expecting convergence next iteration.")

    print(f"  [WARN] Query {query_nr} did NOT converge after {MAX_ITERATIONS} iterations!")
    return _feedback_query_result(
        MAX_ITERATIONS,
        False,
        plan_changed_iterations,
        feedback_unique_plans=len(unique_plan_texts),
        final_plan_fingerprint=last_plan_fingerprint,
        n_injected=feedback_last_n_injected,
    )


# ============================================================================
# MAIN
# ============================================================================

def check_benchmark_prerequisites():
    """Abort early if binary, database, or query directory is unusable."""
    assert os.path.exists(DUCKDB_BIN), f"DuckDB binary not found: {DUCKDB_BIN}"
    assert os.path.exists(DB_FILE), f"Database not found: {DB_FILE}"
    assert os.path.isdir(TPCDS_QUERY_DIR), (
        f"TPC-DS query directory missing: {TPCDS_QUERY_DIR}\n"
        "Run: python3 scripts/export_tpcds_query_files.py"
    )


def print_global_totals(results):
    """Print aggregate stats after all queries finish."""
    total_iterations = sum(r["iterations"] for r in results.values())
    total_plan_changes = sum(len(r["plan_changed_iterations"]) for r in results.values())
    total_injected_sum = sum(r.get("n_injected", 0) for r in results.values())
    sum_unique_plans = sum(r.get("feedback_unique_plans", 0) for r in results.values())
    n_converged = sum(1 for r in results.values() if r.get("converged"))
    n_oscillation = sum(1 for r in results.values() if r.get("oscillation"))
    n_errors = sum(1 for r in results.values() if r.get("error"))

    print("=" * 60)
    print("  GLOBAL TOTALS (sum over queries in this run)")
    print("=" * 60)
    print(f"  Total iterations completed:     {total_iterations}")
    print(f"  Total plan-change events:       {total_plan_changes}")
    print(f"  Sum of last-iter n_injected:    {total_injected_sum}")
    print(f"  Sum of feedback_unique_plans:   {sum_unique_plans}")
    print(f"  Converged / oscillation / error: {n_converged} / {n_oscillation} / {n_errors}")

    # Timing summary table (only queries where plan changed and timing was collected)
    timed = {qn: r["timing"] for qn, r in results.items() if r.get("timing")}
    if timed:
        print("=" * 60)
        print("  TIMING COMPARISON (vanilla vs feedback, plan-changed queries)")
        print("=" * 60)
        print(f"  {'Query':<8} {'Vanilla min':>12} {'Feedback min':>12} {'Speedup':>10}")
        print("  " + "-" * 46)
        for qn in sorted(timed):
            t = timed[qn]
            v_min = t.get("vanilla_min", float("inf"))
            f_min = t.get("feedback_min", float("inf"))
            if v_min < float("inf") and f_min < float("inf") and f_min > 0:
                speedup = v_min / f_min
                tag = "FASTER" if speedup > 1.0 else "SLOWER ⚠"
                print(f"  Q{qn:<6} {v_min:>11.3f}s {f_min:>11.3f}s {speedup:>8.2f}x {tag}")
            else:
                print(f"  Q{qn:<6} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    print("=" * 60)
    print("  Benchmark complete.")


def main():
    """Load queries from disk, run feedback per query, print per-query and global summaries."""
    print("=" * 60)
    print("  TPC-DS Cardinality Feedback Benchmark")
    print(f"  Scale Factor: {SCALE_FACTOR}")
    print(f"  Database: {DB_FILE}")
    print(f"  Queries: {TARGET_QUERIES}")
    print(f"  SQL dir: {TPCDS_QUERY_DIR}")
    print("=" * 60)

    check_benchmark_prerequisites()

    print("\nLoading TPC-DS queries from SQL files...")
    queries = load_tpcds_queries(TARGET_QUERIES)
    assert len(queries) > 0, "No queries loaded — check feedback_queries/tpcds/"
    print(f"  Loaded {len(queries)} queries.\n")

    results = {}
    for query_nr in TARGET_QUERIES:
        if query_nr not in queries:
            print(f"\n  [SKIP] Query {query_nr}: missing file q{query_nr:02d}.sql")
            continue
        results[query_nr] = run_single_query(query_nr, queries[query_nr])
        clear_actual_cardinality_json()
        clear_cardinality_log()

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Query':<10} {'Iterations':<12} {'Converged':<12} {'Plan Changes'}")
    print("-" * 50)

    for query_nr, result in sorted(results.items()):
        if result.get("oscillation"):
            status_str = f"Oscillation (cycle={result['cycle_length']})"
        elif result["converged"]:
            status_str = "Converged"
        elif result.get("error"):
            status_str = "Error"
        else:
            status_str = "Not converged"
        changes = len(result["plan_changed_iterations"])
        print(f"  Q{query_nr:<8} {result['iterations']:<12} {status_str:<25} {changes}")

    print_global_totals(results)


if __name__ == "__main__":
    main()
