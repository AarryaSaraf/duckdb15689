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
  DUCKDB_FEEDBACK_MAX_QUERIES — if set, only Q1..QN run (smoke tests)
"""

import ast
import hashlib
import json
import os
import re
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
DB_FILE = os.environ.get("DUCKDB_FEEDBACK_DB", "/Users/Aarry/Desktop/15689/tpcds_sf200.db").strip()
SCALE_FACTOR = int(os.environ.get("DUCKDB_FEEDBACK_SF", "200"))
TARGET_QUERIES = list(range(1, 100))
_MQQ = os.environ.get("DUCKDB_FEEDBACK_MAX_QUERIES")
if _MQQ:
    _nq = max(1, min(99, int(_MQQ.strip())))
    TARGET_QUERIES = list(range(1, _nq + 1))
MAX_ITERATIONS = 20
MAIN_QUERY_TIMEOUT_SEC = int(os.getenv("DUCKDB_BENCHMARK_MAIN_QUERY_TIMEOUT_SEC", "600"))

# --- Matching / JSON drift thresholds ---
STABLE_PLAN_ABS_DRIFT_TOLERANCE = 100
STABLE_PLAN_REL_DRIFT_TOLERANCE = 0.001  # 0.1%
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
                if table_name:
                    tables.append(table_name)
        else:
            tables_match = re.search(r"Tables: \[(.*?)\]", expression)
            if tables_match:
                tables = [t.strip() for t in tables_match.group(1).split(",") if t.strip()]

        # Parse filters from the expression
        filters_match = re.search(r"Filters: \[(.*)\](?: CtxOcc: \d+)?$", expression)
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

    # Normalize profile join conditions and prepare log conditions
    profile_normalized = []
    for pj in profile_joins:
        conds = parse_explain_conditions(pj["conditions"])
        norm = normalize_condition_set(conds)
        norm = frozenset(c for c in norm if not _is_tautology_condition(c))
        profile_normalized.append(norm)

    log_normalized = []
    log_table_sets = []
    for le in log_entries:
        norm = normalize_condition_set(le["filters"])
        norm = frozenset(c for c in norm if not _is_tautology_condition(c))
        log_normalized.append(norm)
        log_table_sets.append(set(le["tables"]))

    # For each profile join, find best matching log entry
    for pidx, pj in enumerate(profile_joins):
        pnorm = profile_normalized[pidx]
        if not pnorm:
            continue  # No conditions (e.g., cross product)

        p_desc_tables = pj.get("descendant_tables", set())
        p_conditions = pj.get("conditions", "")
        actual_card = int(pj["actual_cardinality"])
        p_join_type = pj.get("join_type", "")
        p_estimated_cardinality = int(pj.get("estimated_cardinality", 0))
        p_operator_name = pj.get("operator_name", "")
        p_plan_path = pj.get("plan_path", [])

        best_lidx = -1
        best_stage = 99
        best_table_delta = float("inf")
        best_cond_diff = float("inf")
        best_est_delta = float("inf")
        best_expr = None
        candidate_count = 0
        p_lineage_incomplete = pj.get("lineage_incomplete", False)
        candidate_debug_rows = []

        for lidx, le in enumerate(log_entries):
            if lidx in used_log_indices:
                continue

            lnorm = log_normalized[lidx]
            if not lnorm:
                continue

            if p_lineage_incomplete:
                if not (lnorm == pnorm or pnorm.issubset(lnorm)):
                    continue
            else:
                if lnorm != pnorm:
                    continue

            l_tables = log_table_sets[lidx]
            if p_desc_tables and p_lineage_incomplete and not (p_desc_tables & l_tables):
                continue
            if p_desc_tables and not p_lineage_incomplete and l_tables != p_desc_tables:
                continue

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

            candidate_count += 1
            expr = le["expression"]
            cand_card = int(le["cardinality"])
            cand_abs_delta = abs(cand_card - actual_card)
            cand_est_delta = abs(cand_card - p_estimated_cardinality)
            relsets_match = re.search(r"RelSets: (\[[^\]]*\])", expr)
            numrels_match = re.search(r"NumRels: (\[[^\]]*\])", expr)
            ctx_inputs_match = re.search(r"CtxInputCards: (\[[^\]]*\])", expr)
            filters_match = re.search(r"Filters: \[(.*)\](?: CtxOcc: \d+)?$", expr)
            expr_relsets = relsets_match.group(1) if relsets_match else None
            expr_numrels = numrels_match.group(1) if numrels_match else None
            expr_ctx_inputs = ctx_inputs_match.group(1) if ctx_inputs_match else None
            expr_filters = filters_match.group(1) if filters_match else None
            if len(candidate_debug_rows) < 6:
                candidate_debug_rows.append(
                    {
                        "log_index": lidx,
                        "expr_tail": expr[-180:] if len(expr) > 180 else expr,
                        "stage": stage,
                        "table_delta": table_delta,
                        "cond_diff": cond_diff,
                        "candidate_cardinality": cand_card,
                        "candidate_is_injected": bool(le["is_injected"]),
                        "candidate_abs_delta_to_actual": cand_abs_delta,
                        "candidate_abs_delta_to_profile_est": cand_est_delta,
                        "candidate_relsets": expr_relsets,
                        "candidate_numrels": expr_numrels,
                        "candidate_ctx_input_cards": expr_ctx_inputs,
                        "candidate_filters_tail": (
                            expr_filters[-180:] if expr_filters and len(expr_filters) > 180 else expr_filters
                        ),
                        "log_tables": sorted(list(l_tables)),
                    }
                )
            use_estimated_tiebreak = p_estimated_cardinality > 0
            if ((use_estimated_tiebreak and (
                    cand_est_delta < best_est_delta or
                    (cand_est_delta == best_est_delta and stage < best_stage) or
                    (cand_est_delta == best_est_delta and stage == best_stage and table_delta < best_table_delta) or
                    (cand_est_delta == best_est_delta and stage == best_stage and table_delta == best_table_delta and cond_diff < best_cond_diff) or
                    (cand_est_delta == best_est_delta and stage == best_stage and table_delta == best_table_delta and cond_diff == best_cond_diff and
                     (best_expr is None or expr < best_expr))
                )) or
                ((not use_estimated_tiebreak) and (
                    stage < best_stage or
                    (stage == best_stage and table_delta < best_table_delta) or
                    (stage == best_stage and table_delta == best_table_delta and cond_diff < best_cond_diff) or
                    (stage == best_stage and table_delta == best_table_delta and cond_diff == best_cond_diff and
                     (best_expr is None or expr < best_expr))
                ))):
                best_est_delta = cand_est_delta
                best_stage = stage
                best_table_delta = table_delta
                best_cond_diff = cond_diff
                best_lidx = lidx
                best_expr = expr

        if best_lidx != -1:
            used_log_indices.add(best_lidx)
            matches.append({
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
            })
        else:
            unresolved.append({
                "profile_join_index": pidx,
                "conditions": pj.get("conditions", ""),
                "descendant_tables": sorted(list(p_desc_tables)),
                "lineage_incomplete": p_lineage_incomplete,
                "actual_cardinality": int(actual_card),
                "estimated_cardinality": int(p_estimated_cardinality),
                "plan_path": list(p_plan_path) if isinstance(p_plan_path, list) else [],
            })

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
# VERIFICATION
# ============================================================================

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
):
    """
    Comprehensive verification checks for iteration 2+.

    Check 1: Injected log lines should match pre-update JSON keys/values; if not, WARN.
    Check 2: All pre-update JSON keys should appear in the cardinality log.
    Check 3: Previously-known matched joins should have an INJECTED line in the log;
             if not, WARN (plan change / optimizer path — same spirit as Check 4).
    Check 4: For each injected expression, its injected cardinality must match
             the actual cardinality from the physical plan. If not, our matching
             between log expressions and physical joins is wrong.
    """
    print(f"    [VERIFY] Running verification for iteration {iteration}...")

    active_pre_update_json = project_json_for_fingerprint(pre_update_json, injection_plan_fingerprint)

    # ----- Check 1: Injected log values vs pre-update JSON -----
    injected_count = 0
    check1_warns = []
    for entry in log_entries:
        if entry["is_injected"]:
            injected_count += 1
            expr = entry["expression"]
            log_val = entry["cardinality"]
            if expr not in active_pre_update_json:
                check1_warns.append(
                    ("missing_key", expr, None, None)
                )
                continue
            json_val = active_pre_update_json[expr]
            if abs(json_val - log_val) >= 1.0:
                check1_warns.append(
                    ("value_mismatch", expr, json_val, log_val)
                )
    if check1_warns:
        print(f"    [VERIFY] Check 1 WARN: {len(check1_warns)} injected log line(s) "
              f"do not align with pre-update JSON (plan/JSON drift or float noise):")
        for kind, expr, jv, lv in check1_warns[:10]:
            ex = expr if len(expr) <= 160 else expr[:160] + "..."
            if kind == "missing_key":
                print(f"      key not in pre-update JSON: {ex}")
            else:
                print(f"      value mismatch: JSON={jv}, Log={lv} — {ex}")
        if len(check1_warns) > 10:
            print(f"      ... and {len(check1_warns) - 10} more")
    else:
        print(f"    [VERIFY] Check 1 PASSED: {injected_count} injected log values "
              f"match the pre-update JSON.")

    # ----- Check 2: All pre-update JSON keys should be in the log -----
    log_expressions = {entry["expression"] for entry in log_entries}
    missing_from_log = []
    for expr in active_pre_update_json:
        if expr not in log_expressions:
            missing_from_log.append(expr)
    if missing_from_log:
        for expr in missing_from_log:
            print(f"    [VERIFY] Check 2 WARN: JSON key not in log: {expr}")
    else:
        print(f"    [VERIFY] Check 2 PASSED: All {len(active_pre_update_json)} JSON keys "
              f"found in the cardinality log.")

    # ----- Check 3: Every PREVIOUSLY-KNOWN matched physical plan join must be INJECTED -----
    # Only check joins whose expression existed in the pre-update JSON.
    # New joins that appeared due to plan changes are legitimately not-yet-injected.
    # The same expression string may appear on many log lines; injection holds if
    # ANY line with that expression is marked INJECTED (not only the first/last).
    def _expr_has_injected_line(expr):
        return any(
            e["expression"] == expr and e["is_injected"] for e in log_entries
        )

    matched_not_injected = []
    matched_injected = []
    matched_new = []  # newly-discovered joins (not in pre-update JSON)
    for match in matches:
        expr = match["expression"]
        actual = match["actual_cardinality"]
        if expr in cte_exprs:
            continue  # CTEs are not expected to be injected
        if expr not in active_pre_update_json:
            matched_new.append(expr)  # new join from plan change, OK
            continue
        if _expr_has_injected_line(expr):
            matched_injected.append(expr)
        else:
            matched_not_injected.append(expr)

    if matched_new:
        print(f"    [VERIFY] Check 3 INFO: {len(matched_new)} new join(s) from "
              f"plan change (not previously injected, OK).")

    if matched_not_injected:
        print(f"    [VERIFY] Check 3 WARN: {len(matched_not_injected)} previously-known "
              f"join(s) have no INJECTED line in this run's cardinality log (plan may have "
              f"changed, or the optimizer did not apply actual_cardinality.json for that key):")
        for expr in matched_not_injected:
            print(f"      NOT INJECTED: {expr}")
    else:
        print(f"    [VERIFY] Check 3 PASSED: All {len(matched_injected)} previously-known "
              f"joins (non-CTE) were INJECTED.")

    # ----- Check 4: Injected cardinality must match actual plan cardinality -----
    # For each match (expression -> actual_cardinality), if that expression was
    # INJECTED, the injected value should equal the actual cardinality.
    # This ONLY holds for converged plans (same structure). For changed plans,
    # actual cardinality may differ from what was injected.
    matched_exprs = {m["expression"]: m["actual_cardinality"] for m in matches}
    injected_vs_actual_mismatches = []
    tolerated_small_drifts = []
    for entry in log_entries:
        if entry["is_injected"] and entry["expression"] in matched_exprs:
            expr = entry["expression"]
            injected_val = int(entry["cardinality"])
            actual_val = matched_exprs[expr]
            abs_delta = abs(int(injected_val) - int(actual_val))
            max_mag = max(abs(int(injected_val)), abs(int(actual_val)), 1)
            rel_delta = float(abs_delta) / float(max_mag)
            is_small_stable_drift = (
                plan_stable is True
                and abs_delta > 0
                and (
                    abs_delta <= STABLE_PLAN_ABS_DRIFT_TOLERANCE
                    or rel_delta <= STABLE_PLAN_REL_DRIFT_TOLERANCE
                )
            )
            if is_small_stable_drift:
                tolerated_small_drifts.append((entry["expression"], injected_val, actual_val, abs_delta, rel_delta))
            elif injected_val != actual_val:
                injected_vs_actual_mismatches.append(
                    (expr, injected_val, actual_val)
                )

    if injected_vs_actual_mismatches:
        print(f"    [VERIFY] Check 4 INFO: {len(injected_vs_actual_mismatches)} "
              f"injected != actual (plan may have changed):")
        for expr, inj, act in injected_vs_actual_mismatches:
            print(f"      {expr}")
            print(f"        Injected: {inj}, Actual: {act}")
        # Note: this is informational, not an assertion failure, because
        # if the plan structure changed, actual cardinality naturally differs.
        # The oscillation detector handles this case.
    else:
        print(f"    [VERIFY] Check 4 PASSED: All injected cardinalities match "
              f"actual plan cardinalities.")
    if tolerated_small_drifts:
        print(f"    [VERIFY] Check 4 INFO: tolerated {len(tolerated_small_drifts)} "
              f"small stable-plan drift(s) (treated as measurement noise).")

    # ----- Check 5: Each match must bind to a distinct cardinality-log line -----
    # The same LOGICAL_JOIN text may appear on multiple log lines (e.g. UNION); those
    # are different entries (different log_index). Duplicate expression strings are OK;
    # reusing the same log line for two physical joins is not.
    log_indices = [m["log_index"] for m in matches]
    dup_log_idx = [i for i, c in Counter(log_indices).items() if c > 1]
    if dup_log_idx:
        print(f"    [VERIFY] Check 5 WARN: duplicate log_index in matches (matcher bug?) "
              f": {dup_log_idx}")
    matched_expr_list = [m["expression"] for m in matches]
    dup_exprs = [expr for expr, c in Counter(matched_expr_list).items() if c > 1]
    if not dup_log_idx:
        if dup_exprs:
            print(f"    [VERIFY] Check 5 PASSED: {len(matches)} mappings; {len(dup_exprs)} "
                  f"expression string(s) repeated across distinct log line(s) (OK).")
        else:
            print(f"    [VERIFY] Check 5 PASSED: {len(matches)} mappings; all expression strings unique.")

    # ----- Check 6: Mapping coverage for joins with conditions -----
    joins_with_conditions = 0
    for pj in profile_joins:
        if parse_explain_conditions(pj.get("conditions", "")):
            joins_with_conditions += 1
    unmatched = joins_with_conditions - len(matches)
    print(f"    [VERIFY] Check 6 INFO: matched {len(matches)}/{joins_with_conditions} "
          f"joins-with-conditions (unmatched={max(unmatched, 0)}).")

    # ----- Check 7: Ambiguous matches resolved by estimate-proxy -----
    ambiguous_matches = [m for m in matches if int(m.get("candidate_count", 0)) > 1]
    if not ambiguous_matches:
        print("    [VERIFY] Check 7 PASSED: no ambiguous match candidates.")
    else:
        stage0_count = sum(1 for m in ambiguous_matches if int(m.get("selected_stage", 99)) == 0)
        non_stage0 = [m for m in ambiguous_matches if int(m.get("selected_stage", 99)) > 0]
        print(
            f"    [VERIFY] Check 7 INFO: {len(ambiguous_matches)} ambiguous match(es); "
            f"stage0={stage0_count}, non_stage0={len(non_stage0)}."
        )
        if non_stage0:
            print("    [VERIFY] Check 7 INFO: non-stage0 ambiguous selections (estimate-proxy likely applied):")
            for m in non_stage0[:6]:
                expr = m.get("expression", "")
                expr_tail = expr[-180:] if len(expr) > 180 else expr
                print(
                    f"      log_index={m.get('log_index')}, stage={m.get('selected_stage')}, "
                    f"est_delta={m.get('selected_abs_delta_to_profile_est')}, "
                    f"profile_est={m.get('profile_estimated_cardinality')}, "
                    f"path={m.get('profile_plan_path')}: {expr_tail}"
                )

# ============================================================================
# MAIN LOOP PER QUERY
# ============================================================================


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

    prev_plan_text = None
    plan_changed_iterations = []
    seen_plan_structures = []  # Track all seen plan structures for oscillation detection
    expr_match_history = {}
    unsafe_expressions = set()
    unique_plan_texts = set()
    last_plan_fingerprint = None
    # Cardinality lines marked INJECTED on the last completed iteration (for benchmarks).
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
        active_pre_update_json = project_json_for_fingerprint(
            pre_update_json, None
        )
        plan_stable = (
            prev_plan_text is not None and current_plan_text == prev_plan_text
        )
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
        # Strict safety gate: never inject ambiguous matches or conflicting
        # same-key cardinalities (beyond tolerated tiny drift).
        newly_unsafe = set()
        for m in matches:
            if int(m.get("candidate_count", 0)) > 1:
                expr = m["expression"]
                print(f"    [ALARM-AMBIGUOUS-MATCH] {expr}")
                newly_unsafe.add(expr)
        batch_actuals = {}
        for m in matches:
            batch_actuals.setdefault(m["expression"], set()).add(
                int(m.get("actual_cardinality", 0))
            )
        for expr, vals in batch_actuals.items():
            if len(vals) > 1:
                print(
                    f"    [ALARM-CONTEXT-COLLISION] same key has conflicting actuals in batch: "
                    f"{sorted(list(vals))} -- {expr}"
                )
                newly_unsafe.add(expr)
        # Dynamic-filter guard: if a matched subtree scan has dynamic filter columns
        # outside this join's own condition columns, treat key as context-unsafe.
        for m in matches:
            expr = m["expression"]
            pidx = m.get("profile_join_index")
            if not (isinstance(pidx, int) and 0 <= pidx < len(profile_joins)):
                continue
            pj = profile_joins[pidx]
            cond_cols = extract_condition_columns(pj.get("conditions", ""))
            dyn_cols = extract_dynamic_filter_columns(
                pj.get("subtree_operator_signatures", [])
            )
            extra_dyn_cols = sorted([c for c in dyn_cols if c not in cond_cols])
            if extra_dyn_cols:
                print(
                    f"    [ALARM-DYNAMIC-FILTER-CONTEXT] dynamic filter cols outside join "
                    f"condition: {extra_dyn_cols} -- {expr}"
                )
                newly_unsafe.add(expr)
        for expr, hist in expr_match_history.items():
            if len(hist) < 2:
                continue
            prev = hist[-2]
            curr = hist[-1]
            prev_actual = int(prev.get("actual_cardinality", 0))
            curr_actual = int(curr.get("actual_cardinality", 0))
            if prev_actual == curr_actual:
                continue
            abs_delta = abs(prev_actual - curr_actual)
            max_mag = max(abs(prev_actual), abs(curr_actual), 1)
            rel_delta = float(abs_delta) / float(max_mag)
            if (
                abs_delta > STABLE_PLAN_ABS_DRIFT_TOLERANCE
                and rel_delta > STABLE_PLAN_REL_DRIFT_TOLERANCE
            ):
                print(
                    f"    [ALARM-CONTEXT-COLLISION] same key changed cardinality across iterations: "
                    f"prev={prev_actual}, curr={curr_actual} -- {expr}"
                )
                newly_unsafe.add(expr)
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
            verify_injection(
                log_entries,
                pre_update_json,
                matches,
                profile_joins,
                cte_exprs,
                iteration,
                plan_stable=plan_stable,
                injection_plan_fingerprint=None,
            )
            print(f"  Verification passed.")

        # Step 11: Check convergence — plan structure hasn't changed
        if prev_plan_text is not None and current_plan_text == prev_plan_text:
            print(f"  Plan CONVERGED after {iteration} iterations.")
            return _feedback_query_result(
                iteration,
                True,
                plan_changed_iterations,
                feedback_unique_plans=len(unique_plan_texts),
                final_plan_fingerprint=last_plan_fingerprint,
                n_injected=feedback_last_n_injected,
            )

        if prev_plan_text is not None:
            plan_changed_iterations.append(iteration)
            print(f"  Plan CHANGED on iteration {iteration}.")

        # Step 11b: Oscillation detection — if we've seen this exact plan before,
        # the optimizer is cycling between plans. This is a valid stopping point.
        if current_plan_text in seen_plan_structures:
            cycle_start = seen_plan_structures.index(current_plan_text) + 1
            cycle_len = iteration - cycle_start
            print(f"  Plan OSCILLATION detected: cycle of length {cycle_len} "
                  f"(iteration {cycle_start} == iteration {iteration}).")
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
