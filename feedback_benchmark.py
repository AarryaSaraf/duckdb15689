"""
TPC-DS Cardinality Feedback Benchmark
======================================
Iteratively runs TPC-DS queries, captures actual join cardinalities from
the physical plan, injects them back into the optimizer via actual_cardinality.json,
and repeats until the physical plan converges (stops changing).

Uses JSON profiling to get structured plan output.
"""

import os
import re
import subprocess
import json
import sys
from collections import Counter

# ============================================================================
# CONSTANTS
# ============================================================================

DUCKDB_DIR = "/Users/Aarry/Desktop/15689/duckdb15689/"
DUCKDB_BIN = os.path.join(DUCKDB_DIR, "build/release/duckdb")
CARDINALITY_LOG = os.path.join(DUCKDB_DIR, "cardinality_log.txt")
ACTUAL_CARDINALITY_JSON = os.path.join(DUCKDB_DIR, "actual_cardinality.json")
PROFILE_OUTPUT = os.path.join(DUCKDB_DIR, "profile_output.json")

DB_FILE = "/Users/Aarry/Desktop/15689/tpcds_sf10.db"
SCALE_FACTOR = 10

# TARGET_QUERIES = list(range(5, 10))
TARGET_QUERIES = [7]
MAX_ITERATIONS = 20                 # safety cap per query

PYTHON_BIN = "/usr/local/bin/python3"

# Operators in the physical plan that represent joins
JOIN_OPERATOR_NAMES = {"HASH_JOIN", "NESTED_LOOP_JOIN", "PIECEWISE_MERGE_JOIN",
                       "CROSS_PRODUCT", "POSITIONAL_JOIN", "BLOCKWISE_NL_JOIN",
                       "IE_JOIN", "ASOF_JOIN", "DELIM_JOIN",
                       "LEFT_DELIM_JOIN", "RIGHT_DELIM_JOIN"}


# ============================================================================
# QUERY EXTRACTION
# ============================================================================

def extract_tpcds_queries(query_nrs):
    """
    Uses the Python duckdb package to extract TPC-DS query SQL text
    for the given query numbers. Returns a dict {query_nr: sql_string}.
    """
    code = f"""
import duckdb, json
con = duckdb.connect(':memory:')
con.execute('INSTALL tpcds; LOAD tpcds')
rows = con.execute('SELECT query_nr, query FROM tpcds_queries()').fetchall()
result = {{}}
for nr, sql in rows:
    if nr in {query_nrs}:
        result[nr] = sql
print(json.dumps(result))
"""
    proc = subprocess.run(
        [PYTHON_BIN, "-c", code],
        capture_output=True, text=True
    )
    assert proc.returncode == 0, f"Failed to extract queries: {proc.stderr}"

    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    assert len(lines) > 0, "No JSON output from query extraction"

    raw = json.loads(lines[-1])
    # Keys come back as strings; convert to int
    return {int(k): v for k, v in raw.items()}


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


# ============================================================================
# QUERY EXECUTION
# ============================================================================

def run_query_with_json_profile(query_sql):
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

    proc = subprocess.run(
        [DUCKDB_BIN, DB_FILE, "-c", full_sql],
        capture_output=True, text=True
    )

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

def get_descendant_tables(node):
    """
    Collect all table names from SEQ_SCAN nodes in the subtree rooted at `node`.
    Returns a set of table base names (stripped of schema prefix).
    """
    tables = set()
    op_name = node.get("operator_name", node.get("name", ""))
    extra = node.get("extra_info", {})
    if op_name == "SEQ_SCAN" and "Table" in extra:
        tbl = extra["Table"].split(".")[-1]  # strip schema prefix
        tables.add(tbl)
    for child in node.get("children", []):
        tables.update(get_descendant_tables(child))
    return tables


def extract_join_nodes(node):
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
    op_name = node.get("operator_name", node.get("name", ""))

    if op_name in JOIN_OPERATOR_NAMES:
        extra = node.get("extra_info", {})
        conditions_raw = extra.get("Conditions", "")
        join_type = extra.get("Join Type", "UNKNOWN")
        est_card_str = extra.get("Estimated Cardinality", "0")

        actual_card = node.get("operator_cardinality", node.get("cardinality", 0))
        desc_tables = get_descendant_tables(node)

        results.append({
            "operator_name": op_name,
            "actual_cardinality": int(actual_card),
            "conditions": conditions_raw.strip() if isinstance(conditions_raw, str) else str(conditions_raw),
            "join_type": join_type,
            "estimated_cardinality": int(est_card_str) if str(est_card_str).isdigit() else 0,
            "descendant_tables": desc_tables,
        })

    for child in node.get("children", []):
        results.extend(extract_join_nodes(child))

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
        LOGICAL_JOIN: RelSets: [...] Tables: [...] Filters: [...] Estimated Cardinality: N
    or:
        LOGICAL_JOIN: RelSets: [...] Tables: [...] Filters: [...] using INJECTED Cardinality: N

    Returns a list of dicts:
        {
            "expression": str,     # the full expression string (key for JSON)
            "tables": list[str],
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
    for line in lines:
        line = line.strip()
        if not line or "LOGICAL_JOIN" not in line:
            continue

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
        cardinality = float(card_str)

        # Parse tables from the expression
        tables_match = re.search(r"Tables: \[(.*?)\]", expression)
        tables = []
        if tables_match:
            tables = [t.strip() for t in tables_match.group(1).split(",") if t.strip()]

        # Parse filters from the expression
        filters_match = re.search(r"Filters: \[(.*)\]$", expression)
        filters = []
        if filters_match:
            filters_raw = filters_match.group(1)
            # Split by " AND " at the top level (filters are separated by AND)
            # Each filter is wrapped in parentheses like (a = b)
            filters = split_filter_string(filters_raw)

        entries.append({
            "expression": expression,
            "tables": tables,
            "filters": filters,
            "cardinality": cardinality,
            "is_injected": is_injected,
        })

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
    # Split by newline or ' AND ' (conditions can be multiline in the JSON)
    parts = re.split(r"\n| AND ", conditions_str)
    return [p.strip() for p in parts if p.strip()]


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

    Returns a list of (expression_string, actual_cardinality) tuples.
    """
    matches = []
    used_log_indices = set()

    # Normalize profile join conditions and prepare log conditions
    profile_normalized = []
    for pj in profile_joins:
        conds = parse_explain_conditions(pj["conditions"])
        norm = normalize_condition_set(conds)
        profile_normalized.append(norm)

    log_normalized = []
    log_table_sets = []
    for le in log_entries:
        log_normalized.append(normalize_condition_set(le["filters"]))
        log_table_sets.append(set(le["tables"]))

    # For each profile join, find best matching log entry
    for pidx, pj in enumerate(profile_joins):
        pnorm = profile_normalized[pidx]
        if not pnorm:
            continue  # No conditions (e.g., cross product)

        p_desc_tables = pj.get("descendant_tables", set())

        best_lidx = -1
        best_table_count = float("inf")
        best_cond_diff = float("inf")

        for lidx, le in enumerate(log_entries):
            if lidx in used_log_indices:
                continue

            lnorm = log_normalized[lidx]
            if not lnorm:
                continue

            # Condition check: profile conditions must be a subset of log conditions
            # (log has the full set of conditions for multi-table joins;
            #  the physical join only applies the outermost condition)
            if not (lnorm == pnorm or pnorm.issubset(lnorm)):
                continue

            # Table check: the physical join's descendant tables must be a subset
            # of the log entry's table set. This ensures the log entry covers
            # all tables that have been joined at this point in the plan.
            l_tables = log_table_sets[lidx]
            if p_desc_tables and not p_desc_tables.issubset(l_tables):
                continue

            # Scoring: prefer smallest table set (most specific match),
            # then smallest condition difference
            cond_diff = abs(len(lnorm) - len(pnorm))
            table_count = len(l_tables)
            if (table_count < best_table_count or
                (table_count == best_table_count and cond_diff < best_cond_diff)):
                best_table_count = table_count
                best_cond_diff = cond_diff
                best_lidx = lidx

        if best_lidx != -1:
            used_log_indices.add(best_lidx)
            actual_card = pj["actual_cardinality"]
            matches.append((log_entries[best_lidx]["expression"], actual_card))

    return matches


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

def update_actual_cardinality_json(matches, cte_expressions):
    """
    Read current actual_cardinality.json, add new entries from matches,
    and write back. Skips CTE duplicate expressions.

    For each match (expression, actual_cardinality):
    - If expression is a CTE duplicate, skip it with a warning.
    - If expression already exists with SAME cardinality, skip (no change).
    - If expression already exists with DIFFERENT cardinality, raise an alarm
      (this shouldn't happen for the same query/plan combination).
    - If expression is new, add it.

    Returns True if any new entries were added, False otherwise.
    """
    current = read_actual_cardinality_json()
    changes_made = False

    for expression, actual_card in matches:
        if expression in cte_expressions:
            print(f"    [SKIP-CTE] {expression}")
            continue

        if expression in current:
            existing_card = current[expression]
            if int(existing_card) != int(actual_card):
                print(f"    [ALARM] Cardinality mismatch for expression!")
                print(f"      Expression: {expression}")
                print(f"      Existing: {int(existing_card)}, New: {int(actual_card)}")
                # Update to new value (the latest run is most accurate)
                current[expression] = float(actual_card)
                changes_made = True
            # else: same value, no change needed
        else:
            current[expression] = float(actual_card)
            changes_made = True
            print(f"    [NEW] {expression} -> {actual_card}")

    if changes_made:
        write_actual_cardinality_json(current)

    return changes_made


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_injection(log_entries, pre_update_json, matches, profile_joins, cte_exprs, iteration):
    """
    Comprehensive verification checks for iteration 2+.

    Check 1: Every injected value in the log must match the corresponding
             value in the pre-update JSON (DuckDB loaded it correctly).
    Check 2: All pre-update JSON keys should appear in the cardinality log.
    Check 3: Every non-CTE expression must have been INJECTED (not Estimated)
             — meaning we captured it in a previous iteration.
    Check 4: For each injected expression, its injected cardinality must match
             the actual cardinality from the physical plan. If not, our matching
             between log expressions and physical joins is wrong.
    """
    print(f"    [VERIFY] Running verification for iteration {iteration}...")

    # ----- Check 1: Injected log values match the pre-update JSON -----
    injected_count = 0
    for entry in log_entries:
        if entry["is_injected"]:
            injected_count += 1
            expr = entry["expression"]
            assert expr in pre_update_json, (
                f"[VERIFY FAIL] Check 1: INJECTED expression not found in "
                f"pre-update JSON: {expr}"
            )
            json_val = pre_update_json[expr]
            log_val = entry["cardinality"]
            assert abs(json_val - log_val) < 1.0, (
                f"[VERIFY FAIL] Check 1: INJECTED cardinality mismatch for "
                f"{expr}: JSON(pre-update)={json_val}, Log={log_val}"
            )
    print(f"    [VERIFY] Check 1 PASSED: {injected_count} injected log values "
          f"match the pre-update JSON.")

    # ----- Check 2: All pre-update JSON keys should be in the log -----
    log_expressions = {entry["expression"] for entry in log_entries}
    missing_from_log = []
    for expr in pre_update_json:
        if expr not in log_expressions:
            missing_from_log.append(expr)
    if missing_from_log:
        for expr in missing_from_log:
            print(f"    [VERIFY] Check 2 WARN: JSON key not in log: {expr}")
    else:
        print(f"    [VERIFY] Check 2 PASSED: All {len(pre_update_json)} JSON keys "
              f"found in the cardinality log.")

    # ----- Check 3: Every PREVIOUSLY-KNOWN matched physical plan join must be INJECTED -----
    # Only check joins whose expression existed in the pre-update JSON.
    # New joins that appeared due to plan changes are legitimately not-yet-injected.
    log_by_expr = {}
    for entry in log_entries:
        if entry["expression"] not in log_by_expr:
            log_by_expr[entry["expression"]] = entry

    matched_not_injected = []
    matched_injected = []
    matched_new = []  # newly-discovered joins (not in pre-update JSON)
    for expr, actual in matches:
        if expr in cte_exprs:
            continue  # CTEs are not expected to be injected
        if expr not in pre_update_json:
            matched_new.append(expr)  # new join from plan change, OK
            continue
        if expr in log_by_expr and log_by_expr[expr]["is_injected"]:
            matched_injected.append(expr)
        else:
            matched_not_injected.append(expr)

    if matched_new:
        print(f"    [VERIFY] Check 3 INFO: {len(matched_new)} new join(s) from "
              f"plan change (not previously injected, OK).")

    if matched_not_injected:
        print(f"    [VERIFY] Check 3 FAIL: {len(matched_not_injected)} previously-known "
              f"join(s) were NOT injected:")
        for expr in matched_not_injected:
            print(f"      NOT INJECTED: {expr}")
        assert False, (
            f"[VERIFY FAIL] Check 3: {len(matched_not_injected)} previously-known "
            f"joins were not injected. See output above."
        )
    else:
        print(f"    [VERIFY] Check 3 PASSED: All {len(matched_injected)} previously-known "
              f"joins (non-CTE) were INJECTED.")

    # ----- Check 4: Injected cardinality must match actual plan cardinality -----
    # For each match (expression -> actual_cardinality), if that expression was
    # INJECTED, the injected value should equal the actual cardinality.
    # This ONLY holds for converged plans (same structure). For changed plans,
    # actual cardinality may differ from what was injected.
    matched_exprs = {expr: actual for expr, actual in matches}
    injected_vs_actual_mismatches = []
    for entry in log_entries:
        if entry["is_injected"] and entry["expression"] in matched_exprs:
            injected_val = int(entry["cardinality"])
            actual_val = matched_exprs[entry["expression"]]
            if injected_val != actual_val:
                injected_vs_actual_mismatches.append(
                    (entry["expression"], injected_val, actual_val)
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


# ============================================================================
# MAIN LOOP PER QUERY
# ============================================================================

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

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n  --- Iteration {iteration} ---")

        # Step 2: Clear the cardinality log (but NOT the JSON — it accumulates)
        clear_cardinality_log()

        # Step 3: Run the query
        profile = run_query_with_json_profile(query_sql)
        if profile is None:
            print(f"  [ERROR] Query {query_nr} failed on iteration {iteration}. Skipping.")
            return {
                "iterations": iteration,
                "converged": False,
                "plan_changed_iterations": plan_changed_iterations,
                "error": True,
            }

        # Step 4: Get plan structure and parse everything FIRST
        root = profile.get("children", [profile])[0] if profile.get("children") else profile
        current_plan_text = get_plan_structure_text(root)

        # Step 5: Parse joins from the JSON profile
        profile_joins = extract_join_nodes(root)
        print(f"  Found {len(profile_joins)} join(s) in physical plan.")

        # Step 6: Parse the cardinality log
        log_entries = parse_cardinality_log()
        print(f"  Found {len(log_entries)} log entries in cardinality_log.txt.")

        # Step 7: Detect CTE duplicates
        cte_exprs = detect_cte_duplicates(log_entries)
        if cte_exprs:
            print(f"  Detected {len(cte_exprs)} CTE/duplicate expression(s).")

        # Step 8: Match log entries to profile joins
        matches = match_joins(profile_joins, log_entries)
        print(f"  Matched {len(matches)} expression(s) to actual cardinalities.")

        # Step 9: Save pre-update JSON snapshot for verification, then update
        pre_update_json = read_actual_cardinality_json()
        changes_made = update_actual_cardinality_json(matches, cte_exprs)

        # Step 10: Verification — runs on EVERY iteration >= 2
        # We verify against the PRE-UPDATE JSON, because that's what DuckDB loaded
        if iteration >= 2:
            verify_injection(log_entries, pre_update_json, matches,
                           profile_joins, cte_exprs, iteration)
            print(f"  Verification passed.")

        # Step 11: Check convergence — plan structure hasn't changed
        if prev_plan_text is not None and current_plan_text == prev_plan_text:
            print(f"  Plan CONVERGED after {iteration} iterations.")
            return {
                "iterations": iteration,
                "converged": True,
                "plan_changed_iterations": plan_changed_iterations,
                "error": False,
            }

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
            return {
                "iterations": iteration,
                "converged": False,
                "plan_changed_iterations": plan_changed_iterations,
                "oscillation": True,
                "cycle_length": cycle_len,
                "error": False,
            }

        seen_plan_structures.append(current_plan_text)
        prev_plan_text = current_plan_text

        if not changes_made and iteration > 1:
            # No new entries added and we already had injections — plan should converge next run
            print(f"  No new cardinality entries. Expecting convergence next iteration.")

    print(f"  [WARN] Query {query_nr} did NOT converge after {MAX_ITERATIONS} iterations!")
    return {
        "iterations": MAX_ITERATIONS,
        "converged": False,
        "plan_changed_iterations": plan_changed_iterations,
        "error": False,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main entry point. Extracts TPC-DS queries, runs the iterative feedback
    loop for each, and prints a final summary.
    """
    print("=" * 60)
    print("  TPC-DS Cardinality Feedback Benchmark")
    print(f"  Scale Factor: {SCALE_FACTOR}")
    print(f"  Database: {DB_FILE}")
    print(f"  Queries: {TARGET_QUERIES}")
    print("=" * 60)

    # Verify DuckDB binary exists
    assert os.path.exists(DUCKDB_BIN), f"DuckDB binary not found: {DUCKDB_BIN}"
    assert os.path.exists(DB_FILE), f"Database not found: {DB_FILE}"

    # Extract queries
    print("\nExtracting TPC-DS queries...")
    queries = extract_tpcds_queries(TARGET_QUERIES)
    assert len(queries) > 0, "No queries extracted!"
    print(f"  Extracted {len(queries)} queries.\n")

    # Run each query
    results = {}
    for query_nr in TARGET_QUERIES:
        if query_nr not in queries:
            print(f"\n  [SKIP] Query {query_nr} not found in TPC-DS query set.")
            continue

        result = run_single_query(query_nr, queries[query_nr])
        results[query_nr] = result

        # Clean up for next query
        clear_actual_cardinality_json()
        clear_cardinality_log()

    # Final Summary
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

    print("=" * 60)
    print("  Benchmark complete.")


if __name__ == "__main__":
    main()
