"""
Oracle Cardinality Injection Benchmark
=======================================
For each TPC-DS query (all 99):
1. Run EXPLAIN to trigger optimizer logging (dry-run).
2. Parse cardinality_log.txt (optional SQL_COUNT_QUERY lines; else COUNT(*) SQL is synthesized from tables + Filters).
3. Execute each COUNT(*) subquery to get the true (oracle) cardinality.
   - If subquery fails (syntax error, projected columns, etc.) → skip.
   - If subquery times out (>120s) → skip, proceed with regular estimation.
4. Write all {expression_key: oracle_cardinality} to actual_cardinality.json.
5. Re-run the real query with injections active.
6. Verify injected values match oracle values.  ALARM if mismatch.
6b. (V6, **on by default**) On the injected run's JSON profile, ALARM when matched
    **physical joins** exceed the symmetric est-vs-actual tolerance. Pass
    ``--disable-profile-ea`` (or ``--disable-prof`` / ``--skip-profile-ea``) to turn
    V6 off. ``--profile-ea-all-operators`` is a noisier debug mode. Same fields as
    EXPLAIN (ANALYZE, FORMAT JSON).
7. Compare plan structure vs baseline (no injection).

Uses SF10 by default.  Focus on cardinalities, not timing.

DuckDB subprocess inherits env for log/JSON paths (see DUCKDB_CARDINALITY_LOG,
DUCKDB_ACTUAL_CARDINALITY_JSON, DUCKDB_FEEDBACK_PLAN_FINGERPRINT in the C++ estimator).

Default strict mode: if the same RelSets:[...] appears with multiple distinct SQL_COUNT_QUERY
texts in one explain pass, skip oracle keys for that relation-set (no single canonical COUNT).
Pass --oracle-permissive-context to inject all keys anyway (legacy behavior).

CLI: python oracle_benchmark.py --audit-sql-counts [cardinality_log.txt [oracle_sql_audit.json]]
  Re-runs every logged SQL_COUNT_QUERY (reason ok) against DB_FILE; writes JSON report and
  prints summary (exit 2 if any SQL error or timeout).

Programmatic: run_single_query(..., baseline_profile=..., skip_baseline_phase=True) reuses a
prior vanilla JSON profile so unified_benchmark.py avoids a duplicate baseline profile run.
V6 runs by default (``skip_profile_ea=False``); pass ``skip_profile_ea=True`` to skip.
Use ``profile_ea_joins_only=False`` to mirror CLI ``--profile-ea-all-operators``.
"""

import os
import re
import subprocess
import json
import sys
import time
import hashlib
import datetime
from collections import Counter

# ============================================================================
# CONSTANTS
# ============================================================================

DUCKDB_DIR = "/Users/Aarry/Desktop/15689/duckdb15689/"
DUCKDB_BIN = os.path.join(DUCKDB_DIR, "build/release/duckdb")
CARDINALITY_LOG = os.path.join(DUCKDB_DIR, "cardinality_log.txt")
ACTUAL_CARDINALITY_JSON = os.path.join(DUCKDB_DIR, "actual_cardinality.json")
PROFILE_OUTPUT = os.path.join(DUCKDB_DIR, "profile_output.json")
RESULTS_LOG = os.path.join(DUCKDB_DIR, "oracle_results.log")

DB_FILE = "/Users/Aarry/Desktop/15689/tpcds_sf200.db"
SCALE_FACTOR = 200

# All 99 TPC-DS queries
TARGET_QUERIES = list(range(0, 100))
# TARGET_QUERIES = [19, 33, 56, 60, 82]

SUBQUERY_TIMEOUT_SEC = 180    # Skip subqueries that take longer than this.

PYTHON_BIN = "/usr/local/bin/python3"

# Strict RelSets-SQL: skip oracle keys when one explain pass logs multiple distinct
# SQL_COUNT strings for the same RelSets tuple. Disabled via --oracle-permissive-context.
ORACLE_STRICT_RELSET_SQL = True

# V6: symmetric relative error |est-actual|/max(|est|,|actual|,1) above this → ALARM
PROFILE_EA_CARD_TOLERANCE = 0.01

# Physical operators whose ``Estimated Cardinality`` comes from the join-order
# pipeline (same place oracle JSON is consulted). Other ops use stats/heuristics.
PROFILE_EA_JOIN_OPERATOR_NAMES = frozenset(
    {
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
)


# ============================================================================
# LOGGING
# ============================================================================

_log_file_handle = None

def init_log():
    global _log_file_handle
    _log_file_handle = open(RESULTS_LOG, "w")
    _log(f"Oracle Cardinality Injection Benchmark")
    _log(f"Started: {datetime.datetime.now().isoformat()}")
    _log(f"Scale Factor: {SCALE_FACTOR}")
    _log(f"Database: {DB_FILE}")
    _log(f"Queries: 1-99")
    _log("=" * 100)

def _log(msg):
    """Write to both stdout and the results log file."""
    print(msg, flush=True)
    if _log_file_handle:
        _log_file_handle.write(msg + "\n")
        _log_file_handle.flush()

def close_log():
    global _log_file_handle
    if _log_file_handle:
        _log_file_handle.close()
        _log_file_handle = None


# ============================================================================
# FILE MANAGEMENT
# ============================================================================

def clear_cardinality_log():
    with open(CARDINALITY_LOG, "w"):
        pass


def clear_actual_cardinality_json():
    if os.path.exists(ACTUAL_CARDINALITY_JSON):
        os.remove(ACTUAL_CARDINALITY_JSON)


def write_actual_cardinality_json(data):
    with open(ACTUAL_CARDINALITY_JSON, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# QUERY EXTRACTION
# ============================================================================

def extract_tpcds_queries(query_nrs):
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
    proc = subprocess.run([PYTHON_BIN, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Failed to extract queries: {proc.stderr}"
    lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    raw = json.loads(lines[-1])
    return {int(k): v for k, v in raw.items()}


# ============================================================================
# QUERY EXECUTION
# ============================================================================

def run_duckdb_sql(sql_text):
    """Run SQL via DuckDB CLI. Returns (stdout, stderr, returncode)."""
    proc = subprocess.run(
        [DUCKDB_BIN, DB_FILE, "-c", sql_text],
        capture_output=True, text=True,
    )
    return proc.stdout, proc.stderr, proc.returncode


def run_duckdb_count(sql_text, timeout=SUBQUERY_TIMEOUT_SEC):
    """Run a SELECT COUNT(*) query via DuckDB CLI in CSV mode. Returns (count, error)."""
    try:
        proc = subprocess.run(
            [DUCKDB_BIN, DB_FILE, "-csv", "-c", sql_text],
            capture_output=True, text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    if proc.returncode != 0:
        return None, proc.stderr[:300]
    lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        try:
            return int(lines[-1]), None
        except ValueError:
            return None, f"Parse error: {lines}"
    return None, f"Unexpected output: {proc.stdout[:200]}"


def run_query_with_json_profile(query_sql):
    """Run a query with JSON profiling. Returns parsed profile dict or None."""
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
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("ERROR IN BASELINE:", proc.stderr)
        return None
    if not os.path.exists(PROFILE_OUTPUT):
        print("ERROR IN BASELINE: No profile output. Details:", proc.stdout[:1000], proc.stderr[:1000])
        return None
    with open(PROFILE_OUTPUT, "r") as f:
        profile = json.load(f)
    os.remove(PROFILE_OUTPUT)
    return profile


# ============================================================================
# LOG PARSING
# ============================================================================

def parse_cardinality_log():
    """
    Parse cardinality_log.txt. Returns list of dicts with expression, tables,
    filter_str, cardinality, is_injected, sql_count_query, num_tables,
    and estimation_detail (parsed from ESTIMATION_DETAIL lines).
    """
    if not os.path.exists(CARDINALITY_LOG):
        return []
    with open(CARDINALITY_LOG) as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or "LOGICAL_JOIN" not in line:
            if line.startswith("SQL_COUNT_QUERY:") and entries:
                entries[-1]["sql_count_query"] = line[len("SQL_COUNT_QUERY:"):].strip()
            elif line.startswith("SQL_COUNT_REASON:") and entries:
                entries[-1]["sql_count_reason"] = line[len("SQL_COUNT_REASON:"):].strip()
            elif line.startswith("SQL_COUNT_COVERAGE:") and entries:
                entries[-1]["sql_count_coverage"] = line[len("SQL_COUNT_COVERAGE:"):].strip()
            elif line.startswith("SQL_COUNT_INJECTABLE:") and entries:
                v = line[len("SQL_COUNT_INJECTABLE:"):].strip().lower()
                entries[-1]["sql_count_injectable"] = v == "yes"
            elif line.startswith("SQL_COUNT_INJECTABLE_REASON:") and entries:
                entries[-1]["sql_count_injectable_reason"] = line[
                    len("SQL_COUNT_INJECTABLE_REASON:"):
                ].strip()
            elif line.startswith("ESTIMATION_DETAIL:") and entries:
                entries[-1]["estimation_detail"] = _parse_estimation_detail(line)
            continue

        is_injected = "using INJECTED Cardinality:" in line
        if is_injected:
            expr_end = line.find(" using INJECTED Cardinality:")
            card_str = line[expr_end + len(" using INJECTED Cardinality:"):].strip()
        else:
            expr_end = line.find(" Estimated Cardinality:")
            if expr_end == -1:
                continue
            card_str = line[expr_end + len(" Estimated Cardinality:"):].strip()

        expression = line[:expr_end]

        rb_match = re.search(r"RelBindings: \[(.*?)\]", expression)
        tables = []
        if rb_match:
            raw = rb_match.group(1)
            for binding in raw.split(" | "):
                binding = binding.strip()
                if ":" in binding:
                    _, table = binding.split(":", 1)
                    table = table.strip().replace("\\|", "|").replace("\\\\", "\\")
                    tables.append(table)

        filters_match = re.search(r"Filters: \[(.*)\](?: CtxOcc: \d+)?$", expression)
        filter_str = filters_match.group(1) if filters_match else ""

        # Parse CtxInputCards
        ctx_match = re.search(r"CtxInputCards: \[([^\]]+)\]", expression)
        ctx_input_cards = {}
        if ctx_match:
            for part in ctx_match.group(1).split(" | "):
                if ":" in part:
                    rel_idx, card_val = part.split(":", 1)
                    try:
                        ctx_input_cards[int(rel_idx.strip())] = int(card_val.strip())
                    except ValueError:
                        pass

        entries.append({
            "expression": expression,
            "tables": tables,
            "filter_str": filter_str,
            "cardinality": float(card_str),
            "is_injected": is_injected,
            "sql_count_query": "",
            "sql_count_reason": "",
            "sql_count_coverage": "",
            "sql_count_injectable": False,
            "sql_count_injectable_reason": "",
            "num_tables": len(tables),
            "ctx_input_cards": ctx_input_cards,
            "estimation_detail": {},
        })

    return entries


def rel_set_key_from_expression(expr: str):
    """Sorted tuple of relation indices from ``RelSets: [0, 1, ...]``, or None."""
    m = re.search(r"RelSets:\s*\[([0-9,\s]+)\]", expr)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",") if p.strip().isdigit()]
    if not parts:
        return None
    return tuple(sorted(int(p) for p in parts))


def sql_normalize_for_ambiguity(sql: str) -> str:
    """Normalize SQL text so trivial formatting differences do not split a group."""
    return " ".join(sql.lower().split())


def ambiguous_rel_set_groups(log_entries):
    """
    Relation-set keys that appear with more than one distinct SQL_COUNT text
    (logged or synthesized) in the same parse pass — unsafe to pick one oracle.
    """
    by_key: dict[tuple[int, ...], set[str]] = {}
    for e in log_entries:
        if e.get("num_tables", 0) < 2:
            continue
        k = rel_set_key_from_expression(e.get("expression", ""))
        if k is None:
            continue
        sql = (e.get("sql_count_query") or "").strip()
        if not sql:
            sql = _synthesize_sql_count_from_log_entry(e).strip()
        if not sql:
            continue
        by_key.setdefault(k, set()).add(sql_normalize_for_ambiguity(sql))
    return {k for k, variants in by_key.items() if len(variants) > 1}


def _parse_estimation_detail(line):
    """Parse an ESTIMATION_DETAIL: line into a dict of components."""
    detail = {"edges": [], "scans": {}}
    # Numerator=N
    m = re.search(r"Numerator=([\d.]+)", line)
    if m:
        detail["numerator"] = float(m.group(1))
    # Denominator=D
    m = re.search(r"Denominator=([\d.]+)", line)
    if m:
        detail["denominator"] = float(m.group(1))
    # EDGE[i]: filter=... has_hll=T/F tdom_hll=N tdom_no_hll=N
    for edge_m in re.finditer(
        r"EDGE\[(\d+)\]: filter=(.*?) has_hll=([TF]) tdom_hll=(\d+) tdom_no_hll=(\d+)", line
    ):
        detail["edges"].append({
            "idx": int(edge_m.group(1)),
            "filter": edge_m.group(2),
            "has_hll": edge_m.group(3) == "T",
            "tdom_hll": int(edge_m.group(4)),
            "tdom_no_hll": int(edge_m.group(5)),
        })
    # SCAN[j]: table=... scan_filter=...
    for scan_m in re.finditer(r"SCAN\[(\d+)\]: table=(\S+) scan_filter=(\S+)", line):
        detail["scans"][int(scan_m.group(1))] = {
            "table": scan_m.group(2),
            "scan_filter": scan_m.group(3),
        }
    return detail


# ============================================================================
# PLAN PARSING
# ============================================================================

def get_plan_structure_text(node, depth=0):
    op_name = node.get("operator_name", node.get("name", "?"))
    extra = node.get("extra_info", {})
    indent = "  " * depth
    lines = [f"{indent}{op_name}"]
    STRUCTURAL_KEYS = {"Join Type", "Conditions", "Filters", "Table",
                       "Projections", "Limit", "Groups", "Aggregates",
                       "Join Condition"}
    for key in sorted(extra.keys()):
        if key in STRUCTURAL_KEYS:
            lines.append(f"{indent}  {key}: {extra[key]}")
    for child in node.get("children", []):
        lines.extend(get_plan_structure_text(child, depth + 1).splitlines())
    return "\n".join(lines)


def compute_plan_fingerprint(plan_text):
    return hashlib.sha1(plan_text.encode("utf-8")).hexdigest()[:12]


# ============================================================================
# CTE DUPLICATE DETECTION
# ============================================================================

def detect_cte_duplicates(log_entries):
    expr_counts = Counter(entry["expression"] for entry in log_entries)
    return {expr for expr, count in expr_counts.items() if count > 1}


# ============================================================================
# ORACLE CARDINALITY COMPUTATION
# ============================================================================

def _synthesize_sql_count_from_log_entry(entry: dict) -> str:
    """
    Build SELECT COUNT(*) ... from LOGICAL_JOIN metadata when C++ did not emit
    SQL_COUNT_QUERY (logs may only have LOGICAL_JOIN + cardinality).

    Comma-FROM uses base table names so WHERE matches filter_str from the
    planner. Skips duplicate physical table names, empty filters (avoids huge
    cross products), and non-catalog-like identifiers.
    """
    tbs = entry.get("tables") or []
    if len(tbs) < 2:
        return ""
    if len(set(tbs)) != len(tbs):
        return ""
    for tb in tbs:
        if not tb or tb == "[unknown]":
            return ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tb):
            return ""
    fs = (entry.get("filter_str") or "").strip()
    if not fs:
        return ""
    if ";" in fs or "--" in fs:
        return ""
    from_clause = ", ".join(tbs)
    return f"SELECT COUNT(*) FROM {from_clause} WHERE ({fs})"


def compute_oracle_cardinalities(
    query_nr,
    log_entries,
    cte_exprs,
    *,
    ambiguous_rel_sets=None,
    strict_relset_sql=True,
    permissive_synthesis=False,
):
    """
    For each multi-table log entry, execute the SQL_COUNT_QUERY to get the
    true cardinality.

    - CTE duplicates → skip.
    - Single-table entries → skip.
    - No SQL_COUNT_QUERY logged → only with ``permissive_synthesis=True`` try synthesized COUNT.
    - Subquery syntax error or projected-column failure → skip gracefully.
    - Subquery timeout (>120s) → skip, proceed with regular estimation.
    - strict_relset_sql: skip entries whose RelSets key is in ambiguous_rel_sets.
    - C++ must emit ``SQL_COUNT_INJECTABLE: yes`` (requires rebuilt ``shell``); otherwise skipped.
    - ``permissive_synthesis``: if True, allow Python-synthesized COUNT when C++ emitted no SQL.

    Returns:
        oracle_map: dict {expression: true_cardinality}
        failures: list of dicts
        skipped: list of (reason, expression)
    """
    if ambiguous_rel_sets is None:
        ambiguous_rel_sets = set()
    oracle_map = {}
    failures = []
    skipped = []
    total = sum(1 for e in log_entries if e["num_tables"] >= 2)
    done = 0

    for entry in log_entries:
        expr = entry["expression"]
        had_logged_sql = bool((entry.get("sql_count_query") or "").strip())
        sql = (entry.get("sql_count_query") or "").strip()

        # Skip CTE duplicates
        if expr in cte_exprs:
            skipped.append(("CTE", expr))
            continue

        # Skip single-table entries
        if entry["num_tables"] <= 1:
            skipped.append(("single-table", expr))
            continue

        rs_key = rel_set_key_from_expression(expr)
        if strict_relset_sql and rs_key is not None and rs_key in ambiguous_rel_sets:
            skipped.append(("ambiguous-relset-sql", expr))
            continue

        if not entry.get("sql_count_injectable", False):
            skipped.append(("not-injectable", expr))
            continue

        if not sql:
            if permissive_synthesis:
                sql = _synthesize_sql_count_from_log_entry(entry).strip()
            if not sql:
                skipped.append(("no-sql", expr))
                continue

        done += 1
        if done % 20 == 0:
            _log(f"    ... computed {done}/{total} subqueries")

        # Execute the subquery
        count, err = run_duckdb_count(sql)
        if count is not None:
            oracle_map[expr] = count
            # #region agent log
            try:
                _agent_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug-367dc7.log"
                )
                with open(_agent_path, "a") as _af:
                    _af.write(
                        json.dumps(
                            {
                                "sessionId": "367dc7",
                                "hypothesisId": "H1",
                                "runId": "oracle-synth",
                                "location": "oracle_benchmark.compute_oracle_cardinalities:success",
                                "message": "oracle_subquery_ok",
                                "data": {
                                    "query_nr": query_nr,
                                    "num_tables": entry.get("num_tables"),
                                    "tables": entry.get("tables"),
                                    "used_cpp_sql_count_query": had_logged_sql,
                                    "filter_str_head": (entry.get("filter_str") or "")[:400],
                                    "sql": sql[:1500],
                                    "count": int(count),
                                },
                                "timestamp": int(time.time() * 1000),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
        else:
            reason = "TIMEOUT" if err == "TIMEOUT" else "SQL_ERROR"
            skipped.append((reason, expr))
            failures.append({
                "expression": expr[:200],
                "error": err,
                "sql": sql[:200],
                "reason": reason,
            })

    return oracle_map, failures, skipped


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_oracle_injection(query_nr, log_entries, oracle_map, cte_exprs, strict_ambiguous_skipped=0):
    """
    After running the real query with oracle injections, verify and produce
    the comprehensive cardinality table.

    Checks:
      V1: For INJECTED lines,  injected value == oracle value.  ALARM on mismatch.
      V2: All oracle keys appeared in the real-run log.
      V3: Oracle keys that were NOT injected in the real run (plan divergence).
      V4: CTE expressions should not be injected.
      V5: Strict RelSets-SQL skips during oracle map build (informational).
    """
    log_expressions = {e["expression"] for e in log_entries}

    alarms = []  # collect all ALARM messages

    # V1 – injected value must equal oracle value
    v1_pass = 0
    v1_fail = []
    for entry in log_entries:
        if entry["is_injected"]:
            expr = entry["expression"]
            injected_val = int(entry["cardinality"])
            if expr in oracle_map:
                oracle_val = oracle_map[expr]
                if injected_val == oracle_val:
                    v1_pass += 1
                else:
                    v1_fail.append((expr, injected_val, oracle_val))

    if v1_fail:
        msg = f"Q{query_nr} V1 ALARM: {len(v1_fail)} injected != oracle"
        _log(f"    ⚠️  {msg}")
        for expr, inj, orc in v1_fail[:5]:
            _log(f"      ALARM: Injected={inj}, Oracle={orc}: {expr[:160]}")
        alarms.append(msg)
    else:
        _log(f"    V1 OK: {v1_pass} injected values match oracle.")

    # V2 – all oracle keys in log
    missing_keys = [e for e in oracle_map if e not in log_expressions]
    if missing_keys:
        _log(f"    V2 WARN: {len(missing_keys)} oracle keys NOT in log.")
    else:
        _log(f"    V2 OK: All {len(oracle_map)} oracle keys in log.")

    # V3 – oracle keys not injected (plan divergence)
    not_injected = [e for e in log_entries
                    if not e["is_injected"] and e["expression"] in oracle_map]
    if not_injected:
        _log(f"    V3 INFO: {len(not_injected)} oracle keys NOT injected (plan divergence).")
    else:
        _log(f"    V3 OK: All oracle keys were injected.")

    # V4 – CTE expressions should not have been injected
    cte_injected = [e for e in log_entries
                    if e["is_injected"] and e["expression"] in cte_exprs]
    if cte_injected:
        _log(f"    V4 WARN: {len(cte_injected)} CTE expressions injected.")
    else:
        _log(f"    V4 OK.")

    if strict_ambiguous_skipped > 0:
        _log(
            f"    V5 INFO: Strict RelSets-SQL skipped {strict_ambiguous_skipped} oracle key(s) "
            f"(ambiguous SQL_COUNT for same RelSets)."
        )
    else:
        _log(f"    V5 OK: No RelSets-SQL ambiguity skips.")

    return {
        "v1_pass": v1_pass,
        "v1_fail": len(v1_fail),
        "v2_missing": len(missing_keys),
        "v3_not_injected": len(not_injected),
        "v4_cte_injected": len(cte_injected),
        "v5_ambiguous_skipped": strict_ambiguous_skipped,
        "alarms": alarms,
    }


PROFILE_EA_SCAN_OPS = frozenset({"TABLE_SCAN", "SEQ_SCAN"})


def _profile_scan_table_basename(extra_info):
    """Base table name (lowercased) from a profile scan node's ``extra_info.Table``."""
    if not isinstance(extra_info, dict):
        return None
    t = extra_info.get("Table")
    if not t:
        return None
    s = str(t).strip()
    if not s:
        return None
    return s.split(".")[-1].lower()


def _collect_subtree_profile_scan_tables(node):
    """Set of base table names (lower) under ``node`` from TABLE_SCAN / SEQ_SCAN."""
    out = set()
    if not isinstance(node, dict):
        return out
    op = (node.get("operator_name") or node.get("operator_type") or "").upper()
    if op in PROFILE_EA_SCAN_OPS:
        b = _profile_scan_table_basename(node.get("extra_info") or {})
        if b:
            out.add(b)
    for ch in node.get("children") or []:
        out.update(_collect_subtree_profile_scan_tables(ch))
    return out


def build_injected_v6_candidates(oracle_log_entries):
    """
    One record per injected multi-table LOGICAL_JOIN line: leaf table multiset plus
    the **injected cardinality** (same value written to ``actual_cardinality.json``).

    V6 ties a physical join to at most one record by matching ``tables`` and
    requiring the profile ``Estimated Cardinality`` to align with ``inj_card``
    (injection is keyed by the full expression; this pairs the operator with the
    same numeric target the planner used).
    """
    out = []
    for e in oracle_log_entries or []:
        if not e.get("is_injected"):
            continue
        tbls = e.get("tables") or []
        if len(tbls) < 2:
            continue
        fs = frozenset(str(t).split(".")[-1].lower() for t in tbls)
        try:
            inj = int(e["cardinality"])
        except (TypeError, ValueError):
            try:
                inj = int(float(e["cardinality"]))
            except (TypeError, ValueError):
                continue
        out.append({"tables": fs, "inj_card": inj})
    return out


def _v6_match_injected_candidate(scan_tables, profile_est, candidates, inj_match_tol):
    """
    Pick the unique injected line that matches this physical join: same leaf
    table multiset, and profile estimated cardinality within ``inj_match_tol``
    (symmetric relative) of that line's injected cardinality.
    """
    if profile_est is None or not candidates:
        return None
    try:
        est_i = int(profile_est)
    except (TypeError, ValueError):
        return None
    same_t = [c for c in candidates if c["tables"] == scan_tables]
    if not same_t:
        return None

    def sym_rel(a, b):
        d = max(abs(int(a)), abs(int(b)), 1)
        return abs(int(a) - int(b)) / float(d)

    exact = [c for c in same_t if int(c["inj_card"]) == est_i]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None

    close = [c for c in same_t if sym_rel(est_i, c["inj_card"]) <= inj_match_tol]
    if len(close) == 1:
        return close[0]
    return None


def _injected_multi_table_join_count(oracle_log_entries):
    return sum(
        1
        for e in oracle_log_entries or []
        if e.get("is_injected") and len(e.get("tables") or []) >= 2
    )


def _duplicate_injected_table_multisets(oracle_log_entries):
    """How many distinct table multisets appear on more than one injected multi-table line."""
    counts = Counter()
    for e in oracle_log_entries or []:
        if not e.get("is_injected"):
            continue
        tbls = e.get("tables") or []
        if len(tbls) < 2:
            continue
        fs = frozenset(str(t).split(".")[-1].lower() for t in tbls)
        counts[fs] += 1
    return sum(1 for fs, n in counts.items() if n > 1)


def _parse_profile_estimated_cardinality(extra_info):
    """Parse extra_info['Estimated Cardinality'] from a JSON profile node."""
    if not isinstance(extra_info, dict):
        return None
    v = extra_info.get("Estimated Cardinality")
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(round(v))
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def verify_profile_est_vs_actual(
    query_nr,
    profile_root,
    rel_tolerance=PROFILE_EA_CARD_TOLERANCE,
    max_report=25,
    *,
    joins_only=True,
    oracle_log_entries=None,
):
    """
    Compare optimizer-estimated vs executed operator cardinalities on the JSON
    profile from the same run as ``run_query_with_json_profile`` (equivalent
    numbers to EXPLAIN (ANALYZE, FORMAT JSON)).

    When ``joins_only`` is True (default), only **physical join** operators in
    ``PROFILE_EA_JOIN_OPERATOR_NAMES`` whose subtree scan tables match a
    **multi-table injected** LOGICAL_JOIN (from ``oracle_log_entries``) are
    checked. Non-injected joins and all other operators are skipped.

    Duplicate table multisets across injected lines are disambiguated with the
    profile ``Estimated Cardinality`` (must match the injected value for that line,
    within ``rel_tolerance``). Injection itself remains keyed by the full logical
    expression in C++/JSON; this is only how V6 attaches profile nodes to those
    injections for the est-vs-actual check.

    When ``joins_only`` is False, every profile node with both estimates is
    checked (very noisy).

    Even injected-join matches can disagree when standalone COUNT scope differs
    from rows reaching that operator in the pipeline.

    Uses symmetric relative error:
        |est - actual| / max(|est|, |actual|, 1) > rel_tolerance
    """
    mismatches = []
    v6_candidates = (
        build_injected_v6_candidates(oracle_log_entries)
        if joins_only
        else []
    )
    if joins_only:
        n_inj_mt = _injected_multi_table_join_count(oracle_log_entries or [])
        if n_inj_mt == 0:
            _log(
                "    V6: no injected multi-table joins in log — skipping est vs actual "
                "(nothing to match against physical joins)."
            )
            return {
                "v6_mismatch": 0,
                "v6_mismatch_samples": [],
                "alarms": [],
            }
        if not v6_candidates:
            _log(
                "    V6: injected multi-table join(s) present but no V6 candidates could be "
                "built (unexpected); skipping profile est vs actual."
            )
            return {
                "v6_mismatch": 0,
                "v6_mismatch_samples": [],
                "alarms": [],
            }
        ndup = _duplicate_injected_table_multisets(oracle_log_entries or [])
        if ndup > 0:
            _log(
                f"    V6: {n_inj_mt} injected multi-table join(s); {ndup} table multiset(s) appear "
                f"on more than one line — disambiguating with (tables + est≈injected_cardinality, "
                f"tol={rel_tolerance * 100:.2f}% rel)."
            )

    def walk(node, depth):
        if not isinstance(node, dict):
            return
        op = node.get("operator_name") or node.get("operator_type") or ""
        extra = node.get("extra_info") or {}
        est = _parse_profile_estimated_cardinality(extra)
        actual_raw = node.get("operator_cardinality")
        if actual_raw is None:
            actual_raw = node.get("cardinality")
        scan_tables = frozenset()
        matched = None
        if joins_only:
            if op not in PROFILE_EA_JOIN_OPERATOR_NAMES:
                consider = False
            else:
                scan_tables = frozenset(_collect_subtree_profile_scan_tables(node))
                if len(scan_tables) >= 2 and est is not None:
                    matched = _v6_match_injected_candidate(
                        scan_tables, est, v6_candidates, rel_tolerance
                    )
                consider = matched is not None
        else:
            consider = True
        if consider and est is not None and actual_raw is not None:
            try:
                actual = int(actual_raw)
            except (TypeError, ValueError):
                try:
                    actual = int(float(actual_raw))
                except (TypeError, ValueError):
                    actual = None
            if actual is not None:
                denom = max(abs(est), abs(actual), 1)
                rel_err = abs(est - actual) / float(denom)
                if rel_err > rel_tolerance:
                    rec = {
                        "operator": op,
                        "est": est,
                        "actual": actual,
                        "rel_err": rel_err,
                        "depth": depth,
                    }
                    if joins_only and scan_tables:
                        rec["scan_tables"] = sorted(scan_tables)
                    mismatches.append(rec)
        for ch in node.get("children") or []:
            walk(ch, depth + 1)

    walk(profile_root, 0)
    mismatches.sort(key=lambda x: -x["rel_err"])
    alarms = []
    if mismatches:
        pct = rel_tolerance * 100.0
        scope = "injected-join operators" if joins_only else "all operators"
        msg = (
            f"Q{query_nr} V6 ALARM: {len(mismatches)} {scope} est vs actual "
            f"> {pct:.2f}% symmetric relative error (JSON execution profile)"
        )
        alarms.append(msg)
        _log(f"    ⚠️  {msg}")
        for m in mismatches[:max_report]:
            tbl = ""
            if m.get("scan_tables"):
                tbl = f" tables={m['scan_tables']}"
            _log(
                f"      V6: {m['operator']}{tbl} est={m['est']:,} actual={m['actual']:,} "
                f"rel_err={m['rel_err'] * 100.0:.2f}%"
            )
        if len(mismatches) > max_report:
            _log(f"      ... and {len(mismatches) - max_report} more (see alarms list)")
    else:
        if joins_only:
            scope = "injected-join profile nodes"
        else:
            scope = "profile operators"
        _log(
            f"    V6 OK: all {scope} within {rel_tolerance * 100.0:.2f}% "
            f"est vs actual (where both present)."
        )

    return {
        "v6_mismatch": len(mismatches),
        "v6_mismatch_samples": mismatches[:max_report],
        "alarms": alarms,
    }


# ============================================================================
# CARDINALITY TABLE
# ============================================================================

def log_cardinality_table(query_nr, explain_entries, oracle_map, oracle_log_entries):
    """
    Log a comprehensive table showing every join's estimated vs oracle
    cardinality, whether it was injected, and the tdom/numerator breakdown
    so we can immediately diagnose estimation errors.
    """
    _log(f"\n  {'Tables':>50} {'Estimated':>12} {'Oracle':>12} {'Ratio':>8} "
         f"{'Injected':>10} {'tdom_hll':>12} {'Input_L':>12} {'Input_R':>12}")
    _log(f"  {'-'*130}")

    injected_set = {e["expression"] for e in oracle_log_entries if e["is_injected"]}

    for entry in explain_entries:
        if entry["num_tables"] < 2:
            continue
        expr = entry["expression"]
        tbl_str = ",".join(entry["tables"])
        if len(tbl_str) > 48:
            tbl_str = tbl_str[:45] + "..."
        est = int(entry["cardinality"])

        # tdom and input card from ESTIMATION_DETAIL
        detail = entry.get("estimation_detail", {})
        edge_tdom = ""
        if detail.get("edges"):
            edge = detail["edges"][0]  # primary join edge
            edge_tdom = f"{edge['tdom_hll']:,}" if edge["has_hll"] else f"~{edge['tdom_no_hll']:,}"
        ctx = entry.get("ctx_input_cards", {})
        input_l = list(ctx.values())[0] if ctx else ""
        input_r = list(ctx.values())[1] if len(ctx) > 1 else ""
        input_l_str = f"{input_l:,}" if isinstance(input_l, int) else "-"
        input_r_str = f"{input_r:,}" if isinstance(input_r, int) else "-"

        if expr in oracle_map:
            orc = oracle_map[expr]
            ratio = orc / max(est, 1)
            was_injected = "✅ YES" if expr in injected_set else "⬜ NO"
            # ALARM if ratio is very large — likely a tdom or scan-filter issue
            alarm = " ⚠️" if ratio > 20 or ratio < 0.05 else ""
            _log(f"  {tbl_str:>50} {est:>12,} {orc:>12,} {ratio:>8.2f} "
                 f"{was_injected:>10} {edge_tdom:>12} {input_l_str:>12} {input_r_str:>12}{alarm}")
        else:
            _log(f"  {tbl_str:>50} {est:>12,} {'(skipped)':>12} {'—':>8} "
                 f"{'⬜ skip':>10} {edge_tdom:>12} {input_l_str:>12} {input_r_str:>12}")


# ============================================================================
# MAIN LOOP PER QUERY
# ============================================================================

def run_single_query(
    query_nr,
    query_sql,
    baseline_profile=None,
    skip_baseline_phase=False,
    *,
    profile_ea_tolerance=PROFILE_EA_CARD_TOLERANCE,
    skip_profile_ea=False,
    profile_ea_joins_only=True,
    oracle_permissive_synthesis=False,
):
    """
    Oracle injection for one TPC-DS query.
    Returns a result dict.

    When ``skip_baseline_phase`` is True, ``baseline_profile`` must be the JSON
    profile dict from a prior vanilla run (same query); Phase 0 does not re-run
    the query. Used by ``unified_benchmark`` to avoid a duplicate baseline
    profile execution.

    ``profile_ea_tolerance``: V6 symmetric relative error threshold for
    estimated vs executed operator cardinalities on the injected run's JSON profile.
    ``skip_profile_ea``: if True, skip V6 entirely. Default False (V6 on).
    ``profile_ea_joins_only``: if True (default), V6 only checks physical joins
    whose leaf scans match a **multi-table injected** LOGICAL_JOIN from the
    post-run cardinality log (non-injected joins are ignored). If False, V6 uses
    every profile node with estimates (``--profile-ea-all-operators``).
    ``oracle_permissive_synthesis``: if True, allow Python COUNT synthesis when C++
    emitted no ``SQL_COUNT_QUERY`` (legacy; default False — inject only C++-certified keys).
    """

    _log(f"\n{'='*60}")
    _log(f"  Query {query_nr}")
    _log(f"{'='*60}")

    # ---- Phase 0: Baseline (no injection) ----
    if skip_baseline_phase:
        if baseline_profile is None:
            _log(f"  [ERROR] skip_baseline_phase=True but baseline_profile is missing for Q{query_nr}.")
            return {"error": True, "error_msg": "baseline_profile required when skip_baseline_phase"}
        _log(f"  Phase 0: Using caller-provided baseline profile (no re-run)")
        prof = baseline_profile
    else:
        _log(f"  Phase 0: Baseline run (no injection)")
        clear_actual_cardinality_json()
        # clear_cardinality_log()
        prof = run_query_with_json_profile(query_sql)
        if prof is None:
            _log(f"  [ERROR] Baseline run failed for Q{query_nr}.")
            return {"error": True, "error_msg": "baseline failed"}
    baseline_root = (prof.get("children", [prof])[0]
                     if prof.get("children") else prof)
    baseline_plan_text = get_plan_structure_text(baseline_root)
    baseline_fingerprint = compute_plan_fingerprint(baseline_plan_text)
    _log(f"  Baseline plan fingerprint: {baseline_fingerprint}")

    # ---- Phase 1: EXPLAIN (dry-run) ----
    _log(f"  Phase 1: EXPLAIN (dry-run)")
    if ORACLE_STRICT_RELSET_SQL:
        clear_cardinality_log()
        _log(f"  (cleared cardinality_log for strict RelSets-SQL pass isolation)")
    explain_sql = f"EXPLAIN {query_sql}"
    _, stderr, rc = run_duckdb_sql(explain_sql)
    if rc != 0:
        _log(f"  [ERROR] EXPLAIN failed: {stderr[:300]}")
        return {"error": True, "error_msg": "explain failed"}

    explain_entries = parse_cardinality_log()
    multi_table = sum(1 for e in explain_entries if e["num_tables"] >= 2)
    with_sql = sum(1 for e in explain_entries if e.get("sql_count_query"))
    noninjectable_entries = [
        e for e in explain_entries if e.get("num_tables", 0) >= 2 and not e.get("sql_count_query")
    ]
    injectable_entries = [
        e for e in explain_entries if e.get("num_tables", 0) >= 2 and e.get("sql_count_query")
    ]
    noninjectable_frac = (len(noninjectable_entries) / float(max(1, multi_table)))
    max_noninjectable_est = max((float(e.get("cardinality", 0.0)) for e in noninjectable_entries), default=0.0)
    max_injectable_est = max((float(e.get("cardinality", 0.0)) for e in injectable_entries), default=0.0)
    # #region agent log
    try:
        reason_counter = Counter(
            (e.get("sql_count_reason") or "missing_reason")
            for e in explain_entries
            if e.get("num_tables", 0) >= 2
        )
        coverage_zero_sql = sum(
            1
            for e in explain_entries
            if e.get("num_tables", 0) >= 2
            and e.get("sql_count_coverage")
            and "sql_conjuncts=0" in e.get("sql_count_coverage", "")
        )
        _agent_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug-367dc7.log")
        with open(_agent_log_path, "a") as _agent_f:
            _agent_f.write(
                json.dumps(
                    {
                        "sessionId": "367dc7",
                        "hypothesisId": "H2",
                        "runId": "q4-regression",
                        "location": "oracle_benchmark.run_single_query:post_explain",
                        "message": "sql_count_generation_summary",
                        "data": {
                            "query_nr": query_nr,
                            "multi_table": multi_table,
                            "with_sql": with_sql,
                            "coverage_zero_sql": coverage_zero_sql,
                            "reason_counts": dict(reason_counter),
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    # #region agent log
    try:
        skipped_by_reason = [
            e for e in explain_entries
            if e.get("num_tables", 0) >= 2 and not e.get("sql_count_query")
        ]
        skipped_by_reason.sort(key=lambda e: float(e.get("cardinality", 0.0)), reverse=True)
        top_skipped = []
        for e in skipped_by_reason[:8]:
            top_skipped.append({
                "tables": e.get("tables", [])[:6],
                "estimated": int(float(e.get("cardinality", 0.0))),
                "reason": e.get("sql_count_reason", ""),
                "coverage": e.get("sql_count_coverage", "")[:200],
            })
        _agent_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug-367dc7.log")
        with open(_agent_log_path, "a") as _agent_f:
            _agent_f.write(
                json.dumps(
                    {
                        "sessionId": "367dc7",
                        "hypothesisId": "H4",
                        "runId": "q4-regression",
                        "location": "oracle_benchmark.run_single_query:skipped_top",
                        "message": "top_noninjectable_joins",
                        "data": {
                            "query_nr": query_nr,
                            "count_noninjectable": len(skipped_by_reason),
                            "top_skipped": top_skipped,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    _log(f"  {len(explain_entries)} log entries, {multi_table} multi-table joins, "
         f"{with_sql} with SQL_COUNT_QUERY.")

    if multi_table == 0:
        _log(f"  No multi-table joins → nothing to inject. Skipping.")
        return {
            "error": False,
            "n_log_entries": len(explain_entries),
            "n_oracle": 0, "n_failures": 0, "n_skipped": 0,
            "n_skipped_ambiguous_relset_sql": 0,
            "n_timeouts": 0, "n_sql_errors": 0,
            "n_injected": 0,
            "plan_changed": False,
            "baseline_fingerprint": baseline_fingerprint,
            "oracle_fingerprint": baseline_fingerprint,
            "verify": {},
            "alarms": [],
        }

    # ---- Phase 2: Detect CTE duplicates ----
    cte_exprs = detect_cte_duplicates(explain_entries)
    if cte_exprs:
        _log(f"  {len(cte_exprs)} CTE/duplicate expression(s) detected.")

    ambiguous_rel_sets = (
        ambiguous_rel_set_groups(explain_entries) if ORACLE_STRICT_RELSET_SQL else set()
    )
    if ORACLE_STRICT_RELSET_SQL and ambiguous_rel_sets:
        _log(
            f"  Strict RelSets-SQL: {len(ambiguous_rel_sets)} relation-set(s) have multiple "
            f"SQL_COUNT variants — oracle keys for those sets will be skipped "
            f"(use --oracle-permissive-context to inject all)."
        )

    # ---- Phase 3: Compute oracle cardinalities ----
    _log(f"  Phase 2: Computing oracle cardinalities ({multi_table} subqueries)...")
    t0 = time.time()
    oracle_map, failures, skipped = compute_oracle_cardinalities(
        query_nr,
        explain_entries,
        cte_exprs,
        ambiguous_rel_sets=ambiguous_rel_sets,
        strict_relset_sql=ORACLE_STRICT_RELSET_SQL,
        permissive_synthesis=oracle_permissive_synthesis,
    )
    elapsed = time.time() - t0
    n_timeouts = sum(1 for f in failures if f.get("reason") == "TIMEOUT")
    n_sql_errors = sum(1 for f in failures if f.get("reason") == "SQL_ERROR")
    n_skipped_ambiguous = sum(1 for r, _ in skipped if r == "ambiguous-relset-sql")
    _log(f"  Computed {len(oracle_map)} oracle cardinalities in {elapsed:.1f}s.")
    _log(f"  Skipped: {len(skipped)} total "
         f"(timeouts={n_timeouts}, sql_errors={n_sql_errors}, "
         f"cte={sum(1 for r,_ in skipped if r=='CTE')}, "
         f"single={sum(1 for r,_ in skipped if r=='single-table')}, "
         f"no_sql={sum(1 for r,_ in skipped if r=='no-sql')}, "
         f"not_injectable={sum(1 for r,_ in skipped if r=='not-injectable')}, "
         f"ambiguous_relset_sql={n_skipped_ambiguous})")
    partial_injection_danger = (
        multi_table > 0
        and noninjectable_frac >= 0.50
        and max_noninjectable_est > max(1.0, max_injectable_est) * 100.0
    )
    # #region agent log
    try:
        _agent_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug-367dc7.log")
        with open(_agent_log_path, "a") as _agent_f:
            _agent_f.write(
                json.dumps(
                    {
                        "sessionId": "367dc7",
                        "hypothesisId": "H5",
                        "runId": "q4-regression",
                        "location": "oracle_benchmark.run_single_query:danger_gate",
                        "message": "partial_injection_safety_gate",
                        "data": {
                            "query_nr": query_nr,
                            "multi_table": multi_table,
                            "noninjectable_frac": noninjectable_frac,
                            "max_noninjectable_est": max_noninjectable_est,
                            "max_injectable_est": max_injectable_est,
                            "gate_triggered": partial_injection_danger,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    if partial_injection_danger:
        _log(
            "  [NOTE] High partial-injection risk detected, but proceeding with oracle injection "
            "as requested."
        )
    if with_sql == 0 and len(oracle_map) > 0:
        _log(
            "  [NOTE] No SQL_COUNT_QUERY in cardinality_log — oracle used Python-synthesized "
            "SELECT COUNT(*) over catalog tables + join filters only (not the full TPC-DS query "
            "predicate). Ratios ≫ 1 vs Estimated are common; equal Oracle counts across joins can "
            "occur when an added table is functionally determined (e.g. item via ss_item_sk)."
        )

    if failures and n_sql_errors > 0:
        _log(f"  SQL errors (first 3):")
        for f in [x for x in failures if x["reason"] == "SQL_ERROR"][:3]:
            _log(f"    {f['error'][:120]}")
    if n_timeouts > 0:
        _log(f"  {n_timeouts} subqueries timed out (>{SUBQUERY_TIMEOUT_SEC}s) → using regular estimation.")

    # ---- Phase 4: Write oracle map to JSON ----
    _log(f"  Phase 3: Injecting {len(oracle_map)} oracle cardinalities.")
    clear_actual_cardinality_json()
    if oracle_map:
        write_actual_cardinality_json({k: float(v) for k, v in oracle_map.items()})

    # ---- Phase 5: Run real query with oracle injection ----
    _log(f"  Phase 4: Running query with oracle injection")
    # clear_cardinality_log()
    oracle_profile = run_query_with_json_profile(query_sql)
    if oracle_profile is None:
        _log(f"  [ERROR] Oracle-injected run failed for Q{query_nr}.")
        return {"error": True, "error_msg": "oracle run failed"}
    oracle_root = (oracle_profile.get("children", [oracle_profile])[0]
                   if oracle_profile.get("children") else oracle_profile)
    oracle_plan_text = get_plan_structure_text(oracle_root)
    oracle_fingerprint = compute_plan_fingerprint(oracle_plan_text)

    # ---- Phase 6: Verify ----
    oracle_log_entries = parse_cardinality_log()
    injected_count = sum(1 for e in oracle_log_entries if e["is_injected"])
    estimated_count = sum(1 for e in oracle_log_entries if not e["is_injected"])
    _log(f"  Phase 5: Verification ({injected_count} injected, {estimated_count} non-injected)")

    verify_results = verify_oracle_injection(
        query_nr,
        oracle_log_entries,
        oracle_map,
        cte_exprs,
        strict_ambiguous_skipped=n_skipped_ambiguous,
    )

    if skip_profile_ea:
        verify_results["v6_mismatch"] = 0
        _log(
            "  Phase 5b: V6 profile est vs actual — skipped "
            "(skip_profile_ea=True / use --disable-profile-ea from CLI)"
        )
    else:
        scope = (
            "injected multi-table joins only (matched by leaf scan tables)"
            if profile_ea_joins_only
            else "all operators"
        )
        _log(
            f"  Phase 5b: V6 profile est vs actual ({scope}, "
            f"symmetric rel err > {profile_ea_tolerance * 100.0:.2f}% → ALARM)"
        )
        ea = verify_profile_est_vs_actual(
            query_nr,
            oracle_root,
            rel_tolerance=profile_ea_tolerance,
            joins_only=profile_ea_joins_only,
            oracle_log_entries=oracle_log_entries,
        )
        verify_results["v6_mismatch"] = ea["v6_mismatch"]
        verify_results["alarms"] = verify_results.get("alarms", []) + ea["alarms"]

    # ---- Cardinality table ----
    log_cardinality_table(query_nr, explain_entries, oracle_map, oracle_log_entries)

    # ---- Phase 7: Plan comparison ----
    plan_changed = baseline_plan_text != oracle_plan_text
    _log(f"\n  Plan: baseline={baseline_fingerprint}  oracle={oracle_fingerprint}  "
         f"changed={'YES' if plan_changed else 'NO'}")
    # #region agent log
    try:
        explain_card_map = {e["expression"]: int(e["cardinality"]) for e in explain_entries}
        ratio_samples = []
        for k, v in oracle_map.items():
            if k in explain_card_map:
                est = max(1, explain_card_map[k])
                ratio_samples.append(float(v) / float(est))
        ratio_min = min(ratio_samples) if ratio_samples else None
        ratio_max = max(ratio_samples) if ratio_samples else None
        ratio_avg = (sum(ratio_samples) / len(ratio_samples)) if ratio_samples else None
        _agent_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor", "debug-367dc7.log")
        with open(_agent_log_path, "a") as _agent_f:
            _agent_f.write(
                json.dumps(
                    {
                        "sessionId": "367dc7",
                        "hypothesisId": "H3",
                        "runId": "q4-regression",
                        "location": "oracle_benchmark.run_single_query:plan_compare",
                        "message": "oracle_ratio_summary",
                        "data": {
                            "query_nr": query_nr,
                            "plan_changed": plan_changed,
                            "n_oracle": len(oracle_map),
                            "ratio_min": ratio_min,
                            "ratio_max": ratio_max,
                            "ratio_avg": ratio_avg,
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    fastest_diff = 0.0
    mean_diff = 0.0
    if plan_changed:
        _log(f"\n  Phase 6: Verifying Performance")
        _log(f"  Measuring baseline plan... (5 runs)")
        clear_actual_cardinality_json()
        bg_times = []
        for _ in range(5):
            t0 = time.time()
            run_duckdb_sql(query_sql)
            bg_times.append(time.time() - t0)
        bg_fastest = min(bg_times)
        bg_mean = sum(bg_times) / 5

        _log(f"  Measuring oracle plan... (5 runs)")
        if oracle_map:
            write_actual_cardinality_json({k: float(v) for k, v in oracle_map.items()})
        or_times = []
        for _ in range(5):
            t0 = time.time()
            run_duckdb_sql(query_sql)
            or_times.append(time.time() - t0)
        or_fastest = min(or_times)
        or_mean = sum(or_times) / 5

        _log(f"  Baseline: fastest={bg_fastest:.3f}s, mean={bg_mean:.3f}s")
        _log(f"  Oracle:   fastest={or_fastest:.3f}s, mean={or_mean:.3f}s")

        fastest_diff = or_fastest - bg_fastest
        mean_diff = or_mean - bg_mean
        
        # Check if the regression exceeds a small threshold to avoid floating point noise on extremely fast queries
        if fastest_diff > 0.01 or mean_diff > 0.01:
            _log(f"  [REGRESSION] Oracle plan is slower! Fastest diff: {fastest_diff:+.3f}s, Mean diff: {mean_diff:+.3f}s ❌")
        else:
            _log(f"  [IMPROVEMENT] Oracle plan is faster (or equal)! Fastest diff: {fastest_diff:+.3f}s, Mean diff: {mean_diff:+.3f}s ✅")

    return {
        "error": False,
        "n_log_entries": len(explain_entries),
        "n_oracle": len(oracle_map),
        "n_failures": len(failures),
        "n_skipped": len(skipped),
        "n_skipped_ambiguous_relset_sql": n_skipped_ambiguous,
        "n_timeouts": n_timeouts,
        "n_sql_errors": n_sql_errors,
        "n_injected": injected_count,
        "plan_changed": plan_changed,
        "baseline_fingerprint": baseline_fingerprint,
        "oracle_fingerprint": oracle_fingerprint,
        "verify": verify_results,
        "alarms": verify_results.get("alarms", []),
        # For debugging / tooling: full oracle key map used on the injected run.
        "oracle_map": dict(oracle_map),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    global ORACLE_STRICT_RELSET_SQL, TARGET_QUERIES
    import argparse

    parser = argparse.ArgumentParser(description="Oracle cardinality injection benchmark")
    parser.add_argument(
        "--oracle-permissive-context",
        action="store_true",
        help="Inject all oracle keys even when the same RelSets has multiple SQL_COUNT variants.",
    )
    parser.add_argument(
        "--enable-profile-ea",
        "--enable-prof",
        action="store_true",
        help="No-op (kept for scripts): V6 is enabled by default.",
    )
    parser.add_argument(
        "--skip-profile-ea",
        "--disable-profile-ea",
        "--disable-prof",
        action="store_true",
        dest="disable_profile_ea",
        help="Disable V6 profile est vs actual check (V6 is on by default).",
    )
    parser.add_argument(
        "--profile-ea-tolerance",
        type=float,
        default=PROFILE_EA_CARD_TOLERANCE,
        metavar="REL",
        help=(
            "V6 threshold: symmetric relative error |est-actual|/max(|est|,|actual|,1). "
            f"Default {PROFILE_EA_CARD_TOLERANCE} (1%%)."
        ),
    )
    parser.add_argument(
        "--profile-ea-all-operators",
        action="store_true",
        help=(
            "V6: compare estimated vs actual on every profile node that has both fields "
            "(very noisy). Default: only physical joins matching an injected multi-table "
            "join (leaf scan table set equals RelBindings from an injected log line)."
        ),
    )
    parser.add_argument(
        "--oracle-permissive-synthesis",
        action="store_true",
        help=(
            "Allow Python-synthesized COUNT(*) when C++ emitted no SQL_COUNT_QUERY. "
            "Default: only keys with SQL_COUNT_INJECTABLE: yes from rebuilt DuckDB."
        ),
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        metavar="N,M,...",
        help="Comma-separated TPC-DS query numbers to run (default: module TARGET_QUERIES).",
    )
    args, unknown_argv = parser.parse_known_args()
    ORACLE_STRICT_RELSET_SQL = not args.oracle_permissive_context
    skip_profile_ea_cli = bool(getattr(args, "disable_profile_ea", False))
    if args.queries:
        parsed_q = []
        for part in args.queries.split(","):
            part = part.strip()
            if part.isdigit():
                parsed_q.append(int(part))
        if parsed_q:
            TARGET_QUERIES = parsed_q

    init_log()
    if unknown_argv:
        _log(f"  [NOTE] Ignored unknown CLI args: {unknown_argv}")
    _log(f"  RelSets-SQL strict mode: {'ON' if ORACLE_STRICT_RELSET_SQL else 'OFF (permissive)'}")
    if skip_profile_ea_cli:
        _log("  V6 profile est vs actual: OFF (--disable-profile-ea / --skip-profile-ea)")
    else:
        v6_scope = (
            "injected multi-table joins (profile ∩ log)"
            if not args.profile_ea_all_operators
            else "all operators"
        )
        _log(
            f"  V6 profile est vs actual: ON ({v6_scope}, "
            f"tolerance={args.profile_ea_tolerance * 100.0:.4f}% symmetric rel err)"
        )
        _log(
            "  [NOTE] V6 can false-alarm: oracle COUNT is for an isolated subquery, while "
            "actual rows at a HASH_JOIN follow the executed pipeline."
        )

    assert os.path.exists(DUCKDB_BIN), f"DuckDB binary not found: {DUCKDB_BIN}"
    assert os.path.exists(DB_FILE), f"Database not found: {DB_FILE}"

    _log("\nExtracting TPC-DS queries...")
    queries = extract_tpcds_queries(TARGET_QUERIES)
    assert len(queries) > 0, "No queries extracted!"
    _log(f"  Extracted {len(queries)} queries.\n")

    results = {}
    all_alarms = []

    for query_nr in TARGET_QUERIES:
        if query_nr not in queries:
            _log(f"\n  [SKIP] Query {query_nr} not found in TPC-DS.")
            continue
        result = run_single_query(
            query_nr,
            queries[query_nr],
            profile_ea_tolerance=args.profile_ea_tolerance,
            skip_profile_ea=skip_profile_ea_cli,
            profile_ea_joins_only=not args.profile_ea_all_operators,
            oracle_permissive_synthesis=args.oracle_permissive_synthesis,
        )
        results[query_nr] = result

        # Collect alarms
        if result.get("alarms"):
            all_alarms.extend(result["alarms"])

        # Clean up
        clear_actual_cardinality_json()
        # clear_cardinality_log()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    _log("\n" + "=" * 110)
    _log("  FINAL SUMMARY")
    _log("=" * 110)
    _log(f"  {'Q':>4} {'Joins':>6} {'Oracle':>7} {'Fail':>5} {'T/O':>4} "
         f"{'Injected':>9} {'PlanChg':>8} "
         f"{'V1 (inj=orc)':>13} {'V3 (miss)':>10} {'V6 injΔ':>9}")
    _log(f"  {'-'*88}")

    total_oracle = 0
    total_injected = 0
    total_plan_changed = 0
    total_errors = 0
    total_alarms = 0
    total_v6_ops = 0

    for query_nr, result in sorted(results.items()):
        if result.get("error"):
            _log(f"  Q{query_nr:>3}  ERROR: {result.get('error_msg', '?')}")
            total_errors += 1
            continue
        v = result.get("verify", {})
        n_join = sum(1 for _ in range(result['n_log_entries']))  # approximate
        v1_total = v.get('v1_pass', 0) + v.get('v1_fail', 0)
        v6n = v.get("v6_mismatch", 0)
        alarm_marker = " ⚠️" if (v.get('v1_fail', 0) > 0 or v6n > 0) else ""
        _log(
            f"  Q{query_nr:>3} "
            f"{result.get('n_log_entries', 0):>6} "
            f"{result['n_oracle']:>7} "
            f"{result['n_failures']:>5} "
            f"{result.get('n_timeouts', 0):>4} "
            f"{result['n_injected']:>9} "
            f"{'YES' if result['plan_changed'] else 'no':>8} "
            f"{v.get('v1_pass', 0)}/{v1_total:>5} "
            f"{v.get('v3_not_injected', 0):>10} "
            f"{v6n:>9}"
            f"{alarm_marker}"
        )
        total_oracle += result["n_oracle"]
        total_injected += result["n_injected"]
        if result["plan_changed"]:
            total_plan_changed += 1
        if v.get("v1_fail", 0) > 0:
            total_alarms += v["v1_fail"]
        total_v6_ops += v6n

    _log(f"\n  {'-'*78}")
    _log(f"  Queries run:    {len(results)}")
    _log(f"  Errors:         {total_errors}")
    _log(f"  Oracle values:  {total_oracle}")
    _log(f"  Injected:       {total_injected}")
    _log(f"  Plans changed:  {total_plan_changed}")
    _log(f"  V1 ALARMs:      {total_alarms}")
    if skip_profile_ea_cli:
        _log(f"  V6 mismatches:  {total_v6_ops} (V6 disabled)")
    else:
        v6_scope = (
            "all profile operators"
            if args.profile_ea_all_operators
            else "injected-join profile nodes (table-set matched)"
        )
        _log(f"  V6 mismatches:  {total_v6_ops} ({v6_scope}, est≠actual beyond tolerance)")

    if all_alarms:
        _log(f"\n  {'='*40}")
        _log(f"  ⚠️  ALARMS  ⚠️")
        _log(f"  {'='*40}")
        for alarm in all_alarms:
            _log(f"  ⚠️  {alarm}")
    else:
        if skip_profile_ea_cli:
            _log(f"\n  ✅ No alarms. V1 injected=oracle OK (V6 was disabled).")
        else:
            _log(
                f"\n  ✅ No alarms. V1 injected=oracle OK; V6 injected-join est vs actual within tolerance."
            )

    _log(f"\n  Full log saved to: {RESULTS_LOG}")
    _log("=" * 110)
    _log("  Benchmark complete.")
    _log(f"  Finished: {datetime.datetime.now().isoformat()}")

    close_log()


if __name__ == "__main__":
    main()
