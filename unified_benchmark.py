"""
Unified cardinality benchmark: per TPC-DS query, run
  (1) vanilla DuckDB (JSON profile, no injection),
  (2) feedback loop until convergence / oscillation / cap,
  (3) oracle cardinality injection (same verifications as oracle_benchmark).

Reuses oracle_benchmark.py and feedback_benchmark.py; configure paths once via
apply_benchmark_config() before importing-driven globals are used.

Usage:
  python unified_benchmark.py --db /path/to/tpcds_sf10.db --queries 1,2,3
  python unified_benchmark.py --db /path/to/tpcds_sf10.db   # all queries 0-99
  python unified_benchmark.py --duckdb build/release/duckdb \\
      --duckdb-feedback build/release/duckdb_feedback ...

Build ``duckdb_feedback`` with ``-DBUILD_FEEDBACK_LEGACY_SHELL=ON`` (default in this fork).

Writes ``<repo>/unified_benchmark_results.csv`` (``pandas.read_csv``): compact
boolean cells (``vanilla_ok`` / errors / ``fb_injects_gt_oracle`` only when
non-default), ``n_inj_feedback`` vs ``n_inj_oracle`` injection line counts, and
``plan_relation`` (fingerprints omitted from CSV; see ``unified_benchmark_summary.txt``
for optional fp columns). Override paths with ``--output-csv`` and ``--output-summary``.

Requires pandas. On Homebrew Python (externally managed), install with:
  python3 -m pip install --break-system-packages pandas
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))

# Default artifact paths (under --repo). Load results with: pandas.read_csv(path)
DEFAULT_OUTPUT_CSV = "unified_benchmark_results.csv"
DEFAULT_OUTPUT_SUMMARY_TXT = "unified_benchmark_summary.txt"

# Columns written to CSV (compact booleans; no per-run fingerprints — use plan_relation).
# Fingerprints remain in the in-memory row dict / DataFrame for verbose printing only.
RESULT_CSV_COLUMNS = [
    "query_nr",
    "plan_relation",
    "vanilla_ok",
    "feedback_iterations",
    "feedback_unique_plans",
    "feedback_converged",
    "feedback_oscillation",
    "feedback_error",
    "n_inj_feedback",
    "n_inj_oracle",
    "n_oracle_keys",
    "fb_injects_gt_oracle",
    "oracle_error",
    "n_oracle_alarms",
]

# Internal columns kept on the DataFrame for tooling / console (not written to CSV).
INTERNAL_FP_COLUMNS = ("fp_vanilla", "fp_feedback", "fp_oracle")

DEFAULT_DB = "/Users/Aarry/Desktop/15689/tpcds_sf200.db"


def apply_benchmark_config(
    repo_root: str,
    db_file: str,
    *,
    duckdb_bin: str | None = None,
    duckdb_bin_feedback: str | None = None,
) -> None:
    """Align paths on oracle_benchmark and feedback_benchmark modules.

    Vanilla + oracle use ``duckdb_bin`` (default: ``<repo>/build/release/duckdb``).
    Feedback uses ``duckdb_bin_feedback`` if that file exists (default:
    ``<repo>/build/release/duckdb_feedback``), else the same binary as oracle.

    Override with env ``DUCKDB_BIN`` / ``DUCKDB_BIN_FEEDBACK`` or the keyword args.
    """
    import oracle_benchmark as ob
    import feedback_benchmark as fb

    main_default = os.path.join(repo_root, "build", "release", "duckdb")
    feedback_default = os.path.join(repo_root, "build", "release", "duckdb_feedback")

    main_bin = duckdb_bin or os.environ.get("DUCKDB_BIN") or main_default
    feedback_candidate = (
        duckdb_bin_feedback or os.environ.get("DUCKDB_BIN_FEEDBACK") or feedback_default
    )
    feedback_bin = feedback_candidate if os.path.isfile(feedback_candidate) else main_bin

    for mod in (ob, fb):
        mod.DUCKDB_DIR = repo_root
        mod.CARDINALITY_LOG = os.path.join(repo_root, "cardinality_log.txt")
        mod.ACTUAL_CARDINALITY_JSON = os.path.join(repo_root, "actual_cardinality.json")
        mod.PROFILE_OUTPUT = os.path.join(repo_root, "profile_output.json")
        mod.DB_FILE = db_file

    ob.DUCKDB_BIN = main_bin
    fb.DUCKDB_BIN = feedback_bin

    ob.RESULTS_LOG = os.path.join(repo_root, "unified_results.log")
    ob.ORACLE_CERTIFICATE_JSON = os.path.join(repo_root, "unified_oracle_certificate.json")


def profile_fingerprint(profile, mod) -> str | None:
    if profile is None:
        return None
    root = profile.get("children", [profile])[0] if profile.get("children") else profile
    text = mod.get_plan_structure_text(root)
    return mod.compute_plan_fingerprint(text)


def classify_plan_relation(fp_v: str | None, fp_f: str | None, fp_o: str | None) -> str:
    if fp_v is None or fp_f is None or fp_o is None:
        return "incomplete"
    eq12 = fp_v == fp_f
    eq13 = fp_v == fp_o
    eq23 = fp_f == fp_o
    if eq12 and eq13:
        return "all_equal"
    if eq12 and not eq13:
        return "vanilla_eq_feedback_only"
    if eq13 and not eq23:
        return "vanilla_eq_oracle_only"
    if eq23 and not eq12:
        return "feedback_eq_oracle_only"
    return "all_different"


def parse_queries_arg(spec: str | None) -> list[int]:
    if not spec or not spec.strip():
        return list(range(0, 100))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a.strip()), int(b.strip()) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def run_unified_single_query(query_nr: int, query_sql: str, ob, fb) -> dict:
    """Run all three phases; return one flat dict for the results table."""
    row: dict = {"query_nr": query_nr}

    # --- (1) Vanilla ---
    ob.clear_actual_cardinality_json()
    ob.clear_cardinality_log()
    vanilla_profile = ob.run_query_with_json_profile(query_sql)
    fp_v = profile_fingerprint(vanilla_profile, ob)
    row["fp_vanilla"] = fp_v
    row["vanilla_ok"] = vanilla_profile is not None

    if vanilla_profile is None:
        row["fp_feedback"] = None
        row["fp_oracle"] = None
        row["feedback_iterations"] = None
        row["feedback_unique_plans"] = None
        row["plan_relation"] = "incomplete"
        row["feedback_converged"] = None
        row["feedback_oscillation"] = None
        row["feedback_error"] = None
        row["oracle_error"] = None
        row["n_oracle_alarms"] = None
        row["n_inj_feedback"] = None
        row["n_inj_oracle"] = None
        row["n_oracle_keys"] = None
        return row

    # --- (2) Feedback ---
    fb.clear_actual_cardinality_json()
    fb.clear_cardinality_log()
    fb_res = fb.run_single_query(query_nr, query_sql)
    row["fp_feedback"] = fb_res.get("final_plan_fingerprint")
    row["feedback_iterations"] = fb_res.get("iterations")
    row["feedback_unique_plans"] = fb_res.get("feedback_unique_plans")
    row["feedback_converged"] = bool(fb_res.get("converged"))
    row["feedback_oscillation"] = bool(fb_res.get("oscillation"))
    row["feedback_error"] = bool(fb_res.get("error"))

    # --- (3) Oracle ---
    ob.clear_actual_cardinality_json()
    ob.clear_cardinality_log()
    o_res = ob.run_single_query(
        query_nr,
        query_sql,
        baseline_profile=vanilla_profile,
        skip_baseline_phase=True,
    )
    row["oracle_error"] = bool(o_res.get("error"))
    row["n_oracle_alarms"] = len(o_res.get("alarms") or [])
    row["fp_oracle"] = o_res.get("oracle_fingerprint")
    row["n_inj_feedback"] = fb_res.get("n_injected")
    row["n_inj_oracle"] = o_res.get("n_injected")
    row["n_oracle_keys"] = o_res.get("n_oracle")

    row["plan_relation"] = classify_plan_relation(
        fp_v, row["fp_feedback"], row["fp_oracle"]
    )

    ob.clear_actual_cardinality_json()
    ob.clear_cardinality_log()
    return row


def _compact_csv_bool_ok(v) -> str:
    """vanilla_ok: show False only when profile failed; blank when OK."""
    if v is None:
        return ""
    if v is False:
        return "False"
    if isinstance(v, str) and v.strip().lower() == "false":
        return "False"
    return ""


def _compact_csv_bool_err(v) -> str:
    """feedback_error / oracle_error: show True only on error."""
    if v is True:
        return "True"
    if isinstance(v, str) and v.strip().lower() == "true":
        return "True"
    return ""


def _fb_injects_gt_oracle_cell(nf, no_log, n_oracle_keys) -> str:
    """Flag when feedback had more INJECTED log lines than the oracle run (same metric).

    Suppressed when oracle wrote no keys to JSON (``n_oracle_keys`` is 0 or missing),
    since the oracle run is then not a meaningful injection baseline.
    """
    try:
        if n_oracle_keys is None or int(n_oracle_keys) <= 0:
            return ""
        if nf is not None and no_log is not None and int(nf) > int(no_log):
            return "yes"
    except (TypeError, ValueError):
        pass
    return ""


def _write_persistent_outputs(
    df,
    *,
    output_csv: str,
    output_summary_txt: str,
    db_file: str,
    verbose: bool,
) -> None:
    """Write CSV (machine-readable) and a fixed-width summary text file."""
    for path in (output_csv, output_summary_txt):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)

    work = df.copy()
    for c in RESULT_CSV_COLUMNS:
        if c not in work.columns:
            work[c] = None

    out_df = work[RESULT_CSV_COLUMNS].copy()
    if "vanilla_ok" in out_df.columns:
        out_df["vanilla_ok"] = out_df["vanilla_ok"].map(_compact_csv_bool_ok)
    for col in ("feedback_error", "oracle_error"):
        if col in out_df.columns:
            out_df[col] = out_df[col].map(_compact_csv_bool_err)
    out_df["fb_injects_gt_oracle"] = [
        _fb_injects_gt_oracle_cell(
            r["n_inj_feedback"], r["n_inj_oracle"], r["n_oracle_keys"]
        )
        for _, r in out_df.iterrows()
    ]
    out_df.to_csv(output_csv, index=False)

    # Summary .txt: same compact columns as CSV plus fingerprints for diffing.
    lines = [
        "unified_benchmark summary",
        f"generated_at: {datetime.datetime.now().isoformat()}",
        f"database: {db_file}",
        "",
    ]
    if out_df.empty:
        lines.append("(no rows)")
    else:
        txt_df = out_df.copy()
        for fc in INTERNAL_FP_COLUMNS:
            if fc in work.columns:
                txt_df[fc] = work[fc].values
        extra_fp = [c for c in INTERNAL_FP_COLUMNS if c in txt_df.columns and c not in RESULT_CSV_COLUMNS]
        lines.append(txt_df[list(RESULT_CSV_COLUMNS) + extra_fp].to_string(index=False))
    body = "\n".join(lines) + "\n"
    with open(output_summary_txt, "w", encoding="utf-8") as f:
        f.write(body)

    if verbose:
        print(f"\nWrote results CSV:    {output_csv}")
        print(f"Wrote summary (txt): {output_summary_txt}")


def print_query_summary(row: dict) -> None:
    q = row["query_nr"]
    print(
        f"  Q{q}: vanilla={row['fp_vanilla']}  feedback={row['fp_feedback']}  "
        f"oracle={row['fp_oracle']}  |  relation={row['plan_relation']}"
    )
    if row.get("feedback_iterations") is not None:
        print(
            f"       feedback: iterations={row['feedback_iterations']}  "
            f"unique_plans={row['feedback_unique_plans']}  "
            f"converged={row['feedback_converged']}  oscillation={row['feedback_oscillation']}  "
            f"error={row['feedback_error']}  "
            f"n_injected(log)={row.get('n_inj_feedback')}"
        )
    inj_warn = _fb_injects_gt_oracle_cell(
        row.get("n_inj_feedback"),
        row.get("n_inj_oracle"),
        row.get("n_oracle_keys"),
    )
    extra = "  [WARN: feedback injected more log lines than oracle]" if inj_warn == "yes" else ""
    print(
        f"       oracle: error={row['oracle_error']}  alarms={row['n_oracle_alarms']}  "
        f"n_injected(log)={row.get('n_inj_oracle')}  "
        f"n_oracle_keys={row.get('n_oracle_keys')}{extra}"
    )


def run_unified_benchmark(
    query_ids: list[int],
    repo_root: str = REPO_ROOT,
    db_file: str = DEFAULT_DB,
    *,
    duckdb_bin: str | None = None,
    duckdb_bin_feedback: str | None = None,
    oracle_log_file: bool = True,
    verbose: bool = True,
    output_csv: str | None = None,
    output_summary_txt: str | None = None,
):
    """
    Run vanilla → feedback → oracle for each query in query_ids.

    Writes ``output_csv`` (default ``<repo>/unified_benchmark_results.csv``) and
    ``output_summary_txt`` (default ``<repo>/unified_benchmark_summary.txt``).
    Load in pandas: ``pandas.read_csv(output_csv)``.

    Returns (pandas.DataFrame, list of row dicts).
    """
    import pandas as pd

    import oracle_benchmark as ob
    import feedback_benchmark as fb

    apply_benchmark_config(
        repo_root,
        db_file,
        duckdb_bin=duckdb_bin,
        duckdb_bin_feedback=duckdb_bin_feedback,
    )

    if output_csv is None:
        output_csv = os.path.join(repo_root, DEFAULT_OUTPUT_CSV)
    if output_summary_txt is None:
        output_summary_txt = os.path.join(repo_root, DEFAULT_OUTPUT_SUMMARY_TXT)

    if oracle_log_file:
        ob.init_log()
        ob._log("Unified benchmark: vanilla → feedback → oracle")
        ob._log(f"Database: {db_file}")

    assert os.path.exists(ob.DUCKDB_BIN), f"DuckDB binary not found: {ob.DUCKDB_BIN}"
    assert os.path.exists(fb.DUCKDB_BIN), f"DuckDB feedback binary not found: {fb.DUCKDB_BIN}"
    assert os.path.exists(db_file), f"Database not found: {db_file}"

    if verbose:
        print(f"DuckDB (vanilla + oracle): {ob.DUCKDB_BIN}")
        print(f"DuckDB (feedback loop):   {fb.DUCKDB_BIN}")
        if fb.DUCKDB_BIN == ob.DUCKDB_BIN:
            print(
                "  (feedback uses the same binary as oracle; build duckdb_feedback or set "
                "DUCKDB_BIN_FEEDBACK for legacy join-order behavior.)"
            )
        print("Extracting TPC-DS queries...")
    queries = ob.extract_tpcds_queries(query_ids)
    if verbose:
        print(f"  Loaded {len(queries)} queries.\n")

    rows: list[dict] = []
    for query_nr in query_ids:
        if query_nr not in queries:
            if verbose:
                print(f"[SKIP] Query {query_nr} not in tpcds_queries()")
            continue
        if verbose:
            print(f"\n{'=' * 70}\nUnified run: Q{query_nr}\n{'=' * 70}")
        row = run_unified_single_query(query_nr, queries[query_nr], ob, fb)
        rows.append(row)
        if verbose:
            print_query_summary(row)

    df = pd.DataFrame(rows)
    if len(rows) == 0:
        df = pd.DataFrame(columns=RESULT_CSV_COLUMNS)

    _write_persistent_outputs(
        df,
        output_csv=output_csv,
        output_summary_txt=output_summary_txt,
        db_file=db_file,
        verbose=verbose,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY (DataFrame)")
        print("=" * 70)
        if not df.empty:
            display_cols = [
                "query_nr",
                "plan_relation",
                "vanilla_ok",
                "feedback_iterations",
                "feedback_unique_plans",
                "feedback_converged",
                "feedback_oscillation",
                "feedback_error",
                "n_inj_feedback",
                "n_inj_oracle",
                "n_oracle_keys",
                "oracle_error",
                "n_oracle_alarms",
                "fp_vanilla",
                "fp_feedback",
                "fp_oracle",
            ]
            cols = [c for c in display_cols if c in df.columns]
            print(df[cols].to_string(index=False))
        else:
            print("(no rows)")

    if oracle_log_file:
        ob._log(f"Unified benchmark results CSV: {output_csv}")
        ob._log(f"Unified benchmark summary txt: {output_summary_txt}")
        ob._log("\nUnified benchmark complete.")
        ob.close_log()
        if verbose:
            print(f"\nOracle-style log also written to: {ob.RESULTS_LOG}")

    return df, rows


def main() -> int:
    try:
        import pandas  # required by run_unified_benchmark
    except ImportError:
        print(
            "unified_benchmark requires pandas. Install for this interpreter, e.g.:\n"
            "  python3 -m pip install --break-system-packages pandas\n"
            "(Homebrew Python is PEP 668 managed; --break-system-packages is required unless you use a venv.)",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description="Vanilla → feedback → oracle unified benchmark")
    parser.add_argument(
        "--repo",
        default=REPO_ROOT,
        help="DuckDB fork root (default: directory of this script)",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to DuckDB database file")
    parser.add_argument(
        "--duckdb",
        default=None,
        metavar="PATH",
        help="DuckDB CLI for vanilla + oracle (default: <repo>/build/release/duckdb or $DUCKDB_BIN)",
    )
    parser.add_argument(
        "--duckdb-feedback",
        default=None,
        metavar="PATH",
        help="DuckDB CLI for feedback phase only (default: <repo>/build/release/duckdb_feedback "
        "or $DUCKDB_BIN_FEEDBACK; falls back to --duckdb if missing)",
    )
    parser.add_argument(
        "--queries",
        default="",
        help="Comma-separated query ids, or ranges like 1-5 (default: 0-99)",
    )
    parser.add_argument(
        "--no-oracle-log",
        action="store_true",
        help="Do not open unified_results.log via oracle_benchmark.init_log (stdout only)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        metavar="PATH",
        help=f"Machine-readable results (default: <repo>/{DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--output-summary",
        default=None,
        metavar="PATH",
        help=f"Human-readable summary table (default: <repo>/{DEFAULT_OUTPUT_SUMMARY_TXT})",
    )
    args = parser.parse_args()

    csv_path = args.output_csv or os.path.join(args.repo, DEFAULT_OUTPUT_CSV)
    summary_path = args.output_summary or os.path.join(args.repo, DEFAULT_OUTPUT_SUMMARY_TXT)

    target = parse_queries_arg(args.queries)
    run_unified_benchmark(
        target,
        repo_root=args.repo,
        db_file=args.db,
        duckdb_bin=args.duckdb,
        duckdb_bin_feedback=args.duckdb_feedback,
        oracle_log_file=not args.no_oracle_log,
        verbose=True,
        output_csv=csv_path,
        output_summary_txt=summary_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
