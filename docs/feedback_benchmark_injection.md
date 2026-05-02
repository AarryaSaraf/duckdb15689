# Feedback benchmark: when injection does not happen (detailed)

This document is **only** about the **iterative feedback** path (`feedback_benchmark.py` + `run_single_query`). It is written for someone who **does not already know DuckDB’s internals** but needs to understand what the feedback code does, **why** it sometimes refuses to write `actual_cardinality.json`, and **where** to improve it.

For oracle-only skip reasons, see `docs/benchmark_injection_gaps.md`.

For an **end-to-end worked example** (TPC-DS Q6, SF10: profiling, log lines, matches, JSON keys, second iteration with `INJECTED`), see [`docs/feedback_benchmark_walkthrough_q6_sf10.md`](feedback_benchmark_walkthrough_q6_sf10.md).

---

## 0. Vocabulary (read this first)


| Term                                | Plain meaning                                                                                                                                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logical join / join-order phase** | Before execution, the optimizer picks join order and **estimates** how many rows each join will produce. That work runs in C++ (`join_order` / `cardinality_estimator`).                                 |
| **`cardinality_log.txt`**          | Text log produced during planning: blocks like `LOGICAL_JOIN: ... Filters: ... Estimated Cardinality: N`. Each block has a long **`expression`** string (the full line) used as a **key** for injection. |
| **Physical join (profile)**         | What actually runs: `HASH_JOIN`, `NESTED_LOOP_JOIN`, etc., with **measured** output cardinality from the JSON execution profile.                                                                         |
| **`match_joins`**                   | Python orchestrator: precomputes normalized predicate sets, then calls **`match_single_profile_join`** once per physical join to pair it with **at most one** log line (exclusive `used_log_indices`).   |
| **`actual_cardinality.json`**      | Mapping `{ expression_string → integer rows }`. The engine reads this on a later run to **inject** cardinalities during estimation (same hook as oracle).                                                |
| **Injection**                       | Overriding the optimizer’s estimate with a stored integer for that expression key.                                                                                                                       |


**Important:** “Logical” here means **optimizer-side estimate identity**, not SQL `LOGICAL` types. The **physical** plan can reorder joins or use different algorithms; the benchmark tries to connect **measured** cardinalities back to **Estimator log lines** by **matching predicates and tables**, not by assuming plan shapes are identical.

---

## 1. How and when DuckDB decides *how many* and *which* join cardinalities to estimate

### 1.1 What “estimate a join” means in this fork

Join **cardinality estimates** are computed while the **join-order optimizer** evaluates candidate trees. Two representative call sites:

**Costing a candidate merge of left and right subplans** — cardinality of the **union of relation ids**:

```13:18:src/optimizer/join_order/cost_model.cpp
double CostModel::ComputeCost(DPJoinNode &left, DPJoinNode &right) {
	auto &combination = query_graph_manager.set_manager.Union(left.set, right.set);
	cardinality_estimator.SetPendingParentSplit(&left.set, &right.set);
	auto join_card = cardinality_estimator.EstimateCardinalityWithSet<double>(combination);
	auto join_cost = join_card;
	return join_cost + left.cost + right.cost;
}
```

**Building a DP join node** — same estimator, stores cardinality on the node:

```135:139:src/optimizer/join_order/plan_enumerator.cpp
	auto cost = cost_model.ComputeCost(left, right);
	auto result = make_uniq<DPJoinNode>(set, best_connection, left.set, right.set, cost);
	cost_model.cardinality_estimator.SetPendingParentSplit(&left.set, &right.set);
	result->cardinality = cost_model.cardinality_estimator.EstimateCardinalityWithSet<idx_t>(set);
	return result;
```

So an “estimated join” in the log corresponds to **one visit** to `EstimateCardinalityWithSet` for a particular **set of base relations** (and filters attached to that subgraph in the estimator), **not** “every possible pairing of tables in the SQL text.”

### 1.2 Why you do not see “every join from the SQL” in the log

- The optimizer explores **dynamic-programming (and possibly greedy)** combinations of relations. Only combinations that are **actually evaluated** generate log entries.
- After **many pair emissions**, enumeration can switch to a **heuristic** path (`TryEmitPair` stops exact search around **10k pairs** in `plan_enumerator.cpp`), which changes **which** subsets get estimated.
- **Rewrites** (subqueries, lateral joins, decorrelation) can change which “relations” exist from the estimator’s point of view versus what your mental picture of “FROM A JOIN B” is.

**Takeaway for readers:** the log is a **sample of estimator activity during one planning episode**, aligned with **relation ids and filters** inside the join-order module—not a mirror of every join-like phrase in the SQL.

---

## 2. What the feedback benchmark does (end-to-end)

**Environment (optional, no code edits):**


| Variable                      | Effect                                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `DUCKDB_FEEDBACK_DB`          | Path to TPC-DS DuckDB file (default: `tpcds_sf200.db` under your tree).                                       |
| `DUCKDB_FEEDBACK_SF`          | Integer scale factor label (printed in the banner only; must match the DB you point at).                      |
| `DUCKDB_FEEDBACK_MAX_QUERIES` | If set (e.g. `5`), only queries **Q1–Qn** run — useful for smoke tests; omit for full **99-query** benchmark. |
| (bundled SQL)                 | TPC-DS text is read from `feedback_queries/tpcds/qNN.sql`; regenerate with `python3 scripts/export_tpcds_query_files.py`. |
| `DUCKDB_BENCHMARK_MAIN_QUERY_TIMEOUT_SEC` | Optional per-query subprocess timeout in seconds (default **600**). |


After a full run, **`GLOBAL TOTALS`** prints summed iterations, plan-change events, last-iteration injected log lines across queries, converged / oscillation / error counts, etc., so two identical configs can be compared without manual tallying.

Per iteration:

1. **Execute** the query with **JSON profiling** → list of **physical** join operators and **actual** row counts (`extract_join_nodes` / profile parsing).
2. **Parse** `cardinality_log.txt` → list of **`LOGICAL_JOIN`** entries (expression string, filters, tables, estimated cardinality, etc.).
3. **`match_joins`** (via **`match_single_profile_join`**) → for each physical join with **non-empty** normalized conditions, pick **at most one** log line (see §3).
4. **`collect_quarantine_unsafe_expressions`** (and **`purge_unsafe_expressions_from_json`**) → may mark matches **unsafe** (ambiguous candidates, same-key actual collisions, dynamic filters, cross-iteration drift, …); quarantined keys are removed from JSON so later iterations cannot inject them.
5. **`update_actual_cardinality_json`** → append/update `{ expression → actual_cardinality }` for matches that survived quarantine (and are not CTE-duplicate expressions in the log).
6. Next iteration: the engine reads JSON during optimization and applies **injected** cardinalities when it recognizes the same expression key.

If matching or quarantine blocks a join, that feedback is **not** written (or is later **purged** from JSON). From iteration 2 onward, **`verify_injection`** runs checks 1–7 (warnings vs passes) against the JSON snapshot **before** that iteration’s updates.

**Per-query startup:** **`run_single_query`** deletes **`actual_cardinality.json`** and truncates **`cardinality_log.txt`** once at the beginning of that query’s loop, then truncates **only** the log at the start of **each** iteration so planning output stays per-iteration while JSON accumulates (subject to quarantine). The full **`main()`** driver clears both files again after each query so the next TPC-DS query starts cold.

---

## 3. How matching works (`precompute_join_match_caches`, `match_joins`, `match_single_profile_join`)

This section answers “what is the code actually doing?” for a new reviewer.

**Inputs:** `profile_joins` (physical operators), `log_entries` (parsed `LOGICAL_JOIN` lines).

**Preparation — `precompute_join_match_caches`:**

- For each profile join, parse **`conditions`**, **`normalize_condition_set`**, drop tautological normalized pieces via **`_is_tautology_condition`**, and store **`pnorm`** (a `frozenset` of normalized predicates).
- For each log entry, normalize **`filters`** the same way into **`lnorm`**, and record **`tables`**.

**Main loop — `match_joins` calls `match_single_profile_join` once per profile index `pidx`:**

1. **Empty `pnorm`** — `match_single_profile_join` returns `(None, None)` immediately (```835:836:feedback_benchmark.py```).
   - Example: cross product, or conditions not exposed in the profile text the parser sees.
   - Result: this physical join is **not** added to `matches` and **not** listed in `unresolved` (it is skipped silently).
2. **Consider each log line `lidx`** unless `lidx` is in **`used_log_indices`** (```853:855:feedback_benchmark.py```).
   - **Critical:** after a log line wins for **some** profile join, that **`lidx` is reserved** — no other profile join may claim it.
   - **Check 5** in **`verify_check_5_distinct_log_bindings`** warns if the same `log_index` appears twice in `matches` (would indicate a matcher bug).
3. **Predicate / table compatibility** — **`_log_candidate_context_ok`** (```756:766:feedback_benchmark.py```):
   - If **`lineage_incomplete`**: allow `lnorm == pnorm` or `pnorm ⊆ lnorm`, with an intersection check on table names when lineage is incomplete.
   - Else: require **`lnorm == pnorm`** and **`l_tables == p_desc_tables`** (exact table set match).
4. **Count candidates:** every log line that passes the gates increments **`candidate_count`** (```863:864:feedback_benchmark.py```).
5. **Pick a winner** — **`_join_match_stage`** assigns a discrete **stage** 0–3 (exact conditions+tables down to weakest), then **`_candidate_beats_best`** tie-breaks (optionally distance to profile **estimated** cardinality, then stage, table delta, condition delta, lexicographic expression).
6. **On success**, return a match dict with **`expression`**, **`log_index`**, **`actual_cardinality`**, **`candidate_count`**, **`selected_stage`**, etc. **If no log line wins**, return **`(None, unresolved_dict)`** so the join is recorded as **unresolved**.

**Outputs:** `matches` and `unresolved` (only joins that had **non-empty** normalized conditions but **no** winning unused log line).

---

## 4. Concrete scenarios by flow

Below, “log line” = one parsed `LOGICAL_JOIN` record in memory (same idea as one block in `cardinality_log.txt`).

### Flow A — happy path

**Scenario:** Single-star schema style query: fact `orders` joined to `customers` on `o_custkey = c_custkey`. The optimizer estimates `orders ⋈ customers` once; the executor runs **one** `HASH_JOIN` with that predicate in the profile.


| Stage                | What you see                                                                                                                                                         |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SQL (simplified)** | `FROM orders o JOIN customers c ON o.o_custkey = c.c_custkey`                                                                                                        |
| **Log**              | One `LOGICAL_JOIN` whose `Filters` normalize to `{ o.o_custkey = c.c_custkey }`, `RelSets` / tables cover `{orders, customers}`, `Estimated Cardinality: 1_500_000`. |
| **Profile**          | One join operator with the **same** predicate text (after parsing), `actual_cardinality = 1_480_000`.                                                                |
| **Matching**         | `pnorm` nonempty; exactly **one** log line passes → `candidate_count == 1`.                                                                                          |
| **JSON**             | Write `full_expression_string → 1480000`.                                                                                                                            |


**Why it works:** one physical operator, one estimator identity, one key—no competition.

---

### Flow B — no match (`unresolved`)

**Scenario A — cross product then filter**

The **optimizer** may still log a selective join for `{A,B}` with predicates. The **executor** might implement “join” as **nested loop with empty join conjuncts** in the profile and push predicates into a **Filter** **above** the join (depending on plan). Then the profile join row’s **`conditions`** field can be **empty** after parsing → **`pnorm`** is empty → **skipped at 835–836** before any log comparison.

```835:836:feedback_benchmark.py
    if not pnorm:
        return None, None
```


| Stage                | What you see                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SQL (conceptual)** | Still `A JOIN B ON …`, but explain JSON shows a cross-type join with **no** conjuncts on the operator.                                            |
| **Profile join**     | `conditions: ""` → `pnorm` empty → **ignored** by matcher (`(None, None)`).                                                                       |
| **Log**              | Still has a rich `LOGICAL_JOIN` for `A⋈B`.                                                                                                        |
| **Ambiguity**        | There is no ambiguity—this physical row is **ineligible** for text-based matching.                                                                  |
| **Outcome**          | **Not** appended to `unresolved` (no diagnostic row for empty `pnorm`). **Nothing** to inject until profiling exposes comparable predicates or the matcher is extended. |


**Scenario B — string normalization mismatch**

The log might say `CAST(x AS INT) = y` while the profile says `x = y` after different formatting. If **`normalize_condition_set`** does not map them to the same canonical form, **`lnorm != pnorm`** and no match occurs—again **`unresolved`**, not “ambiguous.”

**Why this is strict:** injecting from the wrong log line would silently poison cardinality for an unrelated join; empty or mismatched predicates are treated as **unknown**, not guessed.

---

### Flow C — ambiguous match (`candidate_count > 1`)

**Scenario — same predicates, different *estimator context***

The join-order search may visit **the same join predicate** in **multiple DP contexts**. Historically, this caused identical `expression` strings. We have since added **`CtxScanFilters`** and **`CtxOcc`** to the C++ log output to make these signatures as unique as possible. 

However, ambiguity still occurs when a query contains multiple subqueries that are *identical* down to their scan filters (e.g., Q88). Because the physical nodes in Iteration 1 all share the same vanilla estimated cardinality, the Python matcher cannot distinguish them and must **guess** the mapping.

**The Fix:** 
In Iteration 2, DuckDB injects the guessed cardinalities. The physical nodes now have *unique* estimated cardinalities, allowing the Python script to perfectly match them. Although correcting the guess causes a massive cardinality drift, the script recognizes that the previous match was a "vanilla guess" (`is_injected=False`) and safely bypasses the drift quarantine (see Flow G). Thus, Flow C ambiguity is now a temporary, **self-correcting** state!

---

### Flow D — same expression key, conflicting actual cardinalities (batch collision)

This flow described the issue where two physical joins produced **different measured row counts**, yet both matched log entries whose **`expression`** string was **byte-identical**. Then `actual_cardinality.json` could not store both facts under one key.

**The Fix (`CtxOcc`):** 
We added **`CtxOcc: N`** to the end of every `LOGICAL_JOIN` line in the C++ optimizer. This tracks the specific *occurrence* or visit index of the estimator for a given logic. Because the estimator always increments this counter, no two `LOGICAL_JOIN` blocks will ever produce byte-identical strings within the same query execution, even for `UNION ALL` branches.

As a result, Flow D batch collisions have been **effectively eliminated**. The Python script's `[ALARM-CONTEXT-COLLISION]` logic remains in place solely as a safeguard against unforeseen logging bugs.

---

### Flow E — CTE / duplicate expression text in the log file

**What `detect_cte_duplicates` does**

Historically, the estimator would plan a CTE multiple times, repeating **identical** `LOGICAL_JOIN` lines. The benchmark script counted how many times each full **`expression`** string appeared, and skipped injecting any strings that appeared >1 time (`[SKIP-CTE]`).

**The Fix (`CtxOcc`):**
Just like with Flow D, the introduction of **`CtxOcc: N`** assigns a unique occurrence index to the `LOGICAL_JOIN` string. As a result, `detect_cte_duplicates` will virtually never see the exact same expression string more than once. The quarantine heuristic remains in place as a safeguard, but is largely dormant.

---

### Flow F — dynamic filter guard

**What DuckDB is doing (conceptually)**

Parallel pipelines sometimes install **dynamic filters** (bloom / min-max pushed between operators). A hash join’s **static** conjuncts might be `A.id = B.id`, but scans underneath may show `Dynamic Filters=` lines for columns **not** listed in those conjuncts — runtime pruning that affects measured cardinality.

**What the benchmark checks**

For each match, it extracts column-like tokens from the profile join’s **`conditions`** string (`extract_condition_columns`) and scans **`subtree_operator_signatures`** for `Dynamic Filters=` optional columns (`extract_dynamic_filter_columns`). If **any** dynamic column is **not** among the join condition columns, the expression is marked unsafe:

```1410:1431:feedback_benchmark.py
def _quarantine_dynamic_filter_context(matches, profile_joins):
    """Subtree dynamic-filter columns outside join condition columns → unsafe key."""
    ...
        extra_dyn_cols = sorted([c for c in dyn_cols if c not in cond_cols])
        if not extra_dyn_cols:
            continue
        print(
            f"    [ALARM-DYNAMIC-FILTER-CONTEXT] dynamic filter cols outside join "
            f"condition: {extra_dyn_cols} -- {expr}"
        )
        out.add(expr)
```

**Concrete mini-scenario**


| Stage                | Content                                                                                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Physical join**    | `HASH_JOIN` conditions list only `ss_store_sk = s_store_sk`.                                                                                               |
| **Subtree**          | A probe-side scan shows `Dynamic Filters= optional:d_date_sk ...` from another pipeline.                                                                   |
| **`extra_dyn_cols`** | Contains `d_date_sk` (not in join conjuncts).                                                                                                              |
| **Outcome**          | `[ALARM-DYNAMIC-FILTER-CONTEXT]` → expression quarantined: we refuse to treat the logical join key as sufficient context for the **measured** cardinality. |


**Why:** injecting using only the logical join line could **overfit** one runtime pruning regime and harm plans where those dynamics differ.

---

### Flow G — cross-iteration cardinality drift (history)

**Setup**

`_record_match_history` appends, per matched **`expression`**, one record per iteration (actual cardinality, plan fingerprint, profile join index, subtree signatures, **`candidate_count`**, etc.) into **`expr_match_history`**.

**Drift detection — `_quarantine_cross_iteration_cardinality_drift`**

For each expression with **at least two** history entries, the code compares the **last two** iterations’ **`actual_cardinality`** values. If the absolute or relative delta exceeds **`STABLE_PLAN_ABS_DRIFT_TOLERANCE`** / **`STABLE_PLAN_REL_DRIFT_TOLERANCE`**, the expression is quarantined with **`[ALARM-CONTEXT-COLLISION] same key changed cardinality across iterations`** (```1434:1458:feedback_benchmark.py```). This does **not** require `plan_stable`; a changing plan can still trigger it if the same log key keeps matching with wildly different measured rows.

**Exception for Corrected Guesses:** If the previous match was **not** injected (i.e., it was matched against a default vanilla estimate) but the current match **is** injected, the drift is safely ignored. This teaches the script to forgive massive cardinality shifts when it is simply correcting an initial, random mapping guess (e.g., from Iteration 1).

**Concrete mini-scenario**


| Stage              | Content                                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Iteration 3**    | Match says expression `E` has actual **100_000** rows.                                                                                                        |
| **Iteration 4**    | Same or different plan text; measured actual for `E` is **15_000_000**.                                                                                        |
| **Thresholds**     | Default absolute **100** and relative **0.1%** — here both deltas blow past thresholds.                                                                        |
| **Interpretation** | The same JSON key is being tied to **inconsistent** measured cardinalities across runs (wrong identity, plan change, or nondeterminism). Safer to **quarantine** than poison JSON. |


**Contrast with Flow H:** Flow G is **large** drift between consecutive history samples; Flow H is **small** drift on **JSON overwrite policy** when updating an existing key while `plan_stable` is true.

---

### Flow H — small drift ignored (JSON update policy)

When **`update_actual_cardinality_json`** sees an existing key (including legacy unnamespaced keys when the new write uses a namespaced key) and the new actual differs only slightly **while** `plan_stable` is not false, it may **keep the prior JSON value** and print `[INFO] Small stable-plan drift…` (```1042:1058:feedback_benchmark.py```). New keys are written under **`make_namespaced_expression_key(expression, plan_fingerprint)`** when `plan_fingerprint` is set; **`run_single_query`** currently passes **`plan_fingerprint=None`**, so the file uses **raw** `LOGICAL_JOIN` lines as keys unless you call the updater from custom code with a fingerprint.

**Concrete mini-scenario**


| Stage           | Content                                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Stored JSON** | `E → 1_000_000` from iteration 5.                                                                                                |
| **Iteration 6** | Same plan structure; measured actual **999_985** (tiny noise / timing).                                                          |
| **Decision**    | Absolute delta ≤ 100 **or** relative delta ≤ 0.1% → **do not churn JSON** — avoids endless rewrite loops from measurement noise. |


**Trade-off:** the file may lag **slightly** behind the latest measurement on purpose for stability.

---

## 5. C++ fork: cardinality injection, estimator logging, and `legacy_feedback/`

This subsection orients reviewers who open the **engine** side before the Python matcher.

### What changed in `cardinality_estimator.cpp` (high level)

The fork extends **`CardinalityEstimator::EstimateCardinalityWithSet`** so that:

1. **Loading feedback / oracle cardinalities** — At startup of estimation (or first use), the estimator reads **`actual_cardinality.json`**. The path defaults to a workspace constant but can be overridden with **`DUCKDB_ACTUAL_CARDINALITY_JSON`**. Keys are the **full string** of a `LOGICAL_JOIN` line (same format as in the log).
2. **Writing the cardinality log** — Each multi-relation estimate appends to **`cardinality_log.txt`** (override: **`DUCKDB_CARDINALITY_LOG`**). Lines include `LOGICAL_JOIN: … Estimated Cardinality:` or `… using INJECTED Cardinality:` when a JSON key matched.
3. **Plan fingerprint namespacing** — **`DUCKDB_FEEDBACK_PLAN_FINGERPRINT`** can be set by **`run_query_with_json_profile`** (when `plan_fingerprint_hint` is non-`None`) so the estimator aligns with a fingerprinted JSON namespace. **`make_namespaced_expression_key`** / **`project_json_for_fingerprint`** in Python mirror that layout for verification and for JSON writes when `plan_fingerprint` is passed into **`update_actual_cardinality_json`**.
4. **Optional diagnostics** — Environment switches such as **`DUCKDB_DEBUG_CARD_ESTIMATE_NDJSON`** and **`DUCKDB_DUMP_INJECTION_STATS`** add NDJSON / stderr summaries for debugging injection counts (see comments near `DEBUG_CARD_ESTIMATE_*` and `DUMP_INJECTION_STATS` in the source).

The Python benchmark never parses planner internals directly for injection—it relies on these **textual keys** staying aligned with **`match_joins`**.

### `legacy_feedback/` directory

`src/optimizer/join_order/legacy_feedback/` holds **copies** of selected join-order C++ sources (`cardinality_estimator.cpp`, `plan_enumerator.cpp`, etc.) from an **older** revision—useful for **diffing** or for **alternate build targets** that compile this snapshot instead of the main-tree join order. The default **`build/release/duckdb`** path used by **`feedback_benchmark.py`** is the **main** optimizer tree, not this folder.

Day-to-day feedback experiments should focus on **`cardinality_estimator.cpp`** under `src/optimizer/join_order/` plus **`feedback_benchmark.py`**; open **`legacy_feedback/`** when you need a frozen reference of how those files looked in the archived experiment.

---

## 6. Summary table (feedback-specific)


| Mechanism                                     | Where                                      | Effect                                 |
| --------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| No comparable conditions                      | `match_single_profile_join` 835–836        | Silent skip (not in `unresolved`)      |
| No winning log line                           | `match_single_profile_join` 906–917        | `unresolved`                           |
| Multiple log candidates                       | `_quarantine_ambiguous_matches` 1379–1388  | Ambiguous → unsafe                     |
| Same `expression`, different actuals in batch | `_quarantine_same_key_actual_collisions` 1391–1407 | Collision → unsafe             |
| Dynamic filter guard                          | `_quarantine_dynamic_filter_context` 1410–1431 | Unsafe                           |
| CTE duplicate detection                       | `detect_cte_duplicates` 968–977; skip in `update_actual_cardinality_json` 1024–1026 | Skip write |
| Cross-iteration drift                           | `_quarantine_cross_iteration_cardinality_drift` | Unsafe (unless correcting vanilla guess)    |
| Small stable drift                            | `update_actual_cardinality_json` | Keep existing JSON value               |
| Plan oscillation (same structure revisited)   | `run_single_query`               | Stops with `oscillation` in result dict (if plan changed, or no JSON updates pending) |


---

## 7. What can improve with **better code / ideas** (benchmark + fork logging)

These are **mostly** incremental improvements **without** redesigning DuckDB’s planner IR:


| Area                     | Idea                                                                                                                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ambiguous matches**    | Use **distinct_candidate_relsets** / plan_path / structure hashes already logged in debug rows to disambiguate when `candidate_count > 1`; optional per-query override table.     |
| **Collision (Flow D)**   | **Namespace keys** by `log_index` or a hash of `(expression, CtxOcc, RelSets)` when writing JSON; teach the **reader** side to accept those keys (requires C++ reader alignment). |
| **Unresolved (Flow B)**  | Richer **condition parsing**; optional matching on **descendant tables only** for coarse injection (risky—needs experiments).                                                     |
| **Dynamic filter guard** | Tune extraction; add regression tests on known false positives.                                                                                                                   |
| **Thresholds**           | `STABLE_PLAN_`*, `LARGE_DELTA_*` (lines 35–38) are central knobs.                                                                                                                 |


---

## 8. What likely needs **structural** DuckDB / planner changes


| Issue                             | Why scripts alone stall                                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Non-unique expression text**    | Injection is keyed by **string**; duplicates need **planner-issued stable ids** or structured keys in the log. |
| **Log ↔ physical 1:1**            | Complex rewrites break naive correspondence; **provenance** from optimizer to profile would help.              |
| **Condition-less physical joins** | Need either better **profile fields** or explicit cardinality hooks for those operators.                       |


---

## 9. Rough split (qualitative)


| Category                                    | Nature                                                                                               |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Fixable in Python + incremental logging** | Tie-breakers, diagnostics, thresholds, optional namespacing **if** the engine can read it.           |
| **Structural**                              | **Globally unique** join-estimate identity across branches / duplicates; **first-class** provenance. |


---

## 10. Code index (quick navigation)


| Topic                       | File                                                     | Approx. lines                |
| --------------------------- | -------------------------------------------------------- | ---------------------------- |
| Subprocess + env            | `feedback_benchmark.py` `child_duckdb_env`, `run_query_with_json_profile` | 259–316          |
| Profile join extraction     | `feedback_benchmark.py` `extract_join_nodes`             | 375–438                      |
| Match caches + single join  | `feedback_benchmark.py` `precompute_join_match_caches`, `match_single_profile_join`, `match_joins` | 719–961 |
| Parse log                   | `feedback_benchmark.py` `parse_cardinality_log`          | 473+                         |
| CTE duplicate detection     | `feedback_benchmark.py` `detect_cte_duplicates`          | 968–977                      |
| JSON update / skip CTE      | `feedback_benchmark.py` `update_actual_cardinality_json` | 989–1082                     |
| Purge quarantined keys      | `feedback_benchmark.py` `purge_unsafe_expressions_from_json` | 221+                     |
| Verification checks 1–7     | `feedback_benchmark.py` `verify_check_*`, `verify_injection` | 1090–1316                |
| Quarantine union            | `feedback_benchmark.py` `collect_quarantine_unsafe_expressions` | 1462–1469              |
| Per-query feedback loop     | `feedback_benchmark.py` `run_single_query`               | 1500–1692                    |
| Join cardinality invocation | `cost_model.cpp`, `plan_enumerator.cpp`                  | §1                           |
| Estimator / log emission    | `cardinality_estimator.cpp` (main tree)                  | `EstimateCardinalityWithSet` |


---

*This document reflects the fork’s feedback benchmark; re-read `feedback_benchmark.py` after large refactors.*