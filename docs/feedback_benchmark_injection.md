# Feedback benchmark: when injection does not happen (detailed)

This document is **only** about the **iterative feedback** path (`feedback_benchmark.py` + `run_single_query`). It is written for someone who **does not already know DuckDB’s internals** but needs to understand what the feedback code does, **why** it sometimes refuses to write `actual_cardinality.json`, and **where** to improve it.

For oracle-only skip reasons, see `docs/benchmark_injection_gaps.md`.

For an **end-to-end worked example** (TPC-DS Q6, SF10: profiling, log lines, matches, JSON keys, second iteration with `INJECTED`), see [`docs/feedback_benchmark_walkthrough_q6_sf10.md`](feedback_benchmark_walkthrough_q6_sf10.md).

---

## 0. Vocabulary (read this first)


| Term                                | Plain meaning                                                                                                                                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logical join / join-order phase** | Before execution, the optimizer picks join order and **estimates** how many rows each join will produce. That work runs in C++ (`join_order` / `cardinality_estimator`).                                 |
| `**cardinality_log.txt`**           | Text log produced during planning: blocks like `LOGICAL_JOIN: ... Filters: ... Estimated Cardinality: N`. Each block has a long `**expression**` string (the full line) used as a **key** for injection. |
| **Physical join (profile)**         | What actually runs: `HASH_JOIN`, `NESTED_LOOP_JOIN`, etc., with **measured** output cardinality from the JSON execution profile.                                                                         |
| `**match_joins`**                   | Python function that tries to pair **each profile join** with **one** log line, using normalized join conditions + table sets.                                                                           |
| `**actual_cardinality.json`**       | Mapping `{ expression_string → integer rows }`. The engine reads this on a later run to **inject** cardinalities during estimation (same hook as oracle).                                                |
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


After a full run, `**GLOBAL TOTALS`** prints summed iterations, plan-change events, last-iteration injected log lines across queries, etc., so two identical configs can be compared without manual tallying.

Per iteration:

1. **Execute** the query with **JSON profiling** → list of **physical** join operators and **actual** row counts (`extract_join_nodes` / profile parsing).
2. **Parse** `cardinality_log.txt` → list of `**LOGICAL_JOIN`** entries (expression string, filters, tables, estimated cardinality, etc.).
3. `**match_joins**` → for each physical join with usable conditions, pick **at most one** log line (see §3).
4. **Safety gates** in `run_single_query` → may mark matches **unsafe** (ambiguous, collisions, dynamic filters, drift, …).
5. `**update_actual_cardinality_json`** → append/update `{ expression → actual_cardinality }` for **safe** matches only.
6. Next iteration: the engine reads JSON during optimization and applies **injected** cardinalities when it recognizes the same expression key.

If step 3 or 4 fails for a join, that feedback is **not** written (or is later **purged** from JSON).

---

## 3. How `match_joins` works (step-by-step)

This section answers “what is the code actually doing?” for a new reviewer.

**Inputs:** `profile_joins` (physical operators), `log_entries` (parsed `LOGICAL_JOIN` lines).

**Preparation:**

- For each profile join, normalize its **conditions** string into `pnorm` (a `frozenset` of normalized predicates).
- For each log entry, normalize `**filters`** into `lnorm` and record `**tables**`.

**Main loop — one iteration per profile join** (`pidx`):

1. **Skip if `pnorm` is empty** (`continue` at lines 937–938).
  - Example: cross product, or conditions not exposed in the profile text the parser sees.  
  - Result: this physical join **never** enters `matches` → may land in `unresolved` if nothing else saves it.
2. **Consider each log line `lidx`** unless `lidx` is in `**used_log_indices**` (lines 958–960).
  - **Critical:** after a log line wins for **some** profile join, that `**lidx` is reserved** — no other profile join may claim it.  
  - So **two different physical joins cannot both match the same log index.** (If they did, that would be a bug; the benchmark has **Check 5** to warn about duplicate `log_index`.)
3. **Predicate / table compatibility:**
  - If `**lineage_incomplete`**: allow `pnorm ⊆ lnorm` or equality (subset matching).
  - Else: require `**lnorm == pnorm**` (exact equality of normalized sets).
  - Table check: descendant tables from the profile must align with the log’s table set (exact or subset rules depending on lineage).
4. **Count candidates:** every log line that passes the filters increments `**candidate_count`** (line 992).
5. **Pick a winner** among candidates with a **staged** priority (exact tables + exact conditions first, then weaker matches) and **tie-breakers** (estimated cardinality closeness, lexicographic expression order, etc.) — lines 984–1047.
6. **Append a match** with fields including `**expression`** = full string from `log_entries[best_lidx]`, `**actual_cardinality**` from **this** physical join’s measured rows, and `**candidate_count`**.

**Outputs:** `matches` and `unresolved` (physical joins that had conditions but **no** winning log line).

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

The **optimizer** may still log a selective join for `{A,B}` with predicates. The **executor** might implement “join” as **nested loop with empty join conjuncts** in the profile and push predicates into a **Filter** **above** the join (depending on plan). Then the profile join row’s `**conditions`** field can be **empty** after parsing → `**pnorm`** is empty → **skipped at 937–938** before any log comparison.

```937:938:feedback_benchmark.py
        if not pnorm:
            continue  # No conditions (e.g., cross product)
```


| Stage                | What you see                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SQL (conceptual)** | Still `A JOIN B ON …`, but explain JSON shows a cross-type join with **no** conjuncts on the operator.                                            |
| **Profile join**     | `conditions: ""` → `pnorm` empty → **ignored** by matcher.                                                                                        |
| **Log**              | Still has a rich `LOGICAL_JOIN` for `A⋈B`.                                                                                                        |
| **Ambiguity**        | There is no ambiguity—this physical row is **ineligible** for text-based matching.                                                                |
| **Outcome**          | `unresolved` (lines 1174–1183). **Nothing** to inject for this operator until profiling exposes comparable predicates or the matcher is extended. |


**Scenario B — string normalization mismatch**

The log might say `CAST(x AS INT) = y` while the profile says `x = y` after different formatting. If `**normalize_condition_set`** does not map them to the same canonical form, `**lnorm != pnorm**` and no match occurs—again `**unresolved**`, not “ambiguous.”

**Why this is strict:** injecting from the wrong log line would silently poison cardinality for an unrelated join; empty or mismatched predicates are treated as **unknown**, not guessed.

---

### Flow C — ambiguous match (`candidate_count > 1`)

**Scenario — same predicates, different *estimator context***

The join-order search may visit **the same join predicate** in **multiple DP contexts** (different input cardinalities / relation bindings / context occurrence). The **full `expression` string** often differs (e.g. different `CtxInputCards`, `CtxOcc`, `RelSets` embedding), but **after normalization** two log lines can still both satisfy:

- `lnorm == pnorm` (same normalized filters), and  
- the same **descendant table set** check.

Then `**candidate_count == 2`** (or more). The matcher **still picks one winner** (tie-breakers), but `**run_single_query` refuses to inject** because the tie-breaker is a **heuristic**, not a proof of identity:

```2171:2178:feedback_benchmark.py
        # Strict safety gate: never inject ambiguous matches or conflicting
        # same-key cardinalities (beyond tolerated tiny drift).
        newly_unsafe = set()
        for m in matches:
            if int(m.get("candidate_count", 0)) > 1:
                expr = m["expression"]
                print(f"    [ALARM-AMBIGUOUS-MATCH] {expr}")
                newly_unsafe.add(expr)
```


| Stage                        | What you see                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **One physical `HASH_JOIN`** | `actual_cardinality = 50_000`.                                                                                                |
| **Log**                      | Two lines both normalize to the same filter set and pass table checks, but differ elsewhere in the long expression (context). |
| **Matcher**                  | Picks one line (e.g. closer to profile estimated cardinality).                                                                |
| **Safety**                   | `candidate_count > 1` → **quarantine** — **do not trust** injection for that key.                                             |


**Why it happens:** the **logical identity** in the log is **not unique** when you only look at normalized predicates + table sets; the estimator needed **two** rows for **two planning contexts**, but the physical operator only corresponds to **one** of them—without extra disambiguation (RelSets match, plan path, manual map), Python cannot know which row is “the” one.

---

### Flow D — same expression key, conflicting actual cardinalities (batch collision)

This is the flow that feels **counter-intuitive**. Two physical joins produced **different measured row counts**, yet both matched log entries whose `**expression`** string is **byte-identical**. Then `actual_cardinality.json` cannot store both facts under one key.

#### We are **not** missing physics — we are missing a **unique key**

Recall `**used_log_indices`** (§3): **two profile joins cannot consume the same `log_index`.** So this is **not** “two operators stole the same log line.”

What **can** happen is:

1. `**parse_cardinality_log`** produces **two entries** `log_entries[0]` and `log_entries[1]` with **different** `log_index` but `**expression` text exactly the same string**.
2. Physical join **Alpha** matches line **0** → `(expression E, actual 1_000)`.
3. Physical join **Beta** cannot take line **0** (used); matches line **1** → `(expression E, actual 50_000)`.
4. `**update_actual_cardinality_json`** conceptually needs `E → 1000` and `E → 50000` at once—**impossible** for a single JSON map.

The benchmark authors already documented that duplicate expression strings across **distinct** log lines are **expected** in some cases:

```1713:1728:feedback_benchmark.py
    # ----- Check 5: Each match must bind to a distinct cardinality-log line -----
    # The same LOGICAL_JOIN text may appear on multiple log lines (e.g. UNION); those
    # are different entries (different log_index). Duplicate expression strings are OK;
    # reusing the same log line for two physical joins is not.
    log_indices = [m["log_index"] for m in matches]
    ...
    if dup_exprs:
        print(f"    [VERIFY] Check 5 PASSED: {len(matches)} mappings; {len(dup_exprs)} "
              f"expression string(s) repeated across distinct log line(s) (OK).")
```

So **duplicate keys are “OK” for matching** (two branches can bind to two log lines), but they are **not OK for JSON** if the **measured** cardinalities **disagree**: you cannot store two values under one string.

#### Concrete story

Imagine a query shaped like `**UNION ALL` of two similar report queries**, each with `**stores ⋈ customers ON s.id = c.store_id`**. The estimator runs **twice** (once per branch) and emits **two** `LOGICAL_JOIN` blocks. If the serializer prints **identical** full lines (same RelSets/filters string—example only), you get **two** `log_entries` with the **same** `expression`. At execution time:

- Branch 1’s hash join outputs **1_000** rows.  
- Branch 2’s hash join outputs **80_000** rows (different data slice).

Both matches are **legitimate** (different `log_index`, different physical operators), but `**expression` collides** → `**ALARM-CONTEXT-COLLISION`** when building the batch.

```2179:2189:feedback_benchmark.py
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
```

**What we’re “missing” for a fix:** not a deeper understanding of SQL—a **unique injection key** per planner occurrence (e.g. stable join id + occurrence index in the log), or **namespacing** that always distinguishes branches so JSON keys never collide when actuals differ.

---

### Flow E — CTE / duplicate expression text in the log file

**What `detect_cte_duplicates` does**

After parsing `cardinality_log.txt` into `log_entries`, the benchmark counts how many times each full `**expression`** string appears:

```1192:1201:feedback_benchmark.py
def detect_cte_duplicates(log_entries):
    """
    Detect expressions that appear more than once in the cardinality log.
    These are likely CTEs that are planned once but executed multiple times.
    ...
    """
    expr_counts = Counter(entry["expression"] for entry in log_entries)
    return {expr for expr, count in expr_counts.items() if count > 1}
```

Any expression in that **duplicate set** is treated as **unsafe to inject by text alone**: the same key might refer to **multiple** estimator occurrences (CTE inlined several times, duplicate enumeration of the same subgraph, etc.). `**update_actual_cardinality_json`** skips them (`[SKIP-CTE]`).

**Concrete mini-scenario**


| Stage                       | Content                                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **SQL**                     | Query defines `WITH cte AS (SELECT … FROM big_fact JOIN dim ON …) SELECT * FROM cte UNION ALL SELECT * FROM cte …`                    |
| **Planner**                 | May estimate the join inside `cte` **multiple times** as it explores branches; the log can repeat **identical** `LOGICAL_JOIN` lines. |
| `**detect_cte_duplicates`** | Returns that expression string — appears **≥ 2** times in one parse of the file.                                                      |
| **Injection write**         | Skipped: writing **one** cardinality would be meaningless or wrong for the “other” occurrence(s).                                     |


This is **related but not identical** to Flow D: Flow D is “duplicate keys matched to **different actual cardinalities** this iteration”; Flow E is “duplicate keys seen **in the log parse** before matching,” used as a cheap **quarantine** heuristic.

---

### Flow F — dynamic filter guard

**What DuckDB is doing (conceptually)**

Parallel pipelines sometimes install **dynamic filters** (bloom / min-max pushed between operators). A hash join’s **static** conjuncts might be `A.id = B.id`, but scans underneath may show `Dynamic Filters=` lines for columns **not** listed in those conjuncts — runtime pruning that affects measured cardinality.

**What the benchmark checks**

For each match, it extracts column-like tokens from the profile join’s `**conditions`** string (`extract_condition_columns`) and scans `**subtree_operator_signatures**` for `Dynamic Filters=` optional columns (`extract_dynamic_filter_columns`). If **any** dynamic column is **not** among the join condition columns, the expression is marked unsafe:

```2206:2224:feedback_benchmark.py
        # Dynamic-filter guard: if a matched subtree scan has dynamic filter columns
        # outside this join's own condition columns, treat key as context-unsafe.
        for m in matches:
            ...
            extra_dyn_cols = sorted([c for c in dyn_cols if c not in cond_cols])
            if extra_dyn_cols:
                print(
                    f"    [ALARM-DYNAMIC-FILTER-CONTEXT] dynamic filter cols outside join "
                    ...
                )
                newly_unsafe.add(expr)
```

**Concrete mini-scenario**


| Stage                | Content                                                                                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Physical join**    | `HASH_JOIN` conditions list only `ss_store_sk = s_store_sk`.                                                                                               |
| **Subtree**          | A probe-side scan shows `Dynamic Filters= optional:d_date_sk ...` from another pipeline.                                                                   |
| `**extra_dyn_cols`** | Contains `d_date_sk` (not in join conjuncts).                                                                                                              |
| **Outcome**          | `[ALARM-DYNAMIC-FILTER-CONTEXT]` → expression quarantined: we refuse to treat the logical join key as sufficient context for the **measured** cardinality. |


**Why:** injecting using only the logical join line could **overfit** one runtime pruning regime and harm plans where those dynamics differ.

---

### Flow G — plan-stable drift across iterations

**Setup**

`expr_match_history` records, per expression key, each iteration’s matched actual cardinality and plan fingerprint. When the **plan structure text** matches the **previous** iteration (`plan_stable`), we expect relatively stable measurements for the **same** physical mapping.

**Drift detection**

If the **previous** and **current** actual cardinalities for the same expression differ by more than **both** `STABLE_PLAN_ABS_DRIFT_TOLERANCE` (absolute rows) **and** `STABLE_PLAN_REL_DRIFT_TOLERANCE` (relative), the key is treated as colliding across iterations (`[ALARM-CONTEXT-COLLISION]` with cross-iteration messaging, lines ~2246–2307).

**Concrete mini-scenario**


| Stage              | Content                                                                                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Iteration 3**    | Plan shape unchanged from iteration 2; match says expression `E` has actual **100_000** rows.                                                                  |
| **Iteration 4**    | Same plan shape (`plan_stable=True`), but measured actual for `E` is **15_000_000**.                                                                           |
| **Thresholds**     | Default absolute **100** and relative **0.1%** — here both deltas blow past thresholds.                                                                        |
| **Interpretation** | Something “stable” in structure is **not** stable in cardinality (stats bump, nondeterminism, wrong match identity). Safer to **quarantine** than poison JSON. |


**Contrast with Flow H:** Flow G is **large** drift under stability; Flow H is **small** drift treated as noise.

---

### Flow H — small drift ignored (JSON update policy)

When `**update_actual_cardinality_json`** sees an existing key and the new actual differs only slightly **while** `plan_stable` is true, it may **keep the prior JSON value** and print `[INFO] Small stable-plan drift…` (implementation ~lines 1316–1338).

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

The fork extends `**CardinalityEstimator::EstimateCardinalityWithSet`** so that:

1. **Loading feedback / oracle cardinalities** — At startup of estimation (or first use), the estimator reads `**actual_cardinality.json`**. The path defaults to a workspace constant but can be overridden with `**DUCKDB_ACTUAL_CARDINALITY_JSON**`. Keys are the **full string** of a `LOGICAL_JOIN` line (same format as in the log).
2. **Writing the cardinality log** — Each multi-relation estimate appends to `**cardinality_log.txt`** (override: `**DUCKDB_CARDINALITY_LOG**`). Lines include `LOGICAL_JOIN: … Estimated Cardinality:` or `… using INJECTED Cardinality:` when a JSON key matched.
3. **Plan fingerprint namespacing** — `**DUCKDB_FEEDBACK_PLAN_FINGERPRINT`** can scope injected keys when the Python benchmark namespaces JSON entries (`PLANFP:…::…`), so oscillating plans do not overwrite unrelated cardinalities.
4. **Optional diagnostics** — Environment switches such as `**DUCKDB_DEBUG_CARD_ESTIMATE_NDJSON`** and `**DUCKDB_DUMP_INJECTION_STATS**` add NDJSON / stderr summaries for debugging injection counts (see comments near `DEBUG_CARD_ESTIMATE_*` and `DUMP_INJECTION_STATS` in the source).

The Python benchmark never parses planner internals directly for injection—it relies on these **textual keys** staying aligned with `**match_joins`**.

### `legacy_feedback/` directory

`src/optimizer/join_order/legacy_feedback/` is a **frozen snapshot** of join-order sources from an earlier git revision (**“expt 1”**), **not** the code path used by the normal `duckdb` binary. Per `**legacy_feedback/README`**:

- It exists so you can build `**duckdb_feedback**` / `**duckdb_optimizer_join_order_feedback**` and compare behavior against the fork without losing the historical baseline.
- Regenerate with `**scripts/refresh_legacy_feedback_sources.sh**` if you intentionally refresh that snapshot.

Day-to-day feedback experiments should focus on `**cardinality_estimator.cpp**` (main tree) plus `**feedback_benchmark.py**`; use `**legacy_feedback**` only when you need an apples-to-apples comparison against that archived optimizer variant.

---

## 6. Summary table (feedback-specific)


| Mechanism                                     | Where                                | Effect                                 |
| --------------------------------------------- | ------------------------------------ | -------------------------------------- |
| No comparable conditions                      | `match_joins` 937–938                | Physical join skipped in matching loop |
| No winning log line                           | end of `match_joins`                 | `unresolved`                           |
| Multiple log candidates                       | `match_joins` + quarantine 2173–2178 | Ambiguous → unsafe                     |
| Same `expression`, different actuals in batch | 2179–2190                            | Collision → unsafe                     |
| Dynamic filter guard                          | 2206–2224                            | Unsafe                                 |
| CTE duplicate detection                       | 1192–1201 + update 1285–1287         | Skip write                             |
| Historical drift                              | 2246+                                | Unsafe                                 |
| Small stable drift                            | 1316–1338                            | Keep existing JSON value               |


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
| Match profile ↔ log         | `feedback_benchmark.py` `match_joins`                    | 897–1185                     |
| Parse log                   | `feedback_benchmark.py` `parse_cardinality_log`          | 570+                         |
| CTE duplicate detection     | `feedback_benchmark.py` `detect_cte_duplicates`          | 1192–1201                    |
| JSON update / skip CTE      | `feedback_benchmark.py` `update_actual_cardinality_json` | 1208–1400+                   |
| Verify duplicate expr OK    | `feedback_benchmark.py` (Check 5)                        | 1713–1729                    |
| Unsafe quarantine           | `feedback_benchmark.py` `run_single_query`               | 2171–2316                    |
| Join cardinality invocation | `cost_model.cpp`, `plan_enumerator.cpp`                  | §1                           |
| Estimator / log emission    | `cardinality_estimator.cpp`                              | `EstimateCardinalityWithSet` |


---

*This document reflects the fork’s feedback benchmark; re-read `feedback_benchmark.py` after large refactors.*