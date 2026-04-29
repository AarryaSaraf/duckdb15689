# When feedback and oracle benchmarks skip injections

This document explains **why** the harness sometimes does **not** put a cardinality into `actual_cardinality.json`, even though DuckDB **always** computes **some** cardinality estimate for every logical join during optimization.

---

## The core intuition (short answer)

**DuckDB always estimates something.** The join-order estimator combines table stats, distinct counts, and filter factors into a **single number** per logical join (plus internal numerator/denominator detail). You can see that number in `cardinality_log.txt` as `Estimated Cardinality`, and in `ESTIMATION_DETAIL` lines.

So the problem is **not** “we don’t know that an estimate exists.”

The benchmarks add a **second** requirement for **oracle** and **feedback** injection:

> **We only inject when we can tie a logged logical-join key to a cardinality we trust to mean “the same join event” as the optimizer slot, and—for oracle—when we can obtain that cardinality by running a well-defined SQL `COUNT(*)` (or a blessed synthesis path).**

If that link breaks, the harness **skips** adding/updating JSON for that slot. DuckDB then keeps using its **native** estimate for that key (and if your JSON never contains that key, injection simply never fires).

So:

| Question | Answer |
|----------|--------|
| Is the problem that we don’t know what’s being estimated? | **No.** The log names the join (tables, filters, context). |
| Is the problem that we can’t auto-generate SQL? | **Often yes for oracle**, but it’s more precise: we may have **no** `SQL_COUNT_QUERY`, or we have SQL that **does not match** the join slot’s semantics, or SQL **fails** at runtime, or **strict rules** mark the slot “not injectable.” |
| Does skipping oracle computation mean DuckDB can’t inject? | **No.** You could still **hand-edit** JSON; the **benchmark** refuses to **compute** unsafe oracle values. |

---

## Part A — Oracle benchmark (`oracle_benchmark.py`)

### What “oracle injection” means here

1. Run `EXPLAIN` so the estimator logs logical joins.
2. For each candidate multi-table join line, optionally run a `COUNT(*)` query to get a **reference** cardinality.
3. Write `{ logical_join_expression → count }` into `actual_cardinality.json`.
4. Run the real query again; the C++ estimator looks up keys and may replace the model estimate.

Skipping step 2 for a join means **that expression never enters the oracle map**, so the oracle phase does not inject that cardinality (unless it was already in JSON from elsewhere).

### Skip reasons (conceptual)

#### 1. `single-table`

**What:** The log line involves only one base relation (or is treated as single-table for the benchmark).

**Why skip:** The experiment is about **join** cardinalities between multiple tables; single-relation “cardinality” is a different object (often scan cardinality).

**Tiny example:** A filter on `customer` alone may log a 1-relation line; we don’t push oracle counts for that in this workflow.

---

#### 2. `CTE`

**What:** The same long `LOGICAL_JOIN: ...` string appears **more than once** in one explain pass.

**Why skip:** Usually indicates **CTE materialization / reuse**: one logical pattern stands for **multiple** execution contexts. Injecting one number would be ambiguous.

**Tiny example:** A CTE `WITH x AS (SELECT …)` referenced twice can produce two optimizer visits that **share** text but not semantics.

---

#### 3. `not-injectable` (C++ `SQL_COUNT_INJECTABLE: no`)

**What:** The fork’s cardinality estimator decided this join slot must **not** receive a naïve `COUNT(*)` injection.

**Why:** The estimator builds `SQL_COUNT_QUERY` only when it believes the generated SQL answers the **same** relational question as the **cardinality slot for this DP node**. See `cardinality_estimator.cpp`: injectability requires, among other things:

- Non-empty `SQL_COUNT_QUERY` and `SQL_COUNT_REASON: ok`
- **Coverage:** multi-table predicates implied by filters must align with the hypergraph edges used for the estimate (`coverage_multi_missing_from_edges == 0`)
- **Closure match:** the FROM/WHERE closure used for SQL must match the **same relation set** as the join node (`sql_relset_matches_cardinality_join`)

**Tiny example (closure mismatch):** The optimizer may estimate cardinality for a **2-relation** join (e.g. `item` ⋈ `store_sales`), but the generated `COUNT(*)` might legally need **more** tables to express correlated predicates—running that count and injecting it into the **2-relation** slot would be **wrong**. The code marks that slot not injectable.

---

#### 4. `no-sql`

**What:** There is no usable `SQL_COUNT_QUERY`, and synthesis is off.

**Why:** Some joins use filters that don’t serialize to a simple comma-FROM + WHERE over base tables (placeholders, residuals, etc.).

**Tiny example:** An internal filter string containing artifical placeholders (`#[…]`) blocks building SQL.

---

#### 5. `ambiguous-relset-sql` (strict mode)

**What:** The same **relation-set key** (`RelSets: [...]`) appears with **two different** `SQL_COUNT_QUERY` texts in one explain pass.

**Why skip:** There is **no single canonical** “true” count for that key without extra disambiguation—which variant matches **this** join instance?

**Tiny example:** Same tables `{A,B}` appear in two join orders with different residual predicates encoded differently in SQL—picking one COUNT would be arbitrary.

Default `--oracle-permissive-context` relaxes this (inject all variants anyway—legacy behavior).

---

#### 6. `SQL_ERROR` / `TIMEOUT`

**What:** The `COUNT(*)` subquery failed or exceeded the timeout.

**Why skip:** No trustworthy oracle number.

**Tiny example:** Generated SQL references columns or aliases that don’t exist in standalone form.

---

#### 7. Oracle map written but **not injected at runtime** (plan divergence)

**What:** Phase 3 writes keys to JSON, but after re-optimization the **plan changes**, so some logical-join lines from the **original** explain never appear as “injected” in the **new** log.

**Why:** Injection keys are **expression strings**. If the optimizer **does not revisit** the same string, that slot isn’t overwritten—**V3 “not injected”** style messages in logs.

**This is not “we couldn’t estimate”** — it’s “the plan moved, so this logical label didn’t show up with injection.”

---

## Part B — Feedback benchmark (`feedback_benchmark.py`)

Feedback **does not** rely only on `SQL_COUNT_QUERY`. It matches **physical** join operators (from the JSON profile) to **logical** log lines, then writes observed cardinalities into JSON.

### When feedback does **not** inject (or removes keys)

| Situation | Meaning |
|-----------|---------|
| **CTE duplicate expressions** | Same as oracle: don’t inject ambiguous reused expressions. |
| **No match** | A physical join could not be matched to a unique log line (`unresolved`). |
| **Ambiguous match** (`candidate_count > 1`) | Multiple profile joins match the same log key—unsafe. |
| **Conflicting actuals** | Same expression matched with **different** actual cardinalities in one batch. |
| **Dynamic-filter context** | Subtree has dynamic filter columns **outside** the join condition columns—context may differ from the logical key. |
| **Large drift across iterations** | Same expression previously matched a very different actual while plan seemed stable—quarantine. |
| **Unsafe expression set** | Keys get **purged** from JSON when marked unsafe. |

**Tiny example (ambiguous):** Two hash joins in the profile both match the same `LOGICAL_JOIN` signature; we don’t know which actual cardinality belongs in JSON for that key.

### Important distinction

**Feedback** can sometimes inject **even when oracle cannot**, because feedback uses **executed** cardinalities from the profile, not a standalone SQL reconstruction. Conversely, **oracle** can inject joins that **feedback never matched** if matching fails.

---

## Part C — What still happens if JSON is empty or partial?

- **No key in JSON** → C++ uses **native** estimates only.
- **Partial oracle map** → **Some** joins use injected values, others use native estimates in the **same** optimization pass. The optimizer is then optimizing under **mixed** information.

That is why “partial injection” can change plans in subtle ways: you did not replace “all estimates,” only some slots.

---

## Summary diagram (mental model)

```
For each logical join slot during optimization:
  Native estimate ALWAYS exists (internal formula).

  Oracle benchmark may SKIP filling JSON[count for this slot]
      └─ cannot certify a matching COUNT(*) / injectability / ambiguity / runtime failure

  Feedback benchmark may SKIP writing JSON[count for this slot]
      └─ cannot match safely to one physical join + one actual cardinality

At execution lookup time:
  If JSON has key → overwrite estimate with injected value
  Else → keep native estimate
```

---

## Direct answers to “why can’t we inject?”

1. **We usually know what is being estimated** (it’s in the log line).
2. **The blocker is often:** “We cannot **automatically produce** a **standalone SQL query** whose result equals the cardinality of **that** join **in this query’s semantics**,” or “we **can** produce SQL but **should not trust** it for this DP node,” or “the **same key** would receive **conflicting** numbers.”
3. **DuckDB still estimates**; the benchmark is conservative about **what it writes into JSON** so we don’t pretend we measured something we didn’t.

---

## Where to look in code

| Piece | Location |
|-------|----------|
| Oracle skip logic | `oracle_benchmark.compute_oracle_cardinalities` |
| C++ injectability & `SQL_COUNT_INJECTABLE` | `src/optimizer/join_order/cardinality_estimator.cpp` (injectability + `ESTIMATION_DETAIL`) |
| Feedback quarantine | `feedback_benchmark.run_single_query` (unsafe / ambiguous / dynamic filter guards), `update_actual_cardinality_json` |
| JSON lookup at optimization | Same `cardinality_estimator.cpp` injection path |

---

*Last updated to match the fork’s benchmark behavior; re-read code if you change skip rules.*
