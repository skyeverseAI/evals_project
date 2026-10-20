# Changes made for regression testing

Session: `rergression_testing` · 2026-10-20

Goal: make every eval **callable from outside** (inject one pipeline, get a
summary dict back) so `evals/run_suite.py` can run the whole suite against ONE
pipeline and write a snapshot JSON that regression testing diffs against a
baseline.

Nothing here has been run end-to-end yet. `evals/harness.py` now exists (see
its section below), so every eval and `run_suite.py` imports cleanly.

---

## Quality evals — restructured to `run(...)` + `run_local()`

Pattern for each: no work at import time; `run(...)` builds test cases,
calls `evaluate(...)`, and returns `summarize_by_metric(result)`;
`run_local()` builds the default object; `__main__` prints via `print_summary`.
Judge is now built inside `run()`. All imports come from `evals.harness`
(`load_goldens`, `summarize_by_metric`, `print_summary`).

| File | Entry point | Notes |
|---|---|---|
| `evals/evals_retriever.py` | `run(retriever)`, `run_local()` → `build_retriever()` | Hyperparameters now read off the **injected** retriever via `retriever_config()` (reranker → `top_k`, `fetch_k`, `reranker_model`; base → `base_k{k}`). Judge: `openai/gpt-4.1-mini`. |
| `evals/eval_generator.py` | `run()` (no args) | Isolation eval on golden context. `load_chunk_text()` + `golden_context(g, chunk_text)` are now functions. Kept `faithfulness_goldens.json` fields (`question`, `expected_answer`) and the chunk-pointer logic. |
| `evals/eval_rag_pipeline.py` | `run(rag)`, `run_local()` | Triad metrics. `fetch_k`/`top_k` read from `rag.retriever`, so `rag` must expose `.retriever.fetch_k` / `.top_k`. |
| `evals/eval_correctness.py` | `run(rag)`, `run_local()` | **Behaviour change:** now runs the LIVE pipeline instead of reading `generated_answer` saved by `writer.py`. Added a **Style** GEval (reference-free). Kept our correctness wording ("at least 9" on the 0–10 rubric). No hyperparameters logged yet. |

`evals/eval_retriever_with_reranker.py` is untouched and now redundant (the
suite passes the reranker into `evals_retriever.run`).

---

## New merged files — `run_*()` returns a flat dict, combined into a dotted snapshot

### `evals/eval_ops.py` — latency + cost + reliability
`run_latency`, `run_cost`, `run_reliability`, `run_ops(pipeline=None, verbose=True)`
→ keys `latency.*`, `cost.*`, `reliability.*`. No judge, no goldens.

Fixes vs. the pasted template:
1. `lat_summarize` returns NaN instead of crashing when there are no samples (e.g. all TTFT empty); SLO on NaN → FAIL.
2. `retry_rate` = % of requests needing ≥1 retry; raw attempts reported separately as `retry_attempts`.
3. `USD_TO_INR = 95.0` (matches the old `eval_cost.py`).
4. `PRICED_MODEL = "openai/gpt-4o-mini"`; cost report shows `llm.model_name` and warns if it differs from the priced model.
5. Removed unused `avg_cost_in`.

Verified: `ChatOpenRouter` returns `usage_metadata` with `input_token_details.cache_read`, so cost works with our model.

### `evals/evals_safety.py` — toxicity + leakage + scope
`run_toxicity`, `run_leakage`, `run_scope`, `run_safety(rag=None, verbose=True)`
→ keys `toxicity.*`, `leakage.*`, `scope.*`. Self-contained (own `load_goldens` /
`metric_summary`; does not depend on the harness).

- Toxicity: LLM judge + Jev (`typesafe/jev-1.14`), both timed; `TOX_USE_JEV` toggle. Lower = better.
- Leakage: prompt / course_content (GEval) + PII (`PIILeakageMetric`), plus `pii_in_context` count.
- Scope: GEval with expected action + success criteria.
- Each reports `errors` separately so judge failures don't silently drag averages.

**Bug fixed:** old `evals_leakage.py` referenced undefined `system_prompt` (NameError). The merged file uses `prompt.messages[0].prompt.template` from `src/generator.py`.

Old standalone files still present and unchanged: `eval_cost.py`,
`eval_latency.py`, `evals_toxicity.py`, `evals_leakage.py` (still has the bug),
`eval_scope.py`.

---

## `evals/run_suite.py` — orchestrator fixes

1. Imports corrected: `eval_retriever` → `evals_retriever`, `eval_application` → `eval_correctness`, `eval_safety` → `evals_safety`.
2. `evals/metric_registry.py` didn't exist at the time → import is guarded; if missing, ALL metrics are kept (no trimming) and a note is printed. (Registry now exists — see its section.)
3. `_slug()` strips deepeval's `" [GEval]"` suffix (`"Style [GEval]"` → `style`).
4. Retriever eval now tagged correctly when the suite passes `rag.retriever` (the reranker) — via `retriever_config()` in `evals_retriever.py`.
5. Prompt hash uses the template text, not `str(prompt)` (whose repr can change across LangChain versions).
6. Metadata now stamps the measured pipeline (generator model/temp, embedding model, chunk size/overlap, `fetch_k`, `top_k`, reranker model) and each eval's judge model.

Verified with a stub harness: imports resolve, slug/hash/metadata/retriever_config behave as expected.

---

## `evals/harness.py` — reviewed + adapted

Provided file checked against our setup. It fits: returns
`{metric_name: {n, pass_rate, avg/min/max_score}}`, which is exactly what
`run_suite.flatten_nested` expects; `pass_rate` is in % like `eval_ops` /
`evals_safety`; deepeval 4.3.0 exposes `result.test_results[*].metrics_data`
as it assumes; metric names within each eval are distinct, so no bucket collisions.

Changes made:
1. `load_goldens` opens with `encoding="utf-8"` (goldens contain Hinglish / non-ASCII text; the old per-file loaders already did this for some sets).
2. Judge/API **errors** are now counted separately (`errors` stat) and excluded from scores, instead of silently counting as a quality fail. They stay in `n`, so `pass_rate` still drops on errors — but `errors` shows why. Matches `evals_safety.metric_summary`.
3. `print_summary` shows `errors`.

GEval names still come through as e.g. `"Style [GEval]"` here — the suffix is stripped later by `run_suite._slug`, so the summary keys stay deepeval's raw names.

Verified with real deepeval `EvaluationResult` / `TestResult` / `MetricData` objects (incl. one errored metric), `flatten_nested` on the output, `load_goldens` on the three quality golden files (25 each), and all four quality evals + `run_suite` now import cleanly. Not yet run against live APIs.

Optional later: `evals_safety.py` still has its own `load_goldens` / `metric_summary`; it could switch to the harness now that it exists.

---

## `evals/metric_registry.py` — created from the provided template, adapted to our ids

`rule_for(metric_id)` → `{direction, kind (gate | guardrail | info), tol, rel_tol}`.
Judge metrics are judged on **avg_score** (not pass_rate); pass_rate is info.
`run_suite.py` picks this up automatically (its guarded import now succeeds),
so the default snapshot keeps only gates + guardrails.

Changes vs. the template:
1. **Toxicity direction bug fixed.** Template matched `safety.toxicity.avg_toxicity`, an id we never produce. Our id is `safety.toxicity.llm_avg_score`, which the template's generic `safety.* + avg_score` rule would have gated as **higher-is-better** — i.e. *more* toxic would have passed. Toxicity is now matched first and gated lower-is-better.
2. **Jev is info, not a gate.** `safety.toxicity.jev_avg_score` (second judge, for speed/agreement comparison) is tracked only; gating both judges doubles the chance of a noise-driven block.
3. **`slo_ttft_pass` is info.** Template's `endswith("_pass")` caught it, contradicting its own "TTFT too noisy to gate" note. Guarded booleans are now an explicit set: `ops.latency.slo_e2e_pass`, `ops.cost.budget_pass`.
4. **Tolerances flagged as inherited.** The noise-floor numbers (score ~0.033, safety ~0.002, latency ~20%) came from the template author's runs, not ours. Added a TODO to recalibrate (run the suite twice on the identical pipeline, diff, set tol just above observed noise).
5. Docstring lists the actual id shapes `run_suite` writes.

Resulting verdict set — 102 ids in a `--full` snapshot → **21 non-info**:

| Kind | Direction | Metric ids |
|---|---|---|
| gate | lower | `safety.toxicity.llm_avg_score` |
| gate | higher | `safety.leakage.{prompt,content,pii}_avg_score`, `safety.scope.avg_score` |
| guardrail | higher | `{retriever,generator,pipeline,application}.<metric>.avg_score` (10 metrics) |
| guardrail | lower | `ops.latency.e2e_p95_ms` (25%), `ops.cost.cost_per_query_usd` (15%), `ops.reliability.error_rate` (1 pt) |
| guardrail | higher | `ops.reliability.success_rate` (1 pt), `ops.latency.slo_e2e_pass`, `ops.cost.budget_pass` |

Everything else (pass_rate, min/max, n, errors, token counts, p50/p99, ttft, Jev, judge timings, `pii_in_context`) is info.

Verified by building every id the suite emits (quality via `run_suite.flatten_nested`, safety/ops from their return keys) and mapping each through `rule_for` — output matches the table above.

Things for `compare.py` to handle: `avg_score` can be `NaN` (all judge calls errored) — treat as "can't compare", not as pass/fail; `bool` rules compare True/False, not with tolerance; a high `errors` count on a metric should be surfaced since it makes that avg less trustworthy.

---

## Pending (do later)

- [x] **Write `evals/harness.py`** — done (see section above).
- [x] **Write `evals/metric_registry.py`** — done (see section above).
- [ ] Recalibrate registry tolerances from two identical-pipeline runs.
- [ ] Write `compare.py` (baseline vs candidate diff using the registry).
- [ ] Add hyperparameters to `eval_correctness.py`.
- [ ] Decide whether to delete the superseded standalone files listed above.
- [ ] Cost awareness: a full suite run is ~140+ pipeline calls plus all judge calls (toxicity judged twice).
- [ ] Snapshots may contain `NaN` (written as bare `NaN` by `json.dump`) — fine for Python readers, not strict JSON.
