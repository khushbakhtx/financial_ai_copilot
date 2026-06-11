# Financial AI Copilot — Persistent Agent Memory

This file accumulates knowledge across sessions. The orchestrator updates it via
`edit_file` whenever it learns something durable about a user's data, preferences,
or recurring patterns. Never store credentials or raw PII here.

---

## Pipeline Rules (Hardcoded)

- **Always run the full pipeline in order** for a new dataset: EDA → Preprocessing → Feature Engineering → Baseline Model → Error Analysis → Iterative Training → Export Artifacts.
- **MongoDB is the pipeline state bus.** Every script reads its inputs from `pipeline_state` collection at start and writes outputs back at end. Never pass large data between steps in memory — always go through MongoDB.
- **investigation_id is the partition key.** Generate it as `<dataset_stem>_<YYYYMMDD_HHMMSS>` on first use and reuse it for every step in that run.
- **Resumability rule.** Before dispatching a step, call `get_pipeline_state(investigation_id, step)` — if the step already has a completed entry, skip it unless the user explicitly asks to rerun.
- **Leakage threshold: Gini > 0.90** → exclude feature unconditionally. Dominance: top feature > 0.55 AND gap to second > 0.20 → exclude top feature.
- **Classification default** for binary credit/fraud targets. Use regression only when the target is continuous (e.g. loan amount, LTV ratio).
- **Test split default**: random 80/20 stratified. Only use out-of-time split when user explicitly requests it or dataset has a clear temporal column.
- **Minimum viable model bar**: AUC ≥ 0.70 for credit risk, ≥ 0.75 for fraud. Below this, block the export step and flag for investigation.
- **Export artifacts required**: every completed pipeline must produce `model.pkl`, `input_schema.json`, and `example_usage.py` saved to `/tmp/financial_ai/artifacts/<investigation_id>/`.
- **MCP-first reads**: ALL reads from MongoDB collections (findings, experiments, pipeline_state, model_registry, fraud_relationships) MUST use `mongodb__find` or `mongodb__aggregate` MCP tools. Never use run_code() with pymongo to read — that bypasses the MCP layer entirely.
- **Python tools for writes**: `save_finding`, `save_experiment_result`, `save_to_memory`, `save_pipeline_step` handle writes — they add timestamps, IDs, and embeddings automatically. Use them for all structured saves.
- **Read skill at start**: The `mongodb-memory` skill at `.deepagents/skills/mongodb-memory/SKILL.md` contains the full tool reference. Read it at the start of every new investigation thread.

---

## MongoDB Collections Used by Pipelines

| Collection | What goes there | Written by |
|---|---|---|
| `pipeline_state` | Per-step results keyed by `investigation_id` + `step` | Every pipeline script (via `save_pipeline_step`) |
| `findings` | CRITICAL/WARNING/INFO flags from any step | `save_finding` tool |
| `experiments` | Model metrics, hyperparams, CV scores | `save_experiment_result` tool |
| `model_registry` | Final exported model metadata | `07_export_artifacts.py` |
| `datasets` | Dataset metadata + GridFS refs | Terminal server upload endpoint |
| `vector_memory` | Embeddings for semantic search across investigations | `save_to_memory` tool |

---

## Superhuman EDA Protocol (Standard Behavior)

When user uploads a dataset, ALWAYS run these phases before delegating:

### Phase 1 — Structural Foundation (via `run_script("credit-scoring-pipeline/01_eda", ...)`)
- Shape, dtypes, missing rates, identifier detection
- Target distribution, class imbalance ratio
- Quick feature importance (RandomForest 100 trees)
- Leakage flags (importance > 0.90)

### Phase 2 — Segment Risk Analysis (inside 01_eda.py)
- Default/fraud rate by every categorical column
- Age/tenure brackets for continuous risk variables
- Geographic or branch-level variance
- Temporal patterns if date column present

### Phase 3 — Multivariate & Anomaly (inside 01_eda.py)
- Interaction effects: top 3 features cross-tabbed with target
- Informative missingness: NaN vs non-NaN default rate per column
- Deterministic zones: thresholds where outcome probability > 95%
- Batch/synthetic detection: duplicate rows, same-day clusters

### Recurring High-Value Patterns
- **Interest rate > 22%**: often deterministic default signal
- **Loan amount paradox**: risk peaks mid-range, drops for largest loans
- **Missing employer/sector**: proxy for informal employment, higher risk
- **Branch-product failure**: specific branches fail in specific products despite aggregate good performance

---

## User Preferences

*(Agent fills this in across sessions — dataset names, preferred thresholds, business context)*

---

## Model Performance Baselines

*(Agent fills this in — typical AUC/Gini ranges for user's domain, best model types seen)*

---

## Dataset Registry

- Azerbaijan_training_v1_fixed_with_target.csv (investigation_id: Azerbaijan_training_v1_20260530_225521): 15972 rows, 26 cols. Target: is_default (num_days_in_arrears_total >= 60).
