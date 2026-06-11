---
name: credit-scoring-pipeline
description: >
  Full end-to-end credit scoring model pipeline. Covers EDA with segment analysis,
  preprocessing, feature engineering, baseline model comparison, segment-level error
  analysis, iterative retraining, and artifact export (model.pkl, input_schema.json,
  example_usage.py). Every step persists its output to MongoDB pipeline_state collection
  so the pipeline is fully resumable. Use when user wants to build a credit risk / PD
  (probability of default) model from a tabular dataset.
license: Apache-2.0
metadata:
  author: Financial AI Copilot
  version: "1.0"
  use_cases: credit_scoring, probability_of_default, credit_risk, loan_default
allowed-tools: save_pipeline_step, get_pipeline_state, run_script, run_code, save_finding, save_experiment_result, search_memory, save_to_memory
---

# Credit Scoring Pipeline

## Prerequisites

- User has uploaded a dataset (CSV or Parquet) via the UI.
- `download_dataset()` has been called → local path returned.
- `investigation_id` has been generated: `<dataset_stem>_<YYYYMMDD_HHMMSS>`.
- `MONGODB_URI` is set (pipeline_state writes will gracefully warn if not).

---

## Pipeline Overview

```
Step 01 — EDA & Segment Analysis       → MongoDB: pipeline_state.01_eda
Step 02 — Preprocessing                → MongoDB: pipeline_state.02_preprocessing
Step 03 — Feature Engineering          → MongoDB: pipeline_state.03_feature_engineering
Step 04 — Baseline Models              → MongoDB: pipeline_state.04_baseline_model + experiments
Step 05 — Error Analysis               → MongoDB: pipeline_state.05_error_analysis + findings
Step 06 — Iterative Training           → MongoDB: pipeline_state.06_iterative_training + experiments
Step 07 — Export Artifacts             → MongoDB: model_registry + pipeline_state.07_export
```

MongoDB is the **only** state bus between steps. Scripts do not read each other's
files — they query `pipeline_state` via MCP at startup.

---

## Step 01 — EDA & Segment Analysis

**Run**: `run_script("credit-scoring-pipeline/01_eda", dataset_path=<path>, target_col=<col>, investigation_id=<id>)`

**What it does**:
- Structural scan: shape, dtypes, missing rates, identifier detection
- Target distribution + class imbalance ratio
- Quick feature importance (RandomForest 100 trees, top 30 features)
- Leakage detection: flags features with importance > 0.90
- Segment risk: default rate by every categorical column (lift scores)
- Temporal patterns: default rate by month/year if date column exists
- Deterministic zones: thresholds where P(default) > 0.90
- Informative missingness: NaN vs non-NaN default rate per column

**Saves to MongoDB** (`pipeline_state`, step `01_eda`):
```json
{
  "shape": [rows, cols],
  "target_col": "...",
  "target_rate": 0.12,
  "imbalance_ratio": 7.3,
  "columns": [...],
  "dtypes": {...},
  "null_rates": {...},
  "quick_importance": [{"feature": "...", "importance": 0.34}, ...],
  "leakage_flags": ["col_a", "col_b"],
  "high_risk_segments": [{"column": "...", "value": "...", "default_rate": 0.45, "lift": 3.7, "n": 120}, ...],
  "temporal_pattern": {...},
  "deterministic_zones": [...],
  "informative_missingness": [...]
}
```

**Also writes findings** via `save_finding()` for each CRITICAL leakage flag and high-risk segment.

**Success criteria**: Step completes with `shape[0] > 0`, target column confirmed, `quick_importance` populated.

---

## Step 02 — Preprocessing

**Prerequisite**: Step 01 completed. Reads `01_eda` from pipeline_state via MCP.

**Run**: `run_script("credit-scoring-pipeline/02_preprocessing", dataset_path=<path>, target_col=<col>, investigation_id=<id>)`

**What it does**:
- Drops columns: identifiers detected in EDA, leakage flags, >70% missing
- Missing value imputation: median for numeric, mode for categorical
- Constant and near-constant removal (variance < 0.001)
- Outlier capping: IQR ×3 for numeric columns
- Categorical encoding: target encoding for high-cardinality (>20 unique), one-hot for low
- Train/test split: 80/20 stratified random (or out-of-time if `TEST_SPLIT=oot` env var set)
- Saves `train.csv`, `test.csv`, `preprocessor.pkl` to `/tmp/financial_ai/artifacts/<investigation_id>/`

**Saves to MongoDB** (`pipeline_state`, step `02_preprocessing`):
```json
{
  "train_path": "...",
  "test_path": "...",
  "preprocessor_path": "...",
  "train_rows": 8000,
  "test_rows": 2000,
  "features_kept": [...],
  "features_dropped": {"leakage": [...], "high_missing": [...], "constant": [...]},
  "encoding_map": {"col": "target_encoding"},
  "null_fill_map": {"col": 0.45},
  "train_target_rate": 0.118,
  "test_target_rate": 0.122
}
```

**Success criteria**: `train_rows > 100`, `test_rows > 20`, `features_kept` non-empty.

---

## Step 03 — Feature Engineering

**Prerequisite**: Step 02 completed. Reads `02_preprocessing` from pipeline_state.

**Run**: `run_script("credit-scoring-pipeline/03_feature_engineering", investigation_id=<id>)`

**What it does**:
- Reads `train.csv` and `test.csv` paths from pipeline_state
- Ratio features: debt_to_income variants, utilization rates, payment-to-income
- Interaction features: credit_score × employment_years, loan_amount × interest_rate
- Discretization: credit score bands (Poor/Fair/Good/Very Good/Exceptional), DTI buckets
- Polynomial: squares and log-transforms of top 5 features by importance
- Mutual information feature selection: top 50 features by MI score
- Saves `train_fe.csv`, `test_fe.csv` to artifacts dir

**Saves to MongoDB** (`pipeline_state`, step `03_feature_engineering`):
```json
{
  "train_fe_path": "...",
  "test_fe_path": "...",
  "features_engineered": [...],
  "mi_scores": [{"feature": "...", "mi": 0.12}, ...],
  "final_feature_list": [...],
  "n_features": 52
}
```

---

## Step 04 — Baseline Models

**Prerequisite**: Step 03 completed. Reads `03_feature_engineering` from pipeline_state.

**Run**: `run_script("credit-scoring-pipeline/04_baseline_model", investigation_id=<id>)`

**What it does**:
- Reads `train_fe.csv`, `test_fe.csv` paths from pipeline_state
- Trains 5 models: LogisticRegression, RandomForest, XGBoost, LightGBM, CatBoost
- 5-fold stratified CV on train set
- Reports: AUC-ROC, Gini, KS statistic, F1 at 0.5
- Saves best model (by AUC) as `best_model.pkl`, predict_proba scores as `test_scores.csv`
- Calls `save_experiment_result()` for each model

**Saves to MongoDB** (`pipeline_state`, step `04_baseline_model`):
```json
{
  "best_model_name": "XGBoost",
  "best_model_path": "...",
  "test_scores_path": "...",
  "leaderboard": [{"model": "XGBoost", "auc": 0.84, "gini": 0.68, "ks": 0.51}, ...],
  "best_auc": 0.84,
  "best_gini": 0.68,
  "best_ks": 0.51
}
```

**Block condition**: if `best_auc < 0.70`, save CRITICAL finding and stop pipeline. Investigate data quality.

---

## Step 05 — Error Analysis

**Prerequisite**: Step 04 completed. Reads `04_baseline_model` + `02_preprocessing` from pipeline_state.

**Run**: `run_script("credit-scoring-pipeline/05_error_analysis", investigation_id=<id>)`

**What it does**:
- Loads `test_fe.csv` + `test_scores.csv`
- Adds `predict_proba` column to test set
- Segment-level AUC: AUC by every categorical column (region, product, age_band, etc.)
- False negative analysis: high-default customers predicted safe (missed defaults)
- False positive analysis: low-default customers predicted risky (unnecessary rejections)
- Calibration check: predicted probabilities vs actual default rates by decile
- Identifies **weak segments**: segments where model AUC is > 0.05 below overall AUC
- Identifies **bias indicators**: groups with > 2× false positive rate vs overall

**Saves to MongoDB** (`pipeline_state`, step `05_error_analysis`):
```json
{
  "overall_auc": 0.84,
  "segment_aucs": [{"column": "region", "value": "rural", "auc": 0.71, "gap": -0.13, "n": 340}],
  "weak_segments": [...],
  "false_negative_rate": 0.18,
  "false_positive_rate": 0.09,
  "calibration_error": 0.04,
  "bias_flags": [{"group": "age_18_25", "fpr": 0.21, "ratio_vs_overall": 2.3}],
  "improvement_targets": ["rural region", "age_18_25", "product_C"]
}
```

**Also writes findings** for each weak segment and bias flag.

---

## Step 06 — Iterative Training

**Prerequisite**: Step 05 completed. Reads `05_error_analysis` + `03_feature_engineering` from pipeline_state.

**Run**: `run_script("credit-scoring-pipeline/06_iterative_training", investigation_id=<id>)`

**What it does**:
- Reads `improvement_targets` from Step 05 pipeline_state
- For each weak segment: upsamples that segment in training data (× 2 weight)
- Adds segment-specific interaction features for top improvement targets
- Trains XGBoost with `scale_pos_weight` for imbalance + segment weights
- Optuna hyperparameter tuning: 30 trials, optimizing AUC
- Evaluates on full test set AND per-segment
- Iterates up to 3 rounds — stops when overall AUC stops improving by > 0.005
- Saves `best_model_v2.pkl` and updated `test_scores_v2.csv`

**Saves to MongoDB** (`pipeline_state`, step `06_iterative_training`):
```json
{
  "final_model_path": "...",
  "final_scores_path": "...",
  "iterations": 2,
  "auc_progression": [0.84, 0.866, 0.871],
  "final_auc": 0.871,
  "final_gini": 0.742,
  "final_ks": 0.57,
  "best_params": {...},
  "segment_improvement": [{"segment": "rural", "auc_before": 0.71, "auc_after": 0.79}]
}
```

---

## Step 07 — Export Artifacts

**Prerequisite**: Step 06 completed (or Step 04 if Step 06 was skipped — AUC already good).

**Run**: `run_script("credit-scoring-pipeline/07_export_artifacts", investigation_id=<id>)`

**What it does**:
- Reads final model path from Step 06 (or 04) pipeline_state
- Produces `model.pkl` (final fitted model + preprocessor pipeline)
- Produces `input_schema.json`: every feature name, dtype, allowed range, example value
- Produces `example_usage.py`: self-contained script that loads model and scores one row
- Produces `model_card.md`: performance summary, training data description, known limitations, fairness notes
- Registers model in MongoDB `model_registry` collection

**Saves to MongoDB** (`model_registry`):
```json
{
  "investigation_id": "...",
  "model_name": "credit_scoring_xgboost",
  "version": "1.0",
  "artifact_dir": "/tmp/financial_ai/artifacts/<investigation_id>/",
  "metrics": {"auc": 0.871, "gini": 0.742, "ks": 0.57},
  "feature_count": 52,
  "training_rows": 8000,
  "dataset_name": "loans.csv",
  "registered_at": "..."
}
```

**Calls `save_report()`** to publish `model_card.md` to the UI files panel.

---

## Orchestrator Dispatch Pattern

```
1. list_available_datasets()          → find user dataset
2. download_dataset("loans.csv")      → local_path
3. load_dataset_info(local_path)      → quick schema preview
4. search_memory("credit scoring")    → check prior investigations
5. think_tool("Plan...")
6. generate investigation_id          → "loans_20260528_143022"

7. → data-profiling-agent
     run_script("credit-scoring-pipeline/01_eda", dataset_path, target_col, investigation_id)
     → MongoDB: pipeline_state.01_eda written

8. → feature-engineering-agent
     run_script("credit-scoring-pipeline/02_preprocessing", ...)
     run_script("credit-scoring-pipeline/03_feature_engineering", ...)
     → MongoDB: pipeline_state.02, 03 written

9. → model-research-agent
     run_script("credit-scoring-pipeline/04_baseline_model", ...)
     → MongoDB: pipeline_state.04 + experiments written

10. → validation-governance-agent
      run_script("credit-scoring-pipeline/05_error_analysis", ...)
      run_script("credit-scoring-pipeline/06_iterative_training", ...)
      → MongoDB: pipeline_state.05, 06 written

11. → report-generation-agent
      run_script("credit-scoring-pipeline/07_export_artifacts", ...)
      save_report("/tmp/financial_ai/artifacts/<id>/model_card.md")
      → MongoDB: model_registry written
```

## Resuming a Partial Pipeline

If the user says "continue" or "the pipeline crashed":
1. Call `get_pipeline_state(investigation_id)` → see which steps completed
2. Start from the first incomplete step
3. Each script reads its inputs from MongoDB — no manual state passing needed

## MCP Direct Queries (for context loading)

To check what a prior investigation found about credit scoring:
```
mongodb__aggregate → financial_ai_copilot.pipeline_state
filter: { "pipeline": "credit-scoring-pipeline" }
sort: { "timestamp": -1 }
limit: 5
```

To find the best model ever trained:
```
mongodb__find → financial_ai_copilot.model_registry
sort: { "metrics.auc": -1 }
limit: 1
```
