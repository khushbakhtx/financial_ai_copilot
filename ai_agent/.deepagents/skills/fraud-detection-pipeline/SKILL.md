---
name: fraud-detection-pipeline
description: >
  End-to-end fraud detection pipeline for transaction or application datasets.
  Covers EDA with fraud-specific profiling, statistical anomaly detection
  (Isolation Forest / LOF), graph-based fraud ring analysis (NetworkX + MongoDB
  fraud_relationships), velocity feature engineering, imbalance-aware model
  training, and production rule generation. Every step persists structured
  results to MongoDB pipeline_state. Use when user wants to detect fraud,
  flag anomalies, find fraud rings, or build a fraud scoring model.
license: Apache-2.0
metadata:
  author: Financial AI Copilot
  version: "1.0"
  use_cases: fraud_detection, anomaly_detection, transaction_monitoring, fraud_scoring
allowed-tools: save_pipeline_step, get_pipeline_state, run_script, run_code, save_finding, save_experiment_result, search_memory, save_to_memory
---

# Fraud Detection Pipeline

## Prerequisites

- User has uploaded a transaction or application dataset (CSV or Parquet).
- `download_dataset()` has been called → local path returned.
- `investigation_id` generated: `<dataset_stem>_<YYYYMMDD_HHMMSS>`.
- Dataset should have at minimum: a transaction amount or similar numeric field.
- Target column (fraud label) is optional — pipeline runs in unsupervised mode if absent.

---

## Pipeline Overview

```
Step 01 — EDA & Fraud Profiling      → MongoDB: pipeline_state.01_eda
Step 02 — Anomaly Detection          → MongoDB: pipeline_state.02_anomaly + findings
Step 03 — Graph Analysis             → MongoDB: pipeline_state.03_graph + fraud_relationships
Step 04 — Feature Engineering        → MongoDB: pipeline_state.04_features (GridFS: train_fe, test_fe)
Step 05 — Model Training             → MongoDB: pipeline_state.05_model + experiments
Step 06 — Rule Generation            → MongoDB: pipeline_state.06_rules + findings
```

MongoDB `fraud_relationships` collection stores the full entity graph
(accounts ↔ devices ↔ IPs ↔ merchants) for cross-investigation lookups.

---

## Step 01 — EDA & Fraud Profiling

**Run**: `run_script("fraud-detection-pipeline/01_eda", dataset_path, target_col, investigation_id)`

**What it does**:
- Structural scan: shape, dtypes, missing rates, potential ID columns
- Fraud rate (if target present), class imbalance
- Amount distribution: mean/median/99th percentile, log-normality
- Temporal analysis: fraud rate by hour-of-day, day-of-week, month
- Geographic / merchant / channel breakdown: fraud rate per category
- Velocity patterns: transactions per customer per hour (top-N flagged)
- Missing-as-signal: NaN default rate per column

**Saves to MongoDB** (`pipeline_state`, step `01_eda`):
```json
{
  "shape": [rows, cols], "target_col": "...", "fraud_rate": 0.028,
  "has_label": true, "amount_stats": {...}, "temporal_patterns": {...},
  "category_fraud_rates": [...], "velocity_flags": [...],
  "leakage_flags": [...], "id_candidates": [...]
}
```

---

## Step 02 — Anomaly Detection

**Run**: `run_script("fraud-detection-pipeline/02_anomaly_detection", dataset_path, target_col, investigation_id)`

**What it does**:
- Reads `01_eda` from MongoDB for feature guidance
- Isolation Forest (contamination=auto from fraud_rate or 0.05)
- Local Outlier Factor on a 10k sample
- Velocity anomaly: customers with tx_count > mean + 3σ in any 1-hour window
- Amount anomaly: transactions > 99.5th percentile for their merchant category
- Shared-entity detection: accounts sharing device_id / ip_address / email
- Saves anomaly scores + flags back into GridFS as `anomaly_scores` DataFrame

**Saves to MongoDB**:
- `pipeline_state.02_anomaly` — summary stats, threshold used, flag counts
- `findings` — CRITICAL/WARNING per anomaly type

---

## Step 03 — Graph Analysis

**Run**: `run_script("fraud-detection-pipeline/03_graph_analysis", dataset_path, target_col, investigation_id)`

**What it does**:
- Builds bipartite graph: accounts ↔ shared entities (device, IP, merchant, phone)
- NetworkX connected components → fraud rings (components with ≥ 3 nodes)
- Ring risk scoring: rings with any known fraud label → flagged as high-risk
- Writes every edge to MongoDB `fraud_relationships` collection for cross-investigation lookup
- Queries `fraud_relationships` via MCP to check if any entities appeared in prior investigations

**Saves to MongoDB**:
- `pipeline_state.03_graph` — ring count, max ring size, high-risk ring accounts
- `fraud_relationships` — all edges: `{entity_type, entity_id, account_id, investigation_id}`
- `findings` — CRITICAL for each high-risk ring

---

## Step 04 — Feature Engineering

**Run**: `run_script("fraud-detection-pipeline/04_feature_engineering", dataset_path, target_col, investigation_id)`

**What it does**:
- Reads `01_eda` and `02_anomaly` from MongoDB for feature guidance
- Velocity features: tx_count_1h, tx_count_24h, tx_amount_1h (rolling by customer)
- Deviation features: amount_vs_customer_avg, amount_vs_merchant_avg
- Ring membership flag: is_in_fraud_ring (from step 03 graph)
- Anomaly score feature: iso_forest_score from step 02
- Time features: hour_sin/cos, is_weekend, is_night, days_since_first_tx
- Encoding + train/test split (stratified if label present, else random)
- Saves train_fe + test_fe to GridFS

**Saves to MongoDB** (`pipeline_state.04_features`):
```json
{
  "train_fe_path": "gridfs://...", "test_fe_path": "gridfs://...",
  "features_engineered": [...], "n_features": 48
}
```

---

## Step 05 — Model Training

**Run**: `run_script("fraud-detection-pipeline/05_model_training", dataset_path, target_col, investigation_id)`

**What it does**:
- If label present: supervised training (XGBoost + LightGBM with scale_pos_weight)
- If no label: unsupervised scoring only (Isolation Forest ensemble score as final output)
- 5-fold stratified CV, AUC + F1 + Precision@top10% reported
- Saves `fraud_model.pkl` to artifact dir
- Saves `fraud_scores.parquet` to GridFS

**Saves to MongoDB**:
- `pipeline_state.05_model` + `experiments` collection

---

## Step 06 — Rule Generation

**Run**: `run_script("fraud-detection-pipeline/06_rule_generation", dataset_path, target_col, investigation_id)`

**What it does**:
- Reads model feature importances + SHAP values
- Translates top-10 decision paths into human-readable alert rules
- Example: "Flag IF amount > 2500 AND is_night = 1 AND new_device = 1"
- Scores each rule: precision, recall, estimated daily alert volume
- Saves rules as structured JSON to `findings` collection + model card

**Saves to MongoDB**:
- `pipeline_state.06_rules`
- `findings` — one INFO document per rule
- `model_registry` — fraud model registration

---

## Orchestrator Dispatch Pattern

```
1. list_available_datasets() + download_dataset()
2. load_dataset_info() → detect target column (look for: is_fraud, fraud, label, target)
3. generate investigation_id
4. get_pipeline_state(id) → resume check

5. → data-profiling-agent
     run_script("fraud-detection-pipeline/01_eda", ...)

6. [parallel] → fraud-investigation-agent
                 run_script("fraud-detection-pipeline/02_anomaly_detection", ...)
               → fraud-investigation-agent
                 run_script("fraud-detection-pipeline/03_graph_analysis", ...)

7. → feature-engineering-agent
     run_script("fraud-detection-pipeline/04_feature_engineering", ...)

8. → model-research-agent
     run_script("fraud-detection-pipeline/05_model_training", ...)

9. → report-generation-agent
     run_script("fraud-detection-pipeline/06_rule_generation", ...)
     save_report(".../model_card.md")
```

## MCP Queries for Cross-Investigation Context

Check if any entities from this dataset appeared in past fraud investigations:
```
mongodb__find → financial_ai_copilot.fraud_relationships
filter: { "entity_id": { "$in": ["<device_id_list>"] } }
```

Get all high-risk fraud rings ever found:
```
mongodb__find → financial_ai_copilot.findings
filter: { "type": "fraud_ring", "severity": "CRITICAL" }
sort: { "timestamp": -1 }
limit: 10
```
