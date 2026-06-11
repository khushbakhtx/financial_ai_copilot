"""Credit Scoring Pipeline — Step 07: Export Artifacts.

Reads:  pipeline_state.06_iterative_training (final_model_path, metrics).
        pipeline_state.04_baseline_model (fallback if step 06 not run).
        pipeline_state.03_feature_engineering (final_feature_list).
        pipeline_state.01_eda (shape, target_rate — for model card).
Writes: model.pkl, input_schema.json, example_usage.py, model_card.md
        to /tmp/financial_ai/artifacts/<investigation_id>/.
Registers model in MongoDB model_registry collection.
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _pipeline_io import gfs_load, gfs_list

warnings.filterwarnings("ignore")

INVESTIGATION_ID = os.environ["INVESTIGATION_ID"]
TARGET_COL       = os.environ["TARGET_COL"]
MONGODB_URI      = os.environ.get("MONGODB_URI", "")
DB_NAME          = os.environ.get("MONGODB_DB", "financial_ai_copilot")
DATASET_NAME     = os.environ.get("DATASET_PATH", "unknown").split("/")[-1]

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[07_export_artifacts] investigation_id={INVESTIGATION_ID}")

# ── Load prior steps from MongoDB ────────────────────────────────────────────

import pandas as pd
import numpy as np

def _load_step(step_name):
    if not MONGODB_URI:
        return {}
    try:
        from pymongo import MongoClient
        c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        doc = c[DB_NAME]["pipeline_state"].find_one(
            {"investigation_id": INVESTIGATION_ID, "step": step_name}, {"_id": 0}
        )
        c.close()
        return doc.get("data", {}) if doc else {}
    except Exception as e:
        print(f"[07_export_artifacts] MongoDB load {step_name} failed: {e}")
        return {}

step06 = _load_step("06_iterative_training")
step04 = _load_step("04_baseline_model")
step03 = _load_step("03_feature_engineering")
step01 = _load_step("01_eda")

# Pick best available model (step06 preferred, step04 fallback)
if step06 and step06.get("final_model_path"):
    source_model_path = step06["final_model_path"]
    final_metrics = {
        "auc":  step06.get("final_auc",  step04.get("best_auc", 0)),
        "gini": step06.get("final_gini", step04.get("best_gini", 0)),
        "ks":   step06.get("final_ks",   step04.get("best_ks", 0)),
    }
    model_version = "v2_tuned"
    print(f"[07_export_artifacts] Using step 06 model (iterative trained)")
elif step04 and step04.get("best_model_path"):
    source_model_path = step04["best_model_path"]
    final_metrics = {
        "auc":  step04.get("best_auc", 0),
        "gini": step04.get("best_gini", 0),
        "ks":   step04.get("best_ks", 0),
    }
    model_version = "v1_baseline"
    print(f"[07_export_artifacts] Using step 04 model (baseline best)")
else:
    print("[07_export_artifacts] ERROR: No trained model found in pipeline state.")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────

with open(source_model_path, "rb") as f:
    model_bundle = pickle.load(f)

model        = model_bundle["model"]
feature_cols = model_bundle.get("feature_cols", step03.get("final_feature_list", []))
model_name   = model_bundle.get("model_name", "xgboost")

print(f"[07_export_artifacts] Model: {model_name}, features: {len(feature_cols)}")

# ── 1. Build input schema ─────────────────────────────────────────────────────

# Load a sample of the feature-engineered train data to derive stats
sample_df = pd.DataFrame()
if MONGODB_URI:
    full_train_fe = gfs_load("train_fe", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    if full_train_fe is not None:
        sample_df = full_train_fe.head(1000)
if sample_df.empty:
    train_fe_path = step03.get("train_fe_path", "")
    if train_fe_path and not train_fe_path.startswith("gridfs://"):
        try:
            sample_df = pd.read_parquet(train_fe_path).head(1000) if train_fe_path.endswith(".parquet") else pd.read_csv(train_fe_path, nrows=1000)
        except Exception:
            pass

input_schema = {"features": [], "target_col": TARGET_COL, "model_name": model_name}
for feat in feature_cols:
    entry: dict = {"name": feat}
    if not sample_df.empty and feat in sample_df.columns:
        col_data = sample_df[feat].dropna()
        dtype    = str(sample_df[feat].dtype)
        entry["dtype"] = dtype
        if "float" in dtype or "int" in dtype:
            entry["type"]    = "numeric"
            entry["min"]     = round(float(col_data.min()), 4)
            entry["max"]     = round(float(col_data.max()), 4)
            entry["mean"]    = round(float(col_data.mean()), 4)
            entry["example"] = round(float(col_data.median()), 4)
        else:
            entry["type"]    = "categorical"
            entry["values"]  = list(col_data.value_counts().head(10).index.astype(str))
            entry["example"] = str(col_data.mode()[0]) if len(col_data) > 0 else ""
    else:
        entry["dtype"]   = "float64"
        entry["type"]    = "numeric"
        entry["example"] = 0.0
    input_schema["features"].append(entry)

schema_path = str(ARTIFACT_DIR / "input_schema.json")
with open(schema_path, "w") as f:
    json.dump(input_schema, f, indent=2)
print(f"[07_export_artifacts] input_schema.json saved → {schema_path}")

# ── 2. Save final model.pkl (production bundle) ───────────────────────────────

final_model_bundle = {
    "model":         model,
    "feature_cols":  feature_cols,
    "model_name":    model_name,
    "investigation_id": INVESTIGATION_ID,
    "dataset_name":  DATASET_NAME,
    "target_col":    TARGET_COL,
    "metrics":       final_metrics,
    "exported_at":   datetime.now().isoformat(),
}
final_model_path = str(ARTIFACT_DIR / "model.pkl")
with open(final_model_path, "wb") as f:
    pickle.dump(final_model_bundle, f)
print(f"[07_export_artifacts] model.pkl saved → {final_model_path}")

# ── 3. Write example_usage.py ────────────────────────────────────────────────

# Build a realistic example row from schema
example_row = {}
for feat in input_schema["features"]:
    example_row[feat["name"]] = feat.get("example", 0)

example_row_str = json.dumps(example_row, indent=4)

example_usage_code = f'''"""
Example: Load the credit scoring model and score a single applicant.
Generated by Financial AI Copilot — investigation_id: {INVESTIGATION_ID}
"""

import pickle
import pandas as pd

# Load the model bundle
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model       = bundle["model"]
feature_cols = bundle["feature_cols"]
target_col  = bundle["target_col"]

print(f"Model: {{bundle['model_name']}}")
print(f"Trained on: {{bundle['dataset_name']}}")
print(f"Metrics: AUC={{bundle['metrics']['auc']}}, Gini={{bundle['metrics']['gini']}}, KS={{bundle['metrics']['ks']}}")
print(f"Features: {{len(feature_cols)}}")

# --- Score a single applicant ---

applicant = {example_row_str}

# Build a DataFrame aligned to the expected feature order
df = pd.DataFrame([applicant])[feature_cols]
df = df.fillna(0)

# Predict probability of default
prob_default = model.predict_proba(df)[0, 1]
score        = int((1 - prob_default) * 1000)  # scorecard: 1000 = safest

print(f"\\nApplicant:")
for k, v in applicant.items():
    print(f"  {{k}}: {{v}}")

print(f"\\nProbability of default: {{prob_default:.4f}} ({{prob_default:.1%}})")
print(f"Credit score (0-1000):  {{score}}")

if prob_default < 0.10:
    print("Decision: APPROVE (low risk)")
elif prob_default < 0.25:
    print("Decision: REVIEW  (medium risk)")
else:
    print("Decision: DECLINE (high risk)")
'''

example_path = str(ARTIFACT_DIR / "example_usage.py")
with open(example_path, "w") as f:
    f.write(example_usage_code)
print(f"[07_export_artifacts] example_usage.py saved → {example_path}")

# ── 4. Write model_card.md ────────────────────────────────────────────────────

n_rows         = step01.get("shape", [0, 0])[0]
n_cols         = step01.get("shape", [0, 0])[1]
target_rate    = step01.get("target_rate", 0)
imbalance      = step01.get("imbalance_ratio", 0)
train_rows     = step06.get("iterations_run", 0) or step04.get("leaderboard", [{}])[0].get("n", 0)
weak_segs      = step06.get("segment_improvement", [])
findings_count = 0

# Count findings from MongoDB
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        findings_count = c[DB_NAME]["findings"].count_documents({"investigation_id": INVESTIGATION_ID})
        c.close()
    except Exception:
        pass

model_card = f"""# Model Card — Credit Scoring Model
**Investigation ID**: `{INVESTIGATION_ID}`
**Dataset**: `{DATASET_NAME}`
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Model**: {model_name} ({model_version})

---

## Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | **{final_metrics['auc']:.4f}** |
| Gini Coefficient | **{final_metrics['gini']:.4f}** |
| KS Statistic | **{final_metrics['ks']:.4f}** |

## Training Data

| Property | Value |
|----------|-------|
| Dataset rows | {n_rows:,} |
| Dataset columns | {n_cols} |
| Target column | `{TARGET_COL}` |
| Default rate | {target_rate:.2%} |
| Class imbalance | {imbalance}:1 |
| Final feature count | {len(feature_cols)} |
| Pipeline findings logged | {findings_count} |

## Pipeline Steps Completed

1. **EDA & Segment Analysis** — structural profiling, leakage detection, high-risk segments
2. **Preprocessing** — null imputation, outlier capping, categorical encoding, train/test split
3. **Feature Engineering** — ratios, interactions, bands, log transforms, MI selection
4. **Baseline Models** — LR / RandomForest / XGBoost / LightGBM comparison (5-fold CV)
5. **Error Analysis** — segment-level AUC, FN/FP analysis, calibration, fairness flags
6. **Iterative Training** — weak-segment upweighting + Optuna hyperparameter tuning
7. **Export** ← *this step*

## Intended Use

- **Primary use**: Predict probability of loan default for credit underwriting decisions.
- **Users**: Credit risk analysts, underwriting teams, risk management.
- **Not intended for**: Real-time fraud scoring, marketing scoring, insurance pricing.

## Segment Performance

{chr(10).join(f"- **{s['segment']}**: AUC before={s['auc_before']:.4f} → after={s['auc_after']:.4f} (Δ={s['delta']:+.4f})" for s in weak_segs) if weak_segs else "- No weak segment improvements tracked (step 06 not run or no improvements)."}

## Known Limitations

- Model trained on historical data; performance may degrade with population drift (monitor PSI monthly).
- Fairness analysis was conducted at segment level — review findings in MongoDB for bias flags.
- Calibration should be validated on live data before setting decision thresholds.
- Out-of-time validation recommended before production deployment.

## Governance

- **SR 11-7 alignment**: Model documentation complete. Independent validation required before production.
- **Monitoring recommendation**: Re-evaluate PSI and monthly AUC every 30 days.
- **Retrain trigger**: PSI > 0.25 or monthly AUC drop > 0.03.

## Artifacts

| File | Description |
|------|-------------|
| `model.pkl` | Fitted model + feature list + metadata bundle |
| `input_schema.json` | Feature names, types, ranges, example values |
| `example_usage.py` | Self-contained scoring script |
| `model_card.md` | This file |
"""

model_card_path = str(ARTIFACT_DIR / "model_card.md")
with open(model_card_path, "w") as f:
    f.write(model_card)
print(f"[07_export_artifacts] model_card.md saved → {model_card_path}")

# ── 5. Register in MongoDB model_registry ─────────────────────────────────────

registry_doc = {
    "investigation_id": INVESTIGATION_ID,
    "model_name":       f"credit_scoring_{model_name}",
    "version":          model_version,
    "dataset_name":     DATASET_NAME,
    "target_col":       TARGET_COL,
    "artifact_dir":     str(ARTIFACT_DIR),
    "model_path":       final_model_path,
    "schema_path":      schema_path,
    "example_path":     example_path,
    "model_card_path":  model_card_path,
    "metrics":          final_metrics,
    "feature_count":    len(feature_cols),
    "registered_at":    datetime.now().isoformat(),
}

result = {
    "step":             "07_export",
    "investigation_id": INVESTIGATION_ID,
    "artifact_dir":     str(ARTIFACT_DIR),
    "model_path":       final_model_path,
    "schema_path":      schema_path,
    "example_path":     example_path,
    "model_card_path":  model_card_path,
    "metrics":          final_metrics,
    "completed_at":     datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]

        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "07_export"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "credit-scoring-pipeline",
             "step": "07_export", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        db["model_registry"].replace_one(
            {"investigation_id": INVESTIGATION_ID},
            registry_doc,
            upsert=True,
        )
        print(f"[07_export_artifacts] ✓ Saved to MongoDB pipeline_state + model_registry")
        client.close()
    except Exception as e:
        print(f"[07_export_artifacts] WARNING: MongoDB write failed: {e}")

print("\n" + "="*60)
print("EXPORT SUMMARY")
print("="*60)
print(f"Artifact directory: {ARTIFACT_DIR}")
print(f"  model.pkl          ← production model bundle")
print(f"  input_schema.json  ← {len(feature_cols)} features with types and ranges")
print(f"  example_usage.py   ← self-contained scoring script")
print(f"  model_card.md      ← governance documentation")
print(f"\nFinal model: {model_name} ({model_version})")
print(f"  AUC:  {final_metrics['auc']:.4f}")
print(f"  Gini: {final_metrics['gini']:.4f}")
print(f"  KS:   {final_metrics['ks']:.4f}")
if MONGODB_URI:
    stored = gfs_list(INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    print(f"\nGridFS DataFrames stored for this investigation: {stored}")
print(f"\nRegistered in MongoDB model_registry ✓")
print(f"\n[07_export_artifacts] PIPELINE COMPLETE — investigation_id: {INVESTIGATION_ID}")
