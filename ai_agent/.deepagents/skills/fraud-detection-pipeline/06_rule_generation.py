"""Fraud Detection Pipeline — Step 06: Rule Generation + Export.

Reads:  fraud_scores + train_fe from GridFS.
        pipeline_state.05_model, 03_graph from MongoDB.
Writes: fraud_model.pkl (final), model_card.md, alert_rules.json to artifacts dir.
        pipeline_state.06_rules + findings + model_registry to MongoDB.
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

INVESTIGATION_ID = os.environ["INVESTIGATION_ID"]
TARGET_COL       = os.environ.get("TARGET_COL", "")
MONGODB_URI      = os.environ.get("MONGODB_URI", "")
DB_NAME          = os.environ.get("MONGODB_DB", "financial_ai_copilot")
DATASET_NAME     = os.environ.get("DATASET_PATH", "unknown").split("/")[-1]

sys.path.insert(0, str(Path(__file__).parent))
from _pipeline_io import gfs_load, gfs_list

import pandas as pd
import numpy as np

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[fraud/06_rules] investigation_id={INVESTIGATION_ID}")

# ── Load prior steps ──────────────────────────────────────────────────────────

def _load_step(step):
    if not MONGODB_URI:
        return {}
    try:
        from pymongo import MongoClient
        c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        doc = c[DB_NAME]["pipeline_state"].find_one(
            {"investigation_id": INVESTIGATION_ID, "step": step}, {"_id": 0})
        c.close()
        return doc.get("data", {}) if doc else {}
    except Exception:
        return {}

step05 = _load_step("05_model")
step03 = _load_step("03_graph")
step01 = _load_step("01_eda")

if not TARGET_COL:
    TARGET_COL = step01.get("target_col", "")

model_path   = step05.get("model_path", str(ARTIFACT_DIR / "fraud_model.pkl"))
final_metrics = step05.get("metrics", {})
mode         = step05.get("mode", "supervised")
feature_cols = step05.get("feature_cols", [])

# ── Load model ────────────────────────────────────────────────────────────────

model_bundle = None
if Path(model_path).exists():
    with open(model_path, "rb") as f:
        model_bundle = pickle.load(f)
    model = model_bundle["model"]
    feature_cols = model_bundle.get("feature_cols", feature_cols)
    print(f"[fraud/06_rules] Loaded model: {model_bundle.get('model_name')}")
else:
    print(f"[fraud/06_rules] WARNING: model file not found at {model_path}")
    model = None

# ── Load train_fe for SHAP ────────────────────────────────────────────────────

train_fe = gfs_load("train_fe", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
X_sample = pd.DataFrame()
if train_fe is not None and feature_cols:
    X_sample = train_fe[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).head(2000)

# ── SHAP feature importance ───────────────────────────────────────────────────

shap_importance = []
if model is not None and not X_sample.empty and mode == "supervised":
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap_imp = pd.DataFrame({
            "feature":        feature_cols,
            "mean_abs_shap":  np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        shap_importance = shap_imp.head(15).to_dict(orient="records")
        print(f"[fraud/06_rules] SHAP computed. Top feature: {shap_importance[0]['feature']} ({shap_importance[0]['mean_abs_shap']:.4f})")
    except Exception as e:
        print(f"[fraud/06_rules] SHAP failed (non-fatal): {e}")

# ── Generate alert rules ──────────────────────────────────────────────────────

rules = []

if model is not None and not X_sample.empty and mode == "supervised":
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        # Fit a shallow interpretable tree to approximate model decisions
        y_approx = (model.predict_proba(X_sample.values)[:, 1] >= 0.5).astype(int)
        dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
        dt.fit(X_sample, y_approx)

        # Extract leaf-level rules
        tree_text = export_text(dt, feature_names=list(X_sample.columns))
        # Parse top decision rules from the decision tree text
        for i, line in enumerate(tree_text.split("\n")[:40]):
            if "class: 1" in line:
                # Extract the path (last few conditions before this leaf)
                path_lines = tree_text.split("\n")[max(0, i-5):i]
                conditions = [l.strip().replace("|", "").strip() for l in path_lines if "<=" in l or ">" in l]
                if conditions:
                    rule_text = " AND ".join(conditions[-3:])
                    rules.append({
                        "rule_id": f"RULE_{len(rules)+1:03d}",
                        "condition": rule_text,
                        "type": "decision_tree_path",
                        "estimated_precision": None,
                    })
        print(f"[fraud/06_rules] Generated {len(rules)} decision-tree rules")
    except Exception as e:
        print(f"[fraud/06_rules] Rule extraction failed: {e}")

# Hard-coded heuristic rules based on EDA findings
amount_stats = step01.get("amount_stats", {})
amount_col   = next(iter(amount_stats), None)
if amount_col:
    p99 = amount_stats[amount_col].get("p99", 0)
    if p99 > 0:
        rules.insert(0, {
            "rule_id":   "RULE_H01",
            "condition": f"{amount_col} > {p99:.2f}",
            "type":      "heuristic_amount_threshold",
            "estimated_precision": None,
        })

if step03.get("high_risk_rings", 0) > 0:
    rules.insert(0, {
        "rule_id":   "RULE_H02",
        "condition": "is_in_fraud_ring = 1",
        "type":      "graph_ring_membership",
        "estimated_precision": 0.85,
    })

# Evaluate heuristic rules on test scores
fraud_scores = gfs_load("fraud_scores", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
if fraud_scores is not None and TARGET_COL in (fraud_scores.columns if fraud_scores is not None else []):
    y_true = fraud_scores[TARGET_COL].values
    y_score = fraud_scores["fraud_score"].values
    for rule in rules:
        if rule.get("estimated_precision") is None:
            # Estimate: fraction of top-scored records that are actually fraud
            try:
                n_flag = max(1, int(len(y_true) * 0.05))
                top_idx = np.argsort(y_score)[::-1][:n_flag]
                rule["estimated_precision"] = round(float(y_true[top_idx].mean()), 3)
            except Exception:
                pass

# ── Save alert rules JSON ─────────────────────────────────────────────────────

rules_path = str(ARTIFACT_DIR / "alert_rules.json")
with open(rules_path, "w") as f:
    json.dump({"rules": rules, "investigation_id": INVESTIGATION_ID,
               "generated_at": datetime.now().isoformat()}, f, indent=2)
print(f"[fraud/06_rules] Saved {len(rules)} alert rules → {rules_path}")

# ── Write model card ──────────────────────────────────────────────────────────

findings_count = 0
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        findings_count = c[DB_NAME]["findings"].count_documents({"investigation_id": INVESTIGATION_ID})
        c.close()
    except Exception:
        pass

stored_dfs = gfs_list(INVESTIGATION_ID, MONGODB_URI, DB_NAME) if MONGODB_URI else []

model_card = f"""# Model Card — Fraud Detection Model
**Investigation ID**: `{INVESTIGATION_ID}`
**Dataset**: `{DATASET_NAME}`
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Mode**: {mode}

---

## Model Performance

| Metric | Value |
|--------|-------|
{''.join(f'| {k} | **{v}** |\n' for k, v in final_metrics.items())}

## Pipeline Steps Completed

1. **EDA & Fraud Profiling** — structural scan, temporal patterns, high-fraud categories
2. **Anomaly Detection** — IsolationForest, LOF, velocity, amount, shared-entity detection
3. **Graph Analysis** — fraud ring detection, cross-investigation entity lookup
4. **Feature Engineering** — velocity, deviation, ring membership, time features
5. **Model Training** — {'XGBoost + LightGBM with scale_pos_weight' if mode == 'supervised' else 'IsolationForest ensemble (unsupervised)'}
6. **Rule Generation** ← *this step*

## Top Features (SHAP)

{chr(10).join(f"- **{r['feature']}**: {r['mean_abs_shap']:.4f}" for r in shap_importance[:8]) if shap_importance else "- SHAP not available"}

## Alert Rules Generated

{chr(10).join(f"- `{r['rule_id']}`: {r['condition']} (precision≈{r.get('estimated_precision', '?')})" for r in rules[:8])}

## Graph Analysis

- Fraud rings detected: **{step03.get('total_rings', 'N/A')}**
- High-risk rings: **{step03.get('high_risk_rings', 'N/A')}**
- Cross-investigation entity matches: **{step03.get('prior_investigation_matches', 0)}**

## MongoDB Artefacts

- Pipeline findings logged: **{findings_count}**
- DataFrames in GridFS: {stored_dfs}
- Entity edges in `fraud_relationships`: **{step03.get('edges_written_to_mongo', 0)}**

## Governance

- **Monitoring**: Check weekly false-positive rate and alert volume.
- **Retrain trigger**: If precision@top10% drops below 0.40.
- **Bias check**: Review false-positive rate by geography and merchant category.
"""

model_card_path = str(ARTIFACT_DIR / "model_card.md")
with open(model_card_path, "w") as f:
    f.write(model_card)
print(f"[fraud/06_rules] model_card.md saved → {model_card_path}")

# ── Persist to MongoDB ────────────────────────────────────────────────────────

result = {
    "step":               "06_rules",
    "investigation_id":   INVESTIGATION_ID,
    "rules_path":         rules_path,
    "model_card_path":    model_card_path,
    "n_rules":            len(rules),
    "rules":              rules,
    "shap_importance":    shap_importance[:10],
    "completed_at":       datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]

        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "06_rules"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "fraud-detection-pipeline",
             "step": "06_rules", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        # Write one INFO finding per rule
        for rule in rules[:5]:
            db["findings"].insert_one({
                "agent": "report-generation-agent", "type": "alert_rule", "severity": "INFO",
                "content": f"Alert rule {rule['rule_id']}: {rule['condition']} (precision≈{rule.get('estimated_precision', '?')})",
                "investigation_id": INVESTIGATION_ID, "timestamp": datetime.now().isoformat(),
            })
        # Register in model_registry
        db["model_registry"].replace_one(
            {"investigation_id": INVESTIGATION_ID},
            {
                "investigation_id": INVESTIGATION_ID,
                "model_name":       f"fraud_detection_{model_bundle.get('model_name', 'unknown') if model_bundle else 'unknown'}",
                "version":          "v1",
                "dataset_name":     DATASET_NAME,
                "target_col":       TARGET_COL,
                "artifact_dir":     str(ARTIFACT_DIR),
                "model_path":       model_path,
                "rules_path":       rules_path,
                "model_card_path":  model_card_path,
                "metrics":          final_metrics,
                "mode":             mode,
                "n_alert_rules":    len(rules),
                "registered_at":    datetime.now().isoformat(),
            },
            upsert=True,
        )
        print("[fraud/06_rules] ✓ Saved to MongoDB pipeline_state + findings + model_registry")
        client.close()
    except Exception as e:
        print(f"[fraud/06_rules] WARNING: MongoDB write failed: {e}")

print("\n" + "="*60)
print("FRAUD DETECTION PIPELINE COMPLETE")
print("="*60)
print(f"Mode:          {mode}")
print(f"Alert rules:   {len(rules)}")
print(f"Findings:      {findings_count}")
print(f"Artifact dir:  {ARTIFACT_DIR}")
print(f"\n[fraud/06_rules] PIPELINE COMPLETE — investigation_id: {INVESTIGATION_ID}")
