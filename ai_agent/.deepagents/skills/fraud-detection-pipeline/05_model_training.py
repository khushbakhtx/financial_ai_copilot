"""Fraud Detection Pipeline — Step 05: Model Training.

Reads:  train_fe + test_fe from GridFS.
        pipeline_state.04_features (feature list).
Writes: fraud_model.pkl to artifacts dir.
        fraud_scores to GridFS.
        pipeline_state.05_model + experiments to MongoDB.
"""

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

sys.path.insert(0, str(Path(__file__).parent))
from _pipeline_io import gfs_save, gfs_load

import pandas as pd
import numpy as np

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[fraud/05_model] investigation_id={INVESTIGATION_ID}")

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

step04 = _load_step("04_features")
if not TARGET_COL:
    step01 = _load_step("01_eda")
    TARGET_COL = step01.get("target_col", "")

df_train = gfs_load("train_fe", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
df_test  = gfs_load("test_fe",  INVESTIGATION_ID, MONGODB_URI, DB_NAME)

if df_train is None:
    p = step04.get("train_fe_path", "")
    if p and not p.startswith("gridfs://"):
        df_train = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        df_test  = pd.read_parquet(step04["test_fe_path"]) if step04.get("test_fe_path", "").endswith(".parquet") else pd.read_csv(step04.get("test_fe_path", ""))
    else:
        print("[fraud/05_model] ERROR: Cannot load train_fe/test_fe"); sys.exit(1)

has_label = bool(TARGET_COL and TARGET_COL in df_train.columns and df_train[TARGET_COL].sum() > 5)
feature_cols = [c for c in df_train.columns if c != TARGET_COL]

print(f"[fraud/05_model] train={df_train.shape}, has_label={has_label}")

# ── Supervised path ───────────────────────────────────────────────────────────

if has_label:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import xgboost as xgb
    import lightgbm as lgb

    X_train = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = df_train[TARGET_COL].values
    X_test  = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_test  = df_test[TARGET_COL].values

    scale_pos = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=42,
            eval_metric="auc", verbosity=0,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8,
            scale_pos_weight=scale_pos, random_state=42, verbose=-1,
        ),
    }

    results = {}
    fitted  = {}
    for name, model in models.items():
        import time
        t0 = time.time()
        cv_aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_pred = model.predict_proba(X_test)[:, 1]
        fpr_a, tpr_a, _ = roc_curve(y_test, y_pred)
        ks = float(np.max(tpr_a - fpr_a))

        # Precision@top10%: what fraction of top-10% scored are actually fraud
        n_top = max(1, int(len(y_test) * 0.10))
        top_idx = np.argsort(y_pred)[::-1][:n_top]
        p_at_10 = float(y_test[top_idx].mean())

        results[name] = {
            "model": name,
            "cv_auc_mean": round(float(cv_aucs.mean()), 4),
            "test_auc":    round(roc_auc_score(y_test, y_pred), 4),
            "test_ks":     round(ks, 4),
            "precision_at_top10pct": round(p_at_10, 4),
            "train_time_s": round(elapsed, 1),
        }
        fitted[name] = model
        print(f"[fraud/05_model] {name}: AUC={results[name]['test_auc']} KS={results[name]['test_ks']} P@10%={p_at_10:.3f}")

    leaderboard = sorted(results.values(), key=lambda x: x["test_auc"], reverse=True)
    best        = leaderboard[0]
    best_model  = fitted[best["model"]]

    y_pred_final = best_model.predict_proba(X_test)[:, 1]
    scores_df = df_test[[TARGET_COL]].copy()
    scores_df["fraud_score"]     = y_pred_final
    scores_df["predicted_fraud"] = (y_pred_final >= 0.5).astype(int)

    mode = "supervised"
    final_metrics = {"auc": best["test_auc"], "ks": best["test_ks"],
                     "precision_at_top10pct": best["precision_at_top10pct"]}

# ── Unsupervised path ────────────────────────────────────────────────────────

else:
    print("[fraud/05_model] No label — using ensemble anomaly score as fraud score")
    from sklearn.ensemble import IsolationForest

    X_train = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    X_test  = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train)

    scores_df = pd.DataFrame({"fraud_score": -iso.score_samples(X_test)})
    scores_df["predicted_fraud"] = (iso.predict(X_test) == -1).astype(int)

    best_model = iso
    leaderboard = [{"model": "isolation_forest", "unsupervised": True}]
    best = leaderboard[0]
    mode = "unsupervised"
    final_metrics = {"flagged_pct": round(float(scores_df["predicted_fraud"].mean()), 4)}

# ── Save model + scores ───────────────────────────────────────────────────────

model_path = str(ARTIFACT_DIR / "fraud_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": best_model, "feature_cols": feature_cols,
                 "model_name": best["model"], "mode": mode}, f)
print(f"[fraud/05_model] Saved fraud_model.pkl → {model_path}")

if MONGODB_URI:
    gfs_save(scores_df, "fraud_scores", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    scores_path = f"gridfs://{INVESTIGATION_ID}/fraud_scores"
else:
    scores_path = str(ARTIFACT_DIR / "fraud_scores.parquet")
    scores_df.to_parquet(scores_path, index=False)

# ── Persist to MongoDB ────────────────────────────────────────────────────────

result = {
    "step":              "05_model",
    "investigation_id":  INVESTIGATION_ID,
    "mode":              mode,
    "model_path":        model_path,
    "fraud_scores_path": scores_path,
    "leaderboard":       leaderboard,
    "best_model":        best["model"],
    "metrics":           final_metrics,
    "feature_cols":      feature_cols,
    "completed_at":      datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "05_model"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "fraud-detection-pipeline",
             "step": "05_model", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        db["experiments"].insert_one({
            "investigation_id": INVESTIGATION_ID,
            "experiment_name":  f"fraud_baseline_{INVESTIGATION_ID}",
            "model_name":       best["model"],
            "metrics":          final_metrics,
            "parameters":       {},
            "notes":            f"Fraud detection — {mode} mode",
            "timestamp":        datetime.now().isoformat(),
        })
        print("[fraud/05_model] ✓ Saved to MongoDB")
        client.close()
    except Exception as e:
        print(f"[fraud/05_model] WARNING: MongoDB write failed: {e}")

print(f"\n[fraud/05_model] COMPLETE — mode={mode}, best={best['model']}, metrics={final_metrics}")
