"""Credit Scoring Pipeline — Step 04: Baseline Models.

Reads:  train_fe + test_fe DataFrames from MongoDB GridFS (written by step 03).
Writes: best_model.pkl to artifacts dir (binary, stays on disk).
        test_scores DataFrame to MongoDB GridFS.
        pipeline_state.04_baseline_model + experiments collection to MongoDB.
Blocks: if best AUC < 0.70, saves CRITICAL finding and exits with code 2.
"""

import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

INVESTIGATION_ID = os.environ["INVESTIGATION_ID"]
TARGET_COL       = os.environ["TARGET_COL"]
MONGODB_URI      = os.environ.get("MONGODB_URI", "")
DB_NAME          = os.environ.get("MONGODB_DB", "financial_ai_copilot")
MIN_AUC          = float(os.environ.get("MIN_AUC", "0.70"))

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[04_baseline_model] investigation_id={INVESTIGATION_ID}")

# ── Load Step 03 state from MongoDB ──────────────────────────────────────────

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
from _pipeline_io import gfs_save, gfs_load

import pandas as pd
import numpy as np

fe_state = {}
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client_r = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        doc = client_r[DB_NAME]["pipeline_state"].find_one(
            {"investigation_id": INVESTIGATION_ID, "step": "03_feature_engineering"},
            {"_id": 0}
        )
        if doc:
            fe_state = doc.get("data", {})
            print(f"[04_baseline_model] ✓ Loaded feature engineering state from MongoDB")
        else:
            print("[04_baseline_model] ERROR: No feature engineering state. Run step 03 first.")
            sys.exit(1)
        client_r.close()
    except Exception as e:
        print(f"[04_baseline_model] MongoDB read failed: {e}")
        sys.exit(1)
else:
    print("[04_baseline_model] ERROR: MONGODB_URI required for pipeline state.")
    sys.exit(1)

df_train = gfs_load("train_fe", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
df_test  = gfs_load("test_fe",  INVESTIGATION_ID, MONGODB_URI, DB_NAME)

if df_train is None or df_test is None:
    # Local fallback
    train_fe_path = fe_state.get("train_fe_path", "")
    test_fe_path  = fe_state.get("test_fe_path", "")
    if train_fe_path and not train_fe_path.startswith("gridfs://"):
        df_train = pd.read_parquet(train_fe_path) if train_fe_path.endswith(".parquet") else pd.read_csv(train_fe_path)
        df_test  = pd.read_parquet(test_fe_path)  if test_fe_path.endswith(".parquet")  else pd.read_csv(test_fe_path)
    else:
        print("[04_baseline_model] ERROR: Could not load train_fe/test_fe.")
        sys.exit(1)

feature_cols = [c for c in df_train.columns if c != TARGET_COL]
X_train = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = df_train[TARGET_COL].values
X_test  = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
y_test  = df_test[TARGET_COL].values

print(f"[04_baseline_model] Train: {X_train.shape}, Test: {X_test.shape}")
print(f"[04_baseline_model] Train positive rate: {y_train.mean():.3%}")

# ── Model definitions ─────────────────────────────────────────────────────────

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics        import roc_auc_score, f1_score
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
import xgboost as xgb
import lightgbm as lgb

scale_pos = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1, class_weight="balanced",
    ),
    "xgboost": xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric="auc", verbosity=0,
    ),
    "lightgbm": lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=42, verbose=-1,
    ),
}

# ── 5-fold CV + test evaluation ───────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
fitted_models = {}

for name, model in models.items():
    print(f"[04_baseline_model] Training {name}...")
    import time
    t0 = time.time()
    cv_aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_gini = 2 * test_auc - 1

    # KS statistic
    from sklearn.metrics import roc_curve
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_proba)
    ks = float(np.max(tpr_arr - fpr_arr))

    test_f1 = f1_score(y_test, (y_pred_proba >= 0.5).astype(int), zero_division=0)

    results[name] = {
        "model":         name,
        "cv_auc_mean":   round(float(cv_aucs.mean()), 4),
        "cv_auc_std":    round(float(cv_aucs.std()), 4),
        "test_auc":      round(test_auc, 4),
        "test_gini":     round(test_gini, 4),
        "test_ks":       round(ks, 4),
        "test_f1":       round(test_f1, 4),
        "train_time_s":  round(elapsed, 1),
    }
    fitted_models[name] = model
    print(f"  CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}  |  Test AUC: {test_auc:.4f}  KS: {ks:.4f}  ({elapsed:.1f}s)")

# ── Pick best model ───────────────────────────────────────────────────────────

leaderboard = sorted(results.values(), key=lambda x: x["test_auc"], reverse=True)
best        = leaderboard[0]
best_model  = fitted_models[best["model"]]

print(f"\n[04_baseline_model] Best: {best['model']} — AUC={best['test_auc']}, Gini={best['test_gini']}, KS={best['test_ks']}")

# ── Save best model + test scores ─────────────────────────────────────────────

best_model_path = str(ARTIFACT_DIR / "best_model.pkl")
with open(best_model_path, "wb") as f:
    pickle.dump({"model": best_model, "feature_cols": feature_cols, "model_name": best["model"]}, f)
print(f"[04_baseline_model] Saved best_model.pkl → {best_model_path}")

y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
scores_df = df_test[[TARGET_COL]].copy()
scores_df["predict_proba"]   = y_pred_proba_best
scores_df["predicted_label"] = (y_pred_proba_best >= 0.5).astype(int)

if MONGODB_URI:
    gfs_save(scores_df, "test_scores", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    test_scores_path = f"gridfs://{INVESTIGATION_ID}/test_scores"
    print(f"[04_baseline_model] Saved test_scores → GridFS")
else:
    test_scores_path = str(ARTIFACT_DIR / "test_scores.parquet")
    scores_df.to_parquet(test_scores_path, index=False)
    print(f"[04_baseline_model] Saved test_scores → {test_scores_path} (local fallback)")

# ── Charts: feature importance, model comparison, ROC curves ─────────────────

from _pipeline_io import publish_artifact

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Feature importance of the best model
    est = best_model.named_steps["clf"] if hasattr(best_model, "named_steps") else best_model
    imp = None
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        imp = np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    if imp is not None and len(imp) == len(feature_cols):
        order = np.argsort(imp)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
        ax.barh([feature_cols[i] for i in order][::-1], imp[order][::-1], color="#2F6868")
        ax.set_title(f"Feature Importance — {best['model']} (top {len(order)})")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fi_path = ARTIFACT_DIR / "feature_importance.png"
        fig.savefig(fi_path, dpi=130)
        plt.close(fig)
        publish_artifact(fi_path, INVESTIGATION_ID, MONGODB_URI, DB_NAME,
                         kind="image", title="Feature Importance", step="04_baseline_model")

    # 2. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r["model"] for r in leaderboard][::-1]
    aucs  = [r["test_auc"] for r in leaderboard][::-1]
    bars = ax.barh(names, aucs, color=["#2F6868" if n == best["model"] else "#9DB8B3" for n in names])
    ax.set_xlim(0.5, 1.0)
    ax.set_title("Model Comparison — Test AUC")
    ax.set_xlabel("AUC-ROC")
    for b, v in zip(bars, aucs):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    mc_path = ARTIFACT_DIR / "model_comparison.png"
    fig.savefig(mc_path, dpi=130)
    plt.close(fig)
    publish_artifact(mc_path, INVESTIGATION_ID, MONGODB_URI, DB_NAME,
                     kind="image", title="Model Comparison (AUC)", step="04_baseline_model")

    # 3. ROC curves of all models
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, model in fitted_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr_c, tpr_c, _ = roc_curve(y_test, proba)
        ax.plot(fpr_c, tpr_c, label=f"{name} (AUC={results[name]['test_auc']:.3f})",
                linewidth=2 if name == best["model"] else 1.2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Baseline Models")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    roc_path = ARTIFACT_DIR / "roc_curves.png"
    fig.savefig(roc_path, dpi=130)
    plt.close(fig)
    publish_artifact(roc_path, INVESTIGATION_ID, MONGODB_URI, DB_NAME,
                     kind="image", title="ROC Curves", step="04_baseline_model")
except Exception as e:
    print(f"[04_baseline_model] chart generation failed (non-fatal): {e}")

# Publish the trained model binary so it is downloadable from the UI
publish_artifact(best_model_path, INVESTIGATION_ID, MONGODB_URI, DB_NAME,
                 kind="model", title=f"Best baseline model ({best['model']})",
                 step="04_baseline_model")

# ── Persist to MongoDB pipeline_state + experiments ───────────────────────────

result = {
    "step":              "04_baseline_model",
    "investigation_id":  INVESTIGATION_ID,
    "best_model_name":   best["model"],
    "best_model_path":   best_model_path,
    "test_scores_path":  test_scores_path,
    "feature_cols":      feature_cols,
    "leaderboard":       leaderboard,
    "best_auc":          best["test_auc"],
    "best_gini":         best["test_gini"],
    "best_ks":           best["test_ks"],
    "best_f1":           best["test_f1"],
    "completed_at":      datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]

        # pipeline_state
        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "04_baseline_model"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "credit-scoring-pipeline",
             "step": "04_baseline_model", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )

        # experiments collection — one doc per model
        for r in leaderboard:
            db["experiments"].insert_one({
                "investigation_id":  INVESTIGATION_ID,
                "experiment_name":   f"baseline_{INVESTIGATION_ID}",
                "model_name":        r["model"],
                "metrics": {
                    "auc":   r["test_auc"],
                    "gini":  r["test_gini"],
                    "ks":    r["test_ks"],
                    "f1":    r["test_f1"],
                    "cv_auc_mean": r["cv_auc_mean"],
                    "cv_auc_std":  r["cv_auc_std"],
                },
                "parameters":  {},
                "notes":       f"Baseline comparison, step 04",
                "timestamp":   datetime.now().isoformat(),
            })

        # Block condition: flag CRITICAL if AUC too low
        if best["test_auc"] < MIN_AUC:
            db["findings"].insert_one({
                "agent":            "model-research-agent",
                "type":             "model_quality",
                "severity":         "CRITICAL",
                "content":          f"Best model AUC={best['test_auc']} is below minimum threshold {MIN_AUC}. Do not proceed to export. Investigate data quality.",
                "investigation_id": INVESTIGATION_ID,
                "timestamp":        datetime.now().isoformat(),
            })

        print(f"[04_baseline_model] ✓ Saved to MongoDB pipeline_state + experiments")
        client.close()
    except Exception as e:
        print(f"[04_baseline_model] WARNING: MongoDB write failed: {e}")

# ── Block if AUC too low ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("BASELINE MODEL LEADERBOARD")
print("="*60)
print(f"{'Model':<25} {'CV AUC':>8} {'Test AUC':>9} {'Gini':>7} {'KS':>7} {'F1':>7} {'Time':>7}")
print("-"*60)
for r in leaderboard:
    marker = " ◄ BEST" if r["model"] == best["model"] else ""
    print(f"{r['model']:<25} {r['cv_auc_mean']:>8.4f} {r['test_auc']:>9.4f} {r['test_gini']:>7.4f} {r['test_ks']:>7.4f} {r['test_f1']:>7.4f} {r['train_time_s']:>6.1f}s{marker}")

if best["test_auc"] < MIN_AUC:
    print(f"\n[04_baseline_model] BLOCKED: AUC={best['test_auc']} < minimum {MIN_AUC}. Investigate data quality before proceeding.")
    sys.exit(2)

print(f"\n[04_baseline_model] COMPLETE — best model: {best['model']} AUC={best['test_auc']}")
