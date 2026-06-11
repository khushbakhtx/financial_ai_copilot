"""Credit Scoring Pipeline — Step 06: Iterative Training.

Reads:  train_fe + test_fe DataFrames from MongoDB GridFS (written by step 03).
        test + train DataFrames from MongoDB GridFS (written by step 02, for segment weights).
        pipeline_state.05_error_analysis (weak_segments, improvement_targets).
Writes: best_model_v2.pkl to artifacts dir (binary, stays on disk).
        test_scores_v2 DataFrame to MongoDB GridFS.
        pipeline_state.06_iterative_training + experiments to MongoDB.
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
MAX_ITERATIONS   = int(os.environ.get("MAX_ITERATIONS", "3"))
MIN_AUC_GAIN     = float(os.environ.get("MIN_AUC_GAIN", "0.005"))
OPTUNA_TRIALS    = int(os.environ.get("OPTUNA_TRIALS", "30"))

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[06_iterative_training] investigation_id={INVESTIGATION_ID}")

# ── Load prior steps from MongoDB ────────────────────────────────────────────

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
from _pipeline_io import gfs_save, gfs_load

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
        print(f"[06_iterative_training] MongoDB load {step_name} failed: {e}")
        return {}

step05 = _load_step("05_error_analysis")
step03 = _load_step("03_feature_engineering")
step04 = _load_step("04_baseline_model")

if not step05 or not step03:
    print("[06_iterative_training] ERROR: Missing prior step state. Run steps 03-05 first.")
    sys.exit(1)

weak_segments      = step05.get("weak_segments", [])
improvement_targets = step05.get("improvement_targets", [])
baseline_auc       = step05.get("overall_auc", step04.get("best_auc", 0.0))

df_train = gfs_load("train_fe", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
df_test  = gfs_load("test_fe",  INVESTIGATION_ID, MONGODB_URI, DB_NAME)

if df_train is None or df_test is None:
    train_fe_path = step03.get("train_fe_path", "")
    test_fe_path  = step03.get("test_fe_path", "")
    if train_fe_path and not train_fe_path.startswith("gridfs://"):
        df_train = pd.read_parquet(train_fe_path) if train_fe_path.endswith(".parquet") else pd.read_csv(train_fe_path)
        df_test  = pd.read_parquet(test_fe_path)  if test_fe_path.endswith(".parquet")  else pd.read_csv(test_fe_path)
    else:
        print("[06_iterative_training] ERROR: Could not load train_fe/test_fe.")
        sys.exit(1)

feature_cols = [c for c in df_train.columns if c != TARGET_COL]
X_train_base = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
y_train      = df_train[TARGET_COL].values
X_test       = df_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
y_test       = df_test[TARGET_COL].values

scale_pos = float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1))

print(f"[06_iterative_training] Baseline AUC: {baseline_auc}")
print(f"[06_iterative_training] Weak segments to improve: {len(weak_segments)}")
print(f"[06_iterative_training] Max iterations: {MAX_ITERATIONS}, Optuna trials: {OPTUNA_TRIALS}")

# ── Build sample weights for weak segments ────────────────────────────────────

# Load original (pre-feature-engineering) train/test from GridFS for segment weight lookups
step02 = _load_step("02_preprocessing")
test_raw_df     = gfs_load("test",  INVESTIGATION_ID, MONGODB_URI, DB_NAME) or pd.DataFrame()
step02_train_df = gfs_load("train", INVESTIGATION_ID, MONGODB_URI, DB_NAME) or pd.DataFrame()

def _build_sample_weights(df_subset, weak_segs, cat_lookup_df):
    """Build sample weight array: upweight rows in weak segments by 2×."""
    weights = np.ones(len(df_subset))
    if not weak_segs or cat_lookup_df.empty or len(cat_lookup_df) != len(df_subset):
        return weights
    for seg in weak_segs[:5]:
        col, val = seg["column"], seg["value"]
        if col not in cat_lookup_df.columns:
            continue
        mask = cat_lookup_df[col].astype(str) == str(val)
        weights[mask.values] *= 2.0
    return weights

sample_weights = _build_sample_weights(df_train, weak_segments, step02_train_df)
print(f"[06_iterative_training] Sample weights — upweighted {(sample_weights > 1).sum()} rows")

# ── Optuna hyperparameter tuning ──────────────────────────────────────────────

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 500),
            "max_depth":          trial.suggest_int("max_depth", 3, 9),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
            "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 3.0),
            "scale_pos_weight":   scale_pos,
            "random_state":       42,
            "eval_metric":        "auc",
            "verbosity":          0,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train_base, y_train, cv=cv, scoring="roc_auc",
                                 fit_params={"sample_weight": sample_weights})
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos
    best_params["random_state"]     = 42
    best_params["eval_metric"]      = "auc"
    best_params["verbosity"]        = 0
    print(f"[06_iterative_training] Optuna best CV AUC: {study.best_value:.4f}")
    print(f"[06_iterative_training] Best params: {best_params}")
except ImportError:
    print("[06_iterative_training] Optuna not available — using strong defaults")
    best_params = {
        "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
        "scale_pos_weight": scale_pos, "random_state": 42,
        "eval_metric": "auc", "verbosity": 0,
    }

# ── Iterative training loop ───────────────────────────────────────────────────

from sklearn.metrics import roc_curve

auc_progression   = [baseline_auc]
best_model_v2     = None
best_auc_v2       = baseline_auc
best_iteration    = 0

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n[06_iterative_training] Iteration {iteration}/{MAX_ITERATIONS}...")

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_base, y_train, sample_weight=sample_weights, verbose=False)

    y_pred = model.predict_proba(X_test)[:, 1]
    iter_auc = roc_auc_score(y_test, y_pred)
    print(f"  Iteration {iteration} AUC: {iter_auc:.4f} (baseline: {baseline_auc:.4f})")

    auc_progression.append(round(iter_auc, 4))

    if iter_auc > best_auc_v2:
        best_auc_v2   = iter_auc
        best_model_v2 = model
        best_iteration = iteration

    # Stop if no meaningful gain
    if len(auc_progression) >= 3:
        recent_gain = auc_progression[-1] - auc_progression[-2]
        if abs(recent_gain) < MIN_AUC_GAIN:
            print(f"  Early stopping: gain={recent_gain:.4f} < {MIN_AUC_GAIN}")
            break

    # For next iteration: increase weights on still-weak segments
    sample_weights = _build_sample_weights(df_train, weak_segments, step02_train_df)
    sample_weights *= (1.0 + 0.5 * iteration)  # progressive upweighting

# Fallback if model didn't improve
if best_model_v2 is None:
    print("[06_iterative_training] No improvement over baseline — using default params model")
    best_model_v2 = xgb.XGBClassifier(**best_params)
    best_model_v2.fit(X_train_base, y_train, sample_weight=_build_sample_weights(df_train, weak_segments, step02_train_df))
    y_pred = best_model_v2.predict_proba(X_test)[:, 1]
    best_auc_v2 = roc_auc_score(y_test, y_pred)

# ── Final metrics ─────────────────────────────────────────────────────────────

y_pred_final = best_model_v2.predict_proba(X_test)[:, 1]
final_auc    = roc_auc_score(y_test, y_pred_final)
final_gini   = 2 * final_auc - 1
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_pred_final)
final_ks     = float(np.max(tpr_arr - fpr_arr))

# Per-segment improvement check (using test_raw_df if available)
segment_improvement = []
if not test_raw_df.empty and len(test_raw_df) == len(y_test):
    for seg in weak_segments[:5]:
        col, val = seg["column"], seg["value"]
        if col not in test_raw_df.columns:
            continue
        mask = test_raw_df[col].astype(str) == str(val)
        if mask.sum() < 30:
            continue
        yt_seg = y_test[mask.values]
        yp_seg = y_pred_final[mask.values]
        if yt_seg.sum() >= 5 and (yt_seg == 0).sum() >= 5:
            seg_auc_after = roc_auc_score(yt_seg, yp_seg)
            segment_improvement.append({
                "segment":    f"{col}={val}",
                "auc_before": seg["auc"],
                "auc_after":  round(seg_auc_after, 4),
                "delta":      round(seg_auc_after - seg["auc"], 4),
            })

print(f"\n[06_iterative_training] Final AUC: {final_auc:.4f} (baseline: {baseline_auc:.4f}, Δ={final_auc - baseline_auc:+.4f})")

# ── Save final model + scores ─────────────────────────────────────────────────

final_model_path = str(ARTIFACT_DIR / "best_model_v2.pkl")
with open(final_model_path, "wb") as f:
    pickle.dump({
        "model":        best_model_v2,
        "feature_cols": feature_cols,
        "model_name":   "xgboost_tuned",
        "best_params":  best_params,
        "iteration":    best_iteration,
    }, f)
print(f"[06_iterative_training] Saved best_model_v2.pkl → {final_model_path}")

scores_v2 = df_test[[TARGET_COL]].copy()
scores_v2["predict_proba"]   = y_pred_final
scores_v2["predicted_label"] = (y_pred_final >= 0.5).astype(int)

if MONGODB_URI:
    gfs_save(scores_v2, "test_scores_v2", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    final_scores_path = f"gridfs://{INVESTIGATION_ID}/test_scores_v2"
    print(f"[06_iterative_training] Saved test_scores_v2 → GridFS")
else:
    final_scores_path = str(ARTIFACT_DIR / "test_scores_v2.parquet")
    scores_v2.to_parquet(final_scores_path, index=False)
    print(f"[06_iterative_training] Saved test_scores_v2 → {final_scores_path} (local fallback)")

# ── Persist to MongoDB ────────────────────────────────────────────────────────

result = {
    "step":                "06_iterative_training",
    "investigation_id":    INVESTIGATION_ID,
    "final_model_path":    final_model_path,
    "final_scores_path":   final_scores_path,
    "feature_cols":        feature_cols,
    "iterations_run":      len(auc_progression) - 1,
    "best_iteration":      best_iteration,
    "auc_progression":     auc_progression,
    "baseline_auc":        baseline_auc,
    "final_auc":           round(final_auc, 4),
    "final_gini":          round(final_gini, 4),
    "final_ks":            round(final_ks, 4),
    "auc_delta":           round(final_auc - baseline_auc, 4),
    "best_params":         best_params,
    "segment_improvement": segment_improvement,
    "completed_at":        datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]

        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "06_iterative_training"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "credit-scoring-pipeline",
             "step": "06_iterative_training", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )

        db["experiments"].insert_one({
            "investigation_id": INVESTIGATION_ID,
            "experiment_name":  f"iterative_v2_{INVESTIGATION_ID}",
            "model_name":       "xgboost_tuned",
            "metrics": {
                "auc":   round(final_auc, 4),
                "gini":  round(final_gini, 4),
                "ks":    round(final_ks, 4),
                "auc_delta": round(final_auc - baseline_auc, 4),
            },
            "parameters": best_params,
            "notes":      f"Iterative training with weak-segment upweighting + Optuna tuning. {len(auc_progression)-1} iterations.",
            "timestamp":  datetime.now().isoformat(),
        })

        print(f"[06_iterative_training] ✓ Saved to MongoDB pipeline_state + experiments")
        client.close()
    except Exception as e:
        print(f"[06_iterative_training] WARNING: MongoDB write failed: {e}")

print("\n" + "="*60)
print("ITERATIVE TRAINING SUMMARY")
print("="*60)
print(f"Baseline AUC:  {baseline_auc:.4f}")
print(f"Final AUC:     {final_auc:.4f}  (Δ={final_auc - baseline_auc:+.4f})")
print(f"Final Gini:    {final_gini:.4f}")
print(f"Final KS:      {final_ks:.4f}")
print(f"AUC progress:  {' → '.join(str(a) for a in auc_progression)}")
if segment_improvement:
    print(f"\nSegment improvements:")
    for s in segment_improvement:
        print(f"  {s['segment']:35s} {s['auc_before']:.4f} → {s['auc_after']:.4f} (Δ={s['delta']:+.4f})")
print(f"\n[06_iterative_training] COMPLETE")
