"""Credit Scoring Pipeline — Step 05: Error Analysis.

Reads:  test_scores DataFrame from MongoDB GridFS (written by step 04).
        test DataFrame from MongoDB GridFS (written by step 02, for segment columns).
        pipeline_state.01_eda (high_risk_segments — guides which segments to audit).
Writes: pipeline_state.05_error_analysis + findings to MongoDB.
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

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")

print(f"[05_error_analysis] investigation_id={INVESTIGATION_ID}")

# ── Load prior steps from MongoDB ────────────────────────────────────────────

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
from _pipeline_io import gfs_load

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
        print(f"[05_error_analysis] MongoDB load {step_name} failed: {e}")
        return {}

step04 = _load_step("04_baseline_model")
step02 = _load_step("02_preprocessing")
step01 = _load_step("01_eda")

if not step04:
    print("[05_error_analysis] ERROR: No step 04 state. Run step 04 first.")
    sys.exit(1)

# ── Load test data + scores ───────────────────────────────────────────────────

scores_df = gfs_load("test_scores", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
test_df   = gfs_load("test",        INVESTIGATION_ID, MONGODB_URI, DB_NAME)

if scores_df is None:
    p = step04.get("test_scores_path", "")
    if p and not p.startswith("gridfs://"):
        scores_df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    else:
        print("[05_error_analysis] ERROR: Cannot load test_scores.")
        sys.exit(1)

if test_df is None:
    p = step02.get("test_path", "")
    test_df = pd.DataFrame()
    if p and not p.startswith("gridfs://"):
        test_df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)

# Merge on index position (both are aligned slices of the same split)
if not test_df.empty and len(test_df) == len(scores_df):
    # Bring original categorical columns back for segment analysis
    cat_cols = test_df.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        scores_df[c] = test_df[c].values

overall_auc   = step04["best_auc"]
y_true        = scores_df[TARGET_COL].values
y_pred_proba  = scores_df["predict_proba"].values
y_pred_label  = scores_df["predicted_label"].values

print(f"[05_error_analysis] Test set: {len(scores_df)} rows, overall AUC={overall_auc}")

# ── 1. Segment-level AUC ─────────────────────────────────────────────────────

from sklearn.metrics import roc_auc_score, roc_curve

segment_aucs    = []
weak_segments   = []
AUC_GAP_THRESH  = 0.05  # flag if segment AUC is > 0.05 below overall

# Priority: segments flagged in EDA as high-risk + all object/low-cardinality cols
eda_seg_cols = list({s["column"] for s in step01.get("high_risk_segments", [])})
all_seg_cols = list(scores_df.select_dtypes(include="object").columns)
seg_cols_to_check = list(dict.fromkeys(eda_seg_cols + all_seg_cols))  # EDA first, deduped

for col in seg_cols_to_check:
    if col not in scores_df.columns or col == TARGET_COL:
        continue
    for val in scores_df[col].dropna().unique():
        mask = scores_df[col] == val
        n    = int(mask.sum())
        if n < 50:
            continue
        yt = y_true[mask]
        yp = y_pred_proba[mask]
        if yt.sum() < 5 or (yt == 0).sum() < 5:
            continue
        seg_auc = roc_auc_score(yt, yp)
        gap     = round(seg_auc - overall_auc, 4)
        entry   = {
            "column":  col,
            "value":   str(val),
            "auc":     round(seg_auc, 4),
            "gap":     gap,
            "n":       n,
            "pos_rate": round(float(yt.mean()), 4),
        }
        segment_aucs.append(entry)
        if gap < -AUC_GAP_THRESH:
            weak_segments.append(entry)

segment_aucs.sort(key=lambda x: x["gap"])
weak_segments.sort(key=lambda x: x["gap"])

print(f"[05_error_analysis] Segments evaluated: {len(segment_aucs)}, weak: {len(weak_segments)}")
if weak_segments:
    print(f"  Weakest: {weak_segments[0]['column']}={weak_segments[0]['value']} AUC={weak_segments[0]['auc']} (gap={weak_segments[0]['gap']})")

# ── 2. False negative / false positive analysis ───────────────────────────────

fn_mask  = (y_true == 1) & (y_pred_label == 0)  # missed defaults
fp_mask  = (y_true == 0) & (y_pred_label == 1)  # unnecessary flags
tn_mask  = (y_true == 0) & (y_pred_label == 0)
tp_mask  = (y_true == 1) & (y_pred_label == 1)

fn_rate  = float(fn_mask.sum() / max(y_true.sum(), 1))
fp_rate  = float(fp_mask.sum() / max((y_true == 0).sum(), 1))
tn_rate  = 1 - fp_rate
tp_rate  = 1 - fn_rate

print(f"[05_error_analysis] FN rate (missed defaults): {fn_rate:.2%}")
print(f"[05_error_analysis] FP rate (false alarms):    {fp_rate:.2%}")

# Score distribution of FN cases — where in the probability range are they?
fn_scores = y_pred_proba[fn_mask]
fp_scores = y_pred_proba[fp_mask]

fn_analysis = {
    "count":        int(fn_mask.sum()),
    "rate":         round(fn_rate, 4),
    "score_mean":   round(float(fn_scores.mean()), 4) if len(fn_scores) > 0 else 0,
    "score_median": round(float(np.median(fn_scores)), 4) if len(fn_scores) > 0 else 0,
    "score_max":    round(float(fn_scores.max()), 4) if len(fn_scores) > 0 else 0,
}
fp_analysis = {
    "count":        int(fp_mask.sum()),
    "rate":         round(fp_rate, 4),
    "score_mean":   round(float(fp_scores.mean()), 4) if len(fp_scores) > 0 else 0,
    "score_min":    round(float(fp_scores.min()), 4) if len(fp_scores) > 0 else 0,
}

# ── 3. Calibration by decile ─────────────────────────────────────────────────

scores_df["_decile"] = pd.qcut(y_pred_proba, q=10, labels=False, duplicates="drop")
cal_df = scores_df.groupby("_decile").agg(
    pred_mean=("predict_proba", "mean"),
    actual_rate=(TARGET_COL, "mean"),
    n=(TARGET_COL, "count"),
).reset_index()
cal_df["calibration_error"] = (cal_df["pred_mean"] - cal_df["actual_rate"]).abs()
mean_cal_error = float(cal_df["calibration_error"].mean())
calibration = cal_df[["_decile", "pred_mean", "actual_rate", "n", "calibration_error"]].to_dict(orient="records")

print(f"[05_error_analysis] Mean calibration error: {mean_cal_error:.4f}")

# ── 4. Bias / fairness flags ─────────────────────────────────────────────────

bias_flags = []
BIAS_FPR_RATIO = 2.0  # flag if group FPR > 2× overall

for col in seg_cols_to_check:
    if col not in scores_df.columns or col == TARGET_COL:
        continue
    for val in scores_df[col].dropna().unique():
        mask = (scores_df[col] == val) & (y_true == 0)
        n_neg = int(mask.sum())
        if n_neg < 30:
            continue
        group_fpr = float(y_pred_label[mask].mean())
        ratio     = group_fpr / max(fp_rate, 1e-6)
        if ratio > BIAS_FPR_RATIO:
            bias_flags.append({
                "group":          f"{col}={val}",
                "fpr":            round(group_fpr, 4),
                "ratio_vs_overall": round(ratio, 2),
                "n_negative":     n_neg,
            })

bias_flags.sort(key=lambda x: x["ratio_vs_overall"], reverse=True)
print(f"[05_error_analysis] Bias flags: {len(bias_flags)}")

# ── 5. Improvement targets ────────────────────────────────────────────────────

improvement_targets = [f"{s['column']}={s['value']}" for s in weak_segments[:5]]

# ── 6. Persist to MongoDB ─────────────────────────────────────────────────────

result = {
    "step":              "05_error_analysis",
    "investigation_id":  INVESTIGATION_ID,
    "overall_auc":       overall_auc,
    "segment_aucs":      segment_aucs[:30],
    "weak_segments":     weak_segments,
    "fn_analysis":       fn_analysis,
    "fp_analysis":       fp_analysis,
    "false_negative_rate": fn_rate,
    "false_positive_rate": fp_rate,
    "calibration_error": round(mean_cal_error, 4),
    "calibration":       calibration,
    "bias_flags":        bias_flags,
    "improvement_targets": improvement_targets,
    "completed_at":      datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]

        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "05_error_analysis"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "credit-scoring-pipeline",
             "step": "05_error_analysis", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )

        # Write findings for weak segments
        for seg in weak_segments[:5]:
            db["findings"].insert_one({
                "agent":            "validation-governance-agent",
                "type":             "weak_segment",
                "severity":         "WARNING",
                "content":          f"Weak model performance on {seg['column']}={seg['value']}: AUC={seg['auc']} (gap={seg['gap']:+.3f} vs overall {overall_auc}), n={seg['n']}",
                "investigation_id": INVESTIGATION_ID,
                "timestamp":        datetime.now().isoformat(),
            })

        # Write findings for bias flags
        for flag in bias_flags[:3]:
            db["findings"].insert_one({
                "agent":            "validation-governance-agent",
                "type":             "fairness_bias",
                "severity":         "WARNING" if flag["ratio_vs_overall"] < 3.0 else "CRITICAL",
                "content":          f"Bias detected: group {flag['group']} has FPR={flag['fpr']:.2%} which is {flag['ratio_vs_overall']}× the overall FPR. n_negative={flag['n_negative']}",
                "investigation_id": INVESTIGATION_ID,
                "timestamp":        datetime.now().isoformat(),
            })

        print(f"[05_error_analysis] ✓ Saved to MongoDB pipeline_state + findings")
        client.close()
    except Exception as e:
        print(f"[05_error_analysis] WARNING: MongoDB write failed: {e}")

# ── Charts: segment AUC gaps + score distributions ────────────────────────────

from _pipeline_io import publish_artifact

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # 1. Segment AUC gap — worst 12 segments
    worst = segment_aucs[:12][::-1]
    if worst:
        labels = [f"{s['column']}={s['value']}"[:34] for s in worst]
        gaps   = [s["gap"] for s in worst]
        axes[0].barh(labels, gaps,
                     color=["#A85751" if g < -AUC_GAP_THRESH else "#9DB8B3" for g in gaps])
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_title("Segment AUC Gap vs Overall (worst 12)")
        axes[0].set_xlabel("AUC gap")
        axes[0].tick_params(labelsize=8)

    # 2. Score distributions by true class
    axes[1].hist(y_pred_proba[y_true == 0], bins=40, alpha=0.65, label="negative", color="#9DB8B3")
    axes[1].hist(y_pred_proba[y_true == 1], bins=40, alpha=0.65, label="positive", color="#2F6868")
    axes[1].set_title("Predicted Score Distribution by True Class")
    axes[1].set_xlabel("predicted probability")
    axes[1].legend()

    fig.tight_layout()
    seg_chart = ARTIFACT_DIR / "error_analysis.png"
    fig.savefig(seg_chart, dpi=130)
    plt.close(fig)
    publish_artifact(seg_chart, INVESTIGATION_ID, MONGODB_URI, DB_NAME,
                     kind="image", title="Error & Segment Analysis", step="05_error_analysis")
except Exception as e:
    print(f"[05_error_analysis] chart generation failed (non-fatal): {e}")

# ── Print summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ERROR ANALYSIS SUMMARY")
print("="*60)
print(f"Overall AUC:            {overall_auc:.4f}")
print(f"False Negative Rate:    {fn_rate:.2%}  (missed defaults)")
print(f"False Positive Rate:    {fp_rate:.2%}  (false alarms)")
print(f"Calibration Error:      {mean_cal_error:.4f}")
print(f"\nWeakest segments (AUC gap > -{AUC_GAP_THRESH}):")
for s in weak_segments[:5]:
    print(f"  {s['column']}={s['value']:20s}  AUC={s['auc']:.4f}  gap={s['gap']:+.4f}  n={s['n']}")
print(f"\nBias flags ({len(bias_flags)} groups with FPR > {BIAS_FPR_RATIO}× overall):")
for b in bias_flags[:3]:
    print(f"  {b['group']:35s}  FPR={b['fpr']:.2%}  ratio={b['ratio_vs_overall']}×")
print(f"\nImprovement targets for step 06: {improvement_targets}")
print(f"\n[05_error_analysis] COMPLETE")
