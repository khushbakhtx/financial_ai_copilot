"""Credit Scoring Pipeline — Step 01: EDA & Segment Analysis.

Reads: DATASET_PATH, TARGET_COL, INVESTIGATION_ID from environment.
Writes: pipeline_state.01_eda to MongoDB via pymongo.
Also writes findings for CRITICAL leakage flags and high-risk segments.
"""

import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

DATASET_PATH    = os.environ["DATASET_PATH"]
TARGET_COL      = os.environ["TARGET_COL"]
INVESTIGATION_ID = os.environ["INVESTIGATION_ID"]
MONGODB_URI     = os.environ.get("MONGODB_URI", "")
DB_NAME         = os.environ.get("MONGODB_DB", "financial_ai_copilot")

print(f"[01_eda] investigation_id={INVESTIGATION_ID}")
print(f"[01_eda] dataset={DATASET_PATH}, target={TARGET_COL}")

# ── Load dataset ──────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np

if DATASET_PATH.endswith(".parquet"):
    df = pd.read_parquet(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH)

print(f"[01_eda] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

if TARGET_COL not in df.columns:
    print(f"[01_eda] ERROR: target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")
    sys.exit(1)

# ── 1. Structural scan ────────────────────────────────────────────────────────

null_rates  = (df.isnull().sum() / len(df)).round(4).to_dict()
dtypes_map  = {c: str(t) for c, t in df.dtypes.items()}
target_rate = float(df[TARGET_COL].mean())
n_pos       = int(df[TARGET_COL].sum())
n_neg       = len(df) - n_pos
imbalance   = round(n_neg / n_pos, 2) if n_pos > 0 else float("inf")

# Identify likely identifiers (high cardinality, no predictive value)
id_candidates = [
    c for c in df.columns
    if c != TARGET_COL and (
        df[c].nunique() == len(df)
        or any(tok in c.lower() for tok in ("id", "uuid", "key", "hash", "index", "no_"))
    )
]

print(f"[01_eda] target_rate={target_rate:.4f}, imbalance={imbalance}:1")
print(f"[01_eda] Likely identifiers: {id_candidates}")

# ── 2. Quick feature importance (RandomForest) ────────────────────────────────

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

drop_for_model = [TARGET_COL] + id_candidates
feature_cols   = [c for c in df.columns if c not in drop_for_model]

X_raw = df[feature_cols].copy()

# Encode categoricals + fill nulls for RF
for col in X_raw.select_dtypes(include="object").columns:
    X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))
for col in X_raw.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
    X_raw[col] = X_raw[col].astype(np.int64) // 10**9  # epoch seconds

X_raw = X_raw.fillna(-999).replace([np.inf, -np.inf], -999)
y     = df[TARGET_COL].values

sample_n = min(30_000, len(X_raw))
idx      = np.random.default_rng(42).choice(len(X_raw), sample_n, replace=False)
X_sample = X_raw.iloc[idx]
y_sample = y[idx]

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_sample, y_sample)

importance_df = pd.DataFrame({
    "feature":    feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

quick_importance = importance_df.head(30).to_dict(orient="records")
leakage_flags    = importance_df[importance_df["importance"] > 0.90]["feature"].tolist()
dominance_check  = importance_df.head(2)
dominance_gap    = float(dominance_check.iloc[0]["importance"] - dominance_check.iloc[1]["importance"]) if len(dominance_check) >= 2 else 0.0
dominance_flags  = []
if dominance_check.iloc[0]["importance"] > 0.55 and dominance_gap > 0.20:
    dominance_flags = [dominance_check.iloc[0]["feature"]]

print(f"[01_eda] Top features: {importance_df.head(5)[['feature','importance']].values.tolist()}")
print(f"[01_eda] Leakage flags (>0.90): {leakage_flags}")
print(f"[01_eda] Dominance flags: {dominance_flags}")

# ── 3. Segment risk analysis ──────────────────────────────────────────────────

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET_COL and df[c].nunique() <= 50]

high_risk_segments = []
for col in cat_cols:
    seg = (
        df.groupby(col)[TARGET_COL]
        .agg(default_rate="mean", n="count", n_default="sum")
        .reset_index()
    )
    seg["lift"] = (seg["default_rate"] / target_rate).round(3)
    for _, row in seg.iterrows():
        if row["lift"] >= 2.0 and row["n"] >= 30:
            high_risk_segments.append({
                "column":       col,
                "value":        str(row[col]),
                "default_rate": round(float(row["default_rate"]), 4),
                "lift":         float(row["lift"]),
                "n":            int(row["n"]),
            })

high_risk_segments.sort(key=lambda x: x["lift"], reverse=True)
high_risk_segments = high_risk_segments[:20]  # top 20 only
print(f"[01_eda] High-risk segments found: {len(high_risk_segments)}")

# ── 4. Temporal pattern (if date column present) ──────────────────────────────

temporal_pattern = {}
date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
if not date_cols:
    date_candidates = [c for c in df.columns if any(tok in c.lower() for tok in ("date", "time", "period", "month", "year"))]
    for dc in date_candidates:
        try:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")
            if df[dc].notna().sum() > len(df) * 0.5:
                date_cols.append(dc)
                break
        except Exception:
            pass

if date_cols:
    dc = date_cols[0]
    df["_month"] = pd.to_datetime(df[dc], errors="coerce").dt.to_period("M")
    monthly = df.groupby("_month")[TARGET_COL].agg(["mean", "count"]).reset_index()
    monthly.columns = ["month", "default_rate", "n"]
    temporal_pattern = {
        "date_column": dc,
        "monthly": [
            {"month": str(r["month"]), "default_rate": round(float(r["default_rate"]), 4), "n": int(r["n"])}
            for _, r in monthly.iterrows()
        ],
    }
    df.drop(columns=["_month"], inplace=True)
    print(f"[01_eda] Temporal pattern extracted from '{dc}'")

# ── 5. Deterministic zones ────────────────────────────────────────────────────

num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != TARGET_COL]
deterministic_zones = []

for col in num_cols[:15]:  # check top 15 numeric cols
    col_data = df[[col, TARGET_COL]].dropna()
    if len(col_data) < 100:
        continue
    # decile analysis
    col_data["_decile"] = pd.qcut(col_data[col], q=10, duplicates="drop", labels=False)
    dec_stats = col_data.groupby("_decile")[TARGET_COL].agg(["mean", "count"]).reset_index()
    extremes  = dec_stats[dec_stats["mean"] > 0.85]
    for _, row in extremes.iterrows():
        bounds = col_data[col_data["_decile"] == row["_decile"]][col]
        deterministic_zones.append({
            "column":       col,
            "range_min":    round(float(bounds.min()), 4),
            "range_max":    round(float(bounds.max()), 4),
            "default_rate": round(float(row["mean"]), 4),
            "n":            int(row["count"]),
        })

print(f"[01_eda] Deterministic zones found: {len(deterministic_zones)}")

# ── 6. Informative missingness ────────────────────────────────────────────────

informative_missingness = []
for col in df.columns:
    if col in (TARGET_COL,) + tuple(id_candidates):
        continue
    miss_mask = df[col].isnull()
    n_miss    = int(miss_mask.sum())
    if n_miss < 10 or n_miss > len(df) * 0.95:
        continue
    rate_miss    = float(df.loc[miss_mask, TARGET_COL].mean())
    rate_present = float(df.loc[~miss_mask, TARGET_COL].mean())
    lift         = rate_miss / target_rate if target_rate > 0 else 1.0
    if abs(lift - 1.0) > 0.3:
        informative_missingness.append({
            "column":          col,
            "n_missing":       n_miss,
            "missing_pct":     round(n_miss / len(df), 4),
            "default_rate_if_missing":  round(rate_miss, 4),
            "default_rate_if_present":  round(rate_present, 4),
            "lift":            round(float(lift), 3),
        })

informative_missingness.sort(key=lambda x: abs(x["lift"] - 1.0), reverse=True)
print(f"[01_eda] Informative missingness cols: {len(informative_missingness)}")

# ── 7. Assemble result dict ───────────────────────────────────────────────────

result = {
    "step":              "01_eda",
    "investigation_id":  INVESTIGATION_ID,
    "dataset_path":      DATASET_PATH,
    "shape":             list(df.shape),
    "target_col":        TARGET_COL,
    "target_rate":       round(target_rate, 4),
    "n_positive":        n_pos,
    "n_negative":        n_neg,
    "imbalance_ratio":   imbalance,
    "columns":           list(df.columns),
    "dtypes":            dtypes_map,
    "null_rates":        null_rates,
    "id_candidates":     id_candidates,
    "quick_importance":  quick_importance,
    "leakage_flags":     leakage_flags,
    "dominance_flags":   dominance_flags,
    "high_risk_segments": high_risk_segments,
    "temporal_pattern":  temporal_pattern,
    "deterministic_zones": deterministic_zones,
    "informative_missingness": informative_missingness,
    "completed_at":      datetime.now().isoformat(),
}

# ── 8. Persist to MongoDB pipeline_state ─────────────────────────────────────

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db     = client[DB_NAME]
        col    = db["pipeline_state"]
        col.replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "01_eda"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "credit-scoring-pipeline",
             "step": "01_eda", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        print(f"[01_eda] ✓ Saved to MongoDB pipeline_state")

        # Write findings for leakage flags
        for feat in leakage_flags:
            db["findings"].insert_one({
                "agent":            "data-profiling-agent",
                "type":             "leakage_risk",
                "severity":         "CRITICAL",
                "content":          f"Feature '{feat}' has importance > 0.90 — likely data leakage. Exclude before training.",
                "investigation_id": INVESTIGATION_ID,
                "timestamp":        datetime.now().isoformat(),
            })

        # Write findings for dominance flags
        for feat in dominance_flags:
            db["findings"].insert_one({
                "agent":            "data-profiling-agent",
                "type":             "dominance_risk",
                "severity":         "WARNING",
                "content":          f"Feature '{feat}' dominates (importance > 0.55, gap > 0.20). Possible proxy for target.",
                "investigation_id": INVESTIGATION_ID,
                "timestamp":        datetime.now().isoformat(),
            })

        # Write findings for top high-risk segments
        for seg in high_risk_segments[:5]:
            db["findings"].insert_one({
                "agent":            "data-profiling-agent",
                "type":             "high_risk_segment",
                "severity":         "WARNING" if seg["lift"] < 4.0 else "CRITICAL",
                "content":          f"Segment {seg['column']}={seg['value']}: default_rate={seg['default_rate']:.1%}, lift={seg['lift']}×, n={seg['n']}",
                "investigation_id": INVESTIGATION_ID,
                "timestamp":        datetime.now().isoformat(),
            })

        print(f"[01_eda] ✓ Findings written to MongoDB")
        client.close()
    except Exception as e:
        print(f"[01_eda] WARNING: MongoDB write failed: {e}")
else:
    print("[01_eda] WARNING: MONGODB_URI not set — results not persisted to MongoDB")

# ── 8b. Charts: target balance + quick feature importance ────────────────────

try:
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent))
    from _pipeline_io import publish_artifact

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _artifact_dir = _P(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
    _artifact_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(["negative", "positive"], [n_neg, n_pos], color=["#9DB8B3", "#2F6868"])
    axes[0].set_title(f"Target Balance — '{TARGET_COL}' ({target_rate:.2%} positive)")
    for i, v in enumerate([n_neg, n_pos]):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    _top_imp = importance_df.head(15).iloc[::-1]
    axes[1].barh(_top_imp["feature"], _top_imp["importance"], color="#2F6868")
    axes[1].set_title("Quick Feature Importance (top 15)")
    fig.tight_layout()
    _eda_chart = _artifact_dir / "eda_overview.png"
    fig.savefig(_eda_chart, dpi=130)
    plt.close(fig)
    publish_artifact(_eda_chart, INVESTIGATION_ID, MONGODB_URI, DB_NAME,
                     kind="image", title="EDA Overview", step="01_eda")
except Exception as e:
    print(f"[01_eda] chart generation failed (non-fatal): {e}")

# ── 9. Print summary ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("EDA SUMMARY")
print("="*60)
print(f"Dataset:        {DATASET_PATH}")
print(f"Shape:          {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"Target:         '{TARGET_COL}' — {target_rate:.2%} positive, {imbalance}:1 ratio")
print(f"Leakage flags:  {leakage_flags}")
print(f"Dominance:      {dominance_flags}")
print(f"\nTop 10 features by importance:")
for row in quick_importance[:10]:
    bar = "█" * int(row["importance"] * 30)
    print(f"  {row['feature']:40s} {row['importance']:.4f} {bar}")
print(f"\nTop 5 high-risk segments:")
for seg in high_risk_segments[:5]:
    print(f"  {seg['column']}={seg['value']:20s} rate={seg['default_rate']:.2%}  lift={seg['lift']}×  n={seg['n']}")
print(f"\nDeterministic zones:      {len(deterministic_zones)}")
print(f"Informative missing cols: {len(informative_missingness)}")
print(f"\n[01_eda] COMPLETE — investigation_id: {INVESTIGATION_ID}")
