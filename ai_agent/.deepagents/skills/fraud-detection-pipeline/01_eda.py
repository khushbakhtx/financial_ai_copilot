"""Fraud Detection Pipeline — Step 01: EDA & Fraud Profiling.

Reads: DATASET_PATH, TARGET_COL (optional), INVESTIGATION_ID from environment.
Writes: pipeline_state.01_eda to MongoDB.
        findings for leakage flags + high-velocity patterns.
"""

import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

DATASET_PATH     = os.environ["DATASET_PATH"]
TARGET_COL       = os.environ.get("TARGET_COL", "")
INVESTIGATION_ID = os.environ["INVESTIGATION_ID"]
MONGODB_URI      = os.environ.get("MONGODB_URI", "")
DB_NAME          = os.environ.get("MONGODB_DB", "financial_ai_copilot")

print(f"[fraud/01_eda] investigation_id={INVESTIGATION_ID}")
print(f"[fraud/01_eda] dataset={DATASET_PATH}, target={TARGET_COL or '(none — unsupervised)'}")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── Load dataset ──────────────────────────────────────────────────────────────

df = pd.read_parquet(DATASET_PATH) if DATASET_PATH.endswith(".parquet") else pd.read_csv(DATASET_PATH)
print(f"[fraud/01_eda] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── Auto-detect target column if not supplied ─────────────────────────────────

if not TARGET_COL:
    for candidate in ("is_fraud", "fraud", "fraud_label", "label", "target", "is_fraudulent"):
        if candidate in df.columns:
            TARGET_COL = candidate
            print(f"[fraud/01_eda] Auto-detected target: '{TARGET_COL}'")
            break

has_label  = bool(TARGET_COL and TARGET_COL in df.columns)
fraud_rate = float(df[TARGET_COL].mean()) if has_label else None
n_fraud    = int(df[TARGET_COL].sum())    if has_label else None
imbalance  = round((len(df) - n_fraud) / max(n_fraud, 1), 1) if has_label and n_fraud else None

if has_label:
    print(f"[fraud/01_eda] fraud_rate={fraud_rate:.3%}, imbalance={imbalance}:1")
else:
    print("[fraud/01_eda] No label column — unsupervised profiling only")

# ── Structural scan ───────────────────────────────────────────────────────────

null_rates     = (df.isnull().sum() / len(df)).round(4).to_dict()
dtypes_map     = {c: str(t) for c, t in df.dtypes.items()}
id_candidates  = [
    c for c in df.columns
    if df[c].nunique() == len(df) or any(t in c.lower() for t in ("_id", "uuid", "hash", "key", "index"))
]

# ── Amount statistics ─────────────────────────────────────────────────────────

amount_cols = [c for c in df.columns if any(t in c.lower() for t in ("amount", "value", "amt", "sum"))]
amount_stats = {}
for col in amount_cols[:3]:
    s = df[col].dropna()
    amount_stats[col] = {
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "p95": round(float(s.quantile(0.95)), 2),
        "p99": round(float(s.quantile(0.99)), 2),
        "max": round(float(s.max()), 2),
    }
    if has_label and amount_cols:
        fraud_mask  = df[TARGET_COL] == 1
        legit_mask  = df[TARGET_COL] == 0
        amount_stats[col]["fraud_mean"]  = round(float(df.loc[fraud_mask, col].mean()), 2)
        amount_stats[col]["legit_mean"]  = round(float(df.loc[legit_mask, col].mean()), 2)

# ── Temporal analysis ─────────────────────────────────────────────────────────

temporal_patterns = {}
date_cols = []
for c in df.columns:
    if any(t in c.lower() for t in ("date", "time", "timestamp", "created", "period")):
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            if df[c].notna().sum() > len(df) * 0.5:
                date_cols.append(c)
        except Exception:
            pass

if date_cols and has_label:
    dc = date_cols[0]
    df["_hour"] = df[dc].dt.hour
    df["_dow"]  = df[dc].dt.dayofweek
    df["_month"] = df[dc].dt.to_period("M")

    hour_fraud  = df.groupby("_hour")[TARGET_COL].agg(["mean", "count"]).reset_index()
    dow_fraud   = df.groupby("_dow")[TARGET_COL].agg(["mean", "count"]).reset_index()
    month_fraud = df.groupby("_month")[TARGET_COL].agg(["mean", "count"]).reset_index()

    temporal_patterns = {
        "date_column": dc,
        "peak_hour":   int(hour_fraud.loc[hour_fraud["mean"].idxmax(), "_hour"]),
        "peak_dow":    int(dow_fraud.loc[dow_fraud["mean"].idxmax(), "_dow"]),
        "hourly":  [{
            "hour": int(r["_hour"]),
            "fraud_rate": round(float(r["mean"]), 4),
            "n": int(r["count"])
        } for _, r in hour_fraud.iterrows()],
        "monthly": [{
            "month": str(r["_month"]),
            "fraud_rate": round(float(r["mean"]), 4),
            "n": int(r["count"])
        } for _, r in month_fraud.iterrows()],
    }
    for col in ["_hour", "_dow", "_month"]:
        df.drop(columns=[col], inplace=True, errors="ignore")

# ── Category fraud rates ──────────────────────────────────────────────────────

category_fraud_rates = []
if has_label:
    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c != TARGET_COL and df[c].nunique() <= 50]
    for col in cat_cols:
        seg = df.groupby(col)[TARGET_COL].agg(fraud_rate="mean", n="count").reset_index()
        seg["lift"] = (seg["fraud_rate"] / fraud_rate).round(3)
        for _, row in seg[seg["lift"] >= 2.0].iterrows():
            if row["n"] >= 20:
                category_fraud_rates.append({
                    "column": col,
                    "value":  str(row[col]),
                    "fraud_rate": round(float(row["fraud_rate"]), 4),
                    "lift":  float(row["lift"]),
                    "n":     int(row["n"]),
                })
    category_fraud_rates.sort(key=lambda x: x["lift"], reverse=True)

# ── Velocity flags ────────────────────────────────────────────────────────────

velocity_flags = []
cust_cols = [c for c in df.columns if any(t in c.lower() for t in ("customer", "account", "user", "client", "card"))]

if cust_cols and date_cols:
    cc = cust_cols[0]
    dc = date_cols[0]
    try:
        tx_per_cust = df.groupby(cc).size()
        mean_tx, std_tx = tx_per_cust.mean(), tx_per_cust.std()
        high_vel = tx_per_cust[tx_per_cust > mean_tx + 3 * std_tx]
        velocity_flags = [{"entity": str(k), "tx_count": int(v)} for k, v in high_vel.head(20).items()]
        print(f"[fraud/01_eda] High-velocity entities: {len(velocity_flags)}")
    except Exception:
        pass

# ── Quick feature importance (if label) ──────────────────────────────────────

quick_importance = []
leakage_flags    = []
if has_label:
    drop_for_model = [TARGET_COL] + id_candidates + date_cols
    feat_cols = [c for c in df.columns if c not in drop_for_model]
    X_raw = df[feat_cols].copy()
    for col in X_raw.select_dtypes(include="object").columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))
    X_raw = X_raw.fillna(-999).replace([np.inf, -np.inf], -999)
    y = df[TARGET_COL].values
    sample_n = min(20_000, len(X_raw))
    idx = np.random.default_rng(42).choice(len(X_raw), sample_n, replace=False)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_raw.iloc[idx], y[idx])
    imp_df = pd.DataFrame({"feature": feat_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    quick_importance = imp_df.head(20).to_dict(orient="records")
    leakage_flags    = imp_df[imp_df["importance"] > 0.90]["feature"].tolist()
    print(f"[fraud/01_eda] Leakage flags: {leakage_flags}")

# ── Assemble + persist ────────────────────────────────────────────────────────

result = {
    "step":              "01_eda",
    "investigation_id":  INVESTIGATION_ID,
    "dataset_path":      DATASET_PATH,
    "shape":             list(df.shape),
    "target_col":        TARGET_COL,
    "has_label":         has_label,
    "fraud_rate":        fraud_rate,
    "n_fraud":           n_fraud,
    "imbalance_ratio":   imbalance,
    "columns":           list(df.columns),
    "dtypes":            dtypes_map,
    "null_rates":        null_rates,
    "id_candidates":     id_candidates,
    "date_cols":         date_cols,
    "amount_stats":      amount_stats,
    "temporal_patterns": temporal_patterns,
    "category_fraud_rates": category_fraud_rates[:20],
    "velocity_flags":    velocity_flags,
    "quick_importance":  quick_importance,
    "leakage_flags":     leakage_flags,
    "completed_at":      datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "01_eda"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "fraud-detection-pipeline",
             "step": "01_eda", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        for feat in leakage_flags:
            db["findings"].insert_one({
                "agent": "data-profiling-agent", "type": "leakage_risk", "severity": "CRITICAL",
                "content": f"Feature '{feat}' importance > 0.90 — likely data leakage. Exclude before training.",
                "investigation_id": INVESTIGATION_ID, "timestamp": datetime.now().isoformat(),
            })
        for seg in category_fraud_rates[:3]:
            db["findings"].insert_one({
                "agent": "data-profiling-agent", "type": "high_fraud_segment", "severity": "WARNING",
                "content": f"High-fraud segment: {seg['column']}={seg['value']} fraud_rate={seg['fraud_rate']:.1%} lift={seg['lift']}× n={seg['n']}",
                "investigation_id": INVESTIGATION_ID, "timestamp": datetime.now().isoformat(),
            })
        print("[fraud/01_eda] ✓ Saved to MongoDB pipeline_state + findings")
        client.close()
    except Exception as e:
        print(f"[fraud/01_eda] WARNING: MongoDB write failed: {e}")

print("\n" + "="*60)
print("FRAUD EDA SUMMARY")
print("="*60)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
if has_label:
    print(f"Fraud rate: {fraud_rate:.3%}  |  Imbalance: {imbalance}:1")
print(f"High-fraud categories: {len(category_fraud_rates)}")
print(f"High-velocity entities: {len(velocity_flags)}")
print(f"Leakage flags: {leakage_flags}")
if quick_importance:
    print("\nTop 5 features:")
    for r in quick_importance[:5]:
        print(f"  {r['feature']:40s} {r['importance']:.4f}")
print(f"\n[fraud/01_eda] COMPLETE — investigation_id: {INVESTIGATION_ID}")
