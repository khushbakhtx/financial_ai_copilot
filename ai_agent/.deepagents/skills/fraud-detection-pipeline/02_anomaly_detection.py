"""Fraud Detection Pipeline — Step 02: Anomaly Detection.

Reads:  pipeline_state.01_eda (feature guidance, fraud_rate, id_candidates).
Writes: anomaly_scores DataFrame to GridFS.
        pipeline_state.02_anomaly + findings to MongoDB.
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

DATASET_PATH     = os.environ["DATASET_PATH"]
TARGET_COL       = os.environ.get("TARGET_COL", "")
INVESTIGATION_ID = os.environ["INVESTIGATION_ID"]
MONGODB_URI      = os.environ.get("MONGODB_URI", "")
DB_NAME          = os.environ.get("MONGODB_DB", "financial_ai_copilot")

sys.path.insert(0, str(Path(__file__).parent))
from _pipeline_io import gfs_save

import pandas as pd
import numpy as np

print(f"[fraud/02_anomaly] investigation_id={INVESTIGATION_ID}")

# ── Load step 01 from MongoDB ────────────────────────────────────────────────

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
    except Exception as e:
        print(f"[fraud/02_anomaly] MongoDB load {step} failed: {e}")
        return {}

step01 = _load_step("01_eda")
id_candidates = step01.get("id_candidates", [])
leakage_flags = step01.get("leakage_flags", [])
fraud_rate    = step01.get("fraud_rate") or 0.05
has_label     = step01.get("has_label", False)
if not TARGET_COL:
    TARGET_COL = step01.get("target_col", "")

# ── Load dataset ──────────────────────────────────────────────────────────────

df = pd.read_parquet(DATASET_PATH) if DATASET_PATH.endswith(".parquet") else pd.read_csv(DATASET_PATH)
print(f"[fraud/02_anomaly] Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── Prepare numeric feature matrix ───────────────────────────────────────────

from sklearn.preprocessing import StandardScaler, LabelEncoder

drop_cols = [TARGET_COL] + id_candidates + leakage_flags
feat_cols = [c for c in df.columns if c not in drop_cols]

X = df[feat_cols].copy()
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
for col in X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns:
    X[col] = X[col].astype(np.int64) // 10**9
X = X.fillna(-999).replace([np.inf, -np.inf], -999)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 1. Isolation Forest ───────────────────────────────────────────────────────

from sklearn.ensemble import IsolationForest

contamination = min(max(fraud_rate * 1.5, 0.01), 0.15)
print(f"[fraud/02_anomaly] IsolationForest contamination={contamination:.3f}")
iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
iso.fit(X_scaled)
iso_scores  = -iso.score_samples(X_scaled)   # higher = more anomalous
iso_flags   = (iso.predict(X_scaled) == -1).astype(int)
print(f"[fraud/02_anomaly] IsoForest flagged: {iso_flags.sum():,} ({iso_flags.mean():.2%})")

# ── 2. Local Outlier Factor (sampled) ─────────────────────────────────────────

from sklearn.neighbors import LocalOutlierFactor

sample_n  = min(10_000, len(X_scaled))
idx_s     = np.random.default_rng(42).choice(len(X_scaled), sample_n, replace=False)
lof       = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
lof_pred  = lof.fit_predict(X_scaled[idx_s])
lof_flags = np.zeros(len(df), dtype=int)
lof_flags[idx_s] = (lof_pred == -1).astype(int)
print(f"[fraud/02_anomaly] LOF flagged (sample): {lof_flags.sum():,}")

# ── 3. Velocity anomaly ───────────────────────────────────────────────────────

velocity_flags = np.zeros(len(df), dtype=int)
cust_cols = [c for c in df.columns if any(t in c.lower() for t in ("customer", "account", "user", "client", "card"))]
date_cols = [c for c in df.columns if any(t in c.lower() for t in ("date", "time", "timestamp"))]

if cust_cols and date_cols:
    try:
        cc, dc = cust_cols[0], date_cols[0]
        df["_dt"] = pd.to_datetime(df[dc], errors="coerce")
        df["_floor_h"] = df["_dt"].dt.floor("h")
        tx_ph = df.groupby([cc, "_floor_h"])[dc].transform("count")
        mean_v, std_v = tx_ph.mean(), tx_ph.std()
        velocity_flags = (tx_ph > mean_v + 3 * std_v).astype(int).values
        print(f"[fraud/02_anomaly] Velocity flags: {velocity_flags.sum():,}")
        df.drop(columns=["_dt", "_floor_h"], inplace=True, errors="ignore")
    except Exception as e:
        print(f"[fraud/02_anomaly] Velocity analysis failed: {e}")

# ── 4. Amount anomaly ─────────────────────────────────────────────────────────

amount_flags = np.zeros(len(df), dtype=int)
amount_cols  = [c for c in df.columns if any(t in c.lower() for t in ("amount", "value", "amt"))]
if amount_cols:
    ac = amount_cols[0]
    p995 = df[ac].quantile(0.995)
    amount_flags = (df[ac] > p995).astype(int).values
    print(f"[fraud/02_anomaly] Amount anomaly flags (>p99.5={p995:.2f}): {amount_flags.sum():,}")

# ── 5. Shared-entity detection ────────────────────────────────────────────────

shared_entity_flags = np.zeros(len(df), dtype=int)
entity_cols = [c for c in df.columns if any(t in c.lower() for t in ("device", "ip", "email", "phone", "mac"))]
if entity_cols and cust_cols:
    cc = cust_cols[0]
    for ec in entity_cols[:3]:
        accounts_per_entity = df.groupby(ec)[cc].nunique()
        shared = accounts_per_entity[accounts_per_entity > 3].index
        mask = df[ec].isin(shared).astype(int).values
        shared_entity_flags = np.maximum(shared_entity_flags, mask)
    print(f"[fraud/02_anomaly] Shared-entity flags: {shared_entity_flags.sum():,}")

# ── 6. Combine into ensemble score ───────────────────────────────────────────

ensemble_score = (
    0.4 * iso_scores / (iso_scores.max() + 1e-9) +
    0.2 * lof_flags.astype(float) +
    0.15 * velocity_flags.astype(float) +
    0.15 * amount_flags.astype(float) +
    0.10 * shared_entity_flags.astype(float)
)
any_flag = ((iso_flags | lof_flags | velocity_flags | amount_flags | shared_entity_flags) > 0).astype(int)

# ── 7. Evaluate vs label if available ────────────────────────────────────────

precision_iso, recall_iso = None, None
if has_label and TARGET_COL in df.columns:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    y_true = df[TARGET_COL].values
    if y_true.sum() > 0:
        auc_iso  = roc_auc_score(y_true, iso_scores)
        prec_iso = precision_score(y_true, iso_flags, zero_division=0)
        rec_iso  = recall_score(y_true, iso_flags, zero_division=0)
        print(f"[fraud/02_anomaly] IsoForest vs label: AUC={auc_iso:.3f} P={prec_iso:.3f} R={rec_iso:.3f}")
        precision_iso, recall_iso = float(prec_iso), float(rec_iso)

# ── 8. Save anomaly scores to GridFS ─────────────────────────────────────────

scores_df = pd.DataFrame({
    "iso_forest_score":    iso_scores,
    "iso_forest_flag":     iso_flags,
    "lof_flag":            lof_flags,
    "velocity_flag":       velocity_flags,
    "amount_flag":         amount_flags,
    "shared_entity_flag":  shared_entity_flags,
    "ensemble_score":      ensemble_score,
    "any_anomaly_flag":    any_flag,
})
if has_label and TARGET_COL in df.columns:
    scores_df[TARGET_COL] = df[TARGET_COL].values

if MONGODB_URI:
    gfs_save(scores_df, "anomaly_scores", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    scores_path = f"gridfs://{INVESTIGATION_ID}/anomaly_scores"
else:
    from pathlib import Path as _P
    out = _P(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
    out.mkdir(parents=True, exist_ok=True)
    scores_path = str(out / "anomaly_scores.parquet")
    scores_df.to_parquet(scores_path, index=False)

# ── 9. Persist to MongoDB pipeline_state ─────────────────────────────────────

result = {
    "step":                    "02_anomaly",
    "investigation_id":        INVESTIGATION_ID,
    "anomaly_scores_path":     scores_path,
    "contamination_used":      round(contamination, 4),
    "iso_forest_flagged":      int(iso_flags.sum()),
    "iso_forest_flagged_pct":  round(float(iso_flags.mean()), 4),
    "lof_flagged":             int(lof_flags.sum()),
    "velocity_flagged":        int(velocity_flags.sum()),
    "amount_flagged":          int(amount_flags.sum()),
    "shared_entity_flagged":   int(shared_entity_flags.sum()),
    "any_anomaly_flagged":     int(any_flag.sum()),
    "any_anomaly_pct":         round(float(any_flag.mean()), 4),
    "precision_vs_label":      precision_iso,
    "recall_vs_label":         recall_iso,
    "completed_at":            datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        db["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "02_anomaly"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "fraud-detection-pipeline",
             "step": "02_anomaly", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        db["findings"].insert_one({
            "agent": "fraud-investigation-agent", "type": "anomaly_summary", "severity": "WARNING",
            "content": (f"Anomaly detection complete: {any_flag.sum():,} records flagged ({any_flag.mean():.2%}). "
                        f"IsoForest={iso_flags.sum()} | Velocity={velocity_flags.sum()} | "
                        f"Amount={amount_flags.sum()} | SharedEntity={shared_entity_flags.sum()}"),
            "investigation_id": INVESTIGATION_ID, "timestamp": datetime.now().isoformat(),
        })
        print("[fraud/02_anomaly] ✓ Saved to MongoDB")
        client.close()
    except Exception as e:
        print(f"[fraud/02_anomaly] WARNING: MongoDB write failed: {e}")

print("\n" + "="*60)
print("ANOMALY DETECTION SUMMARY")
print("="*60)
print(f"Total records:         {len(df):,}")
print(f"IsolationForest flags: {iso_flags.sum():,} ({iso_flags.mean():.2%})")
print(f"LOF flags (sample):    {lof_flags.sum():,}")
print(f"Velocity anomalies:    {velocity_flags.sum():,}")
print(f"Amount anomalies:      {amount_flags.sum():,}")
print(f"Shared-entity flags:   {shared_entity_flags.sum():,}")
print(f"Any anomaly flagged:   {any_flag.sum():,} ({any_flag.mean():.2%})")
print(f"\n[fraud/02_anomaly] COMPLETE")
