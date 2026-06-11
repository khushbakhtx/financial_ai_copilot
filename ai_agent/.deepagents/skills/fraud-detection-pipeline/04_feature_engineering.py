"""Fraud Detection Pipeline — Step 04: Feature Engineering.

Reads:  pipeline_state.01_eda, 02_anomaly, 03_graph from MongoDB.
        anomaly_scores from GridFS.
Writes: train_fe + test_fe to GridFS.
        pipeline_state.04_features to MongoDB.
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
from _pipeline_io import gfs_save, gfs_load

import pandas as pd
import numpy as np

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[fraud/04_features] investigation_id={INVESTIGATION_ID}")

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
    except Exception as e:
        return {}

step01 = _load_step("01_eda")
step02 = _load_step("02_anomaly")
step03 = _load_step("03_graph")

if not TARGET_COL:
    TARGET_COL = step01.get("target_col", "")
has_label     = step01.get("has_label", bool(TARGET_COL))
id_candidates = step01.get("id_candidates", [])
leakage_flags = step01.get("leakage_flags", [])
date_cols     = step01.get("date_cols", [])
high_risk_accounts = set(step03.get("high_risk_accounts", []))

# ── Load dataset + anomaly scores ─────────────────────────────────────────────

df = pd.read_parquet(DATASET_PATH) if DATASET_PATH.endswith(".parquet") else pd.read_csv(DATASET_PATH)
anomaly_df = gfs_load("anomaly_scores", INVESTIGATION_ID, MONGODB_URI, DB_NAME)

if anomaly_df is not None and len(anomaly_df) == len(df):
    df["iso_forest_score"]   = anomaly_df["iso_forest_score"].values
    df["ensemble_score"]     = anomaly_df["ensemble_score"].values
    df["any_anomaly_flag"]   = anomaly_df["any_anomaly_flag"].values
    print("[fraud/04_features] ✓ Merged anomaly scores from GridFS")
else:
    df["iso_forest_score"] = 0.0
    df["ensemble_score"]   = 0.0
    df["any_anomaly_flag"] = 0

# ── Identify key column types ─────────────────────────────────────────────────

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

cust_cols   = [c for c in df.columns if any(t in c.lower() for t in ("customer", "account", "user", "client", "card", "sender"))]
amount_cols = [c for c in df.columns if any(t in c.lower() for t in ("amount", "value", "amt"))]
entity_cols = [c for c in df.columns if any(t in c.lower() for t in ("device", "ip", "email", "phone", "merchant"))]

cc = cust_cols[0] if cust_cols else None
ac = amount_cols[0] if amount_cols else None

engineered = []

# ── 1. Velocity features ──────────────────────────────────────────────────────

if cc and date_cols:
    try:
        dc = date_cols[0]
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df_s   = df.sort_values([cc, dc])

        df["tx_count_1h"]  = df_s.groupby([cc, df_s[dc].dt.floor("h")])[dc].transform("count").values
        df["tx_count_24h"] = df_s.groupby([cc, df_s[dc].dt.floor("d")])[dc].transform("count").values
        engineered += ["tx_count_1h", "tx_count_24h"]

        if ac:
            df["tx_amount_1h"] = df_s.groupby([cc, df_s[dc].dt.floor("h")])[ac].transform("sum").values
            engineered.append("tx_amount_1h")

        print(f"[fraud/04_features] Velocity features: {engineered}")
    except Exception as e:
        print(f"[fraud/04_features] Velocity features failed: {e}")

# ── 2. Amount deviation features ─────────────────────────────────────────────

if ac and cc:
    cust_mean  = df.groupby(cc)[ac].transform("mean")
    df["amount_vs_cust_avg"] = (df[ac] / cust_mean.replace(0, np.nan)).clip(0, 100).fillna(1.0)
    engineered.append("amount_vs_cust_avg")

    merchant_cols = [c for c in df.columns if "merchant" in c.lower()]
    if merchant_cols:
        mc = merchant_cols[0]
        merch_mean = df.groupby(mc)[ac].transform("mean")
        df["amount_vs_merch_avg"] = (df[ac] / merch_mean.replace(0, np.nan)).clip(0, 100).fillna(1.0)
        engineered.append("amount_vs_merch_avg")

# ── 3. Ring membership flag ───────────────────────────────────────────────────

if cc and high_risk_accounts:
    df["is_in_fraud_ring"] = df[cc].astype(str).isin(high_risk_accounts).astype(int)
    engineered.append("is_in_fraud_ring")
    print(f"[fraud/04_features] Ring membership: {df['is_in_fraud_ring'].sum()} accounts flagged")

# ── 4. Time features ─────────────────────────────────────────────────────────

if date_cols:
    dc = date_cols[0]
    df[dc] = pd.to_datetime(df[dc], errors="coerce")
    df["hour_sin"] = np.sin(2 * np.pi * df[dc].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df[dc].dt.hour / 24)
    df["is_weekend"] = df[dc].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_night"]   = df[dc].dt.hour.between(22, 5).astype(int)
    engineered += ["hour_sin", "hour_cos", "is_weekend", "is_night"]

# ── 5. Shared-entity count per account ───────────────────────────────────────

if cc and entity_cols:
    for ec in entity_cols[:2]:
        shared_count = df.groupby(ec)[cc].transform("nunique")
        col_name = f"{ec}_shared_accounts"
        df[col_name] = shared_count.fillna(1)
        engineered.append(col_name)

# ── 6. Prepare final feature set ─────────────────────────────────────────────

drop_cols = set(id_candidates + leakage_flags + date_cols)
drop_cols.discard(TARGET_COL)
feature_cols = [c for c in df.columns if c not in drop_cols and c != TARGET_COL]

X = df[feature_cols].copy()
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
X = X.fillna(0).replace([np.inf, -np.inf], 0)
df_clean = X.copy()
if TARGET_COL and TARGET_COL in df.columns:
    df_clean[TARGET_COL] = df[TARGET_COL].values

# ── 7. Train/test split ───────────────────────────────────────────────────────

if has_label and TARGET_COL in df_clean.columns and df_clean[TARGET_COL].sum() > 5:
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df_clean, test_size=0.2, stratify=df_clean[TARGET_COL], random_state=42)
else:
    n = len(df_clean)
    df_train = df_clean.iloc[:int(n * 0.8)].copy()
    df_test  = df_clean.iloc[int(n * 0.8):].copy()

print(f"[fraud/04_features] train={df_train.shape}, test={df_test.shape}")

# ── 8. Save to GridFS ─────────────────────────────────────────────────────────

if MONGODB_URI:
    gfs_save(df_train, "train_fe", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    gfs_save(df_test,  "test_fe",  INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    train_fe_path = f"gridfs://{INVESTIGATION_ID}/train_fe"
    test_fe_path  = f"gridfs://{INVESTIGATION_ID}/test_fe"
else:
    train_fe_path = str(ARTIFACT_DIR / "train_fe.parquet")
    test_fe_path  = str(ARTIFACT_DIR / "test_fe.parquet")
    df_train.to_parquet(train_fe_path, index=False)
    df_test.to_parquet(test_fe_path, index=False)

result = {
    "step":              "04_features",
    "investigation_id":  INVESTIGATION_ID,
    "train_fe_path":     train_fe_path,
    "test_fe_path":      test_fe_path,
    "features_engineered": engineered,
    "final_feature_list": [c for c in df_train.columns if c != TARGET_COL],
    "n_features":        len(df_train.columns) - (1 if TARGET_COL in df_train.columns else 0),
    "train_rows":        len(df_train),
    "test_rows":         len(df_test),
    "completed_at":      datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client[DB_NAME]["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "04_features"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "fraud-detection-pipeline",
             "step": "04_features", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        print("[fraud/04_features] ✓ Saved to MongoDB")
        client.close()
    except Exception as e:
        print(f"[fraud/04_features] WARNING: MongoDB write failed: {e}")

print(f"\n[fraud/04_features] COMPLETE — {result['n_features']} features, engineered: {engineered}")
