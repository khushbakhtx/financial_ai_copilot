"""Credit Scoring Pipeline — Step 02: Preprocessing.

Reads:  pipeline_state.01_eda from MongoDB (column info, leakage flags, id candidates).
Writes: train + test DataFrames to MongoDB GridFS (pipeline_dataframes collection).
        preprocessor.pkl to artifacts dir (binary, stays on disk).
        pipeline_state.02_preprocessing to MongoDB.
"""

import json
import os
import pickle
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

DATASET_PATH     = os.environ["DATASET_PATH"]
TARGET_COL       = os.environ["TARGET_COL"]
INVESTIGATION_ID = os.environ["INVESTIGATION_ID"]
MONGODB_URI      = os.environ.get("MONGODB_URI", "")
DB_NAME          = os.environ.get("MONGODB_DB", "financial_ai_copilot")
TEST_SPLIT       = os.environ.get("TEST_SPLIT", "random")  # "random" or "oot"
TEST_SIZE        = float(os.environ.get("TEST_SIZE", "0.2"))

ARTIFACT_DIR = Path(f"/tmp/financial_ai/artifacts/{INVESTIGATION_ID}")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[02_preprocessing] investigation_id={INVESTIGATION_ID}")

# ── Load Step 01 state from MongoDB ──────────────────────────────────────────

eda_state = {}
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        doc = client[DB_NAME]["pipeline_state"].find_one(
            {"investigation_id": INVESTIGATION_ID, "step": "01_eda"},
            {"_id": 0}
        )
        if doc:
            eda_state = doc.get("data", {})
            print(f"[02_preprocessing] ✓ Loaded EDA state from MongoDB")
        else:
            print("[02_preprocessing] WARNING: No EDA state found — running without leakage info")
        client.close()
    except Exception as e:
        print(f"[02_preprocessing] WARNING: MongoDB read failed: {e}")

leakage_flags  = eda_state.get("leakage_flags", [])
dominance_flags = eda_state.get("dominance_flags", [])
id_candidates  = eda_state.get("id_candidates", [])

# ── Load dataset ──────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _pipeline_io import gfs_save

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if DATASET_PATH.endswith(".parquet"):
    df = pd.read_parquet(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH)

print(f"[02_preprocessing] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── 1. Drop columns ───────────────────────────────────────────────────────────

to_drop = set()
to_drop.update(leakage_flags)
to_drop.update(dominance_flags)
to_drop.update(id_candidates)

# Drop high-missing columns (>70%)
null_rates = df.isnull().mean()
high_missing = null_rates[null_rates > 0.70].index.tolist()
to_drop.update(high_missing)

# Drop columns with a single unique value (zero variance)
constant_cols = [c for c in df.columns if c != TARGET_COL and df[c].nunique() <= 1]
to_drop.update(constant_cols)

# Drop datetime cols — will use them only for OOT split
date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
date_candidates = [c for c in df.columns if any(tok in c.lower() for tok in ("date", "time", "period")) and c not in to_drop]
for dc in date_candidates:
    try:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        if df[dc].notna().sum() > len(df) * 0.5:
            date_cols.append(dc)
    except Exception:
        pass

to_drop.update(date_cols)
to_drop.discard(TARGET_COL)

features_dropped = {
    "leakage":     list(leakage_flags),
    "dominance":   list(dominance_flags),
    "identifiers": list(id_candidates),
    "high_missing": list(high_missing),
    "constant":    list(constant_cols),
    "datetime":    list(set(date_cols)),
}

df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)
print(f"[02_preprocessing] Dropped {len(to_drop)} columns → {df.shape[1]} remaining")

# ── 2. Train/test split ───────────────────────────────────────────────────────

if TEST_SPLIT == "oot" and date_cols:
    dc = date_cols[0]
    # Use most recent TEST_SIZE fraction as test
    full_df_with_date = pd.read_csv(DATASET_PATH) if not DATASET_PATH.endswith(".parquet") else pd.read_parquet(DATASET_PATH)
    full_df_with_date[dc] = pd.to_datetime(full_df_with_date[dc], errors="coerce")
    cutoff = full_df_with_date[dc].quantile(1.0 - TEST_SIZE)
    train_idx = full_df_with_date[full_df_with_date[dc] < cutoff].index
    test_idx  = full_df_with_date[full_df_with_date[dc] >= cutoff].index
    df_train  = df.loc[df.index.isin(train_idx)].copy()
    df_test   = df.loc[df.index.isin(test_idx)].copy()
    print(f"[02_preprocessing] OOT split at {cutoff}: train={len(df_train)}, test={len(df_test)}")
else:
    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_COL], random_state=42
    )
    print(f"[02_preprocessing] Random split 80/20: train={len(df_train)}, test={len(df_test)}")

# ── 3. Null imputation ────────────────────────────────────────────────────────

feature_cols = [c for c in df_train.columns if c != TARGET_COL]
null_fill_map = {}

for col in feature_cols:
    if df_train[col].isnull().sum() == 0:
        continue
    if df_train[col].dtype == "object":
        fill_val = df_train[col].mode()[0] if len(df_train[col].mode()) > 0 else "UNKNOWN"
    else:
        fill_val = float(df_train[col].median())
    null_fill_map[col] = fill_val
    df_train[col].fillna(fill_val, inplace=True)
    df_test[col].fillna(fill_val, inplace=True)

print(f"[02_preprocessing] Imputed nulls in {len(null_fill_map)} columns")

# ── 4. Outlier capping (IQR ×3, numeric only) ────────────────────────────────

outlier_caps = {}
for col in df_train.select_dtypes(include=np.number).columns:
    if col == TARGET_COL:
        continue
    q1, q3 = df_train[col].quantile(0.25), df_train[col].quantile(0.75)
    iqr     = q3 - q1
    lo, hi  = q1 - 3 * iqr, q3 + 3 * iqr
    if df_train[col].min() < lo or df_train[col].max() > hi:
        outlier_caps[col] = {"lo": float(lo), "hi": float(hi)}
        df_train[col] = df_train[col].clip(lo, hi)
        df_test[col]  = df_test[col].clip(lo, hi)

print(f"[02_preprocessing] Capped outliers in {len(outlier_caps)} columns")

# ── 5. Categorical encoding ───────────────────────────────────────────────────

encoding_map = {}
label_encoders = {}
target_encoders = {}

for col in df_train.select_dtypes(include="object").columns:
    if col == TARGET_COL:
        continue
    n_unique = df_train[col].nunique()
    if n_unique > 20:
        # Target encoding on train set, apply to test
        target_mean = df_train.groupby(col)[TARGET_COL].mean()
        global_mean = df_train[TARGET_COL].mean()
        df_train[col] = df_train[col].map(target_mean).fillna(global_mean)
        df_test[col]  = df_test[col].map(target_mean).fillna(global_mean)
        target_encoders[col] = {"type": "target", "map": target_mean.to_dict(), "global_mean": float(global_mean)}
        encoding_map[col] = "target_encoding"
    else:
        le = LabelEncoder()
        # Fit on combined unique values to avoid unseen label issues
        all_vals = pd.concat([df_train[col], df_test[col]]).astype(str)
        le.fit(all_vals)
        df_train[col] = le.transform(df_train[col].astype(str))
        df_test[col]  = le.transform(df_test[col].astype(str))
        label_encoders[col] = le
        encoding_map[col] = "label_encoding"

print(f"[02_preprocessing] Encoded {len(encoding_map)} categorical columns")

# ── 6. Near-constant removal (after encoding) ─────────────────────────────────

near_constant = [
    c for c in df_train.columns
    if c != TARGET_COL and df_train[c].std() < 1e-4 and df_train[c].nunique() <= 2
]
if near_constant:
    df_train.drop(columns=near_constant, inplace=True)
    df_test.drop(columns=near_constant, inplace=True)
    print(f"[02_preprocessing] Removed {len(near_constant)} near-constant columns: {near_constant}")

# ── 7. Save train/test CSVs ───────────────────────────────────────────────────

if MONGODB_URI:
    gfs_save(df_train, "train", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    gfs_save(df_test,  "test",  INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    train_path = f"gridfs://{INVESTIGATION_ID}/train"
    test_path  = f"gridfs://{INVESTIGATION_ID}/test"
    print(f"[02_preprocessing] Saved train → GridFS ({len(df_train)} rows)")
    print(f"[02_preprocessing] Saved test  → GridFS ({len(df_test)} rows)")
else:
    # Local fallback when MongoDB not configured
    train_path = str(ARTIFACT_DIR / "train.parquet")
    test_path  = str(ARTIFACT_DIR / "test.parquet")
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    print(f"[02_preprocessing] Saved train → {train_path} (local fallback)")

# ── 8. Save preprocessor state ───────────────────────────────────────────────

preprocessor_state = {
    "null_fill_map":    null_fill_map,
    "outlier_caps":     outlier_caps,
    "target_encoders":  {k: {"type": "target", "map": v["map"], "global_mean": v["global_mean"]} for k, v in target_encoders.items()},
    "label_encoder_classes": {k: list(le.classes_) for k, le in label_encoders.items()},
    "encoding_map":     encoding_map,
    "features_dropped": features_dropped,
    "near_constant_removed": near_constant,
    "feature_cols":     [c for c in df_train.columns if c != TARGET_COL],
}
preprocessor_path = str(ARTIFACT_DIR / "preprocessor.pkl")
with open(preprocessor_path, "wb") as f:
    pickle.dump({
        "null_fill_map":   null_fill_map,
        "outlier_caps":    outlier_caps,
        "target_encoders": target_encoders,
        "label_encoders":  label_encoders,
        "encoding_map":    encoding_map,
        "feature_cols":    [c for c in df_train.columns if c != TARGET_COL],
        "target_col":      TARGET_COL,
    }, f)
print(f"[02_preprocessing] Preprocessor saved → {preprocessor_path}")

# ── 9. Persist to MongoDB pipeline_state ─────────────────────────────────────

result = {
    "step":                 "02_preprocessing",
    "investigation_id":     INVESTIGATION_ID,
    "train_path":           train_path,
    "test_path":            test_path,
    "preprocessor_path":    preprocessor_path,
    "train_rows":           len(df_train),
    "test_rows":            len(df_test),
    "features_kept":        [c for c in df_train.columns if c != TARGET_COL],
    "n_features":           len(df_train.columns) - 1,
    "features_dropped":     features_dropped,
    "encoding_map":         encoding_map,
    "null_fill_map":        {k: str(v) for k, v in null_fill_map.items()},
    "outlier_caps_count":   len(outlier_caps),
    "train_target_rate":    round(float(df_train[TARGET_COL].mean()), 4),
    "test_target_rate":     round(float(df_test[TARGET_COL].mean()), 4),
    "split_method":         TEST_SPLIT,
    "completed_at":         datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client[DB_NAME]["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "02_preprocessing"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "credit-scoring-pipeline",
             "step": "02_preprocessing", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        print(f"[02_preprocessing] ✓ Saved to MongoDB pipeline_state")
        client.close()
    except Exception as e:
        print(f"[02_preprocessing] WARNING: MongoDB write failed: {e}")

print("\n" + "="*60)
print("PREPROCESSING SUMMARY")
print("="*60)
print(f"Train: {len(df_train):,} rows — target rate {result['train_target_rate']:.2%}")
print(f"Test:  {len(df_test):,} rows — target rate {result['test_target_rate']:.2%}")
print(f"Features kept: {result['n_features']}")
print(f"Encoding applied: {len(encoding_map)} columns")
print(f"Dropped — leakage:{len(leakage_flags)} | identifiers:{len(id_candidates)} | high_missing:{len(high_missing)} | constant:{len(constant_cols)}")
print(f"\n[02_preprocessing] COMPLETE")
