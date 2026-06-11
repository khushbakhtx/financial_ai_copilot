"""Credit Scoring Pipeline — Step 03: Feature Engineering.

Reads:  train + test DataFrames from MongoDB GridFS (written by step 02).
        pipeline_state.02_preprocessing (feature_cols, encoding metadata).
        pipeline_state.01_eda (quick_importance for picking square-term candidates).
Writes: train_fe + test_fe DataFrames to MongoDB GridFS.
        pipeline_state.03_feature_engineering to MongoDB.
"""

import os
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
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[03_feature_engineering] investigation_id={INVESTIGATION_ID}")

# ── Load Step 02 state from MongoDB ──────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _pipeline_io import gfs_save, gfs_load

import pandas as pd
import numpy as np

prep_state = {}
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        doc = client[DB_NAME]["pipeline_state"].find_one(
            {"investigation_id": INVESTIGATION_ID, "step": "02_preprocessing"},
            {"_id": 0}
        )
        if doc:
            prep_state = doc.get("data", {})
            print(f"[03_feature_engineering] ✓ Loaded preprocessing state from MongoDB")
        else:
            print("[03_feature_engineering] ERROR: No preprocessing state found in MongoDB. Run step 02 first.")
            sys.exit(1)
        client.close()
    except Exception as e:
        print(f"[03_feature_engineering] MongoDB read failed: {e}")
        sys.exit(1)
else:
    print("[03_feature_engineering] WARNING: No MONGODB_URI — cannot load prior state.")
    sys.exit(1)

df_train = gfs_load("train", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
df_test  = gfs_load("test",  INVESTIGATION_ID, MONGODB_URI, DB_NAME)

if df_train is None or df_test is None:
    train_path = prep_state.get("train_path", "")
    test_path  = prep_state.get("test_path", "")
    if train_path and not train_path.startswith("gridfs://"):
        df_train = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
        df_test  = pd.read_parquet(test_path)  if test_path.endswith(".parquet")  else pd.read_csv(test_path)
    else:
        print("[03_feature_engineering] ERROR: Could not load train/test from GridFS or local path.")
        sys.exit(1)

print(f"[03_feature_engineering] train={df_train.shape}, test={df_test.shape}")

feature_cols = [c for c in df_train.columns if c != TARGET_COL]
num_cols     = df_train[feature_cols].select_dtypes(include=np.number).columns.tolist()

# ── 1. Ratio / financial features ────────────────────────────────────────────

ratio_features_added = []

def _safe_ratio(df, num_col, den_col, new_col, clip=50.0):
    """Add ratio feature only if both columns exist."""
    if num_col in df.columns and den_col in df.columns:
        df[new_col] = (df[num_col] / df[den_col].replace(0, np.nan)).clip(0, clip).fillna(0)
        return new_col
    return None

credit_ratios = [
    ("loan_amount",     "annual_income",      "loan_to_income",         10.0),
    ("debt_to_income",  "annual_income",      "dti_scaled",             1.0),
    ("loan_amount",     "credit_score",       "loan_per_credit_pt",     50.0),
    ("interest_rate",   "annual_income",      "rate_burden",            0.5),
    ("num_open_accounts", "employment_years", "accounts_per_emp_year",  30.0),
]
for num, den, name, clip_v in credit_ratios:
    r = _safe_ratio(df_train, num, den, name, clip_v)
    if r:
        _safe_ratio(df_test, num, den, name, clip_v)
        ratio_features_added.append(name)

print(f"[03_feature_engineering] Ratio features: {ratio_features_added}")

# ── 2. Interaction features ───────────────────────────────────────────────────

interaction_features_added = []
interaction_pairs = [
    ("credit_score",      "employment_years"),
    ("loan_amount",       "interest_rate"),
    ("debt_to_income",    "delinquencies_2yr"),
    ("annual_income",     "loan_term_months"),
    ("credit_score",      "debt_to_income"),
]
for c1, c2 in interaction_pairs:
    if c1 in df_train.columns and c2 in df_train.columns:
        name = f"{c1}_x_{c2}"
        df_train[name] = df_train[c1] * df_train[c2]
        df_test[name]  = df_test[c1]  * df_test[c2]
        interaction_features_added.append(name)

print(f"[03_feature_engineering] Interaction features: {interaction_features_added}")

# ── 3. Discretization / band features ────────────────────────────────────────

band_features_added = []

if "credit_score" in df_train.columns:
    bins   = [0, 580, 670, 740, 800, 851]
    labels = [0, 1, 2, 3, 4]  # Poor, Fair, Good, Very Good, Exceptional
    df_train["credit_score_band"] = pd.cut(df_train["credit_score"], bins=bins, labels=labels, include_lowest=True).astype(float).fillna(0)
    df_test["credit_score_band"]  = pd.cut(df_test["credit_score"],  bins=bins, labels=labels, include_lowest=True).astype(float).fillna(0)
    band_features_added.append("credit_score_band")

if "debt_to_income" in df_train.columns:
    dti_bins   = [0, 0.1, 0.2, 0.35, 0.5, 1.01]
    dti_labels = [0, 1, 2, 3, 4]
    df_train["dti_band"] = pd.cut(df_train["debt_to_income"], bins=dti_bins, labels=dti_labels, include_lowest=True).astype(float).fillna(0)
    df_test["dti_band"]  = pd.cut(df_test["debt_to_income"],  bins=dti_bins, labels=dti_labels, include_lowest=True).astype(float).fillna(0)
    band_features_added.append("dti_band")

if "annual_income" in df_train.columns:
    income_pct = df_train["annual_income"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    df_train["income_band"] = pd.cut(df_train["annual_income"], bins=income_pct, labels=[0,1,2,3,4], include_lowest=True, duplicates="drop").astype(float).fillna(0)
    df_test["income_band"]  = pd.cut(df_test["annual_income"],  bins=income_pct, labels=[0,1,2,3,4], include_lowest=True, duplicates="drop").astype(float).fillna(0)
    band_features_added.append("income_band")

print(f"[03_feature_engineering] Band features: {band_features_added}")

# ── 4. Log transforms for skewed numerics ────────────────────────────────────

log_features_added = []
skew_cols = [c for c in num_cols if df_train[c].skew() > 2.0 and df_train[c].min() >= 0]
for col in skew_cols[:5]:
    name = f"{col}_log1p"
    df_train[name] = np.log1p(df_train[col])
    df_test[name]  = np.log1p(df_test[col])
    log_features_added.append(name)

print(f"[03_feature_engineering] Log transforms: {log_features_added}")

# ── 5. Square terms for top-importance numerics ───────────────────────────────

# Use EDA importance to pick candidates if available; else use variance top-5
eda_state = {}
if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        doc2 = client[DB_NAME]["pipeline_state"].find_one(
            {"investigation_id": INVESTIGATION_ID, "step": "01_eda"}, {"_id": 0}
        )
        if doc2:
            eda_state = doc2.get("data", {})
        client.close()
    except Exception:
        pass

top_features = [r["feature"] for r in eda_state.get("quick_importance", [])[:5]]
top_features = [f for f in top_features if f in num_cols]
if not top_features:
    top_features = sorted(num_cols, key=lambda c: df_train[c].var(), reverse=True)[:5]

sq_features_added = []
for col in top_features[:3]:
    name = f"{col}_sq"
    df_train[name] = df_train[col] ** 2
    df_test[name]  = df_test[col]  ** 2
    sq_features_added.append(name)

print(f"[03_feature_engineering] Squared features: {sq_features_added}")

# ── 6. Mutual Information feature selection ───────────────────────────────────

from sklearn.feature_selection import mutual_info_classif

all_features  = [c for c in df_train.columns if c != TARGET_COL]
X_train_fe    = df_train[all_features].fillna(0).replace([np.inf, -np.inf], 0).values
y_train       = df_train[TARGET_COL].values

print(f"[03_feature_engineering] Computing MI scores for {len(all_features)} features...")
mi_scores = mutual_info_classif(X_train_fe, y_train, random_state=42)
mi_df = pd.DataFrame({"feature": all_features, "mi": mi_scores}).sort_values("mi", ascending=False)

# Keep top 60 by MI (or all if fewer)
n_keep = min(60, len(mi_df))
selected_features = mi_df.head(n_keep)["feature"].tolist()
# Always keep the target
if TARGET_COL not in selected_features:
    selected_features.append(TARGET_COL)

df_train_fe = df_train[selected_features].copy()
df_test_fe  = df_test[[c for c in selected_features if c in df_test.columns]].copy()
# Fill any test cols missing from train (edge case)
for c in selected_features:
    if c not in df_test_fe.columns:
        df_test_fe[c] = 0

mi_scores_list = mi_df.head(30).to_dict(orient="records")
print(f"[03_feature_engineering] MI selection: {n_keep} features kept of {len(all_features)}")
print(f"  Top 5 by MI: {mi_df.head(5)[['feature','mi']].values.tolist()}")

# ── 7. Save engineered datasets ──────────────────────────────────────────────

if MONGODB_URI:
    gfs_save(df_train_fe, "train_fe", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    gfs_save(df_test_fe,  "test_fe",  INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    train_fe_path = f"gridfs://{INVESTIGATION_ID}/train_fe"
    test_fe_path  = f"gridfs://{INVESTIGATION_ID}/test_fe"
    print(f"[03_feature_engineering] Saved train_fe → GridFS ({df_train_fe.shape[0]:,} rows × {df_train_fe.shape[1]} cols)")
    print(f"[03_feature_engineering] Saved test_fe  → GridFS ({df_test_fe.shape[0]:,} rows × {df_test_fe.shape[1]} cols)")
else:
    train_fe_path = str(ARTIFACT_DIR / "train_fe.parquet")
    test_fe_path  = str(ARTIFACT_DIR / "test_fe.parquet")
    df_train_fe.to_parquet(train_fe_path, index=False)
    df_test_fe.to_parquet(test_fe_path, index=False)
    print(f"[03_feature_engineering] Saved train_fe → {train_fe_path} (local fallback)")

# ── 8. Persist to MongoDB pipeline_state ─────────────────────────────────────

all_engineered = ratio_features_added + interaction_features_added + band_features_added + log_features_added + sq_features_added

result = {
    "step":                 "03_feature_engineering",
    "investigation_id":     INVESTIGATION_ID,
    "train_fe_path":        train_fe_path,
    "test_fe_path":         test_fe_path,
    "features_engineered":  all_engineered,
    "mi_scores":            mi_scores_list,
    "final_feature_list":   [c for c in selected_features if c != TARGET_COL],
    "n_features":           len(selected_features) - 1,
    "completed_at":         datetime.now().isoformat(),
}

if MONGODB_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client[DB_NAME]["pipeline_state"].replace_one(
            {"investigation_id": INVESTIGATION_ID, "step": "03_feature_engineering"},
            {"investigation_id": INVESTIGATION_ID, "pipeline": "credit-scoring-pipeline",
             "step": "03_feature_engineering", "data": result, "timestamp": datetime.now().isoformat()},
            upsert=True,
        )
        print(f"[03_feature_engineering] ✓ Saved to MongoDB pipeline_state")
        client.close()
    except Exception as e:
        print(f"[03_feature_engineering] WARNING: MongoDB write failed: {e}")

print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"Original features:    {len(feature_cols)}")
print(f"Engineered features:  {len(all_engineered)}")
print(f"Final after MI sel.:  {result['n_features']}")
print(f"\nTop 10 by Mutual Information:")
for r in mi_scores_list[:10]:
    bar = "█" * int(r["mi"] * 60)
    print(f"  {r['feature']:45s} {r['mi']:.4f} {bar}")
print(f"\n[03_feature_engineering] COMPLETE")
