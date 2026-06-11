"""Fraud Detection Pipeline — Step 03: Graph Analysis.

Reads:  pipeline_state.01_eda (column metadata).
Writes: fraud_relationships collection in MongoDB (all edges).
        pipeline_state.03_graph + findings to MongoDB.
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

import pandas as pd
import numpy as np
import networkx as nx

print(f"[fraud/03_graph] investigation_id={INVESTIGATION_ID}")

# ── Load step 01 ──────────────────────────────────────────────────────────────

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
        print(f"[fraud/03_graph] MongoDB load {step} failed: {e}")
        return {}

step01 = _load_step("01_eda")
if not TARGET_COL:
    TARGET_COL = step01.get("target_col", "")
has_label = step01.get("has_label", False)

# ── Load dataset ──────────────────────────────────────────────────────────────

df = pd.read_parquet(DATASET_PATH) if DATASET_PATH.endswith(".parquet") else pd.read_csv(DATASET_PATH)
print(f"[fraud/03_graph] Dataset: {df.shape[0]:,} rows")

# ── Identify entity columns ───────────────────────────────────────────────────

cust_cols   = [c for c in df.columns if any(t in c.lower() for t in ("customer", "account", "user", "client", "card", "sender"))]
entity_cols = [c for c in df.columns if any(t in c.lower() for t in ("device", "ip", "email", "phone", "mac", "merchant", "receiver", "recipient"))]

if not cust_cols:
    print("[fraud/03_graph] No account/customer column detected — skipping graph analysis")
    result = {"step": "03_graph", "investigation_id": INVESTIGATION_ID,
              "skipped": True, "reason": "no_account_column", "completed_at": datetime.now().isoformat()}
else:
    cc = cust_cols[0]
    print(f"[fraud/03_graph] Account column: '{cc}', Entity columns: {entity_cols}")

    # ── Build bipartite graph ─────────────────────────────────────────────────

    G = nx.Graph()
    edges_for_mongo = []

    for ec in entity_cols[:5]:  # cap at 5 entity types
        shared = df.groupby(ec)[cc].nunique()
        shared = shared[shared > 1]  # entities shared by >1 account
        for entity_val, _ in shared.items():
            accounts = df[df[ec] == entity_val][cc].unique()
            entity_node = f"{ec}:{entity_val}"
            for acc in accounts:
                acc_node = f"account:{acc}"
                G.add_edge(acc_node, entity_node)
                edges_for_mongo.append({
                    "entity_type":      ec,
                    "entity_id":        str(entity_val),
                    "account_id":       str(acc),
                    "investigation_id": INVESTIGATION_ID,
                    "timestamp":        datetime.now().isoformat(),
                })

    print(f"[fraud/03_graph] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Find connected components (fraud rings) ───────────────────────────────

    components = list(nx.connected_components(G))
    rings = [c for c in components if len([n for n in c if n.startswith("account:")]) >= 3]
    rings.sort(key=lambda c: len(c), reverse=True)

    print(f"[fraud/03_graph] Fraud rings (≥3 accounts): {len(rings)}")

    # ── Score rings using known fraud labels ──────────────────────────────────

    ring_summaries = []
    high_risk_account_ids = set()

    for ring in rings[:50]:  # top 50 largest rings
        acc_ids = [n.replace("account:", "") for n in ring if n.startswith("account:")]
        entity_ids = [n for n in ring if not n.startswith("account:")]

        risk_score = 0.0
        n_fraud_in_ring = 0
        if has_label and TARGET_COL in df.columns:
            ring_df = df[df[cc].astype(str).isin(acc_ids)]
            if len(ring_df) > 0:
                ring_fraud_rate = float(ring_df[TARGET_COL].mean())
                n_fraud_in_ring = int(ring_df[TARGET_COL].sum())
                risk_score = ring_fraud_rate
            if risk_score > 0.3:
                high_risk_account_ids.update(acc_ids)

        ring_summaries.append({
            "account_count":   len(acc_ids),
            "entity_count":    len(entity_ids),
            "total_nodes":     len(ring),
            "n_fraud":         n_fraud_in_ring,
            "ring_fraud_rate": round(risk_score, 4),
            "is_high_risk":    risk_score > 0.3,
            "sample_accounts": acc_ids[:5],
        })

    high_risk_rings   = [r for r in ring_summaries if r["is_high_risk"]]
    max_ring_size     = max((r["account_count"] for r in ring_summaries), default=0)

    # ── Cross-investigation lookup via MongoDB ────────────────────────────────

    prior_matches = 0
    if MONGODB_URI and entity_cols:
        try:
            from pymongo import MongoClient
            c = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            # Check if any entity_ids appeared in prior investigations
            sample_entities = [str(e) for e in df[entity_cols[0]].dropna().unique()[:100]]
            prior_matches = c[DB_NAME]["fraud_relationships"].count_documents({
                "entity_id": {"$in": sample_entities},
                "investigation_id": {"$ne": INVESTIGATION_ID},
            })
            c.close()
            if prior_matches > 0:
                print(f"[fraud/03_graph] ⚠ {prior_matches} entities match prior investigations!")
        except Exception as e:
            print(f"[fraud/03_graph] Prior lookup failed: {e}")

    # ── Persist edges to MongoDB fraud_relationships ─────────────────────────

    if MONGODB_URI and edges_for_mongo:
        try:
            from pymongo import MongoClient
            client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            db = client[DB_NAME]

            # Batch insert edges (cap at 5000 to avoid slow inserts)
            batch = edges_for_mongo[:5000]
            db["fraud_relationships"].insert_many(batch, ordered=False)
            print(f"[fraud/03_graph] ✓ Inserted {len(batch)} edges into fraud_relationships")

            # pipeline_state
            result = {
                "step":                  "03_graph",
                "investigation_id":      INVESTIGATION_ID,
                "graph_nodes":           G.number_of_nodes(),
                "graph_edges":           G.number_of_edges(),
                "total_rings":           len(rings),
                "high_risk_rings":       len(high_risk_rings),
                "max_ring_size":         max_ring_size,
                "high_risk_accounts":    list(high_risk_account_ids)[:100],
                "ring_summaries":        ring_summaries[:20],
                "prior_investigation_matches": prior_matches,
                "edges_written_to_mongo": len(batch),
                "completed_at":          datetime.now().isoformat(),
            }
            db["pipeline_state"].replace_one(
                {"investigation_id": INVESTIGATION_ID, "step": "03_graph"},
                {"investigation_id": INVESTIGATION_ID, "pipeline": "fraud-detection-pipeline",
                 "step": "03_graph", "data": result, "timestamp": datetime.now().isoformat()},
                upsert=True,
            )

            # Findings for high-risk rings
            for i, ring in enumerate(high_risk_rings[:5]):
                db["findings"].insert_one({
                    "agent": "fraud-investigation-agent", "type": "fraud_ring", "severity": "CRITICAL",
                    "content": (f"Fraud ring #{i+1}: {ring['account_count']} accounts, "
                                f"{ring['entity_count']} shared entities, "
                                f"fraud_rate={ring['ring_fraud_rate']:.1%}. "
                                f"Sample accounts: {ring['sample_accounts']}"),
                    "investigation_id": INVESTIGATION_ID, "timestamp": datetime.now().isoformat(),
                })

            if prior_matches > 0:
                db["findings"].insert_one({
                    "agent": "fraud-investigation-agent", "type": "cross_investigation_match", "severity": "CRITICAL",
                    "content": f"{prior_matches} entity IDs from this dataset appeared in prior fraud investigations. High-risk signal.",
                    "investigation_id": INVESTIGATION_ID, "timestamp": datetime.now().isoformat(),
                })

            print("[fraud/03_graph] ✓ Saved to MongoDB pipeline_state + findings")
            client.close()
        except Exception as e:
            print(f"[fraud/03_graph] WARNING: MongoDB write failed: {e}")
    else:
        result = {
            "step": "03_graph", "investigation_id": INVESTIGATION_ID,
            "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(),
            "total_rings": len(rings), "high_risk_rings": len(high_risk_rings),
            "completed_at": datetime.now().isoformat(),
        }

print("\n" + "="*60)
print("GRAPH ANALYSIS SUMMARY")
print("="*60)
print(f"Graph nodes: {G.number_of_nodes():,}   edges: {G.number_of_edges():,}")
print(f"Rings (≥3 accounts): {len(rings)}")
print(f"High-risk rings:     {len(high_risk_rings)}")
print(f"Max ring size:       {max_ring_size} accounts")
if prior_matches > 0:
    print(f"⚠ Prior investigation matches: {prior_matches}")
print(f"\n[fraud/03_graph] COMPLETE")
