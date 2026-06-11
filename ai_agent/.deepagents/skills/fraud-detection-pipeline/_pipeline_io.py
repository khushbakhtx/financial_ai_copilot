"""Shared GridFS I/O helpers for the credit-scoring pipeline.

Every pipeline script imports this module to read/write DataFrames from/to
MongoDB GridFS instead of local disk.  The local ARTIFACT_DIR is still used
for final binary outputs (model.pkl, schema, example_usage.py, model_card.md)
because those are not DataFrames and don't need cross-step sharing.

Key names stored in GridFS (all namespaced by investigation_id):
  <investigation_id>/train.parquet         — step 02 output
  <investigation_id>/test.parquet          — step 02 output
  <investigation_id>/train_fe.parquet      — step 03 output
  <investigation_id>/test_fe.parquet       — step 03 output
  <investigation_id>/test_scores.parquet   — step 04 output
  <investigation_id>/test_scores_v2.parquet — step 06 output

Usage in any pipeline script:
    from _pipeline_io import gfs_save, gfs_load, gfs_exists

    gfs_save(df, "train", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
    df = gfs_load("train", INVESTIGATION_ID, MONGODB_URI, DB_NAME)
"""

import io
import os


def _get_gridfs(mongodb_uri: str, db_name: str):
    """Return (db, GridFS bucket) or raise RuntimeError."""
    try:
        import gridfs
        from pymongo import MongoClient
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        return db, gridfs.GridFS(db, collection="pipeline_dataframes")
    except Exception as e:
        raise RuntimeError(f"GridFS init failed: {e}") from e


def _gfs_filename(name: str, investigation_id: str) -> str:
    return f"{investigation_id}/{name}.parquet"


def gfs_save(df, name: str, investigation_id: str, mongodb_uri: str, db_name: str) -> str:
    """Serialise DataFrame as Parquet and store in GridFS.

    Overwrites any previous file with the same name for this investigation_id.
    Returns the GridFS filename stored.
    """
    import pandas as pd
    filename = _gfs_filename(name, investigation_id)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)

    db, fs = _get_gridfs(mongodb_uri, db_name)
    # Remove previous version (GridFS allows multiple files with same filename)
    for old in fs.find({"filename": filename}):
        fs.delete(old._id)
    file_id = fs.put(buf.read(), filename=filename, investigation_id=investigation_id, name=name)
    db.client.close()

    print(f"[gfs] ✓ saved '{filename}' ({df.shape[0]:,} rows × {df.shape[1]} cols) id={file_id}")
    return filename


def gfs_load(name: str, investigation_id: str, mongodb_uri: str, db_name: str):
    """Load a DataFrame from GridFS by name + investigation_id.

    Returns a pandas DataFrame, or None if the file does not exist.
    """
    import pandas as pd
    filename = _gfs_filename(name, investigation_id)
    db, fs = _get_gridfs(mongodb_uri, db_name)
    gf = fs.find_one({"filename": filename})
    if gf is None:
        db.client.close()
        return None
    data = gf.read()
    db.client.close()
    df = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    print(f"[gfs] ✓ loaded '{filename}' ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    return df


def gfs_exists(name: str, investigation_id: str, mongodb_uri: str, db_name: str) -> bool:
    """Return True if the named DataFrame exists in GridFS."""
    filename = _gfs_filename(name, investigation_id)
    db, fs = _get_gridfs(mongodb_uri, db_name)
    exists = fs.exists({"filename": filename})
    db.client.close()
    return exists


def gfs_list(investigation_id: str, mongodb_uri: str, db_name: str) -> list:
    """List all DataFrame names stored for this investigation_id."""
    db, fs = _get_gridfs(mongodb_uri, db_name)
    names = [f.filename.split("/")[-1].replace(".parquet", "")
             for f in fs.find({"investigation_id": investigation_id})]
    db.client.close()
    return names
