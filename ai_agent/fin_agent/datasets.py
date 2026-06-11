"""Dataset management via MongoDB GridFS.

Datasets uploaded by the user are stored in GridFS (binary file storage)
and their metadata goes in the 'datasets' collection. Agents download a
dataset to /tmp/financial_ai/data/ before running sandbox code on it.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(os.getenv("AGENT_WORKDIR", "/tmp/financial_ai")) / "data"

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".json", ".xlsx"}


def _get_gridfs():
    """Return (db, GridFS bucket) or (None, None) if MongoDB not configured."""
    try:
        import gridfs
        from fin_agent.memory import _get_client

        client = _get_client()
        if client is None:
            return None, None

        import os
        db_name = os.getenv("MONGODB_DB", "financial_ai_copilot")
        db = client[db_name]
        return db, gridfs.GridFS(db, collection="dataset_files")
    except Exception as e:
        logger.warning("GridFS init failed: %s", e)
        return None, None


def upload_dataset(file_bytes: bytes, filename: str, content_type: str = "") -> Optional[dict]:
    """Store a dataset file in GridFS and record metadata in 'datasets' collection.

    Returns metadata dict with _id, name, size_mb, columns (if CSV) etc., or None on failure.
    """
    db, fs = _get_gridfs()
    if fs is None:
        return None

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")

    # Store binary in GridFS
    file_id = fs.put(
        file_bytes,
        filename=filename,
        content_type=content_type or _mime(ext),
        upload_date=datetime.utcnow(),
    )

    # Peek at schema for CSV/parquet
    schema_info: dict = {}
    try:
        import io
        import pandas as pd

        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(file_bytes), nrows=500)
        elif ext == ".parquet":
            df = pd.read_parquet(io.BytesIO(file_bytes))
        else:
            df = None

        if df is not None:
            schema_info = {
                "rows_preview": len(df),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "null_counts": df.isnull().sum().to_dict(),
            }
    except Exception as e:
        logger.debug("Schema peek failed: %s", e)

    metadata = {
        "gridfs_id": str(file_id),
        "name": filename,
        "size_bytes": len(file_bytes),
        "size_mb": round(len(file_bytes) / 1024 / 1024, 3),
        "content_type": content_type or _mime(ext),
        "uploaded_at": datetime.utcnow().isoformat(),
        **schema_info,
    }
    db["datasets"].insert_one(dict(metadata))
    logger.info("Dataset uploaded: %s (%s MB)", filename, metadata["size_mb"])
    return metadata


def list_datasets() -> list[dict]:
    """Return metadata for all uploaded datasets, newest first."""
    db, fs = _get_gridfs()
    if db is None:
        return []

    docs = list(
        db["datasets"].find({}, {"_id": 0}).sort("uploaded_at", -1).limit(100)
    )
    return docs


def download_dataset(name_or_id: str) -> Optional[Path]:
    """Download a dataset from GridFS to _DATA_DIR and return its local path.

    Accepts either the filename or the gridfs_id string.
    Returns the local Path, or None if not found.
    """
    db, fs = _get_gridfs()
    if db is None:
        return None

    # Look up metadata
    query = {"$or": [{"name": name_or_id}, {"gridfs_id": name_or_id}]}
    meta = db["datasets"].find_one(query, sort=[("uploaded_at", -1)])
    if meta is None:
        return None

    try:
        from bson import ObjectId

        gridfs_file = fs.get(ObjectId(meta["gridfs_id"]))
        file_bytes = gridfs_file.read()
    except Exception as e:
        logger.error("GridFS download failed: %s", e)
        return None

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _DATA_DIR / meta["name"]
    local_path.write_bytes(file_bytes)
    logger.info("Dataset downloaded to %s", local_path)
    return local_path


def delete_dataset(name_or_id: str) -> bool:
    """Delete a dataset from MongoDB GridFS and the local cache.

    Args:
        name_or_id: Dataset filename or gridfs_id string.

    Returns:
        True if deleted, False if not found.
    """
    db, fs = _get_gridfs()
    deleted = False

    if db is not None:
        import bson
        col = db["datasets"]
        # Try by name first
        meta = col.find_one({"name": name_or_id})
        if meta is None:
            # Try by gridfs_id
            try:
                meta = col.find_one({"gridfs_id": bson.ObjectId(name_or_id)})
            except Exception:
                pass

        if meta:
            try:
                fs.delete(meta["gridfs_id"])
            except Exception:
                pass
            col.delete_one({"_id": meta["_id"]})
            deleted = True
            logger.info("Deleted dataset from GridFS: %s", name_or_id)

    # Also remove local cache file if present
    candidates = [_DATA_DIR / name_or_id]
    # If name_or_id was an id, try to find the local file by iterating the dir
    for p in candidates:
        if p.exists():
            p.unlink()
            logger.info("Removed local cache: %s", p)

    return deleted


def _mime(ext: str) -> str:
    return {
        ".csv": "text/csv",
        ".parquet": "application/octet-stream",
        ".json": "application/json",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(ext, "application/octet-stream")
