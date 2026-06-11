"""MongoDB persistence layer — mirrors agent/orchestrator/database.py exactly."""
from __future__ import annotations

import uuid as _uuid_mod
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import motor.motor_asyncio

_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
_db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _str(v: Any) -> Any:
    if isinstance(v, _uuid_mod.UUID):
        return str(v)
    return v


async def init_db(uri: str, db_name: str = "financial_copilot_backend") -> None:
    global _client, _db
    _client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    _db = _client[db_name]
    await _ensure_indexes()


async def close_db() -> None:
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None


def get_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _db


def _get_db_name() -> str:
    import os
    return os.environ.get("BACKEND_DB_NAME", "financial_copilot_backend")


async def _ensure_indexes() -> None:
    db = get_db()
    await db.assistant.create_index("graph_id")
    await db.assistant.create_index("created_at")
    await db.thread.create_index("created_at")
    await db.thread.create_index("updated_at")
    await db.thread.create_index("status")
    await db.run.create_index("thread_id")
    await db.run.create_index([("thread_id", 1), ("created_at", -1)])
    await db.store.create_index([("prefix", 1), ("key", 1)], unique=True)
    await db.store.create_index("expires_at", sparse=True)


async def _seed_assistants() -> None:
    defaults = [("financial_copilot", "financial_copilot")]
    db = get_db()
    for graph_id, name in defaults:
        exists = await db.assistant.find_one({"graph_id": graph_id})
        if not exists:
            await db.assistant.insert_one({
                "assistant_id": str(_uuid_mod.uuid4()),
                "graph_id": graph_id,
                "name": name,
                "config": {},
                "metadata": {"created_by": "system"},
                "version": 1,
                "description": None,
                "context": None,
                "created_at": _now(),
                "updated_at": _now(),
            })


def _doc(doc: dict) -> dict:
    if doc is None:
        return {}
    out = dict(doc)
    out.pop("_id", None)
    for k, v in out.items():
        if isinstance(v, _uuid_mod.UUID):
            out[k] = str(v)
    return out


# ── Assistants ────────────────────────────────────────────────────────────────

async def create_assistant(
    graph_id: str,
    assistant_id: Optional[str] = None,
    config: dict = {},
    context: Optional[dict] = None,
    metadata: dict = {},
    name: Optional[str] = None,
    description: Optional[str] = None,
    if_exists: str = "raise",
) -> dict[str, Any]:
    db = get_db()
    aid = assistant_id or str(_uuid_mod.uuid4())
    existing = await db.assistant.find_one({"assistant_id": aid})
    if existing:
        if if_exists == "raise":
            raise ValueError(f"Assistant {aid} already exists")
        return _doc(existing)
    doc = {
        "assistant_id": aid,
        "graph_id": graph_id,
        "config": config,
        "context": context,
        "metadata": metadata,
        "name": name,
        "description": description,
        "version": 1,
        "created_at": _now(),
        "updated_at": _now(),
    }
    await db.assistant.insert_one(doc)
    return _doc(doc)


async def get_assistant(assistant_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    doc = await db.assistant.find_one({"assistant_id": assistant_id})
    return _doc(doc) if doc else None


async def get_assistant_by_graph(graph_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    doc = await db.assistant.find_one({"graph_id": graph_id})
    return _doc(doc) if doc else None


async def search_assistants(
    graph_id: Optional[str] = None,
    name: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    metadata_filter: Optional[dict] = None,
) -> list[dict[str, Any]]:
    db = get_db()
    query: dict = {}
    if graph_id:
        query["graph_id"] = graph_id
    if name:
        query["name"] = {"$regex": name, "$options": "i"}
    cursor = db.assistant.find(query).sort("created_at", -1).skip(offset).limit(limit)
    return [_doc(d) async for d in cursor]


async def count_assistants(graph_id: Optional[str] = None) -> int:
    db = get_db()
    query = {"graph_id": graph_id} if graph_id else {}
    return await db.assistant.count_documents(query)


async def patch_assistant(
    assistant_id: str,
    graph_id: Optional[str] = None,
    config: Optional[dict] = None,
    metadata: Optional[dict] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    db = get_db()
    current = await db.assistant.find_one({"assistant_id": assistant_id})
    if not current:
        return None
    update: dict = {"updated_at": _now(), "version": current.get("version", 1) + 1}
    if graph_id is not None:
        update["graph_id"] = graph_id
    if config is not None:
        update["config"] = config
    if metadata is not None:
        update["metadata"] = {**(current.get("metadata") or {}), **metadata}
    if name is not None:
        update["name"] = name
    if description is not None:
        update["description"] = description
    await db.assistant.update_one({"assistant_id": assistant_id}, {"$set": update})
    return _doc(await db.assistant.find_one({"assistant_id": assistant_id}))


async def delete_assistant(assistant_id: str) -> bool:
    db = get_db()
    result = await db.assistant.delete_one({"assistant_id": assistant_id})
    return result.deleted_count > 0


async def list_assistant_versions(assistant_id: str) -> list[dict[str, Any]]:
    doc = await get_assistant(assistant_id)
    return [doc] if doc else []


# ── Threads ───────────────────────────────────────────────────────────────────

async def create_thread(
    thread_id: Optional[str] = None,
    metadata: dict = {},
    config: dict = {},
    if_exists: str = "raise",
) -> dict[str, Any]:
    db = get_db()
    tid = thread_id or str(_uuid_mod.uuid4())
    existing = await db.thread.find_one({"thread_id": tid})
    if existing:
        if if_exists == "raise":
            raise ValueError(f"Thread {tid} already exists")
        return _doc(existing)
    doc = {
        "thread_id": tid,
        "metadata": metadata,
        "config": config,
        "status": "idle",
        "values": None,
        "interrupts": {},
        "error": None,
        "state_updated_at": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    await db.thread.insert_one(doc)
    return _doc(doc)


async def get_thread(thread_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    doc = await db.thread.find_one({"thread_id": thread_id})
    return _doc(doc) if doc else None


async def search_threads(
    ids: Optional[list[str]] = None,
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
) -> list[dict[str, Any]]:
    db = get_db()
    query: dict = {}
    if ids:
        query["thread_id"] = {"$in": ids}
    if status:
        query["status"] = status
    direction = -1 if sort_order == "desc" else 1
    cursor = db.thread.find(query).sort(sort_by, direction).skip(offset).limit(limit)
    return [_doc(d) async for d in cursor]


async def count_threads(status: Optional[str] = None) -> int:
    db = get_db()
    query = {"status": status} if status else {}
    return await db.thread.count_documents(query)


async def patch_thread(
    thread_id: str,
    metadata: Optional[dict] = None,
    config: Optional[dict] = None,
) -> Optional[dict[str, Any]]:
    db = get_db()
    current = await db.thread.find_one({"thread_id": thread_id})
    if not current:
        return None
    update: dict = {"updated_at": _now()}
    if metadata is not None:
        update["metadata"] = {**(current.get("metadata") or {}), **metadata}
    if config is not None:
        update["config"] = config
    await db.thread.update_one({"thread_id": thread_id}, {"$set": update})
    return _doc(await db.thread.find_one({"thread_id": thread_id}))


async def set_thread_status(thread_id: str, status: str) -> None:
    db = get_db()
    await db.thread.update_one(
        {"thread_id": thread_id},
        {"$set": {"status": status, "updated_at": _now()}},
    )


async def update_thread_snapshot(
    thread_id: str,
    values: Any,
    interrupts: Any,
    error: Optional[bytes] = None,
) -> None:
    db = get_db()
    await db.thread.update_one(
        {"thread_id": thread_id},
        {"$set": {
            "values": values if values is not None else {},
            "interrupts": interrupts if interrupts is not None else {},
            "error": error.decode("utf-8", errors="replace") if error else None,
            "state_updated_at": _now(),
            "updated_at": _now(),
        }},
    )


async def delete_thread(thread_id: str) -> bool:
    db = get_db()
    await db.run.delete_many({"thread_id": thread_id})
    result = await db.thread.delete_one({"thread_id": thread_id})
    return result.deleted_count > 0


async def copy_thread(thread_id: str) -> Optional[dict[str, Any]]:
    original = await get_thread(thread_id)
    if not original:
        return None
    return await create_thread(
        metadata=original.get("metadata") or {},
        config=original.get("config") or {},
    )


async def prune_threads(thread_ids: list[str]) -> int:
    if not thread_ids:
        return 0
    db = get_db()
    await db.run.delete_many({"thread_id": {"$in": thread_ids}})
    result = await db.thread.delete_many({"thread_id": {"$in": thread_ids}})
    return result.deleted_count


# ── Runs ──────────────────────────────────────────────────────────────────────

async def create_run(
    thread_id: str,
    assistant_id: str,
    metadata: dict = {},
    kwargs: dict = {},
    multitask_strategy: str = "reject",
    run_id: Optional[str] = None,
    status: str = "pending",
) -> dict[str, Any]:
    db = get_db()
    doc = {
        "run_id": run_id or str(_uuid_mod.uuid4()),
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "metadata": metadata,
        "kwargs": kwargs,
        "multitask_strategy": multitask_strategy,
        "status": status,
        "created_at": _now(),
        "updated_at": _now(),
    }
    await db.run.insert_one(doc)
    return _doc(doc)


async def create_run_and_set_busy(
    thread_id: str,
    assistant_id: str,
    metadata: dict = {},
    kwargs: dict = {},
    multitask_strategy: str = "reject",
    run_id: Optional[str] = None,
    status: str = "pending",
) -> dict[str, Any]:
    run = await create_run(
        thread_id, assistant_id, metadata, kwargs, multitask_strategy,
        run_id=run_id, status=status,
    )
    await set_thread_status(thread_id, "busy")
    return run


async def get_run(run_id: str) -> Optional[dict[str, Any]]:
    db = get_db()
    doc = await db.run.find_one({"run_id": run_id})
    return _doc(doc) if doc else None


async def get_graph_id_for_thread(thread_id: str) -> Optional[str]:
    db = get_db()
    run = await db.run.find_one(
        {"thread_id": thread_id},
        sort=[("created_at", -1)],
    )
    if not run:
        return None
    assistant = await db.assistant.find_one({"assistant_id": run["assistant_id"]})
    return assistant["graph_id"] if assistant else None


async def list_runs(thread_id: str) -> list[dict[str, Any]]:
    db = get_db()
    cursor = db.run.find({"thread_id": thread_id}).sort("created_at", -1)
    return [_doc(d) async for d in cursor]


async def update_run_status(run_id: str, status: str) -> None:
    db = get_db()
    await db.run.update_one(
        {"run_id": run_id},
        {"$set": {"status": status, "updated_at": _now()}},
    )


async def delete_run(run_id: str) -> bool:
    db = get_db()
    result = await db.run.delete_one({"run_id": run_id})
    return result.deleted_count > 0


# ── Store ─────────────────────────────────────────────────────────────────────

def _ns_to_prefix(namespace: list[str]) -> str:
    return "/".join(namespace)


def _store_doc(doc: dict) -> dict:
    if not doc:
        return {}
    out = dict(doc)
    out.pop("_id", None)
    prefix = out.pop("prefix", "")
    out["namespace"] = prefix.split("/") if prefix else []
    return out


async def store_put(
    namespace: list[str],
    key: str,
    value: dict,
    ttl_minutes: Optional[int] = None,
) -> dict[str, Any]:
    db = get_db()
    prefix = _ns_to_prefix(namespace)
    now = _now()
    expires_at = None
    if ttl_minutes is not None:
        expires_at = now + timedelta(minutes=ttl_minutes)
    doc = {
        "prefix": prefix,
        "key": key,
        "value": value,
        "ttl_minutes": ttl_minutes,
        "expires_at": expires_at,
        "created_at": now,
        "updated_at": now,
    }
    await db.store.replace_one({"prefix": prefix, "key": key}, doc, upsert=True)
    return _store_doc(doc)


async def store_get(namespace: list[str], key: str) -> Optional[dict[str, Any]]:
    db = get_db()
    prefix = _ns_to_prefix(namespace)
    query = {
        "prefix": prefix,
        "key": key,
        "$or": [{"expires_at": None}, {"expires_at": {"$gt": _now()}}],
    }
    doc = await db.store.find_one(query)
    return _store_doc(doc) if doc else None


async def store_delete(namespace: list[str], key: str) -> bool:
    db = get_db()
    prefix = _ns_to_prefix(namespace)
    result = await db.store.delete_one({"prefix": prefix, "key": key})
    return result.deleted_count > 0


async def store_search(
    namespace_prefix: list[str],
    limit: int = 10,
    offset: int = 0,
    filter: Optional[dict] = None,
) -> list[dict[str, Any]]:
    db = get_db()
    prefix = _ns_to_prefix(namespace_prefix)
    query: dict = {
        "prefix": {"$regex": f"^{prefix}"},
        "$or": [{"expires_at": None}, {"expires_at": {"$gt": _now()}}],
    }
    cursor = db.store.find(query).sort([("prefix", 1), ("key", 1)]).skip(offset).limit(limit)
    return [_store_doc(d) async for d in cursor]


async def store_list_namespaces(
    prefix: Optional[list[str]] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[list[str]]:
    db = get_db()
    query: dict = {}
    if prefix:
        p = _ns_to_prefix(prefix)
        query["prefix"] = {"$regex": f"^{p}"}
    pipeline = [
        {"$match": query},
        {"$group": {"_id": "$prefix"}},
        {"$sort": {"_id": 1}},
        {"$skip": offset},
        {"$limit": limit},
    ]
    result = []
    async for doc in db.store.aggregate(pipeline):
        result.append(doc["_id"].split("/"))
    return result
