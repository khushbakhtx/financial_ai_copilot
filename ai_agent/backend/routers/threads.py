from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query
from ..models import (
    Thread, ThreadCreate, ThreadPatch,
    ThreadSearchRequest, ThreadCountRequest,
    ThreadPruneRequest, ThreadPruneResponse,
    ThreadState, ThreadStateUpdate, ThreadStateUpdateResponse,
    ThreadStateSearch, CheckpointConfig,
)
from .. import database as db
from ..graphs import get_graph

router = APIRouter(prefix="/threads", tags=["Threads"])


def _to_model(d: dict) -> Thread:
    return Thread(**d)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@router.post("", response_model=Thread)
async def create_thread(body: ThreadCreate):
    try:
        row = await db.create_thread(
            thread_id=body.thread_id,
            metadata=body.metadata,
            if_exists=body.if_exists,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _to_model(row)


@router.post("/search", response_model=list[Thread])
async def search_threads(body: ThreadSearchRequest):
    rows = await db.search_threads(
        ids=body.ids,
        limit=body.limit,
        offset=body.offset,
        status=body.status,
        sort_by=body.sort_by,
        sort_order=body.sort_order,
    )
    return [_to_model(r) for r in rows]


@router.post("/count")
async def count_threads(body: ThreadCountRequest):
    return await db.count_threads(status=body.status)


@router.post("/prune", response_model=ThreadPruneResponse)
async def prune_threads(body: ThreadPruneRequest):
    pruned = await db.prune_threads(thread_ids=body.thread_ids)
    return ThreadPruneResponse(pruned_count=pruned)


@router.get("/{thread_id}", response_model=Thread)
async def get_thread(thread_id: str):
    row = await db.get_thread(thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    return _to_model(row)


@router.patch("/{thread_id}", response_model=Thread)
async def patch_thread(thread_id: str, body: ThreadPatch):
    row = await db.patch_thread(
        thread_id=thread_id,
        metadata=body.metadata,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    return _to_model(row)


@router.delete("/{thread_id}")
async def delete_thread(thread_id: str):
    deleted = await db.delete_thread(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {}


@router.post("/{thread_id}/copy", response_model=Thread)
async def copy_thread(thread_id: str):
    row = await db.copy_thread(thread_id)
    if not row:
        raise HTTPException(status_code=404, detail="Thread not found")
    return _to_model(row)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _checkpoint_config(thread_id: str, cp: Optional[CheckpointConfig] = None) -> dict:
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if cp and cp.checkpoint_id:
        configurable["checkpoint_id"] = cp.checkpoint_id
    if cp and cp.checkpoint_ns is not None:
        configurable["checkpoint_ns"] = cp.checkpoint_ns
    return {"configurable": configurable}


async def _get_graph_for_thread(thread_id: str):
    """Resolve graph for a thread via a single JOIN query (thread → run → assistant)."""
    graph_id = await db.get_graph_id_for_thread(thread_id)
    if not graph_id:
        raise HTTPException(status_code=404, detail="No runs found for thread")
    try:
        return get_graph(graph_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


_EMPTY_THREAD_STATE = ThreadState(values={}, next=[], tasks=[], interrupts=[])


@router.get("/{thread_id}/state", response_model=ThreadState)
async def get_thread_state(thread_id: str):
    thread = await db.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    graph_id = await db.get_graph_id_for_thread(thread_id)
    if not graph_id:
        return _EMPTY_THREAD_STATE
    try:
        graph = get_graph(graph_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    config = _checkpoint_config(thread_id)
    try:
        snapshot = await graph.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from ..streaming import _serialize

    interrupts = []
    tasks_out = []
    for task in (snapshot.tasks or []):
        task_interrupts = []
        for i in (getattr(task, "interrupts", None) or []):
            serialized = {"value": _serialize(getattr(i, "value", i))}
            task_interrupts.append(serialized)
            interrupts.append(serialized)
        tasks_out.append({
            "id": getattr(task, "id", ""),
            "name": getattr(task, "name", ""),
            "error": getattr(task, "error", None),
            "interrupts": task_interrupts,
        })

    return ThreadState(
        values=_serialize(snapshot.values),
        next=list(snapshot.next or []),
        tasks=tasks_out,
        checkpoint=dict(snapshot.config.get("configurable", {})) if snapshot.config else None,
        metadata=snapshot.metadata or {},
        created_at=snapshot.created_at,
        parent_checkpoint=dict(snapshot.parent_config.get("configurable", {})) if snapshot.parent_config else None,
        interrupts=interrupts,
    )


@router.post("/{thread_id}/state", response_model=ThreadStateUpdateResponse)
async def update_thread_state(thread_id: str, body: ThreadStateUpdate):
    thread = await db.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    graph = await _get_graph_for_thread(thread_id)
    config = _checkpoint_config(thread_id, body.checkpoint)
    try:
        new_config = await graph.aupdate_state(
            config,
            body.values,
            as_node=body.as_node,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ThreadStateUpdateResponse(
        checkpoint=dict(new_config.get("configurable", {})) if new_config else None,
    )


@router.get("/{thread_id}/state/{checkpoint_id}", response_model=ThreadState)
async def get_thread_state_at_checkpoint(thread_id: str, checkpoint_id: str):
    thread = await db.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    graph_id = await db.get_graph_id_for_thread(thread_id)
    if not graph_id:
        return _EMPTY_THREAD_STATE
    try:
        graph = get_graph(graph_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    config = _checkpoint_config(thread_id, CheckpointConfig(checkpoint_id=checkpoint_id))
    try:
        snapshot = await graph.aget_state(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from ..streaming import _serialize
    return ThreadState(
        values=_serialize(snapshot.values),
        next=list(snapshot.next or []),
        checkpoint=dict(config.get("configurable", {})),
        metadata=snapshot.metadata or {},
        created_at=snapshot.created_at,
    )


@router.post("/{thread_id}/state/checkpoint")
async def get_state_at_checkpoint_post(thread_id: str, body: CheckpointConfig):
    return await get_thread_state_at_checkpoint(thread_id, body.checkpoint_id or "")


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@router.get("/{thread_id}/history")
@router.post("/{thread_id}/history")
async def get_thread_history(thread_id: str, body: Optional[ThreadStateSearch] = None):
    thread = await db.get_thread(thread_id)
    if not thread:
        # Return empty history for unknown threads (e.g. threads from a previous
        # deployment stored in the client's localStorage) rather than 404-ing.
        return []
    graph_id = await db.get_graph_id_for_thread(thread_id)
    if not graph_id:
        return []
    try:
        graph = get_graph(graph_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    config = _checkpoint_config(thread_id)
    limit = body.limit if body else 10
    # `before` is a CheckpointConfig object — build LangGraph configurable dict
    before_config = None
    if body and body.before and body.before.checkpoint_id:
        configurable: dict = {"checkpoint_id": body.before.checkpoint_id}
        if body.before.checkpoint_ns is not None:
            configurable["checkpoint_ns"] = body.before.checkpoint_ns
        before_config = {"configurable": configurable}

    from ..streaming import _serialize

    def _serialize_tasks(tasks) -> list:
        result = []
        for t in (tasks or []):
            interrupts = []
            for i in (getattr(t, "interrupts", None) or []):
                interrupts.append({"value": _serialize(getattr(i, "value", i))})
            result.append({
                "id": getattr(t, "id", ""),
                "name": getattr(t, "name", ""),
                "error": getattr(t, "error", None),
                "interrupts": interrupts,
            })
        return result

    history = []
    try:
        async for state in graph.aget_state_history(config, limit=limit, before=before_config):
            all_interrupts = []
            for t in (state.tasks or []):
                for i in (getattr(t, "interrupts", None) or []):
                    all_interrupts.append({"value": _serialize(getattr(i, "value", i))})

            history.append({
                "values": _serialize(state.values),
                "next": list(state.next or []),
                "tasks": _serialize_tasks(state.tasks),
                "checkpoint": dict(state.config.get("configurable", {})) if state.config else None,
                "metadata": state.metadata or {},
                "created_at": state.created_at,
                "parent_checkpoint": dict(state.parent_config.get("configurable", {})) if state.parent_config else None,
                "interrupts": all_interrupts,
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return history


# ---------------------------------------------------------------------------
# Stream (GET — reconnect to existing run's stream)
# ---------------------------------------------------------------------------

@router.get("/{thread_id}/stream")
async def stream_thread(thread_id: str):
    """Reconnect endpoint — returns current thread state as a single SSE snapshot."""
    from fastapi.responses import StreamingResponse
    from ..streaming import _sse

    thread = await db.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    async def generate():
        yield _sse("values", {"thread_id": thread_id, "status": thread["status"]})
        yield _sse("end", {})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
