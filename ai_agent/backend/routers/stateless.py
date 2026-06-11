"""Stateless run endpoints — no thread state persistence."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse

from ..models import RunCreateStreamingStateless, RunCreateStateless, RunsCancel
from .. import database as db
from .. import run_manager
from ..graphs import get_graph
from ..streaming import stream_graph, stream_graph_wait, _serialize, _sse

router = APIRouter(prefix="/runs", tags=["Stateless Runs"])


def _normalize_modes(raw: Any) -> list[str]:
    modes = [raw] if isinstance(raw, str) else list(raw) if raw else ["values"]
    if "messages-tuple" in modes:
        modes = [m if m != "messages-tuple" else "messages" for m in modes]
    if "messages" not in modes:
        modes = ["messages"] + modes
    return modes


async def _resolve_graph(assistant_id: str) -> Any:
    row = await db.get_assistant(assistant_id)
    if row:
        return get_graph(row["graph_id"])
    try:
        return get_graph(assistant_id)
    except ValueError:
        pass
    raise HTTPException(status_code=404, detail=f"Assistant not found: {assistant_id}")


def _build_input(body: RunCreateStateless) -> Any:
    if body.command:
        from langgraph.types import Command
        return Command(
            resume=body.command.resume,
            update=body.command.update,
            goto=body.command.goto,
        )
    return body.input


# ---------------------------------------------------------------------------
# Stateless stream
# ---------------------------------------------------------------------------

@router.post("/stream")
async def stream_stateless(body: RunCreateStreamingStateless):
    graph = await _resolve_graph(body.assistant_id)
    # Stateless: use a throwaway thread_id
    thread_id = str(uuid.uuid4())
    lg_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if body.config and body.config.configurable:
        lg_config["configurable"].update(body.config.configurable)
    if body.config and body.config.recursion_limit:
        lg_config["recursion_limit"] = body.config.recursion_limit

    stream_modes = _normalize_modes(body.stream_mode)
    input_data = _build_input(body)

    async def generate():
        async for chunk in stream_graph(
            graph, input_data, lg_config, stream_modes,
            stream_subgraphs=body.stream_subgraphs,
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


# ---------------------------------------------------------------------------
# Stateless wait (blocking)
# ---------------------------------------------------------------------------

@router.post("/wait")
async def wait_stateless(body: RunCreateStateless):
    graph = await _resolve_graph(body.assistant_id)
    thread_id = str(uuid.uuid4())
    lg_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if body.config and body.config.configurable:
        lg_config["configurable"].update(body.config.configurable)

    input_data = _build_input(body)
    try:
        result = await stream_graph_wait(graph, input_data, lg_config)
        return _serialize(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Stateless fire-and-forget
# ---------------------------------------------------------------------------

@router.post("")
async def create_stateless(body: RunCreateStateless):
    graph = await _resolve_graph(body.assistant_id)
    thread_id = str(uuid.uuid4())
    lg_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if body.config and body.config.configurable:
        lg_config["configurable"].update(body.config.configurable)

    input_data = _build_input(body)
    run_id = str(uuid.uuid4())
    run_manager.start(run_id, thread_id, graph, input_data, lg_config)
    return {"run_id": run_id, "thread_id": thread_id, "status": "pending"}


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

@router.post("/batch")
async def batch_runs(body: dict):
    raise HTTPException(status_code=501, detail="Batch runs not implemented")


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

@router.post("/cancel", status_code=204)
async def cancel_runs(body: RunsCancel):
    for run_id in body.run_ids:
        await run_manager.cancel(run_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Crons (stubs)
# ---------------------------------------------------------------------------

@router.post("/crons")
async def create_cron():
    raise HTTPException(status_code=501, detail="Crons not implemented")


@router.post("/crons/search")
async def search_crons():
    return []


@router.post("/crons/count")
async def count_crons():
    return 0


@router.patch("/crons/{cron_id}")
@router.delete("/crons/{cron_id}")
async def manage_cron(cron_id: str):
    raise HTTPException(status_code=501, detail="Crons not implemented")
