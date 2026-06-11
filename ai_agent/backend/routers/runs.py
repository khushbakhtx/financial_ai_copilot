"""Thread-scoped run endpoints — the core of the backend."""
from __future__ import annotations

import asyncio
from typing import Any, Optional
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from deepagent_copilot.ai_agent.backend.models import (
    Run, RunCreateStateful, RunCreateStreamingStateful,
    RunsCancel,
)
import deepagent_copilot.ai_agent.backend.database as db
import deepagent_copilot.ai_agent.backend.run_manager as run_manager
from deepagent_copilot.ai_agent.backend.graphs import get_graph
from deepagent_copilot.ai_agent.backend.streaming import stream_graph, stream_graph_wait, _serialize, _sse

router = APIRouter(prefix="/threads/{thread_id}/runs", tags=["Thread Runs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_modes(raw: Any) -> list[str]:
    modes = [raw] if isinstance(raw, str) else list(raw) if raw else ["values"]
    # LangGraph LOCAL uses "messages" mode (not "messages-tuple") to register
    # StreamMessagesHandler and emit per-message/token chunks. Replace any
    # "messages-tuple" requests with "messages" — the SDK receives SSE event
    # name "messages" either way (our _sse_event_name maps "messages" → "messages").
    if "messages-tuple" in modes:
        modes = [m if m != "messages-tuple" else "messages" for m in modes]
    if "messages" not in modes:
        modes = ["messages"] + modes
    return modes


async def _resolve_graph(assistant_id: str) -> Any:
    """Look up the graph for an assistant_id or graph_id string."""
    row = await db.get_assistant(assistant_id)
    if row:
        return get_graph(row["graph_id"])
    try:
        return get_graph(assistant_id)
    except ValueError:
        pass
    raise HTTPException(status_code=404, detail=f"Assistant not found: {assistant_id}")


async def _resolve_assistant_id(assistant_id: str) -> str:
    """Return a real assistant_id UUID (look up by graph_id if needed)."""
    row = await db.get_assistant(assistant_id)
    if row:
        return row["assistant_id"]
    row = await db.get_assistant_by_graph(assistant_id)
    if row:
        return row["assistant_id"]
    raise HTTPException(status_code=404, detail=f"Assistant not found: {assistant_id}")


async def _resolve_assistant_row(assistant_id: str) -> dict:
    """Return the full assistant row in one lookup (avoids calling get_assistant twice)."""
    row = await db.get_assistant(assistant_id)
    if not row:
        row = await db.get_assistant_by_graph(assistant_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Assistant not found: {assistant_id}")
    return row


def _build_lg_config(thread_id: str, body: RunCreateStateful) -> dict:
    """Build the LangGraph RunnableConfig from the run request."""
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if body.config and body.config.configurable:
        configurable.update(body.config.configurable)
    config: dict[str, Any] = {"configurable": configurable}
    if body.config and body.config.recursion_limit:
        config["recursion_limit"] = body.config.recursion_limit
    if body.config and body.config.tags:
        config["tags"] = body.config.tags
    return config


def _build_input(body: RunCreateStateful) -> Any:
    """Build the graph input — either a Command (resume) or the raw input."""
    if body.command:
        from langgraph.types import Command
        return Command(
            resume=body.command.resume,
            update=body.command.update,
            goto=body.command.goto,
        )
    return body.input


async def _ensure_thread(thread_id: str, if_not_exists: str) -> None:
    thread = await db.get_thread(thread_id)
    if not thread:
        if if_not_exists == "create":
            await db.create_thread(thread_id=thread_id, if_exists="do_nothing")
        else:
            raise HTTPException(status_code=404, detail="Thread not found")


# ---------------------------------------------------------------------------
# Background finalization (runs after end is sent to client)
# ---------------------------------------------------------------------------

async def _finalize_snapshot(thread_id: str, graph: Any, lg_config: dict) -> None:
    """Background task: call aget_state over network, persist snapshot, fix interrupt status.

    This runs AFTER the end SSE event so it never blocks the client.
    aget_state hits the checkpoint store (remote Postgres) which can take
    several seconds — keeping it off the hot path removes the visible 7-second
    delay between the agent finishing and the UI spinner stopping.
    """
    try:
        snapshot = await graph.aget_state(lg_config)
        interrupts: Any = {}
        for task in (snapshot.tasks or []):
            for intr in (getattr(task, "interrupts", None) or []):
                interrupts = _serialize(getattr(intr, "value", intr))
                break
            if interrupts:
                break
        await db.update_thread_snapshot(
            thread_id, _serialize(snapshot.values), interrupts
        )
        # Correct status from "idle" → "interrupted" if the graph paused for input
        if snapshot.tasks and any(getattr(t, "interrupts", None) for t in snapshot.tasks):
            await db.set_thread_status(thread_id, "interrupted")
    except Exception:
        pass  # thread status already set to idle; snapshot update is best-effort


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

@router.post("/stream")
async def stream_run(thread_id: str, body: RunCreateStreamingStateful):
    # One round-trip: get thread (needed for existence check + multitask)
    thread = await db.get_thread(thread_id)
    if not thread:
        if body.if_not_exists == "create":
            thread = await db.create_thread(thread_id=thread_id, if_exists="do_nothing")
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

    # Multitask strategy enforcement (reuse thread fetched above — no extra round-trip)
    if thread.get("status") == "busy":
        strategy = body.multitask_strategy
        if strategy == "reject":
            raise HTTPException(status_code=409, detail="Thread is already busy")
        if strategy == "interrupt":
            runs = await db.list_runs(thread_id)
            for r in runs:
                if r["status"] == "running":
                    await run_manager.cancel(r["run_id"])
                    await db.update_run_status(r["run_id"], "interrupted")
                    break

    # One round-trip: resolve assistant (get graph + assistant_id in one query)
    assistant_row = await db.get_assistant(body.assistant_id)
    if not assistant_row:
        assistant_row = await db.get_assistant_by_graph(body.assistant_id)
    if not assistant_row:
        raise HTTPException(status_code=404, detail=f"Assistant not found: {body.assistant_id}")
    try:
        graph = get_graph(assistant_row["graph_id"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    assistant_id = assistant_row["assistant_id"]

    # One round-trip: create run + set thread busy atomically
    run = await db.create_run_and_set_busy(
        thread_id=thread_id,
        assistant_id=assistant_id,
        metadata=body.metadata,
        kwargs={"stream_mode": body.stream_mode},
        multitask_strategy=body.multitask_strategy,
    )
    run_id = run["run_id"]
    stream_modes = _normalize_modes(body.stream_mode)
    input_data = _build_input(body)
    lg_config = _build_lg_config(thread_id, body)

    async def generate():
        # Fire-and-forget status update — don't block first SSE event on Atlas round-trip
        asyncio.create_task(db.update_run_status(run_id, "running"))
        try:
            async for chunk in stream_graph(
                graph, input_data, lg_config, stream_modes,
                emit_end=False,
                stream_subgraphs=body.stream_subgraphs,
            ):
                yield chunk
            await db.update_run_status(run_id, "success")
        except Exception as e:
            yield _sse("error", {"message": str(e)})
            await db.update_run_status(run_id, "error")
            await db.set_thread_status(thread_id, "error")
            try:
                await db.update_thread_snapshot(thread_id, None, {}, str(e).encode())
            except Exception:
                pass
            yield _sse("end", {})
            return

        # Set idle immediately — two fast DB writes (~200 ms each to Render).
        # The heavy aget_state() checkpoint read happens in the background so it
        # never delays the end event the client is waiting for.
        await db.set_thread_status(thread_id, "idle")

        # Schedule background finalization: aget_state → snapshot → maybe "interrupted"
        asyncio.create_task(_finalize_snapshot(thread_id, graph, lg_config))

        # end arrives < 500 ms after the graph finishes regardless of DB latency
        yield _sse("end", {})

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
# Wait (blocking)
# ---------------------------------------------------------------------------

@router.post("/wait")
async def wait_run(thread_id: str, body: RunCreateStateful):
    await _ensure_thread(thread_id, body.if_not_exists)
    assistant_row = await _resolve_assistant_row(body.assistant_id)
    try:
        graph = get_graph(assistant_row["graph_id"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    run = await db.create_run_and_set_busy(
        thread_id=thread_id,
        assistant_id=assistant_row["assistant_id"],
        metadata=body.metadata,
        multitask_strategy=body.multitask_strategy,
    )
    run_id = run["run_id"]
    input_data = _build_input(body)
    lg_config = _build_lg_config(thread_id, body)

    await db.update_run_status(run_id, "running")
    try:
        result = await stream_graph_wait(graph, input_data, lg_config)
        await db.update_run_status(run_id, "success")
        # Persist snapshot onto thread row
        try:
            snapshot = await graph.aget_state(lg_config)
            interrupts: Any = {}
            for task in (snapshot.tasks or []):
                for intr in (getattr(task, "interrupts", None) or []):
                    interrupts = _serialize(getattr(intr, "value", intr))
                    break
                if interrupts:
                    break
            await db.update_thread_snapshot(
                thread_id, _serialize(snapshot.values), interrupts
            )
            if snapshot.tasks and any(
                getattr(t, "interrupts", None) for t in snapshot.tasks
            ):
                await db.set_thread_status(thread_id, "interrupted")
            else:
                await db.set_thread_status(thread_id, "idle")
        except Exception:
            await db.set_thread_status(thread_id, "idle")
        return _serialize(result)
    except Exception as e:
        await db.update_run_status(run_id, "error")
        await db.set_thread_status(thread_id, "error")
        await db.update_thread_snapshot(thread_id, None, {}, str(e).encode())
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Background (fire-and-forget)
# ---------------------------------------------------------------------------

@router.post("", response_model=Run)
async def create_run(thread_id: str, body: RunCreateStateful):
    await _ensure_thread(thread_id, body.if_not_exists)
    assistant_row = await _resolve_assistant_row(body.assistant_id)
    try:
        graph = get_graph(assistant_row["graph_id"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    run = await db.create_run_and_set_busy(
        thread_id=thread_id,
        assistant_id=assistant_row["assistant_id"],
        metadata=body.metadata,
        multitask_strategy=body.multitask_strategy,
    )
    input_data = _build_input(body)
    lg_config = _build_lg_config(thread_id, body)

    run_manager.start(run["run_id"], thread_id, graph, input_data, lg_config)
    return Run(**run)


# ---------------------------------------------------------------------------
# List / Get / Delete / Cancel / Join / Stream existing run
# ---------------------------------------------------------------------------

@router.get("", response_model=list[Run])
async def list_runs(thread_id: str):
    rows = await db.list_runs(thread_id)
    return [Run(**r) for r in rows]


@router.get("/{run_id}", response_model=Run)
async def get_run(thread_id: str, run_id: str):
    row = await db.get_run(run_id)
    if not row or row["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return Run(**row)


@router.delete("/{run_id}")
async def delete_run(thread_id: str, run_id: str):
    # Cancel if still running
    await run_manager.cancel(run_id)
    deleted = await db.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {}


@router.post("/{run_id}/cancel")
async def cancel_run(
    thread_id: str,
    run_id: str,
    action: str = "interrupt",
    wait: bool = False,
):
    row = await db.get_run(run_id)
    if not row or row["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    cancelled = await run_manager.cancel(run_id)
    if cancelled:
        if action == "rollback":
            await db.delete_run(run_id)
            await db.set_thread_status(thread_id, "idle")
        else:
            await db.update_run_status(run_id, "interrupted")
            await db.set_thread_status(thread_id, "interrupted")
    return {}


@router.get("/{run_id}/join")
async def join_run(thread_id: str, run_id: str):
    """Block until the run completes, then return the run record."""
    import asyncio
    for _ in range(300):  # max 30s polling
        row = await db.get_run(run_id)
        if not row:
            raise HTTPException(status_code=404, detail="Run not found")
        if row["status"] not in ("pending", "running"):
            return Run(**row)
        await asyncio.sleep(0.1)
    raise HTTPException(status_code=408, detail="Run did not complete in time")


@router.get("/{run_id}/stream")
async def stream_existing_run(thread_id: str, run_id: str):
    """Replay and/or live-attach to a background run's SSE stream.

    Behaviour:
    - If the run is still in progress: replay all events emitted so far, then
      stream live events until the run ends.
    - If the run already finished: replay the full event history and close.
    - If the run was never tracked in this process (restart): fall back to a
      current-state snapshot so the client at least gets final values.
    """
    row = await db.get_run(run_id)
    if not row or row["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")

    async def generate():
        replay = run_manager.get_replay(run_id)

        if replay is None:
            # Run was never started in this process (e.g. server restarted).
            # Fall back to a snapshot of the current thread state.
            try:
                thread_row = await db.get_thread(thread_id)
                if thread_row:
                    assistant = await db.get_assistant(row["assistant_id"])
                    if assistant:
                        graph = get_graph(assistant["graph_id"])
                        lg_config = {"configurable": {"thread_id": thread_id}}
                        snapshot = await graph.aget_state(lg_config)
                        yield _sse("values", _serialize(snapshot.values))
            except Exception as e:
                yield _sse("error", {"message": str(e)})
            yield _sse("end", {})
            return

        # Subscribe BEFORE snapshotting the replay buffer so we don't miss any
        # chunks published in the gap between the two steps.
        if not run_manager.is_finished(run_id):
            queue = run_manager.subscribe(run_id)
        else:
            queue = None

        # Send all chunks emitted before we subscribed
        for chunk in list(replay):
            yield chunk

        if queue is None:
            # Run already finished — end event is already in the replay buffer
            return

        # Stream live events until the run ends
        while True:
            chunk = await queue.get()
            if chunk is None:  # sentinel — run finished
                return
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Crons (stub — returns 501 if not implemented)
# ---------------------------------------------------------------------------

@router.post("/crons")
async def create_thread_cron(thread_id: str):
    raise HTTPException(status_code=501, detail="Crons not implemented")
