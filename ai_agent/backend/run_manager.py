"""Background run execution with replay-buffer fan-out and cancellation support."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from deepagent_copilot.ai_agent.backend.database import update_run_status, set_thread_status, update_thread_snapshot

log = logging.getLogger("run_manager")

_active: dict[str, asyncio.Task] = {}
_subscribers: dict[str, list[asyncio.Queue]] = {}
_replay: dict[str, list[str]] = {}
_REPLAY_SENTINEL = None


def subscribe(run_id: str) -> asyncio.Queue:
    """Register a new subscriber queue for a run's SSE events."""
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(run_id, []).append(q)
    return q


def _publish(run_id: str, chunk: str) -> None:
    """Append chunk to replay buffer and push to all subscriber queues."""
    _replay.setdefault(run_id, []).append(chunk)
    for q in _subscribers.get(run_id, []):
        q.put_nowait(chunk)


def _close(run_id: str) -> None:
    """Signal all subscribers that the stream is done, then clean up."""
    for q in _subscribers.get(run_id, []):
        q.put_nowait(_REPLAY_SENTINEL)
    _subscribers.pop(run_id, None)


def get_replay(run_id: str) -> list[str] | None:
    """Return already-emitted chunks for a run (None if run never started here)."""
    return _replay.get(run_id)


def is_finished(run_id: str) -> bool:
    """True if the run task is done (or was never started in this process)."""
    task = _active.get(run_id)
    return task is None or task.done()


async def _execute(
    run_id: str,
    thread_id: str,
    graph: Any,
    input_data: Any,
    config: dict,
) -> None:
    from deepagent_copilot.ai_agent.backend.streaming import stream_graph, _sse, _serialize

    _replay[run_id] = []

    try:
        await update_run_status(run_id, "running")
        await set_thread_status(thread_id, "busy")

        async for chunk in stream_graph(
            graph, input_data, config,
            stream_modes=["values", "messages", "updates"],
            emit_end=False,
            stream_subgraphs=True,
        ):
            _publish(run_id, chunk)

        await update_run_status(run_id, "success")

        try:
            snapshot = await graph.aget_state(config)
            interrupts: Any = {}
            for task in (snapshot.tasks or []):
                for intr in (getattr(task, "interrupts", None) or []):
                    interrupts = _serialize(getattr(intr, "value", intr))
                    break
                if interrupts:
                    break
            await update_thread_snapshot(
                thread_id, _serialize(snapshot.values), interrupts
            )
            if snapshot.tasks and any(
                getattr(t, "interrupts", None) for t in snapshot.tasks
            ):
                await set_thread_status(thread_id, "interrupted")
            else:
                await set_thread_status(thread_id, "idle")
        except Exception:
            await set_thread_status(thread_id, "idle")

        _publish(run_id, _sse("end", {}))

    except asyncio.CancelledError:
        await update_run_status(run_id, "interrupted")
        await set_thread_status(thread_id, "interrupted")
        from deepagent_copilot.ai_agent.backend.streaming import _sse as _sse2
        _publish(run_id, _sse2("end", {}))
    except Exception as e:
        log.exception("run failed  run_id=%s  thread_id=%s", run_id, thread_id)
        await update_run_status(run_id, "error")
        await set_thread_status(thread_id, "error")
        await update_thread_snapshot(thread_id, None, {}, str(e).encode())
        from deepagent_copilot.ai_agent.backend.streaming import _sse as _sse3
        _publish(run_id, _sse3("error", {"message": str(e)}))
        _publish(run_id, _sse3("end", {}))
    finally:
        _active.pop(run_id, None)
        _close(run_id)
        async def _cleanup():
            await asyncio.sleep(30)
            _replay.pop(run_id, None)
        asyncio.create_task(_cleanup())


def start(
    run_id: str,
    thread_id: str,
    graph: Any,
    input_data: Any,
    config: dict,
) -> asyncio.Task:
    task = asyncio.create_task(
        _execute(run_id, thread_id, graph, input_data, config)
    )
    _active[run_id] = task
    return task


async def cancel(run_id: str) -> bool:
    task = _active.get(run_id)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True
    return False


def is_running(run_id: str) -> bool:
    task = _active.get(run_id)
    return task is not None and not task.done()
