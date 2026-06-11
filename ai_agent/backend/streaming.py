"""SSE streaming utilities that replicate LangGraph Platform's wire format."""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from langchain_core.messages import BaseMessage

# LangGraph internal types that need special handling
try:
    from langgraph.types import Overwrite
    _OVERWRITE = Overwrite
except ImportError:
    _OVERWRITE = None

try:
    from langgraph.types import Send as LGSend
    _LG_SEND = LGSend
except ImportError:
    _LG_SEND = None


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """Recursively convert LangChain/LangGraph objects to JSON-safe types."""
    # Plain JSON-safe scalars — return immediately
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # LangGraph Overwrite wrapper — unwrap the inner value
    if _OVERWRITE is not None and isinstance(obj, _OVERWRITE):
        inner = getattr(obj, "value", None)
        if inner is None:
            # Some versions store it differently
            try:
                inner = next(iter(obj))
            except Exception:
                return None
        return _serialize(inner)

    # LangGraph Send — serialize as plain dict
    if _LG_SEND is not None and isinstance(obj, _LG_SEND):
        return {"node": obj.node, "input": _serialize(obj.args)}

    # LangChain messages
    if isinstance(obj, BaseMessage):
        d = obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
        if "type" not in d and hasattr(obj, "type"):
            d["type"] = obj.type
        return _serialize(d)

    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]

    # Pydantic models
    if hasattr(obj, "model_dump"):
        return _serialize(obj.model_dump())
    if hasattr(obj, "dict"):
        return _serialize(obj.dict())

    # Catch-all: any unknown object — try __dict__, then str
    if hasattr(obj, "__dict__"):
        return _serialize(obj.__dict__)

    try:
        import json
        json.dumps(obj)   # test if already serializable
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _sse(event: str, data: Any) -> str:
    """Format one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(_serialize(data))}\n\n"


def _sse_event_name(mode: str, namespace=None) -> str:
    """Map a LangGraph stream mode to the SSE event name the SDK expects.

    The SDK's StreamManager listens for event="messages" (not "messages-tuple").
    messages-tuple is the LangGraph mode name; "messages" is the SSE wire name.
    With subgraphs, namespaced events are "messages|ns1|ns2" etc.
    """
    if mode == "messages-tuple":
        base = "messages"
    else:
        base = mode
    if namespace:
        ns_str = "|".join(str(n).split(":")[0] for n in namespace)
        return f"{base}|{ns_str}"
    return base


# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------

async def stream_graph(
    graph: Any,
    input_data: Any,
    config: dict,
    stream_modes: list[str],
    emit_end: bool = True,
    stream_subgraphs: bool = False,
) -> AsyncIterator[str]:
    """Stream a compiled LangGraph graph as SSE events.

    Yields SSE-formatted strings compatible with the LangGraph SDK.
    Event sequence:
      1. metadata  – {run_id}
      2. <mode>    – one frame per chunk (messages / updates / values / …)
      3. end       – {} signals completion  (only if emit_end=True)
      On error:
      error        – {message}

    Set emit_end=False when the caller needs to do async work (DB writes,
    interrupt detection) before the client sees the final end event.

    Set stream_subgraphs=True to pass subgraphs=True to graph.astream(),
    which enables token-by-token streaming from subagent LLM calls.
    LangGraph yields (namespace, mode, chunk) 3-tuples in this case for
    multi-mode, or (namespace, chunk) 2-tuples for single-mode.
    """
    run_id = str(uuid.uuid4())
    yield _sse("metadata", {"run_id": run_id})

    try:
        # Always pass stream_mode as a list so LangGraph yields (mode, chunk) tuples
        modes: list[str] | str = stream_modes if len(stream_modes) > 1 else stream_modes[0]

        if stream_subgraphs:
            if isinstance(modes, list):
                # multi-mode + subgraphs → 3-tuple (namespace, mode, chunk)
                async for namespace, mode, chunk in graph.astream(
                    input_data, config=config, stream_mode=modes, subgraphs=True
                ):
                    # Skip subgraph-level values events — only root graph values
                    # should update stream.values on the frontend
                    if mode == "values" and namespace:
                        continue
                    # updates: prefix keys with namespace so onUpdateEvent can
                    # identify which subagent produced each update
                    if mode == "updates" and namespace:
                        ns_prefix = "|".join(str(n).split(":")[0] for n in namespace)
                        chunk = {f"{ns_prefix}|{k}": v for k, v in chunk.items()}
                    yield _sse(_sse_event_name(mode, namespace if mode != "updates" else None), chunk)
            else:
                # single-mode + subgraphs → 2-tuple (namespace, chunk)
                async for namespace, chunk in graph.astream(
                    input_data, config=config, stream_mode=modes, subgraphs=True
                ):
                    if modes == "values" and namespace:
                        continue
                    if modes == "updates" and namespace:
                        ns_prefix = "|".join(str(n).split(":")[0] for n in namespace)
                        chunk = {f"{ns_prefix}|{k}": v for k, v in chunk.items()}
                    yield _sse(_sse_event_name(modes, namespace if modes != "updates" else None), chunk)
        else:
            if isinstance(modes, list):
                async for mode, chunk in graph.astream(input_data, config=config, stream_mode=modes):
                    yield _sse(_sse_event_name(mode), chunk)
            else:
                async for chunk in graph.astream(input_data, config=config, stream_mode=modes):
                    yield _sse(_sse_event_name(modes), chunk)

    except Exception as exc:
        yield _sse("error", {"message": str(exc)})
        return

    if emit_end:
        yield _sse("end", {})


async def stream_graph_wait(
    graph: Any,
    input_data: Any,
    config: dict,
) -> Any:
    """Run the graph to completion and return the final state values."""
    result = await graph.ainvoke(input_data, config=config)
    return result
