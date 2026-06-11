"""Load the compiled agent graphs and inject the LangGraph checkpointer.

Reads langgraph.json from the project root — exactly the same source of truth
that `langgraph dev` uses — so adding/renaming graphs there is all you need.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent

# ── Read langgraph.json ────────────────────────────────────────────────────
_langgraph_json = _project_root / "langgraph.json"
with open(_langgraph_json) as _f:
    _lg_config: dict = json.load(_f)

# ── Load .env (path comes from langgraph.json "env" field or default) ──────
_env_file = _project_root / _lg_config.get("env", ".env")
from dotenv import load_dotenv  # noqa: E402
import os as _os  # noqa: E402
load_dotenv(_env_file)

# ── Disable LangSmith tracing — we run our own backend ────────────────────
# The .env may have LANGCHAIN_TRACING_V2=true from when the agent was hosted
# on LangSmith. With our self-hosted backend that key causes every LLM call
# to POST traces to api.smith.langchain.com. If that host is unreachable
# (offline, no key, etc.) the langsmith client retries with backoff, adding
# 5–30 s of overhead per run. Disable it unless explicitly opted in via
# LANGSMITH_TRACING_ENABLED=true in the environment.
if _os.environ.get("LANGSMITH_TRACING_ENABLED", "").lower() != "true":
    _os.environ["LANGCHAIN_TRACING_V2"] = "false"
    _os.environ["LANGSMITH_TRACING_V2"] = "false"

# ── Make project root importable (agent.py lives there) ───────────────────
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Dynamically import every graph listed in langgraph.json ───────────────
# Entry format:  "graph_id": "./module.py:variable"
def _load_graph(entry: str) -> Any:
    """Import and return the graph object from a 'path/to/module.py:var' string."""
    module_path, var_name = entry.rsplit(":", 1)
    # Normalise:  "./agent.py"  →  "agent"
    module_name = Path(module_path).stem
    spec = importlib.util.spec_from_file_location(
        module_name, _project_root / module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    spec.loader.exec_module(module)
    return getattr(module, var_name)


GRAPH_REGISTRY: dict[str, Any] = {
    graph_id: _load_graph(entry)
    for graph_id, entry in _lg_config.get("graphs", {}).items()
}


def get_graph(graph_id: str) -> Any:
    """Return the compiled graph for the given graph_id, or raise ValueError."""
    graph = GRAPH_REGISTRY.get(graph_id)
    if graph is None:
        raise ValueError(f"Unknown graph_id: {graph_id!r}. Valid: {list(GRAPH_REGISTRY)}")
    return graph


def setup_checkpointer(checkpointer: Any) -> None:
    """Inject a LangGraph checkpointer into every registered graph.

    The compiled graph is a Pregel instance whose .checkpointer attribute
    is read on every astream / ainvoke call, so setting it after compilation
    is supported and safe.
    """
    for graph in GRAPH_REGISTRY.values():
        graph.checkpointer = checkpointer
