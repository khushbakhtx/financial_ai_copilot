"""State-extension middleware for the Financial AI Copilot.

deepagents' create_deep_agent does not take a state_schema parameter — custom
state channels must be declared through AgentMiddleware.state_schema (this is
how deepagents' own TodoListMiddleware adds `todos` and FilesystemMiddleware
adds `files`). Without this middleware, Command(update={...}) writes from
tools like update_pipeline_step_status / push_model_leaderboard / store_charts
/ publish_artifact are silently dropped and the UI panels never update.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing_extensions import NotRequired


class FinancialState(AgentState):
    """Extra state channels synced to the frontend via stream.values."""

    # Investigation context
    investigation_id: NotRequired[str]
    dataset_name: NotRequired[str]
    pipeline_type: NotRequired[str]

    # Live panels
    pipeline_steps: NotRequired[List[Dict[str, Any]]]
    findings: NotRequired[List[Dict[str, Any]]]
    model_leaderboard: NotRequired[List[Dict[str, Any]]]
    artifacts: NotRequired[List[Dict[str, Any]]]

    # Header stats
    dataset_shape: NotRequired[List[int]]
    fraud_rate: NotRequired[float]
    best_auc: NotRequired[float]

    # Plotly charts panel
    charts_json: NotRequired[str]
    charts_title: NotRequired[str]


class FinancialStateMiddleware(AgentMiddleware):
    """Registers the financial state channels on the agent graph."""

    state_schema = FinancialState
