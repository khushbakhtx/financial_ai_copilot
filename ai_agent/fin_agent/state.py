"""Shared agent state schema for the Financial AI Copilot.

This TypedDict is used by LangGraph as the graph state AND synchronized
to the frontend via CopilotKitMiddleware. Any field updated via
Command(update={...}) becomes immediately visible in the UI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class PipelineStep(TypedDict):
    step: str           # e.g. "01_eda"
    label: str          # human-readable, e.g. "EDA & Profiling"
    status: str         # "pending" | "running" | "completed" | "error" | "skipped"
    agent: str          # which subagent runs it
    started_at: Optional[str]
    completed_at: Optional[str]
    summary: Optional[str]


class Finding(TypedDict):
    id: str
    severity: str       # "CRITICAL" | "WARNING" | "INFO" | "POSITIVE"
    agent: str
    content: str
    timestamp: str


class ModelResult(TypedDict):
    model_name: str
    auc: float
    gini: Optional[float]
    ks: Optional[float]
    f1: Optional[float]
    rank: int


class FinancialCopilotState(MessagesState):
    """Full shared state — LangGraph graph state + CopilotKit frontend sync."""

    # ── Investigation context (set by orchestrator at start) ──────────────
    investigation_id: str
    dataset_name: str
    pipeline_type: str      # "credit_scoring" | "fraud_detection" | "general"

    # ── Pipeline progress (updated by each step, rendered as progress timeline) ──
    pipeline_steps: List[PipelineStep]

    # ── Live findings feed (populated by save_finding calls) ──────────────
    findings: List[Finding]

    # ── Model leaderboard (populated by model-research-agent) ─────────────
    model_leaderboard: List[ModelResult]

    # ── Summary stats shown in the header ─────────────────────────────────
    dataset_shape: Optional[List[int]]      # [rows, cols]
    fraud_rate: Optional[float]             # fraud detection only
    best_auc: Optional[float]

    # ── File artifacts (save_report writes here) ──────────────────────────
    files: Dict[str, str]

    # ── Downloadable artifacts (charts, models — publish_artifact/sync) ───
    artifacts: List[Dict[str, Any]]

    # ── Todos (orchestrator task tracking, also shown in UI) ──────────────
    todos: List[Any]

    # ── User steering — frontend can write these back ─────────────────────
    # User can edit the plan before execution starts
    user_plan_overrides: Dict[str, Any]
    # User can mark steps to skip
    skipped_steps: List[str]
