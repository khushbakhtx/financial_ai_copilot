"""Financial AI Data Scientist Copilot — Agent Assembly.

Single orchestrator with direct pipeline script execution.
No subagents — the orchestrator calls validated scripts directly via run_script().
"""

import os
from datetime import datetime
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from fin_agent.prompts import ORCHESTRATOR_INSTRUCTIONS
from fin_agent.tools import (
    think_tool,
    load_dataset_info,
    list_available_datasets,
    download_dataset,
    delete_dataset,
    generate_sample_financial_dataset,
    run_code,
    run_script,
    save_finding,
    list_findings,
    save_experiment_result,
    get_experiment_leaderboard,
    calculate_financial_metrics,
    compute_population_stability_index,
    save_pipeline_step,
    get_pipeline_state,
    save_to_memory,
    search_memory,
    save_report,
    update_pipeline_step_status,
    push_model_leaderboard,
    set_investigation_context,
    store_charts,
)

# ── Configuration ─────────────────────────────────────────────────────────────

_AGENT_DIR = Path(__file__).parent
current_date = datetime.now().strftime("%Y-%m-%d")

# ── Model ─────────────────────────────────────────────────────────────────────

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.0,
)

# ── All tools given directly to the orchestrator ──────────────────────────────

all_tools = [
    # Planning
    think_tool,
    # Dataset
    load_dataset_info,
    list_available_datasets,
    download_dataset,
    delete_dataset,
    generate_sample_financial_dataset,
    # Code execution
    run_code,
    run_script,
    # Investigation state
    save_finding,
    list_findings,
    save_experiment_result,
    get_experiment_leaderboard,
    save_pipeline_step,
    get_pipeline_state,
    # Financial metrics
    calculate_financial_metrics,
    compute_population_stability_index,
    # Memory
    save_to_memory,
    search_memory,
    # Reports
    save_report,
    # Live UI sync (updates stream.values → frontend panels update in real-time)
    set_investigation_context,
    update_pipeline_step_status,
    push_model_leaderboard,
    store_charts,
]

# ── Agent ─────────────────────────────────────────────────────────────────────

FULL_PROMPT = ORCHESTRATOR_INSTRUCTIONS + f"\n\nToday's date: {current_date}"

agent = create_deep_agent(
    model=model,
    tools=all_tools,
    system_prompt=FULL_PROMPT,
    subagents=[],
    backend=FilesystemBackend(root_dir=_AGENT_DIR),
    skills=[".deepagents/skills"],
    memory=[".deepagents/AGENTS.md"],
    name="financial-ai-copilot",
)
