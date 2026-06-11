#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# google_cloud_hack/ must be on PYTHONPATH so that
# "deepagent_copilot.ai_agent.backend.*" resolves correctly.
# fin_agent/ lives inside ai_agent/, so ai_agent/ must also be on the path.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting Financial AI Copilot..."
echo "  PYTHONPATH=$PYTHONPATH"

# Unset VIRTUAL_ENV so uv uses its own .venv, not the parent shell's venv
unset VIRTUAL_ENV
unset VIRTUAL_ENV_PROMPT

# Terminal server — receives sandbox output and broadcasts to UI via SSE
uv run python -m uvicorn terminal_server:app --port 8001 &
TERMINAL_PID=$!
echo "Terminal server started (pid=$TERMINAL_PID, port=8001)"

# On exit, kill the terminal server
trap "kill $TERMINAL_PID 2>/dev/null" EXIT

# Main backend — LangGraph API server (LangGraph SDK compatible)
echo "Starting main backend on port 2024..."
uv run python -m uvicorn backend.main:app --host 0.0.0.0 --port 2024 --workers 1
