# Financial AI Copilot

> **Autonomous Financial Data Scientist** — Upload a dataset, get a full ML investigation: credit scoring, fraud detection, and risk analysis — all powered by Gemini and MongoDB Atlas.

Built for the [Google Cloud Rapid Agent Hackathon](https://rapid-agent.devpost.com/) | **Partner Track: MongoDB**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## What It Does

Financial AI Copilot is an **autonomous AI agent** that performs end-to-end financial data science investigations. You upload a CSV — the agent handles everything:

1. **Exploratory Data Analysis** — structural profiling, segment risk analysis, leakage detection
2. **Preprocessing** — missing values, encoding, outlier capping, train/test split
3. **Feature Engineering** — interaction terms, ratios, mutual information scoring
4. **Model Training** — XGBoost, LightGBM, CatBoost comparison with hyperparameter tuning
5. **Error Analysis** — segment-level bias detection, weak cohort identification
6. **Iterative Retraining** — Optuna-based optimization targeting weak segments
7. **Artifact Export** — `model.pkl`, `input_schema.json`, `example_usage.py`

All with **live UI updates**: pipeline progress, model leaderboard, interactive Plotly charts, and downloadable artifacts.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Next.js Frontend                             │
│   Chat │ Pipeline Progress │ Model Leaderboard │ Charts │ Terminal  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ LangGraph SDK streaming
┌──────────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend (port 2024)                        │
│              LangGraph Agent + Gemini 3 Flash Preview                 │
│                                                                      │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────────────┐  │
│  │ Orchestrator │  │ Pipeline Scripts │  │   E2B / Local Sandbox  │  │
│  │  (Gemini 3)  │──│  (validated ML)  │──│   (isolated execution) │  │
│  └──────────────┘  └─────────────────┘  └────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      MongoDB Atlas                                    │
│                                                                      │
│  ┌──────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │  GridFS   │ │  Vector   │ │  Pipeline    │ │  MCP Server      │  │
│  │ (datasets)│ │  Search   │ │  State Bus   │ │  (reads via MCP) │  │
│  └──────────┘ └───────────┘ └──────────────┘ └──────────────────┘  │
│                                                                      │
│  ┌──────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ Findings │ │Experiments│ │    Model      │ │     Fraud        │  │
│  │          │ │           │ │  Registry     │ │  Relationships   │  │
│  └──────────┘ └───────────┘ └──────────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## MongoDB Integration (Partner Track)

MongoDB Atlas is the **central nervous system** of this application — not a bolt-on, but the core infrastructure:

| Use Case | How |
|----------|-----|
| **Dataset Storage** | GridFS stores uploaded CSVs/Parquets (supports files up to 16MB+) |
| **Pipeline State Bus** | Every ML pipeline step reads inputs from and writes outputs to `pipeline_state` collection — enables full resumability |
| **Vector Memory** | `gemini-embedding-001` (3072-dim) embeddings stored in Atlas Vector Search for semantic retrieval of past investigations |
| **MCP Server** | All agent reads go through the official `mongodb-mcp-server` — the agent queries findings, experiments, and state via MCP tools |
| **Checkpointer** | LangGraph thread state persisted via `langgraph-checkpoint-mongodb` |
| **Fraud Graph** | `fraud_relationships` collection stores entity graphs (account ↔ device ↔ IP) for cross-investigation fraud ring detection |
| **Model Registry** | Trained model metadata, metrics, and lineage tracked in `model_registry` |
| **Experiment Tracking** | Every model run saved with metrics, hyperparameters, and timestamps |

---

## Pipelines

### Credit Scoring (7 steps)
```
EDA → Preprocessing → Feature Engineering → Baseline Models →
Error Analysis → Iterative Training → Export Artifacts
```

### Fraud Detection (6 steps)
```
EDA → Anomaly Detection (Isolation Forest + LOF) →
Graph Analysis (NetworkX fraud rings) → Feature Engineering →
Model Training → Rule Generation
```

Both pipelines are **fully resumable** — if interrupted, the agent queries MongoDB for the last completed step and continues from there.

---

## Key Features

- **Gemini 3 Flash Preview** — orchestrator with low thinking latency for fast pipeline dispatch
- **Validated Pipeline Scripts** — the agent never writes ML code itself; it calls pre-written, tested scripts
- **Real-time UI** — pipeline progress, model leaderboard, and charts update live via LangGraph state streaming
- **E2B Sandbox** — code execution in isolated cloud sandboxes (with local subprocess fallback)
- **Cross-Investigation Memory** — vector search finds relevant past investigations when you start a new one
- **Financial Domain Metrics** — PSI (drift), KS statistic, Gini coefficient, segment-level lift scores
- **Leakage Detection** — auto-flags features with importance > 0.90 or dominance gap > 0.20
- **Fraud Ring Detection** — NetworkX graph analysis identifies connected fraud rings across shared entities

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini 3 Flash Preview (`gemini-3-flash-preview`) |
| Agent Framework | LangGraph + deepagents |
| Backend | FastAPI + Uvicorn |
| Database | MongoDB Atlas (GridFS, Vector Search, MCP) |
| Sandbox | E2B Code Interpreter (fallback: local subprocess) |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP |
| Anomaly Detection | PyOD (Isolation Forest, LOF) |
| Graph Analysis | NetworkX |
| Frontend | Next.js 16, React 19, TailwindCSS, Radix UI |
| Charts | Plotly.js (interactive, rendered from agent state) |
| Embeddings | `gemini-embedding-001` (3072 dimensions) |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- MongoDB Atlas cluster (free tier works)
- Gemini API key

### 1. Backend Setup

```bash
cd ai_agent
cp .env.example .env
# Fill in GEMINI_API_KEY and MONGODB_URI in .env

# Install dependencies
pip install uv
uv sync

# Start the agent + terminal server
./start.sh
# → Agent API: http://localhost:2024
# → Terminal SSE: http://localhost:8001
```

### 2. Frontend Setup

```bash
cd agentic-ui
yarn install
yarn dev
# → UI: http://localhost:3000
```

### 3. Use It

1. Open http://localhost:3000
2. Upload a financial dataset (CSV/Parquet)
3. Watch the agent run a full ML investigation autonomously
4. View results: pipeline progress, model leaderboard, charts, downloadable artifacts

---

## Project Structure

```
financial_ai_copilot/
├── ai_agent/                          # Backend + Agent
│   ├── agent.py                       # Agent assembly (Gemini + tools + middleware)
│   ├── fin_agent/                     # Core agent modules
│   │   ├── tools.py                   # 25+ custom tools (dataset, ML, memory, UI sync)
│   │   ├── prompts.py                 # Orchestrator system prompt (pipeline instructions)
│   │   ├── state.py                   # LangGraph state schema
│   │   ├── middleware.py              # State extension for frontend sync
│   │   ├── datasets.py               # MongoDB GridFS integration
│   │   ├── memory.py                  # Vector search + embeddings
│   │   └── sandbox.py                 # E2B / local code execution
│   ├── backend/                       # FastAPI server (LangGraph SDK compatible)
│   │   ├── main.py                    # App + CORS + lifecycle
│   │   ├── database.py               # MongoDB persistence layer (Motor async)
│   │   ├── routers/                   # API endpoints
│   │   └── streaming.py              # SSE streaming for LangGraph
│   ├── terminal_server.py            # Live stdout/stderr → UI terminal
│   ├── .deepagents/skills/           # Validated ML pipeline scripts
│   │   ├── credit-scoring-pipeline/  # 7-step credit scoring
│   │   ├── fraud-detection-pipeline/ # 6-step fraud detection
│   │   └── mongodb-memory/           # MCP tool reference
│   ├── Dockerfile                     # Production container
│   └── pyproject.toml                 # Dependencies + ruff config
│
└── agentic-ui/                        # Frontend
    ├── src/app/components/            # Chat, Pipeline, Terminal, Charts
    ├── src/app/components/genui/      # AI-rendered components (leaderboard, fraud alerts)
    ├── src/app/hooks/                 # useChat, useFinancialCopilot
    ├── src/providers/                 # ChatProvider, ClientProvider
    └── package.json                   # Next.js 16 + React 19
```

---

## How the Agent Works

```
User: "I uploaded loans.csv. Run a full investigation."

Agent (Gemini 3):
  1. list_available_datasets()       → finds loans.csv in GridFS
  2. download_dataset("loans.csv")   → downloads to local disk
  3. load_dataset_info(path)         → reads schema, detects target column
  4. think_tool("credit scoring pipeline - target is 'default'")
  5. set_investigation_context(...)  → UI shows pipeline header
  6. run_script("credit-scoring-pipeline/01_eda", ...)
     → Script runs in E2B sandbox
     → Writes results to MongoDB pipeline_state
     → UI updates: step 1 ✓
  7. run_script("credit-scoring-pipeline/02_preprocessing", ...)
     ...
  8. [continues through all 7 steps]
  9. save_report("final_report.md")  → appears in UI Files panel
```

Each step is **independently resumable** — if the session disconnects, the agent picks up where it left off by querying MongoDB.

---

## Deployment

```bash
# Build Docker image
docker build -t financial-ai-copilot ./ai_agent

# Or use LangGraph Cloud
langgraph build --platform linux/amd64 -t your-registry/financial-ai-copilot:latest
docker push your-registry/financial-ai-copilot:latest
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key |
| `MONGODB_URI` | Yes | MongoDB Atlas connection string |
| `MONGODB_DB` | No | Database name (default: `financial_ai_copilot`) |
| `E2B_API_KEY` | No | E2B sandbox key (falls back to local subprocess) |
| `LANGSMITH_API_KEY` | No | LangSmith tracing |

---

## License

[Apache License 2.0](LICENSE)

---

## Acknowledgments

- [Google Gemini](https://ai.google.dev/) — LLM backbone
- [MongoDB Atlas](https://www.mongodb.com/atlas) — persistent memory, vector search, MCP
- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [E2B](https://e2b.dev/) — secure code sandboxing
- [deepagents](https://github.com/deepagents-ai/deepagents) — agent framework
