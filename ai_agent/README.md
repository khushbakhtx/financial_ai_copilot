# Lucid AI Agent

Built with [deepagents](https://github.com/langchain-ai/deepagents) framework.

## Quick Start

### Prerequisites

Install uv package manager:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

Clone the repository and navigate to the agent directory:
```bash
git clone <repository-url>
cd lucid_agent/deepagent/ai_agent
```

Install dependencies:
```bash
uv sync
```

This reads dependencies from `pyproject.toml` and installs them from PyPI (Python Package Index). On first run, it will:
- Create `uv.lock` (lockfile with pinned versions for reproducibility)
- Create `.venv/` (local virtual environment with all packages)

The lockfile ensures everyone on your team gets the exact same package versions.

### Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

Optional LangSmith tracing (for debugging):
```bash
export LANGSMITH_API_KEY=your_langsmith_api_key_here
```

### Running the Agent

Start the LangGraph development server for testing deepagent:
```bash
uv run langgraph dev
```

The server will open a browser window with the Studio interface at http://localhost:8123

---

## Backend Service

The `backend/` directory is a self-hosted FastAPI server that is 100% API-compatible
with the LangGraph / LangSmith Platform. It replaces the hosted LangSmith deployment
so no agent data leaves your infrastructure.

The frontend's `DEFAULT_DEPLOYMENT_URL` must point to this server (already set to
`http://localhost:8123` in `deep-agents-ui/src/lib/config.ts`).

### File map

```
backend/
  main.py          FastAPI app + lifespan (pool, DB init, checkpointer injection)
  database.py      asyncpg pool, all SQL — schema mirrors LangGraph Platform exactly
  graphs.py        Reads langgraph.json, imports compiled graphs, injects checkpointer
  models.py        Pydantic request/response models (matches OpenAPI spec)
  streaming.py     SSE wire format, _serialize(), stream_graph()
  run_manager.py   asyncio.Task registry for background runs and cancellation
  routers/
    system.py      GET /ok  GET /info  GET /metrics
    assistants.py  Full CRUD + graph / schemas / subgraphs / versions
    threads.py     Full CRUD + state / history / stream
    runs.py        Thread-scoped runs: stream / wait / background / cancel / join
    stateless.py   Stateless runs + bulk cancel
    store.py       Key-value store: put / get / delete / search / namespaces
```

---

### Database schema

Two groups of tables live in the same PostgreSQL instance.

#### Metadata tables (owned by this backend)

| Table | Purpose |
|---|---|
| `assistant` | Assistant records: `graph_id`, `config`, `metadata`, `version` |
| `assistant_versions` | Immutable version snapshot written on every create / PATCH |
| `thread` | Conversation threads — includes denormalised state snapshot columns |
| `run` | One row per execution against a thread |
| `store` | Key-value store with optional TTL |
| `cron` | Scheduled runs (table exists; endpoints return 501) |
| `thread_ttl` | Thread expiry records (table exists; enforcement not yet active) |

Key `thread` columns populated after each run:

| Column | Type | Set when |
|---|---|---|
| `values` | `JSONB` | Run completes — latest graph state |
| `interrupts` | `JSONB NOT NULL DEFAULT '{}'` | Run completes — first pending interrupt value |
| `error` | `BYTEA` | Run fails — UTF-8 encoded exception message |
| `state_updated_at` | `TIMESTAMPTZ` | Same as above |
| `status` | `TEXT` | `idle` / `busy` / `interrupted` / `error` |

#### Checkpoint tables (owned by `AsyncPostgresSaver` — LangGraph library)

| Table | Purpose |
|---|---|
| `checkpoints` | Full graph state snapshot per node execution |
| `checkpoint_blobs` | Binary channel blobs referenced by checkpoints |
| `checkpoint_writes` | Pending task writes between checkpoints |

Created automatically by `AsyncPostgresSaver.setup()` on server startup. Never touched by application code.

---

### API routes

#### System

| Method | Path | Description |
|---|---|---|
| `GET` | `/ok` | Health check — returns `{"ok": true}` |
| `GET` | `/info` | Server version + registered graph IDs |
| `GET` | `/metrics` | Empty stub |

#### Assistants

| Method | Path | Description |
|---|---|---|
| `POST` | `/assistants` | Create assistant |
| `POST` | `/assistants/search` | List assistants with optional graph_id / metadata filter |
| `POST` | `/assistants/count` | Count assistants |
| `GET` | `/assistants/{id}` | Get assistant by ID |
| `PATCH` | `/assistants/{id}` | Update config / metadata / name (shallow-merges metadata) |
| `DELETE` | `/assistants/{id}` | Delete assistant |
| `GET` | `/assistants/{id}/graph` | Graph node + edge schema |
| `GET` | `/assistants/{id}/schemas` | Input / output / config JSON schemas |
| `GET` | `/assistants/{id}/subgraphs` | Registered subgraphs |
| `GET` | `/assistants/{id}/versions` | Full version history |
| `POST` | `/assistants/{id}/versions` | Create new version (same as PATCH) |
| `POST` | `/assistants/{id}/latest` | Return current assistant |

#### Threads

| Method | Path | Description |
|---|---|---|
| `POST` | `/threads` | Create thread (`if_exists: raise\|do_nothing`) |
| `POST` | `/threads/search` | List threads — filter by status / metadata |
| `POST` | `/threads/count` | Count threads |
| `POST` | `/threads/prune` | Delete threads older than `before` timestamp |
| `GET` | `/threads/{id}` | Get thread (includes `values`, `interrupts`, `status`) |
| `PATCH` | `/threads/{id}` | Update thread metadata (shallow merge) |
| `DELETE` | `/threads/{id}` | Delete thread and its runs |
| `POST` | `/threads/{id}/copy` | Duplicate thread |
| `GET` | `/threads/{id}/state` | Current graph state with tasks + interrupts |
| `POST` | `/threads/{id}/state` | Write state update via `aupdate_state` |
| `GET` | `/threads/{id}/state/{checkpoint_id}` | State at a specific checkpoint |
| `POST` | `/threads/{id}/state/checkpoint` | Same via request body |
| `GET` | `/threads/{id}/history` | Checkpoint history (latest N states) |
| `POST` | `/threads/{id}/history` | Same with `limit` / `before` cursor in body |
| `GET` | `/threads/{id}/stream` | Current thread state as a single SSE snapshot |

#### Thread runs

| Method | Path | Description |
|---|---|---|
| `POST` | `/threads/{id}/runs/stream` | Execute graph, stream SSE token by token |
| `POST` | `/threads/{id}/runs/wait` | Execute graph, block until done, return state |
| `POST` | `/threads/{id}/runs` | Fire-and-forget background execution |
| `GET` | `/threads/{id}/runs` | List runs for thread |
| `GET` | `/threads/{id}/runs/{run_id}` | Get run record |
| `DELETE` | `/threads/{id}/runs/{run_id}` | Cancel + delete run |
| `POST` | `/threads/{id}/runs/{run_id}/cancel` | Cancel run |
| `GET` | `/threads/{id}/runs/{run_id}/join` | Poll until run completes (max 30 s) |
| `GET` | `/threads/{id}/runs/{run_id}/stream` | Re-attach: return current thread state as SSE |

#### Stateless runs

| Method | Path | Description |
|---|---|---|
| `POST` | `/runs/stream` | Stateless stream (throwaway thread, no persistence) |
| `POST` | `/runs/wait` | Stateless blocking run |
| `POST` | `/runs` | Stateless fire-and-forget |
| `POST` | `/runs/cancel` | Bulk cancel by run ID list (204) |

#### Store

| Method | Path | Description |
|---|---|---|
| `PUT` | `/store/items` | Upsert key-value item (204) |
| `GET` | `/store/items` | Get item by namespace + key |
| `DELETE` | `/store/items` | Delete item (204) |
| `POST` | `/store/items/search` | Search by namespace prefix + optional filter |
| `POST` | `/store/namespaces` | List distinct namespaces |

---

### Running locally (without Docker)

```bash
# 1. Install dependencies (from the ai_agent/ root)
uv sync

# 2. Set the external Render hostname in .env for local dev
#    (the internal hostname only resolves inside Render's network)
#    DATABASE_URL="postgresql://user:pass@dpg-xxx.oregon-postgres.render.com/dbname"

# 3. Start the backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8123 --reload
```

On startup the server:
1. Opens an asyncpg connection pool to `DATABASE_URL`
2. Runs `init_db()` — creates all metadata tables (`CREATE TABLE IF NOT EXISTS`, safe on every boot)
3. Runs `AsyncPostgresSaver.setup()` — creates / verifies the three LangGraph checkpoint tables
4. Injects the checkpointer into every compiled graph from `langgraph.json`

---

### Running with Docker

Build the image (run from the `ai_agent/` directory — that is the build context):

```bash
docker build -t lucid-backend .
```

Run locally, pointing at your Render Postgres (external hostname):

```bash
docker run -p 8123:8123 \
  -e DATABASE_URL="postgresql://user:pass@dpg-xxx.oregon-postgres.render.com/dbname" \
  -e GOOGLE_API_KEY="your-google-api-key" \
  -e E2B_API_KEY="your-e2b-api-key" \
  lucid-backend
```

Run in production on Render (internal hostname, lower latency):

```bash
docker run -p 8123:8123 \
  -e DATABASE_URL="postgresql://user:pass@dpg-xxx/dbname" \
  -e GOOGLE_API_KEY="your-google-api-key" \
  -e E2B_API_KEY="your-e2b-api-key" \
  lucid-backend
```

Using docker-compose:

```yaml
services:
  backend:
    build: .
    ports:
      - "8123:8123"
    environment:
      DATABASE_URL: "postgresql://user:pass@dpg-xxx.oregon-postgres.render.com/dbname"
      GOOGLE_API_KEY: "${GOOGLE_API_KEY}"
      E2B_API_KEY: "${E2B_API_KEY}"
    restart: unless-stopped
```

> **Important:** Never copy `.env` into the image. Supply all secrets at runtime via
> environment variables. `.dockerignore` already excludes `.env` and `.env.*`.

#### Required environment variables

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `GOOGLE_API_KEY` | Yes | Gemini API key (agent uses `gemini-3-flash-preview`) |
| `E2B_API_KEY` | Yes | E2B sandbox key for `run_code` / `run_script` tools |
| `OPENAI_API_KEY` | No | Only needed if switching the agent model to OpenAI |
| `LANGSMITH_TRACING_ENABLED` | No | Set to `true` to re-enable LangSmith tracing for debugging |

#### DATABASE_URL — internal vs external hostname

Render assigns two hostnames to each Postgres instance:

| Context | Hostname pattern |
|---|---|
| Inside Render (deployed) | `dpg-xxx` (short, internal only) |
| Outside Render (local dev) | `dpg-xxx.oregon-postgres.render.com` |

Use the internal hostname when the backend container is also deployed on Render — DB
queries drop from ~100 ms (public internet) to ~1 ms (internal network), which is the
primary reason to co-locate the backend and database on the same platform.
