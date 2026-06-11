# Backend Endpoint Reference

This document describes every HTTP endpoint exposed by the backend, what role it plays
in the agentic workflow, and the exact shape of its request/response payloads.

The backend is a self-hosted replacement for LangGraph Platform. It speaks the same
HTTP API so the frontend's LangGraph SDK (`useStream`, `client.threads.*`, etc.) works
without modification.

---

## Table of Contents

1. [System](#system)
2. [Assistants](#assistants)
3. [Threads](#threads)
4. [Thread State & History](#thread-state--history)
5. [Thread Runs (stateful streaming)](#thread-runs-stateful-streaming)
6. [Stateless Runs](#stateless-runs)
7. [Key-Value Store](#key-value-store)

---

## System

### `GET /ok`

**Role:** Health check used by load balancers, Render, and the frontend SDK on startup
to confirm the server is reachable.

**Query params:**

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `check_db` | int | `0` | If `1`, runs `SELECT 1` against the Postgres pool before returning. |

**Responses:**
- `200 {"ok": true}` — server is up (and DB is healthy if `check_db=1`).
- `500 {"ok": false, "detail": "..."}` — DB probe failed (only when `check_db=1`).

**When called:** The SDK calls `GET /ok` at startup to validate `DEFAULT_DEPLOYMENT_URL`.
Render's zero-downtime deploys hit it to determine when the new container is ready.

---

### `GET /info`

**Role:** Tells the SDK which graph IDs are registered so it can validate `assistant_id`
references and display a server version in the UI.

**Response:**
```json
{
  "version": "0.1.0",
  "langgraph_py_version": "0.3.x",
  "flags": {},
  "metadata": {
    "graphs": ["lucid_agent", "lucid_autonomous"]
  }
}
```

**When called:** Once on SDK client initialization.

---

### `GET /metrics`

**Role:** Stub required by the LangGraph Platform spec. Returns an empty object.
Not used by the Lucid frontend but keeps API compatibility with any monitoring tooling
that queries the endpoint.

**Response:** `{}`

---

## Assistants

An **assistant** is a named binding between a `graph_id` (e.g. `"lucid_agent"`) and an
optional configuration dict. Every run must reference an `assistant_id`. The frontend
looks up the assistant to know which graph to execute and which config to apply.

### `POST /assistants`

**Role:** Register a new assistant. Called during the seeding step at startup
(`init_db`) to create the two system assistants (`lucid_agent` and `lucid_autonomous`).
Also called if an admin wants to create a custom assistant with different config.

**Request body:**
```json
{
  "graph_id": "lucid_agent",
  "assistant_id": "6c6184c2-...",   // optional — UUID generated if omitted
  "config": {},                      // extra RunnableConfig fields merged into every run
  "metadata": { "created_by": "system" },
  "name": "lucid_agent",
  "if_exists": "do_nothing"          // "raise" (default) | "do_nothing"
}
```

**Response:** Full `Assistant` object.

**`if_exists`:** If you call this again with the same `assistant_id` and `"raise"`, you
get 409. `"do_nothing"` is used during seeding so the server can restart without
re-registering failing.

---

### `POST /assistants/search`

**Role:** The frontend's primary way to discover which assistants exist. Called on every
page load to populate the mode selector (Interactive / Autonomous). Returns the list
ordered by creation time.

**Request body:**
```json
{
  "graph_id": "lucid_agent",   // optional filter
  "name": "lucid",             // optional case-insensitive substring match
  "limit": 10,
  "offset": 0
}
```

**Response:** `[Assistant, ...]`

**Why `POST` not `GET`:** LangGraph Platform uses POST for all search operations so
complex filter objects can be sent in the body rather than query params.

---

### `POST /assistants/count`

**Role:** Returns the total count of assistants matching a filter. Used for pagination
UI in admin tools. Not called by the Lucid frontend directly.

**Request body:** `{ "graph_id": "lucid_agent" }` (both optional)

**Response:** `42` (plain integer)

---

### `GET /assistants/{assistant_id}`

**Role:** Fetch a single assistant by UUID. The frontend calls this when it needs to
confirm an assistant still exists before starting a run, or when displaying assistant
details.

**Response:** `Assistant` object or `404`.

---

### `PATCH /assistants/{assistant_id}`

**Role:** Update an assistant's config, metadata, name, or graph_id. Useful for
switching an assistant from one graph to another without changing the `assistant_id`
that threads already reference. Each PATCH increments the `version` counter and writes
an immutable snapshot to `assistant_versions`.

**Request body (all fields optional):**
```json
{
  "graph_id": "lucid_autonomous",
  "config": { "recursion_limit": 200 },
  "metadata": { "env": "prod" },
  "name": "New Name"
}
```

**Response:** Updated `Assistant` object.

---

### `DELETE /assistants/{assistant_id}`

**Role:** Remove an assistant. Safe to call only if no threads are actively running
against it. In production this is rarely used because the two system assistants are
permanent.

**Response:** `{}` or `404`.

---

### `GET /assistants/{assistant_id}/graph`

**Role:** Returns the compiled graph's node/edge topology for visualization tools (e.g.
LangGraph Studio). The Lucid frontend does not currently use this, but it is part of
the Platform spec.

**Response:**
```json
{
  "nodes": [
    { "id": "__start__", "type": "..." },
    { "id": "model", "type": "RunnableCallable" },
    ...
  ],
  "edges": [
    { "source": "__start__", "target": "MemoryMiddleware.before_agent" },
    ...
  ]
}
```

---

### `GET /assistants/{assistant_id}/schemas`

**Role:** Returns input/output/config JSON schemas for the graph. Stub returning empty
schemas — the Lucid agent has no strict input schema enforced at the API level.

---

### `GET /assistants/{assistant_id}/subgraphs`

**Role:** Lists registered subgraphs. Stub returning `{}`. The Platform spec requires
this endpoint; subgraph introspection is not needed by the frontend.

---

### `GET /assistants/{assistant_id}/versions`

**Role:** Returns the full version history of an assistant. Every `create_assistant`
and `patch_assistant` call writes a snapshot row to `assistant_versions`. Useful for
auditing config changes or rolling back.

**Response:** `[Assistant, ...]` ordered by `version` descending.

---

### `POST /assistants/{assistant_id}/versions`

**Role:** Apply a patch and explicitly create a new version (same as PATCH but via a
different path for SDK compatibility).

**Request body:** Same as `AssistantPatch`.

---

### `POST /assistants/{assistant_id}/latest`

**Role:** Returns the current (latest) assistant record. Used by SDK internals to
confirm the assistant ID is valid before submitting a run.

**Response:** `Assistant` object (no state change).

---

## Threads

A **thread** is a persistent conversation context. It holds a stable `thread_id` that
is stored in the browser's `localStorage`. All messages, tool calls, and checkpoints
for a conversation belong to one thread. A thread can contain many runs over its
lifetime (each user message creates a new run).

### `POST /threads`

**Role:** Create a new thread. Called when the user starts a fresh conversation or when
the SDK detects that `threadId` in localStorage doesn't match any existing thread and
`if_not_exists = "create"` is set on the run request.

**Request body:**
```json
{
  "thread_id": "a1b2c3d4-...",   // optional — UUID generated if omitted
  "metadata": {},
  "if_exists": "raise"           // "raise" | "do_nothing"
}
```

**Response:** `Thread` object with `status: "idle"`.

---

### `POST /threads/search`

**Role:** Populates the thread list panel in the UI sidebar. The frontend calls this
after every run completes (`onFinish`) and on initial load to display recent
conversations sorted by last activity.

**Request body:**
```json
{
  "limit": 20,
  "offset": 0,
  "sort_by": "updated_at",   // "created_at" | "updated_at"
  "sort_order": "desc",      // "asc" | "desc"
  "status": "idle",          // optional: filter by "idle"|"busy"|"interrupted"|"error"
  "ids": ["uuid1", "uuid2"]  // optional: fetch specific threads by ID
}
```

**Response:** `[Thread, ...]` — each thread row includes a denormalised `values` blob
(the last checkpoint's message list) so the UI can render a preview without extra
queries.

---

### `POST /threads/count`

**Role:** Returns the total number of threads matching a filter. Used for pagination.

**Request body:** `{ "status": "idle" }` (optional)

**Response:** `42` (plain integer)

---

### `POST /threads/prune`

**Role:** Delete a batch of threads by explicit ID list. Used by admin scripts to clean
up stale threads. Cascades to the `run` table (deletes associated runs).

**Request body:**
```json
{
  "thread_ids": ["uuid1", "uuid2"],
  "strategy": "delete"   // "delete" | "keep_latest" (keep_latest is a no-op currently)
}
```

**Response:** `{ "pruned_count": 2 }`

---

### `GET /threads/{thread_id}`

**Role:** Fetch the full thread object including its current `status`, `values`
(last checkpoint state), and `interrupts`. The frontend checks `status` to decide
whether to render the interrupt approval UI, and reads `values.messages` for the
conversation history.

**Response:** `Thread` object or `404`.

---

### `PATCH /threads/{thread_id}`

**Role:** Update thread metadata (e.g. add a display name). Does not change the
conversation state.

**Request body:** `{ "metadata": { "title": "My Analysis" } }`

**Response:** Updated `Thread` object.

---

### `DELETE /threads/{thread_id}`

**Role:** Delete a thread and all its runs. Called when the user removes a conversation
from the sidebar. The LangGraph checkpoint rows (in `checkpoints`,
`checkpoint_blobs`, `checkpoint_writes`) are NOT deleted here — they are managed by
LangGraph's own cleanup mechanisms.

**Response:** `{}` or `404`.

---

### `POST /threads/{thread_id}/copy`

**Role:** Clone a thread — creates a new thread row and deep-copies all three
checkpoint tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) with the new
`thread_id`. The new thread starts from the same conversation state as the original.
Used for branching (running alternate responses from the same conversation point).

**Response:** New `Thread` object.

---

## Thread State & History

These endpoints expose LangGraph's checkpoint data. The frontend uses them to render
the full conversation and to detect interrupt state.

### `GET /threads/{thread_id}/state`

**Role:** Returns the **current** graph state — the latest checkpoint. The SDK calls
this after reconnecting to an existing thread to restore `stream.values` (message
history, files, todos) without running the graph again.

Also used to check `tasks[*].interrupts` — if the graph is paused at an interrupt
node, those interrupt values contain the `action_requests` and `review_configs` that
drive the approval UI buttons.

**Response:**
```json
{
  "values": {
    "messages": [...],
    "todos": [...],
    "files": {...}
  },
  "next": ["HumanInTheLoopMiddleware.after_model"],
  "tasks": [
    {
      "id": "task-uuid",
      "name": "HumanInTheLoopMiddleware.after_model",
      "error": null,
      "interrupts": [
        {
          "value": {
            "action_requests": [...],
            "review_configs": [...]
          }
        }
      ]
    }
  ],
  "checkpoint": { "thread_id": "...", "checkpoint_id": "...", "checkpoint_ns": "" },
  "metadata": {},
  "created_at": "2026-05-21T10:00:00Z",
  "parent_checkpoint": { "checkpoint_id": "previous-uuid" },
  "interrupts": [{ "value": { ... } }]
}
```

**Note:** If no runs have been created yet (brand-new thread), returns an empty state
`{ values: {}, next: [], tasks: [], interrupts: [] }` instead of 404.

---

### `POST /threads/{thread_id}/state`

**Role:** Manually inject state into the checkpoint store. Used to set up specific
state for testing or to programmatically add messages (e.g. injecting a tool result
without running the full graph). Also used by `setFiles` in the frontend to update the
`files` dict in the graph state externally.

**Request body:**
```json
{
  "values": { "files": { "train.csv": "s3://..." } },
  "as_node": "model",          // optional — which node the state update is attributed to
  "checkpoint": {              // optional — target a specific checkpoint
    "checkpoint_id": "uuid",
    "checkpoint_ns": ""
  }
}
```

**Response:** `{ "checkpoint": { "checkpoint_id": "new-uuid", ... } }`

---

### `GET /threads/{thread_id}/state/{checkpoint_id}`

**Role:** Retrieve the graph state **at a specific historical checkpoint**. Used when
the user clicks "rewind" or when the frontend needs to re-render state at an older
point in the conversation for branching.

**Response:** Same structure as `GET /threads/{thread_id}/state` but for the requested
checkpoint.

---

### `POST /threads/{thread_id}/state/checkpoint`

**Role:** Same as the GET above but accepts the checkpoint config in a POST body.
Provided for SDK compatibility (some SDK versions use POST for this).

**Request body:** `{ "checkpoint_id": "uuid", "checkpoint_ns": "" }`

---

### `GET /threads/{thread_id}/history` and `POST /threads/{thread_id}/history`

**Role:** Returns the **full sequence of checkpoints** for a thread — one entry per
graph node execution. The SDK's `useStream` calls this via `fetchStateHistory: true`
on mount, which populates `branchContext` used for conversation branching and displaying
the interrupt state after reconnecting to a paused thread.

The POST variant accepts a filter body for pagination.

**POST request body (optional):**
```json
{
  "limit": 10,
  "before": {
    "checkpoint_id": "uuid-of-last-seen-checkpoint"
  }
}
```

`before` enables cursor-based pagination: the response contains only checkpoints
**older** than the given checkpoint. The SDK uses this when loading older history pages.

**Response:** Array of state snapshots, newest first:
```json
[
  {
    "values": { "messages": [...], "todos": [...] },
    "next": [],
    "tasks": [ { "id": "...", "name": "...", "interrupts": [...] } ],
    "checkpoint": { "checkpoint_id": "latest-uuid", "thread_id": "..." },
    "metadata": { "step": 12 },
    "created_at": "2026-05-21T10:05:00Z",
    "parent_checkpoint": { "checkpoint_id": "prev-uuid" },
    "interrupts": []
  },
  ...
]
```

**Why tasks + interrupts matter:** The SDK derives `stream.interrupt` from the last
history entry's `tasks[*].interrupts`. If `interrupts` is non-empty, `useStream`
exposes `stream.interrupt` which triggers the `ToolApprovalInterrupt` component to
render approve/edit/reject buttons. Without `tasks` in the history response, the
interrupt UI would never appear after a page reload.

---

### `GET /threads/{thread_id}/stream`

**Role:** Reconnect endpoint. When a user refreshes the page while a run was in
progress, the SDK attempts to rejoin the stream by calling this endpoint. The current
implementation returns a minimal SSE snapshot of the thread's current status rather
than replaying the full run stream. The `end` event is sent immediately to signal that
there is nothing live to stream.

**Response:** SSE stream:
```
event: values
data: {"thread_id": "...", "status": "idle"}

event: end
data: {}
```

---

## Thread Runs (stateful streaming)

A **run** represents one execution of the graph against a thread. Every user message
triggers a run. Runs are persisted in the `run` table and their progress is tracked via
`status: pending → running → success|error|interrupted`.

### `POST /threads/{thread_id}/runs/stream`

**Role:** The core endpoint — the primary path for every user message. Creates a run,
executes the graph, and streams results back as Server-Sent Events. The frontend's
`useStream` hook connects to this via `client.runs.stream(...)`.

**Request body:**
```json
{
  "assistant_id": "6c6184c2-...",
  "input": {
    "messages": [
      { "type": "human", "content": "Analyze this dataset", "id": "uuid" }
    ]
  },
  "stream_mode": ["values", "updates", "messages-tuple"],
  "stream_subgraphs": true,
  "config": {
    "recursion_limit": 100,
    "configurable": {
      "lucid_partner_name": "bank_name",
      "lucid_projectId": "proj-uuid",
      "agent_key": "secret"
    }
  },
  "multitask_strategy": "enqueue",   // "reject"|"interrupt"|"enqueue"|"rollback"
  "if_not_exists": "reject",         // "create"|"reject" — create thread if missing
  "on_disconnect": "continue"        // "cancel"|"continue"
}
```

**SSE event sequence:**
```
event: metadata
data: {"run_id": "run-uuid"}

event: values
data: {"messages": [<HumanMessage>]}         ← state after input added

event: updates
data: {"MemoryMiddleware.before_agent": {...}}

event: updates
data: {"SkillsMiddleware.before_agent": {...}}

event: messages
data: [<AIMessageChunk>, {"langgraph_node": "model", ...}]  ← token chunks (many)

event: updates
data: {"model": {"messages": [<AIMessage>]}}  ← complete AI response

event: values
data: {"messages": [<Human>, <AI>], "todos": [...]}  ← final state

event: updates
data: {"HumanInTheLoopMiddleware.after_model": {...}}  ← interrupt check

event: end
data: {}
```

**`stream_mode` values:**
- `"values"` — full graph state snapshot after each node (used by SDK to rebuild
  `stream.values` and `stream.messages`)
- `"messages"` (sent as `"messages-tuple"` by SDK but normalized to `"messages"` by
  `_normalize_modes`) — partial AI message chunks for progressive text rendering
- `"updates"` — per-node output dicts (used by `onUpdateEvent` to capture subagent
  messages and update todos)

**`stream_subgraphs: true`:** When the agent delegates to a subagent (e.g.
`data-analysis-agent`), LangGraph emits events with the subgraph's namespace appended
to the event name (e.g. `updates` carries keys like `"train-model-agent|model"`). The
frontend's `onUpdateEvent` uses this to capture subagent messages and display them
under the correct subagent label.

**`multitask_strategy`:**
- `"reject"` — return 409 if thread is `"busy"`
- `"interrupt"` — cancel the current run before starting a new one
- `"enqueue"` — proceed (LangGraph checkpoint serializes access)

**Background finalization:** After `event: end` is sent, a background `asyncio.Task`
calls `graph.aget_state()` to read the final checkpoint, then persists the state
snapshot to `thread.values` and sets `thread.status` to `"idle"` or `"interrupted"`.
This is done in the background so the client sees `end` immediately without waiting for
the Postgres checkpoint read (~500ms).

---

### `POST /threads/{thread_id}/runs/wait`

**Role:** Blocking variant of `/stream`. Runs the graph to completion and returns the
final state values in a single JSON response. Used for non-interactive flows or when
the caller can wait for the full result.

**Request body:** Same as `/stream` minus `on_disconnect`.

**Response:** Final `values` dict (same as the last `event: values` data in the stream).

---

### `POST /threads/{thread_id}/runs`

**Role:** Fire-and-forget background run. Creates a run record, starts the graph in
an `asyncio.Task` via `run_manager`, and returns immediately with the run's metadata.
The caller can check progress by polling `GET /threads/{thread_id}/runs/{run_id}` or
joining with `GET /threads/{thread_id}/runs/{run_id}/join`.

**Request body:** Same as `/stream` minus `on_disconnect`.

**Response:** `Run` object with `status: "pending"`.

---

### `GET /threads/{thread_id}/runs`

**Role:** List all runs for a thread, newest first. Used by the SDK to find the latest
`run_id` for reconnection, and by the thread list to show run count/status.

**Response:** `[Run, ...]`

---

### `GET /threads/{thread_id}/runs/{run_id}`

**Role:** Get the current status and metadata of a specific run. Polled by the SDK
when joining a background run.

**Response:** `Run` object with current `status`.

---

### `DELETE /threads/{thread_id}/runs/{run_id}`

**Role:** Cancel a running run (if active) and delete its DB record. Used when the
user deletes a conversation mid-run.

**Response:** `{}`

---

### `POST /threads/{thread_id}/runs/{run_id}/cancel`

**Role:** Cancel an active run without deleting it. The frontend's stop button calls
this. The `action` param controls what happens to the checkpoint:

**Query params:**

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `action` | string | `"interrupt"` | `"interrupt"` — cancel task, mark run as interrupted. `"rollback"` — cancel task AND delete the run row + set thread idle. |
| `wait` | bool | `false` | Block until cancellation is confirmed. |

**Response:** `{}`

---

### `GET /threads/{thread_id}/runs/{run_id}/join`

**Role:** Block until the run reaches a terminal status (`success`, `error`,
`interrupted`, `timeout`). Polls the DB every 100ms for up to 30 seconds. Used by SDK
callers that submitted a background run and now need the result.

**Response:** Final `Run` object or `408` on timeout.

---

### `GET /threads/{thread_id}/runs/{run_id}/stream`

**Role:** Reconnect to a run that is already complete. Returns the current thread
state as a single-shot SSE stream (one `values` event + `end`). Used by the SDK's
`reconnectOnMount` path when the page reloads and there was a run in progress.

**Response:** SSE stream with `event: values` (the graph state snapshot) and `event: end`.

---

### `POST /threads/{thread_id}/runs/crons`

**Role:** Stub — returns 501. Scheduled runs against a thread are not implemented.
The cron scheduler and `cron` DB table exist but have no execution logic.

---

## Stateless Runs

Stateless runs use a **throwaway thread_id** generated per request. No state is
persisted between calls. Useful for one-shot completions, batch testing, or evaluation
pipelines that don't need conversation history.

### `POST /runs/stream`

**Role:** One-shot streaming graph execution. Same SSE wire format as
`/threads/{id}/runs/stream` but the thread is discarded after the run. No DB rows are
written for the thread or run.

**Request body:** Same as the stateful streaming body but with `on_completion` instead
of `multitask_strategy`:
```json
{
  "assistant_id": "lucid_agent",
  "input": { "messages": [...] },
  "stream_mode": ["values", "messages-tuple"],
  "on_disconnect": "cancel",
  "on_completion": "delete"
}
```

---

### `POST /runs/wait`

**Role:** Blocking one-shot execution. Returns the final state dict synchronously.

---

### `POST /runs`

**Role:** Fire-and-forget stateless run. Returns immediately with a synthetic run ID.
No DB record is written — the run lives only in the `run_manager` asyncio task registry.

**Response:** `{ "run_id": "uuid", "thread_id": "throwaway-uuid", "status": "pending" }`

---

### `POST /runs/cancel`

**Role:** Cancel one or more running stateless runs by ID. Calls `run_manager.cancel`
for each.

**Request body:** `{ "run_ids": ["uuid1", "uuid2"], "action": "interrupt" }`

**Response:** `204 No Content`

---

### `POST /runs/batch` / `POST /runs/crons` / etc.

**Role:** Stubs returning 501. Batch parallel execution and scheduled cron runs are
not implemented.

---

## Key-Value Store

The store is a general-purpose key-value namespace for agent-managed persistent data.
The Lucid agent uses it to store per-user configuration, preferences, or cross-thread
context that should survive beyond a single conversation. Items can have namespaces
(hierarchical path segments) for logical grouping.

### `PUT /store/items`

**Role:** Write or overwrite a value. The agent calls this to persist facts it wants
to remember across threads (e.g. a user's preferred train/test split or model settings).

**Request body:**
```json
{
  "namespace": ["users", "partner_name", "preferences"],
  "key": "split_strategy",
  "value": { "type": "random", "test_size": 0.2 }
}
```

**Response:** `204 No Content`

---

### `GET /store/items`

**Role:** Read a single item by namespace + key. Used by the agent at the start of a
session to restore previously saved preferences.

**Query params:** `namespace=users/partner_name/preferences&key=split_strategy`

**Response:** `Item` object:
```json
{
  "namespace": ["users", "partner_name", "preferences"],
  "key": "split_strategy",
  "value": { "type": "random", "test_size": 0.2 },
  "created_at": "...",
  "updated_at": "..."
}
```

---

### `DELETE /store/items`

**Role:** Remove a stored item. Called when the agent determines a cached value is
stale or the user explicitly asks to reset preferences.

**Request body:** `{ "namespace": [...], "key": "split_strategy" }`

**Response:** `204 No Content`

---

### `POST /store/items/search`

**Role:** List all items under a namespace prefix. Allows the agent to enumerate
everything stored for a given user or project without knowing exact keys in advance.

**Request body:**
```json
{
  "namespace_prefix": ["users", "partner_name"],
  "limit": 20,
  "offset": 0,
  "filter": { "type": "random" }   // optional exact-match filter on value fields
}
```

**Response:** `{ "items": [Item, ...] }`

---

### `POST /store/namespaces`

**Role:** List all unique namespace paths that exist under a prefix. Useful for
discovering what categories of data have been stored (e.g. list all `["users", *]`
namespaces to find all users with stored preferences).

**Request body:**
```json
{
  "namespace_prefix": ["users"],
  "max_depth": 2,
  "limit": 100,
  "offset": 0
}
```

**Response:** `{ "namespaces": [["users", "bank_a"], ["users", "bank_b"]] }`

---

## How the Pieces Fit Together — Full Message Flow

```
User types message
       │
       ▼
useChat.sendMessage()
       │  POST /threads/{id}/runs/stream
       │  body: { assistant_id, input: {messages: [HumanMsg]},
       │          stream_mode: ["values","updates","messages-tuple"],
       │          stream_subgraphs: true, config: {credentials} }
       ▼
Backend stream_run()
  ├─ GET thread from DB (existence + multitask check)
  ├─ GET assistant from DB (→ graph_id)
  ├─ DB: create run + set thread.status="busy"  (single transaction)
  ├─ DB: update_run_status("running")
  │
  └─ graph.astream(input, config, stream_mode=["values","updates","messages"])
         │
         ├── event: metadata  → SDK registers run_id
         ├── event: values    → SDK updates stream.values (human msg visible)
         ├── event: updates   → SDK fires onUpdateEvent (middleware nodes)
         ├── event: messages  → SDK accumulates AIMessageChunk tokens (typing effect)
         ├── event: updates   → SDK fires onUpdateEvent (model node → subagent msgs)
         ├── event: values    → SDK updates stream.values (AI msg visible)
         ├── event: updates   → interrupt check nodes
         └── event: end       → SDK closes stream; calls onFinish()
                │
  ├─ DB: update_run_status("success")
  ├─ DB: set_thread_status("idle")
  └─ asyncio.Task: aget_state() → update_thread_snapshot()  ← background, non-blocking

onFinish() → POST /threads/search  (refresh sidebar thread list)
           → POST /threads/{id}/history  (reload history for branch context)
```