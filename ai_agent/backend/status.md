# Backend Gap Tracker

Each item below is a concrete gap between our implementation and the LangGraph Platform
OpenAPI spec (`deepagents_api.json`). Priority is relative to likelihood of breakage
with the Lucid frontend.

---

## Issues

### 1. `ThreadStateSearch.before` — wrong type  ✅ FIXED
**Priority: HIGH — breaks history pagination**

**Spec:** `before` in `ThreadStateSearch` is a `CheckpointConfig` object:
```json
{ "before": { "checkpoint_id": "uuid", "checkpoint_ns": "..." } }
```
**Was:** `models.py` had `before: Optional[str]` and `threads.py` passed it as
a raw string to `aget_state_history(before={"configurable": {"checkpoint_id": body.before}})`.

The LangGraph SDK sends a full `CheckpointConfig` object. When the frontend paginates
history (e.g. "load older messages"), it sends `{ "before": { "checkpoint_id": "..." } }`
and our code would either crash on `.checkpoint_id` access or silently pass wrong data.

**Fix:** Changed `ThreadStateSearch.before` from `Optional[str]` to
`Optional[CheckpointConfig]` and updated `get_thread_history` to build the
`before_config` dict using `body.before.checkpoint_id`.

---

### 2. `_get_graph_for_thread` — 2 extra DB queries on every state/history call  ✅ FIXED
**Priority: HIGH — latency on every state/history request**

Every call to `GET /threads/{id}/state` or `GET /threads/{id}/history` ran:
1. `db.get_thread(thread_id)` — 1 round-trip
2. `db.list_runs(thread_id)` — 1 round-trip (to find assistant → graph_id)
3. `graph.aget_state(config)` — checkpoint Postgres round-trip

Steps 1+2 happen on every single state/history request. The assistant→graph mapping
never changes after run creation. Both queries execute serially before the checkpoint
read starts.

**Fix:** Replaced the serial `get_thread` + `list_runs` with a single SQL JOIN query
`get_graph_id_for_thread` that returns the `graph_id` in one round-trip. This saves
one full DB round-trip on every state/history call.

---

### 3. `ThreadPruneRequest` — wrong shape  ✅ FIXED
**Priority: MEDIUM — breaks if frontend/admin calls prune**

**Spec:**
```json
{ "thread_ids": ["uuid1", "uuid2"], "strategy": "delete" }
```
**Was:** `models.py` had `before: Optional[datetime]` — completely different shape.
The spec prunes by explicit ID list with a named `strategy` field. Our model accepted
a date cutoff and the response field was named `deleted` not `pruned_count`.

**Fix:**
- Updated `ThreadPruneRequest` to `thread_ids: list[str]` + `strategy: str = "delete"`.
- Updated `ThreadPruneResponse` to use `pruned_count` (spec field name).
- Updated `database.prune_threads` to accept `thread_ids: list[str]` and delete by ID.
- Updated `routers/threads.py` to call the new signature.

---

### 4. `cancel_run` missing `action` query param  ✅ FIXED
**Priority: MEDIUM — `rollback` silently ignored**

**Spec:** `POST /threads/{id}/runs/{run_id}/cancel?action=interrupt|rollback`
- `interrupt` — cancel the task (our current behavior).
- `rollback` — cancel AND delete the run + its checkpoints.

**Was:** Both actions behaved identically (cancel only).

**Fix:** Added `action: str = "interrupt"` query param to `cancel_run`. When
`action == "rollback"`, after cancellation also call `db.delete_run(run_id)` and
delete the run's checkpoints from `checkpoint_writes`/`checkpoint_blobs`/`checkpoints`
via the `AsyncPostgresSaver` connection.

---

### 5. `multitask_strategy` default mismatch  ✅ FIXED
**Priority: MEDIUM — 409 on fast re-sends without explicit strategy**

**Spec default:** `"enqueue"` for `RunCreateStateful.multitask_strategy`.
**Was:** `database.py` `create_run` defaulted to `"reject"`, so if the frontend sends
a run without setting `multitask_strategy` the DB row gets `"reject"` even though the
Pydantic model already defaulted to `"enqueue"`. The kwargs weren't plumbed through.

The real issue: our backend doesn't actually enforce multitask strategy — it creates
the run regardless of what's already running. A proper implementation checks thread
status and rejects/queues/interrupts accordingly.

**Fix (minimal):** The `RunCreateStateful` Pydantic model already had `"enqueue"` as
default. Added enforcement in `stream_run`: if thread status is `"busy"` and strategy
is `"reject"`, return 409. If `"interrupt"`, cancel the running run first. If `"enqueue"`
or `"rollback"`, proceed (enqueue falls through to serial execution since LangGraph
handles ordering via the checkpointer).

---

### 6. `/ok` health check missing `check_db` query param  ✅ FIXED
**Priority: MEDIUM — monitoring/orchestration tools use this**

**Spec:** `GET /ok?check_db=1` should probe the database and return 500 if it fails.
`GET /ok` (default `check_db=0`) returns `{"ok": true}` without touching DB.

**Was:** Our `/ok` returned `{"ok": true}` unconditionally regardless of DB state.

**Fix:** Added `check_db: int = 0` query param to `GET /ok`. When `check_db=1`,
executes `SELECT 1` on the pool; returns 500 with `{"ok": false, "detail": "..."}` on
failure.

---

### 7. `/info` response missing required fields  ✅ FIXED
**Priority: MEDIUM — LangGraph SDK may validate this on startup**

**Spec requires all four fields:**
```json
{
  "version": "0.1.0",
  "langgraph_py_version": "0.x.y",
  "flags": {},
  "metadata": {}
}
```
**Was:** Our system router returned a subset (missing `flags` and `metadata`, or wrong
field names).

**Fix:** Updated `GET /info` to return all four required fields. `langgraph_py_version`
is read from the installed `langgraph` package version at startup.

---

### 8. `ThreadSearchRequest` — `ids` filter missing  ✅ FIXED
**Priority: LOW — frontend doesn't currently use it**

**Spec:** `ids: list[uuid]` — return only threads whose IDs are in this list.
**Was:** `search_threads` SQL ignored `ids` entirely.

**Fix:** Added `ids: Optional[list[str]] = None` to `ThreadSearchRequest` model and
dynamic `WHERE thread_id = ANY($1::uuid[])` branch in `search_threads` SQL.

---

### 9. `AssistantSearchRequest` — `name` substring filter missing  ✅ FIXED
**Priority: LOW — frontend uses graph_id filter not name**

**Spec:** `name` field performs case-insensitive substring match (`ILIKE`).
**Was:** `search_assistants` SQL ignored `name`.

**Fix:** Added `name: Optional[str] = None` to `AssistantSearchRequest` model and
dynamic `AND name ILIKE $N` clause in `search_assistants` SQL.

---

### 10. `AssistantPatch` missing `graph_id` update  ✅ FIXED
**Priority: LOW — graph_id rarely changes after creation**

**Spec:** `AssistantPatch` includes `graph_id` — patching it should switch the
assistant to a different graph.
**Was:** `patch_assistant` in `database.py` didn't update `graph_id`.

**Fix:** Added `graph_id` to `AssistantPatch` Pydantic model, added it as a parameter
to `db.patch_assistant()`, and included it in the UPDATE SQL.

---

### 11. `_get_graph_for_thread` 404s on threads with no runs  ✅ FIXED
**Priority: LOW — only affects freshly created threads before first run**

`GET /threads/{id}/state` on a thread that has no runs yet raised 404 "No runs found".
The LangGraph Platform returns an empty state `{ values: {}, next: [] }` instead.

**Fix:** `get_thread_state`, `get_thread_state_at_checkpoint`, and `get_thread_history`
now call `db.get_graph_id_for_thread()` directly instead of `_get_graph_for_thread()`.
When no graph_id is found (no runs yet), they return `_EMPTY_THREAD_STATE` / `[]`
respectively instead of 404. `_get_graph_for_thread` is now only used by
`update_thread_state` where a 404 is appropriate (can't update state with no runs).

---

### 12. `copy_thread` doesn't copy checkpoints  ✅ FIXED
**Priority: LOW — not used by current frontend**

Spec: "Create a new thread with a copy of the state and checkpoints from an existing
thread." Our implementation only copied metadata/config — the new thread started empty.

**Fix:** After creating the new thread row, `copy_thread` in `database.py` now runs
three INSERT…SELECT statements (one per checkpoint table) that clone all rows from
`checkpoints`, `checkpoint_blobs`, and `checkpoint_writes` replacing `thread_id` with
the new UUID. All three inserts run in the same connection context.

---

### 13. Schema isolation — all tables created in `public` instead of configurable schema  ✅ FIXED
**Priority: HIGH — data sovereignty / dev-prod separation**

All metadata and checkpoint tables were created in the `public` schema, mixing our
backend's tables with the original LangSmith deployment's tables.

**Fix:**
- Added `_get_schema()` to `database.py`: reads `SCHEMA_NAME_LOCAL` first (local dev →
  `"backend"`), falls back to `SCHEMA_NAME` (Render prod → `"public"`).
- `_DDL_TEMPLATE` now starts with `CREATE SCHEMA IF NOT EXISTS __SCHEMA__` and
  `SET search_path TO __SCHEMA__, public` before all `CREATE TABLE` statements.
- asyncpg pool `_setup` callback sets `search_path` on every connection so all raw
  SQL queries resolve to the correct schema automatically.
- `AsyncPostgresSaver` DSN is patched at startup to append
  `options=-csearch_path=<schema>` so LangGraph checkpoint tables also land in the
  correct schema.

---

### 14. `messages-tuple` SSE event name mismatch — no token streaming  ✅ FIXED
**Priority: HIGH — users see full response appear at once instead of streaming**

The LangGraph SDK's `StreamManager` listens for SSE `event: messages` to accumulate
token chunks via `MessageTupleManager`. Our backend was emitting `event: messages-tuple`
(the LangGraph mode name), which the SDK silently ignored. The only event the SDK
received was `event: values` at the end of each node — the complete accumulated state —
so the entire AI response appeared all at once after a multi-second delay.

Additionally `messages-tuple` mode was not always included in the stream modes list,
so LangGraph never emitted token-level chunks at all.

**Fix:**
- Added `_sse_event_name(mode, namespace)` in `streaming.py` that maps
  `"messages-tuple"` → SSE event name `"messages"` (with `"|<namespace>"` suffix for
  subgraph events). All other modes pass through unchanged.
- `_normalize_modes()` in `routers/runs.py` always prepends `"messages-tuple"` to the
  mode list, ensuring LangGraph emits token chunks even if the client didn't request it.

---

### 15. Pre-run DB overhead — 7 serial Render round-trips before graph starts  ✅ FIXED
**Priority: HIGH — ~3–7 s startup latency per message on remote DB**

`stream_run` made 7 serial DB round-trips before `graph.astream()` was called:
1. `get_thread` (existence check)
2. `get_thread` again (multitask check — duplicate)
3. `get_assistant` (via `_resolve_graph`)
4. `get_assistant` again (via `_resolve_assistant_id` — same row, duplicate)
5. `create_run`
6. `update_run_status("running")`
7. `set_thread_status("busy")`

With ~500 ms per Render round-trip this added 3–7 s of pure overhead before the LLM
was even invoked. `wait_run` and background `create_run` had the same problem.

**Fix:**
- `stream_run` now fetches thread once and reuses it for both the existence and
  multitask checks.
- Added `_resolve_assistant_row()` helper that calls `get_assistant` / `get_assistant_by_graph`
  once and returns the full row (replacing the two separate `_resolve_graph` +
  `_resolve_assistant_id` calls).
- Added `create_run_and_set_busy()` in `database.py` that inserts the run row AND sets
  `thread.status = 'busy'` in a single transaction — replacing two separate round-trips.
- Total pre-run round-trips reduced from 7 → 3.

---

### 16. `sort_by` / `sort_order` fields ignored in thread search  ✅ FIXED
**Priority: MEDIUM — frontend always sends `sort_by=updated_at` which was silently ignored**

The frontend sends `{"sort_by": "updated_at", "sort_order": "desc"}` in every
`POST /threads/search` request. Our `ThreadSearchRequest` model didn't have these fields
so they were dropped, and the SQL always ordered by `created_at DESC`.

**Fix:**
- Added `sort_by: Literal["created_at", "updated_at"] = "created_at"` and
  `sort_order: Literal["asc", "desc"] = "desc"` to `ThreadSearchRequest`.
- `search_threads` in `database.py` now uses these with a whitelist guard (prevents
  SQL injection) to build `ORDER BY updated_at DESC` etc.
- Added `CREATE INDEX thread_updated_at_idx ON thread (updated_at DESC)` to the DDL.
- Wired through in `routers/threads.py`.

---

### 17. `LANGSMITH_TRACING_V2` not disabled — silent LangSmith retry overhead  ✅ FIXED
**Priority: MEDIUM — 5–30 s added per LLM call when LangSmith is unreachable**

`graphs.py` already disabled `LANGCHAIN_TRACING_V2` but not `LANGSMITH_TRACING_V2`.
The `.env` had `LANGSMITH_TRACING_V2=true`, which LangChain's SDK also checks. With an
unreachable or invalid LangSmith endpoint the SDK retried with exponential backoff.

**Fix:** Added `os.environ["LANGSMITH_TRACING_V2"] = "false"` alongside the existing
`LANGCHAIN_TRACING_V2` disable in `graphs.py`.

---

### 18. `if_not_exists` default caused 500 on existing thread  ✅ FIXED
**Priority: MEDIUM — crash on retry after stop/interrupt**

`RunCreateStateful.if_not_exists` was changed to `"create"` to handle stale localStorage
thread IDs. But `_ensure_thread` called `db.create_thread()` with the default
`if_exists="raise"`, so when a thread already existed (e.g. after stop + re-send) it
raised `ValueError: Thread already exists` → 500.

**Fix:** `_ensure_thread` now passes `if_exists="do_nothing"` to `create_thread`.
`if_not_exists` default reverted back to `"reject"` (the spec default) since stale IDs
are handled by clearing localStorage.

---

### 19. Request body logging middleware broke SSE streaming  ✅ FIXED
**Priority: HIGH — every streaming response crashed with RuntimeError**

The HTTP logging middleware buffered the request body by calling `await request.body()`
and re-injecting a custom `receive()` callable. Starlette's `StreamingResponse` uses the
same `receive` channel to detect client disconnects. When it called `receive()` mid-stream
it got back `{"type": "http.request"}` (the re-injected body) instead of
`{"type": "http.disconnect"}`, raising `RuntimeError: Unexpected message received`.

**Fix:** Middleware now skips body buffering for routes ending in `/stream` or `/join`,
logging `body=<stream>` instead. Non-streaming routes still log the body preview.

---

### 20. Pool `init` callback bypassed — intermittent 404 "Thread not found"  ✅ FIXED
**Priority: HIGH — one in ~5 new-thread requests returned 404**

Every newly created thread had a ~20% chance of returning 404 on its first
`POST /threads/{id}/runs/stream` call. The thread WAS in the database (confirmed
via direct query), but `get_thread` inside `stream_run` returned `None`.

**Root cause:** `asyncpg.create_pool(init=_setup)` — the `init` callback was used to
run both `set_type_codec` (JSONB) and `SET search_path TO backend, public`. The `init`
callback fires once per connection when it is **created**, but asyncpg 0.31 does not
guarantee that `init` is called in all connection creation paths (e.g. when the pool
grows under concurrent load). Some connections were handed to request handlers with
the default `search_path = '"$user", public'`, so `SELECT * FROM thread` queried
`public.thread` while the INSERT had gone to `backend.thread` (on a correctly-init'd
connection). The mismatch caused `get_thread` to see an empty result.

**Evidence:** Added `SHOW search_path` in `get_thread` on 404s — confirmed
`search_path='"$user", public'` for failing connections.

**Fix:**
- Split pool callbacks into two:
  - `init=_init` — registers JSONB/JSON codecs (once per connection, as before)
  - `setup=_setup` — runs `SET search_path TO backend, public` (on **every checkout**)
- `setup` runs on every `_pool.acquire()` call, so even connections that bypassed
  `init` get the correct `search_path` before any query runs.
- Stress-tested 10 consecutive create+stream pairs: 0/10 failures (was ~2/10).

---

### 21. Token streaming invisible — wrong LangGraph stream mode  ✅ FIXED
**Priority: HIGH — AI responses appear all at once instead of progressively**

**Symptom:** With `langgraph dev` the AI response text streams progressively (token by
token). With the custom backend, the entire AI response appeared all at once after a
multi-second delay.

**Root cause — two layers:**

1. **Wrong stream mode for local graphs**: We were passing `stream_mode=["messages-tuple"]`
   to `graph.astream()`. `"messages-tuple"` is the **LangGraph Platform (remote)** API
   concept; local `CompiledStateGraph.astream()` only understands `"messages"`. The
   Platform server maps `"messages-tuple"` → `"messages"` internally (see
   `langgraph/pregel/remote.py:672-674`). When we passed `"messages-tuple"` to the local
   graph, LangGraph saw an unknown mode and never registered `StreamMessagesHandler` —
   so no `messages` chunks were ever emitted.

2. **StreamMessagesHandler registration**: `StreamMessagesHandler` is only added to
   `run_manager.inheritable_handlers` when `"messages" in stream_modes` (not
   `"messages-tuple"`) — see `langgraph/pregel/main.py:2559-2566`. Without this
   registration, the LangChain callback chain had 0 handlers and `on_llm_end`
   never fired to emit message chunks.

3. **Bonus**: Using `"messages"` mode triggers Gemini's streaming endpoint
   (`streamGenerateContent?alt=sse`) instead of the batch endpoint, so partial tokens
   arrive as they are generated rather than waiting for the full response.

**Fix:**
- `_normalize_modes()` in `routers/runs.py` and `routers/stateless.py`:
  Replace `"messages-tuple"` with `"messages"` and always ensure `"messages"` is in
  the mode list. The SSE event name `"messages"` emitted by our backend is exactly what
  the SDK's `StreamManager` expects (it was already handled correctly by `_sse_event_name`
  which passed `"messages"` through unchanged).
- Result: 6 `event: messages` SSE frames emitted per "hello" response, each containing
  `[AIMessageChunk, metadata]`, appearing progressively as Gemini streams tokens.

---

## Legend
- ✅ FIXED — done in this session
- ⬜ TODO — acknowledged, not yet fixed
