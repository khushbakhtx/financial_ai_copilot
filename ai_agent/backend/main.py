"""Financial AI Copilot — FastAPI backend.

Start:
    cd ai_agent
    ./start.sh
    # orchestrator → http://localhost:2024
    # terminal server → http://localhost:8001

MONGODB_URI must be set in .env.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .graphs import setup_checkpointer
from . import database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongodb_uri = os.environ["MONGODB_URI"]
    db_name = os.environ.get("BACKEND_DB_NAME", "financial_copilot_backend")

    log.info("━━ startup ━━  db=%s", db_name)

    await database.init_db(mongodb_uri, db_name)
    log.info("MongoDB ready")

    await database._seed_assistants()
    log.info("assistants seeded")

    from contextlib import ExitStack
    from langgraph.checkpoint.mongodb import MongoDBSaver
    with ExitStack() as stack:
        checkpointer = stack.enter_context(
            MongoDBSaver.from_conn_string(mongodb_uri, db_name=db_name)
        )
        setup_checkpointer(checkpointer)
        log.info("checkpointer ready (MongoDBSaver)")

        assistants = await database.search_assistants(limit=20)
        for a in assistants:
            log.info(
                "assistant  id=%-36s  graph=%s  name=%s",
                a["assistant_id"], a["graph_id"], a.get("name"),
            )
        if not assistants:
            log.warning("no assistants found — seed may have failed")

        yield

    await database.close_db()
    log.info("shutdown complete")


app = FastAPI(
    title="Financial AI Copilot Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    is_stream = request.url.path.endswith(("/stream", "/join"))
    if is_stream:
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        log.info("%s %s → %d  (%.0f ms)  body=<stream>",
                 request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    body_bytes = await request.body()

    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}
    request._receive = receive  # type: ignore[attr-defined]

    body_preview = body_bytes[:300].decode("utf-8", errors="replace").replace("\n", " ")
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    log.info(
        "%s %s → %d  (%.0f ms)  body=%s",
        request.method, request.url.path,
        response.status_code, elapsed_ms,
        body_preview if body_bytes else "—",
    )
    return response


from .routers import assistants, runs, stateless, store, system, threads

app.include_router(system.router)
app.include_router(assistants.router)
app.include_router(threads.router)
app.include_router(runs.router)
app.include_router(stateless.router)
app.include_router(store.router)
