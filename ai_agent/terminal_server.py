"""Terminal streaming server for the Financial AI Copilot UI.

Exposes SSE endpoint that streams live stdout/stderr from every run_code
execution to the frontend terminal panel.

Run alongside langgraph dev:
    uv run uvicorn terminal_server:app --port 8001 --reload
"""

import asyncio
import json
import logging
import os
from collections import deque
from pathlib import Path

# Load .env from the same directory as this file (agent/.env)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass
from datetime import datetime
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial AI Terminal Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global broadcast queue — sandbox writes here, SSE clients read from it
_broadcast_queue: asyncio.Queue = asyncio.Queue()

# Ring buffer of last 500 lines for reconnecting clients
_history: deque[dict] = deque(maxlen=500)

# Active SSE subscriber queues
_subscribers: list[asyncio.Queue] = []


def _make_event(kind: str, text: str, agent: str = "") -> dict:
    return {
        "kind": kind,          # "stdout" | "stderr" | "system" | "start" | "end"
        "text": text,
        "agent": agent,
        "ts": datetime.now().strftime("%H:%M:%S"),
    }


async def _dispatcher():
    """Background task: fan out from broadcast queue to all subscriber queues."""
    while True:
        try:
            event = await _broadcast_queue.get()
            _history.append(event)
            dead = []
            for q in _subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                try:
                    _subscribers.remove(q)
                except ValueError:
                    pass
        except Exception as e:
            logger.warning("Dispatcher error: %s", e)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_dispatcher())
    logger.info("Terminal server started on port 8001")


@app.get("/terminal/stream")
async def terminal_stream():
    """SSE endpoint — each event is a JSON line prefixed with 'data: '."""
    subscriber_q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _subscribers.append(subscriber_q)

    async def event_generator() -> AsyncIterator[str]:
        # Replay history so reconnecting clients catch up
        for event in list(_history):
            yield f"data: {json.dumps(event)}\n\n"

        # Stream live events
        try:
            while True:
                try:
                    event = await asyncio.wait_for(subscriber_q.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive comment so proxies don't close the connection
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                _subscribers.remove(subscriber_q)
            except ValueError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/terminal/history")
async def terminal_history():
    """Return buffered history as JSON array (for initial load)."""
    return list(_history)


@app.get("/health")
async def health():
    return {"status": "ok", "subscribers": len(_subscribers), "history": len(_history)}


class EmitRequest(BaseModel):
    kind: str
    text: str
    agent: str = ""


@app.post("/terminal/emit")
async def terminal_emit(req: EmitRequest):
    """Receive an event from the LangGraph worker process and broadcast it to SSE clients."""
    emit(req.kind, req.text, req.agent)
    return {"ok": True}


# ── Dataset upload / list endpoints ──────────────────────────────────────────

@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV/parquet/json/xlsx dataset to MongoDB GridFS."""
    from fin_agent.datasets import upload_dataset as _upload, SUPPORTED_EXTENSIONS
    from pathlib import Path

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(file_bytes) > 200 * 1024 * 1024:  # 200 MB limit
        raise HTTPException(status_code=413, detail="File too large (max 200 MB)")

    try:
        meta = _upload(file_bytes, file.filename or "upload", file.content_type or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    if meta is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured — set MONGODB_URI in .env")

    emit("system", f"[dataset] Uploaded: {file.filename} ({meta['size_mb']} MB)")
    return {"ok": True, "dataset": meta}


@app.get("/datasets")
async def list_datasets():
    """Return metadata for all uploaded datasets."""
    from fin_agent.datasets import list_datasets as _list
    return {"datasets": _list()}


# ── Artifact endpoints (charts, models, reports produced by pipeline runs) ───

def _artifacts_db():
    """Return (db, GridFS bucket for artifact_files) or (None, None)."""
    try:
        import gridfs
        from fin_agent.memory import _get_client

        client = _get_client()
        if client is None:
            return None, None
        db = client[os.getenv("MONGODB_DB", "financial_ai_copilot")]
        return db, gridfs.GridFS(db, collection="artifact_files")
    except Exception as e:
        logger.warning("Artifact GridFS init failed: %s", e)
        return None, None


@app.get("/artifacts")
async def list_artifacts(investigation_id: str = ""):
    """Return artifact metadata, newest first. Optional investigation_id filter."""
    db, _ = _artifacts_db()
    if db is None:
        return {"artifacts": []}
    query = {"investigation_id": investigation_id} if investigation_id else {}
    docs = list(db["artifacts"].find(query, {"_id": 0}).sort("created_at", -1).limit(200))
    return {"artifacts": docs}


def _get_artifact_file(gridfs_id: str):
    from bson import ObjectId

    db, fs = _artifacts_db()
    if fs is None:
        raise HTTPException(status_code=503, detail="MongoDB not configured")
    try:
        return fs.get(ObjectId(gridfs_id))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {gridfs_id}")


@app.get("/artifacts/{gridfs_id}/download")
async def download_artifact(gridfs_id: str):
    """Download an artifact as a file attachment."""
    gf = _get_artifact_file(gridfs_id)
    filename = (gf.filename or "artifact").split("/")[-1]
    return StreamingResponse(
        iter([gf.read()]),
        media_type=gf.content_type or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/artifacts/{gridfs_id}/raw")
async def raw_artifact(gridfs_id: str):
    """Serve an artifact inline — used for <img> previews of chart PNGs."""
    gf = _get_artifact_file(gridfs_id)
    return StreamingResponse(
        iter([gf.read()]),
        media_type=gf.content_type or "application/octet-stream",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ── Public API used by sandbox.py ────────────────────────────────────────────

def emit(kind: str, text: str, agent: str = "") -> None:
    """Thread-safe emit from synchronous sandbox code."""
    event = _make_event(kind, text, agent)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(_broadcast_queue.put_nowait, event)
        else:
            _broadcast_queue.put_nowait(event)
    except Exception:
        pass  # Terminal server may not be running — never block the agent


def emit_lines(text: str, kind: str = "stdout", agent: str = "") -> None:
    """Emit text split by newlines, one event per line."""
    for line in text.splitlines():
        if line.strip():
            emit(kind, line, agent)
