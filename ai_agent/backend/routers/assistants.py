from fastapi import APIRouter, HTTPException
from ..models import (
    Assistant, AssistantCreate, AssistantPatch,
    AssistantSearchRequest, AssistantCountRequest,
)
from .. import database as db

router = APIRouter(prefix="/assistants", tags=["Assistants"])


def _to_model(d: dict) -> Assistant:
    return Assistant(**d)


@router.post("", response_model=Assistant)
async def create_assistant(body: AssistantCreate):
    try:
        row = await db.create_assistant(
            graph_id=body.graph_id,
            assistant_id=body.assistant_id,
            config=body.config,
            context=body.context,
            metadata=body.metadata,
            name=body.name,
            description=body.description,
            if_exists=body.if_exists,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _to_model(row)


@router.post("/search", response_model=list[Assistant])
async def search_assistants(body: AssistantSearchRequest):
    rows = await db.search_assistants(
        graph_id=body.graph_id,
        name=body.name,
        limit=body.limit,
        offset=body.offset,
    )
    return [_to_model(r) for r in rows]


@router.post("/count")
async def count_assistants(body: AssistantCountRequest):
    return await db.count_assistants(graph_id=body.graph_id)


@router.get("/{assistant_id}", response_model=Assistant)
async def get_assistant(assistant_id: str):
    row = await db.get_assistant(assistant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return _to_model(row)


@router.patch("/{assistant_id}", response_model=Assistant)
async def patch_assistant(assistant_id: str, body: AssistantPatch):
    row = await db.patch_assistant(
        assistant_id=assistant_id,
        graph_id=body.graph_id,
        config=body.config,
        metadata=body.metadata,
        name=body.name,
        description=body.description,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return _to_model(row)


@router.delete("/{assistant_id}")
async def delete_assistant(assistant_id: str):
    deleted = await db.delete_assistant(assistant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {}


@router.get("/{assistant_id}/graph")
async def get_assistant_graph(assistant_id: str):
    """Return a minimal graph schema (nodes/edges) for the assistant."""
    row = await db.get_assistant(assistant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Assistant not found")
    from ..graphs import get_graph
    try:
        graph = get_graph(row["graph_id"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    # LangGraph compiled graphs expose get_graph() for visualization
    try:
        g = graph.get_graph()
        return {
            "nodes": [{"id": n.id, "type": n.data.__class__.__name__} for n in g.nodes.values()],
            "edges": [{"source": e.source, "target": e.target} for e in g.edges],
        }
    except Exception:
        return {"nodes": [], "edges": []}


@router.get("/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str):
    row = await db.get_assistant(assistant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {"input_schema": {}, "output_schema": {}, "config_schema": {}}


@router.get("/{assistant_id}/subgraphs")
async def get_subgraphs(assistant_id: str):
    row = await db.get_assistant(assistant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {}


@router.get("/{assistant_id}/versions", response_model=list[Assistant])
async def list_assistant_versions(assistant_id: str):
    rows = await db.list_assistant_versions(assistant_id)
    return [_to_model(r) for r in rows]


@router.post("/{assistant_id}/versions", response_model=Assistant)
async def create_assistant_version(assistant_id: str, body: AssistantPatch):
    row = await db.patch_assistant(
        assistant_id=assistant_id,
        config=body.config,
        metadata=body.metadata,
        name=body.name,
        description=body.description,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return _to_model(row)


@router.post("/{assistant_id}/latest", response_model=Assistant)
async def set_latest_version(assistant_id: str):
    row = await db.get_assistant(assistant_id)
    if not row:
        raise HTTPException(status_code=404, detail="Assistant not found")
    return _to_model(row)
