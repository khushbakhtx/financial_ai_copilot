from fastapi import APIRouter, HTTPException, Query
from ..graphs import GRAPH_REGISTRY

router = APIRouter(tags=["System"])


def _langgraph_version() -> str:
    try:
        from importlib.metadata import version
        return version("langgraph")
    except Exception:
        return "unknown"


@router.get("/ok")
async def health_check(check_db: int = Query(default=0)):
    if check_db:
        from .. import database as db
        try:
            async with db._conn() as conn:
                await conn.execute("SELECT 1")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True}


@router.get("/info")
async def server_info():
    return {
        "version": "0.1.0",
        "langgraph_py_version": _langgraph_version(),
        "flags": {},
        "metadata": {
            "graphs": list(GRAPH_REGISTRY.keys()),
        },
    }


@router.get("/metrics")
async def metrics():
    return {}
