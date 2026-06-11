from fastapi import APIRouter, HTTPException, Response
from ..models import (
    Item, StorePutRequest, StoreDeleteRequest,
    StoreSearchRequest, StoreListNamespacesRequest,
    SearchItemsResponse, ListNamespaceResponse,
)
from .. import database as db

router = APIRouter(prefix="/store", tags=["Store"])


@router.put("/items", status_code=204)
async def put_item(body: StorePutRequest):
    await db.store_put(body.namespace, body.key, body.value)
    return Response(status_code=204)


@router.get("/items")
async def get_item(namespace: str, key: str):
    ns = namespace.split("/") if isinstance(namespace, str) else namespace
    result = await db.store_get(ns, key)
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
    return result


@router.delete("/items", status_code=204)
async def delete_item(body: StoreDeleteRequest):
    deleted = await db.store_delete(body.namespace, body.key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")
    return Response(status_code=204)


@router.post("/items/search", response_model=SearchItemsResponse)
async def search_items(body: StoreSearchRequest):
    items = await db.store_search(
        namespace_prefix=body.namespace_prefix,
        limit=body.limit,
        offset=body.offset,
        filter=body.filter,
    )
    return SearchItemsResponse(items=[Item(**i) for i in items])


@router.post("/namespaces", response_model=ListNamespaceResponse)
async def list_namespaces(body: StoreListNamespacesRequest):
    namespaces = await db.store_list_namespaces(
        prefix=body.namespace_prefix,
        limit=body.limit,
        offset=body.offset,
    )
    return ListNamespaceResponse(namespaces=namespaces)
