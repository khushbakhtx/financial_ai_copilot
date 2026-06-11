from __future__ import annotations

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class Config(BaseModel):
    tags: list[str] = []
    recursion_limit: Optional[int] = None
    configurable: dict[str, Any] = {}


class CheckpointConfig(BaseModel):
    checkpoint_id: Optional[str] = None
    checkpoint_ns: Optional[str] = None
    checkpoint_map: Optional[dict[str, Any]] = None


class Send(BaseModel):
    node: str
    input: Any


class Command(BaseModel):
    update: Optional[Any] = None
    resume: Optional[Any] = None
    goto: Optional[Union[Send, list[Send], str, list[str]]] = None


class Interrupt(BaseModel):
    id: Optional[str] = None
    value: Any


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Assistants
# ---------------------------------------------------------------------------

VALID_GRAPH_IDS = Literal["financial_copilot"]


class Assistant(BaseModel):
    assistant_id: str
    graph_id: str
    config: dict[str, Any] = {}
    context: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = {}
    version: int = 1
    name: Optional[str] = None
    description: Optional[str] = None


class AssistantCreate(BaseModel):
    graph_id: str
    assistant_id: Optional[str] = None
    config: dict[str, Any] = {}
    context: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = {}
    if_exists: Literal["raise", "do_nothing"] = "raise"
    name: Optional[str] = None
    description: Optional[str] = None


class AssistantPatch(BaseModel):
    graph_id: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None
    name: Optional[str] = None
    description: Optional[str] = None


class AssistantSearchRequest(BaseModel):
    graph_id: Optional[str] = None
    name: Optional[str] = None
    limit: int = 10
    offset: int = 0
    metadata: Optional[dict[str, Any]] = None


class AssistantCountRequest(BaseModel):
    graph_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

class Thread(BaseModel):
    thread_id: str
    created_at: datetime
    updated_at: datetime
    state_updated_at: Optional[datetime] = None   # spec field; we alias to updated_at
    metadata: dict[str, Any] = {}
    config: dict[str, Any] = {}
    status: Literal["idle", "busy", "interrupted", "error"] = "idle"
    values: Optional[Any] = None
    interrupts: Optional[dict[str, Any]] = None   # object (dict) per spec, not list
    ttl: Optional[dict[str, Any]] = None
    extracted: Optional[dict[str, Any]] = None


class ThreadCreate(BaseModel):
    thread_id: Optional[str] = None
    metadata: dict[str, Any] = {}
    # spec has no config field — accept it for backwards compat but don't store
    if_exists: Literal["raise", "do_nothing"] = "raise"
    ttl: Optional[dict[str, Any]] = None
    supersteps: Optional[list[dict[str, Any]]] = None


class ThreadPatch(BaseModel):
    metadata: Optional[dict[str, Any]] = None
    ttl: Optional[dict[str, Any]] = None


class ThreadSearchRequest(BaseModel):
    ids: Optional[list[str]] = None
    limit: int = 10
    offset: int = 0
    metadata: Optional[dict[str, Any]] = None
    status: Optional[str] = None
    sort_by: Literal["created_at", "updated_at"] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"


class ThreadCountRequest(BaseModel):
    metadata: Optional[dict[str, Any]] = None
    status: Optional[str] = None


class ThreadPruneRequest(BaseModel):
    thread_ids: list[str]
    strategy: Literal["delete", "keep_latest"] = "delete"


class ThreadPruneResponse(BaseModel):
    pruned_count: int


class ThreadStateUpdateResponse(BaseModel):
    """Response for POST /threads/{id}/state — per OpenAPI spec."""
    checkpoint: Optional[dict[str, Any]] = None


class ThreadState(BaseModel):
    values: Any
    next: list[str] = []
    tasks: list[dict[str, Any]] = []
    checkpoint: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = {}
    created_at: Optional[str] = None
    parent_checkpoint: Optional[dict[str, Any]] = None
    interrupts: list[Interrupt] = []


class ThreadStateUpdate(BaseModel):
    values: Optional[Any] = None
    checkpoint: Optional[CheckpointConfig] = None
    as_node: Optional[str] = None


class ThreadStateSearch(BaseModel):
    limit: int = 10
    before: Optional[CheckpointConfig] = None
    metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

StreamMode = Literal[
    "values", "messages", "messages-tuple",
    "tasks", "checkpoints", "updates",
    "events", "debug", "custom",
]


class RunBase(BaseModel):
    assistant_id: str
    input: Optional[Any] = None
    command: Optional[Command] = None
    metadata: dict[str, Any] = {}
    config: Optional[Config] = None
    context: Optional[dict[str, Any]] = None
    stream_mode: Union[StreamMode, list[StreamMode]] = ["values"]
    stream_subgraphs: bool = False
    stream_resumable: bool = False
    interrupt_before: Optional[Union[Literal["*"], list[str]]] = None
    interrupt_after: Optional[Union[Literal["*"], list[str]]] = None
    feedback_keys: list[str] = []
    after_seconds: Optional[float] = None


class RunCreateStateful(RunBase):
    multitask_strategy: Literal["reject", "rollback", "interrupt", "enqueue"] = "enqueue"
    if_not_exists: Literal["create", "reject"] = "reject"


class RunCreateStreamingStateful(RunCreateStateful):
    on_disconnect: Literal["cancel", "continue"] = "continue"


class RunCreateStateless(RunBase):
    on_completion: Literal["delete", "keep"] = "delete"


class RunCreateStreamingStateless(RunCreateStateless):
    on_disconnect: Literal["cancel", "continue"] = "continue"


class RunBatchCreate(BaseModel):
    runs: list[RunCreateStateful]


class Run(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str
    created_at: datetime
    updated_at: datetime
    status: Literal["pending", "running", "error", "success", "timeout", "interrupted"] = "pending"
    metadata: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    multitask_strategy: str = "enqueue"


class RunsCancel(BaseModel):
    run_ids: list[str]
    action: Literal["interrupt", "rollback"] = "interrupt"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class Item(BaseModel):
    namespace: list[str]
    key: str
    value: dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class StorePutRequest(BaseModel):
    namespace: list[str]
    key: str
    value: dict[str, Any]


class StoreDeleteRequest(BaseModel):
    namespace: list[str]
    key: str


class StoreSearchRequest(BaseModel):
    namespace_prefix: list[str]
    filter: Optional[dict[str, Any]] = None
    limit: int = 10
    offset: int = 0
    query: Optional[str] = None


class StoreListNamespacesRequest(BaseModel):
    namespace_prefix: Optional[list[str]] = None
    suffix: Optional[list[str]] = None
    max_depth: Optional[int] = None
    limit: int = 100
    offset: int = 0


class SearchItemsResponse(BaseModel):
    items: list[Item]


class ListNamespaceResponse(BaseModel):
    namespaces: list[list[str]]
