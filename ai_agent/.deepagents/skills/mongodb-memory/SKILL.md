---
name: mongodb-memory
description: MANDATORY — read at the start of every investigation. Full reference for all mongodb__ MCP tools, collection schemas, vector search patterns, and query examples. Required for findings, experiments, pipeline_state, fraud_relationships, and model_registry access.
allowed-tools: aggregate, find, insert-many, collection-indexes, list-collections, delete-many, update-many, count
---

# MongoDB Memory Skill

You are connected to MongoDB Atlas via the `mongodb` MCP server. The database is `financial_ai_copilot`.

## When to Use This Skill

- Saving agent findings, experiment results, or reasoning traces for future investigations
- Retrieving similar past investigations using vector search
- Reading the experiment leaderboard or historical model metrics
- Checking fraud relationship graphs from previous investigations
- Persisting any data that should survive across sessions

## MCP Server Tools Available

The `mongodb` MCP server exposes these tools. Call them as `mongodb__<tool-name>` (two underscores):

### Read Tools (always safe)
| Tool | Purpose |
|------|---------|
| `mongodb__find` | Query any collection with a filter |
| `mongodb__aggregate` | Run an aggregation pipeline (including `$vectorSearch`) |
| `mongodb__aggregate-db` | Aggregate across a whole database |
| `mongodb__count` | Count documents matching a filter |
| `mongodb__list-collections` | List all collections in a database |
| `mongodb__list-databases` | List all databases |
| `mongodb__collection-schema` | Inspect a collection's field schema |
| `mongodb__collection-indexes` | Show indexes including vector search indexes |
| `mongodb__collection-storage-size` | Get collection size |
| `mongodb__db-stats` | Database statistics |

### Write Tools
| Tool | Purpose |
|------|---------|
| `mongodb__insert-many` | Insert documents into a collection |
| `mongodb__update-many` | Update documents matching a filter |
| `mongodb__delete-many` | Delete documents matching a filter |
| `mongodb__create-collection` | Create a new collection |
| `mongodb__create-index` | Create classic, vector search, or full-text index |
| `mongodb__drop-index` | Drop an index |
| `mongodb__drop-collection` | Drop a collection (destructive — confirm first) |
| `mongodb__rename-collection` | Rename a collection |

### Utility Tools
| Tool | Purpose |
|------|---------|
| `mongodb__connect` | Reconnect or switch connection string |
| `mongodb__explain` | Explain a query's execution plan |
| `mongodb__export` | Export query results as JSON |
| `mongodb__mongodb-logs` | Recent MongoDB server logs |
| `mongodb__search-knowledge` | Search MongoDB official documentation |
| `mongodb__list-knowledge-sources` | List available doc sources |

## Collections Reference

| Collection | What's in it | Key fields |
|------------|-------------|------------|
| `findings` | Agent findings — risk flags, anomalies, data quality | `agent`, `type`, `severity`, `content`, `investigation_id`, `timestamp` |
| `experiments` | ML experiment results | `model_name`, `metrics.auc`, `parameters`, `investigation_id` |
| `vector_memory` | Embeddings of past investigations for semantic search | `embedding` (dim 3072), `content`, `summary`, `type`, `tags` |
| `reports` | Final investigation reports | `title`, `content`, `investigation_id` |
| `fraud_relationships` | Fraud graph entities | `entity_type`, `entity_id`, `related_entities` |
| `agent_memory` | Agent reasoning traces | `agent`, `reasoning`, `decision` |
| `conversations` | Investigation conversation history | `thread_id`, `role`, `content` |
| `model_registry` | Trained model metadata | `model_name`, `version`, `metrics` |
| `reasoning_traces` | Full agent reasoning chains | `investigation_id`, `agent`, `steps` |
| `datasets` | Dataset metadata | `name`, `path`, `schema`, `stats` |

## Vector Search (Semantic Memory)

The `vector_memory` collection has a **ready** Vector Search index named `vector_index`:
- Path: `embedding`, Dimensions: 3072, Similarity: cosine
- Filter fields: `type`, `tags`

To find similar past investigations:
```json
{
  "database": "financial_ai_copilot",
  "collection": "vector_memory",
  "pipeline": [
    {
      "$vectorSearch": {
        "index": "vector_index",
        "path": "embedding",
        "queryVector": [/* 3072-dim float array from gemini-embedding-001 */],
        "numCandidates": 50,
        "limit": 5
      }
    },
    {
      "$project": {
        "embedding": 0,
        "score": { "$meta": "vectorSearchScore" }
      }
    }
  ]
}
```

**Note:** To generate the `queryVector`, use `fin_agent.memory.vector_search()` which handles embedding generation automatically. Only drop to raw MCP `$vectorSearch` if you need custom filter fields like `type` or `tags`.

## Common Query Patterns

### Get top experiments by AUC
```json
{
  "database": "financial_ai_copilot",
  "collection": "experiments",
  "filter": {},
  "sort": { "metrics.auc": -1 },
  "limit": 10
}
```

### Get all CRITICAL findings for an investigation
```json
{
  "database": "financial_ai_copilot",
  "collection": "findings",
  "filter": { "severity": "CRITICAL", "investigation_id": "<id>" },
  "sort": { "timestamp": -1 }
}
```

### Get recent fraud relationships
```json
{
  "database": "financial_ai_copilot",
  "collection": "fraud_relationships",
  "filter": { "entity_type": "device" },
  "sort": { "timestamp": -1 },
  "limit": 20
}
```

## Saving Data via MCP

Prefer the Python tools (`save_finding`, `save_experiment_result`, `save_to_memory`) for structured saves — they handle timestamps, IDs, and embeddings automatically.

Use MCP `insert-many` directly only for bulk inserts or when the Python tools are unavailable:
```json
{
  "database": "financial_ai_copilot",
  "collection": "findings",
  "documents": [
    {
      "agent": "FraudInvestigationAgent",
      "type": "fraud_ring",
      "severity": "CRITICAL",
      "content": "Detected 3-node fraud ring sharing device_id 4821",
      "investigation_id": "inv_20260527",
      "timestamp": "2026-05-27T10:00:00Z"
    }
  ]
}
```
