"""MongoDB Atlas integration for persistent financial investigation memory.

Provides collection access, vector search via Atlas Vector Search,
and embedding generation using Google's gemini-embedding-001 model (3072 dimensions).
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_mongo_client = None
_mongo_db = None

VALID_COLLECTIONS = {
    "datasets",
    "experiments",
    "findings",
    "agent_memory",
    "pipeline_state",
    "reports",
    "fraud_relationships",
    "conversations",
    "model_registry",
    "reasoning_traces",
    "vector_memory",
}


def _get_client():
    """Get or create the MongoDB client (lazy initialization)."""
    global _mongo_client, _mongo_db

    if _mongo_client is not None:
        return _mongo_client

    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None

    try:
        from pymongo import MongoClient

        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db_name = os.getenv("MONGODB_DB", "financial_ai_copilot")
        _mongo_db = _mongo_client[db_name]
        _mongo_client.admin.command("ping")
        logger.info("Connected to MongoDB Atlas: %s", db_name)
        return _mongo_client
    except Exception as e:
        logger.warning("MongoDB connection failed: %s", e)
        _mongo_client = None
        return None


def get_mongo_collection(collection_name: str):
    """Get a MongoDB collection by name.

    Args:
        collection_name: Name of the collection to access.

    Returns:
        PyMongo Collection object, or None if MongoDB is not configured.
    """
    if collection_name not in VALID_COLLECTIONS:
        raise ValueError(f"Unknown collection '{collection_name}'. Valid: {VALID_COLLECTIONS}")

    client = _get_client()
    if client is None:
        return None

    return _mongo_db[collection_name]


def _get_embedding(text: str) -> Optional[list[float]]:
    """Generate a text embedding using Google's embedding model.

    Args:
        text: Text to embed.

    Returns:
        List of floats (embedding vector), or None on failure.
    """
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
        return embeddings.embed_query(text)
    except Exception as e:
        logger.warning("Embedding generation failed: %s", e)
        return None


def save_with_embedding(
    collection_name: str,
    document: dict[str, Any],
    embed_field: str = "content",
) -> Optional[str]:
    """Save a document to MongoDB with an embedding for vector search.

    Args:
        collection_name: Target collection name.
        document: Document to save.
        embed_field: Field whose text content will be embedded.

    Returns:
        Inserted document ID as string, or None on failure.
    """
    col = get_mongo_collection(collection_name)
    if col is None:
        return None

    text_to_embed = document.get(embed_field, "")
    if text_to_embed:
        embedding = _get_embedding(str(text_to_embed))
        if embedding:
            document["embedding"] = embedding

    result = col.insert_one(document)
    return str(result.inserted_id)


def vector_search(
    query: str,
    collection: str,
    limit: int = 5,
    index_name: str = "vector_index",
) -> Optional[list[dict]]:
    """Perform a vector similarity search in MongoDB Atlas.

    Requires MongoDB Atlas Vector Search index named 'vector_index' on
    the 'embedding' field with dimension=3072 (gemini-embedding-001).

    Falls back to text search if vector search is unavailable.

    Args:
        query: Natural language query to search for.
        collection: Collection to search.
        limit: Maximum number of results.
        index_name: Name of the Atlas Vector Search index.

    Returns:
        List of matching documents (without embedding field), or None if MongoDB
        is not configured.
    """
    col = get_mongo_collection(collection)
    if col is None:
        return None

    # Try vector search first
    query_embedding = _get_embedding(query)
    if query_embedding:
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit,
                    }
                },
                {
                    "$project": {
                        "embedding": 0,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            results = list(col.aggregate(pipeline))
            if results:
                return [
                    {k: (str(v) if k == "_id" else v) for k, v in doc.items()}
                    for doc in results
                ]
        except Exception as e:
            logger.debug("Vector search failed, falling back to text search: %s", e)

    # Fallback: text search
    try:
        results = list(col.find(
            {"$text": {"$search": query}},
            {"embedding": 0},
            limit=limit,
        ))
        return [
            {k: (str(v) if k == "_id" else v) for k, v in doc.items()}
            for doc in results
        ]
    except Exception:
        # Final fallback: recent documents
        results = list(col.find({}, {"embedding": 0}).sort("timestamp", -1).limit(limit))
        return [
            {k: (str(v) if k == "_id" else v) for k, v in doc.items()}
            for doc in results
        ]


def ensure_indexes():
    """Create necessary MongoDB indexes for efficient querying.

    Call once during initialization. Idempotent — safe to call repeatedly.
    """
    client = _get_client()
    if client is None:
        return

    try:
        from pymongo import ASCENDING, DESCENDING

        # findings: query by investigation, severity, timestamp
        findings_col = _mongo_db["findings"]
        findings_col.create_index([("investigation_id", ASCENDING), ("severity", ASCENDING)])
        findings_col.create_index([("timestamp", DESCENDING)])
        findings_col.create_index([("agent", ASCENDING), ("type", ASCENDING)])

        # experiments: query by model, investigation, AUC
        exp_col = _mongo_db["experiments"]
        exp_col.create_index([("investigation_id", ASCENDING), ("model_name", ASCENDING)])
        exp_col.create_index([("metrics.auc", DESCENDING)])

        # conversations: thread-based retrieval
        conv_col = _mongo_db["conversations"]
        conv_col.create_index([("thread_id", ASCENDING), ("timestamp", ASCENDING)])

        # fraud_relationships: graph lookups
        fraud_col = _mongo_db["fraud_relationships"]
        fraud_col.create_index([("entity_type", ASCENDING), ("entity_id", ASCENDING)])

        # Text index on vector_memory for fallback search
        vm_col = _mongo_db["vector_memory"]
        try:
            vm_col.create_index([("content", "text"), ("summary", "text")])
        except Exception:
            pass  # Text index may already exist

        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.warning("Index creation failed: %s", e)
