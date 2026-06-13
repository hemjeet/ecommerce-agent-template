import asyncio
import logging
import threading
import time
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Optional
from cachetools import TTLCache

logger = logging.getLogger(__name__)

_kb_cache = TTLCache(maxsize=512, ttl=300)
_kb_cache_lock = threading.Lock()

MAX_DOC_CHARS = 800

class KnowledgeBaseSearchResponse(BaseModel):
    answer: Optional[str] = None
    source: Optional[str] = None
    error: Optional[str] = None
    cached: Optional[bool] = None


@tool(
    "search_knowledge_base",
    description=(
        "Search the knowledge base for store policies and FAQs. "
        "Use this tool to answer general questions about return policy, refund policy, "
        "cancellation policy, shipping, warranty, payment, and customer support. "
        "Input should be a free-text search query."
    )
)
async def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    if not query or not query.strip():
        return KnowledgeBaseSearchResponse(
            error="Search query is required."
        ).model_dump_json()

    t_start = time.perf_counter()

    _vectorstore = config['configurable'].get('vectorstore')
    _semantic_cache = config['configurable'].get('semantic_cache')

    # ── 1. Exact-match cache (fast path, no embedding cost) ────────
    cache_key = query.strip().lower()
    with _kb_cache_lock:
        if cache_key in _kb_cache:
            logger.info("Exact-match cache hit for: %s (%.3fs)", query, time.perf_counter() - t_start)
            return _kb_cache[cache_key]

    # ── 2. Embed once — reuse for semantic cache + vectorstore ─────
    query_embedding = None
    embeddings_model = None
    if _semantic_cache is not None:
        embeddings_model = _semantic_cache.embeddings
    elif _vectorstore is not None:
        embeddings_model = _vectorstore.embeddings

    if embeddings_model is not None:
        try:
            t_embed = time.perf_counter()
            query_embedding = await embeddings_model.aembed_query(query)
            logger.info(
                "KB embed (single call): %.3fs, dim=%d",
                time.perf_counter() - t_embed,
                len(query_embedding),
            )
        except Exception as e:
            logger.warning("Embedding failed (falling through): %s", e)

    # ── 3. Semantic cache lookup (uses pre-computed embedding) ─────
    if _semantic_cache is not None and query_embedding is not None:
        try:
            t_cache = time.perf_counter()
            hit = await asyncio.to_thread(
                _semantic_cache.get, query, query_embedding
            )
            logger.info("Semantic cache lookup: %.3fs", time.perf_counter() - t_cache)
            if hit is not None:
                logger.info(
                    "Semantic cache HIT for query=%r (sim=%.4f, total=%.3fs)",
                    query, hit.similarity, time.perf_counter() - t_start,
                )
                return hit.response
        except Exception as e:
            logger.warning("Semantic cache lookup error (falling through): %s", e)

    # ── 4. Cache miss — search vectorstore ─────────────────────────
    try:
        if not _vectorstore:
            return KnowledgeBaseSearchResponse(
                error="Vectorstore not initialized. Cannot search knowledge base.",
            ).model_dump_json()

        t_vs = time.perf_counter()
        if query_embedding is not None:
            docs = await asyncio.to_thread(
                _vectorstore.similarity_search_by_vector, query_embedding, k=3
            )
        else:
            docs = await asyncio.to_thread(
                _vectorstore.similarity_search, query, k=3
            )
        logger.info(
            "Vectorstore search: %.3fs, docs=%d",
            time.perf_counter() - t_vs, len(docs),
        )

        if not docs:
            return KnowledgeBaseSearchResponse(
                error="No relevant information found in the knowledge base.",
                cached=False,
            ).model_dump_json()

        context = "\n\n".join([doc.page_content[:MAX_DOC_CHARS] for doc in docs])
        sources = list({doc.metadata.get("policy_name", "unknown") for doc in docs})
        result = KnowledgeBaseSearchResponse(
            answer=context,
            source=", ".join(sources),
            cached=False,
        ).model_dump_json()

        # ── 5. Store in both caches (reuse embedding) ─────────────
        with _kb_cache_lock:
            _kb_cache[cache_key] = result

        if _semantic_cache is not None and query_embedding is not None:
            try:
                t_put = time.perf_counter()
                await asyncio.to_thread(
                    _semantic_cache.put, query, result, query_embedding
                )
                logger.info("Semantic cache PUT: %.3fs", time.perf_counter() - t_put)
            except Exception as e:
                logger.warning("Semantic cache store error (non-fatal): %s", e)

        logger.info(
            "KB search total: %.3fs (query=%r)",
            time.perf_counter() - t_start, query,
        )
        return result

    except Exception as e:
        logger.error("Error searching knowledge base: %s", e)
        return KnowledgeBaseSearchResponse(
            error=str(e),
        ).model_dump_json()
