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

    # ── 1. Exact-match cache (fast path, no embedding cost) ────────
    cache_key = query.strip().lower()
    with _kb_cache_lock:
        if cache_key in _kb_cache:
            logger.info("Exact-match cache hit for: %s (%.3fs)", query, time.perf_counter() - t_start)
            return _kb_cache[cache_key]

    # ── 2. Vectorstore search ─────────────────────────────────────
    try:
        if not _vectorstore:
            return KnowledgeBaseSearchResponse(
                error="Vectorstore not initialized. Cannot search knowledge base.",
            ).model_dump_json()

        t_vs = time.perf_counter()
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

        # Store in exact-match cache
        with _kb_cache_lock:
            _kb_cache[cache_key] = result

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

