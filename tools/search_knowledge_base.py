import logging
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Optional
from cachetools import TTLCache
import threading

logger = logging.getLogger(__name__)

_kb_cache = TTLCache(maxsize=512, ttl=300)
_kb_cache_lock = threading.Lock()

class KnowledgeBaseSearchResponse(BaseModel):
    answer: Optional[str] = None
    source: Optional[str] = None
    error: Optional[str] = None


@tool(
    "search_knowledge_base",
    description=(
        "Search the knowledge base for store policies and FAQs. "
        "Use this tool to answer general questions about return policy, refund policy, "
        "cancellation policy, shipping, warranty, payment, and customer support. "
        "Input should be a free-text search query."
    )
)
def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    if not query or not query.strip():
        return KnowledgeBaseSearchResponse(
            error="Search query is required."
        ).model_dump_json()

    _vectorstore = config['configurable'].get('vectorstore')

    cache_key = query.strip().lower()
    with _kb_cache_lock:
        if cache_key in _kb_cache:
            logger.info("KB cache hit for: %s", query)
            return _kb_cache[cache_key]

    try:
        if not _vectorstore:
            return KnowledgeBaseSearchResponse(
                error="Vectorstore not initialized. Cannot search knowledge base.",
            ).model_dump_json()

        docs = _vectorstore.similarity_search(query, k=3)
        logger.info("Found %d documents for query: %s", len(docs), query)

        if not docs:
            return KnowledgeBaseSearchResponse(
                error="No relevant information found in the knowledge base."
            ).model_dump_json()

        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list({doc.metadata.get("policy_name", "unknown") for doc in docs})
        result = KnowledgeBaseSearchResponse(
            answer=context,
            source=", ".join(sources),
        ).model_dump_json()
        with _kb_cache_lock:
            _kb_cache[cache_key] = result
        return result

    except Exception as e:
        logger.error("Error searching knowledge base: %s", e)
        return KnowledgeBaseSearchResponse(
            error=str(e),
        ).model_dump_json()
