import os
import uuid
import json
import logging
import sys
import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import dict_row
from psycopg_pool import AsyncConnectionPool

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from slowapi.errors import RateLimitExceeded

from agent import EcomAgent
from tools.order_history import get_order_history
from tools.order_tools import get_order_details
from tools.refund_eligibility import check_refund_eligibility
from tools.refund_amount import calculate_refund_amount
from tools.create_refund_ticket import create_refund_approval_ticket
from tools.process_refund import process_refund
from tools.search_knowledge_base import search_knowledge_base

# ── Logging ────────────────────────────────────────────────────────────
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("agent.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger().handlers[1].setLevel(logging.WARNING)

# Silence noisy third-party loggers
for _noisy in (
    "watchfiles", "httpcore", "httpx", "urllib3", "PIL",
    "openai._base_client", "langsmith", "asyncio",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Shared LLM builder (DeepSeek + OpenAI fallback) ────────────────────
def _build_llm():
    from langchain_deepseek import ChatDeepSeek

    primary = ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )

    # primary = ChatGroq(
    #     model = 'qwen/qwen3.6-27b',
    # )

    fallback = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    llm = primary.with_fallbacks([fallback])
    logger.info("  [ OK ] DeepSeek LLM loaded (%s)", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    logger.info("  [ OK ] OpenAI fallback loaded (%s)", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return llm


# ── Lifespan init helpers ──────────────────────────────────────────────
def _init_vectorstore(postgres_uri: str, embeddings: OpenAIEmbeddings):
    from langchain_postgres import PGVector

    try:
        vs = PGVector(
            embeddings=embeddings,
            collection_name="knowledge_base",
            connection=postgres_uri,
            use_jsonb=True,
        )
        logger.info("  [ OK ] Vectorstore (PGVector) connected")
        return vs
    except Exception as e:
        logger.warning("  [FAIL] Vectorstore connection failed: %s", e)
        return None


def _init_semantic_cache(redis_url: str, embeddings: OpenAIEmbeddings):
    from agent.semantic_cache import RedisSemanticCache
    try:
        cache = RedisSemanticCache(
            redis_url=redis_url,
            embeddings=embeddings,
            threshold=float(os.getenv("REDIS_CACHE_THRESHOLD", "0.92")),
            ttl=int(os.getenv("SEMANTIC_CACHE_TTL", "600")),
        )
        cache.setup()
        cache.cleanup_expired()
        logger.info(
            "  [ OK ] Redis semantic cache ready (threshold=%.2f, ttl=%ds)",
            cache.threshold, cache.ttl,
        )
        return cache
    except Exception as e:
        logger.warning("  [FAIL] Redis semantic cache init failed: %s - disabled", e)
        return None


async def _init_checkpointer(postgres_uri: str):
    try:
        pool = AsyncConnectionPool(
            conninfo=postgres_uri,
            min_size=int(os.getenv("CHECKPOINTER_POOL_MIN", "10")),
            max_size=int(os.getenv("CHECKPOINTER_POOL_MAX", "50")),
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            open=False,
        )
        await pool.open(wait=True, timeout=10)
        async with pool.connection() as conn:
            await AsyncPostgresSaver(conn=conn).setup()
        logger.info(
            "  [ OK ] AsyncPostgresSaver pool ready (min=%d, max=%d)",
            pool.min_size, pool.max_size,
        )
        return pool
    except Exception as e:
        logger.warning("  [SKIP] Checkpointer pool failed: %s - using in-memory", e)
        return None


# ── Pydantic schemas ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID. A new one is generated if omitted.",
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    requires_approval: bool = False
    approval_question: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    reply: str = Field(description="User's reply to the approval prompt (e.g. 'yes' or 'no').")


class HealthResponse(BaseModel):
    status: str
    vectorstore: str


# ── Lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    t0 = time.perf_counter()
    logger.info("─" * 56)
    logger.info("  Startup — EcomAgent Initialization")
    logger.info("─" * 56)

    postgres_uri = os.getenv("POSTGRES_URI")
    embeddings = OpenAIEmbeddings()

    # 1. LLM
    llm = _build_llm()

    # 2. Vectorstore
    vectorstore = _init_vectorstore(postgres_uri, embeddings) if postgres_uri else None
    if not postgres_uri:
        logger.warning("  [SKIP] POSTGRES_URI not set - vectorstore disabled")

    # 3. Semantic cache
    redis_url = os.getenv("REDIS_URL")
    semantic_cache = _init_semantic_cache(redis_url, embeddings) if redis_url else None
    if not redis_url:
        logger.warning("  [SKIP] Semantic cache disabled (no REDIS_URL)")

    # 4. Tools
    tools = [
        get_order_history,
        get_order_details,
        check_refund_eligibility,
        calculate_refund_amount,
        process_refund,
        create_refund_approval_ticket,
        search_knowledge_base,
    ]
    logger.info("  [ OK ] %d tools registered", len(tools))

    # 5. Checkpointer pool
    checkpointer_pool = await _init_checkpointer(postgres_uri) if postgres_uri else None

    # 6. Agent graph
    agent = EcomAgent(llm=llm, tools=tools, checkpointer=None)
    graph = agent.build_graph
    logger.info("  [ OK ] Agent graph compiled")

    # Store on app.state
    app.state.graph = graph
    app.state.vectorstore = vectorstore
    app.state.checkpointer_pool = checkpointer_pool
    app.state.semantic_cache = semantic_cache
    app.state.semantic_cache = semantic_cache

    elapsed = time.perf_counter() - t0
    logger.info("─" * 56)
    logger.info("  Startup complete (%.2fs)", elapsed)
    logger.info("─" * 56)
    yield

    # Shutdown
    logger.info("─" * 56)
    logger.info("  Shutdown")
    logger.info("─" * 56)
    if semantic_cache:
        try:
            deleted = semantic_cache.cleanup_expired()
            logger.info("  [ OK ] Semantic cache: cleaned %d expired entries", deleted)
        except Exception as e:
            logger.warning("  [FAIL] Semantic cache cleanup error: %s", e)
    if checkpointer_pool:
        try:
            await checkpointer_pool.close()
            logger.info("  [ OK ] Checkpointer pool closed")
        except Exception as e:
            logger.warning("  [FAIL] Error closing checkpointer pool: %s", e)


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="ShopAssist API",
    description="E-commerce customer support agent powered by LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


# ── Helpers ───────────────────────────────────────────────────────────
@asynccontextmanager
async def pooled_checkpointer(pool):
    if pool is None:
        yield None
        return
    async with pool.connection() as conn:
        yield AsyncPostgresSaver(conn=conn)


def _get_config(thread_id: str, vectorstore, saver=None, semantic_cache=None):
    cfg = {"configurable": {
        "thread_id": thread_id,
        "vectorstore": vectorstore,
        "semantic_cache": semantic_cache,
    }}
    if saver is not None:
        cfg["configurable"]["checkpointer"] = saver
    return cfg


def _get_request_context(req_thread_id: str | None):
    """Return (thread_id, graph, vectorstore, pool, semantic_cache) from app.state."""
    return (
        req_thread_id or str(uuid.uuid4()),
        app.state.graph,
        app.state.vectorstore,
        app.state.checkpointer_pool,
        app.state.semantic_cache,
    )


async def _check_interrupt(graph, config):
    state = await graph.aget_state(config)
    if state.tasks and getattr(state.tasks[0], "interrupts", None):
        interrupt_value = state.tasks[0].interrupts[0].value
        question = interrupt_value.get("question", "Approval required (yes/no)")
        return True, question
    return False, None


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        vectorstore="connected" if app.state.vectorstore else "disabled",
    )


@app.get("/cache/stats")
async def cache_stats():
    cache = app.state.semantic_cache
    if cache is None:
        return JSONResponse(status_code=503, content={"detail": "Semantic cache is not enabled."})
    return cache.stats()


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(os.getenv("RATE_LIMIT"))
async def chat(request: Request, req: ChatRequest):
    thread_id, graph, vectorstore, pool, semantic_cache = _get_request_context(req.thread_id)

    t0 = time.perf_counter()
    async with pooled_checkpointer(pool) as saver:
        t_conn = time.perf_counter()
        logger.info("TIMING thread=%s conn_acquire=%.2fs", thread_id[:8], t_conn - t0)

        config = _get_config(thread_id, vectorstore, saver, semantic_cache)
        try:
            t_invoke = time.perf_counter()
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                config,
            )
            logger.info("TIMING thread=%s ainvoke=%.2fs", thread_id[:8], time.perf_counter() - t_invoke)
        except Exception as e:
            logger.error("Graph invoke failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        requires_approval, question = await _check_interrupt(graph, config)

    if requires_approval:
        return ChatResponse(response="", thread_id=thread_id, requires_approval=True, approval_question=question)

    final_content = result["messages"][-1].content
    if isinstance(final_content, list):
        final_content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in final_content])

    logger.info("TIMING thread=%s total=%.2fs", thread_id[:8], time.perf_counter() - t0)
    return ChatResponse(response=final_content, thread_id=thread_id)


@app.post("/chat/stream")
@limiter.limit(os.getenv("RATE_LIMIT"))
async def chat_stream(request: Request, req: ChatRequest):
    thread_id, graph, vectorstore, pool, semantic_cache = _get_request_context(req.thread_id)

    async def event_stream():
        async with pooled_checkpointer(pool) as saver:
            config = _get_config(thread_id, vectorstore, saver, semantic_cache)
            try:
                async for msg, metadata in graph.astream(
                    {"messages": [HumanMessage(content=req.message)]},
                    config,
                    stream_mode="messages",
                ):
                    if (
                        isinstance(msg, (AIMessageChunk, AIMessage))
                        and msg.content
                        and metadata.get("langgraph_node") == "llm_call"
                    ):
                        content_str = msg.content
                        if isinstance(content_str, list):
                            content_str = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content_str])
                        if content_str:
                            yield f"data: {json.dumps(content_str)}\n\n"

                requires_approval, question = await _check_interrupt(graph, config)
                if requires_approval:
                    yield f"event: approval_required\ndata: {question}\n\n"
                yield f"event: metadata\ndata: {thread_id}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            except asyncio.CancelledError:
                logger.info("Client disconnected mid-stream (task cancelled)")
                # Cleanly exit the generator to avoid unhandled tracebacks
                return
            except Exception as e:
                logger.error("Stream error: %s", e)
                yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/resume", response_model=ChatResponse)
@limiter.limit(os.getenv("RATE_LIMIT"))
async def chat_resume(request: Request, req: ResumeRequest):
    thread_id, graph, vectorstore, pool, semantic_cache = _get_request_context(req.thread_id)

    t0 = time.perf_counter()
    async with pooled_checkpointer(pool) as saver:
        t_conn = time.perf_counter()
        logger.info("TIMING thread=%s conn_acquire=%.2fs", thread_id[:8], t_conn - t0)

        config = _get_config(thread_id, vectorstore, saver, semantic_cache)
        try:
            result = await graph.ainvoke(Command(resume=req.reply), config)
        except Exception as e:
            logger.error("Resume failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

        requires_approval, question = await _check_interrupt(graph, config)

    if requires_approval:
        return ChatResponse(response="", thread_id=thread_id, requires_approval=True, approval_question=question)

    return ChatResponse(response=result["messages"][-1].content, thread_id=thread_id)


# ── Mount Gradio UI ──────────────────────────────────────────────────
import gradio as gr
from gradio_ui import create_demo

gr.mount_gradio_app(app, create_demo(api_base="http://localhost:8000"), path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_v2:app", host="0.0.0.0", port=8000, reload=True)
    #  uvicorn app_v2:app --reload --port 8000
