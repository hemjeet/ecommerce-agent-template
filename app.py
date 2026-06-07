import os
import uuid
import json
import logging
import sys
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import dict_row
from psycopg import AsyncConnection

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
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
        ]
    )
    logging.getLogger().handlers[1].setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ── Pydantic schemas ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID. A new one is generated if omitted."
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
    """Initialize heavy resources once at startup, clean up on shutdown."""
    load_dotenv()
    start_time = time.perf_counter()
    logger.info("─" * 56)
    logger.info("  Startup — EcomAgent Initialization")
    logger.info("─" * 56)

    # 1. LLM with fallback
    llm = ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
    )
    fallback_llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    llm_with_fallback = llm.with_fallbacks([fallback_llm])
    logger.info("  [ OK ] DeepSeek LLM loaded (%s)", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    logger.info("  [ OK ] OpenAI fallback LLM loaded (%s)", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # 2. Vectorstore (Supabase PGVector)
    postgres_uri = os.getenv("POSTGRES_URI")
    if postgres_uri:
        try:
            vectorstore = PGVector(
                embeddings=OpenAIEmbeddings(),
                collection_name="knowledge_base",
                connection=postgres_uri,
                use_jsonb=True,
            )
            logger.info("  [ OK ] Vectorstore (Supabase/PGVector) connected")
        except Exception as e:
            vectorstore = None
            logger.warning("  [FAIL] Vectorstore connection failed: %s", e)
    else:
        vectorstore = None
        logger.warning("  [SKIP] POSTGRES_URI not set - vectorstore disabled")

    # 3. Tools
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

    # 4. Build agent graph with shared checkpointer
    postgres_uri = os.getenv("POSTGRES_URI")
    checkpointer = None
    if postgres_uri:
        try:
            conn = await AsyncConnection.connect(
                postgres_uri,
                autocommit=True,
                prepare_threshold=0,
                row_factory=dict_row,
            )
            checkpointer = AsyncPostgresSaver(conn=conn)
            await checkpointer.setup()
            logger.info("  [ OK ] AsyncPostgresSaver checkpointer ready")
        except Exception as e:
            checkpointer = None
            logger.warning("  [SKIP] Checkpointer connection failed: %s - using in-memory", e)

    agent = EcomAgent(llm=llm_with_fallback, tools=tools, checkpointer=checkpointer)
    graph = agent.build_graph

    logger.info("  [ OK ] Agent graph compiled")

    # Store on app.state so endpoints can access them
    app.state.graph = graph
    app.state.vectorstore = vectorstore
    app.state.checkpointer = checkpointer

    elapsed = time.perf_counter() - start_time
    logger.info("─" * 56)
    logger.info("  Startup complete (%.2fs)", elapsed)
    logger.info("─" * 56)
    yield
    # Shutdown cleanup
    logger.info("─" * 56)
    logger.info("  Shutdown")
    logger.info("─" * 56)
    if checkpointer is not None:
        try:
            await checkpointer.conn.close()
            logger.info("  [ OK ] AsyncPostgresSaver connection closed")
        except Exception as e:
            logger.warning("  [FAIL] Error closing checkpointer: %s", e)


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
def _get_config(thread_id: str, vectorstore):
    return {"configurable": {"thread_id": thread_id, "vectorstore": vectorstore}}


async def _check_interrupt(graph, config):
    """Return (requires_approval, question) if the graph is paused."""
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


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest):
    """Send a message and receive the full response at once."""
    graph = app.state.graph
    vectorstore = app.state.vectorstore
    thread_id = req.thread_id or str(uuid.uuid4())
    config = _get_config(thread_id, vectorstore)

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config,
        )
    except Exception as e:
        logger.error("Graph invoke failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    requires_approval, question = await _check_interrupt(graph, config)
    if requires_approval:
        return ChatResponse(
            response="",
            thread_id=thread_id,
            requires_approval=True,
            approval_question=question,
        )

    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=thread_id,
    )


@app.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, req: ChatRequest):
    """Send a message and receive the response as a Server-Sent Events stream."""
    graph = app.state.graph
    vectorstore = app.state.vectorstore
    thread_id = req.thread_id or str(uuid.uuid4())
    config = _get_config(thread_id, vectorstore)

    async def event_stream():
        try:
            async for msg, metadata in graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                config,
                stream_mode="messages",
            ):
                if (
                    isinstance(msg, AIMessageChunk)
                    and msg.content
                    and metadata.get("langgraph_node") == "llm_call"
                ):
                    yield f"data: {json.dumps(msg.content)}\n\n"

            requires_approval, question = await _check_interrupt(graph, config)
            if requires_approval:
                yield f"event: approval_required\ndata: {question}\n\n"
            yield f"event: metadata\ndata: {thread_id}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error("Stream error: %s", e)
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/resume", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat_resume(request: Request, req: ResumeRequest):
    """Resume a paused conversation after human approval (yes/no)."""
    graph = app.state.graph
    vectorstore = app.state.vectorstore
    config = _get_config(req.thread_id, vectorstore)

    try:
        result = await graph.ainvoke(Command(resume=req.reply), config)
    except Exception as e:
        logger.error("Resume failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    requires_approval, question = await _check_interrupt(graph, config)
    if requires_approval:
        return ChatResponse(
            response="",
            thread_id=req.thread_id,
            requires_approval=True,
            approval_question=question,
        )

    return ChatResponse(
        response=result["messages"][-1].content,
        thread_id=req.thread_id,
    )


# ── Mount Gradio UI ──────────────────────────────────────────────────
import gradio as gr
from gradio_ui import create_demo

demo = create_demo(api_base="http://localhost:8000")
app = gr.mount_gradio_app(app, demo, path="/ui")
