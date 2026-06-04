# ShopAssist: Architecture & Development Journey

This document outlines the architecture, key technical decisions, and the journey of building **ShopAssist**—a high-performance, AI-powered e-commerce customer support agent capable of handling 10,000+ requests daily.

---

## 1. Project Overview

The goal was to create an intelligent customer support agent capable of:
- Answering policy questions using a Knowledge Base (RAG).
- Fetching specific order details and customer histories.
- Processing refunds autonomously, but with a **Human-in-the-Loop** approval step for safety.
- Providing a fast, real-time streaming chat experience for users.

---

## 2. Technology Stack

- **Core AI Logic:** `LangGraph` & `LangChain` for orchestrating the agentic workflow and tool execution.
- **LLM Provider:** `ChatDeepSeek` / `ChatOpenAI` (with fallback mechanisms).
- **Backend Framework:** `FastAPI` for robust, asynchronous API endpoints.
- **Frontend UI:** `Gradio` mounted directly onto the FastAPI app for a seamless chat interface.
- **Database:** `SQLAlchemy` (PostgreSQL/SQLite) for relational data (Orders, Items, Customers).
- **Vector Store:** `PGVector` for storing and retrieving knowledge base embeddings.

---

## 3. Key Architectural Implementations

### Agentic Workflow (LangGraph)
We utilized LangGraph to define a state machine for the AI.
- **State:** Maintains the conversation history (`messages`).
- **Nodes:** `llm_call` (reasoning), `tools` (execution), and `human_approval` (interruption).
- **Checkpointer:** Used to persist conversation threads so the agent remembers past messages within a session.

### Human-in-the-Loop (Refund Approvals)
To prevent unauthorized refunds, we implemented an interrupt mechanism in LangGraph. When the agent decides to issue a refund, it pauses execution and yields an `approval_required` event to the frontend. The system waits for the user to explicitly type "yes" or "no" before resuming the graph via the `/chat/resume` endpoint.

### Real-Time Streaming (Server-Sent Events)
Instead of waiting for the LLM to generate the entire response, we implemented Server-Sent Events (SSE) in the `/chat/stream` endpoint. As the LLM generates tokens, they are streamed to the frontend instantly, providing a low-latency, conversational feel.

---

## 4. Scaling to 10,000+ Requests/Day (The Challenges & Solutions)

To ensure the application could handle high traffic gracefully, we encountered and solved several critical bottlenecks:

### Challenge 1: The Gradio UI Rendering Lag
**Issue:** The LLM was streaming tokens so fast that yielding every single character to Gradio caused hundreds of UI re-renders per second, freezing the browser.
**Solution:** We implemented **time-based UI throttling** in the Gradio generator. We aggregate the incoming tokens and yield to the frontend at a maximum of 20 frames per second (every 0.05 seconds). This completely eliminated the lag while maintaining the smooth typing effect.

### Challenge 2: Blocking the Async Event Loop
**Issue:** LangGraph's `graph.invoke` and `graph.stream` are synchronous operations. Running them directly in FastAPI endpoints blocked the async event loop, meaning one slow LLM call would freeze the entire server for all other users.
**Solution:** We offloaded the synchronous LangGraph executions to background threads using `asyncio.to_thread` for standard calls, and a dedicated `ThreadPoolExecutor` bridged with an `asyncio.Queue` for the streaming endpoint. This allows FastAPI to handle hundreds of concurrent connections asynchronously.

### Challenge 3: Memory Leaks in Checkpointing
**Issue:** Initially, the graph used `MemorySaver()`. At 10,000 requests/day, storing every conversation thread in RAM would inevitably cause an Out of Memory (OOM) crash.
**Solution:** We migrated to `AsyncPostgresSaver` (connected to Supabase). This persists the graph state to the database, keeping the application stateless and memory usage flat, regardless of how many users chat with the agent.

### Challenge 4: Cache Thread Safety
**Issue:** To speed up knowledge base and order lookups, we introduced `TTLCache`. However, because tool calls were now running in concurrent background threads, simultaneous cache access could crash the app (`RuntimeError: dictionary changed size during iteration`).
**Solution:** We wrapped all cache read/write operations in a `threading.Lock()`, ensuring thread-safe access without sacrificing performance.

### Challenge 5: Rate Limiting
**Issue:** To protect the LLM API budget from abuse or DDoS attacks.
**Solution:** We implemented `slowapi`, setting a reasonable limit of `20 requests/minute` per IP address across all chat endpoints.

---

## 5. Load Testing & Validation

Before considering the system production-ready, we built a comprehensive load test using **Locust** (`loadtest.py`). 

The test simulated realistic traffic:
- 40% Knowledge base queries
- 30% Order/Refund queries
- 20% Streaming chat interactions
- 10% Greetings and health checks

**Results:**
Running 30 concurrent users, the system easily sustained **~5.6 requests per second (RPS)** with a 0% error rate (excluding intentional rate-limit blocks). 
*5.6 RPS translates to nearly 490,000 requests per day—proving the architecture has massive headroom beyond the 10,000 requests/day target.*
