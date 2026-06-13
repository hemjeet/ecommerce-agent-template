# ShopAssist API

ShopAssist is an intelligent e-commerce customer support agent powered by LangGraph, FastAPI, and large language models. It handles customer inquiries, processes refunds, and searches store policies.

## Features

*   **Intelligent Chatbot:** Powered by advanced LLMs (DeepSeek, OpenAI) via LangGraph.
*   **Order Management:** Fetch order history and specific order details.
*   **Automated Refunds:** Check refund eligibility, calculate amounts, and process refunds with a human-in-the-loop approval step.
*   **Knowledge Base Integration:** Answers questions about return policies, shipping, and FAQs using a vector database (PGVector).
*   **Streaming Responses:** Real-time character-by-character responses via Server-Sent Events (SSE).
*   **Gradio UI:** An interactive web interface for users to chat with the agent.
*   **Rate Limiting:** Built-in protection against abuse using `slowapi`.
*   **Production Ready:** Designed to handle high traffic with asynchronous processing and robust database connections.

## Prerequisites

*   Python 3.10+
*   PostgreSQL (optional, defaults to SQLite or local Postgres if configured)
*   API Keys for LLM providers (e.g., DeepSeek, OpenAI)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd ecom-agent-new
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your configuration:
    ```env
    DEEPSEEK_API_KEY=your_deepseek_api_key
    OPENAI_API_KEY=your_openai_api_key
    POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/ecom
    ```
    *(Make sure not to commit the `.env` file!)*

## Running the Application

1.  **Seed the Database (Optional but recommended for testing):**
    ```bash
    python seed_db.py
    python add_more_data.py
    ```

2.  **Start the Server:**
    ```bash
    python app.py
    ```

3.  **Access the UI:**
    Open your browser and navigate to:
    `http://localhost:8000/ui`

## API Endpoints

*   `GET /health`: Health check endpoint.
*   `POST /chat`: Send a message and get a full response.
*   `POST /chat/stream`: Send a message and get a streamed response (SSE).
*   `POST /chat/resume`: Resume a paused conversation (e.g., after human approval).

## Testing

A Locust load testing script is included to test performance.
```bash
locust -f loadtest.py --host http://localhost:8000
```
Then navigate to `http://localhost:8089` to start the test.
