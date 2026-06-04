"""
Load test for ShopAssist API using Locust.

Run with:
    locust -f loadtest.py --host http://localhost:8000

Then open http://localhost:8089 in browser to configure:
  - Number of users: 50
  - Spawn rate: 5/s
  - Run time: 5m

This simulates realistic user behavior across all endpoints.
"""

import uuid
import random
from locust import HttpUser, task, between, tag


# Realistic test data
CUSTOMER_IDS = ["CUST001", "CUST002", "CUST003", "CUST004", "CUST005"]
ORDER_IDS = [
    "ORD1001", "ORD1002", "ORD1003", "ORD1004", "ORD1005",
    "ORD1006", "ORD1007",
    "ORD-005-1", "ORD-005-2", "ORD-005-3", "ORD-005-4", "ORD-005-5",
]

KB_QUERIES = [
    "What is your return policy?",
    "How do I get a refund?",
    "What payment methods do you accept?",
    "Tell me about shipping options",
    "What is the warranty policy?",
    "How can I cancel my order?",
]

ORDER_QUERIES = [
    "Show me order history for customer {cid}",
    "What's the status of order {oid}?",
    "I need a refund for order {oid}",
    "What items did I buy in order {oid}?",
    "Can you check the delivery date for {oid}?",
]


class ShopAssistUser(HttpUser):
    """Simulates a customer interacting with ShopAssist."""

    # Wait 2-6 seconds between requests (realistic chat pacing)
    wait_time = between(2, 6)

    def on_start(self):
        """Each simulated user gets their own thread_id."""
        self.thread_id = str(uuid.uuid4())
        self.customer_id = random.choice(CUSTOMER_IDS)

    # ── Health check (lightweight, ~5% of traffic) ────────────────────

    @tag("health")
    @task(1)
    def health_check(self):
        self.client.get("/health")

    # ── Knowledge base questions (most common, ~40% of traffic) ───────

    @tag("chat", "knowledge")
    @task(8)
    def ask_knowledge_question(self):
        question = random.choice(KB_QUERIES)
        self.client.post(
            "/chat",
            json={
                "message": question,
                "thread_id": self.thread_id,
            },
            name="/chat [knowledge]",
        )

    # ── Order queries (~30% of traffic) ───────────────────────────────

    @tag("chat", "orders")
    @task(6)
    def ask_order_question(self):
        template = random.choice(ORDER_QUERIES)
        question = template.format(
            cid=self.customer_id,
            oid=random.choice(ORDER_IDS),
        )
        self.client.post(
            "/chat",
            json={
                "message": question,
                "thread_id": self.thread_id,
            },
            name="/chat [order]",
        )

    # ── Streaming endpoint (~20% of traffic) ──────────────────────────

    @tag("stream")
    @task(4)
    def stream_chat(self):
        question = random.choice(
            KB_QUERIES + [
                f"Show me order history for customer {self.customer_id}",
                f"What's the status of order {random.choice(ORDER_IDS)}?",
            ]
        )
        # Read the full SSE stream to measure end-to-end latency
        with self.client.post(
            "/chat/stream",
            json={
                "message": question,
                "thread_id": str(uuid.uuid4()),  # fresh thread per stream
            },
            stream=True,
            catch_response=True,
            name="/chat/stream",
        ) as response:
            if response.status_code == 200:
                # Consume the full stream
                content = b""
                for chunk in response.iter_content(chunk_size=1024):
                    content += chunk
                if b"[DONE]" in content:
                    response.success()
                else:
                    response.failure("Stream did not complete with [DONE]")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")

    # ── Greeting (simple, ~5% of traffic) ─────────────────────────────

    @tag("chat", "greeting")
    @task(1)
    def send_greeting(self):
        greetings = ["Hi", "Hello", "Hey there", "Good morning"]
        self.client.post(
            "/chat",
            json={
                "message": random.choice(greetings),
                "thread_id": self.thread_id,
            },
            name="/chat [greeting]",
        )
