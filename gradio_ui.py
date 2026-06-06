"""
Gradio chat interface for ShopAssist.
Connects to the FastAPI backend via HTTP (SSE streaming).
"""
import gradio as gr
import httpx
import uuid
import os
import json

_client = None


def _get_client():
    global _client
    if _client is None:
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=50)
        _client = httpx.Client(timeout=120, limits=limits)
    return _client


def create_demo(api_base: str = None):
    """Build and return a Gradio Blocks demo wired to the FastAPI endpoints."""

    if api_base is None:
        api_base = os.getenv("API_BASE", "http://localhost:8000")

    # ── Streaming chat handler ────────────────────────────────────────
    def respond(message: str, chat_history: list, thread_id: str | None):
        """Stream tokens from /chat/stream and yield updated chat history."""
        if not message or not message.strip():
            yield chat_history, thread_id, gr.update(visible=False), gr.update()
            return

        if not thread_id:
            thread_id = str(uuid.uuid4())

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": "⏳"})

        # Show user message immediately and clear input box
        yield chat_history, thread_id, gr.update(visible=False), gr.update(value="")

        chat_history[-1]["content"] = ""

        accumulated = ""
        approval_needed = False
        approval_question = ""
        current_event = "message"

        try:
            client = _get_client()
            with client.stream(
                "POST",
                f"{api_base}/chat/stream",
                json={"message": message, "thread_id": thread_id},
            ) as response:
                    for line in response.iter_lines():
                        # Empty line = end of SSE event block
                        if not line.strip():
                            current_event = "message"
                            continue

                        if line.startswith("event: "):
                            current_event = line[7:].strip()
                            continue

                        if line.startswith("data: "):
                            raw = line[6:]

                            if current_event == "message":
                                # Decode JSON-encoded token to restore newlines
                                try:
                                    token = json.loads(raw)
                                except json.JSONDecodeError:
                                    token = raw
                                accumulated += token
                                chat_history[-1]["content"] = accumulated
                                yield chat_history, thread_id, gr.update(visible=False), gr.update()

                            elif current_event == "approval_required":
                                approval_question = raw
                                approval_needed = True

                            elif current_event == "metadata":
                                thread_id = raw

                            elif current_event == "error":
                                chat_history[-1]["content"] = f"❌ {raw}"
                                yield chat_history, thread_id, gr.update(visible=False), gr.update()

        except httpx.ConnectError:
            chat_history[-1]["content"] = (
                "❌ Could not connect to the API server. Is it running on "
                f"`{api_base}`?"
            )
            yield chat_history, thread_id, gr.update(visible=False), gr.update()
            return
        except Exception as e:
            chat_history[-1]["content"] = f"❌ {e}"
            yield chat_history, thread_id, gr.update(visible=False), gr.update()
            return

        # If the LLM didn't produce any visible text, remove the empty bubble
        if not accumulated and approval_needed:
            chat_history.pop()

        if approval_needed:
            chat_history.append({
                "role": "assistant",
                "content": (
                    f"🔔 **Approval Required**\n\n{approval_question}\n\n"
                    "Please type **yes** or **no** in the approval box below."
                ),
            })
            yield chat_history, thread_id, gr.update(visible=True), gr.update()
        else:
            yield chat_history, thread_id, gr.update(visible=False), gr.update()

    # ── Approval handler ──────────────────────────────────────────────
    def handle_approval(reply: str, chat_history: list, thread_id: str):
        """Send the approval reply to /chat/resume."""
        if not reply or not reply.strip():
            return chat_history, thread_id, gr.update(visible=True), gr.update()

        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": reply})

        try:
            client = _get_client()
            resp = client.post(
                f"{api_base}/chat/resume",
                json={"thread_id": thread_id, "reply": reply},
            )
            data = resp.json()

            response_text = data.get("response", "")
            if response_text:
                chat_history.append({"role": "assistant", "content": response_text})

            if data.get("requires_approval"):
                q = data.get("approval_question", "Approval required")
                chat_history.append({
                    "role": "assistant",
                    "content": (
                        f"🔔 **Approval Required**\n\n{q}\n\n"
                        "Please type **yes** or **no** in the approval box below."
                    ),
                })
                return chat_history, thread_id, gr.update(visible=True), gr.update()

            return chat_history, thread_id, gr.update(visible=False), gr.update()

        except Exception as e:
            chat_history.append({"role": "assistant", "content": f"❌ {e}"})
            return chat_history, thread_id, gr.update(visible=False), gr.update()

    # ── New chat ──────────────────────────────────────────────────────
    def new_chat():
        return [], None, gr.update(visible=False)

    # ── UI Layout ─────────────────────────────────────────────────────
    custom_css = """
        .gradio-container { max-width: 900px !important; margin: auto; }
        footer { display: none !important; }
        .approval-box {
            border: 2px solid #f59e0b;
            border-radius: 12px;
            padding: 12px 16px;
            background: #fffbeb;
        }
        /* Format markdown tables inside the chatbot */
        .prose table {
            width: 100% !important;
            border-collapse: collapse !important;
            table-layout: auto !important;
            margin: 1em 0 !important;
            font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important;
        }
        .prose th, .prose td {
            border: 1px solid #e5e7eb !important;
            padding: 12px 16px !important;
            text-align: left !important;
            vertical-align: top !important;
            white-space: normal !important;
            word-break: normal !important;
            overflow-wrap: break-word !important;
        }
        .prose th {
            background-color: #f9fafb !important;
            font-weight: 600 !important;
            color: #111827 !important;
        }
        .prose td {
            color: #374151 !important;
        }
    """

    with gr.Blocks(
        title="ShopAssist",
        theme=gr.themes.Soft(
            primary_hue="indigo", 
            secondary_hue="blue",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
        ),
        css=custom_css,
    ) as demo:

        gr.Markdown(
            "# 🛒 ShopAssist\n"
            "> Your AI-powered e-commerce support assistant.  \n"
            "> Ask about **orders**, **refunds**, **policies**, and more."
        )

        thread_state = gr.State(value=None)

        chatbot = gr.Chatbot(
            type="messages",
            height=500,
            show_copy_button=True,
            avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/shopping-cart_1f6d2.png"),
        )

        # ── Main chat input ───────────────────────────────────────────
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Type your message here…",
                show_label=False,
                scale=8,
                container=False,
            )
            send_btn = gr.Button("Send ▶", variant="primary", scale=1, min_width=90)
            clear_btn = gr.Button("🗑️ New Chat", scale=1, min_width=90)

        # ── Approval section (hidden by default) ──────────────────────
        with gr.Column(visible=False, elem_classes="approval-box") as approval_section:
            gr.Markdown("⚠️ **Approval Required** — Type `yes` or `no` below:")
            with gr.Row():
                approval_input = gr.Textbox(
                    placeholder="yes / no",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                approve_btn = gr.Button(
                    "Submit", variant="secondary", scale=1, min_width=90
                )

        # ── Example prompts ───────────────────────────────────────────
        gr.Examples(
            examples=[
                "Hi, what is your return policy?",
                "I need a refund for order ORD-005-1",
                "Show me order history for customer CUST005",
                "What payment methods do you accept?",
            ],
            inputs=msg_input,
            label="💡 Try these",
        )

        # ── Event wiring ──────────────────────────────────────────────
        chat_outputs = [chatbot, thread_state, approval_section, msg_input]

        # Send button
        send_btn.click(
            respond,
            inputs=[msg_input, chatbot, thread_state],
            outputs=chat_outputs,
        )

        # Enter key
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, thread_state],
            outputs=chat_outputs,
        )

        # Approval button
        approve_btn.click(
            handle_approval,
            inputs=[approval_input, chatbot, thread_state],
            outputs=chat_outputs,
        ).then(lambda: "", outputs=approval_input)

        # Approval enter key
        approval_input.submit(
            handle_approval,
            inputs=[approval_input, chatbot, thread_state],
            outputs=chat_outputs,
        ).then(lambda: "", outputs=approval_input)

        # New chat
        clear_btn.click(
            lambda: ([], None, gr.update(visible=False), ""),
            outputs=chat_outputs,
        )

    return demo


# Allow running standalone: python gradio_ui.py
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_port=7860)
    # uvicorn app:app --reload --host 0.0.0.0 --port 8000

