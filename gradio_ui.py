import json
import os
import uuid

import gradio as gr
import httpx


CSS = """
.approval-banner {
    border: 2px solid #f59e0b;
    padding: 14px 18px;
    border-radius: 10px;
    background: #fffbeb;
    margin: 8px 0;
}
.approval-banner h3 {
    color: #b45309;
    margin: 0 0 8px 0;
}
footer { display: none !important; }
"""

_HIDDEN = gr.update(visible=False)
_VISIBLE = gr.update(visible=True)


def create_demo(api_base: str = "http://localhost:8000") -> gr.Blocks:

    async def _stream_chat(
        message: str,
        history: list,
        thread_id: str,
        client: httpx.AsyncClient,
    ):
        """Yield partial chatbot updates as SSE tokens arrive."""
        history = list(history)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})
        yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN

        new_thread_id = thread_id
        accumulated = ""
        approval_pending: str | None = None

        async with client.stream(
            "POST",
            f"{api_base}/chat/stream",
            json={"message": message, "thread_id": thread_id},
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as response:
            if response.status_code != 200:
                history[-1]["content"] = f"**Error ({response.status_code})**"
                yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN
                return

            current_event: str | None = None

            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue

                if line.startswith("data: "):
                    data_content = line[6:]

                    if current_event == "approval_required":
                        approval_pending = data_content.strip()
                        current_event = None
                        continue

                    if current_event == "metadata":
                        new_thread_id = data_content.strip()
                        current_event = None
                        continue

                    if current_event in ("done", None) and data_content == "[DONE]":
                        current_event = None
                        continue

                    if current_event == "error":
                        history[-1]["content"] = f"**Error:** {data_content}"
                        yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN
                        return

                    try:
                        token = json.loads(data_content)
                    except (json.JSONDecodeError, TypeError):
                        token = data_content

                    accumulated += str(token)
                    history[-1]["content"] = accumulated
                    yield history, new_thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN
                    current_event = None

        if approval_pending:
            history[-1]["content"] = f"⚠️  {approval_pending}"
            yield history, new_thread_id, True, approval_pending, _VISIBLE, approval_pending, _VISIBLE, _VISIBLE
        else:
            yield history, new_thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN

    async def _resume_approval(
        reply: str,
        history: list,
        thread_id: str,
        client: httpx.AsyncClient,
    ):
        """Call /chat/resume and yield the result."""
        history = list(history)
        history.append({"role": "user", "content": reply})
        history.append({"role": "assistant", "content": "Processing…"})
        yield history, thread_id, True, "", _VISIBLE, "", _VISIBLE, _VISIBLE

        resp = await client.post(
            f"{api_base}/chat/resume",
            json={"thread_id": thread_id, "reply": reply},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

        if resp.status_code == 200:
            data = resp.json()
            response_text = data.get("response", "")
            if data.get("requires_approval"):
                question = data.get("approval_question", "")
                history[-1]["content"] = f"⚠️  {question}"
                yield history, thread_id, True, question, _VISIBLE, question, _VISIBLE, _VISIBLE
            else:
                history[-1]["content"] = response_text
                yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN
        else:
            history[-1]["content"] = f"**Error ({resp.status_code})**"
            yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN

    async def chat_fn(
        message: str,
        history: list,
        thread_id: str,
        pending_approval: bool,
        approval_question: str,
    ):
        if not message.strip():
            if pending_approval:
                yield history, thread_id, True, approval_question, _VISIBLE, approval_question, _VISIBLE, _VISIBLE
            else:
                yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN
            return

        async with httpx.AsyncClient() as client:
            try:
                if pending_approval:
                    async for result in _resume_approval(message.strip(), history, thread_id, client):
                        yield result
                else:
                    async for result in _stream_chat(message.strip(), history, thread_id, client):
                        yield result
            except httpx.ConnectError:
                history = list(history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": "**Unable to connect to the server.**  \nPlease make sure the API is running."})
                yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN
            except Exception as exc:
                history = list(history)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"**Error:** {exc}"})
                yield history, thread_id, False, "", _HIDDEN, "", _HIDDEN, _HIDDEN

    async def handle_yes(thread_id, pending_approval, history, approval_question):
        async for result in chat_fn("yes", history, thread_id, pending_approval, approval_question):
            yield result

    async def handle_no(thread_id, pending_approval, history, approval_question):
        async for result in chat_fn("no", history, thread_id, pending_approval, approval_question):
            yield result

    def start_new_chat():
        return str(uuid.uuid4()), False, "", [], _HIDDEN, "", _HIDDEN, _HIDDEN

    with gr.Blocks(title="ShopAssist", css=CSS) as demo:
        thread_id_state = gr.State(str(uuid.uuid4()))
        pending_state = gr.State(False)
        approval_q_state = gr.State("")

        gr.Markdown("""
        # ShopAssist — E-Commerce Customer Support

        Ask about your orders, refunds, or store policies.  
        The agent can look up order history, check refund eligibility,
        calculate refund amounts, and process refunds.
        """)

        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
            placeholder="Your conversation will appear here…",
            show_copy_button=True,
            type="messages",
        )

        with gr.Row(visible=False, elem_classes=["approval-banner"]) as approval_row:
            gr.Markdown("### Approval Required")
            approval_display = gr.Markdown("")

        msg = gr.Textbox(
            label="Your message",
            placeholder="e.g. What's the status of order ORD-1729?",
            scale=4,
            container=True,
        )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary", scale=1)
            yes_btn = gr.Button("Yes — Approve", variant="primary", visible=False, scale=1)
            no_btn = gr.Button("No — Cancel", variant="stop", visible=False, scale=1)
            new_chat_btn = gr.Button("New Conversation", variant="secondary", size="sm", scale=1)

        outputs = [
            chatbot,
            thread_id_state,
            pending_state,
            approval_q_state,
            approval_row,
            approval_display,
            yes_btn,
            no_btn,
        ]

        msg.submit(
            fn=chat_fn,
            inputs=[msg, chatbot, thread_id_state, pending_state, approval_q_state],
            outputs=outputs,
        ).then(fn=lambda: "", inputs=None, outputs=[msg])

        send_btn.click(
            fn=chat_fn,
            inputs=[msg, chatbot, thread_id_state, pending_state, approval_q_state],
            outputs=outputs,
        ).then(fn=lambda: "", inputs=None, outputs=[msg])

        yes_btn.click(
            fn=handle_yes,
            inputs=[thread_id_state, pending_state, chatbot, approval_q_state],
            outputs=outputs,
        ).then(fn=lambda: "", inputs=None, outputs=[msg])

        no_btn.click(
            fn=handle_no,
            inputs=[thread_id_state, pending_state, chatbot, approval_q_state],
            outputs=outputs,
        ).then(fn=lambda: "", inputs=None, outputs=[msg])

        new_chat_btn.click(
            fn=start_new_chat,
            inputs=[],
            outputs=[
                thread_id_state,
                pending_state,
                approval_q_state,
                chatbot,
                approval_row,
                approval_display,
                yes_btn,
                no_btn,
            ],
        )

    return demo


if __name__ == "__main__":
    api_base = os.getenv("API_BASE", "http://localhost:8000")
    demo = create_demo(api_base=api_base)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
