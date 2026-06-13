import json
import logging
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.messages import trim_messages, filter_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import interrupt, Command, RetryPolicy
import openai
import httpx
import asyncio
from .prompts import SYSTEM_PROMPT
from .state import EcomAgentState
import tiktoken
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Max tokens to send to the LLM (excluding system prompt)
MAX_CONTEXT_TOKENS = 4000

def tiktoken_counter(messages: list[BaseMessage]) -> int:
    """Fallback token counter using cl100k_base since deepseek model isn't recognized."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    num_tokens = 0
    for msg in messages:
        # Rough approximation: 4 tokens overhead per message
        num_tokens += len(encoding.encode(str(msg.content))) + 4
    return num_tokens


class EcomAgent:
    def __init__(self, llm, tools, checkpointer=None):
        self.tools = tools
        self.base_llm = llm
        self.llm = llm.bind_tools(self.tools)
        self.checkpointer = checkpointer
        self.build_graph = self._graph()

    
    async def _llm_call(self, state: EcomAgentState, config: RunnableConfig):
        messages = state['messages']
        logger.info("Calling LLM with %d messages (before trim)", len(messages))
        
        filtered_messages = filter_messages(messages, include_types=["human", "ai", "tool"])
        
        trimmed = trim_messages(
            filtered_messages,
            max_tokens=MAX_CONTEXT_TOKENS,
            token_counter=tiktoken_counter,
            strategy="last",
            start_on="human",
            include_system=False,  # we prepend SystemMessage ourselves
            allow_partial=False,
        )
        
        
        if not trimmed:
            trimmed = messages[-10:]
        
        logger.info("Trimmed to %d messages (from %d)", len(trimmed), len(messages))


        response = await self.llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT), *trimmed
        ], config)
        logger.info("LLM responded with tool_calls=%s", 
                     [tc['name'] for tc in getattr(response, 'tool_calls', []) or []])
        return {'messages': [response]}
    
    def _evaluate_tool_results(self, state: EcomAgentState):
        """Runs after tools to parse specific tool results and update the state."""
        updates = {}
        
        for msg in reversed(state['messages']):
            if not isinstance(msg, ToolMessage):
                break
            
            logger.info("Evaluating tool result: %s", msg.name)
                
            if msg.name == 'check_refund_eligibility':
                try:
                    result = json.loads(msg.content)
                    updates['refund_elligible'] = result.get('eligible')
                    logger.info("Refund eligibility for order: %s", result.get('eligible'))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse check_refund_eligibility response")
            elif msg.name == 'calculate_refund_amount':
                try:
                    result = json.loads(msg.content)
                    if 'refund_amount' in result and result['refund_amount'] is not None:
                        updates['refund_amount'] = result['refund_amount']
                        logger.info("Refund amount calculated: %s", result['refund_amount'])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse calculate_refund_amount response")
            elif msg.name == 'process_refund':
                try:
                    result = json.loads(msg.content)
                    if result.get('success'):
                        updates['refund_status'] = 'approved'
                        updates['order_id'] = result.get('order_id')
                        logger.info("Refund approved for order %s", result.get('order_id'))
                    elif result.get('amount', 0) >= 500:
                        updates['requires_approval'] = True
                        updates['order_id'] = result.get('order_id')
                        updates['refund_amount'] = result.get('amount')
                        logger.info("Refund requires approval for order %s", result.get('order_id'))
                    else:
                        updates['refund_status'] = 'failed'
                        logger.warning("Refund processing failed: %s", result.get('message'))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse process_refund response")
                    
        return updates

    def _human_approval_node(self, state: EcomAgentState):
        last_message = state['messages'][-1]
        
        amount = state.get("refund_amount", "unknown")
        order_id = state.get("order_id", "unknown")
        
        # Parse from the tool call args if missing from state
        for tc in getattr(last_message, 'tool_calls', []):
            if tc['name'] == 'create_refund_approval_ticket':
                args = tc.get('args', {})
                amount = args.get('amount', amount)
                order_id = args.get('order_id', order_id)
                break

        logger.info("Requesting human approval for order %s, amount %s", order_id, amount)

        user_reply = interrupt({
            "question": (
                f"Refund of {amount} for order {order_id} exceeds 500 "
                f"and needs a support ticket. Create it? (yes / no)"
            )
        })

        logger.info("Human replied: %s", user_reply)

        if isinstance(user_reply, str) and user_reply.lower().strip() == 'yes':
            return {'user_reply': user_reply}
        else:
            # Cancel the tool call
            tool_messages = []
            for tc in getattr(last_message, 'tool_calls', []):
                tool_messages.append(ToolMessage(
                    tool_call_id=tc['id'],
                    name=tc['name'],
                    content="Human denied the request to create an approval ticket."
                ))
            return {'user_reply': str(user_reply), 'messages': tool_messages}

    def _should_continue(self, state: EcomAgentState):
        last_message = state['messages'][-1]

        if getattr(last_message, 'tool_calls', None):
            for tc in last_message.tool_calls:
                if tc['name'] == 'create_refund_approval_ticket':
                    return 'human_approval'
            return 'tools'
        return END

    def _route_after_approval(self, state: EcomAgentState):
        reply = state.get('user_reply', '')
        if isinstance(reply, str) and reply.lower().strip() == 'yes':
            return 'tools'
        return 'llm_call'

    def _graph(self):
        retry_exceptions = (
            ConnectionError,
            TimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.APITimeoutError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
        )
        graph = StateGraph(EcomAgentState)

        graph.add_node('llm_call', self._llm_call, retry=RetryPolicy(
            max_attempts=3,
            retry_on=lambda e: isinstance(e, retry_exceptions)
            and not (isinstance(e, openai.APIStatusError) 
            and e.status_code in (400, 401, 403, 422))
        ))
        graph.add_node('tools', ToolNode(self.tools, handle_tool_errors=True))
        graph.add_node('evaluate_tool_results', self._evaluate_tool_results)
        graph.add_node('human_approval', self._human_approval_node)

        graph.add_edge(START, 'llm_call')
        
        graph.add_conditional_edges(
            'llm_call', self._should_continue, {
                'human_approval': 'human_approval',
                'tools': 'tools',
                END: END
            }
        )
        
        graph.add_conditional_edges(
            'human_approval', self._route_after_approval, {
                'tools': 'tools',
                'llm_call': 'llm_call'
            }
        )

        graph.add_edge('tools', 'evaluate_tool_results')
        graph.add_edge('evaluate_tool_results', 'llm_call')

        checkpointer = self.checkpointer if self.checkpointer is not None else MemorySaver()
        return graph.compile(checkpointer=checkpointer)
