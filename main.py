import uuid
import os
import sys
import logging
from agent import EcomAgent
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from tools.order_history import get_order_history
from tools.order_tools import get_order_details
from tools.refund_eligibility import check_refund_eligibility
from tools.refund_amount import calculate_refund_amount
from tools.create_refund_ticket import create_refund_approval_ticket
from tools.process_refund import process_refund
from tools.search_knowledge_base import search_knowledge_base
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv
from langgraph.types import Command

# Configure logging: file gets everything, console only shows warnings+
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
# Keep console output clean — only warnings and errors
logging.getLogger().handlers[1].setLevel(logging.WARNING)


def main():
    load_dotenv()
    # Initialize LLM and Tools
    llm = ChatDeepSeek(model="deepseek-v4-flash", api_key=os.getenv("DEEPSEEK_API_KEY"))
    
    postgres_uri = os.getenv("POSTGRES_URI")
    if postgres_uri:
        vectorstore = PGVector(
            embeddings=OpenAIEmbeddings(),
            collection_name="knowledge_base",
            connection=postgres_uri,
            use_jsonb=True,
        )
    else:
        vectorstore = None
    fallback_llm = ChatOpenAI(model="gpt-5-nano-2025-08-07")
    llm_with_fallback = llm.with_fallbacks([fallback_llm])
    
    tools = [
        get_order_history, 
        get_order_details, 
        check_refund_eligibility, 
        calculate_refund_amount,
        process_refund,
        create_refund_approval_ticket,
        search_knowledge_base
    ]
    
    # Pass LLM and tools to the agent
    agent = EcomAgent(llm=llm_with_fallback, tools=tools)
    graph = agent.build_graph
    thread_id = str(uuid.uuid4())
    
    print("Agent started. Type 'exit' or 'quit' to stop.")
    pending_interrupt = False
    while True:
        try:
            user_input = input("You: ")
            
            # Allow the user to exit gracefully
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            config = {'configurable': {'thread_id': thread_id, 'vectorstore': vectorstore}}
            if pending_interrupt:
                result = graph.invoke(Command(resume=user_input), config)
                pending_interrupt = False
            else:
                result = graph.invoke({
                    "messages": [HumanMessage(content=user_input)]
                }, config)
            
            # Check if the agent paused using the LangGraph state
            state = graph.get_state(config)
            if state.tasks and getattr(state.tasks[0], 'interrupts', None):
                interrupt_value = state.tasks[0].interrupts[0].value
                question = interrupt_value.get("question", "Approval required (yes/no): ")
                print(f"\n[SYSTEM]: {question}")
                pending_interrupt = True
            else:
                pending_interrupt = False
                chat = result['messages'][-1].content
                print(f"Agent: {chat}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()