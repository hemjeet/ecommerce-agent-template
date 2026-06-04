from typing import TypedDict, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class EcomAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    refund_elligible: Optional[bool]
    refund_amount: Optional[float]
    requires_approval: Optional[bool]
    refund_status : Optional[str]
    order_id: Optional[str]
    user_reply : Optional[str]

    
    

