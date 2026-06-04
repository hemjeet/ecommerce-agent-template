import logging
import uuid
from langchain_core.tools import tool
from data import Order, SessionLocal
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

class ApprovalResponse(BaseModel):
    success: bool
    order_id: str
    message: str
    ticket_id: Optional[str] = None


@tool(
    "create_refund_approval_ticket",
    description=(
        "Create a human approval ticket for a refund request. "
        "Use this when refund amount is greater than 500 and human approval is required. "
        "This does not process the refund."
    )
)
def create_refund_approval_ticket(
    order_id: str,
    customer_id: str,
    amount: float,
    reason: str
) -> str:
    """
    Create a human approval ticket for a refund request.
    Use this when refund amount is greater than 500 and human approval is required.
    This does not process the refund.
    """
    # Input validation
    if not order_id or not order_id.strip():
        return ApprovalResponse(
            success=False, order_id="", message="order_id is required.", ticket_id=None
        ).model_dump_json()
    if not customer_id or not customer_id.strip():
        return ApprovalResponse(
            success=False, order_id=order_id, message="customer_id is required.", ticket_id=None
        ).model_dump_json()
    if amount <= 0:
        return ApprovalResponse(
            success=False, order_id=order_id, message="Amount must be positive.", ticket_id=None
        ).model_dump_json()

    db = SessionLocal()
    try:
        order_id = order_id.upper().strip()
        logger.info("Creating approval ticket for order %s, amount %.2f", order_id, amount)
        
        # Lock the row to prevent duplicate tickets from concurrent requests
        order = db.query(Order).filter(Order.order_id == order_id).with_for_update().first()

        if not order:
            return ApprovalResponse(
                success=False, order_id=order_id, message="Order not found.", ticket_id=None
            ).model_dump_json()
        
        if order.customer_id != customer_id.upper().strip():
            return ApprovalResponse(
                success=False, order_id=order_id, message="Order not found.", ticket_id=None
            ).model_dump_json()

        if order.refund_status != "none":
            logger.warning("Duplicate ticket attempt for order %s (status: %s)", order_id, order.refund_status)
            return ApprovalResponse(
                success=False, order_id=order_id, message="Refund already processed.", ticket_id=None
            ).model_dump_json()
            
        ticket_id = f"TKT{uuid.uuid4().hex[:8].upper()}"
        
        order.refund_status = 'pending_approval'
        order.ticket_id = ticket_id
        db.commit()

        logger.info("Approval ticket %s created for order %s", ticket_id, order_id)
        return ApprovalResponse(
            success=True, order_id=order_id,
            message="Refund approval ticket created successfully.",
            ticket_id=ticket_id,
        ).model_dump_json()
    except Exception as e:
        db.rollback()
        logger.error("Error creating approval ticket for %s: %s", order_id, e)
        raise
    finally:
        db.close()
