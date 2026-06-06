import logging
import uuid
from langchain_core.tools import tool
from data import Order, SessionLocal
from pydantic import BaseModel
from typing import Optional
from .retry import retry_on_db_error

logger = logging.getLogger(__name__)

class RefundResponse(BaseModel):
    success: bool
    order_id: str
    message: str
    amount: float
    refund_id: Optional[str] = None


@tool(
    "process_refund",
    description=(
        "Process the actual refund for a specific order. "
        "Use this only after refund eligibility is confirmed and refund amount is calculated. "
        "Do not use this for refunds above 500 unless human approval has already been granted."
    )
)
@retry_on_db_error()
def process_refund(order_id: str, customer_id: str, amount: float, reason: str) -> str:
    # Input validation
    if not order_id or not order_id.strip():
        return RefundResponse(
            success=False, order_id="", message="order_id is required.", amount=0.0
        ).model_dump_json()
    if not customer_id or not customer_id.strip():
        return RefundResponse(
            success=False, order_id=order_id, message="customer_id is required.", amount=0.0
        ).model_dump_json()
    if amount <= 0:
        return RefundResponse(
            success=False, order_id=order_id, message="Refund amount must be positive.", amount=0.0
        ).model_dump_json()

    db = SessionLocal()
    try:
        order_id = order_id.upper().strip()
        logger.info("Processing refund for order %s, amount %.2f", order_id, amount)
        
        # Lock the row to prevent double-refunds from concurrent requests
        order = db.query(Order).filter(Order.order_id == order_id).with_for_update().first()

        if not order:
            return RefundResponse(
                success=False, order_id=order_id, message="Order not found.", amount=0.0
            ).model_dump_json()
        
        if order.customer_id != customer_id.upper().strip():
            return RefundResponse(
                success=False, order_id=order_id, message="Order not found.", amount=0.0
            ).model_dump_json()

        if order.refund_status != "none":
            logger.warning("Duplicate refund attempt for order %s (status: %s)", order_id, order.refund_status)
            return RefundResponse(
                success=False, order_id=order_id, message="Refund already processed.", amount=0.0
            ).model_dump_json()
            
        if amount >= 500:
            return RefundResponse(
                success=False, order_id=order_id,
                message="Refund amount is 500 or greater. Human approval is required.",
                amount=amount
            ).model_dump_json()
            
        refund_id = f"REF{uuid.uuid4().hex[:8].upper()}"
        
        order.refund_status = 'refunded'
        order.refund_id = refund_id
        db.commit()

        logger.info("Refund %s processed successfully for order %s", refund_id, order_id)
        return RefundResponse(
            success=True, order_id=order_id,
            message="Refund processed successfully.",
            amount=amount, refund_id=refund_id,
        ).model_dump_json()
    except Exception as e:
        db.rollback()
        logger.error("Error processing refund for %s: %s", order_id, e)
        raise
    finally:
        db.close()
