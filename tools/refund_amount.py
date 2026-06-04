import logging
from langchain_core.tools import tool
from data import Order, SessionLocal
from pydantic import BaseModel
from typing import Optional

logger = logging.getLogger(__name__)

class RefundAmountResponse(BaseModel):
    order_id: Optional[str] = None
    refund_amount: Optional[float] = None
    error: Optional[str] = None

@tool(
    "calculate_refund_amount",
    description="Calculate refund amount for a specific order. Returns the refund amount as a number."
)
def calculate_refund_amount(order_id: str) -> str:
    if not order_id or not order_id.strip():
        return RefundAmountResponse(order_id="", error='order_id is required').model_dump_json()

    db = SessionLocal()
    try:
        order_id = order_id.upper().strip()
        logger.info("Calculating refund amount for order: %s", order_id)
        order = db.query(Order).filter(Order.order_id == order_id).first()

        if not order:
            logger.info("Order not found: %s", order_id)
            return RefundAmountResponse(order_id=order_id, error="Order not found").model_dump_json()

        refund_amount = float(order.total_amount) if order.total_amount is not None else 0.0
        logger.info("Refund amount for order %s: %s", order_id, refund_amount)
        return RefundAmountResponse(order_id=order_id, refund_amount=refund_amount).model_dump_json()
    except Exception as e:
        db.rollback()
        logger.error("Error calculating refund amount for %s: %s", order_id, e)
        raise
    finally:
        db.close()