import logging
import datetime as dt
from typing import Optional
from langchain_core.tools import tool
from data import Order, SessionLocal
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EligibilityResponse(BaseModel):
    eligible: bool
    order_id: str
    reason: str
    requires_approval: Optional[bool] = None

@tool(
    "check_refund_eligibility",
    description=(
        "Check whether an order is eligible for refund using order_id. "
        "Use this before creating or processing any refund."
    )
)
def check_refund_eligibility(order_id: str) -> str:
    if not order_id or not order_id.strip():
        return EligibilityResponse(
            eligible=False, order_id="", reason="order_id is required."
        ).model_dump_json()

    db = SessionLocal()
    try:
        order_id = order_id.upper().strip()
        logger.info("Checking refund eligibility for order: %s", order_id)
        order = db.query(Order).filter(Order.order_id == order_id).first()

        if not order:
            logger.info("Order not found: %s", order_id)
            return EligibilityResponse(
                eligible=False,
                order_id=order_id,
                reason="Order not found."
            ).model_dump_json()

        if order.status != "delivered":
            return EligibilityResponse(
                eligible=False,
                order_id=order_id,
                reason=f"Order status is '{order.status}', must be delivered."
            ).model_dump_json()

        if order.payment_status != "paid":
            return EligibilityResponse(
                eligible=False,
                order_id=order_id,
                reason=f"Payment status is '{order.payment_status}', must be paid."
            ).model_dump_json()

        if order.refund_status != "none":
            return EligibilityResponse(
                eligible=False,
                order_id=order_id,
                reason=f"Refund status is '{order.refund_status}'."
            ).model_dump_json()

        if not order.delivery_date:
            return EligibilityResponse(
                eligible=False,
                order_id=order_id,
                reason="Delivery date missing."
            ).model_dump_json()

        try:
            days = (dt.datetime.now().date() -
                    dt.datetime.strptime(order.delivery_date, "%Y-%m-%d").date()).days
        except ValueError:
            return EligibilityResponse(
                eligible=False,
                order_id=order_id,
                reason="Invalid delivery date format."
            ).model_dump_json()

        if days > 7:
            return EligibilityResponse(
                eligible=False,
                order_id=order_id,
                reason=f"Refund window expired. {days} days since delivery, max 7."
            ).model_dump_json()
        
        logger.info("Order %s is eligible for refund (requires_approval=%s)",
                     order_id, order.total_amount > 500)
        return EligibilityResponse(
            eligible=True,
            order_id=order_id,
            reason="Within 7-day window, delivered, paid, no prior refund.",
            requires_approval=order.total_amount > 500,
        ).model_dump_json()
    except Exception as e:
        db.rollback()
        logger.error("Error checking refund eligibility for %s: %s", order_id, e)
        raise
    finally:
        db.close()