import logging
from langchain_core.tools import tool
from data import Order, SessionLocal
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from cachetools import TTLCache
import threading
from .retry import retry_on_db_error

logger = logging.getLogger(__name__)

_order_details_cache = TTLCache(maxsize=1024, ttl=30)
_order_details_cache_lock = threading.Lock()

class OrderDetailsResponse(BaseModel):
    order_id: Optional[str] = None
    customer_id: Optional[str] = None
    amount: Optional[float] = None
    status: Optional[str] = None
    delivery_date: Optional[str] = None
    payment_status: Optional[str] = None
    refund_status: Optional[str] = None
    items: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

@tool("get_order_details",
    description=(
        "Fetch full details for a specific order using order_id. "
        "Use this when the customer asks about one order, gives an order ID, "
        "asks for refund of a specific order, asks order status, delivery status, "
        "payment status, or item details for a specific order."
    ))
@retry_on_db_error()
def get_order_details(order_id: str) -> str:
    if not order_id or not order_id.strip():
        return OrderDetailsResponse(error='order_id is required').model_dump_json()

    cache_key = order_id.upper().strip()
    with _order_details_cache_lock:
        if cache_key in _order_details_cache:
            logger.info("Order details cache hit for: %s", cache_key)
            return _order_details_cache[cache_key]

    db = SessionLocal()
    try:
        order_id = cache_key
        logger.info("Fetching order details for: %s", order_id)

        order_details = db.query(Order).filter(Order.order_id == order_id).first()
        if not order_details:
            logger.info("Order not found: %s", order_id)
            return OrderDetailsResponse(error='No order found', order_id=order_id).model_dump_json()
        
        logger.info("Found order %s with status: %s", order_id, order_details.status)
        
        items_list = []
        for item in order_details.items:
            items_list.append({
                "product_id": item.product_id,
                "name": item.name,
                "price": item.price,
                "quantity": item.quantity
            })

        result = OrderDetailsResponse(
            order_id=order_details.order_id,
            customer_id=order_details.customer_id,
            amount=float(order_details.total_amount) if order_details.total_amount is not None else 0.0,
            status=order_details.status,
            delivery_date=order_details.delivery_date,
            payment_status=order_details.payment_status,
            refund_status=order_details.refund_status,
            items=items_list
        ).model_dump_json()
        with _order_details_cache_lock:
            _order_details_cache[cache_key] = result
        return result
    except Exception as e:
        db.rollback()
        logger.error("Error fetching order details for %s: %s", order_id, e)
        raise
    finally:
        db.close()