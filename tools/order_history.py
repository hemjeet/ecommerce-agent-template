import logging
from langchain_core.tools import tool
from data import Order, SessionLocal
from pydantic import BaseModel
from typing import Optional, Dict, Any
from cachetools import TTLCache
import threading

logger = logging.getLogger(__name__)

_order_history_cache = TTLCache(maxsize=1024, ttl=30)
_order_history_cache_lock = threading.Lock()

class OrderHistoryResponse(BaseModel):
    orders: Optional[Dict[str, Dict[str, Any]]] = None
    error: Optional[str] = None

@tool("get_order_history",
    description=(
        "Fetch orders for a customer using customer_id. "
        "Use this when the customer asks about order history, past orders, or all orders. "
    ))
def get_order_history(customer_id: str) -> str:
    if not customer_id or not customer_id.strip():
        return OrderHistoryResponse(error='customer_id is required').model_dump_json()

    cache_key = customer_id.upper().strip()
    with _order_history_cache_lock:
        if cache_key in _order_history_cache:
            logger.info("Order history cache hit for: %s", cache_key)
            return _order_history_cache[cache_key]

    db = SessionLocal()
    try:
        customer_id = cache_key
        logger.info("Fetching order history for customer: %s", customer_id)

        customer_orders = db.query(Order).filter(Order.customer_id == customer_id).all()
        if not customer_orders:
            logger.info("No orders found for customer: %s", customer_id)
            return OrderHistoryResponse(error='No order found').model_dump_json()
            
        orders_dict = {}
        for order in customer_orders:
            key_name = order.order_id 
            
            items_summary = []
            for item in order.items:
                items_summary.append(f"{item.name} (x{item.quantity})")

            orders_dict[key_name] = {
                "customer_id": order.customer_id,
                "total_amount": float(order.total_amount) if order.total_amount is not None else 0.0,
                "currency": order.currency,
                "status": order.status,
                "order_date": order.order_date,
                "delivery_date": order.delivery_date,
                "payment_status": order.payment_status,
                "refund_status": order.refund_status,
                "items": items_summary
            }

        result = OrderHistoryResponse(orders=orders_dict).model_dump_json()
        with _order_history_cache_lock:
            _order_history_cache[cache_key] = result
        logger.info("Found %d orders for customer: %s", len(orders_dict), customer_id)
        return result
    except Exception as e:
        db.rollback()
        logger.error("Error fetching order history for %s: %s", customer_id, e)
        raise
    finally:
        db.close()