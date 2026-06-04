import datetime as dt
from data.db import SessionLocal, Base, engine
from data.queries import Customer, Order, OrderItem
from sqlalchemy.exc import IntegrityError

def add_more_data():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    try:
        # Check if CUST005 exists
        c5 = db.query(Customer).filter_by(customer_id="CUST005").first()
        if not c5:
            c5 = Customer(
                customer_id="CUST005",
                name="John Doe",
                email="john.doe@example.com",
                phone="555-1234",
                address={"street": "456 Oak St", "city": "Springfield"}
            )
            db.add(c5)
            db.commit()

        # Generate some dynamic dates based on today
        today = dt.datetime.now().date()
        recent_delivery = (today - dt.timedelta(days=2)).strftime("%Y-%m-%d")
        old_delivery = (today - dt.timedelta(days=10)).strftime("%Y-%m-%d")

        # Orders for CUST005
        orders_to_add = [
            Order(
                order_id="ORD-005-1",
                customer_id="CUST005",
                total_amount=120.00,
                currency="INR",
                status="delivered",
                order_date=(today - dt.timedelta(days=5)).strftime("%Y-%m-%d"),
                delivery_date=recent_delivery,
                payment_status="paid",
                refund_status="none"
            ),
            Order(
                order_id="ORD-005-2",
                customer_id="CUST005",
                total_amount=600.00,  # >= 500, needs human approval
                currency="INR",
                status="delivered",
                order_date=(today - dt.timedelta(days=5)).strftime("%Y-%m-%d"),
                delivery_date=recent_delivery,
                payment_status="paid",
                refund_status="none"
            ),
            Order(
                order_id="ORD-005-3",
                customer_id="CUST005",
                total_amount=150.00,
                currency="INR",
                status="delivered",
                order_date=(today - dt.timedelta(days=15)).strftime("%Y-%m-%d"),
                delivery_date=old_delivery,  # > 7 days ago
                payment_status="paid",
                refund_status="none"
            ),
            Order(
                order_id="ORD-005-4",
                customer_id="CUST005",
                total_amount=200.00,
                currency="INR",
                status="shipped",
                order_date=(today - dt.timedelta(days=1)).strftime("%Y-%m-%d"),
                delivery_date=None,  # Not delivered yet
                payment_status="paid",
                refund_status="none"
            ),
            Order(
                order_id="ORD-005-5",
                customer_id="CUST005",
                total_amount=300.00,
                currency="INR",
                status="delivered",
                order_date=(today - dt.timedelta(days=6)).strftime("%Y-%m-%d"),
                delivery_date=recent_delivery,
                payment_status="paid",
                refund_status="refunded",
                refund_id="REF999888"
            )
        ]

        order_items_data = {
            "ORD-005-1": [
                OrderItem(product_id="PROD-101", name="Wireless Mouse", price=120.00, quantity=1)
            ],
            "ORD-005-2": [
                OrderItem(product_id="PROD-102", name="Mechanical Keyboard", price=400.00, quantity=1),
                OrderItem(product_id="PROD-103", name="USB Hub", price=200.00, quantity=1)
            ],
            "ORD-005-3": [
                OrderItem(product_id="PROD-104", name="Ergonomic Mousepad", price=150.00, quantity=1)
            ],
            "ORD-005-4": [
                OrderItem(product_id="PROD-105", name="Webcam", price=200.00, quantity=1)
            ],
            "ORD-005-5": [
                OrderItem(product_id="PROD-106", name="Bluetooth Speaker", price=300.00, quantity=1)
            ]
        }

        for order in orders_to_add:
            # Check if order already exists
            existing_order = db.query(Order).filter_by(order_id=order.order_id).first()
            if not existing_order:
                db.add(order)
            
            # Check if items exist
            existing_items = db.query(OrderItem).filter_by(order_id=order.order_id).all()
            if not existing_items:
                items = order_items_data.get(order.order_id, [])
                for item in items:
                    item.order_id = order.order_id
                    db.add(item)
                
        db.commit()
        print("More data added successfully for CUST005 (Orders & Items)!")

    except IntegrityError as e:
        db.rollback()
        print(f"Data might already exist. IntegrityError: {e}")
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    add_more_data()
