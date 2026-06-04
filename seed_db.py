from data.db import SessionLocal, Base, engine
from data.queries import Customer, Order, OrderItem

def seed_database():
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(Customer).first():
            print("Database already populated!")
            return

        print("Populating database with sample data...")
        
        # 1. Create a Customer
        c1 = Customer(
            customer_id="CUST123",
            name="Alice Smith",
            email="alice@example.com",
            phone="555-0100",
            address={"street": "123 Main St", "city": "Wonderland"}
        )
        db.add(c1)

        # 2. Create Orders for that Customer
        o1 = Order(
            order_id="ORD-001",
            customer_id="CUST123",
            total_amount=120.50,
            currency="USD",
            status="delivered",
            order_date="2023-10-01",
            delivery_date="2023-10-05",
            payment_status="paid",
            refund_status="none"
        )
        
        o2 = Order(
            order_id="ORD-002",
            customer_id="CUST123",
            total_amount=65.00,
            currency="USD",
            status="shipped",
            order_date="2023-10-15",
            delivery_date=None,
            payment_status="paid",
            refund_status="none"
        )
        
        db.add_all([o1, o2])
        db.commit()
        print("Database populated successfully! You can test with customer_id: CUST123 or order_id: ORD-001")
        
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
