from sqlalchemy import Column, String, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .db import Base

class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    phone = Column(String)
    address = Column(JSON)  # Storing the nested address dict as JSON for simplicity

    # Relationship to Order
    orders = relationship("Order", back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, ForeignKey("customers.customer_id"))
    total_amount = Column(Float)
    currency = Column(String)
    status = Column(String)
    order_date = Column(String)
    delivery_date = Column(String, nullable=True)
    payment_status = Column(String)
    refund_status = Column(String)
    refund_id = Column(String, nullable=True)
    ticket_id = Column(String, nullable=True)

    # Relationships
    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "orderitems"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(String, ForeignKey("orders.order_id"))
    product_id = Column(String)
    name = Column(String)
    price = Column(Float)
    quantity = Column(Integer)

    # Relationship to Order
    order = relationship("Order", back_populates="items")
