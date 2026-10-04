import pandas as pd

orders = pd.read_csv(
    "data/raw/olist_orders_dataset.csv"
)

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Create month column
orders["purchase_month"] = (
    orders["order_purchase_timestamp"]
    .dt.to_period("M")
)

print("=" * 70)
print("ORDERS PER MONTH")
print("=" * 70)

monthly_orders = (
    orders
    .groupby("purchase_month")
    .size()
)

print(monthly_orders.to_string())

print("\n" + "=" * 70)
print("DELIVERED ORDERS PER MONTH")
print("=" * 70)

delivered_orders = (
    orders[orders["order_status"] == "delivered"]
    .groupby("purchase_month")
    .size()
)

print(delivered_orders.to_string())

print("\n" + "=" * 70)
print("UNIQUE CUSTOMERS PER MONTH")
print("=" * 70)

monthly_customers = (
    orders
    .groupby("purchase_month")["customer_id"]
    .nunique()
)

print(monthly_customers.to_string())

print("\n" + "=" * 70)
print("DELIVERED CUSTOMERS PER MONTH")
print("=" * 70)

delivered_customers = (
    orders[orders["order_status"] == "delivered"]
    .groupby("purchase_month")["customer_id"]
    .nunique()
)

print(delivered_customers.to_string())
