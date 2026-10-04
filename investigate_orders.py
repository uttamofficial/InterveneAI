import pandas as pd

orders = pd.read_csv(
    "data/raw/olist_orders_dataset.csv"
)

# Convert timestamp columns to datetime
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col])

print("=" * 70)
print("ORDER STATUS")
print("=" * 70)

print(
    orders["order_status"]
    .value_counts()
)

print("\n" + "=" * 70)
print("PURCHASE DATE RANGE")
print("=" * 70)

print(
    "First purchase:",
    orders["order_purchase_timestamp"].min()
)

print(
    "Last purchase:",
    orders["order_purchase_timestamp"].max()
)

print("\n" + "=" * 70)
print("ORDER STATUS PERCENTAGE")
print("=" * 70)

status_percentage = (
    orders["order_status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(status_percentage)

print("\n" + "=" * 70)
print("MISSING VALUES AFTER DATETIME CONVERSION")
print("=" * 70)

print(
    orders[date_columns]
    .isna()
    .sum()
)
