import pandas as pd

customers = pd.read_csv(
    "data/raw/olist_customers_dataset.csv"
)

orders = pd.read_csv(
    "data/raw/olist_orders_dataset.csv"
)

print("=" * 70)
print("CUSTOMER ID ANALYSIS")
print("=" * 70)

print(f"\nTotal customer rows: {len(customers):,}")

print(
    f"Unique customer_id: "
    f"{customers['customer_id'].nunique():,}"
)

print(
    f"Unique customer_unique_id: "
    f"{customers['customer_unique_id'].nunique():,}"
)

print("\nCustomer IDs appearing more than once:")

customer_id_counts = (
    customers["customer_id"]
    .value_counts()
)

print(
    (customer_id_counts > 1).sum()
)

print("\nCustomer unique IDs linked to multiple customer_ids:")

unique_to_customer_count = (
    customers
    .groupby("customer_unique_id")["customer_id"]
    .nunique()
)

multiple_customer_ids = (
    unique_to_customer_count > 1
).sum()

print(multiple_customer_ids)

print("\nDistribution of number of customer_ids per unique customer:")

print(
    unique_to_customer_count
    .value_counts()
    .sort_index()
)

print("\n" + "=" * 70)
print("ORDER → CUSTOMER RELATIONSHIP")
print("=" * 70)

print(
    f"\nTotal orders: "
    f"{len(orders):,}"
)

print(
    f"Unique order IDs: "
    f"{orders['order_id'].nunique():,}"
)

print(
    f"Unique customer IDs in orders: "
    f"{orders['customer_id'].nunique():,}"
)

print(
    f"Orders with customer IDs not found in customers table: "
    f"{(~orders['customer_id'].isin(customers['customer_id'])).sum():,}"
)
