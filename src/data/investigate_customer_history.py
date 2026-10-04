from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/processed/customer_orders.parquet")


def investigate_customer_history():
    df = pd.read_parquet(DATA_PATH)

    print("Shape:", df.shape)

    orders_per_customer = (
        df.groupby("customer_unique_id")
        .size()
        .sort_values(ascending=False)
    )

    print("\nOrders per customer:")
    print(orders_per_customer.describe())

    print("\nCustomers with exactly 1 order:")
    print((orders_per_customer == 1).sum())

    print("\nCustomers with more than 1 order:")
    print((orders_per_customer > 1).sum())

    print("\nCustomers with 2+ orders:")
    print(
        orders_per_customer[orders_per_customer > 1]
        .head(20)
    )


if __name__ == "__main__":
    investigate_customer_history()