from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    customers = pd.read_csv(
        RAW_DIR / "olist_customers_dataset.csv"
    )

    orders = pd.read_csv(
        RAW_DIR / "olist_orders_dataset.csv"
    )

    return customers, orders


def build_customer_orders():
    customers, orders = load_data()

    # Convert purchase timestamp
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    # Attach the real unique customer identifier
    customer_orders = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # Keep only completed purchases for our first analysis
    customer_orders = customer_orders[
        customer_orders["order_status"] == "delivered"
    ].copy()

    # Select the columns we currently need
    customer_orders = customer_orders[
        [
            "order_id",
            "customer_id",
            "customer_unique_id",
            "order_purchase_timestamp",
            "order_status",
        ]
    ]

    # Sort chronologically
    customer_orders = customer_orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp",
        ]
    )

    output_path = (
        PROCESSED_DIR / "customer_orders.parquet"
    )

    customer_orders.to_parquet(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")
    print(
        f"Rows: {len(customer_orders):,}"
    )
    print(
        "Unique customers:",
        customer_orders["customer_unique_id"].nunique(),
    )
    print(
        "Unique orders:",
        customer_orders["order_id"].nunique(),
    )

    print("\nDate range:")
    print(
        customer_orders[
            "order_purchase_timestamp"
        ].min()
    )
    print(
        customer_orders[
            "order_purchase_timestamp"
        ].max()
    )


if __name__ == "__main__":
    build_customer_orders()
