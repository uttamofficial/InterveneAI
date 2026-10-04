from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/customer_orders.parquet")


def investigate_purchase_gaps():
    df = pd.read_parquet(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    df = df.sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    )

    # Difference between consecutive purchases
    df["previous_purchase"] = (
        df.groupby("customer_unique_id")[
            "order_purchase_timestamp"
        ]
        .shift(1)
    )

    df["days_since_previous_purchase"] = (
        df["order_purchase_timestamp"]
        - df["previous_purchase"]
    ).dt.total_seconds() / (60 * 60 * 24)

    # Keep only actual repeat purchases
    repeat_purchases = df[
        df["days_since_previous_purchase"].notna()
    ].copy()

    print("Total delivered orders:", len(df))
    print("Repeat purchases:", len(repeat_purchases))

    print("\nPurchase gap statistics (days):")
    print(
        repeat_purchases[
            "days_since_previous_purchase"
        ].describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print("\nRepeat purchases within 30 days:")
    print(
        (
            repeat_purchases["days_since_previous_purchase"] <= 30
        ).mean()
        * 100
    )

    print("\nRepeat purchases within 60 days:")
    print(
        (
            repeat_purchases["days_since_previous_purchase"] <= 60
        ).mean()
        * 100
    )

    print("\nRepeat purchases within 90 days:")
    print(
        (
            repeat_purchases["days_since_previous_purchase"] <= 90
        ).mean()
        * 100
    )

    print("\nRepeat purchases within 180 days:")
    print(
        (
            repeat_purchases["days_since_previous_purchase"] <= 180
        ).mean()
        * 100
    )


if __name__ == "__main__":
    investigate_purchase_gaps()