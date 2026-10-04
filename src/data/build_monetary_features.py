from pathlib import Path

import pandas as pd


ORDERS_PATH = Path(
    "data/processed/customer_orders.parquet"
)

PAYMENTS_PATH = Path(
    "data/raw/olist_order_payments_dataset.csv"
)

SNAPSHOTS_PATH = Path(
    "data/processed/customer_features_v1.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/customer_features_v2.parquet"
)

HISTORY_DAYS = 180


def build_monetary_features():

    print("Loading delivered orders...")
    orders = pd.read_parquet(ORDERS_PATH)

    print("Loading payments...")
    payments = pd.read_csv(PAYMENTS_PATH)

    print("Orders:", len(orders))
    print("Payment rows:", len(payments))

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    payments["payment_value"] = pd.to_numeric(
        payments["payment_value"],
        errors="coerce",
    )

    # One order can have multiple payment rows.
    # Aggregate payment value at order level first.
    order_payments = (
        payments.groupby("order_id", as_index=False)
        .agg(
            order_value=(
                "payment_value",
                "sum",
            )
        )
    )

    print(
        "Unique orders with payment data:",
        order_payments["order_id"].nunique(),
    )

    # Add order value to delivered orders.
    orders = orders.merge(
        order_payments,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    print(
        "Missing order values:",
        orders["order_value"].isna().sum(),
    )

    snapshots = pd.read_parquet(
        SNAPSHOTS_PATH
    )

    snapshots["snapshot_date"] = pd.to_datetime(
        snapshots["snapshot_date"]
    )

    feature_frames = []

    for snapshot_date in snapshots[
        "snapshot_date"
    ].unique():

        snapshot_date = pd.Timestamp(
            snapshot_date
        )

        history_start = (
            snapshot_date
            - pd.Timedelta(days=HISTORY_DAYS)
        )

        history = orders[
            (orders["order_purchase_timestamp"] >= history_start)
            & (orders["order_purchase_timestamp"] < snapshot_date)
        ].copy()

        features = (
            history.groupby("customer_unique_id")
            .agg(
                total_spend_180d=(
                    "order_value",
                    "sum",
                ),
                avg_order_value_180d=(
                    "order_value",
                    "mean",
                ),
            )
            .reset_index()
        )

        features["snapshot_date"] = snapshot_date

        feature_frames.append(features)

        print(
            f"{snapshot_date.date()} | "
            f"customers={len(features):,}"
        )

    monetary_features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    final_df = snapshots.merge(
        monetary_features,
        on=[
            "customer_unique_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:", OUTPUT_PATH)

    print("\nShape:")
    print(final_df.shape)

    print("\nColumns:")
    print(final_df.columns.tolist())

    print("\nMissing values:")
    print(final_df.isna().sum())


if __name__ == "__main__":
    build_monetary_features()