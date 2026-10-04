from pathlib import Path

import pandas as pd


ORDERS_PATH = Path(
    "data/processed/customer_orders.parquet"
)

SNAPSHOTS_PATH = Path(
    "data/processed/customer_snapshots.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/customer_features_v1.parquet"
)

HISTORY_DAYS = 180


def build_rfm_features():

    print("Loading data...")

    orders = pd.read_parquet(ORDERS_PATH)
    snapshots = pd.read_parquet(SNAPSHOTS_PATH)

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    snapshots["snapshot_date"] = pd.to_datetime(
        snapshots["snapshot_date"]
    )

    feature_frames = []

    for snapshot_date in snapshots["snapshot_date"].unique():

        snapshot_date = pd.Timestamp(snapshot_date)

        history_start = (
            snapshot_date
            - pd.Timedelta(days=HISTORY_DAYS)
        )

        # Orders available in the historical window
        history = orders[
            (orders["order_purchase_timestamp"] >= history_start)
            & (orders["order_purchase_timestamp"] < snapshot_date)
        ].copy()

        # Calculate customer-level features
        features = (
            history.groupby("customer_unique_id")
            .agg(
                last_purchase_date=(
                    "order_purchase_timestamp",
                    "max",
                ),
                purchase_count_180d=(
                    "order_id",
                    "nunique",
                ),
            )
            .reset_index()
        )

        # Recency
        features["recency_days"] = (
            snapshot_date
            - features["last_purchase_date"]
        ).dt.total_seconds() / (60 * 60 * 24)

        features["snapshot_date"] = snapshot_date

        feature_frames.append(features)

        print(
            f"{snapshot_date.date()} | "
            f"customers={len(features):,}"
        )

    features_df = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    # Keep target from snapshot dataset
    final_df = snapshots.merge(
        features_df,
        on=[
            "customer_unique_id",
            "snapshot_date",
        ],
        how="left",
        validate="one_to_one",
    )

    # last_purchase_date is no longer needed as a model feature
    final_df = final_df.drop(
        columns=["last_purchase_date"]
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
    build_rfm_features()