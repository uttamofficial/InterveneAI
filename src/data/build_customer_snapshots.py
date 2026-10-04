from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/customer_orders.parquet")
OUTPUT_PATH = Path("data/processed/customer_snapshots.parquet")

HISTORY_DAYS = 180
TARGET_DAYS = 60


def get_snapshot_dates(df):
    """Return valid monthly snapshot dates."""

    first_date = df["order_purchase_timestamp"].min()
    last_date = df["order_purchase_timestamp"].max()

    earliest_snapshot = first_date + pd.Timedelta(days=HISTORY_DAYS)
    latest_snapshot = last_date - pd.Timedelta(days=TARGET_DAYS)

    # First month-start that is actually >= earliest_snapshot
    first_month = earliest_snapshot.to_period("M").start_time

    if first_month < earliest_snapshot:
        first_month += pd.offsets.MonthBegin(1)

    # Last month-start that is <= latest_snapshot
    last_month = latest_snapshot.to_period("M").start_time

    snapshots = pd.date_range(
        start=first_month,
        end=last_month,
        freq="MS",
    )

    return snapshots


def build_customer_snapshots():
    print("Loading customer orders...")

    df = pd.read_parquet(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    df = df.sort_values("order_purchase_timestamp")

    snapshots = get_snapshot_dates(df)

    print(f"Number of snapshots: {len(snapshots)}")

    all_snapshots = []

    for snapshot_date in snapshots:

        history_start = snapshot_date - pd.Timedelta(days=HISTORY_DAYS)
        target_end = snapshot_date + pd.Timedelta(days=TARGET_DAYS)

        # Orders available before the snapshot
        history = df[
            (df["order_purchase_timestamp"] >= history_start)
            & (df["order_purchase_timestamp"] < snapshot_date)
        ]

        # Future orders after the snapshot
        future = df[
            (df["order_purchase_timestamp"] >= snapshot_date)
            & (df["order_purchase_timestamp"] < target_end)
        ]

        # Customers who already existed before the snapshot
        existing_customers = history["customer_unique_id"].unique()

        snapshot_df = pd.DataFrame(
            {
                "customer_unique_id": existing_customers
            }
        )

        snapshot_df["snapshot_date"] = snapshot_date

        # Customers who purchase during the next 60 days
        future_customers = set(
            future["customer_unique_id"]
        )

        snapshot_df["purchased_next_60_days"] = (
            snapshot_df["customer_unique_id"]
            .isin(future_customers)
            .astype("int8")
        )

        all_snapshots.append(snapshot_df)

        print(
            f"{snapshot_date.date()} | "
            f"customers={len(snapshot_df):,} | "
            f"positive={snapshot_df['purchased_next_60_days'].sum():,}"
        )

    snapshots_df = pd.concat(
        all_snapshots,
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshots_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:", OUTPUT_PATH)
    print("Rows:", len(snapshots_df))
    print(
        "Unique customers:",
        snapshots_df["customer_unique_id"].nunique(),
    )

    print("\nTarget distribution:")

    target_distribution = (
        snapshots_df["purchased_next_60_days"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(target_distribution)


if __name__ == "__main__":
    build_customer_snapshots()