from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/processed/customer_orders.parquet")

HISTORY_DAYS = 180
TARGET_DAYS = 60


def investigate_snapshot_window():
    df = pd.read_parquet(DATA_PATH)

    df["order_purchase_timestamp"] = pd.to_datetime(
        df["order_purchase_timestamp"]
    )

    first_date = df["order_purchase_timestamp"].min()
    last_date = df["order_purchase_timestamp"].max()

    print("First delivered order:", first_date)
    print("Last delivered order:", last_date)

    # Earliest date where 180 days of historical data exists
    earliest_snapshot = first_date + pd.Timedelta(days=HISTORY_DAYS)

    # Latest date where 60 days of future data exists
    latest_snapshot = last_date - pd.Timedelta(days=TARGET_DAYS)

    print("\nSnapshot window:")
    print("Earliest valid snapshot:", earliest_snapshot)
    print("Latest valid snapshot:", latest_snapshot)

    # First month-start snapshot that is >= earliest_snapshot
    earliest_month = earliest_snapshot.to_period("M").start_time

    if earliest_month < earliest_snapshot:
        earliest_month += pd.offsets.MonthBegin(1)

    # Last month-start snapshot that is <= latest_snapshot
    latest_month = latest_snapshot.to_period("M").start_time

    snapshots = pd.date_range(
        start=earliest_month,
        end=latest_month,
        freq="MS",
    )

    print("\nMonthly snapshot dates:")

    for date in snapshots:
        print(date.date())

    print("\nNumber of monthly snapshots:", len(snapshots))


if __name__ == "__main__":
    investigate_snapshot_window()