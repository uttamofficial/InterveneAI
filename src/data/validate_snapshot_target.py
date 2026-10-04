from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/customer_snapshots.parquet")


def validate_target():
    df = pd.read_parquet(DATA_PATH)

    summary = (
        df.groupby("snapshot_date")
        .agg(
            customers=(
                "customer_unique_id",
                "nunique",
            ),
            positive_customers=(
                "purchased_next_60_days",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["positive_rate_pct"] = (
        summary["positive_customers"]
        / summary["customers"]
        * 100
    )

    print("\nTarget by snapshot:\n")
    print(summary.to_string(index=False))

    print("\nOverall:")
    print("Total rows:", len(df))
    print(
        "Positive rows:",
        df["purchased_next_60_days"].sum(),
    )
    print(
        "Positive rate:",
        f"{df['purchased_next_60_days'].mean() * 100:.4f}%",
    )


if __name__ == "__main__":
    validate_target()