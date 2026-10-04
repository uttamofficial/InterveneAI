from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/customer_features_v2.parquet"
)


def analyze_monetary():

    df = pd.read_parquet(DATA_PATH)

    df["spend_bucket"] = pd.qcut(
        df["total_spend_180d"],
        q=5,
        labels=[
            "Q1 - Lowest",
            "Q2",
            "Q3",
            "Q4",
            "Q5 - Highest",
        ],
        duplicates="drop",
    )

    summary = (
        df.groupby(
            "spend_bucket",
            observed=False,
        )
        .agg(
            customers=(
                "customer_unique_id",
                "count",
            ),
            median_spend=(
                "total_spend_180d",
                "median",
            ),
            positive_customers=(
                "purchased_next_60_days",
                "sum",
            ),
            positive_rate=(
                "purchased_next_60_days",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["positive_rate_pct"] = (
        summary["positive_rate"] * 100
    )

    print("\nFuture purchase rate by spending bucket:\n")

    print(
        summary[
            [
                "spend_bucket",
                "customers",
                "median_spend",
                "positive_customers",
                "positive_rate_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    analyze_monetary()