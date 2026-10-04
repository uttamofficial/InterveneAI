from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/customer_features_v2.parquet"
)


def analyze_recency():

    df = pd.read_parquet(DATA_PATH)

    bins = [0, 30, 60, 90, 120, 150, 180]

    labels = [
        "0-30",
        "31-60",
        "61-90",
        "91-120",
        "121-150",
        "151-180",
    ]

    df["recency_bucket"] = pd.cut(
        df["recency_days"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    summary = (
        df.groupby(
            "recency_bucket",
            observed=False,
        )
        .agg(
            customers=(
                "customer_unique_id",
                "count",
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

    print("\nFuture purchase rate by recency:")
    print(
        summary[
            [
                "recency_bucket",
                "customers",
                "positive_customers",
                "positive_rate_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    analyze_recency()