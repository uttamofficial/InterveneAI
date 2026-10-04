from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/customer_features_v2.parquet"
)


def profile_features():

    df = pd.read_parquet(DATA_PATH)

    print("Shape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nFeature statistics:")
    print(
        df[
            [
                "recency_days",
                "purchase_count_180d",
                "total_spend_180d",
                "avg_order_value_180d",
            ]
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print("\nUnique purchase counts:")
    print(
        df["purchase_count_180d"]
        .value_counts()
        .sort_index()
        .head(20)
    )

    print("\nTarget by purchase frequency:")

    frequency_target = (
        df.groupby("purchase_count_180d")
        ["purchased_next_60_days"]
        .agg(
            customers="count",
            positive_rate="mean",
        )
        .reset_index()
    )

    frequency_target["positive_rate_pct"] = (
        frequency_target["positive_rate"] * 100
    )

    print(
        frequency_target[
            [
                "purchase_count_180d",
                "customers",
                "positive_rate_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    profile_features()