"""Phase 2 — Behavioral Feature Engineering.

Builds historical purchase-behavior features for every
(customer_unique_id, snapshot_date) row in
data/processed/customer_features_v2.parquet and writes
data/processed/customer_features_v3.parquet.

New features (all derived ONLY from orders strictly before
the snapshot date — order_purchase_timestamp < snapshot_date):

- purchase_count_30d / purchase_count_90d:
    orders in [snapshot_date - N days, snapshot_date).
- purchase_count_180d:
    already present in v2; recomputed here with the same
    window logic and asserted equal (preservation check).
- days_since_previous_purchase:
    gap in days between the two most recent orders strictly
    before the snapshot (order-level definition adapted from
    src/data/investigate_purchase_gaps.py). NaN when the
    customer has fewer than 2 prior orders — never imputed.
- avg_days_between_orders:
    mean of all consecutive purchase gaps over the customer's
    full history strictly before the snapshot. NaN when fewer
    than 2 prior orders — never imputed.
- purchase_velocity_30d / purchase_velocity_90d:
    purchase_count_Nd / N (orders per day).

Leakage prevention:
- Every contributing order satisfies
    order_purchase_timestamp < snapshot_date (strict).
- Window lower bounds use >= (snapshot - N days), so an order
    stamped exactly at snapshot_date falls outside history and
    can only ever count toward the future target window.
- No target column (purchased_next_60_days) or future order is
    touched during feature computation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ORDERS_PATH = Path(
    "data/processed/customer_orders.parquet"
)

FEATURES_PATH = Path(
    "data/processed/customer_features_v2.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/customer_features_v3.parquet"
)

COUNT_WINDOWS = (30, 90, 180)

SECONDS_PER_DAY = 60 * 60 * 24

# Rows spot-checked order-by-order against raw orders.
LEAKAGE_AUDIT_SAMPLE = 2000
LEAKAGE_AUDIT_SEED = 42


def load_data():
    orders = pd.read_parquet(ORDERS_PATH)
    features = pd.read_parquet(FEATURES_PATH)

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )
    features["snapshot_date"] = pd.to_datetime(
        features["snapshot_date"]
    )

    return orders, features


def add_order_gaps(orders):
    """Attach each order's gap to the customer's previous order.

    gap_days is NaN for a customer's first-ever order.
    Computed once globally (no snapshot involved), then filtered
    per snapshot with the strict < snapshot_date rule.
    """
    orders = orders.sort_values(
        ["customer_unique_id", "order_purchase_timestamp"]
    ).copy()

    orders["previous_purchase_timestamp"] = orders.groupby(
        "customer_unique_id"
    )["order_purchase_timestamp"].shift(1)

    orders["gap_days"] = (
        orders["order_purchase_timestamp"]
        - orders["previous_purchase_timestamp"]
    ).dt.total_seconds() / SECONDS_PER_DAY

    return orders


def build_snapshot_features(snapshot_date, history):
    """Features for one snapshot from orders strictly before it."""
    rows = []

    for window in COUNT_WINDOWS:
        window_start = snapshot_date - pd.Timedelta(
            days=window
        )
        in_window = history[
            history["order_purchase_timestamp"]
            >= window_start
        ]
        counts = (
            in_window.groupby("customer_unique_id")
            .agg(count=("order_id", "nunique"))
            .reset_index()
            .rename(
                columns={
                    "count": f"purchase_count_{window}d"
                }
            )
        )
        rows.append(counts)

    counts_df = rows[0]
    for extra in rows[1:]:
        counts_df = counts_df.merge(
            extra,
            on="customer_unique_id",
            how="outer",
            validate="one_to_one",
        )

    # Gap features use the full history strictly before the
    # snapshot (orders are already filtered to < snapshot_date).
    # `history` is time-sorted, so `last` is the most recent gap.
    gaps_df = (
        history.groupby("customer_unique_id")
        .agg(
            days_since_previous_purchase=(
                "gap_days",
                "last",
            ),
            avg_days_between_orders=(
                "gap_days",
                "mean",
            ),
        )
        .reset_index()
    )

    snapshot_df = counts_df.merge(
        gaps_df,
        on="customer_unique_id",
        how="outer",
        validate="one_to_one",
    )
    snapshot_df["snapshot_date"] = snapshot_date

    return snapshot_df


def build_behavioral_features():
    print("Loading data...")
    orders, features = load_data()
    print(f"Orders: {len(orders):,}")
    print(f"Feature rows: {len(features):,}")

    orders = add_order_gaps(orders)

    feature_frames = []

    for snapshot_date in sorted(
        features["snapshot_date"].unique()
    ):
        snapshot_date = pd.Timestamp(snapshot_date)

        # STRICT temporal boundary: only orders purchased
        # strictly before the snapshot may contribute.
        history = orders[
            orders["order_purchase_timestamp"]
            < snapshot_date
        ]

        snapshot_features = build_snapshot_features(
            snapshot_date, history
        )
        feature_frames.append(snapshot_features)

        print(
            f"{snapshot_date.date()} | "
            f"prior orders={len(history):,} | "
            f"customers={len(snapshot_features):,}"
        )

    new_features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    final_df = features.merge(
        new_features,
        on=["customer_unique_id", "snapshot_date"],
        how="left",
        validate="one_to_one",
    )

    # A v2 row always has >= 1 order in its 180d window (snapshot
    # construction guarantees it), so the recomputed 180d count
    # must be present for every row.
    assert final_df["purchase_count_180d_y"].notna().all(), (
        "Recomputed purchase_count_180d is missing for some rows"
    )

    # Preservation check: recomputation must reproduce v2 exactly.
    assert (
        final_df["purchase_count_180d_x"]
        == final_df["purchase_count_180d_y"]
    ).all(), "purchase_count_180d mismatch with v2 — aborting"

    final_df = final_df.drop(
        columns=["purchase_count_180d_y"]
    ).rename(
        columns={
            "purchase_count_180d_x": "purchase_count_180d"
        }
    )

    # Zero orders in a sub-window is a fact, not a fabrication:
    # fill only counts (gap NaNs mean < 2 prior orders and stay).
    for window in (30, 90):
        col = f"purchase_count_{window}d"
        final_df[col] = (
            final_df[col].fillna(0).astype("int64")
        )
        final_df[f"purchase_velocity_{window}d"] = (
            final_df[col] / float(window)
        )

    final_df["purchase_count_180d"] = final_df[
        "purchase_count_180d"
    ].astype("int64")

    column_order = [
        "customer_unique_id",
        "snapshot_date",
        "purchased_next_60_days",
        "purchase_count_180d",
        "recency_days",
        "total_spend_180d",
        "avg_order_value_180d",
        "purchase_count_30d",
        "purchase_count_90d",
        "days_since_previous_purchase",
        "avg_days_between_orders",
        "purchase_velocity_30d",
        "purchase_velocity_90d",
    ]
    final_df = final_df[column_order]

    validate_features(final_df, orders, features)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    final_df.to_parquet(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)
    print("Shape:", final_df.shape)

    return final_df


def validate_features(df, orders, v2):
    """Fail loudly on leakage or inconsistency; report the rest."""
    print("\n--- Validation ---")

    # 1. No duplicate (customer, snapshot) rows.
    n_dupes = df.duplicated(
        subset=["customer_unique_id", "snapshot_date"]
    ).sum()
    print(f"Duplicate customer+snapshot rows: {n_dupes}")
    assert n_dupes == 0, "Duplicate rows found"

    # 2. No missing customer IDs, and every ID exists in orders.
    n_missing_ids = (
        df["customer_unique_id"].isna().sum()
        + (df["customer_unique_id"] == "").sum()
    )
    print(f"Missing customer IDs: {n_missing_ids}")
    assert n_missing_ids == 0, "Missing customer IDs found"
    unknown = set(df["customer_unique_id"].unique()) - set(
        orders["customer_unique_id"].unique()
    )
    print(f"Customer IDs absent from orders: {len(unknown)}")
    assert not unknown, "Unknown customer IDs found"

    # 3. Valid dates everywhere.
    assert df["snapshot_date"].notna().all(), (
        "NaT snapshot_date found"
    )
    assert orders["order_purchase_timestamp"].notna().all(), (
        "NaT order_purchase_timestamp found"
    )
    print("Invalid dates: 0 (no NaT in snapshot/order dates)")

    # 4. Row/column preservation from v2.
    assert len(df) == len(v2), "Row count changed vs v2"
    for col in v2.columns:
        pd.testing.assert_series_equal(
            df[col].reset_index(drop=True),
            v2[col].reset_index(drop=True),
            check_names=False,
            obj=f"preserved column {col!r}",
        )
    print(
        f"Preservation: OK "
        f"({len(df):,} rows, {len(v2.columns)} v2 columns unchanged)"
    )

    # 5. Nested-window monotonicity on every row.
    monotonic = (
        (df["purchase_count_30d"] <= df["purchase_count_90d"])
        & (
            df["purchase_count_90d"]
            <= df["purchase_count_180d"]
        )
    )
    n_bad = (~monotonic).sum()
    print(f"Monotonicity violations (30<=90<=180): {n_bad}")
    assert n_bad == 0, "Count monotonicity violated"

    # 6. Non-negative integer counts.
    for window in COUNT_WINDOWS:
        col = f"purchase_count_{window}d"
        assert (df[col] >= 0).all(), f"Negative {col}"
        assert pd.api.types.is_integer_dtype(df[col]), (
            f"{col} is not integer dtype"
        )
    print("Counts: all non-negative integers")

    # 7. Velocity consistency: velocity * window == count.
    for window in (30, 90):
        count_col = f"purchase_count_{window}d"
        vel_col = f"purchase_velocity_{window}d"
        assert (df[vel_col] >= 0).all(), (
            f"Negative {vel_col}"
        )
        assert np.allclose(
            df[vel_col] * window,
            df[count_col].to_numpy(dtype="float64"),
            rtol=0,
            atol=1e-9,
        ), f"{vel_col} inconsistent with {count_col}"
    print("Velocities: velocity_Nd * N == count_Nd for all rows")

    # 8. Gap-column coherence: both NaN together (fewer than 2
    #    prior orders), non-negative when present.
    gap_cols = [
        "days_since_previous_purchase",
        "avg_days_between_orders",
    ]
    masks = [df[c].isna() for c in gap_cols]
    assert (masks[0] == masks[1]).all(), (
        "Gap columns have inconsistent NaN masks"
    )
    assert (df[gap_cols].dropna() >= 0).all().all(), (
        "Negative purchase gap found"
    )
    print(
        f"Gaps: NaN masks consistent "
        f"({masks[0].sum():,} single-prior-order rows left as NaN)"
    )

    # 9. Future-order leakage audit: recompute every new feature
    #    order-by-order on a deterministic sample, using ONLY
    #    orders with timestamp strictly before the snapshot.
    rng = np.random.default_rng(LEAKAGE_AUDIT_SEED)
    sample_idx = rng.choice(
        len(df),
        size=min(LEAKAGE_AUDIT_SAMPLE, len(df)),
        replace=False,
    )
    sample = df.iloc[sample_idx]
    checked = 0
    for row in sample.itertuples():
        cust_orders = orders[
            orders["customer_unique_id"]
            == row.customer_unique_id
        ]
        prior = cust_orders[
            cust_orders["order_purchase_timestamp"]
            < row.snapshot_date
        ]
        # The audit itself must never see the future:
        assert not (
            prior["order_purchase_timestamp"]
            >= row.snapshot_date
        ).any(), "Leakage in audit filter"

        for window in (30, 90, 180):
            expected = (
                prior["order_purchase_timestamp"]
                >= row.snapshot_date
                - pd.Timedelta(days=window)
            ).sum()
            assert expected == getattr(
                row, f"purchase_count_{window}d"
            ), (
                f"Leakage/count mismatch for "
                f"{row.customer_unique_id} @ {row.snapshot_date}"
            )

        gaps = (
            prior.sort_values("order_purchase_timestamp")[
                "order_purchase_timestamp"
            ]
            .diff()
            .dt.total_seconds()
            / SECONDS_PER_DAY
        ).dropna()
        expected_last = gaps.iloc[-1] if len(gaps) else np.nan
        expected_mean = gaps.mean() if len(gaps) else np.nan
        for col, expected in (
            ("days_since_previous_purchase", expected_last),
            ("avg_days_between_orders", expected_mean),
        ):
            actual = getattr(row, col)
            if np.isnan(expected):
                assert np.isnan(actual), (
                    f"Expected NaN {col} for "
                    f"{row.customer_unique_id} @ {row.snapshot_date}"
                )
            else:
                assert np.isclose(actual, expected), (
                    f"Gap mismatch {col} for "
                    f"{row.customer_unique_id} @ {row.snapshot_date}"
                )
        checked += 1
    print(
        f"Leakage audit: {checked:,} sampled rows recomputed "
        f"from raw orders with strict < snapshot filter — all match"
    )

    # 10. Missing values report (gap NaNs are legitimate).
    print("\nMissing values:")
    print(df.isna().sum())

    # 11. Summary statistics for the new features.
    print("\nSummary statistics (new features):")
    print(
        df[
            [
                "purchase_count_30d",
                "purchase_count_90d",
                "purchase_count_180d",
                "days_since_previous_purchase",
                "avg_days_between_orders",
                "purchase_velocity_30d",
                "purchase_velocity_90d",
            ]
        ].describe()
    )

    print("\n--- All validation checks passed ---")


if __name__ == "__main__":
    build_behavioral_features()
