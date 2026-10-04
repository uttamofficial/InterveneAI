"""Tests 3, 4: temporal leakage and feature calculations."""

import numpy as np
import pandas as pd

from conftest import sample_rows

WINDOWS = (30, 90, 180)


def _prior_orders(orders, customer_id, snapshot_date):
    cust = orders[orders["customer_unique_id"] == customer_id][
        "order_purchase_timestamp"
    ]
    return cust[cust < snapshot_date].sort_values()


def test_no_future_orders_contribute(orders, features_v3):
    sample = sample_rows(features_v3, n=500)
    for row in sample.itertuples():
        prior = _prior_orders(
            orders, row.customer_unique_id, row.snapshot_date
        )
        assert (prior < row.snapshot_date).all()
        for window in WINDOWS:
            expected = int(
                (prior >= row.snapshot_date - pd.Timedelta(days=window)).sum()
            )
            assert expected == getattr(
                row, f"purchase_count_{window}d"
            ), (row.customer_unique_id, row.snapshot_date, window)


def test_gap_features_from_prior_orders_only(orders, features_v3):
    sample = sample_rows(
        features_v3[features_v3["days_since_previous_purchase"].notna()],
        n=200,
    )
    assert len(sample) > 0
    for row in sample.itertuples():
        prior = _prior_orders(
            orders, row.customer_unique_id, row.snapshot_date
        ).to_numpy()
        assert len(prior) >= 2
        gaps = np.diff(prior).astype("timedelta64[s]").astype(float)
        gaps = gaps / (60 * 60 * 24)
        assert np.isclose(
            row.days_since_previous_purchase, gaps[-1]
        )
        assert np.isclose(row.avg_days_between_orders, gaps.mean())


def test_single_prior_order_gaps_are_nan(orders, features_v3):
    sample = sample_rows(
        features_v3[features_v3["days_since_previous_purchase"].isna()],
        n=300,
    )
    for row in sample.itertuples():
        prior = _prior_orders(
            orders, row.customer_unique_id, row.snapshot_date
        )
        assert len(prior) == 1


def test_count_monotonicity_full_frame(features_v3):
    assert (
        features_v3["purchase_count_30d"]
        <= features_v3["purchase_count_90d"]
    ).all()
    assert (
        features_v3["purchase_count_90d"]
        <= features_v3["purchase_count_180d"]
    ).all()


def test_velocity_identity_full_frame(features_v3):
    for window in (30, 90):
        count = features_v3[f"purchase_count_{window}d"].to_numpy(
            dtype="float64"
        )
        vel = features_v3[f"purchase_velocity_{window}d"].to_numpy()
        assert np.allclose(vel * window, count, rtol=0, atol=1e-9)


def test_gap_nan_masks_consistent(features_v3):
    a = features_v3["days_since_previous_purchase"].isna()
    b = features_v3["avg_days_between_orders"].isna()
    assert (a == b).all()
    assert (features_v3.loc[~a, ["days_since_previous_purchase",
                                 "avg_days_between_orders"]] >= 0).all().all()


def test_v2_features_preserved(features_v2, features_v3):
    cols = [
        "purchase_count_180d",
        "recency_days",
        "total_spend_180d",
        "avg_order_value_180d",
    ]
    merged = features_v2[
        ["customer_unique_id", "snapshot_date"] + cols
    ].merge(
        features_v3[["customer_unique_id", "snapshot_date"] + cols],
        on=["customer_unique_id", "snapshot_date"],
        suffixes=("_v2", "_v3"),
    )
    assert len(merged) == len(features_v2)
    for col in cols:
        assert np.allclose(
            merged[f"{col}_v2"], merged[f"{col}_v3"], equal_nan=True
        ), col
