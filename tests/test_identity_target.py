"""Tests 1, 2, 5: identity integrity, snapshot uniqueness, target integrity."""

import pandas as pd

from conftest import sample_rows


def test_customer_ids_present(orders, snapshots, features_v3):
    for df in (orders, snapshots, features_v3):
        assert df["customer_unique_id"].notna().all()
        assert (df["customer_unique_id"] != "").all()


def test_customer_ids_traceable_to_orders(
    orders, snapshots, features_v3, experiment
):
    order_ids = set(orders["customer_unique_id"].unique())
    for df in (snapshots, features_v3, experiment):
        assert set(df["customer_unique_id"].unique()) <= order_ids


def test_snapshot_uniqueness(snapshots, features_v3, experiment):
    for df in (snapshots, features_v3, experiment):
        assert not df.duplicated(
            subset=["customer_unique_id", "snapshot_date"]
        ).any()


def test_target_binary_and_complete(snapshots, features_v3):
    for df in (snapshots, features_v3):
        col = df["purchased_next_60_days"]
        assert col.notna().all()
        assert set(col.unique()) <= {0, 1}


def test_target_total_positives(features_v3):
    assert int(features_v3["purchased_next_60_days"].sum()) == 2085


def test_target_preserved_v2_to_v3(features_v2, features_v3):
    merged = features_v2[
        ["customer_unique_id", "snapshot_date", "purchased_next_60_days"]
    ].merge(
        features_v3[
            ["customer_unique_id", "snapshot_date", "purchased_next_60_days"]
        ],
        on=["customer_unique_id", "snapshot_date"],
        suffixes=("_v2", "_v3"),
    )
    assert len(merged) == len(features_v2) == len(features_v3)
    assert (
        merged["purchased_next_60_days_v2"]
        == merged["purchased_next_60_days_v3"]
    ).all()


def test_target_spot_check_against_orders(orders, features_v3):
    sample = sample_rows(features_v3, n=300)
    for row in sample.itertuples():
        cust = orders[
            orders["customer_unique_id"] == row.customer_unique_id
        ]["order_purchase_timestamp"]
        future = cust[
            (cust >= row.snapshot_date)
            & (cust < row.snapshot_date + pd.Timedelta(days=60))
        ]
        assert int(len(future) > 0) == row.purchased_next_60_days
