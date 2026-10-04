"""Deployment-asset tests: dashboard runs without the full data store."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

ASSETS = ROOT / "dashboard" / "assets"
EXPECTED_FILES = {
    "decision_snapshot.parquet",
    "test_scores.parquet",
    "shap_background.parquet",
    "recommendations.parquet",
    "overview.json",
    "intervention_params.json",
}
RAW_LIKE_COLUMNS = {
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "order_status",
    "payment_value",
}


def test_all_deployment_assets_exist():
    actual = {p.name for p in ASSETS.iterdir() if p.is_file()}
    assert EXPECTED_FILES <= actual, EXPECTED_FILES - actual


def test_decision_snapshot_schema():
    snap = pd.read_parquet(ASSETS / "decision_snapshot.parquet")
    assert len(snap) == 38376
    for col in (
        ["customer_unique_id", "snapshot_date"]
        + [
            "recency_days",
            "purchase_count_30d",
            "purchase_count_90d",
            "purchase_count_180d",
            "total_spend_180d",
            "avg_order_value_180d",
            "days_since_previous_purchase",
            "avg_days_between_orders",
            "purchase_velocity_30d",
            "purchase_velocity_90d",
        ]
    ):
        assert col in snap.columns, col
    assert (
        pd.to_datetime(snap["snapshot_date"]).dt.date
        == pd.Timestamp("2018-06-01").date()
    ).all()
    assert not snap["customer_unique_id"].duplicated().any()


def test_test_scores_schema():
    scores = pd.read_parquet(ASSETS / "test_scores.parquet")
    assert list(scores.columns) == [
        "customer_unique_id",
        "snapshot_date",
        "y",
        "champion_logreg",
        "tuned_logreg",
        "rf",
        "hgb",
    ]
    assert len(scores) == 114388
    assert set(scores["y"].unique()) <= {0, 1}
    assert int(scores["y"].sum()) == 620


def test_no_unexpected_raw_data_in_assets():
    for name in (
        "decision_snapshot.parquet",
        "test_scores.parquet",
        "shap_background.parquet",
    ):
        cols = set(pd.read_parquet(ASSETS / name).columns)
        assert not (RAW_LIKE_COLUMNS & cols), (name, RAW_LIKE_COLUMNS & cols)
    bg = pd.read_parquet(ASSETS / "shap_background.parquet")
    assert len(bg) == 100


def test_assets_match_local_computation():
    """Spot parity: snapshot slice + one score column recomputed live."""
    v3 = pd.read_parquet(
        ROOT / "data" / "processed" / "customer_features_v3.parquet"
    )
    v3["snapshot_date"] = pd.to_datetime(v3["snapshot_date"])
    snap = pd.read_parquet(ASSETS / "decision_snapshot.parquet")
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    expected = v3[
        v3["snapshot_date"] == pd.Timestamp("2018-06-01")
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(snap, expected)

    champion = joblib.load(ROOT / "models" / "champion_pipeline.pkl")
    test = v3[v3["snapshot_date"] > pd.Timestamp("2018-03-01")].reset_index(
        drop=True
    )
    fresh = champion["pipeline"].predict_proba(test[champion["features"]])[
        :, 1
    ]
    saved = pd.read_parquet(ASSETS / "test_scores.parquet")
    np.testing.assert_allclose(
        saved["champion_logreg"].to_numpy(), fresh, rtol=1e-12, atol=1e-15
    )


def test_dashboard_has_no_full_store_dependency():
    src = (ROOT / "dashboard" / "app.py").read_text()
    assert "customer_features_v3" not in src
    assert "intervention_experiment" not in src
    assert "candidates" not in src


def test_budget_constraint_from_assets():
    from src.optimization.intervention_optimizer import (
        DECISION_SNAPSHOT,
        STRATEGIES,
        build_prediction_frame,
        load_arm_models,
    )

    snap = pd.read_parquet(ASSETS / "decision_snapshot.parquet")
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    _, best = build_prediction_frame(snap, load_arm_models())
    from src.optimization.intervention_optimizer import optimize

    for name, (rank_by, filt) in STRATEGIES.items():
        plan = optimize(best, 20000.0, rank_by, filt)
        assert (
            plan["expected_incremental_cost"].sum() <= 20000.0 + 1e-6
        ), name
    plan_c = optimize(best, 20000.0, "exp_profit", True)
    assert len(plan_c) == 6030
    assert abs(plan_c["expected_incremental_cost"].sum() - 20000.0) < 1e-6
