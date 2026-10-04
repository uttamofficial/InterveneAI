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


def test_simulate_roi_with_frame_never_reads_v3(monkeypatch):
    """Block v3 access: the asset-backed simulator path must not need it."""
    import pandas as pd

    real_read = pd.read_parquet

    def guarded(path, *args, **kwargs):
        if "customer_features_v3" in str(
            path
        ) or "intervention_experiment" in str(path):
            raise FileNotFoundError(f"blocked by test: {path}")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)

    from src.optimization.roi_simulator import simulate_roi

    snap = pd.read_parquet(
        ASSETS / "decision_snapshot.parquet"
    ).head(2000)
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    result = simulate_roi(
        budget=500.0, strategy="C_profit", decision_frame=snap
    )
    assert result["customers_targeted"] > 0
    assert result["campaign_cost"] <= 500.0 + 1e-6
    assert result["expected_incremental_profit"] > 0  # C filters positive


def test_simulate_roi_frame_matches_disk_path():
    from src.optimization.roi_simulator import simulate_roi

    snap = pd.read_parquet(ASSETS / "decision_snapshot.parquet")
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    via_frame = simulate_roi(
        budget=20000.0, strategy="C_profit", decision_frame=snap
    )
    assert via_frame["customers_targeted"] == 6030
    assert abs(via_frame["expected_incremental_profit"] - 86308.33) < 0.01
    assert abs(via_frame["oracle_profit"] - (-11703.72)) < 0.01


def test_deployment_bundles_exist_in_repo():
    import joblib

    for fname in (
        "uplift_tlearner.pkl",
        "uplift_tlearner_free_delivery.pkl",
        "uplift_tlearner_loyalty_points.pkl",
        "champion_pipeline.pkl",
    ):
        path = ROOT / "models" / fname
        assert path.exists(), fname
        assert path.stat().st_size > 0, fname
    bundle = joblib.load(ROOT / "models" / "uplift_tlearner.pkl")
    assert bundle["arm"] == 1


def test_load_arm_models_never_touches_frame(monkeypatch):
    import src.optimization.intervention_optimizer as opt

    def forbidden(*args, **kwargs):
        raise AssertionError("load_uplift_frame must not be called")

    monkeypatch.setattr(opt, "load_uplift_frame", forbidden)
    models = opt.load_arm_models(allow_retrain=False)
    assert set(models) == {1, 2, 3}


def test_missing_bundle_raises_clear_error(tmp_path):
    import shutil

    import src.optimization.intervention_optimizer as opt

    shutil.copy(
        ROOT / "models" / "uplift_tlearner.pkl",
        tmp_path / "uplift_tlearner.pkl",
    )
    diag = opt.diagnose_bundles(tmp_path)
    assert {r["file"].split("/")[-1]: r["exists"] for r in diag} == {
        "uplift_tlearner.pkl": True,
        "uplift_tlearner_free_delivery.pkl": False,
        "uplift_tlearner_loyalty_points.pkl": False,
        "champion_pipeline.pkl": False,
    }
    try:
        opt.load_arm_models(model_dir=tmp_path, allow_retrain=False)
    except RuntimeError as e:
        assert str(e) == (
            "Missing uplift deployment bundle: "
            "uplift_tlearner_free_delivery.pkl "
            "(retraining disabled on this path)"
        )
    else:
        raise AssertionError("RuntimeError not raised for missing bundle")
