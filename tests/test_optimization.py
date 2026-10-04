"""Tests 9, 10: budget constraint and optimization output schema."""

import json

import numpy as np
import pandas as pd

from conftest import ROOT

SPEC_COLUMNS = [
    "customer_unique_id",
    "recommended_intervention",
    "expected_incremental_revenue",
    "expected_incremental_cost",
    "expected_incremental_profit",
]
BUDGETS = ("5000.0", "20000.0", "50000.0", "150000.0")
STRATEGIES = ("A_purchase_prob", "B_uplift", "C_profit")


def _recs():
    return pd.read_parquet(
        ROOT / "data" / "synthetic" / "optimization_recommendations.parquet"
    )


def _summary():
    with open(
        ROOT / "data" / "synthetic" / "optimization_summary.json"
    ) as f:
        return json.load(f)


def test_output_schema():
    recs = _recs()
    for col in SPEC_COLUMNS:
        assert col in recs.columns, col
    assert recs["customer_unique_id"].notna().all()
    assert set(recs["recommended_intervention"].unique()) <= {1, 2, 3}
    assert (recs["expected_incremental_cost"] > 0).all()


def test_profit_identity():
    recs = _recs()
    assert np.allclose(
        recs["expected_incremental_profit"],
        recs["expected_incremental_revenue"]
        - recs["expected_incremental_cost"],
    )


def test_costs_match_assumed_mapping():
    from src.optimization.intervention_optimizer import COSTS

    recs = _recs()
    expected = recs["recommended_intervention"].map(COSTS)
    assert np.allclose(
        recs["expected_incremental_cost"], expected
    )


def test_budget_respected_all_scenarios():
    summary = _summary()
    for budget in BUDGETS:
        for strategy in STRATEGIES:
            s = summary["budgets"][budget][strategy]
            assert s["spend"] <= float(budget) + 1e-6, (budget, strategy)


def test_summary_matches_saved_plans_primary_budget():
    recs = _recs()
    summary = _summary()["budgets"]["20000.0"]
    for strategy in STRATEGIES:
        plan = recs[recs["strategy"] == strategy]
        s = summary[strategy]
        assert len(plan) == s["n_treated"], strategy
        assert abs(plan["expected_incremental_cost"].sum() - s["spend"]) < 1e-6
        assert abs(
            plan["expected_incremental_profit"].sum()
            - s["model_expected_profit"]
        ) < 1e-6


def test_strategy_c_only_profit_positive():
    from src.optimization.intervention_optimizer import (
        DECISION_SNAPSHOT,
        build_prediction_frame,
        load_arm_models,
    )
    from src.causal.uplift_model import load_uplift_frame

    df = load_uplift_frame()
    snap = df[df["snapshot_date"] == DECISION_SNAPSHOT]
    _, best = build_prediction_frame(snap, load_arm_models())
    # C filters to model-positive rows; A/B do not.
    assert (best["exp_profit"] > 0).any()
    assert (best["exp_profit"] <= 0).any()
