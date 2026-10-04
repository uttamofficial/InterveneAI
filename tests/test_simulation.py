"""Test 7: treatment simulation reproducibility and integrity."""

import pandas as pd

from conftest import ROOT


def _build(**kwargs):
    from src.causal.simulate_experiment import build_experiment

    return build_experiment(**kwargs)


def test_reproducible_same_seed():
    a = _build(mechanism="random", arm_probs=None, seed=42)
    b = _build(mechanism="random", arm_probs=None, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = _build(mechanism="random", arm_probs=None, seed=42)
    b = _build(mechanism="random", arm_probs=None, seed=43)
    assert not a["intervention_type"].equals(b["intervention_type"])


def test_saved_file_matches_default_run(experiment):
    rebuilt = _build(mechanism="random", arm_probs=None, seed=42)
    pd.testing.assert_frame_equal(rebuilt, experiment)


def test_arm_proportions_match_config(experiment):
    props = experiment["intervention_type"].value_counts(
        normalize=True
    ).sort_index()
    for arm, expected in ((0, 0.4), (1, 0.2), (2, 0.2), (3, 0.2)):
        assert abs(props[arm] - expected) < 0.01


def test_cost_mapping_and_binary_outcome(experiment):
    from src.causal.simulate_experiment import INTERVENTION_COSTS

    expected = experiment["intervention_type"].map(INTERVENTION_COSTS)
    assert (experiment["intervention_cost"] == expected).all()
    assert set(experiment["outcome"].unique()) <= {0, 1}
    assert experiment["outcome"].notna().all()


def test_real_target_absent(experiment):
    assert "purchased_next_60_days" not in experiment.columns


def test_treatment_flag_consistent(experiment):
    assert (
        (experiment["treatment"] == 1)
        == (experiment["intervention_type"] > 0)
    ).all()
