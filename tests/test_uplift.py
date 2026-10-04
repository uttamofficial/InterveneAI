"""Test 8: uplift calculation integrity."""

import json

import joblib
import numpy as np
import pandas as pd

from conftest import ROOT, sample_rows


def _test_frame():
    from src.causal.uplift_model import FEATURES, load_uplift_frame

    df = load_uplift_frame()
    test = df[df["snapshot_date"] >= pd.Timestamp("2018-04-01")]
    sub = test[test["intervention_type"].isin((0, 1))].copy()
    return sub, FEATURES


def test_uplift_is_treated_minus_control():
    from src.causal.uplift_model import predict_uplift

    bundle = joblib.load(ROOT / "models" / "uplift_tlearner.pkl")
    sub, FEATURES = _test_frame()
    probe = sample_rows(sub, n=200)
    u = predict_uplift(
        bundle["treated_pipeline"],
        bundle["control_pipeline"],
        probe[FEATURES],
    )
    pt = bundle["treated_pipeline"].predict_proba(probe[FEATURES])[:, 1]
    pc = bundle["control_pipeline"].predict_proba(probe[FEATURES])[:, 1]
    assert np.allclose(u, pt - pc)


def test_saved_auuc_matches_recompute():
    from src.causal.uplift_model import auuc, qini_curve

    with open(ROOT / "models" / "uplift_metrics.json") as f:
        saved = json.load(f)
    sub, _ = _test_frame()
    bundle = joblib.load(ROOT / "models" / "uplift_tlearner.pkl")
    from src.causal.uplift_model import predict_uplift

    y = sub["outcome"].to_numpy()
    t = (sub["intervention_type"] == 1).astype(int).to_numpy()
    u = predict_uplift(
        bundle["treated_pipeline"],
        bundle["control_pipeline"],
        sub[bundle["features"]],
    )
    xs, qm = qini_curve(y, t, u)
    assert abs(auuc(xs, qm) - saved["arms"]["1"]["auuc_model"]) < 1e-9


def test_oracle_beats_random_all_arms():
    with open(ROOT / "models" / "uplift_metrics.json") as f:
        saved = json.load(f)
    for arm in ("1", "2", "3"):
        a = saved["arms"][arm]
        assert a["auuc_oracle"] > a["auuc_random"], arm


def test_qini_coefficient_consistent():
    with open(ROOT / "models" / "uplift_metrics.json") as f:
        saved = json.load(f)
    for arm in ("1", "2", "3"):
        a = saved["arms"][arm]
        expected = (a["auuc_model"] - a["auuc_random"]) / (
            a["auuc_oracle"] - a["auuc_random"]
        )
        assert abs(a["qini_coefficient"] - expected) < 1e-9, arm


def test_segments_cover_all_labels():
    with open(ROOT / "models" / "uplift_metrics.json") as f:
        saved = json.load(f)
    for arm in ("1", "2", "3"):
        labels = {s["segment"] for s in saved["arms"][arm]["segments"]}
        assert labels == {"negative", "low", "medium", "high"}, arm
