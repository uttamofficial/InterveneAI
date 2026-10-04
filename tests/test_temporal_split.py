"""Test 6: train/val/test temporal ordering (frozen Phase-4 cuts)."""

import json

import pandas as pd

from conftest import ROOT


def _cuts():
    from src.models.train_baseline import TRAIN_END, VAL_END

    return TRAIN_END, VAL_END


def test_split_months_disjoint_and_ordered(features_v3):
    train_end, val_end = _cuts()
    train = features_v3[features_v3["snapshot_date"] <= train_end]
    val = features_v3[
        (features_v3["snapshot_date"] > train_end)
        & (features_v3["snapshot_date"] <= val_end)
    ]
    test = features_v3[features_v3["snapshot_date"] > val_end]
    assert len(train) and len(val) and len(test)
    assert train["snapshot_date"].max() < val["snapshot_date"].min()
    assert val["snapshot_date"].max() < test["snapshot_date"].min()
    assert set(train["snapshot_date"]).isdisjoint(val["snapshot_date"])
    assert set(val["snapshot_date"]).isdisjoint(test["snapshot_date"])


def test_split_counts_match_saved_metrics(features_v3):
    with open(ROOT / "models" / "baseline_metrics.json") as f:
        saved = json.load(f)
    train_end, val_end = _cuts()
    parts = {
        "train": features_v3[features_v3["snapshot_date"] <= train_end],
        "val": features_v3[
            (features_v3["snapshot_date"] > train_end)
            & (features_v3["snapshot_date"] <= val_end)
        ],
        "test": features_v3[features_v3["snapshot_date"] > val_end],
    }
    assert saved["train_end"] == str(train_end.date())
    assert saved["val_end"] == str(val_end.date())
    for name, part in parts.items():
        assert saved["splits"][name]["rows"] == len(part)
        assert saved["splits"][name]["positives"] == int(
            part["purchased_next_60_days"].sum()
        )


def test_model_comparison_uses_same_splits():
    with open(ROOT / "models" / "model_comparison.json") as f:
        comp = json.load(f)
    assert comp["splits"]["train"][1] == "2017-12-01"
    assert comp["splits"]["val"] == ["2018-01-01", "2018-03-01"]
    assert comp["splits"]["test"] == ["2018-04-01", "2018-06-01"]
