"""Build Streamlit Cloud deployment assets (run once, locally).

Derives small, committable artifacts from EXISTING local outputs only:
- NO retraining, NO parameter changes, NO modified results.
- NO raw Olist data; excluded intermediates are never included.

Assets (dashboard/assets/):
- decision_snapshot.parquet: exact v3 slice at the dashboard decision date.
- test_scores.parquet: target + 4 model scores on the frozen test window,
  recomputed with the SAVED pipelines (bit-identical inputs).
- shap_background.parquet: exact 100-row background sample used by SHAP.
- recommendations.parquet: byte-copy of saved optimization recommendations.
- overview.json: Overview scalars recomputed from v3.
- intervention_params.json: byte-copy of the synthetic-experiment params.

Every asset is validated against the live local computation before the
script exits. Prints "Deployment asset parity: PASSED" only if all
checks hold.
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.explainability import (
    TEST_START,
    design_matrix,
    load_champion,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
V3_PATH = Path("data/processed/customer_features_v3.parquet")
RECS_PATH = Path(
    "data/synthetic/optimization_recommendations.parquet"
)
PARAMS_PATH = Path("data/synthetic/intervention_params.json")
COMP_PATH = Path("models/model_comparison.json")

DECISION_SNAPSHOT = pd.Timestamp("2018-06-01")
SHAP_SEED = 42
SHAP_N = 100


def build_assets():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    v3 = pd.read_parquet(V3_PATH)
    v3["snapshot_date"] = pd.to_datetime(v3["snapshot_date"])

    # 1. Decision snapshot: exact slice, all v3 columns.
    snap = v3[v3["snapshot_date"] == DECISION_SNAPSHOT].copy()
    snap_path = ASSETS_DIR / "decision_snapshot.parquet"
    snap.to_parquet(snap_path, index=False)
    print(f"decision_snapshot: {len(snap):,} rows")

    # 2. Test-window scores from SAVED pipelines (no retraining).
    with open(COMP_PATH) as f:
        comp = json.load(f)
    test = v3[
        v3["snapshot_date"] > pd.Timestamp(comp["splits"]["val"][1])
    ].copy()
    champion = joblib.load("models/champion_pipeline.pkl")
    features = champion["features"]
    frame = pd.DataFrame(
        {
            "customer_unique_id": test["customer_unique_id"].to_numpy(),
            "snapshot_date": test["snapshot_date"].to_numpy(),
            "y": test["purchased_next_60_days"].to_numpy().astype("int8"),
            "champion_logreg": champion["pipeline"]
            .predict_proba(test[features])[:, 1],
        }
    )
    for name in ("logreg", "rf", "hgb"):
        saved = joblib.load(f"models/candidates/{name}.pkl")
        key = name if name != "logreg" else "tuned_logreg"
        frame[key] = saved["pipeline"].predict_proba(test[features])[:, 1]
    scores_path = ASSETS_DIR / "test_scores.parquet"
    frame.to_parquet(scores_path, index=False)
    print(f"test_scores: {len(frame):,} rows")

    # 3. SHAP background: exact 100-row sample (seed 42, as in module).
    bundle = load_champion()
    tst = v3[v3["snapshot_date"] >= TEST_START]
    Xm = design_matrix(bundle, tst)
    rng = np.random.default_rng(SHAP_SEED)
    idx = rng.choice(len(Xm), size=SHAP_N, replace=False)
    bg = tst.iloc[idx][["customer_unique_id"] + bundle["features"]]
    bg_path = ASSETS_DIR / "shap_background.parquet"
    bg.reset_index(drop=True).to_parquet(bg_path, index=False)
    print(f"shap_background: {len(bg):,} rows")

    # 4-5. Byte-copies (no transformation possible).
    shutil.copyfile(
        RECS_PATH, ASSETS_DIR / "recommendations.parquet"
    )
    shutil.copyfile(
        PARAMS_PATH, ASSETS_DIR / "intervention_params.json"
    )
    print("recommendations + intervention_params: copied")

    # 6. Overview scalars recomputed from v3.
    overview = {
        "customers": int(v3["customer_unique_id"].nunique()),
        "observations": int(len(v3)),
        "purchase_rate": float(v3["purchased_next_60_days"].mean()),
        "n_snapshots": int(v3["snapshot_date"].nunique()),
        "decision_snapshot": str(DECISION_SNAPSHOT.date()),
    }
    with open(ASSETS_DIR / "overview.json", "w") as f:
        json.dump(overview, f, indent=2)
    print(f"overview: {overview}")

    for p in sorted(ASSETS_DIR.iterdir()):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")


def validate_parity():
    """Recompute everything live and compare. Raises on mismatch."""
    v3 = pd.read_parquet(V3_PATH)
    v3["snapshot_date"] = pd.to_datetime(v3["snapshot_date"])

    # Snapshot slice parity.
    snap = pd.read_parquet(ASSETS_DIR / "decision_snapshot.parquet")
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    expected = v3[v3["snapshot_date"] == DECISION_SNAPSHOT].reset_index(
        drop=True
    )
    assert list(snap.columns) == list(expected.columns), "snapshot columns"
    assert len(snap) == len(expected) == 38376, "snapshot row count"
    pd.testing.assert_frame_equal(snap, expected)
    banned = {
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "order_status",
        "payment_value",
    }
    assert not (banned & set(snap.columns)), "raw columns leaked"

    # Test scores parity (recompute with saved pipelines).
    with open(COMP_PATH) as f:
        comp = json.load(f)
    test = v3[
        v3["snapshot_date"] > pd.Timestamp(comp["splits"]["val"][1])
    ].reset_index(drop=True)
    assert len(test) == 114388, "test row count"
    saved = pd.read_parquet(ASSETS_DIR / "test_scores.parquet")
    assert list(saved.columns) == [
        "customer_unique_id",
        "snapshot_date",
        "y",
        "champion_logreg",
        "tuned_logreg",
        "rf",
        "hgb",
    ], "scores columns"
    assert (saved["customer_unique_id"].to_numpy() == test[
        "customer_unique_id"
    ].to_numpy()).all(), "scores IDs"
    assert (saved["y"].to_numpy() == test[
        "purchased_next_60_days"
    ].to_numpy().astype("int8")).all(), "scores target"
    champion = joblib.load("models/champion_pipeline.pkl")
    features = champion["features"]
    pipes = {"champion_logreg": champion["pipeline"]}
    for name in ("logreg", "rf", "hgb"):
        key = name if name != "logreg" else "tuned_logreg"
        pipes[key] = joblib.load(f"models/candidates/{name}.pkl")[
            "pipeline"
        ]
    for key, pipe in pipes.items():
        fresh = pipe.predict_proba(test[features])[:, 1]
        # Parquet float64 round-trip noise is ~1e-16; tolerance is strict.
        np.testing.assert_allclose(
            saved[key].to_numpy(), fresh, rtol=1e-12, atol=1e-15
        ), f"scores {key}"

    # SHAP background parity (same seed -> same rows -> same matrix).
    bundle = load_champion()
    tst = v3[v3["snapshot_date"] >= TEST_START].reset_index(drop=True)
    Xm = design_matrix(bundle, tst)
    rng = np.random.default_rng(SHAP_SEED)
    idx = rng.choice(len(Xm), size=SHAP_N, replace=False)
    bg = pd.read_parquet(ASSETS_DIR / "shap_background.parquet")
    assert list(bg.columns) == ["customer_unique_id"] + bundle[
        "features"
    ], "background columns"
    assert (
        bg["customer_unique_id"].to_numpy()
        == tst.iloc[idx]["customer_unique_id"].to_numpy()
    ).all(), "background IDs"
    np.testing.assert_allclose(
        design_matrix(bundle, bg).astype("float64"),
        Xm[idx].astype("float64"),
        rtol=1e-12,
        atol=1e-15,
    )

    # Recommendations + params byte parity.
    assert (
        pd.read_parquet(ASSETS_DIR / "recommendations.parquet")
        .reset_index(drop=True)
        .equals(pd.read_parquet(RECS_PATH).reset_index(drop=True))
    ), "recommendations"
    with open(ASSETS_DIR / "intervention_params.json") as f:
        assert json.load(f) == json.load(open(PARAMS_PATH)), "params"

    # Overview parity.
    with open(ASSETS_DIR / "overview.json") as f:
        ov = json.load(f)
    assert ov["customers"] == int(v3["customer_unique_id"].nunique())
    assert ov["observations"] == len(v3)
    assert ov["purchase_rate"] == float(
        v3["purchased_next_60_days"].mean()
    )
    assert ov["n_snapshots"] == 15

    print("\nDeployment asset parity: PASSED")


if __name__ == "__main__":
    build_assets()
    validate_parity()
