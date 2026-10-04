"""Phase 5 — Advanced Predictive Modeling.

Tuned candidates (Logistic Regression, Random Forest,
HistGradientBoosting) under the frozen Phase-4 chronological
framework. XGBoost/LightGBM were checked and are NOT installed in
this environment; per the no-blind-install rule they are skipped —
HistGradientBoosting covers the gradient-boosting family reliably
via scikit-learn.

Protocol (test stays untouched until final evaluation):
1. Chronological split: train 2017-04..2017-12, val 2018-01..2018-03,
   test 2018-04..2018-06. Preprocessing fit on train only.
2. Tune each family on TRAIN, select configs by VAL PR-AUC only.
   The Phase-4 baseline is carried as an untuned reference row.
3. Single final pass over TEST for every candidate. Champion is the
   best VAL PR-AUC among tuned families; test confirms it.
4. Never accuracy-driven: selection and comparison use PR-AUC
   (primary), ROC-AUC, precision/recall/F1, top-k precision/lift
   and calibration.

Saves:
- models/candidates/<family>.pkl (best tuned pipeline per family)
- models/champion_pipeline.pkl (winner: preprocessing + model)
- models/model_comparison.json (grids, val/test metrics, thresholds,
  top-k, calibration, champion + reasons, seed/versions)
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluate_models import (
    pick_f1_threshold,
    score_metrics,
    threshold_metrics,
    top_k_metrics,
    calibration_data,
)


DATA_PATH = Path(
    "data/processed/customer_features_v3.parquet"
)
MODEL_DIR = Path("models")
CANDIDATE_DIR = MODEL_DIR / "candidates"
CHAMPION_PATH = MODEL_DIR / "champion_pipeline.pkl"
COMPARISON_PATH = MODEL_DIR / "model_comparison.json"
BASELINE_PATH = MODEL_DIR / "baseline_pipeline.pkl"

TARGET = "purchased_next_60_days"

FEATURES = [
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

TRAIN_END = pd.Timestamp("2017-12-01")
VAL_END = pd.Timestamp("2018-03-01")

SEED = 42

# Small, pre-registered grids. No peeking at test, ever.
GRIDS = {
    "logreg": [
        {"C": 0.05},
        {"C": 0.2},
        {"C": 1.0},
        {"C": 5.0},
    ],
    "rf": [
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 100,
        },
        {
            "n_estimators": 500,
            "max_depth": 12,
            "min_samples_leaf": 50,
        },
    ],
    "hgb": [
        {
            "learning_rate": lr,
            "max_leaf_nodes": leaves,
            "min_samples_leaf": msl,
        }
        for lr in (0.05, 0.1)
        for leaves in (15, 31)
        for msl in (50, 200)
    ],
}


def chronological_split(df):
    train = df[df["snapshot_date"] <= TRAIN_END]
    val = df[
        (df["snapshot_date"] > TRAIN_END)
        & (df["snapshot_date"] <= VAL_END)
    ]
    test = df[df["snapshot_date"] > VAL_END]
    assert (
        train["snapshot_date"].max()
        < val["snapshot_date"].min()
        <= val["snapshot_date"].max()
        < test["snapshot_date"].min()
    ), "Split periods overlap"
    return train, val, test


def build_pipeline(family, params):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if family == "logreg":
        steps.append(("scale", StandardScaler()))
        steps.append(
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                    **params,
                ),
            )
        )
    elif family == "rf":
        steps.append(
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                    **params,
                ),
            )
        )
    elif family == "hgb":
        steps.append(
            (
                "clf",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=300,
                    random_state=SEED,
                    **params,
                ),
            )
        )
    else:
        raise ValueError(family)
    return Pipeline(steps)


def tune_family(family, X_train, y_train, X_val, y_val):
    """Fit every grid config on train; rank by val PR-AUC."""
    results = []
    for params in GRIDS[family]:
        pipe = build_pipeline(family, params)
        pipe.fit(X_train, y_train)
        s_val = pipe.predict_proba(X_val)[:, 1]
        m = score_metrics(y_val, s_val)
        results.append((params, pipe, m))
        print(
            f"  {params} -> "
            f"val PR-AUC={m['pr_auc']:.4f} "
            f"ROC-AUC={m['roc_auc']:.4f}"
        )
    results.sort(key=lambda r: r[2]["pr_auc"], reverse=True)
    best_params, best_pipe, best_m = results[0]
    print(
        f"  BEST {family}: {best_params} "
        f"(val PR-AUC={best_m['pr_auc']:.4f})"
    )
    return best_params, best_pipe, results


def final_evaluation(name, pipe, X_test, y_test, X_val, y_val):
    """Single test pass. Threshold comes from validation only."""
    s_val = pipe.predict_proba(X_val)[:, 1]
    s_test = pipe.predict_proba(X_test)[:, 1]
    thr = pick_f1_threshold(y_val, s_val)
    return {
        "val": score_metrics(y_val, s_val),
        "test": score_metrics(y_test, s_test),
        "f1_threshold_from_val": float(thr),
        "test_at_0_5": threshold_metrics(y_test, s_test, 0.5),
        "test_at_f1_threshold": threshold_metrics(
            y_test, s_test, thr
        ),
        "test_top_k": top_k_metrics(y_test, s_test),
        "test_calibration": calibration_data(y_test, s_test),
    }


def train_models():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    train, val, test = chronological_split(df)
    print(
        f"train={len(train):,} val={len(val):,} "
        f"test={len(test):,} (test untouched until final eval)"
    )

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    tuned = {}
    for family in ("logreg", "rf", "hgb"):
        print(f"\nTuning {family} ({len(GRIDS[family])} configs):")
        best_params, best_pipe, full_grid = tune_family(
            family, X_train, y_train, X_val, y_val
        )
        tuned[family] = (best_params, best_pipe, full_grid)
        joblib.dump(
            {
                "pipeline": best_pipe,
                "features": FEATURES,
                "target": TARGET,
                "family": family,
                "params": best_params,
                "seed": SEED,
                "sklearn_version": sklearn.__version__,
                "trained_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
            CANDIDATE_DIR / f"{family}.pkl",
        )

    # Champion = best VAL PR-AUC among tuned families.
    champion = max(
        tuned,
        key=lambda f: tuned[f][2][0][2]["pr_auc"],
    )
    print(f"\nChampion on validation PR-AUC: {champion}")

    print("\nFinal single-pass test evaluation:")
    comparison = {}
    for family, (params, pipe, _) in tuned.items():
        ev = final_evaluation(
            family, pipe, X_test, y_test, X_val, y_val
        )
        ev["family"] = family
        ev["params"] = params
        comparison[family] = ev
        print(
            f"  {family}: test PR-AUC={ev['test']['pr_auc']:.4f} "
            f"ROC-AUC={ev['test']['roc_auc']:.4f} "
            f"Brier={ev['test']['brier']:.5f}"
        )

    # Phase-4 baseline as an untuned reference row (loaded, not refit).
    base_pipe = joblib.load(BASELINE_PATH)["pipeline"]
    ev = final_evaluation(
        "baseline_logreg", base_pipe, X_test, y_test, X_val, y_val
    )
    ev["family"] = "baseline_logreg"
    ev["params"] = {"C": 1.0, "note": "Phase-4 reference"}
    comparison["baseline_logreg"] = ev

    rows = []
    for name, ev in comparison.items():
        top5 = next(
            r for r in ev["test_top_k"] if r["frac"] == 0.05
        )
        rows.append(
            {
                "model": name,
                "val_pr_auc": ev["val"]["pr_auc"],
                "test_pr_auc": ev["test"]["pr_auc"],
                "test_roc_auc": ev["test"]["roc_auc"],
                "test_brier": ev["test"]["brier"],
                "p_at_5": top5["precision"],
                "lift_at_5": top5["lift"],
                "f1": ev["test_at_f1_threshold"]["f1"],
            }
        )

    payload = {
        "target": TARGET,
        "features": FEATURES,
        "seed": SEED,
        "splits": {
            "train": [
                str(train["snapshot_date"].min().date()),
                str(train["snapshot_date"].max().date()),
            ],
            "val": [
                str(val["snapshot_date"].min().date()),
                str(val["snapshot_date"].max().date()),
            ],
            "test": [
                str(test["snapshot_date"].min().date()),
                str(test["snapshot_date"].max().date()),
            ],
        },
        "grids": {
            fam: [dict(p) for p in GRIDS[fam]] for fam in GRIDS
        },
        "val_selection": {
            fam: {
                "best_params": tuned[fam][0],
                "val_pr_auc": tuned[fam][2][0][2]["pr_auc"],
            }
            for fam in tuned
        },
        "test_comparison": rows,
        "details": {
            name: ev for name, ev in comparison.items()
        },
        "champion": champion,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(COMPARISON_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    joblib.dump(
        {
            "pipeline": tuned[champion][1],
            "features": FEATURES,
            "target": TARGET,
            "family": champion,
            "params": tuned[champion][0],
            "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        CHAMPION_PATH,
    )
    print(f"\nSaved: {COMPARISON_PATH}")
    print(f"Saved: {CHAMPION_PATH}")

    # Reload check on the champion.
    reloaded = joblib.load(CHAMPION_PATH)["pipeline"]
    s_a = tuned[champion][1].predict_proba(X_test)[:, 1]
    s_b = reloaded.predict_proba(X_test)[:, 1]
    assert np.allclose(s_a, s_b), "Champion reload mismatch"
    print("Reload check: champion reproduces test scores.")

    return payload


if __name__ == "__main__":
    train_models()
