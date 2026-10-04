"""Phase 4 — Baseline Predictive Model.

Predicts purchased_next_60_days from customer_features_v3.parquet
with a simple Logistic Regression baseline.

Temporal discipline:
- Chronological split on snapshot_date (never a random split):
    train: snapshots 2017-04-01 .. 2017-12-01
    val:   snapshots 2018-01-01 .. 2018-03-01
    test:  snapshots 2018-04-01 .. 2018-06-01
- Imputer/scaler are fit on train only; validation is used only to
  pick the F1 operating threshold; test is touched once for final
  reporting. Features themselves use only pre-snapshot orders
  (see Phase 2 leakage audit).
- Note: rows are stacked (customer, snapshot) combinations, so some
  customers appear in several periods. The split prevents *temporal*
  leakage (no future-period information trains the model); it does
  not isolate customers. Customer overlap is reported, not hidden.

Imbalance discipline:
- class_weight="balanced" (no resampling, no synthetic rows).
- Accuracy is never used for selection or reporting as a headline;
  primary metric is PR-AUC, supported by ROC-AUC, precision/recall/F1,
  confusion matrices, calibration (Brier + curve) and precision@k/lift.

Saves:
- models/baseline_pipeline.pkl (imputer + scaler + logreg, one object)
- models/baseline_metrics.json (split ranges, all metrics, threshold,
  top-k table, calibration points, coefficients, seed/versions)
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(
    "data/processed/customer_features_v3.parquet"
)
MODEL_DIR = Path("models")
PIPELINE_PATH = MODEL_DIR / "baseline_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"

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

# Last snapshot of train / validation. Test is everything later.
TRAIN_END = pd.Timestamp("2017-12-01")
VAL_END = pd.Timestamp("2018-03-01")

SEED = 42
TOP_K_FRACS = (0.01, 0.05, 0.10)


def chronological_split(df):
    """Split strictly by snapshot month. No shuffling, no randomness."""
    train = df[df["snapshot_date"] <= TRAIN_END]
    val = df[
        (df["snapshot_date"] > TRAIN_END)
        & (df["snapshot_date"] <= VAL_END)
    ]
    test = df[df["snapshot_date"] > VAL_END]

    assert len(train) and len(val) and len(test), (
        "Empty chronological split"
    )
    assert (
        train["snapshot_date"].max()
        < val["snapshot_date"].min()
        <= val["snapshot_date"].max()
        < test["snapshot_date"].min()
    ), "Split periods overlap — temporal leakage"
    assert (
        set(train["snapshot_date"])
        .isdisjoint(val["snapshot_date"])
        and set(val["snapshot_date"]).isdisjoint(
            test["snapshot_date"]
        )
    ), "Snapshot months shared across splits"

    for name, part in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        rate = part[TARGET].mean()
        print(
            f"{name:>5}: {part['snapshot_date'].min().date()} .. "
            f"{part['snapshot_date'].max().date()} | "
            f"rows={len(part):,} | "
            f"positives={int(part[TARGET].sum()):,} | "
            f"rate={rate:.4%}"
        )

    overlap = set(test["customer_unique_id"]) & set(
        train["customer_unique_id"]
    )
    print(
        f"Test customers also present in train: "
        f"{len(overlap):,} / "
        f"{test['customer_unique_id'].nunique():,} "
        f"(panel overlap disclosed; periods are disjoint)"
    )

    return train, val, test


def build_pipeline():
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def score_metrics(y_true, y_score):
    y_true = np.asarray(y_true)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(
            average_precision_score(y_true, y_score)
        ),
        "brier": float(brier_score_loss(y_true, y_score)),
    }


def threshold_metrics(y_true, y_score, threshold):
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def pick_f1_threshold(y_true, y_score):
    """Best F1 over a fixed grid — evaluated on validation only."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    best = (0.5, -1.0)
    for thr in np.linspace(0.0, 1.0, 201):
        f1 = f1_score(y_true, y_score >= thr, zero_division=0)
        if f1 > best[1]:
            best = (float(thr), float(f1))
    return best[0]


def top_k_metrics(y_true, y_score, fracs=TOP_K_FRACS):
    """Precision/recall/lift among the top-k scored rows."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="stable")
    ranked = y_true[order]
    prevalence = y_true.mean()
    rows = []
    for frac in fracs:
        k = max(1, int(frac * len(y_true)))
        tp = int(ranked[:k].sum())
        precision = tp / k
        rows.append(
            {
                "frac": float(frac),
                "k": int(k),
                "precision": float(precision),
                "recall": float(tp / y_true.sum()),
                "lift": float(precision / prevalence),
            }
        )
    return rows


def train_baseline():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    print(f"Rows: {len(df):,}")

    print("\nChronological split (snapshot months, no shuffling):")
    train, val, test = chronological_split(df)

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_val = val[FEATURES]
    y_val = val[TARGET]
    X_test = test[FEATURES]
    y_test = test[TARGET]

    print("\nFitting pipeline (imputer/scaler fit on train only)...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    s_val = pipe.predict_proba(X_val)[:, 1]
    s_test = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "target": TARGET,
        "features": FEATURES,
        "seed": SEED,
        "train_end": str(TRAIN_END.date()),
        "val_end": str(VAL_END.date()),
        "splits": {
            name: {
                "from": str(part["snapshot_date"].min().date()),
                "to": str(part["snapshot_date"].max().date()),
                "rows": int(len(part)),
                "positives": int(part[TARGET].sum()),
                "rate": float(part[TARGET].mean()),
            }
            for name, part in (
                ("train", train),
                ("val", val),
                ("test", test),
            )
        },
        "model": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "C": 1.0,
        },
        "val": score_metrics(y_val, s_val),
        "test": score_metrics(y_test, s_test),
        # Random-ranking reference points (computed, not a model).
        "dummy": {
            "roc_auc": 0.5,
            "pr_auc_random": float(y_test.mean()),
        },
    }

    f1_thr = pick_f1_threshold(y_val, s_val)
    metrics["operating_threshold_from_val"] = {
        "threshold": f1_thr,
        "val_f1_at_threshold": float(
            f1_score(y_val, s_val >= f1_thr)
        ),
    }
    metrics["test_at_0.5"] = threshold_metrics(
        y_test, s_test, 0.5
    )
    metrics["test_at_f1_threshold"] = threshold_metrics(
        y_test, s_test, f1_thr
    )
    metrics["test_top_k"] = top_k_metrics(y_test, s_test)

    prob_true, prob_pred = calibration_curve(
        y_test, s_test, n_bins=10, strategy="quantile"
    )
    metrics["test_calibration_quantile10"] = {
        "prob_true": [float(v) for v in prob_true],
        "prob_pred": [float(v) for v in prob_pred],
    }

    clf = pipe.named_steps["clf"]
    metrics["coefficients"] = {
        name: float(coef)
        for name, coef in zip(FEATURES, clf.coef_[0])
    }
    metrics["intercept"] = float(clf.intercept_[0])

    print("\nScore metrics (primary: PR-AUC):")
    for split in ("val", "test"):
        m = metrics[split]
        print(
            f"  {split}: PR-AUC={m['pr_auc']:.4f} | "
            f"ROC-AUC={m['roc_auc']:.4f} | "
            f"Brier={m['brier']:.5f} "
            f"(random PR-AUC={m['prevalence']:.4f})"
        )

    print(f"\nF1 threshold from val: {f1_thr:.3f}")
    for key in ("test_at_0.5", "test_at_f1_threshold"):
        m = metrics[key]
        print(
            f"  {key} (t={m['threshold']:.3f}): "
            f"P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} | "
            f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}"
        )

    print("\nTest precision@k / lift:")
    for row in metrics["test_top_k"]:
        print(
            f"  top {row['frac']:.0%} (k={row['k']:,}): "
            f"P={row['precision']:.4f} R={row['recall']:.4f} "
            f"lift={row['lift']:.2f}x"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipe,
            "features": FEATURES,
            "target": TARGET,
            "train_end": str(TRAIN_END.date()),
            "val_end": str(VAL_END.date()),
            "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        PIPELINE_PATH,
    )
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {PIPELINE_PATH}")
    print(f"Saved: {METRICS_PATH}")

    # Reload check: saved pipeline must reproduce test PR-AUC.
    reloaded = joblib.load(PIPELINE_PATH)["pipeline"]
    s_reload = reloaded.predict_proba(X_test)[:, 1]
    assert np.allclose(s_reload, s_test), (
        "Reloaded pipeline scores differ"
    )
    print("Reload check: saved pipeline reproduces test scores.")

    return metrics


if __name__ == "__main__":
    train_baseline()
