"""Phase 5 — Shared evaluation utilities.

Pure functions used by src/models/train_models.py and
notebooks/05_model_comparison.ipynb. No data loading here; the
notebook imports these to independently verify saved metrics.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TOP_K_FRACS = (0.01, 0.05, 0.10)


def score_metrics(y_true, y_score):
    """Threshold-free ranking metrics + Brier."""
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
    y_true = np.asarray(y_true)
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


def pick_f1_threshold(y_true, y_score, n_grid=201):
    """Best F1 over a fixed grid. Caller decides which split."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.0, 1.0, n_grid):
        f1 = f1_score(y_true, y_score >= thr, zero_division=0)
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), float(f1)
    return best_thr


def top_k_metrics(y_true, y_score, fracs=TOP_K_FRACS):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score, kind="stable")
    ranked = y_true[order]
    prevalence = float(y_true.mean())
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


def calibration_data(y_true, y_score, n_bins=10):
    prob_true, prob_pred = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="quantile"
    )
    return {
        "prob_true": [float(v) for v in prob_true],
        "prob_pred": [float(v) for v in prob_pred],
    }


def evaluate_pipeline(pipe, X, y):
    """Full evaluation bundle for one fitted pipeline."""
    scores = pipe.predict_proba(X)[:, 1]
    return {
        "scores_summary": score_metrics(y, scores),
        "at_0_5": threshold_metrics(y, scores, 0.5),
        "top_k": top_k_metrics(y, scores),
        "calibration": calibration_data(y, scores),
    }


def comparison_frame(rows):
    """Compact test-set comparison table (expects dicts)."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "model": r["model"],
                "val PR-AUC": round(r["val_pr_auc"], 4),
                "test PR-AUC": round(r["test_pr_auc"], 4),
                "test ROC-AUC": round(r["test_roc_auc"], 4),
                "test Brier": round(r["test_brier"], 5),
                "P@5%": round(r["p_at_5"], 4),
                "lift@5%": round(r["lift_at_5"], 2),
                "F1 (val thr)": round(r["f1"], 4),
            }
            for r in rows
        ]
    ).set_index("model")


if __name__ == "__main__":
    import json
    from pathlib import Path

    path = Path("models/model_comparison.json")
    saved = json.loads(path.read_text())
    print(
        comparison_frame(saved["test_comparison"]).to_string()
    )
    print("\nchampion:", saved["champion"])
