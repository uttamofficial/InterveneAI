"""Phase 10 — Explainability for the champion model (SHAP).

The champion is Logistic Regression (linear), so SHAP's LinearExplainer
is exactly compatible: exact Shapley values for the log-odds output,
no sampling approximation. Explanations describe the model's
associational ranking of purchase probability — not causal drivers.

Provides:
- transformed design matrix (median-impute + scale, fit states from
  the saved pipeline — never refit here),
- global importance: mean |SHAP| per feature (+ direction),
- individual explanations: base value, per-feature contributions and
  the resulting log-odds for any row,
- dependence data: feature value vs SHAP value for any feature.

Saves: models/shap_global_importance.json
"""

from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import shap


CHAMPION_PATH = Path("models/champion_pipeline.pkl")
DATA_PATH = Path(
    "data/processed/customer_features_v3.parquet"
)
OUTPUT_PATH = Path("models/shap_global_importance.json")

TEST_START = pd.Timestamp("2018-04-01")
BACKGROUND_N = 100
GLOBAL_N = 2000
SEED = 42


def load_champion():
    bundle = joblib.load(CHAMPION_PATH)
    assert bundle["family"] == "logreg", (
        "LinearExplainer requires the linear champion"
    )
    return bundle


def design_matrix(bundle, df):
    """Apply the SAVED imputer/scaler (no refitting)."""
    pipe = bundle["pipeline"]
    X = pipe.named_steps["impute"].transform(df[bundle["features"]])
    return pipe.named_steps["scale"].transform(X)


def make_explainer(bundle, X_background):
    clf = bundle["pipeline"].named_steps["clf"]
    masker = shap.maskers.Independent(
        X_background, max_samples=BACKGROUND_N
    )
    return shap.LinearExplainer(clf, masker)


def global_importance(bundle, X_sample):
    """Mean |SHAP| per feature on a fixed sample + direction."""
    rng = np.random.default_rng(SEED)
    idx = rng.choice(
        len(X_sample),
        size=min(GLOBAL_N, len(X_sample)),
        replace=False,
    )
    Xs = X_sample[idx]
    background = X_sample[
        rng.choice(len(X_sample), size=BACKGROUND_N, replace=False)
    ]
    explainer = make_explainer(bundle, background)
    values = np.asarray(explainer.shap_values(Xs))
    names = bundle["features"]
    rows = []
    for j, name in enumerate(names):
        col = values[:, j]
        rows.append(
            {
                "feature": name,
                "mean_abs_shap": float(np.mean(np.abs(col))),
                "mean_shap": float(np.mean(col)),
            }
        )
    rows.sort(key=lambda r: r["mean_abs_shap"], reverse=True)
    return {
        "rows": rows,
        "expected_value": float(explainer.expected_value),
        "n": int(len(Xs)),
    }


def individual_explanation(bundle, explainer, x_row):
    """Base value, contributions and log-odds for one row."""
    values = np.asarray(
        explainer.shap_values(x_row.reshape(1, -1))
    )[0]
    total = float(explainer.expected_value + values.sum())
    return {
        "expected_value": float(explainer.expected_value),
        "log_odds": total,
        "probability": float(1.0 / (1.0 + np.exp(-total))),
        "contributions": [
            {"feature": name, "shap_value": float(v)}
            for name, v in zip(bundle["features"], values)
        ],
    }


def main():
    bundle = load_champion()
    df = pd.read_parquet(DATA_PATH)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    test = df[df["snapshot_date"] >= TEST_START]
    X = design_matrix(bundle, test)
    print(f"Test rows for SHAP: {len(test):,}")

    result = global_importance(bundle, X)
    print("\nGlobal importance (mean |SHAP|, log-odds units):")
    for row in result["rows"]:
        print(
            f"  {row['feature']:>32}: "
            f"{row['mean_abs_shap']:.4f} "
            f"(mean {row['mean_shap']:+.4f})"
        )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    # Reproducibility: identical sample -> identical values.
    again = global_importance(bundle, X)
    assert [r["mean_abs_shap"] for r in again["rows"]] == [
        r["mean_abs_shap"] for r in result["rows"]
    ], "SHAP run not reproducible"
    print("Reproducibility check: repeated run identical.")


if __name__ == "__main__":
    main()
