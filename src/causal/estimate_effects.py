"""Phase 7 — Causal Effect Estimation (synthetic experiment only).

No causal-inference package is installed here (dowhy/econml/causalml
all unavailable), and none is installed blindly. All estimators below
are transparent implementations on scipy/statsmodels/scikit-learn.

Estimands (all effects are SYNTHETIC — they recover the assumed DGP
of src/causal/simulate_experiment.py, never real Olist effects):
- Per-arm outcome-rate differences vs control (Newcombe 95% CI, z-test).
- ATE per arm vs control via three estimators:
  a) difference-in-means (unbiased under randomization),
  b) IPW with estimated propensity + bootstrap CI,
  c) regression adjustment (OLS, HC1 CI).
- Segment CATEs (recent-active, frequency, recency buckets) with CIs,
  checked against the oracle true_cate column.

Sanity checks: covariate balance (SMD) per arm, oracle recovery
(estimate CIs must cover the known true ATE), reproducibility.

Saves: data/synthetic/causal_estimates.json
"""

from pathlib import Path

import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)


FEATURES_PATH = Path(
    "data/processed/customer_features_v3.parquet"
)
EXPERIMENT_PATH = Path(
    "data/synthetic/intervention_experiment.parquet"
)
ESTIMATES_PATH = Path(
    "data/synthetic/causal_estimates.json"
)

ARMS = (1, 2, 3)
ARM_NAMES = {
    0: "no_intervention",
    1: "coupon",
    2: "free_delivery",
    3: "loyalty_points",
}
COVARIATES = [
    "recency_days",
    "purchase_count_180d",
    "purchase_count_30d",
]
SEED = 42
BOOT_B = 1000


def load_analysis_frame(experiment_path=EXPERIMENT_PATH):
    """Join synthetic experiment to real covariates (one-to-one)."""
    features = pd.read_parquet(FEATURES_PATH)
    experiment = pd.read_parquet(experiment_path)
    df = experiment.merge(
        features[
            ["customer_unique_id", "snapshot_date"] + COVARIATES
        ],
        on=["customer_unique_id", "snapshot_date"],
        how="left",
        validate="one_to_one",
    )
    assert df[COVARIATES].notna().all().all(), (
        "Covariate join produced missing values"
    )
    return df


def rate_difference(df, arm):
    """Unadjusted treated-vs-control rate diff with Newcombe CI."""
    t = df[df["intervention_type"] == arm]["outcome"].to_numpy()
    c = df[df["intervention_type"] == 0]["outcome"].to_numpy()
    kt, nt, kc, nc = int(t.sum()), len(t), int(c.sum()), len(c)
    lo, hi = confint_proportions_2indep(
        kt, nt, kc, nc, compare="diff", method="newcombe"
    )
    z, p = proportions_ztest([kt, kc], [nt, nc])[:2]
    return {
        "n_treated": nt,
        "n_control": nc,
        "rate_treated": float(kt / nt),
        "rate_control": float(kc / nc),
        "diff": float(kt / nt - kc / nc),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "risk_ratio": float((kt / nt) / (kc / nc)),
        "z_pvalue": float(p),
    }


def ipw_ate(df, arm, seed=SEED, n_boot=BOOT_B):
    """IPW with LogReg-estimated propensity; bootstrap CI (fixed weights)."""
    sub = df[df["intervention_type"].isin((0, arm))].copy()
    sub["d"] = (sub["intervention_type"] == arm).astype(int)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(sub[COVARIATES], sub["d"])
    e = clf.predict_proba(sub[COVARIATES])[:, 1]
    e = np.clip(e, 0.01, 0.99)
    d = sub["d"].to_numpy()
    y = sub["outcome"].to_numpy()

    def ate(w_d, w_y, w_e):
        return float(
            np.mean(w_d * w_y / w_e)
            - np.mean((1 - w_d) * w_y / (1 - w_e))
        )

    point = ate(d, y, e)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sub), size=(n_boot, len(sub)))
    boots = np.array(
        [ate(d[i], y[i], e[i]) for i in idx]
    )
    return {
        "ate": point,
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def regression_adjusted_ate(df, arm):
    """OLS of outcome on treatment + covariates; HC1 CI on treatment."""
    sub = df[df["intervention_type"].isin((0, arm))].copy()
    sub["d"] = (sub["intervention_type"] == arm).astype(int)
    X = sm.add_constant(sub[["d"] + COVARIATES])
    fit = sm.OLS(sub["outcome"], X).fit(cov_type="HC1")
    return {
        "ate": float(fit.params["d"]),
        "ci_lo": float(fit.conf_int().loc["d", 0]),
        "ci_hi": float(fit.conf_int().loc["d", 1]),
    }


def true_ate(df, arm):
    """Oracle ATE from the simulation ground truth (validation only)."""
    hetero = 1.0 + 0.5 * (
        df["purchase_count_30d"].to_numpy() > 0
    ).astype(float)
    tau = {1: 0.008, 2: 0.005, 3: 0.003}[arm]
    return float(np.mean(tau * hetero))


def segment_cates(df, arm):
    """Diff-in-means CATEs per segment with CIs + oracle segment truth."""
    df = df.copy()
    df["seg_recent"] = np.where(
        df["purchase_count_30d"] > 0,
        "active_30d",
        "inactive_30d",
    )
    df["seg_freq"] = np.where(
        df["purchase_count_180d"] >= 3,
        "3+",
        df["purchase_count_180d"].astype(str),
    )
    df["seg_recency"] = pd.cut(
        df["recency_days"],
        bins=[0, 60, 120, 180],
        labels=["0-60d", "61-120d", "121-180d"],
        include_lowest=True,
    ).astype(str)
    rows = []
    for col in ("seg_recent", "seg_freq", "seg_recency"):
        for seg, part in df.groupby(col):
            t = part[part["intervention_type"] == arm][
                "outcome"
            ].to_numpy()
            c = part[part["intervention_type"] == 0][
                "outcome"
            ].to_numpy()
            if len(t) == 0 or len(c) == 0:
                continue
            kt, kc = int(t.sum()), int(c.sum())
            lo, hi = confint_proportions_2indep(
                kt, len(t), kc, len(c),
                compare="diff", method="newcombe",
            )
            oracle = float(
                part[part["intervention_type"] == arm][
                    "true_cate"
                ].mean()
            )
            rows.append(
                {
                    "segment": f"{col}={seg}",
                    "n_treated": int(len(t)),
                    "n_control": int(len(c)),
                    "cate": float(t.mean() - c.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "oracle_cate": oracle,
                    "covers_oracle": bool(lo <= oracle <= hi),
                }
            )
    return rows


def balance_smd(df, arm):
    """Standardized mean differences per covariate (arm vs control)."""
    t = df[df["intervention_type"] == arm]
    c = df[df["intervention_type"] == 0]
    out = {}
    for col in COVARIATES:
        pooled = np.sqrt(
            (t[col].var() + c[col].var()) / 2.0
        )
        out[col] = float(
            (t[col].mean() - c[col].mean()) / pooled
        )
    return out


def estimate_all(experiment_path=EXPERIMENT_PATH, seed=SEED):
    df = load_analysis_frame(experiment_path)
    payload = {
        "input": str(experiment_path),
        "seed": seed,
        "rows": int(len(df)),
        "note": (
            "SYNTHETIC estimates recovering assumed DGP effects. "
            "Not real-world effects."
        ),
        "arms": {},
    }
    for arm in ARMS:
        rd = rate_difference(df, arm)
        ipw = ipw_ate(df, arm, seed=seed)
        reg = regression_adjusted_ate(df, arm)
        truth = true_ate(df, arm)
        payload["arms"][str(arm)] = {
            "name": ARM_NAMES[arm],
            "rate_diff": rd,
            "ate_dim": {
                "ate": rd["diff"],
                "ci_lo": rd["ci_lo"],
                "ci_hi": rd["ci_hi"],
            },
            "ate_ipw": ipw,
            "ate_regadj": reg,
            "true_ate": truth,
            "covers_truth": {
                "dim": bool(
                    rd["ci_lo"] <= truth <= rd["ci_hi"]
                ),
                "ipw": bool(
                    ipw["ci_lo"] <= truth <= ipw["ci_hi"]
                ),
                "regadj": bool(
                    reg["ci_lo"] <= truth <= reg["ci_hi"]
                ),
            },
            "segments": segment_cates(df, arm),
            "balance_smd": balance_smd(df, arm),
        }
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Estimate synthetic intervention effects."
    )
    parser.add_argument(
        "--input", default=str(EXPERIMENT_PATH)
    )
    parser.add_argument(
        "--output", default=str(ESTIMATES_PATH)
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    payload = estimate_all(args.input, args.seed)

    print("\nPer-arm ATE vs control (SIMULATED effects):")
    for arm in ARMS:
        a = payload["arms"][str(arm)]
        print(
            f"  {a['name']}: dim={a['ate_dim']['ate']:.4%} "
            f"({a['ate_dim']['ci_lo']:.4%}, {a['ate_dim']['ci_hi']:.4%}) | "
            f"ipw={a['ate_ipw']['ate']:.4%} | "
            f"regadj={a['ate_regadj']['ate']:.4%} | "
            f"TRUE={a['true_ate']:.4%} | "
            f"coverage={a['covers_truth']}"
        )

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
