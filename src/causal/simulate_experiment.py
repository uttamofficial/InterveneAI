"""Phase 6 — Synthetic Intervention Experiment.

The Olist dataset contains NO real randomized coupon/intervention
experiment: there is no observed treatment, no control group, and no
intervention outcome. This module therefore builds a clearly labelled
SYNTHETIC experimental layer on top of the real customer panel. Every
effect size in the data-generating process (DGP) below is an ASSUMED
constant chosen by the authors — nothing here is a discovered
real-world effect, and results from this file must never be reported
as real treatment effects.

What is real vs simulated (see docs/synthetic_intervention.md):
- REAL (observed): customer_unique_id, snapshot_date, and the
  pre-snapshot covariates used only as simulation inputs.
- SIMULATED: treatment, intervention_type, intervention_cost,
  outcome, true_propensity, true_cate.

Data-generating process (all numbers are assumptions):
- Base purchase probability (probability scale, clipped to [0, 1]):
      base = 0.004
           + 0.004 * (1 - recency_days / 180)
           + 0.004 * I(purchase_count_180d >= 2)
  Calibrated so the control rate lands near the observed ~0.6%.
- Assumed absolute effects (percentage points) vs control:
      coupon (+0.8pp), free_delivery (+0.5pp), loyalty_points (+0.3pp).
- Documented heterogeneity: the effect is 50% larger for customers
  with any purchase in the last 30 days (recently active):
      hetero_mult = 1 + 0.5 * I(purchase_count_30d > 0)
      p = clip(base + TAU[type] * hetero_mult, 0, 1)
      outcome ~ Bernoulli(p) with a seeded RNG.
- Assumed intervention costs (BRL): coupon 5.0, free_delivery 8.0,
  loyalty_points 3.0, control 0.0.

Treatment assignment (configurable via CLI):
- "random": complete randomization with configurable arm
  probabilities (default control 0.4 / 0.2 / 0.2 / 0.2).
- "targeted": assignment probability rises for recently active
  customers (documented confounding — recency also drives the
  outcome, so unadjusted comparisons are biased BY DESIGN).

Oracle columns (ground truth for future estimator validation —
must NEVER be used as model features):
- true_propensity: P(row received its assigned arm).
- true_cate: effect of the received intervention vs control
  (0.0 for control rows).
"""

from pathlib import Path

import argparse
import json
import numpy as np
import pandas as pd


FEATURES_PATH = Path(
    "data/processed/customer_features_v3.parquet"
)
OUTPUT_PATH = Path(
    "data/synthetic/intervention_experiment.parquet"
)
PARAMS_PATH = Path(
    "data/synthetic/intervention_params.json"
)

REAL_TARGET = "purchased_next_60_days"

INTERVENTIONS = {
    0: "no_intervention",
    1: "coupon",
    2: "free_delivery",
    3: "loyalty_points",
}

# Assumed absolute effects in probability units. ASSUMPTIONS, not findings.
TRUE_TAU = {
    0: 0.0,
    1: 0.008,
    2: 0.005,
    3: 0.003,
}

# Assumed costs in BRL. ASSUMPTIONS, not findings.
INTERVENTION_COSTS = {
    0: 0.0,
    1: 5.0,
    2: 8.0,
    3: 3.0,
}

BASE_RATE = 0.004
RECENCY_WEIGHT = 0.004
REPEAT_WEIGHT = 0.004
HETERO_BONUS = 0.5

DEFAULT_ARM_PROBS = {
    0: 0.4,
    1: 0.2,
    2: 0.2,
    3: 0.2,
}

DEFAULT_SEED = 42
DEFAULT_MECHANISM = "random"


def assign_treatment(df, mechanism, arm_probs, seed):
    """Return (arm codes, propensity of received arm). Seeded."""
    rng = np.random.default_rng(seed)
    n = len(df)

    if mechanism == "random":
        arms = list(arm_probs)
        probs = np.array([arm_probs[a] for a in arms])
        assert abs(probs.sum() - 1.0) < 1e-9, (
            "Arm probabilities must sum to 1"
        )
        chosen = rng.choice(arms, size=n, p=probs)
        propensity = np.array([arm_probs[a] for a in chosen])

    elif mechanism == "targeted":
        # Recently active customers are more likely to be treated.
        # Confounded BY DESIGN: recency also raises the outcome.
        p_treat = 0.3 + 0.4 * (
            1 - df["recency_days"].to_numpy() / 180.0
        )
        treated = rng.random(n) < p_treat
        treated_arms = np.array([1, 2, 3])
        chosen = np.zeros(n, dtype=int)
        chosen[treated] = rng.choice(
            treated_arms, size=int(treated.sum())
        )
        propensity = np.where(
            treated, p_treat / 3.0, 1.0 - p_treat
        )

    else:
        raise ValueError(
            f"Unknown mechanism: {mechanism!r} "
            "(use 'random' or 'targeted')"
        )

    return chosen, propensity


def simulate_outcomes(df, arms, seed):
    """Bernoulli outcomes from the documented DGP. Seeded."""
    rng = np.random.default_rng(seed + 1)
    recency = df["recency_days"].to_numpy()
    repeat = (
        df["purchase_count_180d"].to_numpy() >= 2
    ).astype(float)
    recent_active = (
        df["purchase_count_30d"].to_numpy() > 0
    ).astype(float)

    base = (
        BASE_RATE
        + RECENCY_WEIGHT * (1 - recency / 180.0)
        + REPEAT_WEIGHT * repeat
    )
    hetero_mult = 1.0 + HETERO_BONUS * recent_active
    tau = np.array([TRUE_TAU[int(a)] for a in arms])
    cate = tau * hetero_mult

    prob = np.clip(base + cate, 0.0, 1.0)
    outcome = (rng.random(len(df)) < prob).astype("int8")
    return outcome, cate


def build_experiment(
    mechanism=DEFAULT_MECHANISM,
    arm_probs=None,
    seed=DEFAULT_SEED,
):
    """Build the synthetic experimental dataset (no I/O)."""
    features = pd.read_parquet(FEATURES_PATH)
    panel = features[
        ["customer_unique_id", "snapshot_date"]
    ].copy()
    covariates = features[
        [
            "recency_days",
            "purchase_count_180d",
            "purchase_count_30d",
        ]
    ]

    if arm_probs is None:
        arm_probs = dict(DEFAULT_ARM_PROBS)

    arms, propensity = assign_treatment(
        covariates, mechanism, arm_probs, seed
    )
    outcome, cate = simulate_outcomes(
        covariates, arms, seed
    )

    experiment = pd.DataFrame(
        {
            "customer_unique_id": panel["customer_unique_id"],
            "snapshot_date": pd.to_datetime(
                panel["snapshot_date"]
            ),
            "treatment": (arms > 0).astype("int8"),
            "intervention_type": pd.Series(
                arms, dtype="int8"
            ),
            "intervention_cost": pd.Series(arms).map(
                INTERVENTION_COSTS
            ),
            "outcome": outcome,
            "true_propensity": propensity,
            "true_cate": np.where(arms == 0, 0.0, cate),
        }
    )
    return experiment


def validate_experiment(
    experiment, mechanism, arm_probs, seed
):
    """Fail loudly on any inconsistency; report the rest."""
    print("\n--- Validation ---")

    # 1. Required columns, and NO real target or real features.
    required = [
        "customer_unique_id",
        "snapshot_date",
        "treatment",
        "intervention_type",
        "intervention_cost",
        "outcome",
    ]
    assert all(c in experiment.columns for c in required), (
        "Missing required columns"
    )
    assert REAL_TARGET not in experiment.columns, (
        "Real outcome leaked into synthetic dataset"
    )
    print("Columns OK; real target absent.")

    # 2. Treatment counts and proportions.
    counts = (
        experiment["intervention_type"]
        .value_counts()
        .sort_index()
    )
    print("\nTreatment counts:")
    print(counts.to_string())
    assert set(counts.index) == {0, 1, 2, 3}, (
        "Every arm must be present"
    )
    props = counts / len(experiment)
    print("\nTreatment proportions:")
    print(props.to_string())
    if mechanism == "random":
        for arm, expected in arm_probs.items():
            assert (
                abs(props[arm] - expected) < 0.01
            ), f"Arm {arm} proportion off-target"
        print("Proportions match configured arm probabilities.")

    # 3. Outcome rates overall and by arm.
    rates = experiment.groupby("intervention_type")[
        "outcome"
    ].mean()
    print("\nOutcome rates by arm (SIMULATED, assumed effects):")
    for arm, rate in rates.items():
        print(
            f"  {arm} ({INTERVENTIONS[arm]}): {rate:.4%}"
        )
    for arm in (1, 2, 3):
        assert rates[arm] > rates[0], (
            f"Assumed positive effect of arm {arm} not visible"
        )
    assert set(experiment["outcome"].unique()) <= {0, 1}, (
        "Outcome must be binary"
    )
    assert experiment["outcome"].isna().sum() == 0, (
        "Outcome must have no missing values"
    )

    # 4. Intervention costs follow the documented mapping exactly.
    expected_cost = experiment["intervention_type"].map(
        INTERVENTION_COSTS
    )
    assert (experiment["intervention_cost"] == expected_cost).all(), (
        "Cost mapping violated"
    )
    print("\nCosts exact per mapping:")
    print(
        experiment.groupby("intervention_type")[
            "intervention_cost"
        ].agg(["count", "mean", "sum"]).to_string()
    )

    # 5. Reproducibility: same seed -> identical; other seed -> differs.
    again = build_experiment(mechanism, arm_probs, seed)
    pd.testing.assert_frame_equal(experiment, again)
    other = build_experiment(mechanism, arm_probs, seed + 999)
    assert not other["intervention_type"].equals(
        experiment["intervention_type"]
    ), "Different seed produced identical assignment"
    print("\nReproducibility: same seed identical, new seed differs.")

    print("\n--- All validation checks passed ---")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a synthetic intervention experiment."
    )
    parser.add_argument(
        "--mechanism",
        default=DEFAULT_MECHANISM,
        choices=["random", "targeted"],
    )
    parser.add_argument(
        "--arm-probs",
        default=None,
        help=(
            "JSON mapping arm -> probability, e.g. "
            "'{\"0\": 0.5, \"1\": 0.2, \"2\": 0.2, \"3\": 0.1}'. "
            "Random mechanism only."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    arm_probs = (
        {int(k): float(v) for k, v in json.loads(args.arm_probs).items()}
        if args.arm_probs
        else dict(DEFAULT_ARM_PROBS)
    )

    print(f"Mechanism: {args.mechanism} | seed: {args.seed}")
    print(f"Arm probabilities: {arm_probs}")

    experiment = build_experiment(
        args.mechanism, arm_probs, args.seed
    )
    validate_experiment(
        experiment, args.mechanism, arm_probs, args.seed
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    experiment.to_parquet(OUTPUT_PATH, index=False)
    with open(PARAMS_PATH, "w") as f:
        json.dump(
            {
                "mechanism": args.mechanism,
                "arm_probs": arm_probs,
                "seed": args.seed,
                "true_tau_pp": {
                    str(k): v * 100 for k, v in TRUE_TAU.items()
                },
                "costs_brl": INTERVENTION_COSTS,
                "note": (
                    "SYNTHETIC. Effects are assumed constants, "
                    "not measured real-world effects."
                ),
            },
            f,
            indent=2,
        )

    print(f"\nSaved: {OUTPUT_PATH} ({len(experiment):,} rows)")
    print(f"Saved: {PARAMS_PATH}")


if __name__ == "__main__":
    main()
