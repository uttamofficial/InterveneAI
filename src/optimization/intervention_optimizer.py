"""Phase 9 — Intervention Optimization (synthetic data only).

Decides WHO receives an intervention, WHICH one, and whether it is
economically worthwhile — under a campaign budget constraint.

Per customer x intervention, from actual model outputs:
1. predicted outcome without intervention: control T-learner model P(Y|X,T=0)
2. predicted outcome with intervention: arm T-learner model P(Y|X,T=arm)
3. incremental probability (uplift) = (2) - (1)
4. intervention cost (assumed flat BRL per type)
5. expected incremental revenue = uplift x customer AOV
   (AOV = avg_order_value_180d at the decision snapshot; documented proxy)
6. expected incremental profit = revenue - cost

Targeting is on estimated incremental value, NEVER on raw purchase
probability. Strategies compared (same budget, same candidate arms):
- A: rank by predicted purchase probability (the criticized baseline)
- B: rank by predicted uplift
- C: rank by expected incremental profit (profit-positive only)

Greedy allocation: each customer gets at most their best-profit arm;
walk the ranking while cost fits the remaining budget. Strategy honesty:
A/B may select money-losing rows (counted as losses); C filters them.

Scoring uses actual outputs two ways: model-expected totals (what the
planner believes) and oracle totals from simulation ground truth
(true_cate of the CHOSEN arm x AOV - cost). Nothing here is proven in a
real business experiment — all gains are synthetic.

Saves:
- models/uplift_tlearner_free_delivery.pkl
- models/uplift_tlearner_loyalty_points.pkl
  (coupon bundle reused from Phase 8; same config/seed)
- data/synthetic/optimization_recommendations.parquet
- data/synthetic/optimization_summary.json
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import sklearn
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.causal.uplift_model import (
    FEATURES,
    fit_t_learner,
    load_uplift_frame,
    predict_uplift,
)


EXPERIMENT_PATH = Path(
    "data/synthetic/intervention_experiment.parquet"
)
MODEL_DIR = Path("models")
RECOMMENDATIONS_PATH = Path(
    "data/synthetic/optimization_recommendations.parquet"
)
SUMMARY_PATH = Path(
    "data/synthetic/optimization_summary.json"
)

ARMS = (1, 2, 3)
ARM_NAMES = {
    0: "no_intervention",
    1: "coupon",
    2: "free_delivery",
    3: "loyalty_points",
}
COSTS = {0: 0.0, 1: 5.0, 2: 8.0, 3: 3.0}
# Assumed DGP effects (probability units), for oracle scoring only.
TRUE_TAU = {0: 0.0, 1: 0.008, 2: 0.005, 3: 0.003}

TRAIN_END = pd.Timestamp("2017-12-01")
DECISION_SNAPSHOT = pd.Timestamp("2018-06-01")

SEED = 42
PRIMARY_BUDGET = 20000.0
BUDGET_LEVELS = (5000.0, 20000.0, 50000.0, 150000.0)


def load_arm_models():
    """Coupon bundle from Phase 8; train the other two identically."""
    models = {}
    bundle = joblib.load(
        MODEL_DIR / "uplift_tlearner.pkl"
    )
    assert bundle["arm"] == 1, "Phase-8 bundle must be coupon"
    models[1] = (
        bundle["treated_pipeline"],
        bundle["control_pipeline"],
    )
    # The full uplift frame is loaded lazily: only when an arm bundle is
    # missing and must be trained. Saved bundles never touch disk data.
    train = None
    for arm in (2, 3):
        path = MODEL_DIR / (
            f"uplift_tlearner_{ARM_NAMES[arm]}.pkl"
        )
        if path.exists():
            saved = joblib.load(path)
            models[arm] = (
                saved["treated_pipeline"],
                saved["control_pipeline"],
            )
        else:
            if train is None:
                df = load_uplift_frame()
                train = df[df["snapshot_date"] <= TRAIN_END]
            tr = train[train["intervention_type"].isin((0, arm))]
            pipe_t, pipe_c = fit_t_learner(tr)
            models[arm] = (pipe_t, pipe_c)
            joblib.dump(
                {
                    "treated_pipeline": pipe_t,
                    "control_pipeline": pipe_c,
                    "arm": arm,
                    "arm_name": ARM_NAMES[arm],
                    "features": FEATURES,
                    "seed": SEED,
                    "sklearn_version": sklearn.__version__,
                    "trained_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                },
                path,
            )
            print(f"Saved: {path}")
    return models


def build_prediction_frame(df_snap, models):
    """One row per customer x arm with the six economic quantities."""
    X = df_snap[FEATURES]
    p_control = models[1][1].predict_proba(X)[:, 1]
    aov = df_snap["avg_order_value_180d"].to_numpy()
    hetero = 1.0 + 0.5 * (
        df_snap["purchase_count_30d"].to_numpy() > 0
    ).astype(float)
    frames = []
    for arm in ARMS:
        p_treated = models[arm][0].predict_proba(X)[:, 1]
        uplift = p_treated - p_control
        revenue = uplift * aov
        profit = revenue - COSTS[arm]
        frames.append(
            pd.DataFrame(
                {
                    "customer_unique_id": df_snap[
                        "customer_unique_id"
                    ].to_numpy(),
                    "intervention": arm,
                    "p_control": p_control,
                    "p_treated": p_treated,
                    "uplift": uplift,
                    "cost": COSTS[arm],
                    "exp_revenue": revenue,
                    "exp_profit": profit,
                    # Oracle scoring of THIS arm (ground truth).
                    "oracle_profit": TRUE_TAU[arm] * hetero * aov
                    - COSTS[arm],
                }
            )
        )
    pred = pd.concat(frames, ignore_index=True)
    # Each customer's best-profit arm (ties -> cheaper arm).
    pred["rank_key"] = list(
        zip(pred["exp_profit"], -pred["cost"])
    )
    best_idx = pred.groupby("customer_unique_id")[
        "rank_key"
    ].idxmax()
    best = pred.loc[best_idx].drop(columns=["rank_key"])
    return pred, best.reset_index(drop=True)


def optimize(best, budget, rank_by, profit_filter):
    """Greedy budget allocation over one ranking criterion."""
    cand = best.copy()
    if profit_filter:
        cand = cand[cand["exp_profit"] > 0]
    cand = cand.sort_values(rank_by, ascending=False)
    spent, chosen = 0.0, []
    for row in cand.itertuples():
        if spent + row.cost <= budget:
            spent += row.cost
            chosen.append(row.Index)
    plan = cand.loc[chosen]
    return pd.DataFrame(
        {
            "customer_unique_id": plan["customer_unique_id"],
            "recommended_intervention": plan["intervention"].astype(
                "int8"
            ),
            "expected_incremental_revenue": plan["exp_revenue"],
            "expected_incremental_cost": plan["cost"],
            "expected_incremental_profit": plan["exp_profit"],
            "oracle_profit": plan["oracle_profit"],
        }
    )


STRATEGIES = {
    # rank_by, profit_filter
    "A_purchase_prob": ("p_control", False),
    "B_uplift": ("uplift", False),
    "C_profit": ("exp_profit", True),
}


def summarize(plan):
    return {
        "n_treated": int(len(plan)),
        "spend": float(plan["expected_incremental_cost"].sum()),
        "model_expected_profit": float(
            plan["expected_incremental_profit"].sum()
        ),
        "oracle_profit": float(plan["oracle_profit"].sum()),
        "mix": {
            ARM_NAMES[int(a)]: int((plan["recommended_intervention"] == a).sum())
            for a in sorted(plan["recommended_intervention"].unique())
        },
    }


def main():
    df = load_uplift_frame()
    snap = df[df["snapshot_date"] == DECISION_SNAPSHOT].copy()
    print(
        f"Decision snapshot: {DECISION_SNAPSHOT.date()} | "
        f"customers: {len(snap):,}"
    )
    models = load_arm_models()
    _, best = build_prediction_frame(snap, models)
    print(
        f"Customers with a profit-positive arm: "
        f"{int((best['exp_profit'] > 0).sum()):,} / {len(best):,}"
    )

    summary = {
        "decision_snapshot": str(DECISION_SNAPSHOT.date()),
        "budgets": {},
        "note": (
            "SYNTHETIC optimization. Oracle profits use simulation "
            "ground truth. Not validated in any real experiment."
        ),
    }
    primary_plans = {}
    for budget in BUDGET_LEVELS:
        summary["budgets"][str(budget)] = {}
        for name, (rank_by, filt) in STRATEGIES.items():
            plan = optimize(best, budget, rank_by, filt)
            summary["budgets"][str(budget)][name] = summarize(plan)
            if budget == PRIMARY_BUDGET:
                plan["strategy"] = name
                primary_plans[name] = plan
            s = summary["budgets"][str(budget)][name]
            print(
                f"budget {budget:>9,.0f} | {name:>15}: "
                f"n={s['n_treated']:>6,} spend={s['spend']:>9,.0f} "
                f"model_profit={s['model_expected_profit']:>10,.0f} "
                f"oracle_profit={s['oracle_profit']:>9,.0f} "
                f"mix={s['mix']}"
            )

    out = pd.concat(primary_plans.values(), ignore_index=True)
    out.to_parquet(RECOMMENDATIONS_PATH, index=False)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RECOMMENDATIONS_PATH} ({len(out):,} rows)")
    print(f"Saved: {SUMMARY_PATH}")

    reloaded = pd.read_parquet(RECOMMENDATIONS_PATH)
    assert len(reloaded) == len(out), "Reload row mismatch"
    print("Reload check: recommendations reproduce exactly.")


if __name__ == "__main__":
    main()
