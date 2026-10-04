"""Phase 10 — Business ROI simulator (model estimates, not results).

Interactive planning function over the Phase-9 prediction frame at the
frozen decision snapshot (2018-06-01). Every assumption is an argument:
campaign budget, intervention costs, expected order value, customer cap
and targeting strategy. Outputs are MODEL ESTIMATES from synthetic-data
models — never real business results. Oracle counterparts (simulation
ground truth) are reported alongside so miscalibration is visible.
"""

from pathlib import Path

import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.optimization.intervention_optimizer import (
    DECISION_SNAPSHOT,
    STRATEGIES,
    build_prediction_frame,
    load_arm_models,
    optimize,
)
from src.causal.uplift_model import load_uplift_frame


def simulate_roi(
    budget=20000.0,
    intervention_costs=None,
    order_value=None,
    n_customers=None,
    strategy="C_profit",
):
    """Run one ROI scenario. All quantities are estimates.

    budget: max campaign spend (BRL).
    intervention_costs: {arm: cost} override (default assumed costs).
    order_value: scalar BRL override for expected revenue per incremental
        purchase (default None = each customer's own historical AOV).
    n_customers: cap on ranked candidates before budget allocation.
    strategy: one of A_purchase_prob / B_uplift / C_profit.
    """
    from src.optimization.intervention_optimizer import COSTS

    costs = dict(COSTS)
    if intervention_costs:
        costs.update({int(k): float(v) for k, v in intervention_costs.items()})

    models = load_arm_models()
    df = load_uplift_frame()
    snap = df[df["snapshot_date"] == DECISION_SNAPSHOT].copy()
    _, best = build_prediction_frame(snap, models)

    # Apply assumption overrides to a working copy.
    best = best.copy()
    best["cost"] = best["intervention"].map(costs)
    # Revenue base: per-customer historical AOV, or a scalar override.
    aov = snap.set_index("customer_unique_id")["avg_order_value_180d"]
    revenue_base = (
        aov.loc[best["customer_unique_id"]].to_numpy()
        if order_value is None
        else float(order_value)
    )
    best["exp_revenue"] = best["uplift"].to_numpy() * revenue_base
    best["exp_profit"] = best["exp_revenue"] - best["cost"].to_numpy()
    # oracle_profit untouched: ground-truth scoring of chosen arms.

    rank_by, filt = STRATEGIES[strategy]
    if strategy == "C_profit":
        filt = True
    cand = best.copy()
    if filt:
        cand = cand[cand["exp_profit"] > 0]
    cand = cand.sort_values(rank_by, ascending=False)
    if n_customers is not None:
        cand = cand.head(int(n_customers))

    spent, picks = 0.0, []
    for idx, row in cand.iterrows():
        if spent + row["cost"] <= budget:
            spent += row["cost"]
            picks.append(idx)
    plan = cand.loc[picks]

    purchases = float(plan["uplift"].sum())
    revenue = float(plan["exp_revenue"].sum())
    profit = float(plan["exp_profit"].sum())
    oracle = float(plan["oracle_profit"].sum())
    result = {
        "strategy": strategy,
        "budget": float(budget),
        "assumed_costs": {str(k): float(v) for k, v in costs.items()},
        "order_value": (
            "per-customer historical AOV" if order_value is None
            else float(order_value)
        ),
        "n_customers_cap": n_customers,
        "customers_targeted": int(len(plan)),
        "expected_incremental_purchases": purchases,
        "expected_incremental_revenue": revenue,
        "campaign_cost": spent,
        "expected_incremental_profit": profit,
        "roi": (profit / spent) if spent > 0 else None,
        "oracle_profit": oracle,
        "label": (
            "MODEL ESTIMATES on synthetic data — "
            "not real business results."
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Business ROI simulator (estimates only)."
    )
    parser.add_argument("--budget", type=float, default=20000.0)
    parser.add_argument("--order-value", type=float, default=None)
    parser.add_argument("--n-customers", type=int, default=None)
    parser.add_argument(
        "--strategy", default="C_profit", choices=list(STRATEGIES)
    )
    parser.add_argument(
        "--costs",
        default=None,
        help='JSON, e.g. \'{"1": 4.0, "2": 6.0, "3": 2.5}\'',
    )
    args = parser.parse_args()
    result = simulate_roi(
        budget=args.budget,
        intervention_costs=json.loads(args.costs)
        if args.costs
        else None,
        order_value=args.order_value,
        n_customers=args.n_customers,
        strategy=args.strategy,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
