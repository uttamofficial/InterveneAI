"""Phase 11 — InterveneAI Streamlit dashboard.

Every number is loaded from pipeline artifacts (models/*.pkl,
models/*.json, data/synthetic/*.json/.parquet); nothing is hardcoded
and no fake business numbers appear. Synthetic quantities are labeled
as model estimates throughout.

Run from the repo root:
    .venv/bin/streamlit run dashboard/app.py
"""

from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.evaluate_models import comparison_frame

DECISION_SNAPSHOT = pd.Timestamp("2018-06-01")
ARMS = (1, 2, 3)
ARM_NAMES = {
    0: "no_intervention",
    1: "coupon",
    2: "free_delivery",
    3: "loyalty_points",
}
EST_LABEL = "Model estimates on synthetic data — not real business results."


def artifact(*parts):
    p = ROOT.joinpath(*parts)
    assert p.exists(), f"Missing artifact: {p}"
    return p


@st.cache_resource
def load_bundles():
    champion = joblib.load(
        artifact("models", "champion_pipeline.pkl")
    )
    uplift = {}
    uplift[1] = joblib.load(
        artifact("models", "uplift_tlearner.pkl")
    )
    for arm, fname in (
        (2, "uplift_tlearner_free_delivery.pkl"),
        (3, "uplift_tlearner_loyalty_points.pkl"),
    ):
        saved = joblib.load(artifact("models", fname))
        uplift[arm] = saved
    return champion, uplift


@st.cache_resource
def load_json(*parts):
    with open(artifact(*parts)) as f:
        return json.load(f)


@st.cache_data
def load_overview():
    with open(
        artifact("dashboard", "assets", "overview.json")
    ) as f:
        return json.load(f)


@st.cache_data
def load_snapshot():
    df = pd.read_parquet(
        artifact("dashboard", "assets", "decision_snapshot.parquet")
    )
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


@st.cache_data
def load_recommendations():
    recs = pd.read_parquet(
        artifact("dashboard", "assets", "recommendations.parquet")
    )
    return recs


@st.cache_data
def load_test_scores():
    """Precomputed target + model scores on the frozen test window."""
    frame = pd.read_parquet(
        artifact("dashboard", "assets", "test_scores.parquet")
    )
    return {
        "y": frame["y"].to_numpy(),
        "champion_logreg": frame["champion_logreg"].to_numpy(),
        "tuned_logreg": frame["tuned_logreg"].to_numpy(),
        "rf": frame["rf"].to_numpy(),
        "hgb": frame["hgb"].to_numpy(),
    }


@st.cache_resource
def shap_explainer():
    from src.evaluation.explainability import (
        design_matrix,
        load_champion,
        make_explainer,
    )

    bundle = load_champion()
    bg = pd.read_parquet(
        artifact("dashboard", "assets", "shap_background.parquet")
    )
    X_bg = design_matrix(bundle, bg)
    explainer = make_explainer(bundle, X_bg)
    return bundle, explainer


def section_overview():
    st.header("Overview")
    ov = load_overview()
    comp = load_json("models", "model_comparison.json")
    champ = comp["details"]["logreg"]["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{ov['customers']:,}")
    c2.metric("Observations", f"{ov['observations']:,}")
    c3.metric(
        "Purchase rate",
        f"{100 * ov['purchase_rate']:.2f}%",
    )
    c4.metric(
        "Champion test PR-AUC",
        f"{champ['pr_auc']:.4f}",
        help="From models/model_comparison.json; random = "
        f"{champ['prevalence']:.4f}",
    )
    st.caption(
        "Stacked customer-snapshot panel (15 monthly snapshots). "
        "Champion = tuned Logistic Regression selected on validation PR-AUC."
    )


def customer_row(customer_id):
    snap = load_snapshot()
    hit = snap[snap["customer_unique_id"] == customer_id]
    return hit, snap


def section_customer():
    st.header("Customer Intelligence")
    st.caption(f"Decision snapshot {DECISION_SNAPSHOT.date()}.")
    hit, snap = None, None
    snap = load_snapshot()
    sample = snap.sample(500, random_state=42)["customer_unique_id"].tolist()
    choice = st.selectbox("Customer (500-row sample)", sample)
    typed = st.text_input(
        "…or paste a full customer_unique_id", value=""
    ).strip()
    cid = typed if typed else choice
    hit = snap[snap["customer_unique_id"] == cid]
    if hit.empty:
        st.warning("Customer ID not present at the decision snapshot.")
        return
    row = hit.iloc[0]
    champion, uplift = load_bundles()
    feats = champion["features"]
    p_buy = float(
        champion["pipeline"].predict_proba(hit[feats])[:, 1][0]
    )
    st.subheader(f"Purchase probability: {p_buy:.4f}")
    st.write("Feature snapshot:")
    st.dataframe(hit[feats].T.rename(columns={hit.index[0]: "value"}))
    st.write("Per-intervention uplift (T-learner estimates):")
    rows = []
    for arm in ARMS:
        b = uplift[arm]
        pt = float(
            b["treated_pipeline"].predict_proba(hit[feats])[:, 1][0]
        )
        pc = float(
            b["control_pipeline"].predict_proba(hit[feats])[:, 1][0]
        )
        rows.append(
            {
                "intervention": ARM_NAMES[arm],
                "P(Y|T=1)": round(pt, 4),
                "P(Y|T=0)": round(pc, 4),
                "uplift": round(pt - pc, 4),
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("intervention"))
    recs = load_recommendations()
    mine = recs[
        (recs["strategy"] == "C_profit")
        & (recs["customer_unique_id"] == cid)
    ]
    if len(mine):
        r = mine.iloc[0]
        st.success(
            f"Recommended intervention (strategy C @20k budget): "
            f"{ARM_NAMES[int(r['recommended_intervention'])]} | "
            f"expected profit "
            f"{r['expected_incremental_profit']:.2f} BRL (estimate)"
        )
    else:
        st.info(
            "No intervention recommended for this customer under "
            "strategy C at the 20k budget (not profit-positive)."
        )


def section_simulator():
    st.header("Intervention Simulator")
    st.warning(EST_LABEL)
    st.selectbox(
        "Targeting strategy",
        ["C_profit", "B_uplift", "A_purchase_prob"],
        key="sim_strategy",
    )
    budget = st.number_input(
        "Campaign budget (BRL)",
        min_value=0.0,
        value=20000.0,
        step=1000.0,
    )
    c1, c2, c3 = st.columns(3)
    costs = {
        1: c1.number_input("Coupon cost", value=5.0, step=0.5),
        2: c2.number_input("Free-delivery cost", value=8.0, step=0.5),
        3: c3.number_input("Loyalty cost", value=3.0, step=0.5),
    }
    use_aov = st.checkbox("Use per-customer historical AOV", value=True)
    order_value = None
    if not use_aov:
        order_value = st.number_input(
            "Expected order value (BRL)", value=150.0, step=10.0
        )
    cap = st.number_input(
        "Customer cap (0 = no cap)", min_value=0, value=0, step=500
    )
    if st.button("Run simulation"):
        from src.optimization.roi_simulator import simulate_roi

        with st.spinner("Scoring customers…"):
            r = simulate_roi(
                budget=float(budget),
                intervention_costs={str(k): float(v) for k, v in costs.items()},
                order_value=order_value,
                n_customers=int(cap) if cap else None,
                strategy=st.session_state.sim_strategy,
                decision_frame=load_snapshot(),
            )
        cols = st.columns(3)
        cols[0].metric("Customers targeted", f"{r['customers_targeted']:,}")
        cols[1].metric(
            "Expected incremental purchases",
            f"{r['expected_incremental_purchases']:.1f}",
        )
        cols[2].metric("Campaign cost", f"{r['campaign_cost']:,.0f} BRL")
        cols = st.columns(3)
        cols[0].metric(
            "Expected incremental revenue",
            f"{r['expected_incremental_revenue']:,.0f} BRL",
        )
        cols[1].metric(
            "Expected incremental profit",
            f"{r['expected_incremental_profit']:,.0f} BRL",
        )
        roi = r["roi"]
        cols[2].metric(
            "ROI", "—" if roi is None else f"{roi:.2f}"
        )
        st.caption(
            f"Oracle profit (simulation ground truth): "
            f"{r['oracle_profit']:,.0f} BRL. {r['label']}"
        )


def section_performance():
    st.header("Model Performance")
    comp = load_json("models", "model_comparison.json")
    st.write("Frozen chronological test window "
             f"{comp['splits']['test'][0]} → {comp['splits']['test'][1]}.")
    st.dataframe(comparison_frame(comp["test_comparison"]))
    st.caption(
        "Champion selected on validation PR-AUC "
        f"({comp['champion']}); accuracy never used."
    )
    scores = load_test_scores()
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("champion_logreg", "tuned_logreg", "rf", "hgb"):
        if name not in scores:
            continue
        p, r, _ = precision_recall_curve(scores["y"], scores[name])
        axes[0].step(r, p, where="post", label=name)
        f, t, _ = roc_curve(scores["y"], scores[name])
        axes[1].plot(f, t, label=name)
    axes[0].axhline(scores["y"].mean(), color="grey", linestyle=":")
    axes[0].set_title("Precision–Recall")
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle=":")
    axes[1].set_title("ROC")
    axes[1].legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("Calibration (test, quantile bins)")
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("logreg", "rf", "hgb"):
        cal = comp["details"][name]["test_calibration"]
        ax.plot(
            cal["prob_pred"],
            cal["prob_true"],
            marker="o",
            label=f"{name} (Brier "
            f"{comp['details'][name]['test']['brier']:.4f})",
        )
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("Precision@k / lift@k (test)")
    rows = []
    for name in ("logreg", "rf", "hgb"):
        for tk in comp["details"][name]["test_top_k"]:
            rows.append(
                {
                    "model": name,
                    "top": f"{int(100 * tk['frac'])}%",
                    "precision": round(tk["precision"], 4),
                    "lift": round(tk["lift"], 2),
                }
            )
    st.dataframe(
        pd.DataFrame(rows).set_index(["model", "top"])
    )


def section_explain():
    st.header("Explainability")
    imp = load_json("models", "shap_global_importance.json")
    tab = pd.DataFrame(imp["rows"]).set_index("feature")
    st.write("Global importance — mean |SHAP|, log-odds "
             f"(n={imp['n']:,} test rows).")
    st.bar_chart(tab["mean_abs_shap"])
    st.caption(
        "Associational description of the linear champion, not causal "
        "drivers. Duplicate velocity/count features split credit."
    )
    st.subheader("Customer-level explanation")
    snap = load_snapshot()
    sample = snap.sample(200, random_state=7)["customer_unique_id"].tolist()
    cid = st.selectbox("Customer", sample, key="shap_customer")
    hit = snap[snap["customer_unique_id"] == cid]
    if hit.empty:
        return
    bundle, explainer = shap_explainer()
    from src.evaluation.explainability import individual_explanation

    row = hit[bundle["features"]]
    Xi = bundle["pipeline"].named_steps["impute"].transform(row)
    Xs = bundle["pipeline"].named_steps["scale"].transform(Xi)
    exp = individual_explanation(bundle, explainer, Xs[0])
    st.write(
        f"Predicted purchase probability: **{exp['probability']:.4f}** "
        f"(log-odds {exp['log_odds']:.3f}, base "
        f"{exp['expected_value']:.3f})"
    )
    contrib = pd.DataFrame(exp["contributions"]).set_index("feature")
    top = contrib.reindex(
        contrib["shap_value"].abs().sort_values(ascending=False).index
    ).head(8)
    st.bar_chart(top["shap_value"])


def section_method():
    st.header("Methodology / Assumptions")
    st.subheader("Real Olist data")
    st.write(
        "Customer panel, snapshot design with strict pre-snapshot features "
        "(`order_purchase_timestamp < snapshot_date`), purchase/count/"
        "monetary/behavioral features, and chronological train/val/test "
        "splits. No temporal leakage by construction (audited in Phase 2)."
    )
    st.subheader("Simulated experiment")
    params = load_json("dashboard", "assets", "intervention_params.json")
    st.write(
        "Olist contains no randomized intervention. Treatment, costs and "
        "outcomes are synthetic with documented assumed effects "
        f"(coupon +{params['true_tau_pp']['1']}pp, free_delivery "
        f"+{params['true_tau_pp']['2']}pp, loyalty "
        f"+{params['true_tau_pp']['3']}pp; seed {params['seed']}). "
        "Oracle columns exist for method validation only."
    )
    st.subheader("Causal assumptions")
    st.write(
        "Consistency, SUTVA (no spillover), ignorability under randomization, "
        "positivity. Targeted assignment is confounded by design. Uplift "
        "scores and ROI outputs are model estimates, never measurements."
    )
    st.subheader("Limitations")
    comp = load_json("models", "model_comparison.json")
    test_pr = comp["details"]["logreg"]["test"]["pr_auc"]
    st.write(
        "Rare positives (~0.6%) with weak features: ranking is modest "
        f"(test PR-AUC {test_pr:.4f}), uplift ranking near-random, "
        "probabilities miscalibrated by balanced weights, and assumed "
        "economics make most campaigns oracle-unprofitable. Nothing here "
        "predicts real campaign performance without a real randomized "
        "experiment."
    )


st.set_page_config(page_title="InterveneAI", layout="wide")
st.title("InterveneAI — Intervention Analytics")

page = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "Customer Intelligence",
        "Intervention Simulator",
        "Model Performance",
        "Explainability",
        "Methodology / Assumptions",
    ],
)

{
    "Overview": section_overview,
    "Customer Intelligence": section_customer,
    "Intervention Simulator": section_simulator,
    "Model Performance": section_performance,
    "Explainability": section_explain,
    "Methodology / Assumptions": section_method,
}[page]()
