"""Phase 8 — Uplift Modeling with a T-learner (synthetic data only).

No uplift package is installed here (scikit-uplift/causalml absent),
and none is installed blindly. The T-learner is implemented
transparently: two Logistic Regression classifiers estimate
P(Y=1 | X, T=1) and P(Y=1 | X, T=0); uplift = treated_prob - control_prob.

Key concept: uplift is NOT purchase probability. A customer with high
P(Y=1 | X, T=0) buys anyway — treating them wastes budget. Targeting
needs the incremental effect, which no standard classifier estimates.

Protocol:
- One T-learner per intervention arm vs control (primary: coupon).
- Train on snapshots <= 2017-12-01, evaluate on 2018-04-01..2018-06-01
  (same frozen test window as Phases 4-5; no tuning, so no val split).
- Preprocessing (median impute + scale) fit on train only.

Evaluation (all on synthetic outcomes):
- Qini curve + AUUC vs random targeting vs oracle (true_cate) ranking,
  plus the Qini coefficient.
- Uplift-by-decile calibration: predicted uplift vs observed
  treated-minus-control rate per decile (Newcombe CIs).
- Uplift segments: negative (< 0) / low / medium / high (tertiles of
  the non-negative predictions), each with observed diff and oracle mean.
- Spearman correlation of predicted uplift with oracle true_cate.

Saves:
- models/uplift_tlearner.pkl (primary coupon T-learner + preprocessing)
- models/uplift_metrics.json (per-arm AUUC, deciles, segments, config)
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from statsmodels.stats.proportion import confint_proportions_2indep


FEATURES_PATH = Path(
    "data/processed/customer_features_v3.parquet"
)
EXPERIMENT_PATH = Path(
    "data/synthetic/intervention_experiment.parquet"
)
MODEL_PATH = Path("models/uplift_tlearner.pkl")
METRICS_PATH = Path("models/uplift_metrics.json")

ARMS = (1, 2, 3)
ARM_NAMES = {
    1: "coupon",
    2: "free_delivery",
    3: "loyalty_points",
}
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
TEST_START = pd.Timestamp("2018-04-01")

SEED = 42
PRIMARY_ARM = 1


def load_uplift_frame():
    features = pd.read_parquet(FEATURES_PATH)
    experiment = pd.read_parquet(EXPERIMENT_PATH)
    df = experiment.merge(
        features[["customer_unique_id", "snapshot_date"] + FEATURES],
        on=["customer_unique_id", "snapshot_date"],
        how="left",
        validate="one_to_one",
    )
    assert len(df), "Empty uplift frame"
    # Gap features are legitimately NaN for ~97% of rows;
    # pipelines median-impute them (fit on train only).
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def fit_t_learner(train_arm):
    """Two logreg classifiers: one on treated, one on control rows."""
    treated = train_arm[train_arm["intervention_type"] != 0]
    control = train_arm[train_arm["intervention_type"] == 0]
    assert len(treated) and len(control), "Arm missing in train"
    mk = lambda: Pipeline(
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
    pipe_t, pipe_c = mk(), mk()
    pipe_t.fit(treated[FEATURES], treated["outcome"])
    pipe_c.fit(control[FEATURES], control["outcome"])
    return pipe_t, pipe_c


def predict_uplift(pipe_t, pipe_c, X):
    return (
        pipe_t.predict_proba(X)[:, 1]
        - pipe_c.predict_proba(X)[:, 1]
    )


def qini_curve(y, t, scores, n_points=100):
    """Qini curve: cumulative incremental gains ranked by score.

    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k) over the top-k rows.
    Returns xs (fraction ranked) and Q values.
    """
    order = np.argsort(-np.asarray(scores), kind="stable")
    y = np.asarray(y)[order]
    t = np.asarray(t)[order]
    n = len(y)
    ks = np.unique(
        np.clip((np.linspace(0, 1, n_points) * n).astype(int), 1, n)
    )
    qs = []
    for k in ks:
        yt = float(y[:k][t[:k] == 1].sum())
        yc = float(y[:k][t[:k] == 0].sum())
        nt = float((t[:k] == 1).sum())
        nc = float((t[:k] == 0).sum())
        qs.append(yt - yc * (nt / nc) if nc > 0 else yt)
    return (ks / n).tolist(), qs


def auuc(xs, qs):
    return float(np.trapezoid(qs, xs))


def uplift_deciles(y, t, uplift):
    """Observed treated-minus-control rate per predicted-uplift decile."""
    df = pd.DataFrame(
        {"y": np.asarray(y), "t": np.asarray(t),
         "u": np.asarray(uplift)}
    )
    df["decile"] = pd.qcut(df["u"], 10, duplicates="drop")
    rows = []
    for dec, part in df.groupby("decile", observed=False):
        yt = part[part["t"] == 1]["y"].to_numpy()
        yc = part[part["t"] == 0]["y"].to_numpy()
        if len(yt) == 0 or len(yc) == 0:
            continue
        lo, hi = confint_proportions_2indep(
            int(yt.sum()), len(yt), int(yc.sum()), len(yc),
            compare="diff", method="newcombe",
        )
        rows.append(
            {
                "decile": str(dec),
                "n": int(len(part)),
                "mean_predicted_uplift": float(part["u"].mean()),
                "observed_diff": float(yt.mean() - yc.mean()),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        )
    return rows


def uplift_segments(y, t, uplift, oracle):
    """Negative / low / medium / high predicted-uplift segments."""
    u = np.asarray(uplift)
    labels = np.full(len(u), "", dtype=object)
    labels[u < 0] = "negative"
    pos = u >= 0
    if pos.sum():
        cuts = np.quantile(u[pos], [1 / 3, 2 / 3])
        rest = np.full(len(u), "", dtype=object)
        rest[pos & (u < cuts[0])] = "low"
        rest[pos & (u >= cuts[0]) & (u < cuts[1])] = "medium"
        rest[pos & (u >= cuts[1])] = "high"
        labels[pos] = rest[pos]
    out = []
    for name in ("negative", "low", "medium", "high"):
        m = labels == name
        if not m.any():
            continue
        yt = np.asarray(y)[m & (np.asarray(t) == 1)]
        yc = np.asarray(y)[m & (np.asarray(t) == 0)]
        diff = (
            float(yt.mean() - yc.mean())
            if len(yt) and len(yc) else None
        )
        out.append(
            {
                "segment": name,
                "n": int(m.sum()),
                "mean_predicted_uplift": float(u[m].mean()),
                "observed_diff": diff,
                "oracle_mean": float(
                    np.asarray(oracle)[m].mean()
                ),
            }
        )
    return out


def evaluate_arm(df_test, arm, pipe_t, pipe_c):
    sub = df_test[df_test["intervention_type"].isin((0, arm))].copy()
    y = sub["outcome"].to_numpy()
    t = (sub["intervention_type"] == arm).astype(int).to_numpy()
    u = predict_uplift(pipe_t, pipe_c, sub[FEATURES])
    oracle = sub["true_cate"].to_numpy()

    xs_m, q_m = qini_curve(y, t, u)
    # Random targeting = straight line to the total incremental gain
    # (order-free, computed directly).
    total = q_m[-1] if len(q_m) else 0.0
    # Recompute totals directly (order-free).
    yt = float(y[t == 1].sum())
    yc = float(y[t == 0].sum())
    nt = float((t == 1).sum())
    nc = float((t == 0).sum())
    total = yt - yc * (nt / nc)
    q_r = [total * x for x in xs_m]
    xs_o, q_o = qini_curve(y, t, oracle)
    a_m, a_r, a_o = (
        auuc(xs_m, q_m),
        auuc(xs_m, q_r),
        auuc(xs_o, q_o),
    )
    qini_coef = (
        float((a_m - a_r) / (a_o - a_r)) if a_o != a_r else 0.0
    )
    spearman = float(
        pd.Series(u).corr(pd.Series(oracle), method="spearman")
    )
    return {
        "n_test": int(len(sub)),
        "n_treated": int((t == 1).sum()),
        "n_control": int((t == 0).sum()),
        "auuc_model": a_m,
        "auuc_random": a_r,
        "auuc_oracle": a_o,
        "qini_coefficient": qini_coef,
        "spearman_vs_oracle": spearman,
        "qini_curve": {
            "xs": xs_m,
            "model": q_m,
            "random": q_r,
            "oracle": q_o,
        },
        "deciles": uplift_deciles(y, t, u),
        "segments": uplift_segments(y, t, u, oracle),
    }


def main():
    df = load_uplift_frame()
    train = df[df["snapshot_date"] <= TRAIN_END]
    test = df[df["snapshot_date"] >= TEST_START]
    print(
        f"train rows: {len(train):,} "
        f"(to {TRAIN_END.date()}) | test rows: {len(test):,} "
        f"(from {TEST_START.date()})"
    )

    results = {}
    models = {}
    for arm in ARMS:
        tr = train[train["intervention_type"].isin((0, arm))]
        print(
            f"\n{ARM_NAMES[arm]}: train treated="
            f"{int((tr['intervention_type'] == arm).sum()):,} "
            f"control={int((tr['intervention_type'] == 0).sum()):,}"
        )
        pipe_t, pipe_c = fit_t_learner(tr)
        models[arm] = (pipe_t, pipe_c)
        ev = evaluate_arm(test, arm, pipe_t, pipe_c)
        results[str(arm)] = {"name": ARM_NAMES[arm], **ev}
        print(
            f"  AUUC model={ev['auuc_model']:.2f} random={ev['auuc_random']:.2f} "
            f"oracle={ev['auuc_oracle']:.2f} | "
            f"Qini coef={ev['qini_coefficient']:.3f} | "
            f"spearman(uplift, oracle)={ev['spearman_vs_oracle']:.3f}"
        )

    pipe_t, pipe_c = models[PRIMARY_ARM]
    joblib.dump(
        {
            "treated_pipeline": pipe_t,
            "control_pipeline": pipe_c,
            "arm": PRIMARY_ARM,
            "arm_name": ARM_NAMES[PRIMARY_ARM],
            "features": FEATURES,
            "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        MODEL_PATH,
    )
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {
                "arms": results,
                "primary_arm": PRIMARY_ARM,
                "train_end": str(TRAIN_END.date()),
                "test_start": str(TEST_START.date()),
                "seed": SEED,
                "note": (
                    "SYNTHETIC uplift estimates. "
                    "Not real-world effects."
                ),
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {METRICS_PATH}")

    reloaded = joblib.load(MODEL_PATH)
    u_a = predict_uplift(pipe_t, pipe_c, test[FEATURES][:1000])
    u_b = predict_uplift(
        reloaded["treated_pipeline"],
        reloaded["control_pipeline"],
        test[FEATURES][:1000],
    )
    assert np.allclose(u_a, u_b), "Reload mismatch"
    print("Reload check: saved T-learner reproduces uplift scores.")


if __name__ == "__main__":
    main()
