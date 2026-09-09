"""
Reject-inference credit scoring pipeline (synthetic data).

Adapted from `reject_inference_v3_stronger_bias_and_xgboost.py` (a Snowflake
notebook script) for standalone use in a Streamlit app: same data-generating
process, same three reject-inference methods (baseline / fuzzy augmentation /
parceling), same PSI and bootstrap-significance diagnostics -- just running
in memory instead of round-tripping through Snowflake tables.

Call `train_all()` once (the app caches it with st.cache_resource) to get an
`Artifacts` bundle with the population, every trained model, and every
evaluation table the app's tabs need.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "AGE",
    "INCOME",
    "CREDIT_HISTORY_MONTHS",
    "NUM_CREDIT_ACCOUNTS",
    "DEBT_RATIO",
    "NUM_LATE_PAYMENTS",
]

# Display labels + input-widget bounds, derived from the generating process
# in the source script (rng.integers(18, 75), lognormal(10.5, 0.5), etc.)
FEATURE_META = {
    "AGE": dict(label="Age (Years)", help="Applicant's age", min=18, max=90, default=35, step=1, kind="int"),
    "INCOME": dict(label="Annual Income ($)", help="Total annual income before tax", min=0, max=500_000, default=45_000, step=1_000, kind="int"),
    "CREDIT_HISTORY_MONTHS": dict(label="Credit History (Months)", help="Length of active credit history", min=0, max=240, default=60, step=1, kind="int"),
    "NUM_CREDIT_ACCOUNTS": dict(label="Open Credit Accounts", help="Valid credit lines (cards, loans)", min=0, max=20, default=3, step=1, kind="int"),
    "DEBT_RATIO": dict(label="Debt-to-Income Ratio", help="Decimal format (e.g., 0.30 for 30%)", min=0.0, max=1.0, default=0.28, step=0.01, kind="float"),
    "NUM_LATE_PAYMENTS": dict(label="Historical Late Payments", help="Count of past-due incidents", min=0, max=20, default=1, step=1, kind="int"),
}

SCORE_MIN, SCORE_MAX = 300, 850

RISK_BANDS = [
    (750, SCORE_MAX, "Excellent", "#22c55e"),
    (700, 750, "Good", "#84cc16"),
    (650, 700, "Fair", "#eab308"),
    (600, 650, "Poor", "#f97316"),
    (SCORE_MIN, 600, "High Risk", "#ef4444"),
]


def probability_to_score(p_bad: np.ndarray | float) -> np.ndarray | float:
    """Linear map from P(bad) to a 300-850 score -- higher score = lower risk."""
    p_bad = np.clip(p_bad, 0.0, 1.0)
    return SCORE_MIN + (1 - p_bad) * (SCORE_MAX - SCORE_MIN)


def risk_category(score: float) -> tuple[str, str]:
    for lo, hi, name, color in RISK_BANDS:
        if score >= lo:
            return name, color
    return RISK_BANDS[-1][2], RISK_BANDS[-1][3]


def psi(score_expected, score_actual, bins=10, w_actual=None) -> float:
    """Population Stability Index: how far score_actual's distribution has
    drifted from score_expected's, using score_expected's own decile edges
    as the reference grid. w_actual lets a reweighted sample (fuzzy
    augmentation, parceling) contribute per-row weights."""
    score_expected = np.asarray(score_expected)
    score_actual = np.asarray(score_actual)
    edges = np.quantile(score_expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    expected_counts, _ = np.histogram(score_expected, bins=edges)
    expected_pct = expected_counts / expected_counts.sum()
    if w_actual is None:
        actual_counts, _ = np.histogram(score_actual, bins=edges)
        actual_pct = actual_counts / actual_counts.sum()
    else:
        w_actual = np.asarray(w_actual)
        actual_pct = np.array([
            w_actual[(score_actual >= edges[i]) & (score_actual < edges[i + 1])].sum()
            for i in range(bins)
        ])
        actual_pct = actual_pct / actual_pct.sum()
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def bootstrap_auc_diff(y_true, score_a, score_b, n_boot=1000, seed=0) -> dict:
    """Paired bootstrap CI for AUC(score_a) - AUC(score_b) on the same
    population. If the 95% CI excludes 0, the difference is unlikely noise."""
    rng_b = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng_b.integers(0, n, n)
        diffs[i] = roc_auc_score(y_true[idx], score_a[idx]) - roc_auc_score(y_true[idx], score_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def generate_population(n_applicants: int = 7000, seed: int = 789) -> pd.DataFrame:
    """Same data-generating process as Cell 19 of the source script: a
    synthetic applicant population with a hidden TRUE_BAD outcome and an
    observed STATUS (approved/rejected) driven by a *harder*, more
    deterministic selection rule than a naive baseline (noise sd 0.25)."""
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n_applicants)
    income = rng.lognormal(10.5, 0.5, n_applicants).round(2)
    credit_history_months = rng.integers(0, 240, n_applicants)
    num_credit_accounts = rng.poisson(3, n_applicants)
    debt_ratio = rng.beta(2, 5, n_applicants).round(3)
    num_late_payments = rng.poisson(1.2, n_applicants)

    df = pd.DataFrame({
        "AGE": age,
        "INCOME": income,
        "CREDIT_HISTORY_MONTHS": credit_history_months,
        "NUM_CREDIT_ACCOUNTS": num_credit_accounts,
        "DEBT_RATIO": debt_ratio,
        "NUM_LATE_PAYMENTS": num_late_payments,
    })

    logit = (
        0.5
        - 0.02 * (df["AGE"] - 45).abs()
        - 0.00002 * df["INCOME"]
        - 0.01 * df["CREDIT_HISTORY_MONTHS"]
        + 0.05 * df["NUM_CREDIT_ACCOUNTS"]
        + 2.5 * df["DEBT_RATIO"]
        + 0.35 * df["NUM_LATE_PAYMENTS"]
    )
    true_pd = 1 / (1 + np.exp(-logit))
    df["TRUE_BAD"] = rng.binomial(1, true_pd)

    approval_score = (
        0.02 * df["CREDIT_HISTORY_MONTHS"]
        - 3.0 * df["DEBT_RATIO"]
        - 0.6 * df["NUM_LATE_PAYMENTS"]
        + 0.00003 * df["INCOME"]
        + rng.normal(0, 0.25, n_applicants)
    )
    cutoff = np.quantile(approval_score, 1 - 5000 / 7000)
    df["STATUS"] = np.where(approval_score >= cutoff, "approved", "rejected")
    df["TARGET"] = np.where(df["STATUS"] == "approved", df["TRUE_BAD"], np.nan)
    df.insert(0, "APPLICANT_ID", range(1, len(df) + 1))
    return df


@dataclass
class Artifacts:
    df: pd.DataFrame
    approved: pd.DataFrame
    rejected: pd.DataFrame

    # Logistic-regression models + their scalers, one per method
    lr_baseline: LogisticRegression
    lr_fuzzy: LogisticRegression
    lr_parcel: LogisticRegression
    scaler_baseline: StandardScaler
    scaler_fuzzy: StandardScaler
    scaler_parcel: StandardScaler

    # XGBoost models, one per method (trees -> no scaling needed)
    xgb_baseline: xgb.XGBClassifier
    xgb_fuzzy: xgb.XGBClassifier
    xgb_parcel: xgb.XGBClassifier
    xgb_models: dict          # method_name -> XGBClassifier, for generic lookup
    deployed_method: str       # method (of xgb_models) with the best true_population_auc
    deployed_model: xgb.XGBClassifier  # = xgb_models[deployed_method]

    # Diagnostics
    psi_values: dict
    lr_summary: pd.DataFrame
    xgb_summary: pd.DataFrame
    bootstrap_results: dict
    selection_bias: pd.DataFrame
    score_shift: pd.DataFrame
    deployed_feature_importance: pd.DataFrame

    metrics: dict = field(default_factory=dict)


def train_all(n_applicants: int = 7000, seed: int = 789) -> Artifacts:
    df = generate_population(n_applicants, seed)
    approved = df[df["STATUS"] == "approved"].copy()
    rejected = df[df["STATUS"] == "rejected"].copy()
    approved["BAD"] = approved["TARGET"]

    X, y = approved[FEATURES], approved["BAD"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ---- Baseline: logistic regression, accepts-only, no correction ----
    scaler_base = StandardScaler().fit(X_train)
    lr_baseline = LogisticRegression(max_iter=1000).fit(scaler_base.transform(X_train), y_train)
    baseline_auc = roc_auc_score(y_test, lr_baseline.predict_proba(scaler_base.transform(X_test))[:, 1])

    X_rej = scaler_base.transform(rejected[FEATURES])
    p_bad_rej = lr_baseline.predict_proba(X_rej)[:, 1]

    # ---- Fuzzy augmentation ----
    rej_good = rejected[FEATURES].copy(); rej_good["BAD"] = 0; rej_good["weight"] = 1 - p_bad_rej
    rej_bad = rejected[FEATURES].copy(); rej_bad["BAD"] = 1; rej_bad["weight"] = p_bad_rej
    approved_aug = approved[FEATURES + ["BAD"]].copy(); approved_aug["weight"] = 1.0
    augmented = pd.concat([approved_aug, rej_good, rej_bad], ignore_index=True)

    scaler_fuzzy = StandardScaler().fit(augmented[FEATURES])
    lr_fuzzy = LogisticRegression(max_iter=1000).fit(
        scaler_fuzzy.transform(augmented[FEATURES]), augmented["BAD"], sample_weight=augmented["weight"]
    )
    fuzzy_auc = roc_auc_score(y_test, lr_fuzzy.predict_proba(scaler_fuzzy.transform(X_test))[:, 1])

    # ---- Parceling ----
    inflation_factor = p_bad_rej.mean() / y.mean()
    n_bins = 10
    score_approved_base = lr_baseline.predict_proba(scaler_base.transform(approved[FEATURES]))[:, 1]
    score_rejected_base = lr_baseline.predict_proba(scaler_base.transform(rejected[FEATURES]))[:, 1]
    combo_scores = np.concatenate([score_approved_base, score_rejected_base])
    bin_edges = np.quantile(combo_scores, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    approved_bin = np.digitize(score_approved_base, bin_edges) - 1
    bin_bad_rate = pd.Series(approved["BAD"].values).groupby(approved_bin).mean()
    reject_bin = np.digitize(score_rejected_base, bin_edges) - 1
    reject_base_rate = pd.Series(reject_bin).map(bin_bad_rate).fillna(bin_bad_rate.mean()).values
    reject_rate_inflated = np.clip(reject_base_rate * inflation_factor, 0, 1)

    rej_good_p = rejected[FEATURES].copy(); rej_good_p["BAD"] = 0; rej_good_p["weight"] = 1 - reject_rate_inflated
    rej_bad_p = rejected[FEATURES].copy(); rej_bad_p["BAD"] = 1; rej_bad_p["weight"] = reject_rate_inflated
    augmented_p = pd.concat([approved_aug, rej_good_p, rej_bad_p], ignore_index=True)
    scaler_parcel = StandardScaler().fit(augmented_p[FEATURES])
    lr_parcel = LogisticRegression(max_iter=1000).fit(
        scaler_parcel.transform(augmented_p[FEATURES]), augmented_p["BAD"], sample_weight=augmented_p["weight"]
    )
    parcel_auc = roc_auc_score(y_test, lr_parcel.predict_proba(scaler_parcel.transform(X_test))[:, 1])

    # ---- PSI (training sample vs full population), per LR method ----
    score_full_baseline = lr_baseline.predict_proba(scaler_base.transform(df[FEATURES]))[:, 1]
    score_train_baseline = lr_baseline.predict_proba(scaler_base.transform(approved[FEATURES]))[:, 1]
    psi_baseline = psi(score_full_baseline, score_train_baseline)

    score_full_fuzzy = lr_fuzzy.predict_proba(scaler_fuzzy.transform(df[FEATURES]))[:, 1]
    score_train_fuzzy = lr_fuzzy.predict_proba(scaler_fuzzy.transform(augmented[FEATURES]))[:, 1]
    psi_fuzzy = psi(score_full_fuzzy, score_train_fuzzy, w_actual=augmented["weight"].values)

    score_full_parcel = lr_parcel.predict_proba(scaler_parcel.transform(df[FEATURES]))[:, 1]
    score_train_parcel = lr_parcel.predict_proba(scaler_parcel.transform(augmented_p[FEATURES]))[:, 1]
    psi_parcel = psi(score_full_parcel, score_train_parcel, w_actual=augmented_p["weight"].values)

    psi_values = {"baseline_accepts_only": psi_baseline, "fuzzy_augmentation": psi_fuzzy, "parceling": psi_parcel}

    # ---- Ground-truth validation (only possible because this is synthetic
    # data -- a real lender never observes TRUE_BAD for declined applicants) ----
    y_true_full = df["TRUE_BAD"]
    score_baseline_full = score_full_baseline
    score_fuzzy_full = score_full_fuzzy
    score_parcel_full = score_full_parcel
    auc_true_baseline = roc_auc_score(y_true_full, score_baseline_full)
    auc_true_fuzzy = roc_auc_score(y_true_full, score_fuzzy_full)
    auc_true_parcel = roc_auc_score(y_true_full, score_parcel_full)

    lr_summary = pd.DataFrame({
        "method": ["baseline_accepts_only", "fuzzy_augmentation", "parceling"],
        "model_class": ["logistic_regression"] * 3,
        "accepts_test_auc": [baseline_auc, fuzzy_auc, parcel_auc],
        "true_population_auc": [auc_true_baseline, auc_true_fuzzy, auc_true_parcel],
        "psi_train_vs_full_population": [psi_baseline, psi_fuzzy, psi_parcel],
    })

    # ---- XGBoost: same three methods, same sample weights, no scaling ----
    xgb_baseline = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="auc", random_state=42)
    xgb_baseline.fit(X_train, y_train)
    xgb_baseline_auc = roc_auc_score(y_test, xgb_baseline.predict_proba(X_test)[:, 1])

    xgb_fuzzy = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="auc", random_state=42)
    xgb_fuzzy.fit(augmented[FEATURES], augmented["BAD"], sample_weight=augmented["weight"])
    xgb_fuzzy_auc = roc_auc_score(y_test, xgb_fuzzy.predict_proba(X_test)[:, 1])

    xgb_parcel = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="auc", random_state=42)
    xgb_parcel.fit(augmented_p[FEATURES], augmented_p["BAD"], sample_weight=augmented_p["weight"])
    xgb_parcel_auc = roc_auc_score(y_test, xgb_parcel.predict_proba(X_test)[:, 1])

    score_baseline_full_xgb = xgb_baseline.predict_proba(df[FEATURES])[:, 1]
    score_fuzzy_full_xgb = xgb_fuzzy.predict_proba(df[FEATURES])[:, 1]
    score_parcel_full_xgb = xgb_parcel.predict_proba(df[FEATURES])[:, 1]
    auc_true_baseline_xgb = roc_auc_score(y_true_full, score_baseline_full_xgb)
    auc_true_fuzzy_xgb = roc_auc_score(y_true_full, score_fuzzy_full_xgb)
    auc_true_parcel_xgb = roc_auc_score(y_true_full, score_parcel_full_xgb)

    brier_baseline_xgb = brier_score_loss(y_true_full, score_baseline_full_xgb)
    brier_fuzzy_xgb = brier_score_loss(y_true_full, score_fuzzy_full_xgb)
    brier_parcel_xgb = brier_score_loss(y_true_full, score_parcel_full_xgb)

    xgb_summary = pd.DataFrame({
        "method": ["baseline_accepts_only", "fuzzy_augmentation", "parceling"],
        "model_class": ["xgboost"] * 3,
        "accepts_test_auc": [xgb_baseline_auc, xgb_fuzzy_auc, xgb_parcel_auc],
        "true_population_auc": [auc_true_baseline_xgb, auc_true_fuzzy_xgb, auc_true_parcel_xgb],
        "true_population_brier": [brier_baseline_xgb, brier_fuzzy_xgb, brier_parcel_xgb],
    })

    bootstrap_results = {
        "lr_fuzzy_vs_baseline": bootstrap_auc_diff(y_true_full, score_fuzzy_full, score_baseline_full),
        "lr_parcel_vs_baseline": bootstrap_auc_diff(y_true_full, score_parcel_full, score_baseline_full),
        "xgb_fuzzy_vs_baseline": bootstrap_auc_diff(y_true_full, score_fuzzy_full_xgb, score_baseline_full_xgb),
        "xgb_parcel_vs_baseline": bootstrap_auc_diff(y_true_full, score_parcel_full_xgb, score_baseline_full_xgb),
        "xgb_vs_lr_baseline": bootstrap_auc_diff(y_true_full, score_baseline_full_xgb, score_baseline_full),
    }

    # ---- Selection bias: approved vs rejected, by feature ----
    rows = []
    for f in FEATURES:
        a_mean, r_mean = approved[f].mean(), rejected[f].mean()
        rows.append({
            "feature": f,
            "approved_mean": a_mean,
            "rejected_mean": r_mean,
            "pct_diff": (r_mean - a_mean) / a_mean * 100 if a_mean else np.nan,
        })
    selection_bias = pd.DataFrame(rows)

    # ---- Score shift by reject-inference method: how well each method's
    # predicted bad-rate on rejects matches the (normally hidden) truth ----
    true_bad_rate_rejected = rejected["TRUE_BAD"].mean()
    declines_called_bad = {
        "baseline_accepts_only": float((p_bad_rej >= 0.5).mean()),
        "fuzzy_augmentation": float((lr_fuzzy.predict_proba(scaler_fuzzy.transform(rejected[FEATURES]))[:, 1] >= 0.5).mean()),
        "parceling": float((reject_rate_inflated >= 0.5).mean()),
    }
    score_shift = pd.DataFrame([
        {
            "method": m,
            "declines_called_bad": declines_called_bad[m],
            "actually_bad": true_bad_rate_rejected,
            "error_pp": (declines_called_bad[m] - true_bad_rate_rejected) * 100,
            "score_shift_psi": psi_values[m],
        }
        for m in ["baseline_accepts_only", "fuzzy_augmentation", "parceling"]
    ])

    # ---- Deploy the XGBoost method with the best true-population AUC. This
    # is a data-driven pick, not a fixed choice -- see xgb_summary above:
    # in this run, aggressive parceling actually overcorrects and *hurts*
    # true AUC relative to baseline, so it should not automatically win. ----
    xgb_models = {"baseline_accepts_only": xgb_baseline, "fuzzy_augmentation": xgb_fuzzy, "parceling": xgb_parcel}
    best_row = xgb_summary.loc[xgb_summary["true_population_auc"].idxmax()]
    deployed_method = best_row["method"]
    deployed_model = xgb_models[deployed_method]

    booster = deployed_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp_rows = [{"feature": f, "gain": gain.get(f, 0.0)} for f in FEATURES]
    deployed_feature_importance = pd.DataFrame(imp_rows).sort_values("gain", ascending=False).reset_index(drop=True)

    metrics = {
        "n_applicants": n_applicants,
        "n_approved": len(approved),
        "n_rejected": len(rejected),
        "good_rate_approved": float(1 - y.mean()),
        "deployed_true_auc": float(best_row["true_population_auc"]),
    }

    return Artifacts(
        df=df, approved=approved, rejected=rejected,
        lr_baseline=lr_baseline, lr_fuzzy=lr_fuzzy, lr_parcel=lr_parcel,
        scaler_baseline=scaler_base, scaler_fuzzy=scaler_fuzzy, scaler_parcel=scaler_parcel,
        xgb_baseline=xgb_baseline, xgb_fuzzy=xgb_fuzzy, xgb_parcel=xgb_parcel,
        xgb_models=xgb_models, deployed_method=deployed_method, deployed_model=deployed_model,
        psi_values=psi_values, lr_summary=lr_summary, xgb_summary=xgb_summary,
        bootstrap_results=bootstrap_results, selection_bias=selection_bias,
        score_shift=score_shift, deployed_feature_importance=deployed_feature_importance,
        metrics=metrics,
    )


def score_applicant(artifacts: Artifacts, inputs: dict) -> dict:
    """Score one applicant with every XGBoost method (baseline / fuzzy /
    parceling) plus the LR-parceling model for an apples-to-apples
    cross-check, and add a "deployed" entry pointing at whichever XGBoost
    method had the best true-population AUC in training. `inputs` maps
    FEATURES -> value."""
    row = pd.DataFrame([{f: inputs[f] for f in FEATURES}])

    results = {}
    for method_name, mdl in artifacts.xgb_models.items():
        results[f"xgb_{method_name}"] = float(mdl.predict_proba(row[FEATURES])[:, 1][0])
    results["lr_parcel"] = float(
        artifacts.lr_parcel.predict_proba(artifacts.scaler_parcel.transform(row[FEATURES]))[:, 1][0]
    )

    out = {k: {"p_bad": v, "score": probability_to_score(v)} for k, v in results.items()}
    out["deployed"] = out[f"xgb_{artifacts.deployed_method}"]
    return out


def explain_applicant(artifacts: Artifacts, inputs: dict) -> pd.DataFrame:
    """Per-feature SHAP-style contribution (in log-odds/margin units) from
    the deployed model, via XGBoost's native pred_contribs -- no extra
    `shap` dependency needed."""
    row = pd.DataFrame([{f: inputs[f] for f in FEATURES}])
    dmat = xgb.DMatrix(row[FEATURES], feature_names=FEATURES)
    contribs = artifacts.deployed_model.get_booster().predict(dmat, pred_contribs=True)[0]
    contrib_df = pd.DataFrame({
        "feature": FEATURES + ["base_value"],
        "contribution": contribs,
    })
    contrib_df["label"] = contrib_df["feature"].map(lambda f: FEATURE_META.get(f, {}).get("label", f))
    return contrib_df


def whatif_score(artifacts: Artifacts, inputs: dict, feature: str, new_value) -> float:
    """Recompute the deployed model's score with one feature changed --
    used to generate concrete, non-hallucinated improvement estimates."""
    new_inputs = dict(inputs)
    new_inputs[feature] = new_value
    row = pd.DataFrame([{f: new_inputs[f] for f in FEATURES}])
    p_bad = float(artifacts.deployed_model.predict_proba(row[FEATURES])[:, 1][0])
    return float(probability_to_score(p_bad))
