"""
CreditAI-style Streamlit app, built on a real reject-inference + XGBoost
credit scoring pipeline (see model.py, adapted from
reject_inference_v3_stronger_bias_and_xgboost.py).

Layout mirrors https://credit-ai-seven.vercel.app/ (Credit Scoring / Data
Analysis / PSI Diagnostics / About), with an "AI Pipeline"-style tab
replaced by a "Model Insights" tab that shows the actual LR-vs-XGBoost,
baseline-vs-reject-inference comparison the source notebook produced --
real model diagnostics instead of a simulated multi-agent log, and no
OpenAI key required.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import model
from model import FEATURES, FEATURE_META, SCORE_MIN, SCORE_MAX, RISK_BANDS

st.set_page_config(page_title="CreditAI", page_icon="💳", layout="wide")

# ----------------------------------------------------------------------------
# Styling -- dark theme + card look, in the spirit of the reference app
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.credit-header { text-align: center; padding: 0.5rem 0 1.5rem 0; }
.credit-header h1 {
    background: linear-gradient(90deg, #818cf8, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 2.4rem; margin-bottom: 0.2rem;
}
.credit-header p { color: #94a3b8; font-size: 1.05rem; }
div[data-testid="stMetric"] {
    background: rgba(148,163,184,0.08);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 10px; padding: 0.8rem 1rem;
}
.risk-badge {
    display: inline-block; padding: 0.35rem 0.9rem; border-radius: 999px;
    font-weight: 700; font-size: 0.95rem; color: white;
}
.explain-box {
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px; padding: 1rem 1.2rem; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="credit-header"><h1>💳 CreditAI</h1>'
    '<p>Reject-Inference-Corrected Credit Scoring &amp; Risk Analysis</p></div>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading CreditAI model pipeline...")
def get_artifacts():
    return model.get_or_train_artifacts()


artifacts = get_artifacts()
DEPLOYED_LABELS = {"baseline_accepts_only": "no correction", "fuzzy_augmentation": "fuzzy augmentation", "parceling": "parceling"}
deployed_label = DEPLOYED_LABELS[artifacts.deployed_method]

tab_score, tab_data, tab_psi, tab_insights, tab_about = st.tabs(
    ["💳 Credit Scoring", "📊 Data Analysis", "📈 PSI Diagnostics", "🔍 Model Insights", "ℹ️ About"]
)

# ============================================================================
# TAB 1 -- Credit Scoring
# ============================================================================
with tab_score:
    col_form, col_result = st.columns([1, 1.3], gap="large")

    with col_form:
        st.subheader("Applicant Information")
        with st.form("scoring_form"):
            inputs = {}
            for f in FEATURES:
                meta = FEATURE_META[f]
                if meta["kind"] == "int":
                    inputs[f] = st.number_input(
                        meta["label"], min_value=meta["min"], max_value=meta["max"],
                        value=meta["default"], step=meta["step"], help=meta["help"],
                    )
                else:
                    inputs[f] = st.number_input(
                        meta["label"], min_value=meta["min"], max_value=meta["max"],
                        value=meta["default"], step=meta["step"], format="%.2f", help=meta["help"],
                    )
            submitted = st.form_submit_button("Analyze Credit Risk", use_container_width=True, type="primary")

    with col_result:
        st.subheader("Score")
        if not submitted:
            st.info("Enter applicant details and click **Analyze Credit Risk**.")
        else:
            all_scores = model.score_applicant(artifacts, inputs)
            deployed = all_scores["deployed"]
            score = deployed["score"]
            p_bad = deployed["p_bad"]
            approval_prob = (1 - p_bad) * 100
            category, color = model.risk_category(score)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [SCORE_MIN, SCORE_MAX], "tickwidth": 1},
                    "bar": {"color": color},
                    "steps": [{"range": [lo, hi], "color": bg} for lo, hi, _, _, bg in RISK_BANDS],
                },
                domain={"x": [0, 1], "y": [0, 1]},
            ))
            gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(gauge, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Credit Score", f"{score:.0f}", help="300 (High Risk) — 850 (Excellent)")
            m2.metric("Approval Probability", f"{approval_prob:.1f}%")
            with m3:
                st.markdown(f"**Risk Category**<br><span class='risk-badge' style='background:{color}'>{category}</span>",
                             unsafe_allow_html=True)

            # ---- Feature contributions (XGBoost native SHAP-style contribs) ----
            st.markdown("#### What drove this score")
            contrib = model.explain_applicant(artifacts, inputs)
            contrib_features = contrib[contrib["feature"] != "base_value"].copy()
            contrib_features = contrib_features.reindex(
                contrib_features["contribution"].abs().sort_values(ascending=True).index
            )
            fig_contrib = px.bar(
                contrib_features, x="contribution", y="label", orientation="h",
                color=contrib_features["contribution"] > 0,
                color_discrete_map={True: "#ef4444", False: "#22c55e"},
                labels={"contribution": "Impact on risk (log-odds of default)", "label": ""},
            )
            fig_contrib.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_contrib, use_container_width=True)
            st.caption("Red bars push risk up (score down); green bars push risk down (score up). "
                       "Computed directly from the deployed model's own math (XGBoost `pred_contribs`) — not an LLM guess.")

            top_risk = contrib_features.sort_values("contribution", ascending=False).iloc[-1] \
                if (contrib_features["contribution"] > 0).any() else None
            worst = contrib_features.loc[contrib_features["contribution"].idxmax()]
            best = contrib_features.loc[contrib_features["contribution"].idxmin()]
            st.markdown(
                f"<div class='explain-box'>Biggest risk driver: <b>{worst['label']}</b>. "
                f"Biggest positive factor: <b>{best['label']}</b>.</div>",
                unsafe_allow_html=True,
            )

            # ---- Improvement tips: real recomputation, not canned text ----
            with st.expander("Improvement tips (recomputed, not generic advice)"):
                tips = []
                if inputs["DEBT_RATIO"] > 0.15:
                    new_score = model.whatif_score(artifacts, inputs, "DEBT_RATIO", max(0.10, inputs["DEBT_RATIO"] - 0.10))
                    tips.append(f"Lowering **Debt-to-Income Ratio** by 0.10 (to {max(0.10, inputs['DEBT_RATIO']-0.10):.2f}) "
                                f"moves the estimated score to **{new_score:.0f}** ({new_score - score:+.0f}).")
                if inputs["NUM_LATE_PAYMENTS"] > 0:
                    new_score = model.whatif_score(artifacts, inputs, "NUM_LATE_PAYMENTS", max(0, inputs["NUM_LATE_PAYMENTS"] - 1))
                    tips.append(f"One fewer **Late Payment** moves the estimated score to **{new_score:.0f}** "
                                f"({new_score - score:+.0f}).")
                if inputs["CREDIT_HISTORY_MONTHS"] < 120:
                    new_score = model.whatif_score(artifacts, inputs, "CREDIT_HISTORY_MONTHS", inputs["CREDIT_HISTORY_MONTHS"] + 24)
                    tips.append(f"24 more months of **Credit History** moves the estimated score to **{new_score:.0f}** "
                                f"({new_score - score:+.0f}).")
                if tips:
                    for t in tips:
                        st.markdown(f"- {t}")
                else:
                    st.markdown("No high-impact single-feature changes found for this profile.")

            # ---- Cross-check across methods, ties to PSI Diagnostics tab ----
            with st.expander("Cross-check: score by method"):
                method_display = {
                    "baseline_accepts_only": "XGBoost, no reject-inference correction",
                    "fuzzy_augmentation": "XGBoost + fuzzy augmentation",
                    "parceling": "XGBoost + parceling",
                }
                rows = []
                for m, label in method_display.items():
                    key = f"xgb_{m}"
                    tag = "  (deployed)" if m == artifacts.deployed_method else ""
                    rows.append({"Model": label + tag, "P(bad)": all_scores[key]["p_bad"], "Score": all_scores[key]["score"]})
                rows.append({"Model": "Logistic regression + parceling", "P(bad)": all_scores["lr_parcel"]["p_bad"], "Score": all_scores["lr_parcel"]["score"]})
                cross = pd.DataFrame(rows)
                cross["P(bad)"] = (cross["P(bad)"] * 100).round(1).astype(str) + "%"
                cross["Score"] = cross["Score"].round(0).astype(int)
                st.dataframe(cross, hide_index=True, use_container_width=True)
                st.caption("Method choice changes the number — see the PSI Diagnostics and Model Insights tabs for why "
                           f"**{method_display[artifacts.deployed_method]}** is deployed here (best true-population AUC).")

# ============================================================================
# TAB 2 -- Data Analysis
# ============================================================================
with tab_data:
    df = artifacts.df
    st.subheader("Applicant Population Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants", f"{artifacts.metrics['n_applicants']:,}")
    c2.metric("Approved", f"{artifacts.metrics['n_approved']:,}")
    c3.metric("Rejected", f"{artifacts.metrics['n_rejected']:,}")
    c4.metric("Good Rate (Approved)", f"{artifacts.metrics['good_rate_approved']*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Application Status**")
        status_counts = df["STATUS"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.pie(status_counts, names="Status", values="Count", hole=0.5,
                     color="Status", color_discrete_map={"approved": "#22c55e", "rejected": "#ef4444"})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Credit Worthiness (Approved)**")
        approved = artifacts.approved
        good_bad = approved["BAD"].map({0: "Good", 1: "Bad"}).value_counts().reset_index()
        good_bad.columns = ["Outcome", "Count"]
        fig = px.pie(good_bad, names="Outcome", values="Count", hole=0.5,
                     color="Outcome", color_discrete_map={"Good": "#22c55e", "Bad": "#ef4444"})
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Income Distribution by Application Status**")
    fig = px.histogram(df, x="INCOME", color="STATUS", nbins=50, barmode="overlay", opacity=0.65,
                        color_discrete_map={"approved": "#22c55e", "rejected": "#ef4444"})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    def approval_rate_by_bin(frame, col, bins, labels=None):
        binned = pd.cut(frame[col], bins=bins, labels=labels)
        out = frame.groupby(binned, observed=True)["STATUS"].apply(lambda s: (s == "approved").mean() * 100)
        out = out.reset_index()
        out.columns = [col, "approval_rate"]
        return out

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Credit History Impact**")
        bins = [0, 24, 60, 120, 180, 240]
        labels = ["0-2y", "2-5y", "5-10y", "10-15y", "15-20y"]
        agg = approval_rate_by_bin(df, "CREDIT_HISTORY_MONTHS", bins, labels)
        fig = px.bar(agg, x="CREDIT_HISTORY_MONTHS", y="approval_rate",
                     labels={"CREDIT_HISTORY_MONTHS": "Credit History", "approval_rate": "Approval Rate (%)"},
                     color_discrete_sequence=["#6366f1"])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("**Debt Ratio vs. Approval**")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        labels = ["0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ".5+"]
        agg = approval_rate_by_bin(df, "DEBT_RATIO", bins, labels)
        fig = px.bar(agg, x="DEBT_RATIO", y="approval_rate",
                     labels={"DEBT_RATIO": "Debt-to-Income Ratio", "approval_rate": "Approval Rate (%)"},
                     color_discrete_sequence=["#f97316"])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Late Payments vs. Approval**")
    agg = df.groupby("NUM_LATE_PAYMENTS", observed=True)["STATUS"].apply(lambda s: (s == "approved").mean() * 100).reset_index()
    agg.columns = ["Late Payments", "Approval Rate (%)"]
    agg = agg[agg["Late Payments"] <= 8]
    fig = px.bar(agg, x="Late Payments", y="Approval Rate (%)", color_discrete_sequence=["#38bdf8"])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3 -- PSI Diagnostics
# ============================================================================
with tab_psi:
    st.subheader("Population Stability Index (PSI)")
    st.markdown(
        "PSI measures how far a training sample's score distribution has drifted from the "
        "full population's. Thresholds: **< 0.10** stable · **0.10 – 0.25** moderate shift · **≥ 0.25** severe shift."
    )

    psi_df = pd.DataFrame({
        "method": list(artifacts.psi_values.keys()),
        "psi": list(artifacts.psi_values.values()),
    })
    fig = go.Figure()
    fig.add_bar(x=psi_df["method"], y=psi_df["psi"], marker_color=["#ef4444", "#22c55e", "#22c55e"])
    fig.add_hline(y=0.10, line_dash="dash", line_color="#eab308", annotation_text="0.10 moderate")
    fig.add_hline(y=0.25, line_dash="dash", line_color="#ef4444", annotation_text="0.25 severe")
    fig.update_layout(height=340, yaxis_title="PSI (log scale not applied)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    worst_feature_note = artifacts.selection_bias.reindex(
        artifacts.selection_bias["pct_diff"].abs().sort_values(ascending=False).index
    ).iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Worst raw-feature shift", FEATURE_META[worst_feature_note["feature"]]["label"],
              f"{worst_feature_note['pct_diff']:+.1f}% (rej. vs appr.)")
    severe = int((psi_df["psi"] >= 0.25).sum())
    k2.metric("Methods in severe band", f"{severe} / {len(psi_df)}")
    k3.metric("Largest score shift", f"{psi_df['psi'].max():.2f}", psi_df.loc[psi_df['psi'].idxmax(), 'method'])
    k4.metric("Applicants scored", f"{artifacts.metrics['n_applicants']:,}")

    st.markdown("#### Selection bias — approved vs. rejected, by feature")
    st.dataframe(
        artifacts.selection_bias.style.format({"approved_mean": "{:.2f}", "rejected_mean": "{:.2f}", "pct_diff": "{:+.1f}%"}),
        hide_index=True, use_container_width=True,
    )
    st.caption("Credit history and late payments show the largest gaps — the model class that can least ignore "
               "that gap (and correct for it) should benefit most from reject inference. See Model Insights.")

    st.markdown("#### Score shift — by reject-inference method")
    st.dataframe(
        artifacts.score_shift.style.format({
            "declines_called_bad": "{:.1%}", "actually_bad": "{:.1%}",
            "error_pp": "{:+.1f}", "score_shift_psi": "{:.2f}",
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption("`actually_bad` uses the synthetic ground truth (normally unobservable for real declined applicants) "
               "so the error column shows exactly how far each method's predicted bad-rate on declines misses the truth.")

    st.markdown("#### Logistic-regression method comparison (accepts-only test set vs. true population)")
    st.dataframe(
        artifacts.lr_summary.style.format({
            "accepts_test_auc": "{:.4f}", "true_population_auc": "{:.4f}", "psi_train_vs_full_population": "{:.4f}",
        }),
        hide_index=True, use_container_width=True,
    )

    with st.expander("Bootstrapped AUC significance (logistic regression, 95% CI)"):
        for k in ["lr_fuzzy_vs_baseline", "lr_parcel_vs_baseline"]:
            r = artifacts.bootstrap_results[k]
            verdict = "✅ statistically significant" if r["significant"] else "— within sampling noise"
            st.write(f"**{k}**: mean diff {r['mean_diff']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] — {verdict}")

# ============================================================================
# TAB 4 -- Model Insights
# ============================================================================
with tab_insights:
    st.subheader("Model Class Comparison: Logistic Regression vs. XGBoost")
    st.markdown("Same reject-inference logic (fuzzy augmentation / parceling weights) applied to both a linear "
                "model and a tree ensemble, evaluated against the (synthetic) true population — the only fair "
                "way to tell whether a fancier model class is actually buying anything here.")

    combined = pd.concat([artifacts.lr_summary.assign(true_population_brier=np.nan), artifacts.xgb_summary], ignore_index=True)
    fig = px.bar(
        combined, x="method", y="true_population_auc", color="model_class", barmode="group",
        labels={"true_population_auc": "True-population AUC", "method": "", "model_class": "Model"},
        color_discrete_map={"logistic_regression": "#818cf8", "xgboost": "#38bdf8"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        artifacts.xgb_summary.style.format({
            "accepts_test_auc": "{:.4f}", "true_population_auc": "{:.4f}", "true_population_brier": "{:.4f}",
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption("Brier score: lower is better-calibrated (0 = perfect). AUC alone can hide a model that ranks well "
               "but is overconfident.")

    with st.expander("Bootstrapped AUC significance (XGBoost, 95% CI)"):
        for k in ["xgb_fuzzy_vs_baseline", "xgb_parcel_vs_baseline", "xgb_vs_lr_baseline"]:
            r = artifacts.bootstrap_results[k]
            verdict = "✅ statistically significant" if r["significant"] else "— within sampling noise"
            st.write(f"**{k}**: mean diff {r['mean_diff']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] — {verdict}")

    st.markdown(f"#### Deployed model (XGBoost + {deployed_label}) — feature importance")
    fi = artifacts.deployed_feature_importance.copy()
    fi["label"] = fi["feature"].map(lambda f: FEATURE_META[f]["label"])
    fig = px.bar(fi.sort_values("gain"), x="gain", y="label", orientation="h",
                 labels={"gain": "Average gain", "label": ""}, color_discrete_sequence=["#6366f1"])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Deployed for live scoring (Credit Scoring tab): XGBoost + {deployed_label}** — "
            f"chosen automatically as the XGBoost method with the best true-population AUC "
            f"({artifacts.metrics['deployed_true_auc']:.3f}) in the table above. "
            f"Retrained each session on a fresh synthetic population (seed 789, 7,000 applicants).")

# ============================================================================
# TAB 5 -- About
# ============================================================================
with tab_about:
    st.subheader("About CreditAI")
    st.markdown(f"""
This app demonstrates an end-to-end **reject-inference credit scoring pipeline** on synthetic data —
adapted directly from a working notebook script
(`reject_inference_v3_stronger_bias_and_xgboost.py`), not a toy re-skin of the UI alone.

**Data.** {artifacts.metrics['n_applicants']:,} synthetic applicants are generated from six features
(age, income, credit history, open accounts, debt ratio, late payments). A hidden `TRUE_BAD` outcome
is drawn from a logistic risk model; an `APPROVED` decision is then drawn from a **separate**,
more deterministic rule (low noise, sd = 0.25) correlated with the same risk features — a strong
missing-not-at-random selection problem, since only approved applicants' outcomes are ever observed
in the training data (5,000 approved / 2,000 rejected).

**The problem this creates.** A model trained only on approved applicants sees a systematically
different population than the one it will score in production. That gap shows up two ways: as
**selection bias** (approved vs. rejected applicants have different feature distributions — see
*PSI Diagnostics*) and as degraded discrimination on the true population.

**Reject inference.** Two standard corrections are implemented and compared against a
no-correction baseline:
- **Fuzzy augmentation** — every rejected applicant is duplicated into a "good" and "bad" copy,
  weighted by the baseline model's own predicted P(bad).
- **Parceling** — rejected applicants are binned by baseline score, and each bin's assumed bad rate
  is the *approved* population's bad rate in that bin, inflated by the overall approve/reject bad-rate
  ratio.

**Model classes.** Both corrections are applied to a logistic regression and to an XGBoost classifier
(300 trees, depth 3), so "does a harder selection problem help reject inference?" and "does a more
flexible model help?" stay separable in the results (see *Model Insights*).

**Deployed scorer.** The Credit Scoring tab uses **XGBoost + {deployed_label}** for its headline score —
picked automatically each session as the XGBoost method with the best true-population AUC (see
*Model Insights*) — with the other three models shown as a cross-check. Explanations use XGBoost's
own SHAP-style `pred_contribs` — no external LLM call, no API key.

**Ground-truth validation.** Because this is synthetic data, `TRUE_BAD` is available even for
rejected applicants — something a real lender never has. That lets every method be scored against
reality directly (with bootstrap confidence intervals), rather than only against the biased
accepts-only test set.

**Caveat.** All data here is synthetic and generated for demonstration. Nothing in this app should be
used to make an actual lending decision.
""")
