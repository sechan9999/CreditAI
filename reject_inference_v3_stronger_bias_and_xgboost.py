# =====================================================================================
# Reject inference, round 2: a harder selection-bias scenario + a model-class comparison
#
# Paste each "# %% Cell N" block into its own new cell at the bottom of
# reject_inference_model.ipynb, in order, and run top to bottom. Everything here is
# self-contained (re-imports, redefines FEATURES, redefines a fresh `psi()` helper) so it
# does not depend on any variable from your existing V2 cells still being in memory.
#
# WHY THIS EXISTS
# On V2, fuzzy augmentation and parceling drove PSI (population stability) from 0.33 to
# ~0.00 -- a clean fix to the *feature-distribution* mismatch between the accepts-only
# training sample and the full population -- but true-population AUC barely moved
# (0.7325 baseline vs 0.7327 fuzzy vs 0.7317 parceling: differences smaller than sampling
# noise). The likely reason: V2's approval rule had a large random component
# (rng.normal(0, 1)), so the accepts-only model was already extrapolating fairly well and
# there wasn't much true bias left to correct.
#
# V3 below cuts that noise term from sd=1.0 to sd=0.25, making approval much more
# deterministically driven by the same features that drive true risk -- a harder,
# more realistic missing-not-at-random selection problem. If reject inference is doing
# real work (not just cosmetic distribution-matching), this is the scenario where it
# should show a visible AUC gain over baseline. It also adds an XGBoost comparison,
# since you asked why the pipeline used logistic regression instead -- same reject
# inference logic (fuzzy augmentation / parceling weights), swapped model class, so the
# two questions ("does a harder selection problem help reject inference?" and "does a
# more flexible model help?") stay separable in the results.
# =====================================================================================

# %% Cell 19: generate V3 -- same risk model as V2, a much more deterministic approval rule
import pandas as pd
import numpy as np
from snowflake.snowpark.context import get_active_session

session = get_active_session()
rng = np.random.default_rng(789)
n_applicants = 7000

age = rng.integers(18, 75, n_applicants)
income = rng.lognormal(10.5, 0.5, n_applicants).round(2)
credit_history_months = rng.integers(0, 240, n_applicants)
num_credit_accounts = rng.poisson(3, n_applicants)
debt_ratio = rng.beta(2, 5, n_applicants).round(3)
num_late_payments = rng.poisson(1.2, n_applicants)

synthetic3 = pd.DataFrame({
    "AGE": age,
    "INCOME": income,
    "CREDIT_HISTORY_MONTHS": credit_history_months,
    "NUM_CREDIT_ACCOUNTS": num_credit_accounts,
    "DEBT_RATIO": debt_ratio,
    "NUM_LATE_PAYMENTS": num_late_payments,
})

# Same TRUE_BAD risk model as V2, so the underlying risk relationship is unchanged --
# only the approval mechanism below changes.
logit3 = (
    0.5
    - 0.02 * (synthetic3["AGE"] - 45).abs()
    - 0.00002 * synthetic3["INCOME"]
    - 0.01 * synthetic3["CREDIT_HISTORY_MONTHS"]
    + 0.05 * synthetic3["NUM_CREDIT_ACCOUNTS"]
    + 2.5 * synthetic3["DEBT_RATIO"]
    + 0.35 * synthetic3["NUM_LATE_PAYMENTS"]
)
true_pd3 = 1 / (1 + np.exp(-logit3))
synthetic3["TRUE_BAD"] = rng.binomial(1, true_pd3)

# KEY CHANGE vs V2: noise sd 0.25 instead of 1.0 -- approval is now much more tightly
# determined by the same risk-correlated features, i.e. a stronger MNAR-style selection
# effect for reject inference to try to correct.
approval_score3 = (
    0.02 * synthetic3["CREDIT_HISTORY_MONTHS"]
    - 3.0 * synthetic3["DEBT_RATIO"]
    - 0.6 * synthetic3["NUM_LATE_PAYMENTS"]
    + 0.00003 * synthetic3["INCOME"]
    + rng.normal(0, 0.25, n_applicants)
)
cutoff3 = np.quantile(approval_score3, 1 - 5000 / 7000)  # same 5000/2000 approve/reject split as V2
synthetic3["STATUS"] = np.where(approval_score3 >= cutoff3, "approved", "rejected")
synthetic3["TARGET"] = np.where(synthetic3["STATUS"] == "approved", synthetic3["TRUE_BAD"], np.nan)

print("V3 (stronger, more deterministic approval rule -- noise sd 0.25 vs 1.0 in V2)")
print(synthetic3["STATUS"].value_counts())
print("Overall TRUE_BAD rate (all 7000, normally unobservable):        ", round(synthetic3["TRUE_BAD"].mean(), 3))
print("Bad rate among approved only (what a real lender sees):         ", round(synthetic3.loc[synthetic3["STATUS"] == "approved", "TRUE_BAD"].mean(), 3))
print("Bad rate among rejected (never observed in real life):          ", round(synthetic3.loc[synthetic3["STATUS"] == "rejected", "TRUE_BAD"].mean(), 3))
# Compare this approved-vs-rejected gap to V2's -- it should be noticeably wider here,
# which is exactly the "stronger selection bias" you want to stress-test reject inference against.
synthetic3.head(10)

# %% Cell 20: persist V3 to Snowflake (mirrors how V2 was persisted)
observed3 = synthetic3.drop(columns=["TRUE_BAD"]).copy()
observed3.insert(0, "APPLICANT_ID", range(1, len(observed3) + 1))
ground_truth3 = synthetic3[["TRUE_BAD"]].copy()
ground_truth3.insert(0, "APPLICANT_ID", range(1, len(ground_truth3) + 1))

session.write_pandas(observed3, "TELECOM_CREDIT_V3", database="TELECOM_ANALYSIS", schema="PUBLIC", auto_create_table=True, overwrite=True)
session.write_pandas(ground_truth3, "TELECOM_CREDIT_V3_TRUE_LABELS", database="TELECOM_ANALYSIS", schema="PUBLIC", auto_create_table=True, overwrite=True)
print("Wrote", len(observed3), "applicants to TELECOM_ANALYSIS.PUBLIC.TELECOM_CREDIT_V3")
print("Wrote hidden ground-truth labels to TELECOM_ANALYSIS.PUBLIC.TELECOM_CREDIT_V3_TRUE_LABELS")

# %% Cell 21: shared setup for everything below -- imports, FEATURES, a fresh psi() helper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

FEATURES = ["AGE", "INCOME", "CREDIT_HISTORY_MONTHS", "NUM_CREDIT_ACCOUNTS", "DEBT_RATIO", "NUM_LATE_PAYMENTS"]

def psi(score_expected, score_actual, bins=10, w_actual=None):
    """Population Stability Index: how much score_actual's distribution has drifted
    from score_expected's, using score_expected's own decile edges as the reference grid.
    w_actual lets a reweighted/augmented sample (fuzzy augmentation, parceling) contribute
    its per-row weights instead of being treated as a plain unweighted sample."""
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

def bootstrap_auc_diff(y_true, score_a, score_b, n_boot=2000, seed=0):
    """Paired bootstrap CI for AUC(score_a) - AUC(score_b) on the same population.
    If the 95% CI excludes 0, the difference is unlikely to be noise."""
    rng_b = np.random.default_rng(seed)
    y_true = np.asarray(y_true); score_a = np.asarray(score_a); score_b = np.asarray(score_b)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng_b.integers(0, n, n)
        diffs[i] = roc_auc_score(y_true[idx], score_a[idx]) - roc_auc_score(y_true[idx], score_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_diff": float(diffs.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "significant": bool(lo > 0 or hi < 0)}

print("FEATURES, psi(), and bootstrap_auc_diff() are ready.")

# %% Cell 22: load V3, baseline logistic regression (accepts-only) -- same recipe as V2's baseline
df3 = session.table("TELECOM_ANALYSIS.PUBLIC.TELECOM_CREDIT_V3").to_pandas()
print("Fresh applicant batch (V3):", df3.shape)
print(df3["STATUS"].value_counts())
approved3 = df3[df3["STATUS"] == "approved"].copy()
rejected3 = df3[df3["STATUS"] == "rejected"].copy()
approved3["BAD"] = approved3["TARGET"]
X3 = approved3[FEATURES]
y3 = approved3["BAD"]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.25, random_state=42, stratify=y3)
scaler3 = StandardScaler().fit(X_train3)
X_train3_s = scaler3.transform(X_train3)
X_test3_s = scaler3.transform(X_test3)
baseline_model3 = LogisticRegression(max_iter=1000)
baseline_model3.fit(X_train3_s, y_train3)
baseline_auc3 = roc_auc_score(y_test3, baseline_model3.predict_proba(X_test3_s)[:, 1])
print("Baseline (accepts-only) LR test AUC on V3:", round(baseline_auc3, 4))

# %% Cell 23: fuzzy augmentation on V3 (identical recipe to V2's fuzzy-augmentation cell)
X_rej3 = scaler3.transform(rejected3[FEATURES])
p_bad_rej3 = baseline_model3.predict_proba(X_rej3)[:, 1]
print("Baseline model mean predicted P(bad) on REJECTED V3 applicants:", round(p_bad_rej3.mean(), 3))
print("Observed bad rate on APPROVED V3 applicants:", round(y3.mean(), 3))

rej_good3 = rejected3[FEATURES].copy()
rej_good3["BAD"] = 0
rej_good3["weight"] = 1 - p_bad_rej3
rej_bad3 = rejected3[FEATURES].copy()
rej_bad3["BAD"] = 1
rej_bad3["weight"] = p_bad_rej3
approved_aug3 = approved3[FEATURES + ["BAD"]].copy()
approved_aug3["weight"] = 1.0
augmented3 = pd.concat([approved_aug3, rej_good3, rej_bad3], ignore_index=True)

X_aug3 = augmented3[FEATURES]
y_aug3 = augmented3["BAD"]
w_aug3 = augmented3["weight"]
scaler_aug3 = StandardScaler().fit(X_aug3)
X_aug3_s = scaler_aug3.transform(X_aug3)
ri_model3 = LogisticRegression(max_iter=1000)
ri_model3.fit(X_aug3_s, y_aug3, sample_weight=w_aug3)
X_test3_s_ri = scaler_aug3.transform(X_test3)
ri_auc3 = roc_auc_score(y_test3, ri_model3.predict_proba(X_test3_s_ri)[:, 1])
print("Fuzzy-augmentation LR AUC on V3 (accepts test set):", round(ri_auc3, 4))

# %% Cell 24: parceling on V3 (identical recipe to V2's parceling cell)
inflation_factor3 = p_bad_rej3.mean() / y3.mean()
print("Parceling inflation factor (V3):", round(inflation_factor3, 2), "x")
n_bins = 10
score_approved_base3 = baseline_model3.predict_proba(scaler3.transform(approved3[FEATURES]))[:, 1]
score_rejected_base3 = baseline_model3.predict_proba(scaler3.transform(rejected3[FEATURES]))[:, 1]
combo_scores3 = np.concatenate([score_approved_base3, score_rejected_base3])
bin_edges3 = np.quantile(combo_scores3, np.linspace(0, 1, n_bins + 1))
bin_edges3[0], bin_edges3[-1] = -np.inf, np.inf
approved_bin3 = np.digitize(score_approved_base3, bin_edges3) - 1
bin_bad_rate3 = pd.Series(approved3["BAD"].values).groupby(approved_bin3).mean()
reject_bin3 = np.digitize(score_rejected_base3, bin_edges3) - 1
reject_base_rate3 = pd.Series(reject_bin3).map(bin_bad_rate3).fillna(bin_bad_rate3.mean()).values
reject_rate_inflated3 = np.clip(reject_base_rate3 * inflation_factor3, 0, 1)

rej_good_p3 = rejected3[FEATURES].copy()
rej_good_p3["BAD"] = 0
rej_good_p3["weight"] = 1 - reject_rate_inflated3
rej_bad_p3 = rejected3[FEATURES].copy()
rej_bad_p3["BAD"] = 1
rej_bad_p3["weight"] = reject_rate_inflated3
augmented_p3 = pd.concat([approved_aug3, rej_good_p3, rej_bad_p3], ignore_index=True)
scaler_p3 = StandardScaler().fit(augmented_p3[FEATURES])
parcel_model3 = LogisticRegression(max_iter=1000).fit(scaler_p3.transform(augmented_p3[FEATURES]), augmented_p3["BAD"], sample_weight=augmented_p3["weight"])
parcel_auc3 = roc_auc_score(y_test3, parcel_model3.predict_proba(scaler_p3.transform(X_test3))[:, 1])
print("Parceling LR AUC on V3 (accepts test set):", round(parcel_auc3, 4))

# %% Cell 25: PSI diagnostic on V3 (same structure as V2's PSI cell)
score_full_baseline3 = baseline_model3.predict_proba(scaler3.transform(df3[FEATURES]))[:, 1]
score_train_baseline3 = baseline_model3.predict_proba(scaler3.transform(approved3[FEATURES]))[:, 1]
psi_baseline3 = psi(score_full_baseline3, score_train_baseline3)
score_full_fuzzy3 = ri_model3.predict_proba(scaler_aug3.transform(df3[FEATURES]))[:, 1]
score_train_fuzzy3 = ri_model3.predict_proba(scaler_aug3.transform(augmented3[FEATURES]))[:, 1]
psi_fuzzy3 = psi(score_full_fuzzy3, score_train_fuzzy3, w_actual=augmented3["weight"].values)
score_full_parcel3 = parcel_model3.predict_proba(scaler_p3.transform(df3[FEATURES]))[:, 1]
score_train_parcel3 = parcel_model3.predict_proba(scaler_p3.transform(augmented_p3[FEATURES]))[:, 1]
psi_parcel3 = psi(score_full_parcel3, score_train_parcel3, w_actual=augmented_p3["weight"].values)
print("V3 PSI (training sample vs full population):")
print("  Baseline, accepts-only, no correction :", round(psi_baseline3, 4))
print("  Fuzzy augmentation                    :", round(psi_fuzzy3, 4))
print("  Parceling, inflation-adjusted         :", round(psi_parcel3, 4))

# %% Cell 26: ground-truth validation on V3 (same structure as V2's ground-truth cell)
truth3 = session.table("TELECOM_ANALYSIS.PUBLIC.TELECOM_CREDIT_V3_TRUE_LABELS").to_pandas()
full_eval3 = df3.merge(truth3, on="APPLICANT_ID")
y_true_full3 = full_eval3["TRUE_BAD"]
score_baseline_full3 = baseline_model3.predict_proba(scaler3.transform(full_eval3[FEATURES]))[:, 1]
score_fuzzy_full3 = ri_model3.predict_proba(scaler_aug3.transform(full_eval3[FEATURES]))[:, 1]
score_parcel_full3 = parcel_model3.predict_proba(scaler_p3.transform(full_eval3[FEATURES]))[:, 1]
auc_true_baseline3 = roc_auc_score(y_true_full3, score_baseline_full3)
auc_true_fuzzy3 = roc_auc_score(y_true_full3, score_fuzzy_full3)
auc_true_parcel3 = roc_auc_score(y_true_full3, score_parcel_full3)

results_summary_v3 = pd.DataFrame({
    "method": ["baseline_accepts_only", "fuzzy_augmentation", "parceling"],
    "model_class": ["logistic_regression"] * 3,
    "accepts_test_auc": [baseline_auc3, ri_auc3, parcel_auc3],
    "true_population_auc": [auc_true_baseline3, auc_true_fuzzy3, auc_true_parcel3],
    "psi_train_vs_full_population": [psi_baseline3, psi_fuzzy3, psi_parcel3],
})
print(results_summary_v3.to_string(index=False))

# Statistical significance: does fuzzy/parceling actually beat baseline on true AUC,
# or is it within noise (as it was on V2)? 95% CI excluding 0 means "yes, really different."
print("\nBootstrapped AUC difference vs baseline (true population, V3, LR):")
print("  fuzzy - baseline   :", bootstrap_auc_diff(y_true_full3, score_fuzzy_full3, score_baseline_full3))
print("  parceling - baseline:", bootstrap_auc_diff(y_true_full3, score_parcel_full3, score_baseline_full3))

# %% Cell 27: check XGBoost/LightGBM are available in this notebook environment
try:
    import xgboost as xgb
    print("xgboost:", xgb.__version__)
except ImportError as e:
    print("xgboost not available:", e, "-- run `!pip install xgboost` in a cell first, or add it via the notebook's package picker.")
try:
    import lightgbm as lgb
    print("lightgbm:", lgb.__version__)
except ImportError as e:
    print("lightgbm not available:", e)

# %% Cell 28: same reject-inference logic, XGBoost instead of logistic regression
# Trees don't need feature scaling, and they accept sample_weight natively, so the fuzzy
# augmentation / parceling weighting scheme carries over unchanged -- only the model class
# and the (unscaled) feature matrices differ from cells 22-24.
import xgboost as xgb

xgb_baseline = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="auc", random_state=42)
xgb_baseline.fit(X_train3, y_train3)
xgb_baseline_auc = roc_auc_score(y_test3, xgb_baseline.predict_proba(X_test3)[:, 1])

xgb_fuzzy = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="auc", random_state=42)
xgb_fuzzy.fit(X_aug3, y_aug3, sample_weight=w_aug3)
xgb_fuzzy_auc = roc_auc_score(y_test3, xgb_fuzzy.predict_proba(X_test3)[:, 1])

xgb_parcel = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, eval_metric="auc", random_state=42)
xgb_parcel.fit(augmented_p3[FEATURES], augmented_p3["BAD"], sample_weight=augmented_p3["weight"])
xgb_parcel_auc = roc_auc_score(y_test3, xgb_parcel.predict_proba(X_test3)[:, 1])

print("XGBoost accepts-test AUC  -- baseline:", round(xgb_baseline_auc, 4), " fuzzy:", round(xgb_fuzzy_auc, 4), " parceling:", round(xgb_parcel_auc, 4))

# Ground-truth validation for the XGBoost models, on the same full_eval3 population
score_baseline_full_xgb = xgb_baseline.predict_proba(full_eval3[FEATURES])[:, 1]
score_fuzzy_full_xgb = xgb_fuzzy.predict_proba(full_eval3[FEATURES])[:, 1]
score_parcel_full_xgb = xgb_parcel.predict_proba(full_eval3[FEATURES])[:, 1]
auc_true_baseline_xgb = roc_auc_score(y_true_full3, score_baseline_full_xgb)
auc_true_fuzzy_xgb = roc_auc_score(y_true_full3, score_fuzzy_full_xgb)
auc_true_parcel_xgb = roc_auc_score(y_true_full3, score_parcel_full_xgb)

# Calibration check -- AUC alone can hide a model that ranks well but is overconfident.
# Brier score is lower = better calibrated (0 = perfect).
brier_baseline_lr = brier_score_loss(y_true_full3, score_baseline_full3)
brier_baseline_xgb = brier_score_loss(y_true_full3, score_baseline_full_xgb)
brier_fuzzy_lr = brier_score_loss(y_true_full3, score_fuzzy_full3)
brier_fuzzy_xgb = brier_score_loss(y_true_full3, score_fuzzy_full_xgb)
brier_parcel_lr = brier_score_loss(y_true_full3, score_parcel_full3)
brier_parcel_xgb = brier_score_loss(y_true_full3, score_parcel_full_xgb)

results_summary_v3_xgb = pd.DataFrame({
    "method": ["baseline_accepts_only", "fuzzy_augmentation", "parceling"],
    "model_class": ["xgboost"] * 3,
    "accepts_test_auc": [xgb_baseline_auc, xgb_fuzzy_auc, xgb_parcel_auc],
    "true_population_auc": [auc_true_baseline_xgb, auc_true_fuzzy_xgb, auc_true_parcel_xgb],
    "true_population_brier": [brier_baseline_xgb, brier_fuzzy_xgb, brier_parcel_xgb],
})
print(results_summary_v3_xgb.to_string(index=False))

print("\nBootstrapped AUC difference vs XGBoost baseline (true population, V3):")
print("  XGB fuzzy - XGB baseline    :", bootstrap_auc_diff(y_true_full3, score_fuzzy_full_xgb, score_baseline_full_xgb))
print("  XGB parceling - XGB baseline:", bootstrap_auc_diff(y_true_full3, score_parcel_full_xgb, score_baseline_full_xgb))
print("\nBootstrapped AUC difference, XGBoost baseline vs LR baseline (does model class alone matter?):")
print("  XGB baseline - LR baseline  :", bootstrap_auc_diff(y_true_full3, score_baseline_full_xgb, score_baseline_full3))

# %% Cell 29: final side-by-side -- V2 (LR, weak selection) vs V3 (LR, strong selection) vs V3 (XGBoost)
# V2 numbers are hard-coded from your last run (results_summary from cell 17) so this table
# stands on its own even if V2's kernel state is no longer live.
final_comparison = pd.DataFrame({
    "method": ["baseline_accepts_only", "fuzzy_augmentation", "parceling"] * 3,
    "scenario": ["V2_weak_selection_LR"] * 3 + ["V3_strong_selection_LR"] * 3 + ["V3_strong_selection_XGB"] * 3,
    "true_population_auc": [
        0.732549, 0.732668, 0.731565,                                   # V2, from your last run
        auc_true_baseline3, auc_true_fuzzy3, auc_true_parcel3,          # V3, logistic regression
        auc_true_baseline_xgb, auc_true_fuzzy_xgb, auc_true_parcel_xgb,  # V3, XGBoost
    ],
})
print(final_comparison.to_string(index=False))
print("\nRead this next to the bootstrap CIs printed in cells 26 and 28: those tell you which")
print("of these differences are real and which are within sampling noise -- this table alone")
print("cannot distinguish the two.")
