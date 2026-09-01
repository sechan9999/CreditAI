import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scoring_model import CreditScoringModel
from reject_inference_methods import RejectInference


def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index between two 1-D numeric distributions.

    PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

    Bucket edges are derived from `expected` (the reference distribution)
    using quantiles, then reused unchanged on `actual` so both
    distributions are compared on identical bins.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(breakpoints) < 3:
        # Expected distribution is (near-)constant -- can't bucket meaningfully.
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Floor empty bins so log() / division stay defined.
    epsilon = 1e-4
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def rate_psi(psi_value):
    """Standard industry rule-of-thumb PSI interpretation."""
    if psi_value < 0.10:
        return "Stable"
    elif psi_value < 0.25:
        return "Moderate shift"
    else:
        return "Severe shift"


def selection_bias_psi(approved_df, rejected_df, feature_cols, buckets=10):
    """
    Legitimate use #1: diagnose selection bias.

    Compares the raw *feature* distributions of the approved vs. rejected
    populations, feature by feature. This is exactly the population-shift
    question PSI was built for: how different are the applicants you
    reject from the applicants you approve?
    """
    rows = []
    for col in feature_cols:
        psi = calculate_psi(approved_df[col], rejected_df[col], buckets=buckets)
        rows.append({'feature': col, 'psi': psi, 'rating': rate_psi(psi)})
    return pd.DataFrame(rows).sort_values('psi', ascending=False).reset_index(drop=True)


def _fit_weighted(model, X, y, sample_weight):
    """Fit a CreditScoringModel with per-row sample weights (fuzzy augmentation)."""
    X_scaled = model.scaler.fit_transform(X[model.feature_cols])
    model.model.fit(X_scaled, y, sample_weight=sample_weight)
    model.is_fitted = True
    model.coefficients_ = pd.DataFrame({
        'feature': model.feature_cols,
        'coefficient': model.model.coef_[0],
        'odds_ratio': np.exp(model.model.coef_[0])
    }).sort_values('coefficient', ascending=False)
    return model


def score_shift_psi(approved_df, rejected_df, feature_cols, methods=None,
                     buckets=10, target_col='target'):
    """
    Legitimate use #2: compare how much each reject-inference method shifts
    the model's *score output* relative to an accepts-only baseline.

    Computed on predicted probability rather than on the clipped score, so the
    result is a property of the model and not of the current PDO.

    NOT legitimate, and deliberately not what this function does: comparing
    the raw-*feature* PSI of each method's augmented training sample against
    the population. Fuzzy Augmentation and Parceling both reuse every
    rejected applicant's real, unmodified feature vector -- they only differ
    in what target label (and weight) gets attached to it. A feature-level
    PSI on the augmented rows therefore reads ~0 for every method by
    construction, regardless of how differently the methods actually behave.

    What *does* discriminate between them: retrain a scoring model on each
    method's augmented data, score the full population (approved + rejected)
    with that model and with the accepts-only baseline model, and compare
    those two score distributions with PSI. That captures the actual effect
    of each method's labeling strategy on model behavior.
    """
    if methods is None:
        methods = ['hard_cutoff', 'fuzzy_augmentation', 'parceling']

    ri = RejectInference(approved_df, rejected_df, feature_cols, target_col=target_col)
    full_population = pd.concat([approved_df, rejected_df], ignore_index=True)
    # Compared on the model's own output, not on the clipped 300-850 score.
    # PSI with quantile bins taken from the baseline is invariant under any
    # strictly monotone transform of both distributions, so probability,
    # log-odds and an unclipped score all give the same number. np.clip is not
    # monotone at the boundaries: it piles mass onto the floor, and how much
    # mass depends on the chosen PDO. Measured through the clip this diagnostic
    # moved from 0.438 to 0.711 on a pure relabelling of the scale, which means
    # it was reporting a property of the display rather than of the model.
    baseline_scores = ri.base_model.predict_proba(full_population)

    rows = []
    score_distributions = {'baseline': baseline_scores}
    for method in methods:
        if method == 'hard_cutoff':
            combined, _, _ = ri.hard_cutoff()
            retrained = CreditScoringModel(feature_cols).fit(combined, combined[target_col])
        elif method == 'fuzzy_augmentation':
            combined, _ = ri.fuzzy_augmentation()
            retrained = _fit_weighted(CreditScoringModel(feature_cols), combined,
                                       combined[target_col], combined['weight'])
        elif method == 'parceling':
            combined, _, _ = ri.parceling()
            retrained = CreditScoringModel(feature_cols).fit(combined, combined[target_col])
        else:
            raise ValueError(f"Unknown reject-inference method: {method}")

        new_scores = retrained.predict_proba(full_population)
        psi = calculate_psi(baseline_scores, new_scores, buckets=buckets)
        rows.append({'method': method, 'psi': psi, 'rating': rate_psi(psi)})
        score_distributions[method] = new_scores

    result = pd.DataFrame(rows).sort_values('psi', ascending=False).reset_index(drop=True)
    return result, score_distributions


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'raw', 'telecom_data.csv')
    df = pd.read_csv(data_path)

    approved_df = df[df['status'] == 'approved'].copy()
    rejected_df = df[df['status'] == 'rejected'].copy()
    feature_cols = ['age', 'income', 'credit_history_months',
                     'num_credit_accounts', 'debt_ratio', 'num_late_payments']

    bias_report = selection_bias_psi(approved_df, rejected_df, feature_cols)
    shift_report, score_distributions = score_shift_psi(approved_df, rejected_df, feature_cols)

    output = "\n" + "=" * 80 + "\n"
    output += "PSI DIAGNOSTICS\n"
    output += "=" * 80 + "\n\n"
    output += "1) Selection bias -- approved vs. rejected raw feature distributions\n"
    output += "-" * 72 + "\n"
    output += bias_report.to_string(index=False) + "\n\n"
    output += "2) Score shift -- each reject-inference method vs. accepts-only baseline\n"
    output += "-" * 72 + "\n"
    output += shift_report.to_string(index=False) + "\n\n"
    output += (
        "Note: a raw-feature PSI comparing each method's augmented training\n"
        "sample to the population is NOT included above. Fuzzy Augmentation and\n"
        "Parceling both reuse every rejected applicant's real feature vector, so\n"
        "that comparison reads ~0 for both methods by construction and cannot\n"
        "discriminate between them. The score-shift PSI above is computed on\n"
        "model output instead, which does discriminate.\n"
    )

    print(output)

    reports_dir = os.path.join(base_path, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'psi_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(output)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].barh(bias_report['feature'], bias_report['psi'], color='#4C72B0')
        axes[0].axvline(0.10, color='orange', linestyle='--', linewidth=1)
        axes[0].axvline(0.25, color='red', linestyle='--', linewidth=1)
        axes[0].set_xlabel('PSI (approved vs. rejected)')
        axes[0].set_title('Selection Bias by Feature')

        axes[1].bar(shift_report['method'], shift_report['psi'], color='#55A868')
        axes[1].axhline(0.10, color='orange', linestyle='--', linewidth=1)
        axes[1].axhline(0.25, color='red', linestyle='--', linewidth=1)
        axes[1].set_ylabel('PSI (method score vs. baseline score)')
        axes[1].set_title('Score Shift by Reject-Inference Method')
        axes[1].tick_params(axis='x', rotation=15)

        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, 'psi_chart.png'), dpi=120)
        print(f"\n[Info] Chart saved to: {os.path.join(reports_dir, 'psi_chart.png')}")
    except ImportError:
        print("\n[Info] matplotlib not available -- skipping chart generation.")

    print(f"[Info] Report saved to: {os.path.join(reports_dir, 'psi_analysis_report.txt')}")
