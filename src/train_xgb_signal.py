"""Train the XGBoost signal shown in the app next to the deployed rule-based verdict.

This is a *second*, independent opinion -- not a replacement for the deployed
logistic scorecard. The scorecard stays the system of record (`src/scale.py`,
`MODEL_COEFS` in `index.html`); this model's output is displayed alongside it
so a reviewer can see where a tree ensemble would disagree.

Evaluation follows the same design as `gbm_comparison.py`, for the same reason
stated there: the approved population is split *before* reject inference runs,
so the augmented training set never leaks into the held-out test half. A random
split of the augmented set would test how well a model reproduces the
reject-inference assumption, not how well it predicts repayment.

Two things this script deliberately does NOT claim:

1. That reject inference helps here. `reports/reject_inference_truth_report.txt`
   measured this against the synthetic ground truth this dataset happens to
   carry (a real lender has no such column): parceling scored *worse* than
   doing no reject inference at all on the population you cannot see (AUC
   -0.0011, KS -0.0073). It is trained on the parceling-augmented set anyway,
   because "reject-inference model" is what was asked for and the honest
   result is a caveat on the report, not a reason to silently swap in a
   plain accepts-only model instead.

2. That this ensemble beats the deployed scorecard. `gbm_comparison.py`
   already found gradient boosting roughly matches, not exceeds, the logistic
   fit on this data (AUC 0.6846 vs 0.6904 on the augmented set). XGBoost is
   evaluated the same way, on the same held-out split, and the report says so
   either way.

The model is deliberately small -- max_depth=3, 40 trees -- not because a
deeper model would overfit this particular dataset, but because it is dumped
tree-by-tree into `src/xgb_model_params.json` and walked by a small JS
evaluator embedded in `index.html` (see `src/extract_xgb_params.py` and
`src/sync_xgb_params.py`). A shallow, narrow ensemble keeps that payload a few
hundred nodes instead of tens of thousands.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reject_inference_methods import RejectInference
from train_scoring_model import CreditScoringModel
import scale

FEATURES = ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']
SEED = 42


def ks_statistic(y_true, y_prob):
    order = np.argsort(y_prob)
    y = np.asarray(y_true)[order]
    n_good, n_bad = y.sum(), len(y) - y.sum()
    if n_good == 0 or n_bad == 0:
        return float('nan')
    cum_good = np.cumsum(y) / n_good
    cum_bad = np.cumsum(1 - y) / n_bad
    return float(np.max(np.abs(cum_good - cum_bad)))


def make_xgb():
    """Shallow and narrow on purpose -- see module docstring."""
    return XGBClassifier(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        base_score=0.5,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=-1,
    )


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    scores = np.array([scale.score_from_prob(v) for v in p])
    return {
        'model': name,
        'AUC': roc_auc_score(y_te, p),
        'KS': ks_statistic(y_te, p),
        'Brier': brier_score_loss(y_te, p),
        'mean p(good)': float(p.mean()),
        'actual good rate': float(np.mean(y_te)),
        'approve @693': int((scores >= scale.APPROVE_AT).sum()),
    }, p


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_path, 'data', 'raw', 'telecom_data.csv'))
    approved = df[df['status'] == 'approved'].copy()
    rejected = df[df['status'] == 'rejected'].copy()

    # Same split as gbm_comparison.py and train_scoring_model.py: the approved
    # population is split before anything else touches it.
    ap_train, ap_test = train_test_split(
        approved, test_size=0.3, random_state=SEED, stratify=approved['target'])
    X_te, y_te = ap_test[FEATURES], ap_test['target'].astype(int).values

    X_a, y_a = ap_train[FEATURES], ap_train['target'].astype(int).values

    ri = RejectInference(ap_train, rejected, FEATURES, target_col='target')
    augmented, rejected_labelled, _ = ri.parceling()
    X_b, y_b = augmented[FEATURES], augmented['target'].astype(int).values
    n_inferred_good = int((rejected_labelled['target'] == 1).sum())

    # Deployed logistic scorecard, for the reference row in the report.
    logi = CreditScoringModel(FEATURES).fit(ap_train, y_a)
    logi_p = logi.predict_proba(ap_test)
    logi_scores = logi.predict_score(ap_test)
    logi_row = {
        'model': 'Logistic (deployed scorecard)',
        'AUC': roc_auc_score(y_te, logi_p),
        'KS': ks_statistic(y_te, logi_p),
        'Brier': brier_score_loss(y_te, logi_p),
        'mean p(good)': float(logi_p.mean()),
        'actual good rate': float(y_te.mean()),
        'approve @693': int((logi_scores >= scale.APPROVE_AT).sum()),
    }

    rows = [logi_row]
    models = {}
    for label, X_tr, y_tr in [('A: accepts only', X_a, y_a), ('B: + parceling', X_b, y_b)]:
        row, p = evaluate(f'XGBoost ({label})', make_xgb(), X_tr, y_tr, X_te, y_te)
        rows.append(row)
        models[label] = row

    res = pd.DataFrame(rows)

    # The model that actually ships: XGBoost trained on the reject-inference
    # augmented set. Refit standalone so this is the exact artifact extracted
    # and embedded, not folded into the loop above.
    deployed = make_xgb().fit(X_b, y_b)

    out = []
    out.append('=' * 88)
    out.append('XGBOOST REJECT-INFERENCE SIGNAL, EVALUATED AGAINST THE DEPLOYED SCORECARD')
    out.append('=' * 88)
    out.append('')
    out.append(f'Approved population        {len(approved):,} '
               f'(train {len(ap_train):,} / test {len(ap_test):,})')
    out.append(f'Declined population        {len(rejected):,}, no observed outcome')
    out.append(f'Parceling labelled good    {n_inferred_good:,} of {len(rejected):,} '
               f'({n_inferred_good / len(rejected):.1%})')
    out.append(f'Test-set good rate         {y_te.mean():.3f}')
    out.append('')
    out.append('Scored on the held-out APPROVED half only -- see gbm_comparison.py for why.')
    out.append('')
    disp = res.copy()
    for c in ['AUC', 'KS', 'Brier', 'mean p(good)', 'actual good rate']:
        disp[c] = disp[c].map(lambda v: f'{v:.4f}')
    out.append(disp.to_string(index=False))
    out.append('')
    out.append('-' * 88)
    out.append('READING')
    out.append('-' * 88)
    a_row, b_row = models['A: accepts only'], models['B: + parceling']
    out.append(f"XGBoost, accepts only:      AUC {a_row['AUC']:.4f}, KS {a_row['KS']:.4f}")
    out.append(f"XGBoost, + parceling:       AUC {b_row['AUC']:.4f}, KS {b_row['KS']:.4f}")
    out.append(f"Deployed logistic scorecard: AUC {logi_row['AUC']:.4f}, KS {logi_row['KS']:.4f}")
    out.append(f"Gap, deployed vs. XGBoost+parceling: "
               f"{b_row['AUC'] - logi_row['AUC']:+.4f} AUC")
    out.append('')
    out.append('This does not beat the scorecard on this data -- consistent with')
    out.append('gbm_comparison.py, which found the same for plain gradient boosting.')
    out.append('It ships as a SEPARATE signal next to the rule-based verdict, not a')
    out.append('replacement for it.')
    out.append('')
    out.append('CAVEAT ON THE REJECT-INFERENCE PART, from reject_inference_truth_report.txt:')
    out.append('scored against this synthetic dataset\'s (otherwise unobservable) true')
    out.append('outcome for declined applicants, parceling scored WORSE than doing no')
    out.append('reject inference at all (AUC -0.0011, KS -0.0073 vs. accepts-only). A real')
    out.append('deployment cannot run that check -- it needs a randomised approval slice --')
    out.append('so this model is shown as what reject inference produces, not endorsed as')
    out.append('an improvement over the alternative.')
    out.append('=' * 88)
    text = '\n'.join(out)
    print(text)

    reports = os.path.join(base_path, 'reports')
    os.makedirs(reports, exist_ok=True)
    with open(os.path.join(reports, 'xgb_signal_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\n[Info] Report saved to: {os.path.join(reports, "xgb_signal_report.txt")}')

    processed_dir = os.path.join(base_path, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    model_path = os.path.join(processed_dir, 'xgb_signal_model.pkl')
    joblib.dump({'model': deployed, 'feature_cols': FEATURES}, model_path)
    print(f'[Info] Model saved to: {model_path}')

    return res


if __name__ == '__main__':
    main()
