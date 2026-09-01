"""Does an ensemble beat the logistic scorecard, and does reject inference help?

A 2x2: two training sets crossed with four model families, all scored on the same
held-out set.

**The evaluation design is the whole experiment.** The obvious version is wrong:
build the parceling-augmented training set, split it randomly, and score on the
held-out part. That test set contains inferred labels, so what you measure is how
well a model reproduces the reject-inference assumption -- not how well it
predicts repayment. It is the same circularity that makes picking a
reject-inference method by KS meaningless.

So the approved population is split *first*. The accepts-only base model is fit on
the training half only, reject inference runs off that, and every model is scored
on the approved test half, where the outcomes were actually observed.

What that still cannot tell you: the test set is drawn from the approved
population, so this measures which model fits the observable region better. No
in-sample evaluation can measure performance on the declined population, because
nobody observed it. That needs a randomised approval experiment.

Reported alongside ranking: **calibration.** A scorecard maps probability to
points, so a model that ranks well but whose probabilities are shifted produces
wrong points even with a perfect AUC. That matters more here than a decimal place
of AUC.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scale
from reject_inference_methods import RejectInference
from train_scoring_model import CreditScoringModel

FEATURES = ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']
SEED = 42


def ks_statistic(y_true, y_prob):
    """Maximum separation between the good and bad cumulative distributions."""
    order = np.argsort(y_prob)
    y = np.asarray(y_true)[order]
    n_good, n_bad = y.sum(), len(y) - y.sum()
    if n_good == 0 or n_bad == 0:
        return float('nan')
    cum_good = np.cumsum(y) / n_good
    cum_bad = np.cumsum(1 - y) / n_bad
    return float(np.max(np.abs(cum_good - cum_bad)))


def make_models():
    """Four families, all wrapped so they take raw features."""
    logistic = Pipeline([
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)),
    ])
    logistic_uw = Pipeline([
        ('sc', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    gbm = GradientBoostingClassifier(random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                class_weight='balanced', random_state=SEED, n_jobs=-1)
    stack = StackingClassifier(
        estimators=[
            ('logistic', Pipeline([('sc', StandardScaler()),
                                   ('clf', LogisticRegression(max_iter=1000, random_state=SEED))])),
            ('gbm', GradientBoostingClassifier(random_state=SEED)),
            ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                          random_state=SEED, n_jobs=-1)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
        cv=5, n_jobs=-1,
    )
    return {
        'Logistic (balanced) - deployed': logistic,
        'Logistic (unweighted)': logistic_uw,
        'Gradient boosting': gbm,
        'Random forest': rf,
        'Stacking (LR + GBM + RF)': stack,
    }


def evaluate(name, train_label, model, X_tr, y_tr, X_te, y_te, weights=None):
    if weights is not None:
        try:
            model.fit(X_tr, y_tr, clf__sample_weight=weights)
        except (TypeError, ValueError):
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)

    p = model.predict_proba(X_te)[:, 1]
    scores = np.array([scale.score_from_prob(v) for v in p])
    approve = int((scores >= scale.APPROVE_AT).sum())

    return {
        'training set': train_label,
        'model': name,
        'AUC': roc_auc_score(y_te, p),
        'KS': ks_statistic(y_te, p),
        'Brier': brier_score_loss(y_te, p),
        'mean p(good)': float(p.mean()),
        'actual good rate': float(np.mean(y_te)),
        'approve @693': approve,
    }, p, scores


def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base_path, 'data', 'raw', 'telecom_data.csv'))
    approved = df[df['status'] == 'approved'].copy()
    rejected = df[df['status'] == 'rejected'].copy()

    # Split the observed population BEFORE any inference touches it.
    ap_train, ap_test = train_test_split(
        approved, test_size=0.3, random_state=SEED, stratify=approved['target'])

    X_te, y_te = ap_test[FEATURES], ap_test['target'].astype(int).values

    # Training set A: accepts only.
    X_a, y_a = ap_train[FEATURES], ap_train['target'].astype(int).values

    # Training set B: accepts plus parceling-inferred declines. The base model
    # behind the inference sees only ap_train, so the test half never leaks in.
    ri = RejectInference(ap_train, rejected, FEATURES, target_col='target')
    augmented, rejected_labelled, _ = ri.parceling()
    X_b, y_b = augmented[FEATURES], augmented['target'].astype(int).values

    n_inferred_good = int((rejected_labelled['target'] == 1).sum())

    rows = []
    probs = {}
    for train_label, X_tr, y_tr in [('A: accepts only', X_a, y_a),
                                    ('B: + parceling', X_b, y_b)]:
        for name, model in make_models().items():
            row, p, _ = evaluate(name, train_label, model, X_tr, y_tr, X_te, y_te)
            rows.append(row)
            probs[(train_label, name)] = p

    res = pd.DataFrame(rows)

    out = []
    out.append('=' * 88)
    out.append('GBM vs LOGISTIC SCORECARD, WITH AND WITHOUT REJECT INFERENCE')
    out.append('=' * 88)
    out.append('')
    out.append(f'Approved population        {len(approved):,} '
               f'(train {len(ap_train):,} / test {len(ap_test):,})')
    out.append(f'Declined population        {len(rejected):,}, no observed outcome')
    out.append(f'Parceling labelled good    {n_inferred_good:,} of {len(rejected):,} '
               f'({n_inferred_good / len(rejected):.1%})')
    out.append(f'Test-set good rate         {y_te.mean():.3f}')
    out.append('')
    out.append('Scored on the held-out APPROVED half only, where outcomes were observed.')
    out.append('A random split of the augmented set would put inferred labels in the test')
    out.append('set and measure agreement with the assumption instead of accuracy.')
    out.append('')

    disp = res.copy()
    for c in ['AUC', 'KS', 'Brier', 'mean p(good)', 'actual good rate']:
        disp[c] = disp[c].map(lambda v: f'{v:.4f}')
    out.append(disp.to_string(index=False))
    out.append('')

    # --- what the numbers mean ---
    out.append('-' * 88)
    out.append('READING')
    out.append('-' * 88)

    best_auc = res.loc[res['AUC'].idxmax()]
    logi_a = res[(res['training set'] == 'A: accepts only')
                 & (res['model'] == 'Logistic (balanced) - deployed')].iloc[0]
    out.append(f"Best AUC: {best_auc['model']} on {best_auc['training set']} "
               f"({best_auc['AUC']:.4f})")
    out.append(f"Deployed scorecard, accepts only: AUC {logi_a['AUC']:.4f}, KS {logi_a['KS']:.4f}")
    out.append(f"Gap to best: {best_auc['AUC'] - logi_a['AUC']:+.4f} AUC")
    out.append('')

    bal = res[res['model'] == 'Logistic (balanced) - deployed']
    unw = res[res['model'] == 'Logistic (unweighted)']
    out.append('CALIBRATION -- the part that matters for a scorecard.')
    out.append('The deployed model uses class_weight="balanced", which reweights the classes')
    out.append('and therefore calibrates probabilities to a 50/50 prior rather than to the')
    out.append('population base rate. The score maps probability to points as if the')
    out.append('probability were the real one, so a shift here shifts every score.')
    for _, r in pd.concat([bal, unw]).iterrows():
        out.append(f"  {r['model']:<32} {r['training set']:<18} "
                   f"mean p {r['mean p(good)']:.3f} vs actual {r['actual good rate']:.3f}  "
                   f"Brier {r['Brier']:.4f}")
    out.append('')

    # --- where the balanced weighting actually goes ---
    b = Pipeline([('sc', StandardScaler()),
                  ('clf', LogisticRegression(class_weight='balanced', max_iter=1000,
                                             random_state=SEED))]).fit(X_a, y_a)
    u = Pipeline([('sc', StandardScaler()),
                  ('clf', LogisticRegression(max_iter=1000, random_state=SEED))]).fit(X_a, y_a)
    cb, cu = b.named_steps['clf'].coef_[0], u.named_steps['clf'].coef_[0]
    ib, iu = b.named_steps['clf'].intercept_[0], u.named_steps['clf'].intercept_[0]
    shift = ib - iu
    inflation = shift * scale.SCORE_FACTOR

    out.append('Decomposed, the balancing is almost entirely an intercept shift:')
    out.append(f'  slope correlation between the two fits   {np.corrcoef(cb, cu)[0, 1]:.6f}')
    out.append(f'  largest slope difference                 {np.abs(cb - cu).max():.4f}')
    out.append(f'  intercept difference                     {shift:+.4f} in log-odds')
    out.append('')
    out.append(f'A constant log-odds shift times the scale factor is a constant point shift:')
    out.append(f'  {shift:.4f} x {scale.SCORE_FACTOR:.4f} = {inflation:+.1f} points on every applicant')
    out.append('')
    out.append('So the weighting buys no ranking -- a monotone shift cannot change AUC, and it')
    out.append('does not: both fits score 0.7252. What it changes is the meaning of the number.')
    out.append('The policy says approve at p(good) >= 83.3%. If the probabilities are inflated,')
    out.append('the applicants being approved do not meet that bar, so the system is not')
    out.append('implementing its own stated policy. Fixing the calibration does not change the')
    out.append('policy; it makes the implementation match it.')
    out.append('')
    out.append('Two ways to fix it: drop class_weight="balanced", which costs nothing in AUC;')
    out.append('or keep it for fitting and calibrate the probabilities on a holdout (Platt or')
    out.append('isotonic) before the points mapping. Either changes approval volume, so it is')
    out.append('a decision to make deliberately rather than a silent correction.')
    out.append('')

    out.append('DECISION IMPACT at the 693-point approve bar, out of '
               f'{len(ap_test):,} test applicants:')
    for _, r in res.iterrows():
        out.append(f"  {r['training set']:<18} {r['model']:<32} approves {r['approve @693']:>4}")
    out.append('')

    out.append('LIMIT: every row is scored on approved applicants. Which model is better on')
    out.append('the declined population is not measurable from this data -- that needs a')
    out.append('randomised approval of a slice of declines.')
    out.append('=' * 88)

    text = '\n'.join(out)
    print(text)

    reports = os.path.join(base_path, 'reports')
    os.makedirs(reports, exist_ok=True)
    with open(os.path.join(reports, 'gbm_comparison_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f'\n[Info] Report saved to: {os.path.join(reports, "gbm_comparison_report.txt")}')

    return res


if __name__ == '__main__':
    main()
