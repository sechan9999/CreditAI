"""Score reject inference against the outcome it is normally guessing at.

Every evaluation of reject inference in this repository until now has had the
same hole in it, and said so: you can measure a model on the applicants you
approved, but the whole point of reject inference is the applicants you did not,
and nobody observes those. Picking a method by KS on the accepted population, or
by KS on labels the method itself invented, is circular.

This data carries `true_good` for every applicant, including the declined ones.
That is the held-back answer -- what would have happened. So for once the
question is answerable:

  1. split the approved population, fit the accepts-only baseline on the train
     half, and run each reject-inference method off that baseline;
  2. retrain on each method's augmented set;
  3. score every model twice -- on the approved test half, where the outcome was
     observed in the ordinary way, and on the **entire declined population**
     against `true_good`, which no real lender would have.

The second column is the one that matters, and the comparison between the two is
the finding: a method can look better on the population you can see while being
worse on the population you cannot.

`true_good` is never a feature and never a training label. It is only ever read
at scoring time, on rows no model was fit on.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scale
from psi_analysis import _fit_weighted
from reject_inference_methods import RejectInference
from train_scoring_model import CreditScoringModel

FEATURES = ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']
SEED = 42


def ks(y, p):
    order = np.argsort(p)
    y = np.asarray(y)[order]
    ng, nb = y.sum(), len(y) - y.sum()
    if ng == 0 or nb == 0:
        return float('nan')
    return float(np.max(np.abs(np.cumsum(y) / ng - np.cumsum(1 - y) / nb)))


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(base, 'data', 'raw', 'telecom_data.csv'))

    approved = df[df['status'] == 'approved'].copy()
    declined = df[df['status'] == 'rejected'].copy()

    ap_tr, ap_te = train_test_split(approved, test_size=0.3, random_state=SEED,
                                    stratify=approved['target'])

    ri = RejectInference(ap_tr, declined, FEATURES, target_col='target')

    builds = {}
    builds['none (accepts only)'] = (ap_tr, None)
    combined, _, _ = ri.hard_cutoff()
    builds['hard cutoff'] = (combined, None)
    combined, _ = ri.fuzzy_augmentation()
    builds['fuzzy augmentation'] = (combined, combined['weight'].values)
    combined, _, _ = ri.parceling()
    builds['parceling'] = (combined, None)

    rows = []
    for name, (train_df, weights) in builds.items():
        model = CreditScoringModel(FEATURES)
        if weights is not None:
            _fit_weighted(model, train_df, train_df['target'], weights)
        else:
            model.fit(train_df, train_df['target'])

        p_seen = model.predict_proba(ap_te)
        y_seen = ap_te['target'].astype(int).values

        p_unseen = model.predict_proba(declined)
        y_unseen = declined['true_good'].astype(int).values

        s_unseen = np.array([scale.score_from_prob(v) for v in p_unseen])
        would_approve = s_unseen >= scale.APPROVE_AT
        # of the declined applicants this model would now approve, how many were
        # actually good? That is the business question reject inference is for.
        precision = float(y_unseen[would_approve].mean()) if would_approve.any() else float('nan')

        rows.append({
            'method': name,
            'AUC seen': roc_auc_score(y_seen, p_seen),
            'KS seen': ks(y_seen, p_seen),
            'AUC unseen': roc_auc_score(y_unseen, p_unseen),
            'KS unseen': ks(y_unseen, p_unseen),
            'would approve': int(would_approve.sum()),
            'of those, good': precision,
        })

    res = pd.DataFrame(rows)

    out = []
    out.append('=' * 92)
    out.append('REJECT INFERENCE, SCORED AGAINST THE OUTCOME IT IS GUESSING AT')
    out.append('=' * 92)
    out.append('')
    out.append(f'  approved            {len(approved):,}  (train {len(ap_tr):,} / test {len(ap_te):,})')
    out.append(f'  declined            {len(declined):,}  -- true_good known here, and used only for scoring')
    out.append(f'  good rate, approved {approved["true_good"].mean():.4f}')
    out.append(f'  good rate, declined {declined["true_good"].mean():.4f}')
    out.append('')
    out.append('  "seen"   = approved test half, outcome observed the ordinary way')
    out.append('  "unseen" = the whole declined population, scored against true_good')
    out.append('')

    disp = res.copy()
    for c in ['AUC seen', 'KS seen', 'AUC unseen', 'KS unseen', 'of those, good']:
        disp[c] = disp[c].map(lambda v: f'{v:.4f}' if pd.notna(v) else '-')
    out.append(disp.to_string(index=False))
    out.append('')

    out.append('-' * 92)
    out.append('READING')
    out.append('-' * 92)
    best_seen = res.loc[res['AUC seen'].idxmax(), 'method']
    best_unseen = res.loc[res['AUC unseen'].idxmax(), 'method']
    out.append(f'  best on the population you can see    {best_seen}')
    out.append(f'  best on the population you cannot     {best_unseen}')
    if best_seen != best_unseen:
        out.append('')
        out.append('  These disagree, which is the whole argument. Selecting a')
        out.append('  reject-inference method on the accepted population picks the wrong one.')
    else:
        out.append('')
        out.append('  They agree here. That is worth knowing and is not guaranteed -- it means')
        out.append('  on this data the cheap evaluation happened to rank the methods correctly.')

    baseline = res[res['method'] == 'none (accepts only)'].iloc[0]
    out.append('')
    out.append('  Lift over doing no reject inference at all, on the declined population:')
    for _, r in res.iterrows():
        if r['method'] == 'none (accepts only)':
            continue
        out.append(f"    {r['method']:<22} AUC {r['AUC unseen'] - baseline['AUC unseen']:+.4f}   "
                   f"KS {r['KS unseen'] - baseline['KS unseen']:+.4f}")
    out.append('')
    out.append('  The last two columns are the business question: if this model were used to')
    out.append('  reconsider the declined applicants, how many would it approve, and what')
    out.append(f'  share of those were actually good? The declined base rate is '
               f'{declined["true_good"].mean():.1%}.')
    out.append('')
    out.append('  CAVEAT: true_good exists here because the data is synthetic. A real lender')
    out.append('  has no such column, which is why reject inference exists. What this shows')
    out.append('  is which method to reach for -- not a measurement you can repeat in')
    out.append('  production without deliberately approving a randomised slice of declines.')
    out.append('=' * 92)

    text = '\n'.join(out)
    print(text)

    reports = os.path.join(base, 'reports')
    os.makedirs(reports, exist_ok=True)
    with open(os.path.join(reports, 'reject_inference_truth_report.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write(text + '\n')
    print(f'\n[Info] Report saved to: '
          f'{os.path.join(reports, "reject_inference_truth_report.txt")}')
    return res


if __name__ == '__main__':
    main()
