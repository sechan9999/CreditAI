"""Load the supplied applicant export into the repository's conventions.

Run once per new export. Everything downstream reads
`data/raw/telecom_data.csv`, so this is the only place the source file's
conventions are translated -- and every translation is asserted rather than
assumed, because the two that bit this repo before were both silent.

**Polarity.** The export's `TARGET` is 1 = bad: on approved rows it equals
`TRUE_BAD` in 100% of cases. The pipeline's `target` has always been 1 = good.
Loading the export unchanged would flip the sign of every coefficient and the
model would still fit, still report a plausible AUC, and rank every applicant
backwards. So the flip happens here, once, and is checked.

**Units.** The previous data carried income as monthly Korean 10,000-won units
while `model_params.json` had been extracted from an earlier run in annual
units, an 11x mismatch that nothing detected -- it declined roughly 190
applicants per 7,000 who the actual trained model would have approved. This
export is annual USD. It is stored as annual USD, the app asks for annual USD,
and no layer converts. One unit, stated on screen, is the only reliable fix.

**`true_good`.** The export adds `TRUE_BAD` for every applicant including the
declined ones -- the outcome nobody normally observes. It is stored as
`true_good` (1 = good) so it shares the polarity of `target` rather than
opposing it, which is how the bug above happens. It must never be used as a
training feature or label for the declined population: it is the held-back
answer that makes reject inference measurable instead of assumed.
"""

import os
import sys

import numpy as np
import pandas as pd

FEATURES = ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']

RENAME = {
    'APPLICANT_ID': 'applicant_id',
    'AGE': 'age',
    'INCOME': 'income',
    'CREDIT_HISTORY_MONTHS': 'credit_history_months',
    'NUM_CREDIT_ACCOUNTS': 'num_credit_accounts',
    'DEBT_RATIO': 'debt_ratio',
    'NUM_LATE_PAYMENTS': 'num_late_payments',
    'STATUS': 'status',
}


def load(source_path):
    raw = pd.read_csv(source_path)

    missing = set(RENAME) | {'TARGET', 'TRUE_BAD'}
    missing -= set(raw.columns)
    assert not missing, f'export is missing columns: {sorted(missing)}'

    df = raw.rename(columns=RENAME).copy()

    # --- polarity, verified rather than trusted ---
    approved = df['status'] == 'approved'
    agreement = (raw.loc[approved.values, 'TARGET'] == raw.loc[approved.values, 'TRUE_BAD']).mean()
    assert agreement > 0.99, (
        f'TARGET and TRUE_BAD agree on only {agreement:.1%} of approved rows. '
        'The assumption that both are 1 = bad no longer holds -- check the export '
        'before flipping anything.'
    )

    df['target'] = np.where(approved, 1 - raw['TARGET'], np.nan)   # 1 = good
    df['true_good'] = 1 - raw['TRUE_BAD']                          # 1 = good, all rows

    assert df.loc[approved, 'target'].notna().all(), 'approved rows must carry a target'
    assert df.loc[~approved, 'target'].isna().all(), 'declined rows must not carry a target'
    assert np.array_equal(df.loc[approved, 'target'].values,
                          df.loc[approved, 'true_good'].values), \
        'on approved rows target and true_good must be the same observed outcome'

    # --- direction of effect, as a guard against a silent flip ---
    bad = 1 - df['true_good']
    signs = {'income': -1, 'credit_history_months': -1,
             'debt_ratio': +1, 'num_late_payments': +1}
    for col, expected in signs.items():
        r = np.corrcoef(df[col], bad)[0, 1]
        assert np.sign(r) == expected, (
            f'{col} correlates {r:+.4f} with being bad; expected sign {expected:+d}. '
            'Either the labels are inverted or this is not the data you think it is.'
        )

    # --- units ---
    mean_income = df['income'].mean()
    assert 10_000 < mean_income < 200_000, (
        f'mean income is {mean_income:,.0f}. This loader expects annual USD. '
        'If the export changed units, change the app label and the income bands too, '
        'not just this assertion.'
    )

    df = df[['applicant_id'] + FEATURES + ['status', 'target', 'true_good']]
    return df, raw


def report(df):
    ap = df[df['status'] == 'approved']
    rj = df[df['status'] == 'rejected']
    out = []
    out.append('=' * 78)
    out.append('DATA INGESTED')
    out.append('=' * 78)
    out.append(f'  rows                      {len(df):,}')
    out.append(f'  approved / declined       {len(ap):,} / {len(rj):,}')
    out.append(f'  observed good rate        {ap["target"].mean():.4f}  (approved only)')
    out.append('')
    out.append('  true_good is present for every applicant, including the declined:')
    out.append(f'    good rate, approved     {ap["true_good"].mean():.4f}')
    out.append(f'    good rate, declined     {rj["true_good"].mean():.4f}')
    out.append(f'    the declined population is {(1-rj["true_good"].mean())/(1-ap["true_good"].mean()):.2f}x '
               'more likely to be bad')
    out.append('')
    out.append('  This is the column that makes reject inference testable. It is the')
    out.append('  answer, not a feature: train without it, then score against it.')
    out.append('')
    out.append(f'  {"feature":<24}{"min":>12}{"mean":>12}{"max":>12}')
    out.append('  ' + '-' * 60)
    for f in FEATURES:
        out.append(f'  {f:<24}{df[f].min():>12.3f}{df[f].mean():>12.3f}{df[f].max():>12.3f}')
    out.append('')
    out.append('  income is ANNUAL USD. Nothing downstream converts it.')
    out.append('=' * 78)
    return '\n'.join(out)


def main():
    if len(sys.argv) < 2:
        print('usage: python src/ingest_data.py <export.csv>', file=sys.stderr)
        return 2

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df, _ = load(sys.argv[1])

    text = report(df)
    print(text)

    dest = os.path.join(base, 'data', 'raw', 'telecom_data.csv')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.to_csv(dest, index=False)
    print(f'\n[Info] Written to {dest}')

    reports = os.path.join(base, 'reports')
    os.makedirs(reports, exist_ok=True)
    with open(os.path.join(reports, 'data_ingest_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
