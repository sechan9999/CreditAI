"""A reject-inference method must not depend on how the score is displayed.

Parceling used to bin on the 300-850 score with equal-width bins. That score is
an affine transform of log-odds with a chosen PDO and anchor, and it is clipped
at both ends, so the bin edges moved whenever the scorecard was rescaled. With
identical data and an identical seed, the method assigned 261, 281 and 278
rejected applicants to Good under three different display scales.

Rescaling a scorecard changes no decision -- it is a relabelling. It must
therefore not change which applicants a reject-inference method calls good, or
the labels the model trains on become a function of a cosmetic choice.

These tests pin that down.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import scale  # noqa: E402
from reject_inference_methods import RejectInference  # noqa: E402

FEATURES = ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']

SCALES = [(20, 5), (40, 1), (75, 3)]


@pytest.fixture(scope='module')
def populations():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'telecom_data.csv'))
    return (df[df['status'] == 'approved'].copy(),
            df[df['status'] == 'rejected'].copy())


def _assign_under(pdo, base_odds, approved, rejected, monkeypatch):
    """Run parceling with the module's display scale set to (pdo, base_odds)."""
    factor = pdo / math.log(2)
    monkeypatch.setattr(scale, 'PDO', pdo)
    monkeypatch.setattr(scale, 'BASE_ODDS', base_odds)
    monkeypatch.setattr(scale, 'SCORE_FACTOR', factor)
    monkeypatch.setattr(scale, 'SCORE_OFFSET', scale.BASE_SCORE - factor * math.log(base_odds))

    ri = RejectInference(approved, rejected, FEATURES, target_col='target')
    _, rejected_copy, _ = ri.parceling()
    return rejected_copy['target'].to_numpy()


class TestParcelingIsScaleInvariant:
    def test_same_applicants_labelled_good(self, populations, monkeypatch):
        approved, rejected = populations
        runs = [_assign_under(p, b, approved, rejected, monkeypatch) for p, b in SCALES]
        for i in range(1, len(runs)):
            assert np.array_equal(runs[0], runs[i]), (
                f'parceling assignment changed between PDO {SCALES[0]} and {SCALES[i]}: '
                f'{int(runs[0].sum())} vs {int(runs[i].sum())} labelled good'
            )

    def test_bins_are_populated(self, populations):
        """Quantile bins exist so no bad rate is estimated from a handful of rows.

        The equal-width version produced bins holding 3 and 11 approved
        applicants, and rejects landing in them inherited a bad rate of exactly
        1.0 or 0.0 estimated from those few observations.
        """
        approved, rejected = populations
        ri = RejectInference(approved, rejected, FEATURES, target_col='target')
        probs = ri.base_model.predict_proba(approved)
        edges = np.unique(np.quantile(probs, np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        counts = pd.cut(pd.Series(probs), bins=edges, labels=False).value_counts()
        assert counts.min() >= 100, f'a quantile bin holds only {counts.min()} approved applicants'

    def test_does_not_bin_on_the_displayed_score(self):
        """The binning must not read predict_score, which carries the scale."""
        src = open(os.path.join(ROOT, 'src', 'reject_inference_methods.py'), encoding='utf-8').read()
        body = src[src.index('def parceling'):]
        body = body[:body.index('return combined')]
        assert 'predict_score' not in body, (
            'parceling reads predict_score again -- it is back on the display scale'
        )
