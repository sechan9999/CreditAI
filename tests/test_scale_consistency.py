"""The scale is written out in three places. This asserts they agree.

`src/scale.py` is the definition. `lambda_function.py` and `index.html` each keep
a copy because both ship as self-contained artifacts and cannot import it -- the
Lambda is a deployment bundle, the page is a single HTML file served statically.

A copy that can drift will drift. It already had: the page was running invented
scaler statistics and a different set of coefficients, and scored the same
applicant 122 points below the trained model. These tests are the thing that
would have caught it.
"""

import io
import json
import math
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import scale  # noqa: E402
import lambda_function as lam  # noqa: E402

HTML = io.open(os.path.join(ROOT, 'index.html'), encoding='utf-8').read()


def js_number(name):
    """Pull a literal `const NAME = <number>` out of index.html.

    Handles both `const A = 1;` and `const A = 1, B = 2;`. Only literals are
    readable this way -- the derived constants are checked by comparing the
    literals they are built from, plus a textual check that the page uses the
    same formulas.
    """
    m = re.search(r'(?:const\s+|,\s*)' + name + r'\s*=\s*(-?[\d.]+)\s*[,;]', HTML)
    assert m, f'{name} not found as a numeric const in index.html'
    return float(m.group(1))


class TestLambdaMatchesScale:
    @pytest.mark.parametrize('name', [
        'BASE_SCORE', 'PDO', 'BASE_ODDS', 'APPROVE_ODDS', 'REVIEW_ODDS',
    ])
    def test_constant(self, name):
        assert getattr(lam, name) == getattr(scale, name), f'{name} differs'

    @pytest.mark.parametrize('name', ['SCORE_FACTOR', 'SCORE_OFFSET', 'APPROVE_AT', 'REVIEW_AT'])
    def test_derived(self, name):
        assert getattr(lam, name) == pytest.approx(getattr(scale, name))

    def test_same_decision_across_the_range(self):
        for s in range(300, 851, 5):
            assert lam.get_decision(s)[0] == scale.decision(s)[0], f'differ at {s}'


class TestPageMatchesScale:
    @pytest.mark.parametrize('name', ['BASE_SCORE', 'PDO', 'BASE_ODDS'])
    def test_scale_constant(self, name):
        assert js_number(name) == pytest.approx(getattr(scale, name)), f'{name} differs'

    @pytest.mark.parametrize('js,py', [
        ('PREV_PDO', '_PREV_PDO'),
        ('PREV_BASE_SCORE', '_PREV_BASE_SCORE'),
        ('PREV_BASE_ODDS', '_PREV_BASE_ODDS'),
    ])
    def test_policy_input_constant(self, js, py):
        """The policy bars are derived, so compare what they are derived from."""
        assert js_number(js) == pytest.approx(getattr(scale, py)), f'{js} differs'

    @pytest.mark.parametrize('expr', [
        'PREV_PDO / Math.LN2',
        'PREV_BASE_SCORE - PREV_FACTOR * Math.log(PREV_BASE_ODDS)',
        'Math.exp((550 - PREV_OFFSET) / PREV_FACTOR)',
        'PDO / Math.LN2',
        'BASE_SCORE - SCORE_FACTOR * Math.log(BASE_ODDS)',
        'SCORE_OFFSET + SCORE_FACTOR * Math.log(APPROVE_ODDS)',
        'SCORE_OFFSET + SCORE_FACTOR * Math.log(REVIEW_ODDS)',
    ])
    def test_derivation_formula_present(self, expr):
        """The derived values cannot be read by regex, so check the arithmetic.

        If the page ever hardcodes one of these instead of deriving it, that is
        the point at which it can drift from src/scale.py, so the formula being
        present is the thing worth asserting.
        """
        assert expr in HTML, f'page no longer derives: {expr}'

    def test_page_model_matches_trained_params(self):
        """The page's hardcoded coefficients must be the trained ones.

        This is the assertion that the 122-point bug would have failed.
        """
        params = json.load(io.open(os.path.join(ROOT, 'src', 'model_params.json'), encoding='utf-8'))
        feats = params['features']

        m = re.search(r'const MODEL_COEFS = \{(.*?)\};', HTML, re.S)
        assert m, 'MODEL_COEFS not found in index.html'
        coefs = dict(re.findall(r'(\w+)\s*:\s*(-?[\d.eE+-]+)', m.group(1)))

        assert float(coefs['intercept']) == pytest.approx(params['intercept'])
        for i, f in enumerate(feats):
            assert float(coefs[f]) == pytest.approx(params['coefs'][i]), f'coef {f} differs'

        m = re.search(r'const SCALER = \{(.*?)\n        \};', HTML, re.S)
        assert m, 'SCALER not found in index.html'
        means = dict(re.findall(r'(\w+)\s*:\s*(-?[\d.eE+-]+)', m.group(1).split('scale:')[0]))
        scales = dict(re.findall(r'(\w+)\s*:\s*(-?[\d.eE+-]+)', m.group(1).split('scale:')[1]))
        for i, f in enumerate(feats):
            assert float(means[f]) == pytest.approx(params['means'][i]), f'mean {f} differs'
            assert float(scales[f]) == pytest.approx(params['scales'][i]), f'scale {f} differs'

    def test_page_has_no_literal_decision_thresholds(self):
        """Decisions must come from the derived cutoffs, not from numbers."""
        for bad in ['score >= 600', 'score >= 550', 'lastScore >= 600', 'lastScore >= 550']:
            assert bad not in HTML, f'literal threshold left in the page: {bad}'


class TestPolicyIsPreservedByRecalibration:
    """The point of stating the policy in odds.

    Rescaling changes every score but must change no decision, so the odds a
    cutoff represents is the thing that has to stay fixed.
    """

    def test_cutoff_odds_are_the_incumbent_bars(self):
        assert scale.APPROVE_ODDS == 5
        old_factor = 20 / math.log(2)
        old_offset = 600 - old_factor * math.log(5)
        old_review_odds = math.exp((550 - old_offset) / old_factor)
        assert scale.REVIEW_ODDS == pytest.approx(old_review_odds, rel=1e-6)

    def test_cutoffs_ordered_and_inside_range(self):
        assert scale.SCORE_MIN < scale.REVIEW_AT < scale.APPROVE_AT < scale.SCORE_MAX

    def test_scale_is_monotone(self):
        prev = -1
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            s = scale.score_from_prob(p)
            assert s > prev
            prev = s

    def test_anchor_holds(self):
        assert scale.score_from_odds(scale.BASE_ODDS) == pytest.approx(scale.BASE_SCORE)

    def test_pdo_holds(self):
        """Doubling the odds must move the score by exactly PDO points."""
        a = scale.score_from_odds(2.0)
        b = scale.score_from_odds(4.0)
        assert b - a == pytest.approx(scale.PDO, abs=1e-6)
