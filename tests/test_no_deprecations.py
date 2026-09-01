"""A canary for library deprecations in the model we actually ship.

`penalty='l2'` sat in `CreditScoringModel` for a long time. It was the default
in every scikit-learn this project supports, so it bought nothing, and from
scikit-learn 1.8 it emitted a FutureWarning saying it would be removed in 1.10.
The warning was visible in every single test run and nobody acted on it, which
is the normal fate of a warning that scrolls past 17 times.

So the warning is a test now. If a future scikit-learn deprecates something else
this model passes, this fails and someone looks -- which is the entire point.

This test is deliberately strict, and it will occasionally fail for a reason
that is not a bug: an upstream deprecation in a code path we do not control.
That is an acceptable trade. Read the message, and if it is genuinely not ours,
narrow the filter rather than deleting the test.
"""

import ast
import os
import sys
import warnings

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from train_scoring_model import CreditScoringModel  # noqa: E402

FEATURES = ['age', 'income', 'credit_history_months',
            'num_credit_accounts', 'debt_ratio', 'num_late_payments']


@pytest.fixture(scope='module')
def approved():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'telecom_data.csv'))
    return df[df['status'] == 'approved'].copy()


def _fit_capturing(approved):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = CreditScoringModel(FEATURES)
        model.fit(approved, approved['target'].astype(int))
        model.predict_proba(approved)
        model.predict_score(approved)
    return model, caught


class TestNoDeprecationWarnings:
    def test_fit_and_score_raise_no_future_warnings(self, approved):
        _, caught = _fit_capturing(approved)
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert not future, (
            'fitting or scoring raised FutureWarning(s):\n  '
            + '\n  '.join(f'{w.category.__name__}: {w.message}' for w in future)
        )

    def test_fit_and_score_raise_no_deprecation_warnings(self, approved):
        _, caught = _fit_capturing(approved)
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not dep, (
            'fitting or scoring raised DeprecationWarning(s):\n  '
            + '\n  '.join(f'{w.category.__name__}: {w.message}' for w in dep)
        )

    def test_penalty_is_not_passed_explicitly(self):
        """The specific thing that was wrong, pinned so it cannot come back.

        Passing `penalty` is what produced the warning. The documented migration
        is `l1_ratio=0`, which would raise on the older scikit-learn that
        requirements.txt still permits, so the right answer is to pass neither
        and let the default stand.

        Parsed with `ast` rather than searched for as text. The first version of
        this test sliced the source between the call's parentheses and looked for
        the word, and it failed on the comment explaining why the argument was
        removed -- a substring match cannot tell an argument from prose about an
        argument.
        """
        source = open(os.path.join(ROOT, 'src', 'train_scoring_model.py'),
                      encoding='utf-8').read()
        tree = ast.parse(source)

        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == 'LogisticRegression']
        assert calls, 'no LogisticRegression call found in train_scoring_model.py'

        for call in calls:
            kwargs = {kw.arg for kw in call.keywords if kw.arg}
            assert 'penalty' not in kwargs, 'penalty is being passed again'
            assert 'l1_ratio' not in kwargs, (
                'l1_ratio raises on scikit-learn below 1.8, which requirements.txt allows'
            )
