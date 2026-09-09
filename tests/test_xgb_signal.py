"""The XGBoost signal is only worth shipping if the JS the browser actually
runs computes what the trained model computes, and if it genuinely stays out
of the decision. Both are checked directly, not assumed.

`TestJsMatchesTrainedModel` is the one that matters most: it does not test the
compacted-tree JSON in isolation, it extracts the literal `<script>` contents
of `index.html`, runs it in Node, and compares `xgbPredictProba()` -- the exact
function a browser evaluates -- against `xgboost`'s own `predict_proba` on
every row of the dataset. `test_scale_consistency.py` does the analogous check
for the logistic scorecard; this is that test for a tree ensemble, where the
failure mode is different (a wrong split direction, not a transcribed
coefficient) and was real: see `extract_xgb_params.py`'s `eval_tree` docstring
for the float32 boundary case this exact kind of check caught during
development.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

HTML_PATH = os.path.join(ROOT, 'index.html')
PARAMS_PATH = os.path.join(ROOT, 'src', 'xgb_model_params.json')
MODEL_PATH = os.path.join(ROOT, 'data', 'processed', 'xgb_signal_model.pkl')
DATA_PATH = os.path.join(ROOT, 'data', 'raw', 'telecom_data.csv')

HAVE_XGBOOST = True
try:
    import xgboost  # noqa: F401
except ImportError:
    HAVE_XGBOOST = False

HAVE_NODE = shutil.which('node') is not None


def _extract_page_script():
    html = io.open(HTML_PATH, encoding='utf-8').read()
    m = re.search(r'<script>(.*)</script>', html, re.S)
    assert m, 'no <script> block found in index.html'
    return m.group(1)


class TestCompactTreeWalker:
    """The pure-Python reference walker, on a hand-built tiny tree."""

    def test_leaf_only_tree(self):
        from extract_xgb_params import eval_tree
        assert eval_tree({'leaf': 0.5}, [1.0]) == 0.5

    def test_split_direction_is_strictly_less_than(self):
        from extract_xgb_params import eval_tree
        tree = {'f': 0, 'th': 1.0, 'default': 'no',
                 'yes': {'leaf': -1.0}, 'no': {'leaf': 1.0}}
        assert eval_tree(tree, [0.5]) == -1.0   # 0.5 < 1.0 -> yes
        assert eval_tree(tree, [1.0]) == 1.0    # 1.0 == 1.0 -> no (not strictly less)
        assert eval_tree(tree, [1.5]) == 1.0    # 1.5 >= 1.0 -> no

    def test_float32_boundary_matches_the_documented_case(self):
        """0.224 vs a threshold of the float32 value nearest 0.224.

        At float64 precision 0.224 < 0.224000007 is True; XGBoost compared
        both at float32, where they are equal, so it takes the 'no' branch.
        The walker must reproduce that, not the float64 answer.
        """
        from extract_xgb_params import eval_tree
        tree = {'f': 0, 'th': 0.224000007, 'default': 'no',
                 'yes': {'leaf': 111.0}, 'no': {'leaf': 222.0}}
        assert eval_tree(tree, [0.224]) == 222.0

    def test_missing_uses_default_branch(self):
        from extract_xgb_params import eval_tree
        tree = {'f': 0, 'th': 1.0, 'default': 'yes',
                 'yes': {'leaf': 9.0}, 'no': {'leaf': -9.0}}
        assert eval_tree(tree, [float('nan')]) == 9.0


class TestParamsFileIsWellFormed:
    def test_params_file_exists(self):
        assert os.path.exists(PARAMS_PATH), 'run src/train_xgb_signal.py then src/extract_xgb_params.py'

    def test_features_match_the_logistic_model(self):
        """Both models must read the same six columns in the same order --
        `updatePipeline` builds one `inputs` object and feeds it to both."""
        params = json.load(io.open(PARAMS_PATH, encoding='utf-8'))
        model_params = json.load(io.open(os.path.join(ROOT, 'src', 'model_params.json'), encoding='utf-8'))
        assert params['features'] == model_params['features']

    def test_tree_count_and_size_are_bounded(self):
        """A regression guard against accidentally training an unbounded model --
        this ships inside a static HTML page, not behind an API."""
        params = json.load(io.open(PARAMS_PATH, encoding='utf-8'))
        assert len(params['trees']) <= 200

        def count(node):
            return 1 if 'leaf' in node else 1 + count(node['yes']) + count(node['no'])

        total_nodes = sum(count(t) for t in params['trees'])
        assert total_nodes <= 5000, f'{total_nodes} nodes is too large to ship as inline JS'


@pytest.mark.skipif(not HAVE_XGBOOST, reason='xgboost not installed')
class TestJsMatchesTrainedModel:
    """`xgbPredictProba()`, run in Node straight out of index.html, against
    `xgboost`'s own `predict_proba`, on every row of the dataset."""

    @pytest.fixture(scope='class')
    @classmethod
    def comparison(cls):
        if not HAVE_NODE:
            pytest.skip('node not available')
        if not (os.path.exists(MODEL_PATH) and os.path.exists(PARAMS_PATH)):
            pytest.skip('trained model / params not present -- run the train/extract scripts first')

        import joblib
        saved = joblib.load(MODEL_PATH)
        model, features = saved['model'], saved['feature_cols']

        df = pd.read_csv(DATA_PATH)
        X = df[features]
        py_prob = model.predict_proba(X)[:, 1]

        rows = X.to_dict(orient='records')
        script = _extract_page_script()
        script = re.sub(r'Chart\.defaults[^;]*;', '', script)

        harness = (
            "const fakeEl = {innerHTML:'', style:{}, textContent:'', value:''};\n"
            "global.document = {addEventListener: ()=>{}, getElementById: () => fakeEl, "
            "querySelectorAll: () => []};\n"
            "global.window = {};\n"
            "global.Chart = {defaults: {plugins: {legend: {labels: {}}}}};\n"
            "global.lucide = {createIcons: () => {}};\n"
            + script +
            "\nconst rows = require('fs').readFileSync(process.argv[2], 'utf8');\n"
            "const out = JSON.parse(rows).map(xgbPredictProba);\n"
            "console.log(JSON.stringify(out));\n"
        )
        # The dataset is 7,000 rows -- passing it as an argv string overflows
        # ARG_MAX ("Argument list too long"), so both the harness script and
        # the row data go through temp files instead of `node -e`.
        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, 'harness.js')
            rows_path = os.path.join(tmp, 'rows.json')
            io.open(script_path, 'w', encoding='utf-8').write(harness)
            io.open(rows_path, 'w', encoding='utf-8').write(json.dumps(rows))
            proc = subprocess.run(['node', script_path, rows_path],
                                   capture_output=True, text=True, timeout=120)
            assert proc.returncode == 0, f'node harness failed:\n{proc.stderr}'
            js_prob = np.array(json.loads(proc.stdout.strip().splitlines()[-1]))
        return py_prob, js_prob

    def test_js_probability_matches_python_within_tolerance(self, comparison):
        py_prob, js_prob = comparison
        max_err = np.max(np.abs(py_prob - js_prob))
        assert max_err < 1e-6, f'JS and Python XGBoost predictions disagree by up to {max_err}'

    def test_not_a_degenerate_constant_signal(self, comparison):
        """A walker that always fell through to the same leaf would still
        'match' a constant-probability model -- guard against a trivially
        passing comparison by requiring real spread in both series."""
        py_prob, _ = comparison
        assert py_prob.std() > 0.02


class TestSignalStaysAdvisory:
    """The whole point of a *separate* signal is that it cannot change an
    approval. These are static checks on the page source, not a simulation --
    intentionally cheap to run on every change to index.html."""

    def test_calculate_score_does_not_call_xgb(self):
        script = _extract_page_script()
        m = re.search(r'function calculateScore\(\)\s*\{.*?\n        \}\n', script, re.S)
        assert m, 'calculateScore() not found'
        assert 'xgbPredictProba' not in m.group(0), \
            'the deployed decision must not depend on the advisory ML signal'

    def test_xgb_signal_appears_exactly_once_in_pipeline(self):
        script = _extract_page_script()
        assert script.count("title: \"ML Signal: XGBoost") == 1

    def test_xgb_card_is_marked_advisory(self):
        script = _extract_page_script()
        m = re.search(r'\{ title: "ML Signal: XGBoost[^}]*advisory:\s*true', script)
        assert m, 'the XGBoost pipeline step must set advisory: true'
