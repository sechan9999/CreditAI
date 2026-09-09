"""Write the trained XGBoost signal's trees to the one file the page reads.

Mirrors `extract_model_params.py`: the pickle is the training artifact, this
JSON is the single description of the deployed tree ensemble, and
`sync_xgb_params.py` copies it into `index.html` because the page is a static
file and cannot read JSON off disk at runtime.

Unlike the logistic scorecard, this model needs no scaler -- tree splits
compare a raw feature value against a threshold, so standardising first would
only relabel the thresholds. What has to survive the trip to JSON is the tree
structure itself: for each internal node, which feature and threshold it
splits on and which child handles a value below versus at-or-above it; for
each leaf, the value it contributes to the sum.

`base_score` was fixed to 0.5 at training time (see `train_xgb_signal.py`),
which is `logit(0.5) = 0` in margin space, so no separate base-margin constant
needs to travel alongside the trees -- the sum of leaf values a row lands in,
across every tree, IS the margin. That is verified below, not assumed: every
row's dumped-tree sum is checked against the booster's own `output_margin`
prediction before anything is written out.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, 'data', 'processed', 'xgb_signal_model.pkl')
OUT_PATH = os.path.join(BASE, 'src', 'xgb_model_params.json')
DATA_PATH = os.path.join(BASE, 'data', 'raw', 'telecom_data.csv')


def compact_tree(node, feature_index):
    """Recursively turn one `get_dump(dump_format='json')` tree into a small,
    JS-friendly shape: {f, th, yes, no, default} for a split, {leaf} for a leaf.

    `yes`/`no`/`missing` in the raw dump are node ids; `children` carries the
    actual child dicts, in no guaranteed order, so children are matched to
    `yes`/`no` by id rather than by position.
    """
    if 'leaf' in node:
        return {'leaf': float(node['leaf'])}

    by_id = {c['nodeid']: c for c in node['children']}
    yes_id, no_id, missing_id = node['yes'], node['no'], node.get('missing', node['yes'])
    feat = node['split']
    return {
        'f': feature_index[feat],
        'th': float(node['split_condition']),
        'yes': compact_tree(by_id[yes_id], feature_index),
        'no': compact_tree(by_id[no_id], feature_index),
        'default': 'yes' if missing_id == yes_id else 'no',
    }


def eval_tree(node, row):
    """Pure-Python walker -- the reference the generated JS must match.

    XGBoost builds its DMatrix in float32 and splits at float32 thresholds
    (dumped as e.g. 0.224000007, the float32 value nearest a training row's
    own 0.224). Comparing at float64 precision instead disagrees right at
    that boundary -- 0.224 < 0.224000007 is True in float64 but False once
    both sides are rounded to the float32 XGBoost actually compared -- which
    silently sent rows down the wrong branch on 13 of 8,000 (row, tree) pairs
    before this cast was added. The embedded JS walker rounds the same way,
    with `Math.fround`.
    """
    while 'leaf' not in node:
        v = row[node['f']]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            branch = node['default']
        else:
            branch = 'yes' if np.float32(v) < np.float32(node['th']) else 'no'
        node = node[branch]
    return node['leaf']


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'no model at {MODEL_PATH} -- run src/train_xgb_signal.py first', file=sys.stderr)
        return 1

    saved = joblib.load(MODEL_PATH)
    model, features = saved['model'], saved['feature_cols']
    booster = model.get_booster()
    feature_index = {f: i for i, f in enumerate(features)}

    dumps = booster.get_dump(dump_format='json')
    trees = [compact_tree(json.loads(d), feature_index) for d in dumps]

    params = {
        'features': features,
        'objective': 'binary:logistic',
        'base_score': 0.5,
        'trees': trees,
    }

    # Verify against the booster's own output before writing anything: sum the
    # compacted trees for every row in the dataset and compare to XGBoost's
    # own margin prediction. This is the check that would catch a wrong split
    # direction or a mismatched child id, not a smoke test.
    df = pd.read_csv(DATA_PATH)
    X = df[features]
    margin = booster.predict(xgb.DMatrix(X), output_margin=True)
    py_margin = np.array([sum(eval_tree(t, row) for t in trees) for row in X.itertuples(index=False)])
    max_err = float(np.max(np.abs(margin - py_margin)))
    assert max_err < 1e-6, f'compacted trees disagree with the booster by {max_err}'

    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(params, fh, indent=2)
        fh.write('\n')

    n_nodes = sum(_count_nodes(t) for t in trees)
    print(f'[Info] {len(trees)} trees, {n_nodes} nodes total, written to {OUT_PATH}')
    print(f'[Info] Max |dumped-tree sum - booster output_margin| over {len(X):,} rows: {max_err:.2e}')
    print('Now run: python src/sync_xgb_params.py')
    return 0


def _count_nodes(node):
    if 'leaf' in node:
        return 1
    return 1 + _count_nodes(node['yes']) + _count_nodes(node['no'])


if __name__ == '__main__':
    sys.exit(main())
