"""Write the trained model's parameters to the one file everything else reads.

There used to be two `model_params.json` files with different key names. This
script wrote `data/processed/model_params.json` with keys `scale_mean` /
`scale_scale` / `coef`, while the page, the Lambda and the tests read
`src/model_params.json` with keys `means` / `scales` / `coefs`. So retraining
never reached the deployed model. The consequence was measured: the page was
standardising income against moments from a run in different units, an 11x
mismatch that declined roughly 190 applicants per 7,000 the trained model
would have approved.

Now there is one file, `src/model_params.json`, with one schema, written here.

It deliberately holds no scale constants. `pdo` and `base_odds` used to live in
this file as well as in `src/scale.py`, `lambda_function.py` and `index.html`,
which is exactly how four copies of a number drift apart. The scale belongs to
`src/scale.py`; this file describes the model.
"""

import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_scoring_model import CreditScoringModel  # noqa: F401,E402  (needed to unpickle)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, 'data', 'processed', 'scoring_model.pkl')
OUT_PATH = os.path.join(BASE, 'src', 'model_params.json')


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'no model at {MODEL_PATH} -- run src/train_scoring_model.py first',
              file=sys.stderr)
        return 1

    wrapper = joblib.load(MODEL_PATH)
    scaler, model = wrapper.scaler, wrapper.model

    params = {
        'features': list(wrapper.feature_cols),
        'means': [float(v) for v in scaler.mean_],
        'scales': [float(v) for v in scaler.scale_],
        'coefs': [float(v) for v in model.coef_[0]],
        'intercept': float(model.intercept_[0]),
    }

    n = len(params['features'])
    for key in ('means', 'scales', 'coefs'):
        assert len(params[key]) == n, f'{key} has {len(params[key])} entries, expected {n}'
    assert all(s > 0 for s in params['scales']), 'a scale of zero would divide by zero'

    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(params, fh, indent=4)
        fh.write('\n')

    print(f'[Info] Written to {OUT_PATH}')
    print()
    print(f'{"feature":<24}{"mean":>14}{"sd":>13}{"coef":>12}')
    print('-' * 63)
    for i, f in enumerate(params['features']):
        print(f'{f:<24}{params["means"][i]:>14.4f}'
              f'{params["scales"][i]:>13.4f}{params["coefs"][i]:>+12.6f}')
    print(f'{"intercept":<24}{"":>27}{params["intercept"]:>+12.6f}')
    print()
    print('Now update the two copies that ship self-contained -- the constants in')
    print('index.html and in lambda_function.py -- and run')
    print('  pytest tests/test_scale_consistency.py')
    print('which fails if any of the three disagree.')

    stale = os.path.join(BASE, 'data', 'processed', 'model_params.json')
    if os.path.exists(stale):
        print()
        print(f'[Warn] A second params file still exists: {stale}')
        print('       Nothing should read it. Delete it.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
