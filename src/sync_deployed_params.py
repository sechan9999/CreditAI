"""Copy `src/model_params.json` into the two artifacts that ship self-contained.

`index.html` is served as a single static file and `lambda_function.py` as a
deployment bundle, so neither can read the JSON at runtime. Both therefore carry
a copy of the numbers -- and a copy that is maintained by hand is a copy that
drifts. It did: the page ran the accepts-only model against fabricated scaler
moments while the Lambda ran a different fit entirely.

So the copies are generated, not typed. The loop is:

    python src/train_scoring_model.py       # fit, write the pickle
    python src/extract_model_params.py      # pickle -> src/model_params.json
    python src/sync_deployed_params.py      # JSON  -> index.html, lambda_function.py
    pytest tests/test_scale_consistency.py  # fails if any copy disagrees

The scale constants are deliberately not touched here. They live in
`src/scale.py` and are a policy decision, not a training output.
"""

import io
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(BASE, 'src', 'model_params.json')
HTML = os.path.join(BASE, 'index.html')
LAMBDA = os.path.join(BASE, 'lambda_function.py')
CRLF = '\r\n'


def load(path):
    raw = io.open(path, encoding='utf-8', newline='').read()
    return raw.replace(CRLF, '\n'), (CRLF in raw)


def save(path, text, crlf):
    if crlf:
        text = text.replace('\n', CRLF)
    io.open(path, 'w', encoding='utf-8', newline='').write(text)


def sync_html(p):
    s, crlf = load(HTML)
    feats = p['features']

    coefs = ['        const MODEL_COEFS = {',
             f'            intercept: {p["intercept"]!r},']
    for i, f in enumerate(feats):
        coefs.append(f'            {f}: {p["coefs"][i]!r},')
    coefs[-1] = coefs[-1].rstrip(',')
    coefs.append('        };')

    scaler = ['        const SCALER = {', '            mean: {']
    for i, f in enumerate(feats):
        scaler.append(f'                {f}: {p["means"][i]!r},')
    scaler[-1] = scaler[-1].rstrip(',')
    scaler.append('            },')
    scaler.append('            scale: {')
    for i, f in enumerate(feats):
        scaler.append(f'                {f}: {p["scales"][i]!r},')
    scaler[-1] = scaler[-1].rstrip(',')
    scaler.append('            }')
    scaler.append('        };')

    s, n1 = re.subn(r'        const MODEL_COEFS = \{.*?\n        \};',
                    '\n'.join(coefs), s, flags=re.S)
    assert n1 == 1, f'MODEL_COEFS block not found exactly once (n={n1})'
    s, n2 = re.subn(r'        const SCALER = \{.*?\n        \};',
                    '\n'.join(scaler), s, flags=re.S)
    assert n2 == 1, f'SCALER block not found exactly once (n={n2})'

    save(HTML, s, crlf)
    print(f'  index.html          MODEL_COEFS + SCALER updated')


def sync_lambda(p):
    s, crlf = load(LAMBDA)

    def arr(key):
        return '[' + ', '.join(repr(v) for v in p[key]) + ']'

    block = (
        'MODEL_PARAMS = {\n'
        '    "features": [' + ', '.join(f'"{f}"' for f in p['features']) + '],\n'
        f'    "means": {arr("means")},\n'
        f'    "scales": {arr("scales")},\n'
        f'    "coefs": {arr("coefs")},\n'
        f'    "intercept": {p["intercept"]!r}\n'
        '}'
    )
    s, n = re.subn(r'MODEL_PARAMS = \{.*?\n\}', block, s, flags=re.S)
    assert n == 1, f'MODEL_PARAMS block not found exactly once (n={n})'

    save(LAMBDA, s, crlf)
    print(f'  lambda_function.py  MODEL_PARAMS updated')


def main():
    if not os.path.exists(PARAMS):
        print(f'no {PARAMS} -- run src/extract_model_params.py first', file=sys.stderr)
        return 1
    p = json.load(io.open(PARAMS, encoding='utf-8'))

    print(f'Syncing from {os.path.relpath(PARAMS, BASE)}:')
    sync_html(p)
    sync_lambda(p)
    print()
    print('Now run: pytest tests/test_scale_consistency.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
