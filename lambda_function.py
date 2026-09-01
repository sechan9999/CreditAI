import json
import math
import base64

# --- Hardcoded Model Parameters ---
MODEL_PARAMS = {
    "features": ["age", "income", "credit_history_months", "num_credit_accounts", "debt_ratio", "num_late_payments"],
    "means": [33.712, 3142.937, 32.673, 2.727, 0.342, 1.567],
    "scales": [9.775, 1739.786, 33.188, 1.701, 0.194, 1.531],
    "coefs": [0.3504, 0.6578, 0.4044, 0.2297, -0.7421, -0.9985],
    "intercept": -0.5985
}

# --- Scale ---
# score = SCORE_OFFSET + SCORE_FACTOR * ln(odds), where ln(odds) is the model's
# linear predictor. Must stay identical to the constants in index.html.
#
# Recalibrated from PDO 20 with 600 at 5:1. On that scale the 7,000-applicant
# population spanned only 300-658, so the top third of a declared 300-850 range
# was never used. PDO 40 with 600 at even money spreads the same population
# across 300-809, and the anchor states itself: 600 points means an even chance
# of being good.
BASE_SCORE = 600
PDO = 40          # points to double the odds
BASE_ODDS = 1     # odds of being good at BASE_SCORE -- even money
SCORE_FACTOR = PDO / math.log(2)
SCORE_OFFSET = BASE_SCORE - SCORE_FACTOR * math.log(BASE_ODDS)

# --- Policy ---
# Stated in odds; the point cutoffs are derived. These are the bars the previous
# scale's 600 and 550 represented, preserved exactly, which is what makes the
# recalibration a relabelling rather than a change of policy.
# Derived from the previous scale rather than transcribed. Writing the review bar
# out as a literal is how a rounding slip becomes a policy change: a hand-computed
# 0.8839527 against the exact 0.8838834764831849 moves the boundary just enough to
# flip an applicant sitting on it. Deriving it makes that impossible.
#
# These are the bars the previous scale's 600 and 550 points represented. Replacing
# them with chosen round odds is a policy decision and should be made as one.
_PREV_PDO, _PREV_BASE_SCORE, _PREV_BASE_ODDS = 20, 600, 5
_prev_factor = _PREV_PDO / math.log(2)
_prev_offset = _PREV_BASE_SCORE - _prev_factor * math.log(_PREV_BASE_ODDS)

APPROVE_ODDS = _PREV_BASE_ODDS                                # the old 600 bar, p(good) = 83.3%
REVIEW_ODDS = math.exp((550 - _prev_offset) / _prev_factor)    # the old 550 bar, p(good) = 46.9%
APPROVE_AT = SCORE_OFFSET + SCORE_FACTOR * math.log(APPROVE_ODDS)
REVIEW_AT = SCORE_OFFSET + SCORE_FACTOR * math.log(REVIEW_ODDS)

def predict(input_dict):
    log_odds = MODEL_PARAMS['intercept']
    for i, feat in enumerate(MODEL_PARAMS['features']):
        val = float(input_dict.get(feat, 0))
        mean = MODEL_PARAMS['means'][i]
        scale = MODEL_PARAMS['scales'][i]
        coef = MODEL_PARAMS['coefs'][i]
        log_odds += ((val - mean) / scale) * coef

    prob_good = 1 / (1 + math.exp(-log_odds))
    odds = prob_good / (1 - prob_good + 1e-10)

    score = SCORE_OFFSET + SCORE_FACTOR * math.log(odds + 1e-10)
    return max(300, min(850, score)), prob_good

def get_decision(score):
    """Decide from the derived point cutoffs, never from literals.

    The policy lives in APPROVE_ODDS and REVIEW_ODDS. Because the cutoffs are
    derived from the same scale the score is, a recalibration moves both
    together and no applicant's decision changes.
    """
    if score >= APPROVE_AT: return "Approve", "Minimal/Low Risk"
    elif score >= REVIEW_AT: return "Review", "Medium Risk"
    else: return "Reject", "High/Very High Risk"

def lambda_handler(event, context):
    # REMOVED MANUAL CORS HEADERS TO AVOID CONFLICT WITH AWS CONSOLE SETTINGS
    
    try:
        # Body Parsing
        body_raw = event.get('body')
        is_base64 = event.get('isBase64Encoded', False)
        
        if not body_raw:
            body = event if 'age' in event else {}
        else:
            if is_base64:
                body_raw = base64.b64decode(body_raw).decode('utf-8')
            body = json.loads(body_raw)
        
        # Predict
        score, prob = predict(body)
        decision, risk = get_decision(score)
        
        return {
            'statusCode': 200,
            # No headers here, AWS Lambda Function URL config adds them!
            'body': json.dumps({
                "credit_score": round(score, 1),
                "probability_good": round(prob, 4),
                "risk_level": risk,
                "decision": decision
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
