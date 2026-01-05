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

def predict(input_dict):
    log_odds = MODEL_PARAMS['intercept']
    for i, feat in enumerate(MODEL_PARAMS['features']):
        val = float(input_dict.get(feat, 0))
        mean = MODEL_PARAMS['means'][i]
        scale = MODEL_PARAMS['scales'][i]
        coef = MODEL_PARAMS['coefs'][i]
        log_odds += ((val - mean) / scale) * coef
        
    prob_good = 1 / (1 + math.exp(-log_odds))
    
    base_score = 600
    pdo = 20
    base_odds = 50
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    
    score = offset + factor * log_odds
    return max(300, min(850, score)), prob_good

def get_decision(score):
    if score >= 600: return "Approve", "Minimal/Low Risk"
    elif score >= 550: return "Review", "Medium Risk"
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
