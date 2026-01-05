import math
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

# --- Hardcoded Model Parameters for Lightweight Lambda ---
MODEL_PARAMS = {
    "features": ["age", "income", "credit_history_months", "num_credit_accounts", "debt_ratio", "num_late_payments"],
    "means": [33.71204081632653, 3142.9372472184195, 32.67306122448979, 2.726530612244898, 0.3421508740102015, 1.5673469387755101],
    "scales": [9.775277955146588, 1739.7855186869183, 33.18780893728111, 1.7005131890713896, 0.19394932709543317, 1.5314909042621074],
    "coefs": [0.35040862346963164, 0.6578123678635274, 0.40440611833726114, 0.22966313352192982, -0.742082716270213, -0.9984769472049088],
    "intercept": -0.5984832925687965
}
# -------------------------------------------------------

app = FastAPI(title="Telco Credit API (Lightweight)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ApplicantData(BaseModel):
    age: int
    income: float
    credit_history_months: int
    num_credit_accounts: int
    debt_ratio: float
    num_late_payments: int

def predict(input_dict):
    # 1. Scaling
    scaled_values = []
    for i, feat in enumerate(MODEL_PARAMS['features']):
        val = input_dict[feat]
        mean = MODEL_PARAMS['means'][i]
        scale = MODEL_PARAMS['scales'][i]
        scaled_val = (val - mean) / scale
        scaled_values.append(scaled_val)
    
    # 2. Linear Combination
    log_odds = MODEL_PARAMS['intercept']
    for i, val in enumerate(scaled_values):
        log_odds += val * MODEL_PARAMS['coefs'][i]
        
    # 3. Probability (Sigmoid)
    prob_good = 1 / (1 + math.exp(-log_odds))
    
    # 4. Score Conversion
    base_score = 600
    pdo = 20
    base_odds = 50
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    
    # Odds for scoring (using prob)
    # odds = p / (1-p) same as exp(log_odds)
    # score = offset + factor * log(odds) = offset + factor * log_odds
    
    score = offset + factor * log_odds
    score = max(300, min(850, score)) # Clip
    
    return score, prob_good

def get_decision(score):
    if score >= 600: return "Approve", "Minimal/Low Risk"
    elif score >= 550: return "Review", "Medium Risk"
    else: return "Reject", "High/Very High Risk"

@app.get("/")
def home():
    return {"message": "Telco Credit API (Lightweight Lambda Version)"}

@app.post("/predict")
def predict_endpoint(data: ApplicantData):
    try:
        input_dict = data.dict()
        score, prob = predict(input_dict)
        decision, risk = get_decision(score)
        
        return {
            "applicant_id": "N/A", 
            "credit_score": round(score, 1),
            "probability_good": round(prob, 4),
            "risk_level": risk,
            "decision": decision,
            "input_summary": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)
