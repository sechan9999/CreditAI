from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from mangum import Mangum
import json
import os
import math
import scale

# Initialize App
app = FastAPI(
    title="Telco Credit Assessment API (Lite)",
    description="Lightweight API for AWS Lambda",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = Mangum(app)

# Load Parameters
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# One params file, one schema: src/model_params.json, written by
# src/extract_model_params.py. This used to read a second file in
# data/processed/ with different key names, which is how retraining stopped
# reaching anything that served predictions.
PARAMS_PATH = os.path.join(BASE_PATH, 'src', 'model_params.json')

try:
    with open(PARAMS_PATH, 'r') as f:
        params = json.load(f)
    print(f"[Info] Params loaded successfully from {PARAMS_PATH}")
except Exception as e:
    print(f"[Error] Failed to load params: {e}")
    params = None

# Inference Logic (No Sklearn/Pandas)
def predict(input_dict):
    if not params:
        raise ValueError("Model parameters not loaded")

    features = params['features']
    scale_mean = params['means']
    scale_scale = params['scales']
    coef = params['coefs']
    intercept = params['intercept']

    # 1. Prepare Vector & Scale
    x_scaled = []
    for i, feat in enumerate(features):
        val = input_dict.get(feat)
        if val is None:
            raise ValueError(f"Missing feature: {feat}")

        # Standard Scaler: z = (x - u) / s
        scaled_val = (val - scale_mean[i]) / scale_scale[i]
        x_scaled.append(scaled_val)

    # 2. Linear Combination
    linear_pred = intercept + sum(x * c for x, c in zip(x_scaled, coef))

    # 3. Sigmoid (Probability)
    prob = 1 / (1 + math.exp(-linear_pred))

    # 4. Score
    #
    # The scale comes from src/scale.py, not from the params file. It used to be
    # carried in the params JSON as well, which meant pdo and base_odds existed in
    # four places at once and drifted apart.
    odds = prob / (1 - prob + 1e-10)
    score = scale.SCORE_OFFSET + scale.SCORE_FACTOR * math.log(odds + 1e-10)
    score = max(scale.SCORE_MIN, min(scale.SCORE_MAX, score))

    return score, prob

# Define Input Schema with validation
class ApplicantData(BaseModel):
    age: int
    income: float
    credit_history_months: int
    num_credit_accounts: int
    debt_ratio: float
    num_late_payments: int

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 18 or v > 100:
            raise ValueError('Age must be between 18 and 100')
        return v

    @field_validator('income')
    @classmethod
    def validate_income(cls, v):
        if v < 0:
            raise ValueError('Income must be non-negative')
        return v

    @field_validator('debt_ratio')
    @classmethod
    def validate_debt_ratio(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Debt ratio must be between 0 and 1')
        return v

    @field_validator('num_late_payments')
    @classmethod
    def validate_late_payments(cls, v):
        if v < 0:
            raise ValueError('Number of late payments must be non-negative')
        return v

def get_decision(score):
    if score >= 600:
        return "Approve", "Minimal/Low Risk"
    elif score >= 550:
        return "Review", "Medium Risk"
    else:
        return "Reject", "High/Very High Risk"

@app.get("/")
def home():
    return {"message": "Welcome to Telco Credit Assessment API (Lite)"}

@app.get("/health")
def health():
    if params:
        return {"status": "healthy", "params_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Model parameters not loaded")

@app.post("/predict")
def predict_credit_score(data: ApplicantData):
    try:
        try:
            input_dict = data.model_dump()
        except AttributeError:
            input_dict = data.dict()

        score, prob_good = predict(input_dict)

        score = round(score, 1)
        decision, risk_level = get_decision(score)

        return {
            "applicant_id": "N/A",
            "credit_score": score,
            "probability_good": round(prob_good, 4),
            "risk_level": risk_level,
            "decision": decision,
            "input_summary": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
