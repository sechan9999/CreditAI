from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from mangum import Mangum
import os
import sys
import joblib
import pandas as pd
import uvicorn

# Ensure src module is in path to load custom class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_scoring_model import CreditScoringModel

# Initialize App
app = FastAPI(
    title="Telco Credit Assessment API",
    description="API for evaluating creditworthiness of telecom applicants",
    version="1.0.0"
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lambda Handler
handler = Mangum(app)

# Load Model
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'scoring_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print(f"[Info] Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[Error] Failed to load model: {e}")
    model = None

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

    @field_validator('credit_history_months')
    @classmethod
    def validate_credit_history(cls, v):
        if v < 0:
            raise ValueError('Credit history months must be non-negative')
        return v

    @field_validator('num_credit_accounts')
    @classmethod
    def validate_credit_accounts(cls, v):
        if v < 0:
            raise ValueError('Number of credit accounts must be non-negative')
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

# Define Decision Logic
def get_decision(score):
    if score >= 600:
        return "Approve", "Minimal/Low Risk"
    elif score >= 550:
        return "Review", "Medium Risk"
    else:
        return "Reject", "High/Very High Risk"

@app.get("/")
def home():
    return {"message": "Welcome to Telco Credit Assessment API. Use /predict to score applicants."}

@app.get("/health")
def health():
    if model:
        return {"status": "healthy", "model_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict")
def predict_credit_score(data: ApplicantData):
    if not model:
        raise HTTPException(status_code=503, detail="Model service unavailable")

    # Prepare input dataframe
    try:
        input_dict = data.model_dump()
    except AttributeError:
        input_dict = data.dict()

    input_data = pd.DataFrame([input_dict])

    try:
        # Calculate Score
        score = float(model.predict_score(input_data)[0])
        score = round(score, 1)

        # Calculate Probability (of being Good)
        prob_good = float(model.predict_proba(input_data)[0])

        # Determine Decision
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
