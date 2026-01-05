
import os
import joblib
import json
import numpy as np
import pandas as pd
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_scoring_model import CreditScoringModel

# Load Model
current_dir = os.path.dirname(os.path.abspath(__file__))
# Depending on where this script is run, adjust path
# Assuming run from root
model_path = os.path.join('data', 'processed', 'scoring_model.pkl')

if not os.path.exists(model_path):
    # Try src relative
    model_path = os.path.join(current_dir, '..', 'data', 'processed', 'scoring_model.pkl')

print(f"Loading model from: {model_path}")
model_wrapper = joblib.load(model_path)
model = model_wrapper.model
scaler = model_wrapper.scaler

# Extract Parameters
params = {
    'features': model_wrapper.feature_cols,
    'scale_mean': scaler.mean_.tolist(),
    'scale_scale': scaler.scale_.tolist(),
    'coef': model.coef_[0].tolist(),
    'intercept': model.intercept_[0],
    'base_score': 600,
    'pdo': 20,
    'base_odds': 5 # Updated base_odds
}

# Save to JSON
output_path = os.path.join('data', 'processed', 'model_params.json')
with open(output_path, 'w') as f:
    json.dump(params, f, indent=4)

print(f"Model parameters saved to: {output_path}")
print(params)
