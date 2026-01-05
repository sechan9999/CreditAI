import joblib
import pandas as pd
import os
import json

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_path, 'data', 'processed', 'best_scoring_model.pkl')

try:
    model_obj = joblib.load(model_path)
    
    # Extract Scaler Params
    means = model_obj.scaler.mean_.tolist()
    scales = model_obj.scaler.scale_.tolist()
    
    # Extract Model Params
    coefs = model_obj.model.coef_[0].tolist()
    intercept = model_obj.model.intercept_[0]
    
    feature_names = model_obj.feature_cols
    
    params = {
        'features': feature_names,
        'means': means,
        'scales': scales,
        'coefs': coefs,
        'intercept': intercept
    }
    
    # Print formatted for easy copy-paste or programmatic use
    print(json.dumps(params, indent=4))
    
    # Save to file just in case
    with open(os.path.join(base_path, 'src', 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)
        
except Exception as e:
    print(e)
