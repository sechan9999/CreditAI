
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import joblib
import sys

# Ensure clearer text rendering
plt.rcParams['figure.dpi'] = 150

# Add src to path to allow importing CreditScoringModel
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from train_scoring_model import CreditScoringModel
except ImportError:
    # Fallback if run from a different context
    sys.path.append(os.path.join(current_dir, 'src'))
    from train_scoring_model import CreditScoringModel

def main():
    print("Starting SHAP analysis...")
    
    # 1. Define Paths
    # current_dir is '.../src'. Parent is root.
    base_path = os.path.dirname(current_dir) 
    data_path = os.path.join(base_path, 'data', 'raw', 'telecom_data.csv')
    model_path = os.path.join(base_path, 'data', 'processed', 'scoring_model.pkl')
    
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")

    # 2. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Filter approved like in training (as we are explaining the scoring model for approved customers)
    if 'status' in df.columns:
        approved_df = df[df['status'] == 'approved'].copy()
    else:
        approved_df = df.copy()

    feature_cols = ['age', 'income', 'credit_history_months', 
                    'num_credit_accounts', 'debt_ratio', 'num_late_payments']
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in approved_df.columns:
            print(f"Error: Missing feature column '{col}'")
            return

    X = approved_df[feature_cols]
    
    # 3. Load Model
    if not os.path.exists(model_path):
        print("Model not found. Please run train_scoring_model.py first.")
        # Optional: could trigger training here, but better to warn
        return

    try:
        credit_model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Create SHAP Explainer
    # We want to explain the predict_proba output.
    # Because the model includes internal scaling, we pass the original X to the masker
    # and use a wrapper function that calls the model's prediction method.
    
    # Use a background dataset for the masker (e.g., 100 samples)
    masker_data = X.sample(n=100, random_state=42)
    masker = shap.maskers.Independent(data=masker_data)

    def predict_wrapper(data):
        # Ensure input is DataFrame for the custom model pipeline
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data, columns=feature_cols)
        # Return probability of 'Good' (class 1)
        return credit_model.predict_proba(data)

    print("Calculating SHAP values... (this may take a moment)")
    explainer = shap.Explainer(predict_wrapper, masker)
    
    # Calculate SHAP values for a larger sample for the summary plot
    X_display = X.sample(n=500, random_state=42)
    shap_values = explainer(X_display)

    # 5. Generate Summary Plot (Global)
    print("Generating Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_display, show=False)
    # plt.title("SHAP Summary Plot") # summary_plot usually adds its own or looks better without overlapping title
    summary_plot_path = os.path.join(base_path, 'shap_summary.png')
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {summary_plot_path}")

    # 6. Generate Waterfall Plot (Local)
    print("Generating Waterfall Plot...")
    # Find a sample with a relatively low probability (higher risk) to explain a "denial" or risk factor
    probs = predict_wrapper(X_display)
    # Get index of minimum probability
    min_prob_idx = np.argmin(probs)
    
    plt.figure()
    # Create a waterfall plot for the chosen instance
    shap.plots.waterfall(shap_values[min_prob_idx], show=False)
    # plt.title(f"Waterfall Plot (Sample {min_prob_idx})")
    waterfall_plot_path = os.path.join(base_path, 'shap_waterfall.png')
    plt.savefig(waterfall_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {waterfall_plot_path}")
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
