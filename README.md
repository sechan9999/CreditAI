# Telco Credit Assessment Pipeline

This project builds a comprehensive End-to-End machine learning pipeline to assess the creditworthiness of applicants for telephone company subscriptions. It simulates the entire lifecycle of a credit scoring project from data generation to API deployment.

## Project Structure

```
telco_credit_assessment/
├── data/
│   ├── raw/                  # Generated synthetic data
│   └── processed/            # Trained models and augemented datasets
├── reports/                  # Analysis reports and charts
└── src/
    ├── generate_data.py            # Step 1: Generate synthetic telecom data
    ├── generate_report.py          # Step 2: EDA and HTML report generation
    ├── train_scoring_model.py      # Step 3: Train baseline Logistic Regression
    ├── reject_inference.py         # Step 4: Concept proof of reject inference
    ├── reject_inference_methods.py # Step 4-2: Compare Hard Cutoff, Fuzzy, Parceling
    ├── ks_analysis.py              # Step 5: KS Analysis class and plotting
    ├── final_comparison.py         # Step 6: Compare all models
    ├── create_scorecard_policy.py  # Step 7: Create Scorecard and Policy Rules
    ├── psi_analysis.py             # Step 7-2: PSI diagnostics (selection bias + score shift)
    ├── app.py                      # Step 8: FastAPI Server
    └── load_test.py                # Step 9: Load testing script
```

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib plotly joblib fastapi uvicorn requests pydantic
    ```

2.  **Run Pipeline Steps:**
    ```bash
    # 1. Generate Data
    python src/generate_data.py
    
    # 2. Generate EDA Report
    python src/generate_report.py
    
    # 3. Train Baseline Model
    python src/train_scoring_model.py
    
    # 4. Perform Reject Inference & Compare Methods
    python src/reject_inference_methods.py
    
    # 5. Create Final Scorecard & Policy
    python src/create_scorecard_policy.py

    # 6. PSI Diagnostics (selection bias + reject-inference method comparison)
    python src/psi_analysis.py
    ```

3.  **Run API Server:**
    ```bash
    python src/app.py
    ```
  

4.  **Run Load Test (in another terminal):**
    ```bash
    python src/load_test.py
    ```

## Key Results
- **Selected Method:** Parceling (or Hard Cutoff depending on run)
- **Max KS:** ~40-50%
- **Policy:** Auto-Approve (>= 600), Review (550-600), Reject (< 550)

### 🔍 Model Explainability (XAI)
To ensure transparency in our credit decisions, I implemented **SHAP** analysis:
- **Global Interpretation:** Visualized the primary drivers of default risk across the entire population.
- **Local Interpretation:** Provided clear "Reason Codes" for individual loan denials using Waterfall plots.

![SHAP Summary](shap_summary.png)
*Figure 1: Global feature importance showing the impact of variables on the model output.*

![SHAP Waterfall](shap_waterfall.png)
*Figure 2: Waterfall plot showing feature contributions for a specific prediction.*

### 📐 PSI: what it can and can't tell you

`src/psi_analysis.py` adds Population Stability Index (PSI) diagnostics on top of the reject-inference comparison, and is deliberately scoped to the two places PSI is actually a valid tool here:

1. **Selection bias** (`selection_bias_psi`) — PSI between the *raw feature* distributions of the approved vs. rejected populations. This answers "how different are the applicants we reject from the applicants we approve?", which is exactly the population-shift question PSI was designed for.
2. **Score shift** (`score_shift_psi`) — PSI between the accepts-only baseline model's score distribution and the score distribution each reject-inference method produces after retraining on its augmented data.

**What it's *not* used for, and why:** it's tempting to also run PSI on the *raw feature vectors* of each method's augmented training sample against the population, to compare Hard Cutoff vs. Fuzzy Augmentation vs. Parceling. That comparison is a trap. Fuzzy Augmentation and Parceling both reuse every rejected applicant's real, unmodified feature vector — they only differ in what target label (Fuzzy: soft weight; Parceling: a per-applicant Bernoulli draw from the score-bin bad rate) gets attached to it. So a feature-level PSI on the augmented rows reads ~0 for both methods by construction, no matter how differently they actually behave — it's tautological, not diagnostic.

The score-shift PSI is the non-tautological alternative: since it measures the model's *output* after retraining on each method's labels, it actually discriminates between the methods. In a typical run: Fuzzy Augmentation stays close to the baseline (PSI < 0.10, "Stable") because its soft/continuous weighting doesn't inject hard label noise, Hard Cutoff shows a moderate shift, and Parceling shows a severe shift (PSI > 0.25) because its per-applicant random draw adds real label noise on top of the same feature vectors.

Running `python src/psi_analysis.py` writes both tables to `reports/psi_analysis_report.txt` and a two-panel chart to `reports/psi_chart.png`.

## A note on the data and the scale

The dataset is **synthetic**, generated by `src/generate_data.py`. Every figure in this
repository is therefore a property of a simulation and not a business outcome, and the
KS and PSI numbers should be read as evidence about the *methods* rather than about a
real book of loans.

The 300&ndash;850 score range is the familiar consumer-credit range, but the calibration
here is local to this model: **600 points is anchored at 5:1 odds of being good, and every
20 points doubles the odds.** Those three constants (`BASE_SCORE`, `PDO`, `BASE_ODDS` in
`index.html`, mirrored in `lambda_function.py`) define the whole scale. A score from this
model is not comparable to a bureau score that happens to use the same range.
