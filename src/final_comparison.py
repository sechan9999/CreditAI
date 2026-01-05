import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from train_scoring_model import CreditScoringModel
from ks_analysis import KSCalculator
from reject_inference_methods import RejectInference

def train_and_evaluate(df, name, feature_cols, sample_weight=None):
    """모델 학습 및 KS 계산"""
    X = df[feature_cols]
    y = df['target']
    
    # Train/Test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 가중치 처리
    if 'weight' in df.columns:
        train_weights = df.loc[X_train.index, 'weight'].values
    else:
        train_weights = None
    
    # 모델 학습
    model = CreditScoringModel(feature_cols)
    if train_weights is not None:
        # 가중치가 있으면 직접 fit
        X_scaled = model.scaler.fit_transform(X_train)
        model.model.fit(X_scaled, y_train, sample_weight=train_weights)
        
        # Manually set coefs for consistency if needed, or rely on internal object
        model.is_fitted = True
    else:
        model.fit(pd.DataFrame(X_train.values, columns=feature_cols), y_train)
    
    # 예측
    test_proba = model.predict_proba(pd.DataFrame(X_test.values, columns=feature_cols))
    
    # KS 계산
    ks = KSCalculator(y_test.values, test_proba, n_bins=10)
    
    return {
        'name': name,
        'n_samples': len(df),
        'bad_rate': (y == 0).mean() * 100,
        'bad_rate_weighted': ((y==0) * df.get('weight', 1)).sum() / df.get('weight', 1).sum() * 100 if 'weight' in df else (y==0).mean() * 100,
        'ks': ks.ks_statistic,
        'model': model,
        'ks_calc': ks
    }

if __name__ == "__main__":
    # 데이터 로드
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'raw', 'telecom_data.csv')
    df = pd.read_csv(data_path)
    
    approved_df = df[df['status'] == 'approved'].copy()
    rejected_df = df[df['status'] == 'rejected'].copy()
    feature_cols = ['age', 'income', 'credit_history_months', 
                    'num_credit_accounts', 'debt_ratio', 'num_late_payments']

    # Reject Inference 실행 to get datasets
    print("Preparing Datasets...")
    ri = RejectInference(approved_df, rejected_df, feature_cols)
    combined_hard, _, _ = ri.hard_cutoff(cutoff=0.5)
    combined_fuzzy, _ = ri.fuzzy_augmentation(scaling_factor=0.7)
    combined_parcel, _, _ = ri.parceling(n_bins=10)
    
    # 각 방법별 모델 학습 및 평가
    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON AFTER REJECT INFERENCE")
    print("=" * 80)

    results = []

    # 1. Approved Only (baseline)
    results.append(train_and_evaluate(approved_df, "Approved Only (Baseline)", feature_cols))

    # 2. Hard Cutoff
    results.append(train_and_evaluate(combined_hard, "Hard Cutoff", feature_cols))

    # 3. Fuzzy Augmentation
    results.append(train_and_evaluate(combined_fuzzy, "Fuzzy Augmentation", feature_cols))

    # 4. Parceling
    results.append(train_and_evaluate(combined_parcel, "Parceling", feature_cols))

    # 결과 출력
    comparison_df = pd.DataFrame([{
        'Method': r['name'],
        'N Samples': r['n_samples'],
        'Bad Rate (%)': round(r['bad_rate_weighted'], 2), 
        'KS (%)': round(r['ks'], 2)
    } for r in results])

    print("\n" + comparison_df.to_string(index=False))
    print("\n" + "=" * 80)
    
    # Save comparison to file
    with open(os.path.join(base_path, 'reports', 'final_model_comparison.txt'), 'w', encoding='utf-8') as f:
        f.write("📊 MODEL COMPARISON AFTER REJECT INFERENCE\n")
        f.write("=" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n" + "=" * 80 + "\n")
