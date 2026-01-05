import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from train_scoring_model import CreditScoringModel
from ks_analysis import KSCalculator
from reject_inference_methods import RejectInference
from final_comparison import train_and_evaluate

def create_scorecard(model, feature_cols, base_score=600, pdo=20):
    """
    Logistic Regression 계수를 스코어카드로 변환
    
    Score = Σ (WoE_i × β_i × Factor) + Offset
    
    여기서는 간소화된 버전 사용
    """
    
    coefficients = model.model.coef_[0]
    intercept = model.model.intercept_[0]
    
    factor = pdo / np.log(2)
    
    scorecard = []
    
    for i, feat in enumerate(feature_cols):
        coef = coefficients[i]
        # 스케일된 계수를 점수 기여도로 변환
        points = coef * factor
        scorecard.append({
            'Feature': feat,
            'Coefficient': round(coef, 4),
            'Points per Std': round(points, 2)
        })
    
    scorecard_df = pd.DataFrame(scorecard)
    
    output = "\n" + "=" * 60 + "\n"
    output += "📋 CREDIT SCORECARD\n"
    output += "=" * 60 + "\n"
    output += f"Base Score: {base_score}\n"
    output += f"PDO (Points to Double Odds): {pdo}\n"
    output += f"Intercept: {intercept:.4f}\n\n"
    output += scorecard_df.to_string(index=False) + "\n"
    output += "=" * 60 + "\n"
    print(output)
    
    return scorecard_df, output

def create_decision_rules(model, df, feature_cols):
    """승인/거절 기준 수립"""
    
    scores = model.predict_score(df)
    targets = df['target'].values
    
    # 점수 구간별 분석
    score_ranges = [
        (300, 500, "Very High Risk"),
        (500, 550, "High Risk"),
        (550, 600, "Medium Risk"),
        (600, 650, "Low Risk"),
        (650, 700, "Very Low Risk"),
        (700, 850, "Minimal Risk")
    ]
    
    output = "\n" + "=" * 70 + "\n"
    output += "📱 전화 가입 승인 결정 규칙\n"
    output += "=" * 70 + "\n"
    
    results = []
    for low, high, risk_level in score_ranges:
        mask = (scores >= low) & (scores < high)
        if mask.sum() > 0:
            n = mask.sum()
            bad_rate = (targets[mask] == 0).mean() * 100
            results.append({
                'Score Range': f"{low}-{high}",
                'Risk Level': risk_level,
                'N Customers': n,
                'Bad Rate (%)': round(bad_rate, 1),
                'Decision': 'Reject' if bad_rate > 30 else ('Review' if bad_rate > 15 else 'Approve')
            })
    
    rules_df = pd.DataFrame(results)
    output += rules_df.to_string(index=False) + "\n"
    
    output += "\n" + "-" * 70 + "\n"
    output += "💡 권장 정책:\n"
    output += "  • Score >= 600: 자동 승인\n"
    output += "  • 550 <= Score < 600: 수동 심사\n"
    output += "  • Score < 550: 자동 거절 또는 보증금 요구\n"
    output += "=" * 70 + "\n"
    print(output)
    
    return rules_df, output

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
    
    print("\n" + "=" * 80)
    print("📊 Evaluating Models to Select Best One...")
    print("=" * 80)

    results = []
    results.append(train_and_evaluate(approved_df, "Approved Only (Baseline)", feature_cols))
    results.append(train_and_evaluate(combined_hard, "Hard Cutoff", feature_cols))
    results.append(train_and_evaluate(combined_fuzzy, "Fuzzy Augmentation", feature_cols))
    results.append(train_and_evaluate(combined_parcel, "Parceling", feature_cols))

    # 최종 모델 선택 (최대 KS)
    best_result = max(results[1:], key=lambda x: x['ks'])
    print(f"\n🏆 Best Method: {best_result['name']} (KS = {best_result['ks']:.2f}%)")
    
    # Save Best Model Logic
    processed_dir = os.path.join(base_path, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    best_model_path = os.path.join(processed_dir, 'best_scoring_model.pkl')
    joblib.dump(best_result['model'], best_model_path)
    print(f"[Info] Best model saved to: {best_model_path}")

    # 스코어카드 생성
    scorecard, sc_output = create_scorecard(best_result['model'], feature_cols)
    
    # 결정 규칙 생성
    decision_rules, rules_output = create_decision_rules(
        best_result['model'], 
        approved_df, 
        feature_cols
    )
    
    # Save Reports
    with open(os.path.join(base_path, 'reports', 'final_scorecard_policy.txt'), 'w', encoding='utf-8') as f:
        f.write(f"🏆 Best Method Selected: {best_result['name']} (KS = {best_result['ks']:.2f}%)\n")
        f.write(sc_output)
        f.write(rules_output)
