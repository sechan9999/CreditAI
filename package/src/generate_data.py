import numpy as np
import pandas as pd
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# 시나리오: 전화회사 가입 신청자 데이터
# - approved: 가입 승인된 고객 (성과 관찰 가능)
# - rejected: 가입 거절된 고객 (성과 관찰 불가 → Reject Inference 필요)
# =============================================================================

def generate_telecom_data(n_approved=5000, n_rejected=2000):
    """
    전화회사 가입 신청 데이터 생성
    
    Features:
    - age: 나이
    - income: 연소득 (만원)
    - credit_history_months: 신용거래 기간 (개월)
    - num_credit_accounts: 신용 계좌 수
    - debt_ratio: 부채 비율
    - num_late_payments: 연체 횟수
    """
    
    # === 승인된 고객 (Known Good/Bad) ===
    approved_data = pd.DataFrame({
        'age': np.random.normal(35, 10, n_approved).clip(20, 70).astype(int),
        'income': np.random.lognormal(8, 0.5, n_approved).clip(2000, 15000),
        'credit_history_months': np.random.exponential(36, n_approved).clip(1, 240).astype(int),
        'num_credit_accounts': np.random.poisson(3, n_approved).clip(0, 15),
        'debt_ratio': np.random.beta(2, 5, n_approved),
        'num_late_payments': np.random.poisson(1, n_approved).clip(0, 10)
    })
    
    # 타겟 변수: 1 = Good (정상 납부), 0 = Bad (연체/해지)
    # 로지스틱 함수로 확률 계산
    log_odds = (
        -2.0 
        + 0.03 * approved_data['age']
        + 0.0003 * approved_data['income']
        + 0.01 * approved_data['credit_history_months']
        + 0.1 * approved_data['num_credit_accounts']
        - 3.0 * approved_data['debt_ratio']
        - 0.5 * approved_data['num_late_payments']
    )
    prob_good = 1 / (1 + np.exp(-log_odds))
    approved_data['target'] = (np.random.random(n_approved) < prob_good).astype(int)
    approved_data['status'] = 'approved'
    
    # === 거절된 고객 (성과 미관측) ===
    # 거절 고객은 일반적으로 더 높은 위험 프로파일
    rejected_data = pd.DataFrame({
        'age': np.random.normal(30, 12, n_rejected).clip(18, 70).astype(int),
        'income': np.random.lognormal(7.5, 0.6, n_rejected).clip(1000, 10000),
        'credit_history_months': np.random.exponential(24, n_rejected).clip(0, 120).astype(int),
        'num_credit_accounts': np.random.poisson(2, n_rejected).clip(0, 10),
        'debt_ratio': np.random.beta(3, 3, n_rejected),
        'num_late_payments': np.random.poisson(3, n_rejected).clip(0, 15)
    })
    rejected_data['target'] = np.nan  # 성과 미관측!
    rejected_data['status'] = 'rejected'
    
    return approved_data, rejected_data

if __name__ == "__main__":
    # 데이터 생성
    approved_df, rejected_df = generate_telecom_data()
    full_df = pd.concat([approved_df, rejected_df], ignore_index=True)

    print("=" * 60)
    print("📱 전화회사 가입 신청 데이터 요약")
    print("=" * 60)
    print(f"승인된 고객: {len(approved_df):,}명")
    print(f"  - Good (정상): {approved_df['target'].sum():,}명 ({approved_df['target'].mean()*100:.1f}%)")
    print(f"  - Bad (연체): {(approved_df['target']==0).sum():,}명 ({(1-approved_df['target'].mean())*100:.1f}%)")
    print(f"거절된 고객: {len(rejected_df):,}명 (성과 미관측)")
    print("=" * 60)

    # 데이터 저장
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_path, 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'telecom_data.csv')
    full_df.to_csv(output_path, index=False)
    print(f"\n[Info] Data saved to: {output_path}")
