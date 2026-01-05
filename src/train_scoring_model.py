import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CreditScoringModel:
    """
    신용평가 모델 클래스
    - Logistic Regression 기반
    - 확률을 점수로 변환
    """
    
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            penalty='l2',           # L2 정규화
            C=1.0,                  # 정규화 강도
            class_weight='balanced', # 클래스 불균형 처리
            max_iter=1000,
            random_state=42
        )
        self.is_fitted = False
        
    def fit(self, X, y):
        """모델 학습"""
        X_scaled = self.scaler.fit_transform(X[self.feature_cols])
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # 계수 저장
        self.coefficients_ = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': self.model.coef_[0],
            'odds_ratio': np.exp(self.model.coef_[0])
        }).sort_values('coefficient', ascending=False)
        
        return self
    
    def predict_proba(self, X):
        """Good 확률 예측"""
        X_scaled = self.scaler.transform(X[self.feature_cols])
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_score(self, X, base_score=600, pdo=20, base_odds=5):
        """
        확률을 신용 점수로 변환
        
        Score = base_score + pdo * log(odds) / log(2)
        
        Parameters:
        - base_score: 기준 점수 (일반적으로 600)
        - pdo: Points to Double Odds (20점 증가 시 odds 2배)
        - base_odds: 기준 odds (Good:Bad = 50:1)
        """
        prob = self.predict_proba(X)
        odds = prob / (1 - prob + 1e-10)
        
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        
        score = offset + factor * np.log(odds + 1e-10)
        return np.clip(score, 300, 850)  # 점수 범위 제한
    
    def summary(self):
        """모델 요약"""
        summary_str = ""
        summary_str += "\n" + "=" * 60 + "\n"
        summary_str += "📈 Logistic Regression 모델 계수\n"
        summary_str += "=" * 60 + "\n"
        summary_str += self.coefficients_.to_string(index=False) + "\n"
        summary_str += f"\nIntercept: {self.model.intercept_[0]:.4f}\n"
        summary_str += "=" * 60 + "\n"
        return summary_str

if __name__ == "__main__":
    # 데이터 로드
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'raw', 'telecom_data.csv')
    df = pd.read_csv(data_path)
    
    # 승인된 고객만 사용
    approved_df = df[df['status'] == 'approved'].copy()
    
    # 피처 정의
    feature_cols = ['age', 'income', 'credit_history_months', 
                    'num_credit_accounts', 'debt_ratio', 'num_late_payments']

    # 승인 고객 데이터로 Train/Test 분리
    X_approved = approved_df[feature_cols]
    y_approved = approved_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X_approved, y_approved, test_size=0.3, random_state=42, stratify=y_approved
    )

    # 모델 학습
    credit_model = CreditScoringModel(feature_cols)
    credit_model.fit(pd.DataFrame(X_train, columns=feature_cols), y_train)
    
    report = credit_model.summary()
    print(report)

    # 예측
    train_proba = credit_model.predict_proba(pd.DataFrame(X_train, columns=feature_cols))
    test_proba = credit_model.predict_proba(pd.DataFrame(X_test, columns=feature_cols))
    train_scores = credit_model.predict_score(pd.DataFrame(X_train, columns=feature_cols))
    test_scores = credit_model.predict_score(pd.DataFrame(X_test, columns=feature_cols))

    stats = f"\n📊 점수 분포:\n"
    stats += f"Train - Mean: {train_scores.mean():.1f}, Std: {train_scores.std():.1f}\n"
    stats += f"Test  - Mean: {test_scores.mean():.1f}, Std: {test_scores.std():.1f}\n"
    print(stats)
    
    # Save Report
    with open(os.path.join(base_path, 'reports', 'scoring_model_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(report + stats)
    
    # 모델 저장
    processed_dir = os.path.join(base_path, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    model_path = os.path.join(processed_dir, 'scoring_model.pkl')
    joblib.dump(credit_model, model_path)
    print(f"\n[Info] Scoring Model saved to: {model_path}")
