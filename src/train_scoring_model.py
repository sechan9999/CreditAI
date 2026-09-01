import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scale

class CreditScoringModel:
    """
    신용평가 모델 클래스
    - Logistic Regression 기반
    - 확률을 점수로 변환
    """
    
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        # class_weight='balanced' was removed deliberately.
        #
        # It buys nothing here and costs a lot, measured on this data with the
        # approved test half held out: AUC 0.6916 either way, a difference of
        # 0.00001, and the two score orderings correlate at 0.99987 by Spearman.
        # For logistic regression the balancing is almost purely an intercept
        # shift -- the slopes correlate at 0.99996 -- and a constant shift in
        # log-odds cannot change a ranking. Brier, which does see calibration,
        # goes the other way: 0.2249 balanced against 0.1947 unweighted.
        #
        # The class split here is 69/31. "balanced" is a tool for severe
        # imbalance; at this ratio it solves a problem the data does not have.
        #
        # What it does change is the meaning of the probability, and this model's
        # probability is not a diagnostic. It is fed straight into a scorecard
        # anchored on odds: score = offset + factor * ln(odds). On this data the
        # balancing moved the intercept by -0.7894, which is -45.6 points on every
        # applicant, and dropped approvals at the 693-point bar from 257 to 41 out
        # of 1,500. The policy says approve at p(good) >= 83.3%; with probabilities
        # 25 points low the system was declining applicants who met its own bar.
        #
        # If a future dataset is imbalanced enough that the fit needs weighting,
        # weight it and then calibrate (Platt or isotonic on a holdout) before the
        # points mapping -- do not feed weighted probabilities into an odds-anchored
        # scale.
        # penalty='l2' was passed explicitly and has been dropped. It was the
        # default in every scikit-learn this project supports, so passing it
        # bought nothing -- and from 1.8 it emits a FutureWarning, removed in
        # 1.10, which would have turned into a hard failure on the next upgrade.
        #
        # The documented migration is l1_ratio=0, but that would raise on the
        # older scikit-learn that requirements.txt still allows (>=1.0), where
        # l1_ratio was only valid with solver='saga' and penalty='elasticnet'.
        # Relying on the default is correct on every version.
        #
        # Verified behaviour-preserving: fitting with and without the argument
        # gives bit-identical coefficients and intercept on this data, so no
        # model artifact or report needed regenerating.
        self.model = LogisticRegression(
            C=1.0,                  # 정규화 강도 (L2, the default penalty)
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
    
    def predict_score(self, X, base_score=None, pdo=None, base_odds=None):
        """Convert probability to a credit score.

            score = offset + factor * ln(odds),   factor = pdo / ln2

        Defaults come from src/scale.py and are defined nowhere else. While this
        function carried its own defaults (pdo=20, base_odds=5) the web page, the
        Lambda handler and this module could each use a different scale, and they
        did: the same applicant scored 122 points lower on the page than the
        trained model gave. Passing the arguments explicitly still works, which is
        what you want when comparing two scales deliberately.
        """
        base_score = scale.BASE_SCORE if base_score is None else base_score
        pdo = scale.PDO if pdo is None else pdo
        base_odds = scale.BASE_ODDS if base_odds is None else base_odds

        prob = self.predict_proba(X)
        odds = prob / (1 - prob + 1e-10)

        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)

        score = offset + factor * np.log(odds + 1e-10)
        return np.clip(score, scale.SCORE_MIN, scale.SCORE_MAX)
    
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
