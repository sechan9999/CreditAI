import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from train_scoring_model import CreditScoringModel

class RejectInference:
    """
    Reject Inference 클래스
    
    Methods:
    1. Hard Cutoff: 확률 기준으로 Good/Bad 할당
    2. Fuzzy Augmentation: 확률을 가중치로 사용
    3. Parceling: 각 점수 구간별 Bad Rate 적용
    """
    
    def __init__(self, approved_df, rejected_df, feature_cols, target_col='target'):
        self.approved_df = approved_df.copy()
        self.rejected_df = rejected_df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # 승인 데이터로 기본 모델 학습 (Check train_scoring_model.py for class definition)
        self.base_model = CreditScoringModel(feature_cols)
        self.base_model.fit(approved_df, approved_df[target_col])
        
        # 거절 고객 확률 예측
        self.rejected_df['pred_prob'] = self.base_model.predict_proba(rejected_df)
        self.rejected_df['pred_score'] = self.base_model.predict_score(rejected_df)
        
    def hard_cutoff(self, cutoff=0.5):
        """
        Method 1: Hard Cutoff
        - 확률이 cutoff 이상이면 Good, 미만이면 Bad
        - 가장 간단하지만 정보 손실 있음
        """
        output = f"\n{'='*60}\n"
        output += f"🔹 Hard Cutoff Method (cutoff={cutoff})\n"
        output += f"{'='*60}\n"
        
        rejected_copy = self.rejected_df.copy()
        rejected_copy[self.target_col] = (rejected_copy['pred_prob'] >= cutoff).astype(int)
        
        # 결과 통계
        n_good = (rejected_copy[self.target_col] == 1).sum()
        n_bad = (rejected_copy[self.target_col] == 0).sum()
        
        output += f"거절 고객 중 추정 Good: {n_good:,} ({n_good/len(rejected_copy)*100:.1f}%)\n"
        output += f"거절 고객 중 추정 Bad:  {n_bad:,} ({n_bad/len(rejected_copy)*100:.1f}%)\n"
        
        # 통합 데이터
        combined = pd.concat([self.approved_df, rejected_copy], ignore_index=True)
        
        return combined, rejected_copy, output
    
    def fuzzy_augmentation(self, scaling_factor=0.8):
        """
        Method 2: Fuzzy Augmentation
        - 각 거절 고객을 Good과 Bad로 복제
        - 예측 확률을 가중치로 사용
        - scaling_factor: 거절 고객의 Bad Rate 조정 (< 1이면 더 나쁜 것으로 간주)
        """
        output = f"\n{'='*60}\n"
        output += f"🔹 Fuzzy Augmentation Method (scaling={scaling_factor})\n"
        output += f"{'='*60}\n"
        
        rejected_copy = self.rejected_df.copy()
        
        # 확률 조정 (거절 고객은 일반적으로 더 나쁨)
        adjusted_prob = rejected_copy['pred_prob'] * scaling_factor
        
        # Good 레코드 생성 (가중치 = 조정된 확률)
        good_records = rejected_copy.copy()
        good_records[self.target_col] = 1
        good_records['weight'] = adjusted_prob
        
        # Bad 레코드 생성 (가중치 = 1 - 조정된 확률)
        bad_records = rejected_copy.copy()
        bad_records[self.target_col] = 0
        bad_records['weight'] = 1 - adjusted_prob
        
        # 승인 데이터에 가중치 추가
        approved_weighted = self.approved_df.copy()
        approved_weighted['weight'] = 1.0
        
        # 통합
        combined = pd.concat([approved_weighted, good_records, bad_records], ignore_index=True)
        
        # 통계
        effective_good = good_records['weight'].sum()
        effective_bad = bad_records['weight'].sum()
        
        output += f"거절 고객 Effective Good: {effective_good:,.1f}\n"
        output += f"거절 고객 Effective Bad:  {effective_bad:,.1f}\n"
        output += f"Effective Bad Rate: {effective_bad/(effective_good+effective_bad)*100:.1f}%\n"
        
        return combined, output
    
    def parceling(self, n_bins=10):
        """
        Method 3: Parceling
        - 점수 구간별로 승인 고객의 Bad Rate 계산
        - 거절 고객에게 해당 구간의 Bad Rate 확률로 Good/Bad 할당
        """
        np.random.seed(42)  # For reproducibility
        
        output = f"\n{'='*60}\n"
        output += f"🔹 Parceling Method ({n_bins} quantile bins on predicted probability)\n"
        output += f"{'='*60}\n"
        
        # Quantile bins on predicted probability -- not equal-width bins on the
        # displayed 300-850 score, which is what this used to do.
        #
        # Two things were wrong with binning on the score. The score is an affine
        # transform of log-odds with a chosen PDO and anchor, and it is clipped at
        # 300 and 850, so equal-width bins on it made this method depend on a
        # cosmetic display choice. Measured directly: the same data and the same
        # seed assign 261, 281 and 278 rejected applicants to Good under PDO 20/5:1,
        # PDO 40/1:1 and PDO 75/3:1. Relabelling a score changes no decision, so it
        # must not change which applicants a reject-inference method calls good.
        #
        # And equal-width bins left some bins nearly empty -- 3 approved applicants
        # in one and 11 in another -- so rejects landing there were assigned from a
        # bad rate estimated on 3 observations. Quantile bins put roughly equal
        # counts in each, which is the point of stratifying at all.
        approved_copy = self.approved_df.copy()
        approved_copy['pred_prob'] = self.base_model.predict_proba(approved_copy)

        edges = np.unique(np.quantile(approved_copy['pred_prob'], np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf

        approved_copy['score_bin'] = pd.cut(approved_copy['pred_prob'], bins=edges, labels=False)
        
        # 구간별 Bad Rate 계산
        bin_stats = approved_copy.groupby('score_bin').agg({
            self.target_col: ['count', 'mean']
        }).reset_index()
        bin_stats.columns = ['score_bin', 'count', 'good_rate']
        bin_stats['bad_rate'] = 1 - bin_stats['good_rate']
        
        output += "\n📊 Probability Bin Statistics (quantile bins):\n"
        output += bin_stats.to_string(index=False) + "\n"
        
        # 거절 고객에 적용
        rejected_copy = self.rejected_df.copy()
        rejected_copy['score_bin'] = pd.cut(rejected_copy['pred_prob'], bins=edges, labels=False)
        
        # 각 구간의 Bad Rate를 확률로 사용하여 Good/Bad 할당
        def assign_target(row):
            bin_idx = row['score_bin']
            if pd.isna(bin_idx) or bin_idx not in bin_stats['score_bin'].values:
                # 구간 밖이면 전체 Bad Rate 사용
                bad_rate = approved_copy[self.target_col].mean()
            else:
                bad_rate = bin_stats.loc[bin_stats['score_bin'] == bin_idx, 'bad_rate'].values[0]
            
            # Bad Rate만큼의 확률로 Bad(0) 할당
            return 0 if np.random.random() < bad_rate else 1
        
        rejected_copy[self.target_col] = rejected_copy.apply(assign_target, axis=1)
        
        # 결과 통계
        n_good = (rejected_copy[self.target_col] == 1).sum()
        n_bad = (rejected_copy[self.target_col] == 0).sum()
        
        output += f"\n거절 고객 할당 결과:\n"
        output += f"  Good: {n_good:,} ({n_good/len(rejected_copy)*100:.1f}%)\n"
        output += f"  Bad:  {n_bad:,} ({n_bad/len(rejected_copy)*100:.1f}%)\n"
        
        # 통합
        combined = pd.concat([
            self.approved_df, 
            rejected_copy.drop(columns=['pred_prob', 'pred_score', 'score_bin'])
        ], ignore_index=True)
        
        return combined, rejected_copy, output

if __name__ == "__main__":
    # Load data
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'raw', 'telecom_data.csv')
    df = pd.read_csv(data_path)
    
    approved_df = df[df['status'] == 'approved'].copy()
    rejected_df = df[df['status'] == 'rejected'].copy()
    feature_cols = ['age', 'income', 'credit_history_months', 
                    'num_credit_accounts', 'debt_ratio', 'num_late_payments']
    
    # Run Inference
    ri = RejectInference(approved_df, rejected_df, feature_cols)

    # 1. Hard Cutoff
    combined_hard, rejected_hard, out_hard = ri.hard_cutoff(cutoff=0.5)
    
    # 2. Fuzzy Augmentation
    combined_fuzzy, out_fuzzy = ri.fuzzy_augmentation(scaling_factor=0.7)
    
    # 3. Parceling
    combined_parcel, rejected_parcel, out_parcel = ri.parceling(n_bins=10)
    
    # Print and Save
    full_report = "\n" + "=" * 80 + "\n"
    full_report += "🔄 REJECT INFERENCE COMPARISON\n"
    full_report += "=" * 80 + "\n"
    full_report += out_hard
    full_report += out_fuzzy
    full_report += out_parcel
    
    print(full_report)
    
    with open(os.path.join(base_path, 'reports', 'reject_inference_methods.txt'), 'w', encoding='utf-8') as f:
        f.write(full_report)
