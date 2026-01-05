import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class KSCalculator:
    """
    KS (Kolmogorov-Smirnov) 통계량 계산기
    
    KS는 Good과 Bad의 누적 분포 차이의 최대값
    - KS > 0.4: Excellent
    - 0.35 < KS <= 0.4: Very Good
    - 0.30 < KS <= 0.35: Good
    - 0.20 < KS <= 0.30: Moderate
    - KS <= 0.20: Poor
    """
    
    def __init__(self, y_true, y_prob, n_bins=10):
        self.y_true = np.array(y_true)
        self.y_prob = np.array(y_prob)
        self.n_bins = n_bins
        self._calculate()
        
    def _calculate(self):
        """KS 테이블 계산"""
        # 확률로 정렬 (내림차순 - 높은 확률 = Good)
        sorted_idx = np.argsort(-self.y_prob)
        sorted_prob = self.y_prob[sorted_idx]
        sorted_target = self.y_true[sorted_idx]
        
        # 구간별 분할
        n = len(sorted_target)
        bin_size = n // self.n_bins
        
        results = []
        cum_good = 0
        cum_bad = 0
        total_good = (self.y_true == 1).sum()
        total_bad = (self.y_true == 0).sum()
        
        for i in range(self.n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < self.n_bins - 1 else n
            
            bin_target = sorted_target[start:end]
            bin_prob = sorted_prob[start:end]
            
            bin_good = (bin_target == 1).sum()
            bin_bad = (bin_target == 0).sum()
            bin_total = len(bin_target)
            
            cum_good += bin_good
            cum_bad += bin_bad
            
            cum_good_pct = cum_good / total_good * 100
            cum_bad_pct = cum_bad / total_bad * 100
            ks = cum_good_pct - cum_bad_pct
            
            results.append({
                'decile': i + 1,
                'min_prob': bin_prob.min(),
                'max_prob': bin_prob.max(),
                'total': bin_total,
                'good': bin_good,
                'bad': bin_bad,
                'bad_rate': bin_bad / bin_total * 100,
                'cum_good': cum_good,
                'cum_bad': cum_bad,
                'cum_good_pct': cum_good_pct,
                'cum_bad_pct': cum_bad_pct,
                'ks': ks
            })
        
        self.ks_table = pd.DataFrame(results)
        self.ks_statistic = self.ks_table['ks'].max()
        self.ks_decile = self.ks_table.loc[self.ks_table['ks'].idxmax(), 'decile']
        
    def summary(self):
        """KS 요약 출력"""
        output = ""
        output += "\n" + "=" * 80 + "\n"
        output += "📊 KS (Kolmogorov-Smirnov) Analysis\n"
        output += "=" * 80 + "\n"
        
        display_cols = ['decile', 'total', 'good', 'bad', 'bad_rate', 
                       'cum_good_pct', 'cum_bad_pct', 'ks']
        output += self.ks_table[display_cols].round(2).to_string(index=False) + "\n"
        
        output += "\n" + "-" * 80 + "\n"
        output += f"🎯 Maximum KS: {self.ks_statistic:.2f}% at Decile {self.ks_decile}\n"
        
        # 판별력 평가
        if self.ks_statistic > 40:
            rating = "Excellent ⭐⭐⭐⭐⭐"
        elif self.ks_statistic > 35:
            rating = "Very Good ⭐⭐⭐⭐"
        elif self.ks_statistic > 30:
            rating = "Good ⭐⭐⭐"
        elif self.ks_statistic > 20:
            rating = "Moderate ⭐⭐"
        else:
            rating = "Poor ⭐"
            
        output += f"📈 Model Discrimination: {rating}\n"
        output += "=" * 80 + "\n"
        print(output)
        return output
        
    def plot(self, save_path, figsize=(12, 5)):
        """KS 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: 누적 분포
        ax1 = axes[0]
        deciles = self.ks_table['decile']
        ax1.plot(deciles, self.ks_table['cum_good_pct'], 'g-o', label='Cumulative Good %', linewidth=2)
        ax1.plot(deciles, self.ks_table['cum_bad_pct'], 'r-s', label='Cumulative Bad %', linewidth=2)
        
        # KS 최대 지점 표시
        ks_idx = self.ks_table['ks'].idxmax()
        ax1.vlines(self.ks_decile, 
                  self.ks_table.loc[ks_idx, 'cum_bad_pct'],
                  self.ks_table.loc[ks_idx, 'cum_good_pct'],
                  colors='blue', linestyles='--', linewidth=2, label=f'KS = {self.ks_statistic:.1f}%')
        
        ax1.set_xlabel('Decile')
        ax1.set_ylabel('Cumulative %')
        ax1.set_title('KS Chart - Cumulative Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bad Rate by Decile
        ax2 = axes[1]
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, self.n_bins))
        ax2.bar(deciles, self.ks_table['bad_rate'], color=colors)
        ax2.axhline(y=(self.y_true == 0).mean() * 100, color='red', linestyle='--', label='Overall Bad Rate')
        ax2.set_xlabel('Decile (1=Best, 10=Worst)')
        ax2.set_ylabel('Bad Rate (%)')
        ax2.set_title('Bad Rate by Decile')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n📈 KS Chart saved to: {save_path}")

if __name__ == "__main__":
    # Load Environment
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'raw', 'telecom_data.csv')
    model_path = os.path.join(base_path, 'data', 'processed', 'scoring_model.pkl')
    
    # Load Data & Model
    df = pd.read_csv(data_path)
    
    # Import class
    import sys
    sys.path.append(os.path.join(base_path, 'src'))
    from train_scoring_model import CreditScoringModel 
    
    model = joblib.load(model_path)
    
    # Re-create Test Set (must match seed)
    approved_df = df[df['status'] == 'approved'].copy()
    feature_cols = ['age', 'income', 'credit_history_months', 
                    'num_credit_accounts', 'debt_ratio', 'num_late_payments']
    
    X = approved_df[feature_cols]
    y = approved_df['target']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Predict Probabilities
    test_proba = model.predict_proba(pd.DataFrame(X_test, columns=feature_cols))
    
    # Calculate KS
    ks_calc = KSCalculator(y_test.values, test_proba, n_bins=10)
    
    # Output
    report = ks_calc.summary()
    
    # Save Report
    with open(os.path.join(base_path, 'reports', 'ks_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
        
    # Save Plot
    plot_path = os.path.join(base_path, 'reports', 'ks_chart.png')
    ks_calc.plot(plot_path)
