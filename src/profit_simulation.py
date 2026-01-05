import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 1. 가정 설정 (비즈니스 시나리오)
    profit_per_tp = 1000  # 대출 상환 시 이익 ($)
    loss_per_fp = 5000    # 대출 부도 시 손실 ($)

    # 2. 임계값 후보군 (0.1부터 0.9까지)
    thresholds = np.linspace(0.1, 0.9, 9)
    profits = []

    # 3. 각 임계값별 기대 이익 계산 시뮬레이션
    for t in thresholds:
        # 예시 데이터 (TP와 FP는 모델 성능에 따라 변함)
        # 실제로는 model.predict_proba() 결과로 계산합니다.
        tp = 900 * (1 - t**2) # 임계값이 높을수록 TP는 줄어듬
        fp = 100 * (1 - t)    # 임계값이 높을수록 FP도 줄어듬
        
        current_profit = (tp * profit_per_tp) - (fp * loss_per_fp)
        profits.append(current_profit)

    # 4. 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, profits, marker='o', linestyle='-', color='b')
    
    # 최적 임계값 찾기
    optimal_idx = np.argmax(profits)
    optimal_threshold = thresholds[optimal_idx]
    max_profit = profits[optimal_idx]
    
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.1f}')
    plt.scatter([optimal_threshold], [max_profit], color='r', zorder=5)
    plt.text(optimal_threshold, max_profit, f" ${max_profit:,.0f}", verticalalignment='bottom')

    plt.title("Expected Profit by Approval Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Expected Profit ($)")
    plt.legend()
    plt.grid(True)
    
    # Save to current directory
    output_path = 'profit_optimization.png'
    plt.savefig(output_path)
    print(f"Graph saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
