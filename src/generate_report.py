import pandas as pd
import os
import json

# Load data
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_path, 'data', 'raw', 'telecom_data.csv')
df = pd.read_csv(data_path)

# Prepare Data for Charts

# 1. Status Counts
status_counts = df['status'].value_counts()
status_data = {
    'labels': status_counts.index.tolist(),
    'data': status_counts.values.tolist()
}

# 2. Target Counts (Approved Only)
approved = df[df['status'] == 'approved']
target_counts = approved['target'].value_counts().sort_index()
target_data = {
    'labels': ['Bad (0)', 'Good (1)'], # assuming 0 and 1
    'data': [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))]
}

# 3. Histogram Data Helper
def get_hist_data(series, bins=20):
    counts, bin_edges = pd.cut(series, bins=bins, retbins=True).value_counts().sort_index(), pd.cut(series, bins=bins, retbins=True)[1]
    # Format bin labels
    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
    return labels, counts.values.tolist()

rejected = df[df['status'] == 'rejected']

# Collect histogram data for key features
features = ['income', 'credit_history_months', 'debt_ratio', 'num_late_payments']
feature_charts = {}

for feat in features:
    # Determine common bins range
    min_val = min(df[feat].min(), 0)
    max_val = df[feat].max()
    # Create approx 15 bins
    step = (max_val - min_val) / 15
    bins = [min_val + i*step for i in range(16)]
    
    # Approved hist
    a_counts = pd.cut(approved[feat], bins=bins).value_counts().sort_index()
    # Rejected hist
    r_counts = pd.cut(rejected[feat], bins=bins).value_counts().sort_index()
    
    labels = [f"{int(bins[i])}" for i in range(len(bins)-1)]
    
    feature_charts[feat] = {
        'labels': labels,
        'approved': a_counts.values.tolist(),
        'rejected': r_counts.values.tolist()
    }

# Convert to JSON for JS injection
charts_json = json.dumps({
    'status': status_data,
    'target': target_data,
    'features': feature_charts
})

# Generate HTML with Chart.js
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telco Credit Assessment Data Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f7f6; color: #333; }}
        h1 {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
        .dashboard-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .full-width {{ grid-column: 1 / -1; }}
        .stats-container {{ display: flex; justify-content: space-around; margin-bottom: 20px; }}
        .stat-item {{ text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #3498db; }}
        .stat-label {{ color: #7f8c8d; font-size: 0.9rem; }}
        canvas {{ max-width: 100%; height: 300px; }}
    </style>
</head>
<body>
    <h1>📱 Telco Credit Data Analysis Report</h1>
    
    <div class="dashboard-grid">
        <div class="card full-width">
            <div class="stats-container">
                <div class="stat-item">
                    <div class="stat-value">{len(df):,}</div>
                    <div class="stat-label">Total Applicants</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(approved):,}</div>
                    <div class="stat-label">Approved</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(rejected):,}</div>
                    <div class="stat-label">Rejected</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{approved['target'].mean()*100:.1f}%</div>
                    <div class="stat-label">Good Rate (Approved)</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>Applicant Status</h3>
            <canvas id="statusChart"></canvas>
        </div>
        <div class="card">
            <h3>Credit Worthiness (Approved)</h3>
            <canvas id="targetChart"></canvas>
        </div>

        <div class="card">
            <h3>Income Distribution</h3>
            <canvas id="incomeChart"></canvas>
        </div>
        <div class="card">
            <h3>Credit History (Months)</h3>
            <canvas id="historyChart"></canvas>
        </div>
        <div class="card">
            <h3>Debt Ratio</h3>
            <canvas id="debtChart"></canvas>
        </div>
        <div class="card">
            <h3>Late Payments</h3>
            <canvas id="lateChart"></canvas>
        </div>
    </div>

    <script>
        const data = {charts_json};

        // Common Options
        const commonOptions = {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom' }} }}
        }};

        // Status Chart
        new Chart(document.getElementById('statusChart'), {{
            type: 'doughnut',
            data: {{
                labels: data.status.labels,
                datasets: [{{
                    data: data.status.data,
                    backgroundColor: ['#3498db', '#e74c3c']
                }}]
            }},
            options: commonOptions
        }});

        // Target Chart
        new Chart(document.getElementById('targetChart'), {{
            type: 'pie',
            data: {{
                labels: data.target.labels,
                datasets: [{{
                    data: data.target.data,
                    backgroundColor: ['#e74c3c', '#2ecc71']
                }}]
            }},
            options: commonOptions
        }});

        // Feature Histograms Helper
        function createComparisonChart(canvasId, featureKey, label) {{
            new Chart(document.getElementById(canvasId), {{
                type: 'bar',
                data: {{
                    labels: data.features[featureKey].labels,
                    datasets: [
                        {{
                            label: 'Approved',
                            data: data.features[featureKey].approved,
                            backgroundColor: 'rgba(46, 204, 113, 0.6)',
                            borderColor: 'rgba(46, 204, 113, 1)',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Rejected',
                            data: data.features[featureKey].rejected,
                            backgroundColor: 'rgba(231, 76, 60, 0.6)',
                            borderColor: 'rgba(231, 76, 60, 1)',
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    ...commonOptions,
                    scales: {{
                        x: {{ stacked: false }},
                        y: {{ beginAtZero: true }}
                    }}
                }}
            }});
        }}

        createComparisonChart('incomeChart', 'income', 'Income');
        createComparisonChart('historyChart', 'credit_history_months', 'History');
        createComparisonChart('debtChart', 'debt_ratio', 'Debt Ratio');
        createComparisonChart('lateChart', 'num_late_payments', 'Late Payments');

    </script>
</body>
</html>
"""

report_path = os.path.join(base_path, 'reports', 'data_analysis.html')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Report generated at: {report_path}")
