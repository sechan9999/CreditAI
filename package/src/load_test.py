import requests
import pandas as pd
import numpy as np
import time
import concurrent.futures
import random

API_URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 500
CONCURRENT_USERS = 20

def generate_random_applicant():
    return {
        "age": random.randint(20, 70),
        "income": round(random.uniform(2000, 15000), 1),
        "credit_history_months": random.randint(1, 240),
        "num_credit_accounts": random.randint(0, 15),
        "debt_ratio": round(random.random(), 2),
        "num_late_payments": random.choice([0, 0, 0, 1, 2, 5])
    }

def send_request(request_id):
    data = generate_random_applicant()
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=data)
        latency = (time.time() - start_time) * 1000 # ms
        return {
            "id": request_id,
            "status": response.status_code,
            "latency": latency,
            "decision": response.json().get('decision') if response.status_code == 200 else "Error"
        }
    except Exception as e:
        return {
            "id": request_id, 
            "status": "Fail", 
            "latency": 0, 
            "decision": str(e)
        }

print(f"🚀 Starting Load Test: {NUM_REQUESTS} requests with {CONCURRENT_USERS} concurrent users...")
start_test = time.time()

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
    futures = [executor.submit(send_request, i) for i in range(NUM_REQUESTS)]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

total_time = time.time() - start_test
df_res = pd.DataFrame(results)

print("\n" + "=" * 60)
print("📊 LOAD TEST RESULTS")
print("=" * 60)
print(f"Total Requests: {NUM_REQUESTS}")
print(f"Total Time: {total_time:.2f} sec")
print(f"Throughput: {NUM_REQUESTS / total_time:.1f} req/sec")
print(f"Success Rate: {(df_res['status'] == 200).mean() * 100:.1f}%")
print("-" * 60)
print(f"Average Latency: {df_res['latency'].mean():.2f} ms")
print(f"P95 Latency: {df_res['latency'].quantile(0.95):.2f} ms")
print(f"P99 Latency: {df_res['latency'].quantile(0.99):.2f} ms")
print("-" * 60)
print("Decisions Distribution:")
print(df_res['decision'].value_counts().to_string())
print("=" * 60)
