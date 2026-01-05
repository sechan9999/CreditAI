# Deploying Telco Credit Assessment API to AWS

This guide explains how to deploy your FastAPI application to AWS Cloud.

## Prerequisite
- You have an **AWS Account**.
- You have the code pushed to GitHub (which we have done).

## Method 1: AWS App Runner (Recommended - Easiest)
AWS App Runner accesses your GitHub repo and automatically builds and deploys your code.

1.  **Log in to AWS Console** and search for **"App Runner"**.
2.  Click **"Create an App Runner service"**.
3.  **Source:** Select **"Source code repository"**.
4.  **Connect to GitHub:**
    *   Add your GitHub account if not connected.
    *   Select Repository: `sechan9999/credit_asessment`
    *   Branch: `main`
5.  **Deployment Settings:** Select **"Automatic"** (redeploys on every push) or "Manual".
6.  **Build Settings:** Select **"Configure all settings here"**.
    *   **Runtime:** Python 3
    *   **Build command:** `pip install -r requirements.txt`
    *   **Start command:** `python src/app.py` 
        *   *Note:* You might need to edit `src/app.py` to listen on port 8080 (App Runner default) or configure the port in App Runner settings. 
        *   *Better Option:* Use **configuration file (apprunner.yaml)** if we created one, or simply use the Docker method below which is more robust.

## Method 2: AWS App Runner (via Docker/Container) -> **BEST WAY**
Since we created a `Dockerfile`, this is the most reliable method.

1.  **Go to App Runner Console**.
2.  **Source:** Repository (GitHub).
3.  **Configure Build:**
    *   Instead of "Code", create a file named `apprunner.yaml` in your repo (I will create this for you next).
    *   OR, just select **"Source Code"** -> **"Python 3"** like above.
    
    *Wait, actually the easiest way with Dockerfile is to push the Image to ECR, but connecting GitHub to App Runner directly with Python runtime is easier for you now.*

    **Let's use the Python Runtime method directly from GitHub:**
    
    1.  **Build Command:** `pip install -r requirements.txt`
    2.  **Start Command:** `uvicorn src.app:app --host 0.0.0.0 --port 8080`
    3.  **Port:** 8080

## Method 3: AWS Elastic Beanstalk
1.  Search **"Elastic Beanstalk"**.
2.  Create Application -> **"Python"** Platform.
3.  Upload your code (zip `src`, `data`, `requirements.txt`).
4.  Deploy.

## What I have prepared for you
I have added:
- `requirements.txt`: List of python libraries.
- `Dockerfile`: If you want to build a container image later.

**Next Steps for YOU:**
1.  Go to **[AWS App Runner Console](https://console.aws.amazon.com/apprunner)**.
2.  Create Service -> Connect your GitHub.
3.  Use the Build/Start commands mentioned in Method 1.
