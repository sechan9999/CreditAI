# Deploying to AWS Lambda with Function URL

This project relies on `Mangum` to adapt FastAPI for AWS Lambda.

## Prerequsite
- AWS Account
- Python 3.9+ installed locally

## Steps to Deploy Manually

1.  **Prepare Deployment Package (Zip)**
    AWS Lambda requires all dependencies to be included in the zip file.
    
    ```powershell
    # 1. Install dependencies to a local folder
    mkdir package
    pip install -r requirements.txt --target ./package
    
    # 2. Copy source code into package
    xcopy /E /I src package\src
    xcopy /E /I data package\data
    
    # 3. Create Zip file (Requires 7-zip or similar, or python command)
    # Using Python to zip 'package' folder content into 'function.zip'
    python -c "import shutil; shutil.make_archive('function', 'zip', 'package')"
    ```

2.  **Create Lambda Function**
    - Go to **AWS Lambda Console**.
    - **Create Function** -> "Author from scratch".
    - **Runtime:** Python 3.11 (or match your local version).
    - **Architecture:** x86_64.
    - Click **Create function**.

3.  **Upload Code**
    - **Code Source** -> **Upload from** -> **.zip file**.
    - Upload `function.zip`.

4.  **Configure Handler**
    - Go to **Runtime settings** -> **Edit**.
    - **Handler:** `src.app.handler` (pointing to the Mangum handler object).

5.  **Enable Function URL**
    - Go to **Configuration** tab -> **Function URL**.
    - **Create function URL**.
    - **Auth type:** `NONE` (for public access like the demo link you shared).
    - **CORS:** Enable. Allow origins `*`.
    - Click **Save**.

6.  **Test**
    - Copy the Function URL (e.g., `https://xyz...lambda-url.region.on.aws`).
    - Use it in your Dashboard or Curl.

## Note on Model File
The machine learning model (`best_scoring_model.pkl`) is included in the package. Ensure the Lambda size limits (50MB direct upload, 250MB unzipped) are respected. Our model is small so it fits easily.
