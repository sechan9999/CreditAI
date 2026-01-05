# Deploying Dashboard to AWS S3 (Static Website Hosting)

Now that your API is running on AWS Lambda, let's host the dashboard HTML on S3 so anyone can access it.

## 1. Create S3 Bucket
1. Go to **AWS Console** -> **S3**.
2. Click **Create bucket**.
3. **Bucket name:** `telco-credit-dashboard-sechan` (Must be unique globally, try adding numbers if taken).
4. **Object Ownership:** ACLs disabled (Recommended).
5. **Block Public Access settings for this bucket:**
   - ⚠️ **Uncheck** "Block all public access". (We need it public).
   - Check the warning box "I acknowledge that...".
6. Click **Create bucket**.

## 2. Upload File
1. Click on your new bucket name.
2. Click **Upload**.
3. Click **Add files** -> Select `reports/dashboard.html` from your project folder.
   - Or just drag and drop the file.
4. Click **Upload**.

## 3. Configure Static Website Hosting
1. Go to **Properties** (속성) tab of the bucket.
2. Scroll to the very bottom: **Static website hosting**.
3. Click **Edit**.
4. Select **Enable**.
5. **Index document:** `dashboard.html`.
6. Click **Save changes**.

## 4. Make File Public (Policy)
1. Go to **Permissions** (권한) tab.
2. Scroll to **Bucket policy** -> **Edit**.
3. Paste this JSON (Replace `YOUR_BUCKET_NAME` with your actual bucket name):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/*"
        }
    ]
}
```
4. Click **Save changes**.

## 5. Access Your Dashboard
1. Go back to **Properties** -> **Static website hosting** (bottom).
2. You will see a **Bucket website endpoint** URL (e.g., `http://telco-credit...s3-website...`).
3. Click it! Your dashboard is now live.
