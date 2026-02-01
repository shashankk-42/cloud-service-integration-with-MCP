# AWS Credential Setup Guide

Follow these steps to get your AWS credentials for the Cloud Service Integration project.

## Step 1: Create an AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click **Create an AWS Account**
3. Follow the signup process (credit card required, but free tier available)

## Step 2: Create an IAM User

1. In AWS Console, go to **IAM** (search in top bar)
2. Click **Users** → **Create user**
3. Enter a username (e.g., `cloud-integration-dev`)
4. Click **Next**

## Step 3: Set Permissions

1. Select **Attach policies directly**
2. Search and check **AdministratorAccess** (for development)
3. Click **Next** → **Create user**

## Step 4: Create Access Keys

1. Click on the user you just created
2. Go to **Security credentials** tab
3. Click **Create access key**
4. Select **Command Line Interface (CLI)**
5. Click **Next** → **Create access key**
6. **Download the .csv file** or copy the keys

## Step 5: Update .env

Open your `.env` file and replace the placeholders:

```
AWS_ACCESS_KEY_ID=your_actual_access_key_id
AWS_SECRET_ACCESS_KEY=your_actual_secret_access_key
```

## Google Gemini API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Click **Get API Key** → **Create API Key**
3. Copy the key and add to `.env`:

```
GOOGLE_API_KEY=your_actual_gemini_api_key
```

## Verification

After setting credentials, run:

```bash
python -c "from langchain_google_genai import ChatGoogleGenerativeAI; print('Gemini OK')"
```
