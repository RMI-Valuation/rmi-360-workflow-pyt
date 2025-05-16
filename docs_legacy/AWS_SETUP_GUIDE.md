# 🛠️ AWS Setup Guide for RMI 360 Imaging Workflow Python Toolbox

This guide documents how to configure AWS credentials, roles, and permissions securely required to support the following 
RMI 360 Imaging Workflow Python Toolbox scripts:

- `copy_to_aws.py`
- `deploy_lambda_monitor.py`
- `lambda_progress_monitor.py`
- `disable_rule.py`

---

## ✅ IAM User Setup (for Local Scripts)
Create a dedicated IAM user (e.g., `project-deployer`) with:
- **Programmatic access only**
- An access key stored in **keyring** under a unique service name (e.g., `aws_s3`)

### Required IAM Policy (Custom)
Attach a policy like this to the user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3Upload",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::<YOUR-BUCKET-NAME>",
        "arn:aws:s3:::<YOUR-BUCKET-NAME>/*"
      ]
    },
    {
      "Sid": "LambdaDeploy",
      "Effect": "Allow",
      "Action": [
        "lambda:CreateFunction",
        "lambda:GetFunction",
        "lambda:UpdateFunctionCode",
        "lambda:AddPermission"
      ],
      "Resource": [
        "arn:aws:lambda:<YOUR-AWS-REGION>:<YOUR-AWS-ACCOUNT-ID>:function:ProgressMonitor", 
        "arn:aws:lambda:<YOUR-AWS-REGION>:<YOUR-AWS-ACCOUNT-ID>:function:DisableUploadMonitorRule"
      ]
    },
    {
      "Sid": "ScheduleRule",
      "Effect": "Allow",
      "Action": [
        "events:PutRule",
        "events:PutTargets"
      ],
      "Resource": "arn:aws:events:<YOUR-AWS-REGION>:<YOUR-AWS-ACCOUNT-ID>:rule/UploadMonitorRule*"
    },
    {
      "Sid": "AllowSTS",
      "Effect": "Allow",
      "Action": "sts:GetCallerIdentity",
      "Resource": "*"
    },
    {
      "Sid": "PassLambdaExecutionRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::<YOUR-AWS-ACCOUNT-ID>:role/LambdaProgressMonitorRole"
    }
  ]
}
```
> - Update `<YOUR-AWS-ACCOUNT-ID>` with your actual AWS account number.
> - Update `<YOUR-BUCKET-NAME>` with your actual AWS bucket name.

### Script Configuration
In `config.yaml`, include:
```yaml
aws:
  region: <YOUR-AWS-REGION>
  s3_bucket: <YOUR-BUCKET-NAME>
  keyring_aws: true
  keyring_service_name: aws-deployer
```

---

## ✅ Lambda Execution Role Setup

### Lambda Role: `LambdaProgressMonitorRole`
Attach these policies to the role used **by the Lambda functions**:

#### 1. **AWSLambdaBasicExecutionRole** *(for logging)*
```json
{
  "Effect": "Allow",
  "Action": [
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": "arn:aws:logs:<YOUR-AWS-REGION>:<YOUR-AWS-ACCOUNT-ID>:log-group:/aws/lambda/LambdaProgressMonitorRole*"
}
```

#### 2. **LambdaUploadMonitorAccess** *(custom policy)*
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListUploads",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::<YOUR-BUCKET-NAME>"
    },
    {
      "Sid": "ReadWriteStatusFiles",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::<YOUR-BUCKET-NAME>/status/*"
    },
    {
      "Sid": "DisableScheduleRule",
      "Effect": "Allow",
      "Action": "events:DisableRule",
      "Resource": "arn:aws:events:<YOUR-AWS-REGION>:<YOUR-AWS-ACCOUNT-ID>:rule/UploadMonitorRule*"
    },
    {
      "Sid": "InvokeDisableRuleLambda",
      "Effect": "Allow",
      "Action": "lambda:InvokeFunction",
      "Resource": "arn:aws:lambda:<YOUR-AWS-REGION>:<YOUR-AWS-ACCOUNT-ID>:function:DisableUploadMonitorRule"
    }
  ]
}
```
> - Update `<YOUR-BUCKET-NAME>` with your actual AWS bucket name.
> - Update `<YOUR-AWS-REGION>` with your actual AWS region.
> - Update `<YOUR-AWS-ACCOUNT-ID>` with your actual AWS account number.

---

## ✅ Lambda Scripts Overview

### `lambda_progress_monitor.py`
- Monitors progress of image upload.
- Generates `status/status.html` and updates `progress_<slug>.json`.
- Invokes `DisableUploadMonitorRule` if stalled or complete.
- Needs access to:
  - `s3:GetObject`, `s3:PutObject` for `status/`
  - `s3:ListBucket`
  - `lambda:InvokeFunction`
  - `events:DisableRule`

### `disable_rule.py`
- Simple Lambda to disable a named CloudWatch Events rule.
- Requires:
  - `events:DisableRule`

Enable static website hosting on the bucket and configure the index document.

---

## 🗝️ Secure Credential Storage
Use Python's `keyring` library to securely store and retrieve access keys:

```bash
keyring set aws-deployer AWS_ACCESS_KEY_ID
keyring set aws-deployer AWS_SECRET_ACCESS_KEY
```
Or programmatically:
```python
keyring.set_password("aws_s3", "AWS_ACCESS_KEY_ID", "<YOUR_AWS_ACCESS_KEY>")
keyring.set_password("aws_s3", "AWS_SECRET_ACCESS_KEY", "<YOUR_AWS_SECRET_KEY>")
```
You may also use the `SetAWSKeyringCredentialsTool` (ArcGIS Python Toolbox) to securely store credentials via UI:
- Label: **Set AWS Keyring Credentials**
- Category: **Setup** 
- Prompts for Access Key ID and Secret Access Key 
- Stores credentials in the service name defined in config (default: `aws_s3` or user-defined)

---

## 🧪 Testing
After setup:
- Run `deploy_lambda_monitor.py` to create Lambdas, CloudWatch rule, and upload initial progress JSON.
- Run `copy_to_aws.py` to upload images.
- Monitor progress at: `https://<bucket>.s3.amazonaws.com/status/status.html`

---

## ✅ Recap: Key Components
| Component                    | Who Uses It             | Access Level           |
|-----------------------------|--------------------------|-------------------------|
| IAM User: project-deployer  | Scripts (local)          | Full S3 + deploy perms  |
| LambdaProgressMonitorRole   | Lambda functions         | S3 read/write, logs     |
| Keyring Service: aws-deployer | Scripts                | AWS credentials source  |