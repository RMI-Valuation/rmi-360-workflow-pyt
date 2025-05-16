# 🛠️ Tool: Copy to AWS (and Set AWS Keyring Credentials)

## 🧑‍💻 Tool Name
**09 – Copy To AWS**  
**(includes: Set AWS Keyring Credentials)**

---

## 📝 Purpose

Uploads final, renamed, and metadata-tagged images to an AWS S3 bucket using multithreaded parallel uploads. Supports credentials from system keyring or config, uploads only new/updated files, logs results, supports dry-run, and can deploy a Lambda monitor for progress tracking. Includes tool for securely storing AWS credentials.

---

## 🧰 Parameters

| Parameter                   | Required | Description                                                         |
|-----------------------------|----------|---------------------------------------------------------------------|
| Input Folder for Images     | ✅       | Folder containing `.jpg` images to upload                           |
| Skip Existing Files in S3?  | ⬜️      | If checked, skips images already present in S3                      |
| Project Folder              | ✅       | Root folder for the project (for logs, assets)                      |
| Config File                 | ✅       | Path to `config.yaml` with AWS and project settings                 |
| Deploy Upload Monitor?      | ⬜️      | Whether to deploy AWS Lambda monitor before transfer                |

**Set AWS Keyring Credentials** (sub-tool):

| Parameter             | Required | Description                     |
|----------------------|----------|---------------------------------|
| AWS Access Key ID    | ✅       | Your AWS access key ID          |
| AWS Secret Access Key| ✅       | Your AWS secret key (stored securely) |

---

## 🗂️ Scripts & Components

| Script                              | Role/Responsibility                                |
|-------------------------------------|----------------------------------------------------|
| `tools/copy_to_aws_tool.py`         | ArcGIS Toolbox wrapper, parameter handling          |
| `utils/copy_to_aws.py`              | Handles upload logic, retry, logging                |
| `utils/deploy_lambda_monitor.py`    | Deploys AWS Lambda monitor for upload tracking      |
| `utils/manager/config_manager.py`   | Loads and validates configuration                  |

---

## ⚙️ Behavior / Logic

1. Loads AWS credentials from keyring or config.
2. Scans the input folder for `.jpg` images.
3. Optionally skips files already present in S3.
4. Uploads files in parallel using TransferManager.
5. Writes detailed logs and summary CSV.
6. Optionally deploys Lambda monitor for progress tracking.

---

## 🗃️ Inputs

- Folder of renamed and tagged images
- Project YAML config with AWS credentials and S3 settings

---

## 📤 Outputs

- Images uploaded to target S3 bucket/folder
- Upload logs and summary CSV in the project folder

---

## 🗝️ Configuration / Notes

- AWS credentials and S3 settings are defined in `config.yaml`:

```yaml
aws:
  keyring_aws: true
  access_key_id: "..."  # Only if keyring_aws is false
  secret_access_key: "..."  # Only if keyring_aws is false
  s3_bucket: "rmi-orient-img-test"
  s3_folder: "project_name"
```

- If `keyring_aws: true`, credentials are securely stored in the system keyring.
- Supports resumable uploads, concurrency, and cancel triggers.

---

## 🧩 Dependencies

- Python with `boto3`, `botocore`
- ArcGIS Pro
- Project YAML config

---

## 🔗 Related Tools

- Rename and Tag Images
- Add Images to OID
- Generate OID Service

---

## ✅ Validation

Run via `validate_tool_copy_to_aws()`

- Confirms presence of: `region`, `s3_bucket`, `s3_bucket_folder`
- Verifies credentials are resolvable via keyring or config
- Validates all optional parameters (`retries`, `max_workers`, etc.)
- Ensures upload folder is resolvable

---

## 📝 Notes

- 🛠 Run this **after `Rename and Tag Images`** is complete
- ✅ Safe to re-run with `skip_existing = true`
- 🧵 Multithreaded: Uploads use 5 threads by default
- 🔐 Keyring storage is cross-platform secure via the OS’s credential manager
