# 🛠️ Tool: Generate OID Service

## 🧑‍💻 Tool Name
**Generate OID Service**

---

## 📝 Purpose

Publishes an Oriented Imagery Dataset (OID) as a web service, enabling access in ArcGIS Pro, web apps, and other clients. Automates configuration, registration, and deployment to ArcGIS Enterprise or Online, including setting permissions and logging.

---

## 🧰 Parameters

| Parameter            | Required | Description                                      |
|----------------------|----------|--------------------------------------------------|
| OID Feature Class    | ✅       | Input OID to publish                             |
| Config File          | ✅       | Path to `config.yaml` with service settings       |
| Project Folder       | ✅       | Project root for resolving outputs                |

---

## 🗂️ Scripts & Components

| Script                                  | Role/Responsibility                |
|-----------------------------------------|------------------------------------|
| `tools/generate_oid_service_tool.py`    | ArcGIS Toolbox wrapper             |
| `utils/generate_oid_service.py`         | Core publishing logic              |
| `utils/manager/config_manager.py`       | Loads and validates configuration  |

---

## ⚙️ Behavior / Logic

1. Loads service parameters from config.
2. Registers OID with ArcGIS Enterprise/Online.
3. Publishes as a web service.
4. Sets sharing and access permissions.
5. Logs results and errors.

---

## 🗃️ Inputs

- OID feature class
- Project YAML config with service settings

---

## 📤 Outputs

- Published OID web service
- Service URL and logs

---

## 🗝️ Configuration / Notes

From `config.yaml`:

```yaml
generate_oid_service:
  portal_url: "https://myportal.domain.com/portal"
  service_name: "project_oid_service"
  sharing: "public"
  folder: "OID Services"
  portal_tags:
    - "config.project.number"
    - "Oriented Imagery"
  summary: "'Oriented Imagery for ' + config.project.number + ' ' + config.project.rr_name + ' - ' + config.project.description"
```

---

## 📤 Output

| Output | Description |
|--------|-------------|
| OID Feature Class | Copied and updated with public AWS URLs |
| Portal Item | Hosted oriented imagery item published using ArcGIS Pro credentials |
| Sharing | Controlled via `portal.share_with` (`PRIVATE`, `ORGANIZATION`, or `PUBLIC`) |

---

## ✅ Validation

Validation is performed by `validate_tool_generate_oid_service()` in `utils/validators`:

- Checks:
  - `portal.project_folder`, `portal.share_with`, and `portal_tags` are defined
  - `aws.s3_bucket`, `region`, and `s3_bucket_folder` resolve correctly
  - Optional fields like `summary` are validated and resolved from config

---

## 📝 Notes

- This tool does **not upload images** – images must already be uploaded via `Copy to AWS`
- Publishing requires being signed in to ArcGIS Online or Enterprise Portal in ArcGIS Pro
- The duplicated OID (e.g., `OID_AWS`) is stored in the same GDB as the original
- Safe to re-run: existing hosted services will be replaced if the name matches
