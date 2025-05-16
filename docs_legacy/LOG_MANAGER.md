
# 🧾 LogManager Utility

> **Module:** `utils/log_manager.py`  
> **Project:** RMI 360 Imaging Workflow Python Toolbox  
> **Author:** RMI Valuation, LLC  
> **Version:** 1.0.0  
> **Created:** 2025-05-10

---

## 📌 Overview

`LogManager` provides structured, multi-format logging for command-line workflows, ArcGIS Pro tools, and orchestration scripts.

It supports:
- Multi-level logging (`info`, `debug`, `warning`, `error`)
- Context metadata (e.g., `tool`, `reel`, `duration`)
- HTML logs with collapsible steps and toggleable timestamps
- JSON logs for structured post-analysis
- ArcPy-compatible messages when a `messages` object is passed
- Easy integration with `PathManager` for export control

---

## ✅ Features

- Plaintext logging to `.txt`
- Structured JSON export via `.export_json()`
- Beautiful HTML logging with:
  - Indentation tracking via `push()` / `pop()`
  - `step()` context manager for scoped logging
  - Toggleable timestamps
- Fun extras: `success()`, `party()`, `fireworks()`

---

## 🛠 Initialization

```python
from utils.manager.log_manager import LogManager
from utils.manager.path_manager import PathManager

pm = PathManager(project_base="path/to/project")
log = LogManager(config={"debug_messages": True}, path_manager=pm)
```

---

## ✏️ Logging Examples

```python
log.info("Starting enhancement", context={"tool": "Enhance"})
log.debug("Loaded EXIF data", context={"images": 112})

log.push("Apply corrections")
log.warn("Missing metadata", context={"reel": "r002"})
log.pop("Corrections applied")

with log.step("Write outputs"):
    log.info("Saved images to disk")

log.success("Enhancement complete")
log.export_all("enhancement_log")
```

---

## 📤 Export Methods

```python
log.export_txt("mylog.txt")
log.export_html("mylog.html")
log.export_json("mylog.json")
log.export_all("mylog")  # exports all 3
```

---

## 📦 Context Metadata

Every log call can accept a `context={}` dictionary, which:
- Appends key-value info to `.txt`/`.html`
- Is structured in `.json`

```python
log.info("Exporting", context={"bucket": "prod-bucket", "file_count": 24})
```

---

## 📄 Method Summary

| Method               | Description                                      |
|----------------------|--------------------------------------------------|
| `log()`              | Core method for logging all levels               |
| `info()` / `warn()`  | Informational and warning logging                |
| `debug()` / `error()`| Debugging or error-level logging                 |
| `push()` / `pop()`   | Track step entry and exit with indentation       |
| `step()`             | Context manager version of push/pop              |
| `export_txt()`       | Write plaintext log                              |
| `export_html()`      | Write styled HTML log                            |
| `export_json()`      | Export structured log entries                    |
| `export_all()`       | Run all exports with shared filename             |
| `success()`          | Log a celebratory message                        |
| `fireworks()`        | Log celebratory emoji                            |
| `party()`            | Log ASCII celebration                            |

---

## 🔁 Dependencies

- Internal: `utils/path_manager.py`
- External: `html`, `json`, `contextlib`, `time`, `datetime`
