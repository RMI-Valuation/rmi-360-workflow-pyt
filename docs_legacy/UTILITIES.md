
# 🧰 Shared Utilities – RMI 360 Imaging Workflow Python Toolboxx

This guide explains the utility functions and helpers that power nearly every tool in the Mosaic 360 Toolbox. These are mostly located in the `utils/` directory and are reused across tools, orchestrators, and validation scripts.


---

## 🧠 Expression Resolution

### Location: `utils/expression_utils.py`

Used for resolving dynamic expressions in filenames, metadata, and custom fields.

- `resolve_expression(expr, row, config)` — main resolver
- `resolve_field_expr()` — parses `field.AcquisitionDate.date(...)`
- `resolve_config_expr()` — parses `config.project.slug.strip("-")`

Supports chaining, functions, and date formatting.

---

## 🧾 CSV and EXIF Helpers

### `write_csv(path, rows, headers)`
- Writes a list of dicts to a CSV file

### `build_exiftool_args_file()`
- Used to create `.args` batch file for ExifTool
- Automatically quoted and formatted for multiline execution

---

## 📚 Related Documentation

- [Configuration Guide](./CONFIG_GUIDE.md)
- [Tool Guides](./TOOL_GUIDES.md)
