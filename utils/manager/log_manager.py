# =============================================================================
# 🗾 Log Manager Utility (utils/manager/log_manager.py)
# -----------------------------------------------------------------------------
# Purpose:             Provides structured logging across CLI, ArcGIS, and web output.
# Project:             RMI 360 Imaging Workflow Python Toolbox
# Version:             1.1.0
# Author:              RMI Valuation, LLC
# Created:             2025-05-11
# Last Updated:        2025-05-15
#
# Description:
#   Centralized logging framework that supports text, HTML, and JSON output.
#   Includes step-based indentation, duration tracking, context metadata, and
#   optional "celebration" features. Designed to support both ArcPy tools and
#   CLI pipelines. Compatible with PathManager for output directory control.
#
# File Location:        /utils/manager/log_manager.py
# Called By:            Tools, orchestrators, testing pipelines
# Int. Dependencies:    utils/manager/path_manager
# Ext. Dependencies:    json, time, html, contextlib, typing, datetime
#
# Documentation:
#   See: docs_legacy/LOG_MANAGER.md
#   (Ensure this doc is current; update if needed.)
#
# Notes:
#   - HTML logs are collapsible with toggleable timestamps.
#   - JSON output includes metadata for downstream automation.
#   - Supports playful methods like .party(), .fireworks(), and .success().
#   - Integrates with PathManager for file output and log organization.
# =============================================================================

from datetime import datetime
from typing import Optional, List, Type, Dict, Generator
from contextlib import contextmanager
import time
import html
import json
from utils.manager.path_manager import PathManager

class LogManager:
    """
    LogManager provides structured logging for command-line, ArcGIS, and web-based workflows.

    Logs are collected in plain text, HTML, and JSON formats with support for:
    - Multiple log levels (info, warning, error, debug)
    - Step-based indentation and timing
    - Context metadata (tool, file, reel, etc.)
    - ArcGIS Pro messaging (if a message object is provided)
    - Export to .txt, .html, and .json files via PathManager

    Designed for use in orchestration scripts, CLI tools, and ArcGIS Python Toolbox tools.
    
    """

    LEVEL_PREFIX = {
        "debug": "🛠️",
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅",
        "custom": ""  # Emoji passed as parameter
    }
    # When True, show [indent] after emoji in log output if debug_messages is True
    SHOW_INDENT_LEVEL_IN_DEBUG = True
    # When True, show context in log output if debug_messages is True
    SHOW_CONTEXT_IN_DEBUG = False

    def __init__(
            self,
            messages=None,
            config: Optional[dict] = None,
            path_manager: Optional[PathManager] = None,
            enable_file_output: bool = True,
    ):
        """
        Initialize the LogManager.

        Args:
            messages (optional): ArcPy messaging object for integrated tool messages.
            config (dict, optional): Project-level configuration, including debug flag.
            path_manager (PathManager, optional): Provides log file directories.
            enable_file_output (bool): If True, log to .txt, .html, and .json files.
        """
        self.messages = messages
        self.config = config or {}
        self.path_manager = path_manager
        self.enable_file_output = enable_file_output
        self.entries: List[str] = []
        self.html_blocks: List[str] = []
        self.records: List[Dict] = []
        self.depth = 0
        self.indent_char = "    "
        self._timing_stack: List[float] = []
        self._context_stack: List[Optional[Dict]] = []

    def push(self, msg: str) -> None:
        """
        Start a new indented log section and begin timing it.
        Args:
            msg (str): Header message for the step or section.
        """
        self.info(msg)
        self.depth += 1
        self._timing_stack.append(time.time())

    def log(self, msg: str, level: str = "info", context: Optional[Dict] = None, indent: int = 0, error_type: Optional[Type[Exception]] = None, emoji: Optional[str] = None) -> None:
        """
        Log a message with a specified level, optional context, and explicit indentation.

        Args:
            msg (str): The message to log.
            level (str): Logging level. One of 'debug', 'info', 'warning', 'error', 'success', or 'custom'.
            context (dict, optional): Metadata context to include in the log (default: None).
            indent (int, optional): Indentation level (number of indent units, default: 0).
            error_type (Type[Exception], optional): Exception type to raise if level is 'error' (default: None).
            emoji (str, optional): Emoji to use for 'custom' level messages (default: None).

        For 'custom' level, provide an emoji.
        For 'error', if error_type is given, raises after logging (indent forced to 0 unless overridden).
        """
        valid_levels = {"debug", "info", "warning", "error", "success", "custom"}
        if level not in valid_levels:
            self.log(f"⚠️ Invalid log level '{level}' (defaulting to info)", level="warning", indent=indent)
            level = "info"

        if level == "debug" and not self.config.get("debug_messages", False):
            return

        # Use context from stack if not provided
        if context is None and self._context_stack:
            context = self._context_stack[-1]

        # For custom, emoji must be provided
        if level == "custom":
            prefix = emoji or self.LEVEL_PREFIX["custom"]
        else:
            prefix = self.LEVEL_PREFIX.get(level, "")

        # Add indent level in debug mode if enabled
        if self.config.get("debug_messages", False) and self.SHOW_INDENT_LEVEL_IN_DEBUG:
            prefix = f"{prefix} [{indent}]"

        context_str = ""
        if context and isinstance(context, dict):
            if self.config.get("debug_messages", False) and self.SHOW_CONTEXT_IN_DEBUG:
                context_str = "  " + " ".join(f"{k}={v}" for k, v in context.items())

        indent_str = self.indent_char * indent
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        iso_timestamp = datetime.now().isoformat()
        full_msg = f"[{timestamp}] {prefix} {indent_str}{msg}{context_str}"
        self.entries.append(full_msg)

        self.records.append({
            "timestamp": iso_timestamp,
            "level": level,
            "message": msg,
            "indent": indent,
            "context": context or {}
        })

        css_class = level
        # Enhanced HTML log generation for indentation and collapsibility
        # Use <details>/<summary> for steps (indent=0 with separator), .block for indented messages
        if not hasattr(self, '_open_details'):
            self._open_details = False
        html_line = None
        if indent == 0 and (msg.strip() == '=' * 40 or (level == 'custom' and emoji == '▶️')):
            # Start or end of a step, use <details>/<summary>
            if msg.strip() == '=' * 40:
                # Separator, close previous <details> if open
                if self._open_details:
                    self.html_blocks.append('</div></details>')
                    self._open_details = False
            else:
                # Start a new collapsible block for the step
                if self._open_details:
                    self.html_blocks.append('</div></details>')
                summary = html.escape(msg + (context_str if context_str else ''))
                self.html_blocks.append(f'<details open><summary><span class="ts">[{timestamp}]</span> {prefix} {summary}</summary><div class="block">')
                self._open_details = True
            # Do not add the separator/step message as a normal line
            html_line = None
        else:
            block_class = 'block' if indent > 0 else ''
            html_line = f'<div class="{css_class} {block_class}"><span class="ts">[{timestamp}]</span> {prefix} {html.escape(msg + context_str)}</div>'
        if html_line:
            self.html_blocks.append(html_line)


        if self.messages:
            if level == "warning" and hasattr(self.messages, "addWarningMessage"):
                self.messages.addWarningMessage(full_msg)
            elif level == "error" and hasattr(self.messages, "addErrorMessage"):
                self.messages.addErrorMessage(full_msg)
            elif hasattr(self.messages, "addMessage"):
                self.messages.addMessage(full_msg)
        else:
            print(full_msg)

        if self.enable_file_output and self.path_manager:
            try:
                log_file = self.path_manager.logs / "process_log.txt"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(full_msg + "\n")
            except Exception as e:
                if level == "warning":
                    print(f"⚠️ Could not write to log file: {e}")
                else:
                    # Avoid recursion
                    print(f"Could not write to log file: {e}")

        if level == "error" and error_type:
            raise error_type(full_msg)

    def debug(self, msg: str, context: Optional[Dict] = None, indent: int = 0) -> None:
        self.log(msg, level="debug", context=context, indent=indent)

    def info(self, msg: str, context: Optional[Dict] = None, indent: int = 0) -> None:
        self.log(msg, level="info", context=context, indent=indent)

    def warning(self, msg: str, context: Optional[Dict] = None, indent: int = 0) -> None:
        self.log(msg, level="warning", context=context, indent=indent)

    def error(self, msg: str, context: Optional[Dict] = None, indent: int = 0, error_type: Optional[Type[Exception]] = None) -> None:
        # If error_type is provided, always log at indent=0 unless overridden
        use_indent = 0 if error_type is not None and indent == 0 else indent
        self.log(msg, level="error", context=context, indent=use_indent, error_type=error_type)

    def success(self, msg: str, context: Optional[Dict] = None, indent: int = 0) -> None:
        self.log(msg, level="success", context=context, indent=indent)

    def custom(self, msg: str, emoji: str, context: Optional[Dict] = None, indent: int = 0) -> None:
        self.log(msg, level="custom", context=context, indent=indent, emoji=emoji)



    @staticmethod
    def _format_duration(seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s" if mins else f"{secs}s"

    def get_messages(self) -> List[str]:
        return self.entries

    def export_txt(self, filename: str = "process_log.txt") -> None:
        """Export plain text log to .txt file.

        Args:
            filename (str): Output filename for the text log.
        """
        if not self.path_manager:
            self.warning("[LogManager] No PathManager available for export_txt().")
            return
        try:
            txt_path = self.path_manager.logs / filename
            with open(txt_path, "w", encoding="utf-8") as f:
                for msg in self.entries:
                    f.write(msg + "\n")
        except Exception as e:
            self.warning(f"Failed to write TXT log: {e}")

    def export_json(self, filename: str = "process_log.json") -> None:
        """Export log records to a structured JSON file.

        Args:
            filename (str): Filename for the JSON export.
        """
        if not self.path_manager:
            self.warning("[LogManager] No PathManager available for export_json().")
            return
        try:
            json_path = self.path_manager.logs / filename
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            self.warning(f"Failed to export JSON log: {e}")

    def export_html(self, filename: Optional[str] = "process_log.html") -> None:
        """Export log output to an HTML file with collapsible blocks.

        Args:
            filename (str): Output filename for the HTML log.
        """
        if not self.path_manager:
            self.warning("[LogManager] No PathManager available for export_html().")
            return
        try:
            html_path = self.path_manager.logs / filename
            html_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure any open <details> is closed
            if hasattr(self, '_open_details') and self._open_details:
                self.html_blocks.append('</div></details>')
                self._open_details = False
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(HTML_LOG_TEMPLATE_HEAD)
                for line in self.html_blocks:
                    f.write(line + "\n")
                f.write(HTML_LOG_TEMPLATE_FOOT)
        except Exception as e:
            self.warning(f"Failed to export HTML log: {e}")

    def export_all(self, basename: str = "process_log") -> None:
        """Export log to .txt, .html, and .json using a common base filename."""
        self.export_txt(f"{basename}.txt")
        self.export_json(f"{basename}.json")
        self.export_html(f"{basename}.html")

    @contextmanager
    def step(self, msg: str, context: Optional[Dict] = None):
        """
        Context manager for logging the start and end of a logical step.
        Logs separators before and after the step message at indent=0.
        Any log messages inside should use indent=1 for proper nesting.
        Usage:
            with logger.step("Run Mosaic Processor"):
                logger.info("Hello", indent=1)
        """
        self._context_stack.append(context)
        sep = "=" * 40
        self.custom(sep, emoji="▶️", indent=0)
        self.custom(msg, emoji="▶️", indent=0, context=context)
        self.custom(sep, emoji="▶️", indent=0)
        start_time = time.time()
        try:
            yield
        except Exception as e:
            elapsed = time.time() - start_time
            elapsed_str = self._format_duration(elapsed)
            self.error(f"{msg} failed: {e} (Elapsed: {elapsed_str})", indent=0)
            self.custom(sep, emoji="▶️", indent=0)
            raise
        else:
            elapsed = time.time() - start_time
            elapsed_str = self._format_duration(elapsed)
            self.success(f"{msg} complete (Elapsed: {elapsed_str})", indent=0)
        finally:
            self._context_stack.pop()


# --- HTML Log Template Components ---

HTML_LOG_TEMPLATE_HEAD = """
<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>Process Log</title>
<style>
    body {
        font-family: "Fira Code", Consolas, monospace;
        background-color: #ffffff;
        color: #222;
        padding: 2em;
        line-height: 1.6;
    }
    h1 {
        font-size: 1.8em;
        margin-bottom: 0.5em;
    }
    summary {
        font-weight: bold;
        font-size: 1.05em;
        cursor: pointer;
        margin-top: 1em;
        color: #333;
    }
    .ts {
        color: #888;
        font-size: 0.9em;
        display: inline-block;
        width: 180px;
        margin-right: 1em;
    }
    .info {
        color: #333;
    }
    .warning {
        color: #e65100;
        background-color: #fff8e1;
        padding: 2px 4px;
        border-left: 4px solid #f57c00;
    }
    .error {
        color: #b71c1c;
        background-color: #ffebee;
        padding: 2px 4px;
        border-left: 4px solid #d32f2f;
        font-weight: bold;
    }
    .debug {
        color: #777;
        font-style: italic;
    }
    .block {
        margin-left: 1.5em;
    }
    pre {
        white-space: pre-wrap;
    }
    label {
        font-size: 0.9em;
        display: block;
        margin-bottom: 1em;
        color: #444;
    }
</style>
<script>
function toggleTimestamps() {
    const ts = document.querySelectorAll('.ts');
    ts.forEach(span => span.style.display = span.style.display === 'none' ? 'inline-block' : 'none');
}
</script>
</head>
<body>
<h1>📄 Process Log</h1>
<label><input type="checkbox" onclick="toggleTimestamps()"> Show/Hide Timestamps</label>
<hr>
"""

HTML_LOG_TEMPLATE_FOOT = "</body></html>"
