# =============================================================================
# 🤖 OpenAI ChatGPT Utilities (openai_utils.py)
# -----------------------------------------------------------------------------
# Purpose:     Handles prompt formatting, API key retrieval, and ChatGPT response parsing
#
# Project:     RMI 360 Imaging Workflow Python Toolbox
# Version:     1.0.0
# Author:      RMI Valuation, LLC
# Created:     2025-05-10
#
# Description:
#   - Sends user prompts to the OpenAI Chat API
#   - Supports keyring or direct API key config
#   - Cleans and parses markdown-wrapped or malformed JSON replies
#
# Exposed Functions:
#   - ask_chatgpt
#   - get_openai_api_key
#   - clean_json_response
#
# Dependencies:
#   - openai, keyring, re, ast, json
# =============================================================================
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
import re
import ast
import json
import keyring
from typing import Any, Dict, Optional

from utils.arcpy_utils import log_message


def clean_json_response(text: str | dict) -> Any:
    """
    Attempts to clean and parse JSON returned by ChatGPT, even if wrapped in markdown or already parsed.

    Supports:
    - Directly returning pre-parsed dicts
    - Removing markdown fences like ```json
    - Fallback to Python literal evaluation

    Args:
        text (str | dict): Raw text response from ChatGPT, or already-parsed dict.

    Returns:
        Any: Parsed Python object (typically a dict), or raises ValueError on failure.
    """
    if isinstance(text, dict):
        return text  # Already parsed — return as-is

    text = text.strip()

    # Remove markdown backticks or ```json
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Failed to parse GPT JSON response: {e}")


def get_openai_api_key(config: Dict) -> str:
    """
    Retrieves the OpenAI API key from the config or system keyring.

    Args:
        config (dict): Config dictionary with OpenAI section.

    Returns:
        str: The OpenAI API key.

    Raises:
        RuntimeError: If keyring is enabled but key is not found.
    """
    openai_cfg = config.get("openai", {})
    use_keyring = openai_cfg.get("keyring_openai", False)
    service_name = openai_cfg.get("keyring_service_name", "rmi_openai")
    if use_keyring:
        api_key = keyring.get_password(service_name, "openai_api_key")
        if not api_key:
            raise RuntimeError(f"❌ OpenAI API key not found in keyring for service '{service_name}'.")
        return api_key
    else:
        api_key = openai_cfg.get("api_key")
        if not api_key:
            raise RuntimeError("❌ OpenAI API key not provided in config.")
        return api_key


def ask_chatgpt(prompt: str, config: Dict, messages: list) -> Optional[dict]:
    """
    Sends a prompt to the OpenAI Chat API and returns the parsed JSON result.

    Args:
        prompt (str): The full formatted user prompt.
        config (dict): Config including OpenAI API model and key info.
        messages (list): Optional message handler for logging.

    Returns:
        dict | None: Parsed JSON response from ChatGPT, or None on error or malformed reply.
    """
    model = config.get("openai", {}).get("model", "gpt-4o")
    api_key = get_openai_api_key(config)
    messages_payload: list[ChatCompletionUserMessageParam] = [{"role": "user", "content": prompt}]
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages_payload,
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        log_message(f"ChatGPT error: {e}", messages, level="error", error_type=RuntimeError, config=config)
        return None
