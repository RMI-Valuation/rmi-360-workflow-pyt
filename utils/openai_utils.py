from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
import json
import keyring

from utils.arcpy_utils import log_message


def get_openai_api_key(config):
    openai_cfg = config.get("openai", {})
    use_keyring = openai_cfg.get("keyring_openai", False)
    service_name = openai_cfg.get("keyring_service_name", "rmi_openai")
    if use_keyring:
        api_key = keyring.get_password(service_name, "openai_api_key")
        if not api_key:
            raise RuntimeError(f"❌ OpenAI API key not found in keyring for service '{service_name}'.")
        return api_key
    else:
        api_key = openai_cfg.get("api_key", {})
        return api_key


def ask_chatgpt(prompt, config, messages):
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
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            log_message(f"⚠️ GPT returned non-JSON output: {content}", messages, level="error", config=config)
            return None

    except Exception as e:
        log_message(f"ChatGPT error: {e}", messages, level="error", error_type=RuntimeError, config=config)
        return None