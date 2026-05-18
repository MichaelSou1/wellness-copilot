from typing import List
from langchain_openai import ChatOpenAI
from . import config  # Ensure .env is loaded

LLM_MAX_ATTEMPTS = 3
# ChatOpenAI/OpenAI SDK `max_retries` counts retries after the first attempt.
LLM_MAX_RETRIES = LLM_MAX_ATTEMPTS - 1


def _validate_llm_config():
    if not config.LLM_MODEL:
        raise ValueError("LLM_MODEL is not set. Please configure it in your .env file.")
    if not config.LLM_API_KEY:
        raise ValueError("LLM_API_KEY is not set. Please configure it in your .env file.")


def _base_llm_kwargs():
    _validate_llm_config()
    kwargs = {
        "model": config.LLM_MODEL,
        "base_url": config.LLM_BASE_URL,
        "api_key": config.LLM_API_KEY,
        "max_retries": LLM_MAX_RETRIES,
    }

    # GLM-4.7 / GLM-5 系列用 thinking.type 控制思考模式。
    # ChatOpenAI 会把 extra_body 合并进 OpenAI-compatible 请求体。
    if config.LLM_DISABLE_THINKING:
        kwargs["extra_body"] = {
            "thinking": {
                "type": "disabled",
            }
        }

    return kwargs


def _responses_mode_kwargs():
    return {
        "use_responses_api": True,
        "output_version": config.LLM_OUTPUT_VERSION,
    }


def create_llm():
    kwargs = _base_llm_kwargs()
    if config.LLM_API_MODE != "responses":
        return ChatOpenAI(**kwargs)

    try:
        # Check if model_kwargs exists in kwargs to avoid duplicate keyword argument conflict
        base_model_kwargs = kwargs.pop("model_kwargs", {})
        # If model_kwargs exists, we might need to pass it merged if supported or separately
        if base_model_kwargs:
            return ChatOpenAI(**kwargs, model_kwargs=base_model_kwargs, **_responses_mode_kwargs())
        return ChatOpenAI(**kwargs, **_responses_mode_kwargs())
    except TypeError:
        # 兼容旧版 langchain-openai：将 responses 配置下沉到 model_kwargs。
        responses_kwargs = _responses_mode_kwargs()
        if "base_model_kwargs" in locals() and base_model_kwargs:
            responses_kwargs.update(base_model_kwargs)
        return ChatOpenAI(
            **kwargs,
            model_kwargs=responses_kwargs,
        )


def _collect_text_parts(value, parts: List[str]):
    if value is None:
        return
    if hasattr(value, "content"):
        _collect_text_parts(value.content, parts)
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            parts.append(text)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text_parts(item, parts)
        return
    if isinstance(value, dict):
        text_value = value.get("text")
        if isinstance(text_value, str):
            text = text_value.strip()
            if text:
                parts.append(text)
            return
        if text_value is not None:
            _collect_text_parts(text_value, parts)
            return
        for key in ("content", "output_text", "value"):
            if key in value and value[key] is not None:
                _collect_text_parts(value[key], parts)
        return


def extract_text_content(message_or_content) -> str:
    parts: List[str] = []
    _collect_text_parts(message_or_content, parts)
    return "\n".join(parts)


llm = create_llm()
