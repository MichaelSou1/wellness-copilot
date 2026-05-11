from typing import List
from langchain_openai import ChatOpenAI
from . import config  # Ensure .env is loaded


def _validate_llm_config():
    if not config.LLM_MODEL:
        raise ValueError("LLM_MODEL is not set. Please configure it in your .env file.")
    if not config.LLM_API_KEY:
        raise ValueError("LLM_API_KEY is not set. Please configure it in your .env file.")


def _base_llm_kwargs():
    _validate_llm_config()
    return {
        "model": config.LLM_MODEL,
        "base_url": config.LLM_BASE_URL,
        "api_key": config.LLM_API_KEY,
    }


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
        return ChatOpenAI(**kwargs, **_responses_mode_kwargs())
    except TypeError:
        # 兼容旧版 langchain-openai：将 responses 配置下沉到 model_kwargs。
        return ChatOpenAI(
            **kwargs,
            model_kwargs=_responses_mode_kwargs(),
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
