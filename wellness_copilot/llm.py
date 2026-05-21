from typing import List
from langchain_openai import ChatOpenAI
from . import config  # Ensure .env is loaded

LLM_MAX_ATTEMPTS = 3
# ChatOpenAI/OpenAI SDK `max_retries` counts retries after the first attempt.
LLM_MAX_RETRIES = LLM_MAX_ATTEMPTS - 1


def _validate_llm_config(profile: dict):
    if not profile.get("model"):
        raise ValueError(f"{profile['label']}_MODEL is not set. Please configure it in your .env file.")
    if not profile.get("api_key"):
        raise ValueError(f"{profile['label']}_API_KEY is not set. Please configure it in your .env file.")


def _profile(
    *,
    label: str,
    base_url: str,
    api_key: str | None,
    model: str | None,
    api_mode: str,
    output_version: str,
    disable_thinking: bool,
) -> dict:
    return {
        "label": label,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "api_mode": api_mode,
        "output_version": output_version,
        "disable_thinking": disable_thinking,
    }


DEFAULT_LLM_PROFILE = _profile(
    label="LLM",
    base_url=config.LLM_BASE_URL,
    api_key=config.LLM_API_KEY,
    model=config.LLM_MODEL,
    api_mode=config.LLM_API_MODE,
    output_version=config.LLM_OUTPUT_VERSION,
    disable_thinking=config.LLM_DISABLE_THINKING,
)

ORCHESTRATOR_LLM_PROFILE = _profile(
    label="ORCHESTRATOR_LLM",
    base_url=config.ORCHESTRATOR_LLM_BASE_URL,
    api_key=config.ORCHESTRATOR_LLM_API_KEY,
    model=config.ORCHESTRATOR_LLM_MODEL,
    api_mode=config.ORCHESTRATOR_LLM_API_MODE,
    output_version=config.ORCHESTRATOR_LLM_OUTPUT_VERSION,
    disable_thinking=config.ORCHESTRATOR_LLM_DISABLE_THINKING,
)


def _base_llm_kwargs(profile: dict):
    _validate_llm_config(profile)
    kwargs = {
        "model": profile["model"],
        "base_url": profile["base_url"],
        "api_key": profile["api_key"],
        "max_retries": LLM_MAX_RETRIES,
    }

    # GLM-4.7 / GLM-5 系列用 thinking.type 控制思考模式。
    # ChatOpenAI 会把 extra_body 合并进 OpenAI-compatible 请求体。
    if profile.get("disable_thinking"):
        kwargs["extra_body"] = {
            "thinking": {
                "type": "disabled",
            }
        }

    return kwargs


def _responses_mode_kwargs(profile: dict):
    return {
        "use_responses_api": True,
        "output_version": profile["output_version"],
    }


def create_llm(profile: dict | None = None):
    profile = profile or DEFAULT_LLM_PROFILE
    kwargs = _base_llm_kwargs(profile)
    if profile["api_mode"] != "responses":
        return ChatOpenAI(**kwargs)

    try:
        # Check if model_kwargs exists in kwargs to avoid duplicate keyword argument conflict
        base_model_kwargs = kwargs.pop("model_kwargs", {})
        # If model_kwargs exists, we might need to pass it merged if supported or separately
        if base_model_kwargs:
            return ChatOpenAI(**kwargs, model_kwargs=base_model_kwargs, **_responses_mode_kwargs(profile))
        return ChatOpenAI(**kwargs, **_responses_mode_kwargs(profile))
    except TypeError:
        # 兼容旧版 langchain-openai：将 responses 配置下沉到 model_kwargs。
        responses_kwargs = _responses_mode_kwargs(profile)
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
default_llm = llm
orchestrator_llm = create_llm(ORCHESTRATOR_LLM_PROFILE)
