"""Vision helpers used by the orchestrator's multimodal processor.

The public functions here are provider-agnostic wrappers around an
OpenAI-compatible VLM endpoint. They return low-confidence fallback payloads
instead of raising so text-only and offline flows keep working.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.request
from typing import Any

from .. import config


_MEAL_SYSTEM = """\
你是健康饮食图片识别助手。请从图片估算餐盘中的食物和宏量营养素。

只输出 JSON，不要 markdown。字段：
{
  "items": [{"name": "...", "estimated_amount": "..."}],
  "kcal": 0,
  "protein_g": 0,
  "carbs_g": 0,
  "fat_g": 0,
  "confidence": 0.0,
  "notes": "一句话说明不确定性"
}

要求：
- 数字为估算值，未知则用 0
- confidence 范围 0-1
- 若图片不像餐食，items 为空且 confidence <= 0.2
"""

_GENERIC_SYSTEM = """\
你是 Wellness Copilot 的多模态图片理解节点。你的任务是把图片转成后续健康咨询可用的文本 grounding。

只输出 JSON，不要 markdown。字段：
{
  "description": "面向用户问题的图片描述，2-5 句，具体但不夸大",
  "query_focus": "图片中与用户问题最相关的细节",
  "visible_elements": ["可见元素1", "可见元素2"],
  "health_relevance": "可能影响饮食、训练、恢复、心理或医学建议的观察；不能诊断",
  "uncertainty": "哪些内容看不清、无法从图片确认或需要用户补充",
  "content_type": "meal|body|exercise|medical_document|product|environment|other",
  "confidence": 0.0,
  "meal_estimate": {
    "items": [{"name": "...", "estimated_amount": "..."}],
    "kcal": 0,
    "protein_g": 0,
    "carbs_g": 0,
    "fat_g": 0,
    "confidence": 0.0,
    "notes": "若不是餐食，items 为空"
  }
}

要求：
- 围绕用户问题放大注意力；与问题无关的背景简略带过
- 不要识别或猜测真实身份、敏感属性；不要做疾病诊断或处方判断
- 食物、体态、伤口、化验单、包装标签等都可以描述，但必须标注不确定性
- 数字都是估算；未知则用 0；confidence 范围 0-1
- 若不是餐食，meal_estimate.items 为空且 meal_estimate.confidence <= 0.2
"""


def _empty_result(reason: str, confidence: float = 0.0) -> dict:
    return {
        "items": [],
        "kcal": 0,
        "protein_g": 0,
        "carbs_g": 0,
        "fat_g": 0,
        "confidence": confidence,
        "notes": reason,
        "provider": config.MULTIMODAL_LLM_PROVIDER or "disabled",
    }


def _empty_multimodal_result(reason: str, confidence: float = 0.0) -> dict:
    return {
        "description": "",
        "query_focus": "",
        "visible_elements": [],
        "health_relevance": "",
        "uncertainty": reason,
        "content_type": "unknown",
        "confidence": confidence,
        "provider": config.MULTIMODAL_LLM_PROVIDER or "disabled",
    }


def _coerce_image_url(image_bytes_or_url: bytes | str) -> str:
    if isinstance(image_bytes_or_url, bytes):
        encoded = base64.b64encode(image_bytes_or_url).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    value = str(image_bytes_or_url or "").strip()
    if value.startswith("data:image/"):
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if re.fullmatch(r"[A-Za-z0-9+/=\s]+", value) and len(value) > 100:
        return f"data:image/jpeg;base64,{''.join(value.split())}"
    return value


def _extract_json(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_meal(parsed: dict) -> dict:
    result = _empty_result("", confidence=0.0)
    result.update({k: parsed.get(k, result[k]) for k in result if k in parsed})
    items = parsed.get("items")
    if isinstance(items, list):
        result["items"] = items[:10]
    for key in ("kcal", "protein_g", "carbs_g", "fat_g"):
        try:
            result[key] = int(round(float(parsed.get(key) or 0)))
        except Exception:
            result[key] = 0
    try:
        result["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence") or 0)))
    except Exception:
        result["confidence"] = 0.0
    notes = parsed.get("notes")
    result["notes"] = str(notes).strip() if notes else "图片营养素为估算值。"
    result["provider"] = config.MULTIMODAL_LLM_PROVIDER
    return result


def _text_field(parsed: dict, key: str) -> str:
    value = parsed.get(key)
    return str(value).strip() if value is not None else ""


def _normalize_multimodal(parsed: dict) -> dict:
    result = _empty_multimodal_result("", confidence=0.0)
    result["description"] = _text_field(parsed, "description")
    result["query_focus"] = _text_field(parsed, "query_focus")
    result["health_relevance"] = _text_field(parsed, "health_relevance")
    result["uncertainty"] = _text_field(parsed, "uncertainty")
    result["content_type"] = _text_field(parsed, "content_type") or "other"
    elements = parsed.get("visible_elements")
    if isinstance(elements, list):
        result["visible_elements"] = [str(item).strip() for item in elements[:12] if str(item).strip()]
    try:
        result["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence") or 0)))
    except Exception:
        result["confidence"] = 0.0

    meal_raw = parsed.get("meal_estimate") or parsed.get("meal")
    if isinstance(meal_raw, dict):
        meal = _normalize_meal(meal_raw)
        if meal.get("items") or any(int(meal.get(k) or 0) > 0 for k in ("kcal", "protein_g", "carbs_g", "fat_g")):
            result["meal"] = meal
    result["provider"] = config.MULTIMODAL_LLM_PROVIDER
    if not result["description"]:
        result["description"] = "图片内容无法被可靠描述。"
    if not result["uncertainty"]:
        result["uncertainty"] = "图片理解为模型估计，无法确认不可见的成分、重量、病因或身份信息。"
    return result


def _extract_responses_text(response: dict) -> str:
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                return str(part.get("text") or "")
        text = item.get("text")
        if isinstance(text, str) and text:
            return text
    choices = response.get("choices") or []
    if choices:
        return str((choices[0].get("message") or {}).get("content") or "")
    return ""


def _responses_api_meal(image_url: str) -> dict:
    if not config.MULTIMODAL_LLM_BASE_URL or not config.MULTIMODAL_LLM_API_KEY or not config.MULTIMODAL_LLM_MODEL:
        return _empty_result("multimodal_processor 未配置 MULTIMODAL_LLM_BASE_URL / MULTIMODAL_LLM_API_KEY / MULTIMODAL_LLM_MODEL。")
    if not image_url.startswith(("http://", "https://", "data:image/")):
        return _empty_result("图片来源不是可访问 URL 或 data URL，已跳过视觉识别。")

    payload = {
        "model": config.MULTIMODAL_LLM_MODEL,
        "input": [
            {"role": "system", "content": _MEAL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "请识别这张餐盘照并估算宏量营养素。"},
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{config.MULTIMODAL_LLM_BASE_URL.rstrip('/')}/responses"
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {config.MULTIMODAL_LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.MULTIMODAL_LLM_TIMEOUT_SEC) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    content = _extract_responses_text(response)
    parsed = _extract_json(content)
    if not parsed:
        return _empty_result("Vision 返回无法解析，已降级为不确定。", confidence=0.1)
    return _normalize_meal(parsed)


def _responses_api_image_description(image_url: str, user_query: str = "") -> dict:
    if not config.MULTIMODAL_LLM_BASE_URL or not config.MULTIMODAL_LLM_API_KEY or not config.MULTIMODAL_LLM_MODEL:
        return _empty_multimodal_result("multimodal_processor 未配置 MULTIMODAL_LLM_BASE_URL / MULTIMODAL_LLM_API_KEY / MULTIMODAL_LLM_MODEL。")
    if not image_url.startswith(("http://", "https://", "data:image/")):
        return _empty_multimodal_result("图片来源不是可访问 URL 或 data URL，已跳过视觉识别。")

    query = (user_query or "").strip() or "用户只发送了图片，请描述其中与健康咨询可能相关的信息。"
    payload = {
        "model": config.MULTIMODAL_LLM_MODEL,
        "input": [
            {"role": "system", "content": _GENERIC_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "用户问题：\n"
                            f"{query}\n\n"
                            "请把图片转成后续健康咨询可用的文字 grounding。"
                        ),
                    },
                    {"type": "input_image", "image_url": image_url},
                ],
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{config.MULTIMODAL_LLM_BASE_URL.rstrip('/')}/responses"
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {config.MULTIMODAL_LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.MULTIMODAL_LLM_TIMEOUT_SEC) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    content = _extract_responses_text(response)
    parsed = _extract_json(content)
    if not parsed:
        result = _empty_multimodal_result("Vision 返回不是 JSON，已保留原始文本作为低置信度描述。", confidence=0.3)
        result["description"] = str(content or "").strip()
        return result
    return _normalize_multimodal(parsed)


def _openai_compatible_meal(image_url: str) -> dict:
    if not config.MULTIMODAL_LLM_BASE_URL or not config.MULTIMODAL_LLM_API_KEY or not config.MULTIMODAL_LLM_MODEL:
        return _empty_result("multimodal_processor 未配置 MULTIMODAL_LLM_BASE_URL / MULTIMODAL_LLM_API_KEY / MULTIMODAL_LLM_MODEL。")
    if not image_url.startswith(("http://", "https://", "data:image/")):
        return _empty_result("图片来源不是可访问 URL 或 data URL，已跳过视觉识别。")

    payload = {
        "model": config.MULTIMODAL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": _MEAL_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请识别这张餐盘照并估算宏量营养素。"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{config.MULTIMODAL_LLM_BASE_URL.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {config.MULTIMODAL_LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.MULTIMODAL_LLM_TIMEOUT_SEC) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    content = response["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        return _empty_result("Vision 返回无法解析，已降级为不确定。", confidence=0.1)
    return _normalize_meal(parsed)


def _openai_compatible_image_description(image_url: str, user_query: str = "") -> dict:
    if not config.MULTIMODAL_LLM_BASE_URL or not config.MULTIMODAL_LLM_API_KEY or not config.MULTIMODAL_LLM_MODEL:
        return _empty_multimodal_result("multimodal_processor 未配置 MULTIMODAL_LLM_BASE_URL / MULTIMODAL_LLM_API_KEY / MULTIMODAL_LLM_MODEL。")
    if not image_url.startswith(("http://", "https://", "data:image/")):
        return _empty_multimodal_result("图片来源不是可访问 URL 或 data URL，已跳过视觉识别。")

    query = (user_query or "").strip() or "用户只发送了图片，请描述其中与健康咨询可能相关的信息。"
    payload = {
        "model": config.MULTIMODAL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": _GENERIC_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "用户问题：\n"
                            f"{query}\n\n"
                            "请把图片转成后续健康咨询可用的文字 grounding。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        "temperature": 0,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{config.MULTIMODAL_LLM_BASE_URL.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {config.MULTIMODAL_LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.MULTIMODAL_LLM_TIMEOUT_SEC) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    content = response["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        result = _empty_multimodal_result("Vision 返回不是 JSON，已保留原始文本作为低置信度描述。", confidence=0.3)
        result["description"] = str(content or "").strip()
        return result
    return _normalize_multimodal(parsed)


def analyze_meal_image(image_bytes_or_url: bytes | str) -> dict:
    """Return structured meal estimates from an image.

    Providers currently use an OpenAI-compatible /chat/completions request for
    portability across OpenAI, Tongyi, Zhipu, and compatible gateways. When no
    provider is configured, the function returns a low-confidence result rather
    than raising, so text-only and offline flows keep working.
    """
    if not config.MULTIMODAL_LLM_ENABLED:
        return _empty_result("multimodal_processor 已通过 MULTIMODAL_LLM_ENABLED=false 关闭。")
    provider = (config.MULTIMODAL_LLM_PROVIDER or "disabled").lower()
    if provider in {"", "disabled", "none", "mock"}:
        return _empty_result("multimodal_processor provider 未启用；请配置 MULTIMODAL_LLM_PROVIDER 和模型凭证。")

    image_url = _coerce_image_url(image_bytes_or_url)
    try:
        if config.MULTIMODAL_LLM_API_MODE == "responses":
            return _responses_api_meal(image_url)
        return _openai_compatible_meal(image_url)
    except Exception as exc:
        return _empty_result(f"Vision 调用失败：{type(exc).__name__}", confidence=0.1)


def analyze_image_for_query(image_bytes_or_url: bytes | str, user_query: str = "") -> dict:
    """Return a query-focused text grounding for any image type.

    This is the generic VLM entry point used by the orchestrator tool. It can
    describe meals, exercise posture, body-region photos, product labels,
    documents, environments, and other visual inputs. Meal estimates are kept
    as an optional nested field for backward compatibility with nutrition and
    critic logic.
    """
    if not config.MULTIMODAL_LLM_ENABLED:
        return _empty_multimodal_result("multimodal_processor 已通过 MULTIMODAL_LLM_ENABLED=false 关闭。")
    provider = (config.MULTIMODAL_LLM_PROVIDER or "disabled").lower()
    if provider in {"", "disabled", "none", "mock"}:
        return _empty_multimodal_result("multimodal_processor provider 未启用；请配置 MULTIMODAL_LLM_PROVIDER 和模型凭证。")

    image_url = _coerce_image_url(image_bytes_or_url)
    try:
        if config.MULTIMODAL_LLM_API_MODE == "responses":
            return _responses_api_image_description(image_url, user_query=user_query)
        return _openai_compatible_image_description(image_url, user_query=user_query)
    except Exception as exc:
        return _empty_multimodal_result(f"Vision 调用失败：{type(exc).__name__}", confidence=0.1)


def analyze_form_image(image_bytes: bytes, exercise_hint: str = "") -> dict[str, Any]:
    """Compatibility wrapper for exercise-form analysis."""
    return {
        "exercise_hint": exercise_hint,
        "confidence": 0.0,
        "notes": "请优先使用 analyze_image_for_query 获取面向用户问题的通用图片描述；本函数仅保留兼容占位。",
    }
