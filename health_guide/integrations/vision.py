"""Vision helpers used by MultiModalPreprocessor.

This module is intentionally not exposed as an LLM tool. The graph calls it
once before routing so the parent agent and child experts can see grounded
meal data without giving the model a way to repeatedly invoke Vision.
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


def _empty_result(reason: str, confidence: float = 0.0) -> dict:
    return {
        "items": [],
        "kcal": 0,
        "protein_g": 0,
        "carbs_g": 0,
        "fat_g": 0,
        "confidence": confidence,
        "notes": reason,
        "provider": config.VISION_PROVIDER or "disabled",
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
    result["provider"] = config.VISION_PROVIDER
    return result


def _openai_compatible_meal(image_url: str) -> dict:
    if not config.VISION_BASE_URL or not config.VISION_API_KEY or not config.VISION_MODEL:
        return _empty_result("Vision 未配置 VISION_BASE_URL / VISION_API_KEY / VISION_MODEL。")
    if not image_url.startswith(("http://", "https://", "data:image/")):
        return _empty_result("图片来源不是可访问 URL 或 data URL，已跳过视觉识别。")

    payload = {
        "model": config.VISION_MODEL,
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
    url = f"{config.VISION_BASE_URL.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "Authorization": f"Bearer {config.VISION_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.VISION_TIMEOUT_SEC) as resp:
        response = json.loads(resp.read().decode("utf-8"))
    content = response["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        return _empty_result("Vision 返回无法解析，已降级为不确定。", confidence=0.1)
    return _normalize_meal(parsed)


def analyze_meal_image(image_bytes_or_url: bytes | str) -> dict:
    """Return structured meal estimates from an image.

    Providers currently use an OpenAI-compatible /chat/completions request for
    portability across OpenAI, Tongyi, Zhipu, and compatible gateways. When no
    provider is configured, the function returns a low-confidence result rather
    than raising, so text-only and offline flows keep working.
    """
    if not config.VISION_ENABLED:
        return _empty_result("Vision 已通过 VISION_ENABLED=false 关闭。")
    provider = (config.VISION_PROVIDER or "disabled").lower()
    if provider in {"", "disabled", "none", "mock"}:
        return _empty_result("Vision provider 未启用；请配置 VISION_PROVIDER 和模型凭证。")

    image_url = _coerce_image_url(image_bytes_or_url)
    try:
        return _openai_compatible_meal(image_url)
    except Exception as exc:
        return _empty_result(f"Vision 调用失败：{type(exc).__name__}", confidence=0.1)


def analyze_form_image(image_bytes: bytes, exercise_hint: str = "") -> dict[str, Any]:
    """Stretch placeholder for exercise-form analysis."""
    return {
        "exercise_hint": exercise_hint,
        "confidence": 0.0,
        "notes": "姿势识别尚未启用；当前 MVP 只处理餐食图片。",
    }
