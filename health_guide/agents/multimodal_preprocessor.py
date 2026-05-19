"""Turn-scoped multimodal grounding before specialist routing."""
from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import HumanMessage

from ..integrations.vision import analyze_meal_image
from ..llm import extract_text_content


def _latest_human_message(messages: list[Any]):
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            return msg
    return None


def _normalize_image_part(part: dict, index: int) -> dict | None:
    if not isinstance(part, dict):
        return None
    part_type = str(part.get("type") or "").lower()
    image_url = part.get("image_url")
    if part_type in {"image_url", "input_image"} or image_url:
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        url = url or part.get("url")
        if url:
            return {"index": index, "kind": "url", "url": url}
    media_id = part.get("media_id")
    if media_id:
        return {"index": index, "kind": "media_id", "media_id": media_id}
    raw_b64 = part.get("image_bytes_b64") or part.get("data")
    if raw_b64:
        return {
            "index": index,
            "kind": "bytes_b64",
            "image_bytes_b64": raw_b64,
            "mime_type": part.get("mime_type") or "image/jpeg",
        }
    return None


def _extract_image_inputs(message) -> list[dict]:
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return []
    images = []
    for i, part in enumerate(content):
        normalized = _normalize_image_part(part, i)
        if normalized:
            images.append(normalized)
    return images


def _payload_for_vision(image: dict):
    if image.get("kind") == "bytes_b64":
        try:
            return base64.b64decode(str(image.get("image_bytes_b64") or ""))
        except Exception:
            return ""
    return image.get("url") or image.get("media_id") or ""


def _looks_like_meal_turn(text: str) -> bool:
    if not text:
        return True
    meal_words = (
        "餐",
        "饭",
        "吃",
        "菜",
        "热量",
        "蛋白",
        "碳水",
        "脂肪",
        "宏量",
        "营养",
        "增肌",
        "减脂",
        "meal",
        "food",
        "kcal",
    )
    return any(word in text.lower() for word in meal_words)


def multimodal_preprocessor_node(state):
    """Extract image parts and ground likely meal photos into state.

    Pure text turns return an empty update and do not call any vision model.
    """
    latest = _latest_human_message(state.get("messages") or [])
    images = _extract_image_inputs(latest)
    if not images:
        return {}

    user_text = extract_text_content(latest).strip()
    update = {"image_inputs": images}
    if not _looks_like_meal_turn(user_text):
        return update

    first_image = images[0]
    meal = analyze_meal_image(_payload_for_vision(first_image))
    if meal:
        update["vision_extractions"] = {"meal": meal}
    return update
