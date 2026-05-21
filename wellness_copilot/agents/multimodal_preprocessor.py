"""Turn-scoped image input collection before orchestrator tool use.

This node deliberately does not call a VLM. The orchestrator receives the
available image handles in state and can call its `multimodal_processor` tool
when visual grounding is useful for the user's query.
"""
from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import HumanMessage


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


def multimodal_preprocessor_node(state):
    """Extract image parts into state.

    Pure text turns return an empty update. Image turns only expose normalized
    handles to the orchestrator; the VLM call is done lazily through a tool.
    """
    latest = _latest_human_message(state.get("messages") or [])
    images = _extract_image_inputs(latest)
    if not images:
        return {}
    return {"image_inputs": images}
