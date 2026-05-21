"""Accumulate fragmented WeChat inputs into one actionable user turn."""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage

from ..llm import extract_text_content


_QUESTION_OR_COMMAND = re.compile(
    r"[？?]|怎么|如何|能不能|能否|可以吗|可不可以|吗\b|呢\b|"
    r"帮我|给我|建议|推荐|安排|计划|记录|提醒|分析|看看|判断|测试|test|在吗|在不在|"
    r"多少|够不够|要不要|该不该|是不是|行不行|怎么办",
    re.IGNORECASE,
)
_TEXT_EXPECTS_IMAGE = re.compile(
    r"图|图片|照片|截图|这张|拍的|发的图|这幅|这页|这个动作|姿势|化验单|报告单|包装|标签|配料表|营养成分表",
    re.IGNORECASE,
)
_STANDALONE_CHAT = re.compile(
    r"^(你好|您好|hello|hi|嗨|测试|test|在吗|在不在|谢谢|感谢|多谢|再见|拜拜|早安|晚安|收到|好的|好)[。.!！?？~～]*$",
    re.IGNORECASE,
)


def _latest_human_message(messages: list[Any]):
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", None) == "human":
            return msg
    return None


def _image_part(part: dict) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = str(part.get("type") or "").lower()
    return bool(
        part.get("image_url")
        or part.get("image_bytes_b64")
        or part.get("media_id")
        or part_type in {"image_url", "input_image"}
    )


def _copy_image_part(part: dict) -> dict:
    copied = dict(part)
    image_url = copied.get("image_url")
    if isinstance(image_url, dict):
        copied["image_url"] = dict(image_url)
    return copied


def _fragment_from_message(message) -> dict:
    content = getattr(message, "content", None)
    text = extract_text_content(message).strip()
    images: list[dict] = []
    if isinstance(content, list):
        for part in content:
            if _image_part(part):
                images.append(_copy_image_part(part))
    return {"text": text, "images": images, "message_id": getattr(message, "id", None)}


def _is_empty(fragment: dict) -> bool:
    return not (fragment.get("text") or fragment.get("images"))


def _merge_fragments(fragments: list[dict]) -> tuple[str, list[dict]]:
    texts = [str(f.get("text") or "").strip() for f in fragments if str(f.get("text") or "").strip()]
    images = []
    for fragment in fragments:
        images.extend(fragment.get("images") or [])
    return "\n".join(texts).strip(), images


def _looks_complete(text: str, images: list[dict], is_wechat: bool) -> tuple[bool, str]:
    if not is_wechat:
        return True, "non_wechat"
    if not text:
        return False, "waiting_for_text"
    if not images and _STANDALONE_CHAT.search(text.strip()):
        return True, "standalone_chat"
    if _TEXT_EXPECTS_IMAGE.search(text) and not images:
        return False, "waiting_for_image"
    if _QUESTION_OR_COMMAND.search(text):
        return True, "question_or_command"
    if not images and len(text.strip()) <= 20:
        return True, "short_text"
    return False, "waiting_for_question_or_command"


def _is_standalone_chat_text(text: str) -> bool:
    return bool(_STANDALONE_CHAT.search((text or "").strip()))


def _fused_content(text: str, images: list[dict]):
    if not images:
        return text
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.extend(images)
    return parts


def _remove_fragment_messages(fragments: list[dict]) -> list[RemoveMessage]:
    removes = []
    seen = set()
    for fragment in fragments:
        message_id = fragment.get("message_id")
        if message_id and message_id not in seen:
            seen.add(message_id)
            removes.append(RemoveMessage(id=message_id))
    return removes


def input_accumulator_node(state):
    """Buffer fragmented WeChat messages until the turn is actionable.

    WeChat users often send an image and its question as separate messages, or
    split one thought into several short messages. This node persists those
    fragments in graph state. Once the cumulative text contains a clear
    question/command, it replaces the latest input with a fused HumanMessage and
    clears the buffer. Non-WeChat inputs pass through immediately.
    """
    latest = _latest_human_message(state.get("messages") or [])
    if latest is None:
        return {
            "input_accumulator_status": "READY",
            "input_accumulator_reason": "no_human_message",
        }

    wechat_context = state.get("wechat_context") or {}
    is_wechat = bool(wechat_context)
    fragment = _fragment_from_message(latest)
    if _is_empty(fragment):
        return {
            "input_accumulator_status": "WAITING" if is_wechat else "READY",
            "input_accumulator_reason": "empty_fragment",
        }

    if not is_wechat or wechat_context.get("pre_accumulated"):
        return {
            "input_accumulator_status": "READY",
            "input_accumulator_reason": "worker_pre_accumulated" if is_wechat else "non_wechat",
            "pending_input_fragments": [],
        }

    fragments = list(state.get("pending_input_fragments") or [])
    if fragments and _is_standalone_chat_text(fragment.get("text") or ""):
        fragments = []
    fragments.append(fragment)
    text, images = _merge_fragments(fragments)
    complete, reason = _looks_complete(text, images, is_wechat=True)
    if not complete:
        return {
            "pending_input_fragments": fragments,
            "input_accumulator_status": "WAITING",
            "input_accumulator_reason": reason,
        }

    return {
        "messages": _remove_fragment_messages(fragments) + [HumanMessage(content=_fused_content(text, images))],
        "pending_input_fragments": [],
        "input_accumulator_status": "READY",
        "input_accumulator_reason": reason,
    }
