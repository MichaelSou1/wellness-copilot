"""Thin HTTP client for WeChat iLink / ClawBot style bot APIs.

The public protocol is young and endpoint shapes may change. To keep the repo
deployable, every endpoint path is configurable through environment variables
and responses are normalized defensively.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .. import config
from .local_logs import get_kv, set_kv


class WeChatILinkError(RuntimeError):
    pass


@dataclass
class NormalizedUpdate:
    update_id: str
    context_token: str
    user_wxid: str
    chat_type: str
    text: str
    media_ids: list[str]
    raw: dict


class WeChatILinkClient:
    def __init__(self, bot_token: str | None = None, base_url: str | None = None):
        self.bot_token = bot_token if bot_token is not None else config.WECHAT_BOT_TOKEN
        self.base_url = (base_url or config.WECHAT_ILINK_BASE_URL).rstrip("/")

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _headers(self, auth: bool = True) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if config.WECHAT_APP_ID:
            headers["X-WeChat-AppId"] = config.WECHAT_APP_ID
        if auth:
            if not self.bot_token:
                raise WeChatILinkError("WECHAT_BOT_TOKEN is not configured")
            headers["Authorization"] = f"Bearer {self.bot_token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        auth: bool = True,
        timeout: int | float = 30,
        raw: bool = False,
    ):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=self._url(path),
            data=data,
            headers=self._headers(auth=auth),
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if raw:
                    return body
                if not body:
                    return {}
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise WeChatILinkError(f"HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise WeChatILinkError(str(exc.reason)) from exc

    def get_bot_qrcode(self) -> dict:
        payload = {
            "app_id": config.WECHAT_APP_ID,
            "app_secret": config.WECHAT_APP_SECRET,
        }
        return self._request("POST", config.WECHAT_ENDPOINT_QRCODE, payload, auth=False)

    def poll_qrcode_status(self, qrcode_id: str, timeout: int = 5) -> dict:
        payload = {"qrcode_id": qrcode_id}
        return self._request("POST", config.WECHAT_ENDPOINT_QRCODE_STATUS, payload, auth=False, timeout=timeout)

    def get_updates(self, timeout: int | None = None, offset: str | int | None = None) -> list[dict]:
        timeout = int(timeout or config.WECHAT_POLL_TIMEOUT_SEC)
        payload = {"timeout": timeout}
        if offset is not None and str(offset) != "":
            payload["offset"] = offset
        result = self._request(
            "POST",
            config.WECHAT_ENDPOINT_UPDATES,
            payload,
            timeout=timeout + 10,
        )
        if isinstance(result, list):
            return result
        for key in ("updates", "messages", "data", "items"):
            value = result.get(key) if isinstance(result, dict) else None
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for nested_key in ("updates", "messages", "items"):
                    nested = value.get(nested_key)
                    if isinstance(nested, list):
                        return nested
        return []

    def send_message(
        self,
        context_token: str,
        text: str = "",
        *,
        image: str = "",
        voice: str = "",
        file: str = "",
    ) -> dict:
        payload: dict[str, Any] = {"context_token": context_token}
        if text:
            payload.update({"msg_type": "text", "text": text})
        elif image:
            payload.update({"msg_type": "image", "image": image})
        elif voice:
            payload.update({"msg_type": "voice", "voice": voice})
        elif file:
            payload.update({"msg_type": "file", "file": file})
        else:
            raise ValueError("send_message requires text, image, voice, or file")
        return self._request("POST", config.WECHAT_ENDPOINT_SEND, payload)

    def push_to_user(self, wxid: str, text: str) -> dict:
        payload = {"wxid": wxid, "msg_type": "text", "text": text}
        return self._request("POST", config.WECHAT_ENDPOINT_PUSH, payload)

    def download_media(self, media_id: str) -> bytes:
        path = config.WECHAT_ENDPOINT_MEDIA.format(media_id=urllib.parse.quote(str(media_id), safe=""))
        return self._request("GET", path, auth=True, timeout=60, raw=True)


def _first(*values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _collect_media_ids(raw: dict) -> list[str]:
    media_ids: list[str] = []
    for key in ("media_id", "mediaId", "image_media_id", "imageMediaId"):
        value = raw.get(key)
        if value:
            media_ids.append(str(value))
    for key in ("images", "media", "attachments"):
        value = raw.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    media_id = _first(item.get("media_id"), item.get("mediaId"), item.get("id"))
                    if media_id:
                        media_ids.append(media_id)
                elif item:
                    media_ids.append(str(item))
    seen = set()
    unique = []
    for media_id in media_ids:
        if media_id not in seen:
            seen.add(media_id)
            unique.append(media_id)
    return unique


def normalize_update(update: dict) -> NormalizedUpdate:
    msg = update.get("message") if isinstance(update.get("message"), dict) else update
    sender = msg.get("sender") if isinstance(msg.get("sender"), dict) else {}
    chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
    update_id = _first(update.get("update_id"), update.get("id"), msg.get("message_id"), msg.get("msgid"), time.time())
    context_token = _first(update.get("context_token"), msg.get("context_token"), chat.get("context_token"))
    user_wxid = _first(
        update.get("user_wxid"),
        update.get("from_wxid"),
        msg.get("from_wxid"),
        sender.get("wxid"),
        sender.get("id"),
        chat.get("user_wxid"),
    )
    chat_type = _first(update.get("chat_type"), msg.get("chat_type"), chat.get("type"), "private")
    text = _first(msg.get("text"), msg.get("content"), update.get("text"), update.get("content"))
    return NormalizedUpdate(
        update_id=str(update_id),
        context_token=context_token,
        user_wxid=user_wxid,
        chat_type=chat_type,
        text=text,
        media_ids=_collect_media_ids(msg),
        raw=update,
    )


_CLIENT: WeChatILinkClient | None = None


def get_client() -> WeChatILinkClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = WeChatILinkClient()
    return _CLIENT


def get_last_offset() -> str:
    return get_kv("wechat_ilink:last_offset", "")


def set_last_offset(offset: str) -> None:
    set_kv("wechat_ilink:last_offset", str(offset))
