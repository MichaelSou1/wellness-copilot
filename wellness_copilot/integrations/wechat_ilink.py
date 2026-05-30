"""Synchronous client for the WeChatBot / iLink Bot HTTP APIs.

The protocol is implemented from the public WeChatBot SDK:
https://www.wechatbot.dev/zh/protocol
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7

from .. import config
from .local_logs import get_kv, set_kv

MESSAGE_TYPE_USER = 1
MESSAGE_TYPE_BOT = 2
MESSAGE_STATE_FINISH = 2
ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5
_HEX_32 = re.compile(r"^[0-9a-fA-F]{32}$")
_BOT_TOKEN_KEY = "wechat_ilink:bot_token"
_BOT_BASE_URL_KEY = "wechat_ilink:base_url"
_BOT_ACCOUNT_ID_KEY = "wechat_ilink:account_id"
_BOT_LOGIN_USER_ID_KEY = "wechat_ilink:login_user_id"
_BOT_UPDATED_AT_KEY = "wechat_ilink:login_updated_at"


class WeChatILinkError(RuntimeError):
    pass


def get_runtime_bot_token() -> str:
    return os.environ.get("WECHAT_BOT_TOKEN") or get_kv(_BOT_TOKEN_KEY, "")


def get_runtime_base_url() -> str:
    return (
        os.environ.get("WECHAT_ILINK_BASE_URL")
        or get_kv(_BOT_BASE_URL_KEY, "")
        or config.WECHAT_ILINK_BASE_URL
    )


def save_runtime_login(
    *,
    bot_token: str,
    base_url: str = "",
    account_id: str = "",
    login_user_id: str = "",
) -> dict[str, Any]:
    token = str(bot_token or "").strip()
    if not token:
        raise ValueError("bot_token must not be empty")
    set_kv(_BOT_TOKEN_KEY, token)
    if base_url:
        set_kv(_BOT_BASE_URL_KEY, str(base_url).strip())
    if account_id:
        set_kv(_BOT_ACCOUNT_ID_KEY, str(account_id).strip())
    if login_user_id:
        set_kv(_BOT_LOGIN_USER_ID_KEY, str(login_user_id).strip())
    set_kv(_BOT_UPDATED_AT_KEY, str(int(time.time())))
    return runtime_login_status()


def runtime_login_status() -> dict[str, Any]:
    env_token = os.environ.get("WECHAT_BOT_TOKEN", "")
    stored_token = get_kv(_BOT_TOKEN_KEY, "")
    base_url = get_runtime_base_url()
    return {
        "configured": bool(env_token or stored_token),
        "source": "env" if env_token else ("runtime" if stored_token else ""),
        "base_url": base_url,
        "account_id": os.environ.get("WECHAT_ACCOUNT_ID") or get_kv(_BOT_ACCOUNT_ID_KEY, ""),
        "login_user_id": os.environ.get("WECHAT_LOGIN_USER_ID") or get_kv(_BOT_LOGIN_USER_ID_KEY, ""),
        "updated_at": get_kv(_BOT_UPDATED_AT_KEY, ""),
    }


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
        self.bot_token = bot_token if bot_token is not None else get_runtime_bot_token()
        self.base_url = (base_url or get_runtime_base_url()).rstrip("/")
        self.last_updates_cursor = ""

    def _url(
        self,
        path: str,
        *,
        base_url: str | None = None,
        query: dict[str, Any] | None = None,
    ) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            root = (base_url or self.base_url).rstrip("/")
            url = f"{root}/{path.lstrip('/')}"
        if query:
            encoded = urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{encoded}"
        return url

    def _headers(self, auth: bool = True) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "iLink-App-Id": config.WECHAT_ILINK_APP_ID,
            "iLink-App-ClientVersion": _build_client_version(),
        }
        if auth:
            if not self.bot_token:
                raise WeChatILinkError("WECHAT_BOT_TOKEN is not configured")
            headers["AuthorizationType"] = "ilink_bot_token"
            headers["Authorization"] = f"Bearer {self.bot_token}"
            headers["X-WECHAT-UIN"] = _random_wechat_uin()
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        auth: bool = True,
        query: dict[str, Any] | None = None,
        base_url: str | None = None,
        timeout: int | float = 30,
        raw: bool = False,
    ):
        method = method.upper()
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._headers(auth=auth)
        if method == "GET":
            headers.pop("Content-Type", None)
        req = urllib.request.Request(
            url=self._url(path, base_url=base_url, query=query),
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if raw:
                    return body
                if not body:
                    return {}
                result = json.loads(body.decode("utf-8"))
                _raise_for_api_error(result, path)
                return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            msg = _api_error_message(detail) or detail[:300]
            raise WeChatILinkError(f"HTTP {exc.code}: {msg}") from exc
        except urllib.error.URLError as exc:
            raise WeChatILinkError(str(exc.reason)) from exc
        except TimeoutError as exc:
            raise WeChatILinkError("request timed out") from exc

    def get_bot_qrcode(self) -> dict:
        return self._request(
            "GET",
            config.WECHAT_ENDPOINT_QRCODE,
            auth=False,
            base_url=config.WECHAT_QR_BASE_URL,
            query={"bot_type": config.WECHAT_BOT_TYPE},
        )

    def poll_qrcode_status(
        self,
        qrcode_id: str,
        timeout: int = 45,
        *,
        base_url: str | None = None,
    ) -> dict:
        return self._request(
            "GET",
            config.WECHAT_ENDPOINT_QRCODE_STATUS,
            auth=False,
            base_url=base_url or config.WECHAT_QR_BASE_URL,
            query={"qrcode": qrcode_id},
            timeout=timeout,
        )

    def get_updates(self, timeout: int | None = None, offset: str | int | None = None) -> list[dict]:
        timeout = int(timeout or config.WECHAT_POLL_TIMEOUT_SEC)
        payload = {"get_updates_buf": str(offset or ""), "base_info": _base_info()}
        result = self._request(
            "POST",
            config.WECHAT_ENDPOINT_UPDATES,
            payload,
            timeout=timeout + 15,
        )
        self.last_updates_cursor = _first(
            result.get("get_updates_buf") if isinstance(result, dict) else "",
            result.get("cursor") if isinstance(result, dict) else "",
            result.get("next_cursor") if isinstance(result, dict) else "",
            result.get("offset") if isinstance(result, dict) else "",
        )
        updates: list[dict] = []
        if isinstance(result, list):
            updates = [item for item in result if isinstance(item, dict)]
        for key in ("msgs", "updates", "messages", "data", "items"):
            value = result.get(key) if isinstance(result, dict) else None
            if isinstance(value, list):
                updates = [item for item in value if isinstance(item, dict)]
                break
            if isinstance(value, dict):
                for nested_key in ("msgs", "updates", "messages", "items"):
                    nested = value.get(nested_key)
                    if isinstance(nested, list):
                        updates = [item for item in nested if isinstance(item, dict)]
                        break
        if self.last_updates_cursor:
            for item in updates:
                item.setdefault("_wechat_get_updates_buf", self.last_updates_cursor)
        return updates

    def remember_context(self, user_id: str, context_token: str) -> None:
        if user_id and context_token:
            set_kv(f"wechat_ilink:context:{user_id}", context_token)

    def send_message(
        self,
        context_token: str,
        text: str = "",
        *,
        user_id: str = "",
        image: str = "",
        voice: str = "",
        file: str = "",
    ) -> dict:
        if text:
            if not context_token:
                raise ValueError("send_message requires context_token")
            user_id = user_id or os.environ.get("WECHAT_TARGET_WXID", "")
            if not user_id:
                raise WeChatILinkError("send_message requires user_id for the WeChatBot protocol")
            self.remember_context(user_id, context_token)
            result: dict[str, Any] = {}
            for chunk in _chunk_text(text, 4000):
                msg = _build_text_message(user_id, context_token, chunk)
                result = self._request(
                    "POST",
                    config.WECHAT_ENDPOINT_SEND,
                    {"msg": msg, "base_info": _base_info()},
                )
            return result
        elif image:
            raise NotImplementedError("WeChat image replies need CDN upload support")
        elif voice:
            raise NotImplementedError("WeChat voice replies need CDN upload support")
        elif file:
            raise NotImplementedError("WeChat file replies need CDN upload support")
        else:
            raise ValueError("send_message requires text, image, voice, or file")

    def push_to_user(self, wxid: str, text: str, *, context_token: str = "") -> dict:
        context_token = (
            context_token
            or (os.environ.get("WECHAT_CONTEXT_TOKEN", "") if wxid == os.environ.get("WECHAT_TARGET_WXID", "") else "")
            or get_kv(f"wechat_ilink:context:{wxid}", "")
        )
        if not context_token:
            raise WeChatILinkError(
                f"No context_token for {wxid}; send/receive a message with this user before pushing"
            )
        return self.send_message(context_token, text=text, user_id=wxid)

    def download_media(self, media_id: str) -> bytes:
        ref = _decode_media_ref(media_id)
        if not ref:
            if not config.WECHAT_ENDPOINT_MEDIA:
                raise WeChatILinkError("Unsupported media reference")
            path = config.WECHAT_ENDPOINT_MEDIA.format(media_id=urllib.parse.quote(str(media_id), safe=""))
            return self._request("GET", path, auth=True, timeout=60, raw=True)

        media = ref.get("media") if isinstance(ref.get("media"), dict) else {}
        fallback_url = _first(ref.get("url"), media.get("full_url"))
        encrypted_query_param = _first(
            media.get("encrypt_query_param"),
            media.get("encrypted_query_param"),
            media.get("encryptQueryParam"),
        )
        if not encrypted_query_param:
            if fallback_url:
                return _download_url(fallback_url)
            raise WeChatILinkError("Media reference did not contain encrypted_query_param")

        aes_key = _first(ref.get("aeskey"), ref.get("aes_key"), media.get("aes_key"), media.get("aeskey"))
        if not aes_key:
            raise WeChatILinkError("Media reference did not contain aes_key")

        url = (
            f"{config.WECHAT_CDN_BASE_URL.rstrip('/')}/download"
            f"?encrypted_query_param={urllib.parse.quote(encrypted_query_param, safe='')}"
        )
        ciphertext = _download_url(url)
        return _decrypt_aes_ecb(ciphertext, _decode_aes_key(aes_key))


def _first(*values) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _build_client_version() -> str:
    parts = str(config.WECHAT_CHANNEL_VERSION or "0.1.0").split(".")
    try:
        major = int(parts[0]) & 0xFF if len(parts) > 0 else 0
        minor = int(parts[1]) & 0xFF if len(parts) > 1 else 0
        patch = int(parts[2]) & 0xFF if len(parts) > 2 else 0
    except Exception:
        major = minor = patch = 0
    return str((major << 16) | (minor << 8) | patch)


def _random_wechat_uin() -> str:
    val = struct.unpack(">I", os.urandom(4))[0]
    return base64.b64encode(str(val).encode("utf-8")).decode("ascii")


def _base_info() -> dict[str, str]:
    return {"channel_version": config.WECHAT_CHANNEL_VERSION}


def _raise_for_api_error(payload: Any, label: str) -> None:
    if not isinstance(payload, dict):
        return
    ret = payload.get("ret")
    errcode = payload.get("errcode")
    failed_ret = isinstance(ret, int) and ret != 0
    failed_err = isinstance(errcode, int) and errcode != 0
    if failed_ret or failed_err:
        code = errcode if failed_err else ret
        msg = payload.get("errmsg") or f"{label} failed (ret={ret} errcode={errcode})"
        raise WeChatILinkError(f"{msg} [code={code}]")


def _api_error_message(text: str) -> str:
    try:
        payload = json.loads(text)
    except Exception:
        return ""
    if isinstance(payload, dict):
        return str(payload.get("errmsg") or payload.get("error") or "")
    return ""


def _chunk_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        window = text[:limit]
        cut = -1
        for marker in ("\n\n", "\n", " "):
            idx = window.rfind(marker)
            if idx > limit * 3 // 10:
                cut = idx + len(marker)
                break
        if cut == -1:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:]
    return chunks or [""]


def _build_text_message(user_id: str, context_token: str, text: str) -> dict[str, Any]:
    return {
        "from_user_id": "",
        "to_user_id": user_id,
        "client_id": str(uuid4()),
        "message_type": MESSAGE_TYPE_BOT,
        "message_state": MESSAGE_STATE_FINISH,
        "context_token": context_token,
        "item_list": [{"type": ITEM_TEXT, "text_item": {"text": text}}],
    }


def _extract_item_text(items: list[dict]) -> str:
    parts: list[str] = []
    for item in items:
        item_type = _as_int(item.get("type"))
        if item_type == ITEM_TEXT:
            parts.append(str(item.get("text_item", {}).get("text", "")).strip())
        elif item_type == ITEM_VOICE:
            parts.append(str(item.get("voice_item", {}).get("text", "")).strip())
        elif item_type == ITEM_FILE:
            parts.append(str(item.get("file_item", {}).get("file_name", "[file]")).strip())
        elif item_type == ITEM_VIDEO:
            parts.append("[video]")
    return "\n".join(part for part in parts if part)


def _collect_media_ids(raw: dict) -> list[str]:
    media_ids: list[str] = []
    items = raw.get("item_list")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict) or _as_int(item.get("type")) != ITEM_IMAGE:
                continue
            image = item.get("image_item")
            if not isinstance(image, dict):
                continue
            media = image.get("media") if isinstance(image.get("media"), dict) else None
            thumb_media = image.get("thumb_media") if isinstance(image.get("thumb_media"), dict) else None
            ref = {
                "type": "image",
                "media": media or thumb_media or {},
                "aeskey": image.get("aeskey") or image.get("aes_key"),
                "url": image.get("url"),
            }
            if ref["media"] or ref["url"]:
                media_ids.append(json.dumps(ref, ensure_ascii=False, sort_keys=True))

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


def _decode_media_ref(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _download_url(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise WeChatILinkError(f"CDN HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise WeChatILinkError(str(exc.reason)) from exc


def _decode_aes_key(encoded: str) -> bytes:
    if _HEX_32.match(encoded):
        return binascii.unhexlify(encoded)
    try:
        decoded = base64.b64decode(encoded)
    except Exception as exc:
        raise WeChatILinkError(f"Cannot base64 decode aes_key: {exc}") from exc
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        try:
            hex_text = decoded.decode("ascii")
            if _HEX_32.match(hex_text):
                return binascii.unhexlify(hex_text)
        except Exception:
            pass
    raise WeChatILinkError(f"Decoded aes_key has unexpected length {len(decoded)}")


def _decrypt_aes_ecb(ciphertext: bytes, key: bytes) -> bytes:
    if len(key) != 16:
        raise WeChatILinkError(f"AES key must be 16 bytes, got {len(key)}")
    if len(ciphertext) % 16 != 0:
        raise WeChatILinkError(f"Ciphertext length {len(ciphertext)} is not a multiple of 16")
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    unpadder = PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


def normalize_update(update: dict) -> NormalizedUpdate:
    msg = update.get("message") if isinstance(update.get("message"), dict) else update
    sender = msg.get("sender") if isinstance(msg.get("sender"), dict) else {}
    chat = msg.get("chat") if isinstance(msg.get("chat"), dict) else {}
    items = msg.get("item_list") if isinstance(msg.get("item_list"), list) else []
    update_id = _first(
        update.get("update_id"),
        update.get("id"),
        msg.get("message_id"),
        msg.get("msgid"),
        msg.get("client_id"),
        msg.get("create_time_ms"),
        time.time(),
    )
    context_token = _first(update.get("context_token"), msg.get("context_token"), chat.get("context_token"))
    user_wxid = _first(
        update.get("user_wxid"),
        update.get("from_wxid"),
        msg.get("from_user_id"),
        msg.get("from_wxid"),
        sender.get("wxid"),
        sender.get("id"),
        chat.get("user_wxid"),
    )
    chat_type = _first(update.get("chat_type"), msg.get("chat_type"), chat.get("type"), "private")
    text = _first(_extract_item_text(items), msg.get("text"), msg.get("content"), update.get("text"), update.get("content"))
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
    token = get_runtime_bot_token()
    base_url = get_runtime_base_url().rstrip("/")
    if _CLIENT is None or _CLIENT.bot_token != token or _CLIENT.base_url != base_url:
        _CLIENT = WeChatILinkClient()
    return _CLIENT


def get_last_offset() -> str:
    return get_kv("wechat_ilink:last_offset", "")


def set_last_offset(offset: str) -> None:
    set_kv("wechat_ilink:last_offset", str(offset))
