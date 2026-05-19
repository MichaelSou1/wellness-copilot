"""Long-poll WeChat iLink updates and pass them into the Health Guide graph."""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from contextlib import contextmanager

from langchain_core.messages import HumanMessage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from health_guide import config as cfg
from health_guide.detail import display_role, set_detail
from health_guide.graph import graph
from health_guide.integrations.wechat_ilink import (
    WeChatILinkError,
    get_client,
    get_last_offset,
    normalize_update,
    set_last_offset,
)
from health_guide.llm import extract_text_content
from health_guide.observability import ObservabilityTracker, TurnRecord


@contextmanager
def _worker_env(user_id: str, target_wxid: str, context_token: str):
    old = {
        "HEALTH_GUIDE_USER_ID": os.environ.get("HEALTH_GUIDE_USER_ID"),
        "WECHAT_TARGET_WXID": os.environ.get("WECHAT_TARGET_WXID"),
        "WECHAT_CONTEXT_TOKEN": os.environ.get("WECHAT_CONTEXT_TOKEN"),
    }
    os.environ["HEALTH_GUIDE_USER_ID"] = user_id
    os.environ["WECHAT_TARGET_WXID"] = target_wxid
    os.environ["WECHAT_CONTEXT_TOKEN"] = context_token
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _next_offset(update_id: str) -> str:
    try:
        return str(int(update_id) + 1)
    except Exception:
        return str(update_id)


def _build_content(client, text: str, media_ids: list[str]):
    if not media_ids:
        return text
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for media_id in media_ids:
        try:
            image_bytes = client.download_media(media_id)
            encoded = base64.b64encode(image_bytes).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    "media_id": media_id,
                }
            )
        except Exception as exc:
            parts.append(
                {
                    "type": "text",
                    "text": f"[图片 media_id={media_id} 下载失败：{type(exc).__name__}]",
                }
            )
    return parts


def _process_update(client, update: dict, tracker: ObservabilityTracker, detail: bool = False) -> None:
    msg = normalize_update(update)
    if not msg.text and not msg.media_ids:
        set_last_offset(_next_offset(msg.update_id))
        return

    user_id = msg.user_wxid or os.environ.get("HEALTH_GUIDE_USER_ID", "wechat_user")
    thread_id = f"wechat:{user_id}"
    content = _build_content(client, msg.text, msg.media_ids)
    config = {"configurable": {"thread_id": thread_id}}
    start = time.perf_counter()

    final_answer = ""
    routes = []
    tools_used = []
    retrieval_hits = 0
    actuation_events = []
    vision_calls = 0

    with _worker_env(user_id, msg.user_wxid, msg.context_token):
        stream_iter = graph.stream(
            {
                "messages": [HumanMessage(content=content)],
                "profile_user_id": user_id,
                "wechat_context": {
                    "context_token": msg.context_token,
                    "chat_type": msg.chat_type,
                    "user_wxid": msg.user_wxid,
                },
            },
            config,
        )
        for event in stream_iter:
            for key, value in event.items():
                value = value or {}
                if detail:
                    print(f"[node] {key}")
                if value.get("messages"):
                    final_answer = extract_text_content(value["messages"][-1])
                if value.get("last_tools"):
                    tools_used.extend(t for t in value["last_tools"] if t != "__RESET__")
                if value.get("retrieval_hits") is not None:
                    retrieval_hits += int(value.get("retrieval_hits") or 0)
                if value.get("executed"):
                    routes = value["executed"]
                if value.get("actuation_log"):
                    actuation_events.extend(value["actuation_log"])
                if value.get("vision_extractions"):
                    vision_calls += 1

    if not final_answer:
        final_answer = "抱歉，我这轮没有生成有效回复，请再发一次。"

    if msg.context_token:
        client.send_message(msg.context_token, text=final_answer)
    elif msg.user_wxid:
        client.push_to_user(msg.user_wxid, final_answer)
    else:
        print(f"[wechat_worker] no reply target; answer={final_answer[:120]}")

    latency_ms = (time.perf_counter() - start) * 1000
    tracker.log_turn(
        TurnRecord(
            thread_id=thread_id,
            turn_index=int(time.time()),
            route=",".join(r for r in routes if r != "FINISH") or "FINISH",
            user_query=msg.text or "[image]",
            final_answer=final_answer,
            tools_used=tools_used,
            retrieval_hits=retrieval_hits,
            citations_count=final_answer.count("[source:"),
            latency_ms=latency_ms,
            actuation_count=len(actuation_events),
            vision_calls=vision_calls,
            wechat_msgs_in=1,
            wechat_msgs_out=1,
        )
    )
    if detail:
        print(
            f"[wechat_worker] replied to {display_role(user_id)} "
            f"tools={tools_used} actuation={len(actuation_events)}"
        )
    set_last_offset(_next_offset(msg.update_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WeChat iLink long-poll worker")
    parser.add_argument("--once", action="store_true", help="Process one polling batch and exit")
    parser.add_argument("--detail", action="store_true", help="Print node/tool details")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_detail(args.detail)
    tracker = ObservabilityTracker()
    client = get_client()
    if not cfg.WECHAT_BOT_TOKEN:
        print("WECHAT_BOT_TOKEN is not configured. Run scripts/wechat_login.py first.")
        if args.once:
            return

    backoff = 1.0
    while True:
        try:
            if not cfg.WECHAT_BOT_TOKEN:
                time.sleep(min(30, backoff))
                backoff = min(60, backoff * 2)
                if args.once:
                    return
                continue
            updates = client.get_updates(timeout=cfg.WECHAT_POLL_TIMEOUT_SEC, offset=get_last_offset())
            backoff = 1.0
            for update in updates:
                _process_update(client, update, tracker, detail=args.detail)
            if args.once:
                return
            time.sleep(cfg.WECHAT_WORKER_IDLE_SEC)
        except (WeChatILinkError, OSError) as exc:
            print(f"[wechat_worker] transient error: {type(exc).__name__}: {exc}")
            time.sleep(backoff)
            backoff = min(60, backoff * 2)
            if args.once:
                return
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
