"""解析 OneBot 消息、戳一戳通知和当前图片引用，并构造发出的消息段。"""

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(slots=True)
class MessageView:
    """消息简化视图：文本、被 @ 用户，以及当前图片引用。"""

    text: str = ""
    mentions: list[int] = field(default_factory=list)
    # 只保留当前事件的图片引用，不下载、不保存图片内容。
    images: list[dict[str, str]] = field(default_factory=list)
    reply_to: str = ""
    parts: list[dict] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PokeView:
    """已确认由他人戳向机器人的通知；group_id=0 表示私聊。"""

    user_id: int
    self_id: int
    group_id: int = 0


def parse_poke_event(event: dict[str, Any]) -> PokeView | None:
    """NapCat 的私聊 user_id 是会话对象，sender_id 才是实际发起者。"""
    if (event.get("post_type"), event.get("notice_type"), event.get("sub_type")) != ("notice", "notify", "poke"):
        return None

    def event_id(value: object) -> int:
        if type(value) is int:
            return value if 0 <= value <= 2**63 - 1 else -1
        if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 19:
            number = int(value)
            return number if number <= 2**63 - 1 else -1
        return -1

    self_id = event_id(event.get("self_id"))
    target_id = event_id(event.get("target_id"))
    group_id = event_id(event.get("group_id", 0))
    actor = event.get("user_id") if group_id > 0 else event.get("sender_id", event.get("user_id"))
    user_id = event_id(actor)
    if self_id <= 0 or user_id <= 0 or group_id < 0 or target_id != self_id or user_id == self_id:
        return None
    return PokeView(user_id=user_id, self_id=self_id, group_id=group_id)


def parse_message(message: object, *, image_limit: int | None = 3) -> MessageView:
    """保留安全消息段的原顺序；附件地址只留给下载器，不进入文本视图。"""
    if isinstance(message, str):
        return MessageView(text=message.strip())
    if not isinstance(message, list):
        return MessageView()

    text_parts: list[str] = []
    mentions: list[int] = []
    images: list[dict[str, str]] = []
    parts: list[dict] = []
    text_budget = 4000
    reply_to = ""
    for segment in message:
        if not isinstance(segment, dict):
            continue
        kind = segment.get("type")
        data = segment.get("data") or {}
        if not isinstance(data, dict):
            continue
        if kind == "text":
            text = str(data.get("text", ""))[:text_budget]
            text_budget -= len(text)
            text_parts.append(text)
            parts.append({'type': 'text', 'text': text})
        elif kind == "at":
            qq = str(data.get("qq", ""))
            # 只提取具体用户的数字 QQ 号，忽略 @全体成员等非数字目标。
            if qq.isdigit():
                mentions.append(int(qq))
                text_parts.append(f"@{qq}")
                parts.append({'type': 'at', 'qq': int(qq)})
        elif kind == "image" and (image_limit is None or len(images) < image_limit):
            # 不将带签名的下载地址、文件路径或不可信 summary 暴露给模型。
            images.append({
                "url": data.get("url", "")[:8193] if isinstance(data.get("url"), str) else "",
                "file_size": str(data.get("file_size", ""))[:20],
            })
            parts.append({'type': 'image', 'index': len(images) - 1})
        elif kind == "reply":
            reply_to = normalize_message_id(data.get("id"))
        elif kind in {'record', 'video'}:
            label = 'audio' if kind == 'record' else 'video'
            parts.append({'type': label})
            text_parts.append(f'[{"语音" if label == "audio" else "视频"}，内容未解析]')
        elif kind == 'json':
            try:
                raw = data.get('data', '')
                card = json.loads(raw) if isinstance(raw, str) and len(raw) <= 65536 else {}
                meta = card.get('meta', {}) if isinstance(card, dict) else {}
                detail = next((v for v in meta.values() if isinstance(v, dict) and ('title' in v or 'desc' in v)), {}) if isinstance(meta, dict) else {}
                app = detail.get('source', '')
                platform = app.get('name', '') if isinstance(app, dict) else app
                part = {'type': 'card', 'platform': str(platform)[:60],
                        'title': str(detail.get('title', ''))[:200], 'description': str(detail.get('desc', ''))[:200]}
            except (ValueError, TypeError):
                part = {'type': 'card'}
            parts.append(part)
            text_parts.append('[分享卡片，正文未读取] ' + part.get('title', '') + ' ' + part.get('description', ''))
    return MessageView(text="".join(text_parts).strip(), mentions=mentions, images=images, reply_to=reply_to, parts=parts)


def normalize_message_id(value: object) -> str:
    """OneBot 消息号可能为负数；不接受任意 CQ 码或非数字目标。"""
    if type(value) not in {int, str}:
        return ""
    text = str(value)
    return text if text and len(text) <= 24 and text.lstrip("-").isdigit() and text.count("-") <= 1 else ""


def reply_segment(message_id: str) -> dict[str, Any]:
    return {"type": "reply", "data": {"id": message_id}}


def text_segment(text: str) -> dict[str, Any]:
    """将普通文本包装成 OneBot 文本消息段。"""
    return {"type": "text", "data": {"text": text}}


def image_segment(file: str) -> dict[str, Any]:
    """构造图片段；本地表情包使用Base64跨容器发送。"""
    return {"type": "image", "data": {"file": file}}


def at_segment(user_id: int) -> dict[str, Any]:
    """构造 @ 指定用户的消息段，协议中的 QQ 号使用字符串。"""
    return {"type": "at", "data": {"qq": str(user_id)}}
