"""解析 OneBot 消息，并构造群聊和私聊使用的文本以及群聊 @ 消息段。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MessageView:
    """消息的简化视图：合并后的文本，以及按出现顺序提取的被 @ 用户 QQ 号。"""

    text: str = ""
    mentions: list[int] = field(default_factory=list)


def parse_message(message: object) -> MessageView:
    """解析数组消息中的文本和 @ 消息段；纯字符串仅作为文本，不解析 CQ 码。"""
    if isinstance(message, str):
        return MessageView(text=message.strip())
    if not isinstance(message, list):
        return MessageView()

    text_parts: list[str] = []
    mentions: list[int] = []
    for segment in message:
        if not isinstance(segment, dict):
            continue
        kind = segment.get("type")
        data = segment.get("data") or {}
        if kind == "text":
            text_parts.append(str(data.get("text", "")))
        elif kind == "at":
            qq = str(data.get("qq", ""))
            # 只提取具体用户的数字 QQ 号，忽略 @全体成员等非数字目标。
            if qq.isdigit():
                mentions.append(int(qq))
                text_parts.append(f"@{qq}")
        # 图片、语音等其他消息段暂不参与文本解析。
    return MessageView(text="".join(text_parts).strip(), mentions=mentions)


def text_segment(text: str) -> dict[str, Any]:
    """将普通文本包装成 OneBot 文本消息段。"""
    return {"type": "text", "data": {"text": text}}


def at_segment(user_id: int) -> dict[str, Any]:
    """构造 @ 指定用户的消息段，协议中的 QQ 号使用字符串。"""
    return {"type": "at", "data": {"qq": str(user_id)}}
