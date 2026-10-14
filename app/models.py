"""解析 OneBot 文本、@ 和当前图片引用，并构造发出的消息段。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MessageView:
    """消息简化视图：文本、被 @ 用户，以及按顺序截取的前三张图片引用。"""

    text: str = ""
    mentions: list[int] = field(default_factory=list)
    # 只保留当前事件的图片引用，不下载、不保存图片内容。
    images: list[dict[str, str]] = field(default_factory=list)
    reply_to: str = ""


def parse_message(message: object) -> MessageView:
    """解析数组消息中的文本、@ 和图片；纯字符串仅作文本，不解析 CQ 码。"""
    if isinstance(message, str):
        return MessageView(text=message.strip())
    if not isinstance(message, list):
        return MessageView()

    text_parts: list[str] = []
    mentions: list[int] = []
    images: list[dict[str, str]] = []
    reply_to = ""
    for segment in message:
        if not isinstance(segment, dict):
            continue
        kind = segment.get("type")
        data = segment.get("data") or {}
        if not isinstance(data, dict):
            continue
        if kind == "text":
            text_parts.append(str(data.get("text", "")))
        elif kind == "at":
            qq = str(data.get("qq", ""))
            # 只提取具体用户的数字 QQ 号，忽略 @全体成员等非数字目标。
            if qq.isdigit():
                mentions.append(int(qq))
                text_parts.append(f"@{qq}")
        elif kind == "image" and len(images) < 3:
            # 不将带签名的下载地址、文件路径或不可信 summary 暴露给模型。
            images.append({
                "url": data.get("url", "")[:8193] if isinstance(data.get("url"), str) else "",
                "file_size": str(data.get("file_size", ""))[:20],
            })
        elif kind == "reply":
            reply_to = normalize_message_id(data.get("id"))
        # 语音、视频等其他消息段暂不参与解析。
    return MessageView(text="".join(text_parts).strip(), mentions=mentions, images=images, reply_to=reply_to)


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
