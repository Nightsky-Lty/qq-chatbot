"""绑定当前会话的发送工具，限量、去重并等待连接层确认。"""

import asyncio
import re
from typing import Any, Awaitable, Callable

from app.database import Database
from app.models import normalize_message_id, reply_segment, text_segment
from app.tools import ToolRegistry

SEND_MESSAGE_TOOL = {"type": "function", "function": {
    "name": "send_message", "description": "向当前会话发送一条文字，只回应一条新消息和一个话题。默认普通发送，不显示QQ引用框；仅需明确指向时传reply_to_message_id引用本批允许的新消息。普通群聊最多100字且不换行。",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 4000},
        "reply_to_message_id": {"type": "string", "minLength": 1, "maxLength": 24,
                                "description": "可选：显式显示QQ引用框。省略则普通发送，仍回应当前焦点；只在多人容易串话或需明确指向时填写，不要每条都引用。私聊不提供。"},
    }, "required": ["text"], "additionalProperties": False},
}}


class MessageSession:
    def __init__(self, registry: ToolRegistry, database: Database,
                 sender: Callable[[int, list[dict[str, Any]]], Awaitable[Any]], target: int,
                 *, private: bool, automatic: bool, preserve_layout: bool, lock: asyncio.Lock,
                 reply_targets: set[str] | None = None, default_reply: str = "", bot_id: int = 0,
                 max_chars: int = 100, max_messages: int | None = None):
        self.registry = registry
        self.db = database
        self.sender = sender
        self.target = target
        self.private = private
        self.preserve_layout = preserve_layout
        self.limit = max_messages if max_messages is not None else (1 if private or automatic else 2)
        self.count = 0
        self.sent_texts: set[str] = set()
        self.lock = lock
        self.reply_targets = reply_targets or set()
        self.default_reply = default_reply
        self.bot_id = bot_id
        self.max_chars = max_chars
        self.last_reply = default_reply
        self.selected_reply: str | None = None

    def resolve_reply(self, arguments: dict[str, Any]) -> str:
        """解析逻辑回应对象；是否显示QQ引用仅取决于显式参数。"""
        reply_to = arguments.get("reply_to_message_id", self.selected_reply or self.default_reply)
        if (reply_to and (self.private or reply_to not in self.reply_targets
                          or normalize_message_id(reply_to) != reply_to)):
            raise ValueError("只能引用程序提供的本批新消息，不能引用历史、其他群或编造的消息号。")
        if self.selected_reply is not None and reply_to != self.selected_reply:
            raise ValueError("本批已经回应一个焦点；文字和表情补充都必须针对同一条消息，不切换话题。")
        return reply_to

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = arguments["text"].strip()
        if not text or "[CQ:" in text or "[不插话]" in text:
            return {"ok": False, "error": "请发送真实非空文字，不发送CQ码或内部标记。"}
        try:
            reply_to = self.resolve_reply(arguments)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        preserve_layout = self.preserve_layout and (not self.default_reply or reply_to == self.default_reply)
        if not self.private and not preserve_layout:
            if len(text) > self.max_chars or "\n" in text or "\r" in text or "[分条]" in text:
                return {"ok": False, "error": f"普通群回复最多{self.max_chars}字且不换行，只接一条新消息的一个话题；不要拆段总结历史。"}
        # 对旧模型输出的分条/换行仍作兼容，但所有文字都经过此发送工具。
        if self.private:
            parts = [text.replace("[分条]", "\n").strip()]
        else:
            chunks = re.split(r"(?m)^[ \t]*\[分条\][ \t]*\r?$", text)
            parts = [part.strip() for chunk in chunks
                     for part in ([chunk] if preserve_layout else chunk.splitlines()) if part.strip()]
        if not parts:
            return {"ok": False, "error": "没有可发送的文字，不要只传分条标记。"}
        if self.registry.receipt_error and self.registry.receipts:
            # 备忘动作有失败时只回真实回执，不能发送模型的错误“已全部保存”确认。
            parts = ["\n".join(dict.fromkeys(self.registry.receipts))]
        async with self.lock:
            if self.registry.delivery_uncertain:
                return {"ok": False, "error": "上一条发送未确认，本轮不再发送，避免重复。"}
            signature = f"{reply_to}:{text}"
            if signature in self.sent_texts:
                return {"ok": True, "duplicate": True, "note": "此文字本轮已经发送，不重复提交。"}
            if self.count >= self.limit:
                return {"ok": False, "error": "已达到本轮文字发送上限，请finish。"}
            parts = parts[:self.limit - self.count]
            if self.registry.receipts:
                receipts = [receipt for receipt in dict.fromkeys(self.registry.receipts) if receipt not in parts[-1]]
                if receipts:
                    parts[-1] += "\n\n" + "\n".join(receipts)
            self.registry.output_attempted = True  # 超时/取消前就占位，不能误重发。
            self.registry.delivery_uncertain = True
            self.selected_reply = reply_to
            sent = 0
            try:
                for part in parts:
                    quote_to = arguments.get("reply_to_message_id", "")
                    segments = ([reply_segment(quote_to)] if quote_to else []) + [text_segment(part)]
                    sent_id = await self.sender(self.target, segments)
                    self.count += 1
                    sent += 1
                    self.registry.outputs_sent += 1
                    if self.private:
                        self.db.log_private_message(self.target, "assistant", part)
                    else:
                        self.db.log_message(self.target, self.bot_id, "机器人", part, role="assistant",
                                            message_id=normalize_message_id(sent_id), reply_to_message_id=reply_to)
            except Exception:
                return {"ok": False, "error": "QQ发送未确认，本轮不自动重试。请结束，不能声称用户已收到。"}
            self.registry.delivery_uncertain = False
            self.sent_texts.add(signature)
            self.last_reply = reply_to
            return {"ok": True, "sent": sent, "remaining": self.limit - self.count,
                    "note": "NapCat接受发送，不代表用户已读；无需再把同样文字作为最终回答输出。"}
