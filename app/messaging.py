"""绑定当前会话的发送工具，限量、去重并等待连接层确认。"""

import asyncio
import re
from typing import Any, Awaitable, Callable

from app.database import Database
from app.models import text_segment
from app.tools import ToolRegistry

SEND_MESSAGE_TOOL = {"type": "function", "function": {
    "name": "send_message", "description": "向当前会话实际发送一条聊天文字。收件人已绑定，不接受QQ号、群号、CQ码或图片；每次最多4000字。",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 4000},
    }, "required": ["text"], "additionalProperties": False},
}}


class MessageSession:
    def __init__(self, registry: ToolRegistry, database: Database,
                 sender: Callable[[int, list[dict[str, Any]]], Awaitable[None]], target: int,
                 *, private: bool, automatic: bool, preserve_layout: bool, lock: asyncio.Lock):
        self.registry = registry
        self.db = database
        self.sender = sender
        self.target = target
        self.private = private
        self.preserve_layout = preserve_layout
        self.limit = 1 if private or automatic else 2
        self.count = 0
        self.sent_texts: set[str] = set()
        self.lock = lock

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        text = arguments["text"].strip()
        if not text or "[CQ:" in text or "[不插话]" in text:
            return {"ok": False, "error": "请发送真实非空文字，不发送CQ码或内部标记。"}
        # 对旧模型输出的分条/换行仍作兼容，但所有文字都经过此发送工具。
        if self.private:
            parts = [text.replace("[分条]", "\n").strip()]
        else:
            chunks = re.split(r"(?m)^[ \t]*\[分条\][ \t]*\r?$", text)
            parts = [part.strip() for chunk in chunks
                     for part in ([chunk] if self.preserve_layout else chunk.splitlines()) if part.strip()]
        if not parts:
            return {"ok": False, "error": "没有可发送的文字，不要只传分条标记。"}
        if self.registry.receipt_error and self.registry.receipts:
            # 备忘动作有失败时只回真实回执，不能发送模型的错误“已全部保存”确认。
            parts = ["\n".join(dict.fromkeys(self.registry.receipts))]
        async with self.lock:
            if self.registry.delivery_uncertain:
                return {"ok": False, "error": "上一条发送未确认，本轮不再发送，避免重复。"}
            if text in self.sent_texts:
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
            sent = 0
            try:
                for part in parts:
                    await self.sender(self.target, [text_segment(part)])
                    self.count += 1
                    sent += 1
                    self.registry.outputs_sent += 1
                    if self.private:
                        self.db.log_private_message(self.target, "assistant", part)
            except Exception:
                return {"ok": False, "error": "QQ发送未确认，本轮不自动重试。请结束，不能声称用户已收到。"}
            self.registry.delivery_uncertain = False
            self.sent_texts.add(text)
            return {"ok": True, "sent": sent, "remaining": self.limit - self.count,
                    "note": "NapCat接受发送，不代表用户已读；无需再把同样文字作为最终回答输出。"}
