"""绑定当前会话的发送工具，限量、去重并等待连接层确认。"""

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

from app.database import Database
from app.models import at_segment, normalize_message_id, reply_segment, text_segment
from app.tools import ToolRegistry

SEND_MESSAGE_TOOL = {"type": "function", "function": {
    "name": "send_message", "description": "向当前会话发送一条文字，只回应一条新消息和一个话题。默认不引用、不@；群聊可用at_user_ids主动@本轮允许的成员，引用另用reply_to_message_id。普通群聊最多100字且不换行。",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 4000},
        "reply_to_message_id": {"type": "string", "minLength": 1, "maxLength": 24,
                                "description": "可选：显式显示QQ引用框。省略则普通发送，仍回应当前焦点；只在多人容易串话或需明确指向时填写，不要每条都引用。私聊不提供。"},
        "at_user_ids": {"type": "array", "maxItems": 3,
                        "items": {"type": "integer", "minimum": 1},
                        "description": "可选：真实@的QQ号列表，最多3人，只能使用程序列出的本轮允许@成员。可围绕当前话题主动点名、邀请或提醒，不必等对方要求；不要每条回复都@，不能@全体、自己或猜测QQ号。私聊不提供。"},
    }, "required": ["text"], "additionalProperties": False},
}}


class MessageSession:
    def __init__(self, registry: ToolRegistry, database: Database,
                 sender: Callable[[int, list[dict[str, Any]]], Awaitable[Any]], target: int,
                 *, private: bool, automatic: bool, preserve_layout: bool, lock: asyncio.Lock,
                 reply_targets: set[str] | None = None, default_reply: str = "", bot_id: int = 0,
                 max_chars: int = 100, max_messages: int | None = None,
                 mention_targets: set[int] | None = None):
        self.registry = registry
        self.db = database
        self.sender = sender
        self.target = target
        self.private = private
        self.preserve_layout = preserve_layout
        self.limit = max_messages if max_messages is not None else (1 if private or automatic else 2)
        self.count = 0
        self.sent_texts: set[str] = set()
        self.delivered_texts: set[str] = set()
        self.spoken_texts: set[str] = set()
        self.lock = lock
        self.reply_targets = reply_targets or set()
        self.default_reply = default_reply
        self.bot_id = bot_id
        self.max_chars = max_chars
        self.last_reply = default_reply
        self.selected_reply: str | None = None
        self.mention_targets = frozenset(mention_targets or ()) - {bot_id}

    def resolve_mentions(self, arguments: dict[str, Any]) -> list[int]:
        """只接受上下文快照内的本群成员；不从模型输出的文字中猜测QQ号。"""
        if self.private and "at_user_ids" in arguments:
            raise ValueError("私聊不支持@参数。")
        targets = arguments.get("at_user_ids", [])
        if not isinstance(targets, list) or len(targets) > 3:
            raise ValueError("at_user_ids必须是最多3个QQ号的数组。")
        if any(type(target) is not int or target <= 0 for target in targets):
            raise ValueError("@目标必须是正整数QQ号，不支持@全体。")
        if any(target not in self.mention_targets for target in targets):
            raise ValueError("只能@程序列出的本轮允许成员，不能@自己、其他群成员或编造的QQ号。")
        return list(dict.fromkeys(targets))

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
            mentions = self.resolve_mentions(arguments)
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
            if text in self.spoken_texts:
                return {"ok": True, "duplicate": True, "note": "同一句话本轮已用语音发送，不重复文字提交。"}
            signature = json.dumps([reply_to, text, sorted(mentions)], ensure_ascii=False)
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
                for index, part in enumerate(parts):
                    quote_to = arguments.get("reply_to_message_id", "")
                    # 长文分条仅第一条带@，避免同一次工具调用重复提醒。
                    part_mentions = mentions if index == 0 else []
                    segments = ([reply_segment(quote_to)] if quote_to else [])
                    segments += [at_segment(target) for target in part_mentions]
                    segments.append(text_segment((" " if part_mentions else "") + part))
                    sent_id = await self.sender(self.target, segments)
                    self.count += 1
                    sent += 1
                    self.registry.outputs_sent += 1
                    self.delivered_texts.add(part)
                    if self.private:
                        self.db.log_private_message(self.target, "assistant", part)
                    else:
                        recorded = "".join(f"@{target} " for target in part_mentions) + part
                        self.db.log_message(self.target, self.bot_id, "机器人", recorded, role="assistant",
                                            message_id=normalize_message_id(sent_id), reply_to_message_id=reply_to,
                                            mentions=part_mentions)
            except Exception:
                return {"ok": False, "error": "QQ发送未确认，本轮不自动重试。请结束，不能声称用户已收到。"}
            self.registry.delivery_uncertain = False
            self.sent_texts.add(signature)
            self.delivered_texts.add(text)
            self.last_reply = reply_to
            return {"ok": True, "sent": sent, "remaining": self.limit - self.count,
                    "note": "NapCat接受发送，不代表用户已读；无需再把同样文字作为最终回答输出。"}
