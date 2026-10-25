"""绑定当前会话的发送工具，限量、去重并等待连接层确认。"""

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

from app.database import Database
from app.models import at_segment, normalize_message_id, reply_segment, text_segment
from app.tools import ToolRegistry

SEND_MESSAGE_TOOL = {"type": "function", "function": {
    "name": "send_message", "description": "向当前会话发送口语短消息，普通文字每条1～20字；完整HTTP(S)网址单独一条，代码用完整代码块单独一条，二者允许超长且不拆内容。text中用换行分隔自然短句；整轮条数遵守运行环境，超额整次拒绝。只回应一个话题，默认不引用、不@；群聊引用/@仅用授权参数。",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 4000,
                 "description": "普通口语每条不超过运行环境上限，多条用换行分隔，通常1～3条即可。网址用完整http://或https://链接独占一行，不混说明或Markdown链接；代码用完整围栏代码块，保留换行缩进。网址/代码不受20字限制，但text合计最多4000字符。"},
        "reply_to_message_id": {"type": "string", "minLength": 1, "maxLength": 24,
                                "description": "可选：显式显示QQ引用框。省略则普通发送，仍回应当前焦点；只在多人容易串话或需明确指向时填写，不要每条都引用。私聊不提供。"},
        "at_user_ids": {"type": "array", "maxItems": 3,
                        "items": {"type": "integer", "minimum": 1},
                        "description": "可选：真实@的QQ号列表，最多3人，只能使用程序列出的本轮允许@成员。可围绕当前话题主动点名、邀请或提醒，不必等对方要求；不要每条回复都@，不能@全体、自己或猜测QQ号。私聊不提供。"},
    }, "required": ["text"], "additionalProperties": False},
}}


CODE_BLOCK = re.compile(r'(?m)^[ \t]*(?P<fence>`{3,}|~{3,})[^\r\n]*\r?\n[\s\S]*?^[ \t]*(?P=fence)[ \t]*\r?$')
PROTECTED_CODE = re.compile(CODE_BLOCK.pattern + r'|(?<!`)`[^`\r\n]+`(?!`)', re.M)
URL = re.compile(r'https?://[^\s<>"`，。！？；：“”‘’「」【】]+', re.I)
BARE_URL = re.compile(r'https?://[^\s<>"`]+', re.I)
MARKDOWN_LINK = re.compile(r'\[([^\]\r\n]+)\]\((https?://[^\s<>()]+(?:\([^\s<>()]*\)[^\s<>()]*)*)\)', re.I)


def _short_text(text: str, max_chars: int) -> list[str]:
    """普通文字优先采用模型给出的断点，再按标点与字符兜底。"""
    parts = []
    for line in text.replace("[分条]", "\n").splitlines():
        remaining = line.strip()
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            boundaries = list(re.finditer(r'[，。！？；、,.!?;：:…]+[”’」』）)]*|\s+', window))
            # 不为了开头的语气词拆出大量一两个字的碎片。
            boundary = next((m.end() for m in reversed(boundaries) if m.end() >= max_chars // 2), max_chars)
            parts.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].strip()
        if remaining:
            parts.append(remaining)
    return [part for part in parts if part]


def _inline_links(text: str, max_chars: int) -> list[str]:
    # 兼容模型偶尔输出Markdown链接，说明和真实网址分开发送。
    text = MARKDOWN_LINK.sub(lambda m: f'{m[1]}\n{m[2]}\n', text)
    parts = []
    start = 0
    for match in URL.finditer(text):
        parts.extend(_short_text(text[start:match.start()], max_chars))
        value = match[0]
        tail = ''
        # 自然语言中的句末标点/外层括号不属于网址；平衡的URL括号保留。
        isolated = not text[:match.start()].split('\n')[-1].strip() and not text[match.end():].split('\n')[0].strip()
        if not isolated:
            while value and (value[-1] in '.,!?;:' or
                             value[-1] in ')]}' and value.count(value[-1]) > value.count({')': '(', ']': '[', '}': '{'}[value[-1]])):
                tail = value[-1] + tail
                value = value[:-1]
        if value:
            parts.append(value)
        parts.extend(_short_text(tail, max_chars))
        start = match.end()
    parts.extend(_short_text(text[start:], max_chars))
    return parts


def _text_and_links(text: str, max_chars: int) -> list[str]:
    parts = []
    for line in text.splitlines():
        # 模型按约定单独给出的网址原样保留，包括Unicode路径与参数。
        if BARE_URL.fullmatch(line.strip()):
            parts.append(line.strip())
        else:
            parts.extend(_inline_links(line, max_chars))
    return parts


def short_messages(text: str, max_chars: int = 20) -> list[str]:
    """普通文字拆短句；独立网址和代码保留完整内容，各占一个消息气泡。"""
    parts = []
    start = 0
    for match in PROTECTED_CODE.finditer(text):
        parts.extend(_text_and_links(text[start:match.start()], max_chars))
        parts.append(match[0])
        start = match.end()
    parts.extend(_text_and_links(text[start:], max_chars))
    return parts


class MessageSession:
    def __init__(self, registry: ToolRegistry, database: Database,
                 sender: Callable[[int, list[dict[str, Any]]], Awaitable[Any]], target: int,
                 *, private: bool, automatic: bool, preserve_layout: bool, lock: asyncio.Lock,
                 reply_targets: set[str] | None = None, default_reply: str = "", bot_id: int = 0,
                 max_chars: int = 20, max_messages: int | None = None,
                 mention_targets: set[int] | None = None):
        self.registry = registry
        self.db = database
        self.sender = sender
        self.target = target
        self.private = private
        self.limit = max_messages if max_messages is not None else 6
        self.count = 0
        self.sent_texts: set[str] = set()
        self.delivered_texts: set[str] = set()
        self.spoken_texts: set[str] = set()
        self.lock = lock
        self.reply_targets = reply_targets or set()
        self.default_reply = default_reply
        self.bot_id = bot_id
        self.max_chars = max(1, min(20, max_chars))
        self.last_reply = default_reply
        self.selected_reply: str | None = None
        self.mention_targets = frozenset(mention_targets or ()) - {bot_id}

    def log_group_reply(self, content: str, *, message_id: str = "", reply_to: str = "",
                        mentions: list[int] | None = None) -> int:
        source_id = self.db.log_message(self.target, self.bot_id, "机器人", content, role="assistant",
                                        message_id=message_id, reply_to_message_id=reply_to, mentions=mentions)
        return source_id

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
        if self.registry.receipt_error and self.registry.receipts:
            # 备忘动作有失败时只回真实回执，不能发送模型的错误“已全部保存”确认。
            text = "\n".join(dict.fromkeys(self.registry.receipts))
        async with self.lock:
            if self.registry.delivery_uncertain:
                return {"ok": False, "error": "上一条发送未确认，本轮不再发送，避免重复。"}
            if text in self.spoken_texts:
                return {"ok": True, "duplicate": True, "note": "同一句话本轮已用语音发送，不重复文字提交。"}
            signature = json.dumps([reply_to, text, sorted(mentions)], ensure_ascii=False)
            if signature in self.sent_texts:
                return {"ok": True, "duplicate": True, "note": "此文字本轮已经发送，不重复提交。"}
            receipts = list(dict.fromkeys(self.registry.receipts))
            receipt_only = bool(receipts) and text == "\n".join(receipts)
            parts = [] if receipt_only else short_messages(text, self.max_chars)
            remaining = max(0, self.limit - self.count)
            if len(parts) > remaining:
                return {"ok": False, "error": f"本次需要{len(parts)}条，但本轮只剩{remaining}条；尚未发送本次文字。请保留要点，重写为每条1～{self.max_chars}字的口语短句，不要拆长文刷屏。"}
            # 可信操作回执不因普通回复预算而消失，也始终按单条字数拆分。
            pending_receipts = [receipt for receipt in receipts if receipt and receipt not in self.delivered_texts
                                and (receipt_only or receipt not in text)]
            parts.extend(part for receipt in pending_receipts for part in short_messages(receipt, self.max_chars))
            if not parts:
                return {"ok": False, "error": "没有可发送的非空短句。"}
            self.registry.output_attempted = True  # 超时/取消前就占位，不能误重发。
            self.registry.delivery_uncertain = True
            self.selected_reply = reply_to
            sent = 0
            try:
                for index, part in enumerate(parts):
                    quote_to = arguments.get("reply_to_message_id", "") if index == 0 else ""
                    # 同次调用仅第一条带@，避免重复提醒。
                    part_mentions = mentions if index == 0 else []
                    segments = ([reply_segment(quote_to)] if quote_to else [])
                    segments += [at_segment(target) for target in part_mentions]
                    segments.append(text_segment(part))
                    sent_id = await self.sender(self.target, segments)
                    self.count += 1
                    sent += 1
                    self.registry.outputs_sent += 1
                    self.delivered_texts.add(part)
                    if self.private:
                        self.db.log_private_message(self.target, "assistant", part)
                    else:
                        recorded = "".join(f"@{target} " for target in part_mentions) + part
                        self.log_group_reply(recorded, message_id=normalize_message_id(sent_id),
                                             reply_to=reply_to, mentions=part_mentions)
            except Exception:
                return {"ok": False, "error": "QQ发送未确认，本轮不自动重试。请结束，不能声称用户已收到。"}
            self.registry.delivery_uncertain = False
            self.sent_texts.add(signature)
            self.delivered_texts.add(text)
            self.delivered_texts.update(self.registry.receipts)
            self.last_reply = reply_to
            return {"ok": True, "sent": sent, "remaining": max(0, self.limit - self.count),
                    "note": "NapCat接受发送，不代表用户已读；无需再把同样文字作为最终回答输出。"}
