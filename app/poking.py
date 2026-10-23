"""绑定当前会话的主动戳一戳工具：目标白名单、冷却和确认式发送。"""

import math
import time
from typing import Any, Awaitable, Callable

from app.config import Settings
from app.messaging import MessageSession

SendPoke = Callable[[int | None, int], Awaitable[None]]

SEND_POKE_TOOL = {"type": "function", "function": {
    "name": "send_poke", "description": "在当前会话真实戳一戳某人，由你按聊天语境决定是否使用，不自动戳回。省略user_id则戳当前发言人；群聊可选本轮允许成员，私聊仅当前对象。每轮最多一次，受会话冷却限制。",
    "parameters": {"type": "object", "properties": {
        "user_id": {"type": "integer", "minimum": 1,
                    "description": "可选：目标QQ号。省略为当前发言人；群聊仅可用程序列出的本轮允许成员，不能猜QQ、跨群戳、戳自己或全体；私聊仅当前对象。"},
    }, "required": [], "additionalProperties": False},
}}

POKE_TOOL_INSTRUCTIONS = """\n\n【主动戳一戳工具】
本轮提供send_poke，可按当前语境偶尔主动戳人或戳回，不必等用户明确要求；收到戳一戳也不是必须戳回。
省略user_id即当前发言人；群聊其他目标只选本轮允许@成员列表中的QQ，私聊只能戳当前对象。
每轮最多一次，冷却时不换目标绕过限制，不连续戳人；也可以不戳，用文字或表情正常回复。
轻互动可以只戳一戳；明确的问题仍应通过send_message（或已提供的send_voice）给出实际回答，不能只戳不答。
只有工具成功回执才能说已戳；失败或未确认不重试，不把“戳一戳”文字、@或CQ码当成真实动作。
戳成功后无需再发“已戳”确认；想补充文字仍须用send_message。自己的戳一戳不会再次唤醒自己。
"""


class PokeService:
    """每个群/私聊共享冷却，跨Agent轮次生效；不持久化，重启后重置。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._next_allowed: dict[tuple[str, int], float] = {}

    def reserve(self, conversation: tuple[str, int]) -> int:
        now = time.monotonic()
        self._next_allowed = {key: deadline for key, deadline in self._next_allowed.items() if deadline > now}
        remaining = self._next_allowed.get(conversation, now) - now
        if remaining > 0:
            return math.ceil(remaining)
        # 提交之前占用冷却，超时/断线/取消也不允许下一轮立即重试。
        self._next_allowed[conversation] = now + self.settings.poke_cooldown_seconds
        return 0


class PokeSession:
    def __init__(self, service: PokeService, messages: MessageSession, sender: SendPoke,
                 *, default_user_id: int, allowed_targets: set[int]):
        self.service, self.messages, self.sender = service, messages, sender
        self.default_user_id = default_user_id
        targets = {messages.target} if messages.private else allowed_targets
        self.allowed_targets = frozenset(targets) - {messages.bot_id}
        self.attempted = False

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target = arguments.get("user_id", self.default_user_id)
        if type(target) is not int or target <= 0 or target not in self.allowed_targets:
            return {"ok": False, "error": "只能戳当前会话程序允许的QQ号；私聊仅当前对象，不能戳自己、其他群成员或编造的QQ号。"}
        messages = self.messages
        registry = messages.registry
        conversation = ("private" if messages.private else "group", messages.target)
        async with messages.lock:
            if not self.service.settings.poke_enabled:
                return {"ok": False, "error": "主动戳一戳已关闭。"}
            if registry.delivery_uncertain:
                return {"ok": False, "error": "上一条动作未确认，本轮不再发送或戳人，避免重复。"}
            if self.attempted:
                return {"ok": False, "error": "本轮已经尝试过戳一戳，最多一次，不更换目标或重试。"}
            if registry.receipts and not messages.count:
                return {"ok": False, "error": "备忘操作已有回执，请先用send_message发送真实回执，不能只戳一戳代替确认。"}
            try:
                reply_to = messages.resolve_reply({})
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            remaining = self.service.reserve(conversation)
            if remaining:
                return {"ok": False, "retry_after_seconds": remaining,
                        "error": f"当前会话戳一戳冷却中，还需{remaining}秒；不要换目标重试，可正常聊天。"}
            self.attempted = True
            registry.output_attempted = True
            registry.delivery_uncertain = True
            messages.selected_reply = reply_to
            try:
                await self.sender(None if messages.private else messages.target, target)
                registry.outputs_sent += 1
                content = f"[戳一戳动作：机器人戳了戳QQ{target}]"
                if messages.private:
                    messages.db.log_private_message(messages.target, "assistant", content)
                else:
                    messages.log_group_reply(content, reply_to=reply_to)
            except Exception:
                return {"ok": False, "error": "戳一戳发送失败或未确认，本轮不重试；不能声称已经戳到。"}
            registry.delivery_uncertain = False
            messages.last_reply = reply_to
            return {"ok": True, "sent": 1, "user_id": target,
                    "note": "NapCat已确认接受戳一戳，不代表对方已看到；无需额外播报已戳。"}
