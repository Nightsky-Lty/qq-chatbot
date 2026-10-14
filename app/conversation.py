"""群会话调度：固定聚批、串行处理和由工具控制的话题活跃期。"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agent import AgentOutcome
from app.config import Settings

logger = logging.getLogger(__name__)
Sender = Callable[[int, list[dict[str, Any]]], Awaitable[Any]]


@dataclass(slots=True)
class GroupMessage:
    storage_id: int
    message_id: str
    user_id: int
    nickname: str
    text: str
    self_id: int
    mentions: list[int] = field(default_factory=list)
    reply_to: str = ""
    images: list[dict[str, str]] = field(default_factory=list)
    direct: bool = False
    eligible: bool = True
    received_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class GroupBatch:
    entries: list[GroupMessage]
    topic: str = ""

    @property
    def candidates(self) -> list[GroupMessage]:
        return [entry for entry in self.entries if entry.direct or entry.eligible]

    @property
    def focus(self) -> GroupMessage:
        direct = [entry for entry in self.entries if entry.direct]
        if direct:
            last = direct[-1]
            supplements = [entry for entry in self.candidates
                           if entry.user_id == last.user_id and entry.received_at >= last.received_at]
            return next((entry for entry in reversed(supplements) if entry.text), supplements[-1])
        candidates = self.candidates
        return next((entry for entry in reversed(candidates) if entry.text), candidates[-1])

    @property
    def direct(self) -> bool:
        return any(entry.direct for entry in self.entries)

    @property
    def reply_targets(self) -> set[str]:
        return {entry.message_id for entry in self.candidates if entry.message_id}


@dataclass(slots=True)
class ConversationState:
    sender: Sender
    pending: list[GroupMessage] = field(default_factory=list)
    worker: asyncio.Task[None] | None = None
    topic: str = ""
    idle_until: float = 0
    hard_until: float = 0
    next_run_at: float = 0

    def active(self, now: float) -> bool:
        if self.topic and now < min(self.idle_until, self.hard_until):
            return True
        self.topic = ""
        return False


Runner = Callable[[int, GroupBatch, Sender], Awaitable[AgentOutcome | None]]


class GroupConversations:
    def __init__(self, settings: Settings, runner: Runner):
        self.settings, self.runner = settings, runner
        self.states: dict[int, ConversationState] = {}
        self.closed = False

    async def submit(self, group_id: int, entry: GroupMessage, sender: Sender) -> str:
        """只入队，不等待计时或模型。低活跃期普通消息仅留在数据库。"""
        if self.closed:
            return "ignored"
        now = time.monotonic()
        for old_id, old in list(self.states.items()):
            if (not old.active(now) and now >= old.next_run_at and not old.pending
                    and (old.worker is None or old.worker.done())):
                self.states.pop(old_id, None)
        state = self.states.get(group_id)
        if state is not None and state.sender != sender:
            await self.cancel(group_id=group_id)
            state = None
        if state is None:
            if not entry.direct:
                return "ignored"
            if len(self.states) >= self.settings.chat_max_pending_groups:
                return "full"
            state = self.states[group_id] = ConversationState(sender)
        if not entry.direct and not state.active(now) and not state.pending and state.worker is None:
            return "ignored"
        if any(item.message_id == entry.message_id for item in state.pending) and entry.message_id:
            return "accepted"
        if len(state.pending) >= self.settings.chat_batch_max_messages:
            return "full"
        state.pending.append(entry)
        if state.worker is None or state.worker.done():
            state.worker = asyncio.create_task(self._work(group_id, state), name=f"group-conversation-{group_id}")
        return "accepted"

    async def _work(self, group_id: int, state: ConversationState) -> None:
        try:
            while state.pending and not self.closed:
                now = time.monotonic()
                active = state.active(now)
                # finish 后未@的新消息不能重新启动话题；明确的新提问不丢弃。
                if not active and not any(entry.direct for entry in state.pending):
                    state.pending.clear()
                    break
                first = state.pending[0]
                due = max(first.received_at + self.settings.chat_batch_seconds, state.next_run_at)
                await asyncio.sleep(max(0, due - now))
                now = time.monotonic()
                fresh = [entry for entry in state.pending
                         if now - entry.received_at < self.settings.chat_queue_wait_seconds]
                expired_direct = any(entry.direct for entry in state.pending if entry not in fresh)
                state.pending.clear()
                if expired_direct:
                    await state.sender(group_id, [{"type": "text", "data": {"text": "这次等得太久了，请重新@我。"}}])
                if not fresh:
                    continue
                active = state.active(now)
                if not active and not any(entry.direct for entry in fresh):
                    continue
                batch = GroupBatch(fresh, state.topic if active else "")
                if not batch.candidates:
                    continue
                logger.info("Group batch: group=%s messages=%s direct=%s active=%s",
                            group_id, len(fresh), batch.direct, active)
                outcome = await self.runner(group_id, batch, state.sender)
                now = time.monotonic()
                state.next_run_at = now + self.settings.ai_cooldown_seconds
                if outcome is not None and not outcome.error and outcome.wait_for_messages:
                    # 不发送的等待不能开启话题或刷新寿命，否则模型会永久保持活跃。
                    if outcome.sent and (batch.direct or not state.hard_until or now < state.hard_until):
                        if not batch.topic or (batch.direct and now >= state.hard_until):
                            state.hard_until = now + self.settings.chat_active_max_seconds
                        state.topic = outcome.topic[:120]
                        state.idle_until = now + self.settings.chat_active_idle_seconds
                    elif not state.active(now):
                        state.topic = ""
                else:
                    state.topic = ""
                logger.info("Group conversation state: group=%s active=%s", group_id, state.active(now))
        except asyncio.CancelledError:
            raise
        except Exception:
            state.topic = ""
            state.pending.clear()
            logger.exception("Group conversation failed: group=%s", group_id)
        finally:
            state.worker = None

    async def cancel(self, *, sender: Sender | None = None, group_id: int | None = None) -> None:
        targets = [(key, state) for key, state in self.states.items()
                   if (sender is None or state.sender == sender) and (group_id is None or key == group_id)]
        tasks = []
        for key, state in targets:
            self.states.pop(key, None)
            state.pending.clear()
            state.topic = ""
            if state.worker is not None:
                state.worker.cancel()
                tasks.append(state.worker)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        self.closed = True
        await self.cancel()
