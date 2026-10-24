"""群会话调度：固定聚批、串行处理和不绑定语义话题的活跃跟进。"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from app.config import Settings

if TYPE_CHECKING:
    from app.agent import AgentOutcome

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
    random_trigger: bool = False
    # QQ补发的离线旧消息不参加随机抽签，但仍可保存并用于理解上下文。
    random_eligible: bool = True
    # 动作通知没有可引用的OneBot消息号，不冒充普通文字或记忆来源。
    poke: bool = False
    # 仅入口确认时间足够新的普通群图片可供主动收藏，不从历史补收。
    collectible: bool = False
    parts: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class GroupBatch:
    entries: list[GroupMessage]

    @property
    def candidates(self) -> list[GroupMessage]:
        return [entry for entry in self.entries if entry.direct or entry.eligible]

    @property
    def focus(self) -> GroupMessage:
        direct = [entry for entry in self.entries if entry.direct]
        triggers = direct or [entry for entry in self.entries if entry.random_trigger]
        if triggers:
            last = triggers[-1]
            supplements = [entry for entry in self.candidates
                           if entry.user_id == last.user_id and entry.received_at >= last.received_at]
            return next((entry for entry in reversed(supplements) if entry.text), supplements[-1])
        candidates = self.candidates
        return next((entry for entry in reversed(candidates) if entry.text), candidates[-1])

    @property
    def direct(self) -> bool:
        return any(entry.direct for entry in self.entries)

    @property
    def random_trigger(self) -> bool:
        return not self.direct and any(entry.random_trigger for entry in self.entries)

    @property
    def wakes_followup(self) -> bool:
        return self.direct or self.random_trigger

    @property
    def reply_targets(self) -> set[str]:
        return {entry.message_id for entry in self.candidates if entry.message_id and not entry.poke}


@dataclass(slots=True)
class ConversationState:
    sender: Sender
    pending: list[GroupMessage] = field(default_factory=list)
    worker: asyncio.Task[None] | None = None
    following: bool = False
    idle_until: float = 0
    hard_until: float = 0
    next_run_at: float = 0

    def active(self, now: float) -> bool:
        if self.following and now < min(self.idle_until, self.hard_until):
            return True
        self.stop_followup()
        return False

    def stop_followup(self) -> None:
        self.following = False
        self.idle_until = self.hard_until = 0


Runner = Callable[[int, GroupBatch, Sender], Awaitable["AgentOutcome | None"]]


@dataclass(slots=True)
class GroupActivity:
    participants: dict[int, float] = field(default_factory=dict)
    last_triggered_at: float = float("-inf")


class GroupConversations:
    def __init__(self, settings: Settings, runner: Runner):
        self.settings, self.runner = settings, runner
        self.states: dict[int, ConversationState] = {}
        self.activity: dict[int, GroupActivity] = {}
        self.closed = False

    def _record_activity(self, group_id: int, entry: GroupMessage, now: float) -> GroupActivity:
        cutoff = now - self.settings.random_chat_window_seconds
        for old_id, activity in list(self.activity.items()):
            activity.participants = {user: seen for user, seen in activity.participants.items() if seen > cutoff}
            if (not activity.participants
                    and now - activity.last_triggered_at >= self.settings.random_chat_cooldown_seconds):
                self.activity.pop(old_id, None)
        activity = self.activity.setdefault(group_id, GroupActivity())
        if entry.random_eligible and (entry.text.strip() or entry.images):
            activity.participants[entry.user_id] = now
        return activity

    def random_probability(self, participants: int) -> float:
        if participants <= 0:
            return 0.0
        return min(self.settings.random_chat_max_probability,
                   self.settings.random_chat_min_probability
                   + max(0, participants - 1) * self.settings.random_chat_probability_per_user)

    async def submit(self, group_id: int, entry: GroupMessage, sender: Sender) -> str:
        """只入队，不等待模型；低活跃期由程序按不同发言人数抽签唤醒。"""
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
        activity = self._record_activity(group_id, entry, now) if self.settings.random_chat_enabled else None
        idle = state is None or (not state.active(now) and not state.pending
                                 and (state.worker is None or state.worker.done()))
        if idle and not entry.direct:
            if (activity is None or not entry.eligible or not entry.random_eligible
                    or not (entry.text.strip() or entry.images) or entry.text.lstrip().startswith("/")
                    or now - activity.last_triggered_at < self.settings.random_chat_cooldown_seconds
                    or (state is not None and now < state.next_run_at)
                    or (state is None and len(self.states) >= self.settings.chat_max_pending_groups)):
                return "ignored"
            probability = self.random_probability(len(activity.participants))
            if random.random() >= probability:
                return "ignored"
            entry.random_trigger = True
            activity.last_triggered_at = now
            logger.info("Random group join: group=%s participants=%s probability=%.3f",
                        group_id, len(activity.participants), probability)
        if state is None:
            if not (entry.direct or entry.random_trigger):
                return "ignored"
            if len(self.states) >= self.settings.chat_max_pending_groups:
                return "full"
            state = self.states[group_id] = ConversationState(sender)
        if not (entry.direct or entry.random_trigger) and not state.active(now) and not state.pending and state.worker is None:
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
                # 只有明确提问或程序随机抽中的新消息能开启活跃跟进。
                if not active and not any(entry.direct or entry.random_trigger for entry in state.pending):
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
                if not active and not any(entry.direct or entry.random_trigger for entry in fresh):
                    continue
                batch = GroupBatch(fresh)
                if not batch.candidates:
                    continue
                logger.info("Group batch: group=%s messages=%s direct=%s random=%s active=%s",
                            group_id, len(fresh), batch.direct, batch.random_trigger, active)
                outcome = await self.runner(group_id, batch, state.sender)
                now = time.monotonic()
                state.next_run_at = now + self.settings.ai_cooldown_seconds
                if outcome is not None and not outcome.error and outcome.wait_for_messages:
                    # 沉默不续期；普通跟进不能跨过超时重新开启，明确唤醒可以。
                    still_active = state.active(now)
                    if outcome.sent and (still_active or batch.wakes_followup):
                        if not still_active:
                            state.hard_until = now + self.settings.chat_active_max_seconds
                        state.following = True
                        state.idle_until = now + self.settings.chat_active_idle_seconds
                else:
                    state.stop_followup()
                logger.info("Group conversation state: group=%s active=%s", group_id, state.active(now))
        except asyncio.CancelledError:
            raise
        except Exception:
            state.stop_followup()
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
            state.stop_followup()
            if state.worker is not None:
                state.worker.cancel()
                tasks.append(state.worker)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        self.closed = True
        await self.cancel()
        self.activity.clear()
