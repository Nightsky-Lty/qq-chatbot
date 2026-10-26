"""私聊有界排队：连续补充聚合，同一用户始终串行生成与发送。"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from app.config import Settings

if TYPE_CHECKING:
    from app.chat_images import ChatImageSession
    from app.prompting import ContextPacket

logger = logging.getLogger(__name__)
Sender = Callable[[int, list[dict[str, Any]]], Awaitable[Any]]


@dataclass(slots=True)
class PrivateRequest:
    source_id: int
    prompt: str
    context: "ContextPacket"
    image_session: "ChatImageSession | None" = None
    collection_jobs: dict = field(default_factory=dict)
    received_at: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class PrivateState:
    sender: Sender
    pending: list[PrivateRequest] = field(default_factory=list)
    worker: asyncio.Task[None] | None = None
    next_run_at: float = 0


class PrivateConversations:
    QUIET_SECONDS = 1.0
    MAX_BATCH_SECONDS = 3.0
    MAX_USERS = 32
    MAX_PENDING = 30

    def __init__(self, settings: Settings, slots: asyncio.Semaphore,
                 runner: Callable[[int, list[PrivateRequest], Sender], Awaitable[Any]]):
        self.settings, self.slots, self.runner = settings, slots, runner
        self.states: dict[int, PrivateState] = {}
        self.closed = False

    async def submit(self, user_id: int, request: PrivateRequest, sender: Sender) -> PrivateState | None:
        if self.closed:
            return None
        now = time.monotonic()
        for key, old in list(self.states.items()):
            if old.worker is None and not old.pending and now >= old.next_run_at:
                self.states.pop(key)
        state = self.states.get(user_id)
        if state is not None and state.sender != sender:
            await self.cancel(user_id=user_id)
            state = None
        if state is None:
            if len(self.states) >= self.MAX_USERS:
                return None
            state = self.states[user_id] = PrivateState(sender)
        if len(state.pending) >= self.MAX_PENDING:
            return None
        state.pending.append(request)
        if state.worker is None:
            state.worker = asyncio.create_task(self._work(user_id, state), name=f"private-conversation-{user_id}")
        return state

    async def _wait_batch(self, state: PrivateState) -> None:
        while state.pending:
            due = max(state.next_run_at, min(
                state.pending[-1].received_at + self.QUIET_SECONDS,
                state.pending[0].received_at + self.MAX_BATCH_SECONDS))
            due = min(due, state.pending[0].received_at + self.settings.chat_queue_wait_seconds)
            delay = due - time.monotonic()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def _work(self, user_id: int, state: PrivateState) -> None:
        try:
            while state.pending and not self.closed:
                await self._wait_batch(state)
                now = time.monotonic()
                fresh = [r for r in state.pending if now - r.received_at < self.settings.chat_queue_wait_seconds]
                expired = len(fresh) != len(state.pending)
                state.pending = fresh
                if expired:
                    await state.sender(user_id, [{"type": "text", "data": {"text": "前面的消息排队太久了，请重新发一下。"}}])
                if not state.pending:
                    continue
                remaining = (state.pending[0].received_at + self.settings.chat_queue_wait_seconds
                             - time.monotonic())
                acquired = False
                try:
                    try:
                        await asyncio.wait_for(self.slots.acquire(), max(0, remaining))
                    except TimeoutError:
                        continue
                    acquired = True
                    # 等全局名额期间的补充仍合并，实际运行前才确定本批边界。
                    await self._wait_batch(state)
                    now = time.monotonic()
                    batch = [r for r in state.pending
                             if now - r.received_at < self.settings.chat_queue_wait_seconds]
                    expired = len(batch) != len(state.pending)
                    state.pending.clear()
                    if expired:
                        await state.sender(user_id, [{"type": "text", "data": {"text": "前面的消息排队太久了，请重新发一下。"}}])
                    if batch:
                        state.next_run_at = time.monotonic() + self.settings.private_ai_cooldown_seconds
                        logger.info("Private batch: messages=%s", len(batch))
                        await self.runner(user_id, batch, state.sender)
                finally:
                    if acquired:
                        self.slots.release()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Private conversation failed")
        finally:
            state.worker = None
            if self.states.get(user_id) is state and time.monotonic() >= state.next_run_at:
                self.states.pop(user_id, None)

    async def cancel(self, *, sender: Sender | None = None, user_id: int | None = None) -> None:
        tasks = []
        for key, state in list(self.states.items()):
            if (sender is None or state.sender == sender) and (user_id is None or key == user_id):
                self.states.pop(key)
                state.pending.clear()
                if state.worker is not None:
                    state.worker.cancel()
                    tasks.append(state.worker)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        self.closed = True
        await self.cancel()
