"""机器人业务逻辑：群命令与召集、私聊、自动接话、后台聊天和风格模仿。"""

import asyncio
import copy
import json
import logging
import re
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from app.config import Settings
from app.balance import BalanceService, BalanceUnavailable
from app.server_status import ServerStatusService
from app.image_cache import ImageCache
from app.agent import AgentOutcome, ReactAgent
from app.conversation import GroupBatch, GroupConversations, GroupMessage
from app.chat_images import CHAT_IMAGE_TOOL, ChatImageService, ChatImageSession
from app.database import Database
from app.llm import DeepSeekClient, SYSTEM_PROMPT
from app.persona import PersonaSelector
from app.focus_context import focus_background, memory_context_query
from app.models import (
    MessageView, PokeView, at_segment, normalize_message_id, parse_message,
    parse_poke_event, reply_segment, text_segment,
)
from app.messaging import MessageSession, SEND_MESSAGE_TOOL
from app.poking import PokeService, PokeSession, SendPoke, SEND_POKE_TOOL
from app.memory import MemoryService
from app.memory_capture import MemoryCapture
from app.working_memory import WorkingMemoryManager
from app.dialogue import DialogueService, ThreadDeferral
from app.thread_history import READ_THREAD_HISTORY_TOOL, ThreadHistorySession
from app.memory_tools import register_memory_tools
from app.group_context import addressing, attach_dialogue_relation, build_group_history
from app.prompting import ContextPacket, PromptBuilder, PromptScene
from app.reminders import REMINDER_TOOLS, ReminderService
from app.stickers import STICKER_TOOLS, StickerService, StickerSession
from app.sticker_index import StickerIndexer
from app.sticker_collect import (
    COLLECT_STICKER_TOOL, CollectionJob, CollectionSession, StickerCollector,
)
from app.tools import FINISH_TOOL, WAIT_FOR_MESSAGES_TOOL, ToolRegistry
from app.tool_search import ToolRetriever, enable_tool_search
from app.web_search import WEB_TOOLS, WebSearchService, WebSession, user_urls
from app.voice import LIST_VOICE_CHARACTERS_TOOL, SEND_VOICE_TOOL, VoiceService, VoiceSession, VoiceTransport

logger = logging.getLogger(__name__)
# 发送回调由连接层提供：业务层传入群号或 QQ 号和消息段，不直接操作 WebSocket。
SendMessage = Callable[[int, list[dict[str, Any]]], Awaitable[Any]]
SendGroup = SendMessage
SendPrivate = SendMessage
# 会话类型也参与冷却和未完成请求的键，群号与 QQ 号相同也不会串话。
AIKey = tuple[str, int, int]


HELP_TEXT = """可用命令：
/名单 创建 名单名｜删除名单 名单名
/名单 添加 名单名 @某人｜删除 名单名 @某人
/名单 查看 [名单名]
/关键词 添加 词｜删除 词｜查看
/关键词 绑定 词 名单名｜解绑 词
只有完整发送已绑定的关键词，才会 @ 对应名单；改绑需先解绑。
/问 内容（或直接 @机器人）
/锐评 内容
/模仿 @某人 主题（无需授权，至少需要 5 条本群普通文字发言）
/余额（仅机器人管理员）
/服务器状态 或 /状态（仅机器人管理员）
/帮助
可在聊天中要求发表情包，图片需先由部署者放入本地图库。
群聊也可以要求语音回复，使用QQ提供的AI音色；暂不识别收到的语音。
@或戳一戳机器人、使用/问、/锐评唤醒，等3秒聚合连发文字和图片，模型按需看图。
话题活跃期间无需每条@；低活跃期普通消息也可能按群活跃人数被程序随机抽中接话。
群图片可按概率随机收藏，也可在聊天中要求收藏当前表情图，先去重再审核。
开启图片缓存时，接收到的图片会短期保存到服务器并发送给概要模型；缓存不等于表情收藏。
也可以私聊直接发文字聊天，私聊 /帮助 查看专用命令。"""

PRIVATE_HELP_TEXT = """私聊可直接发送文字聊天，无需 /问 或 @。
也可以戳一戳机器人唤醒聊天。
/问 内容
/锐评 内容
/余额（仅机器人管理员）
/服务器状态 或 /状态（仅机器人管理员）
/帮助
可直接说“明天上午九点提醒我开会”“查看我的提醒”“取消提醒3”。
提醒使用北京时间，目前只支持一次性提醒。
也可以说“发个开心的表情包”，需要本地图库中有对应图片。
可发图片或文字附图；开启缓存后支持近期图片指代，每轮最多查看3张原图。
名单、关键词召集和模仿功能请在群里使用。
私聊文本和机器人回复会保存在服务器，相关上下文会发送给 DeepSeek。
开启图片缓存时，收到的图片会短期保存到服务器并发送给概要模型，普通群图也一样。
缓存不等于收藏；看原图时所选图片还会发送给聊天模型。"""


class BotService:
    """协调配置、数据库和模型客户端，隔离处理群聊及每位用户的私聊。"""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        llm: DeepSeekClient,
        reminders: ReminderService | None = None,
    ):
        """保存依赖，并初始化按用户计时的冷却记录和按群分配的发送锁。"""
        self.settings = settings
        self.db = database
        self.llm = llm
        self.balance = BalanceService(settings, llm.client)
        self.server_status = ServerStatusService(settings)
        self.agent = ReactAgent(settings, llm)
        self.sticker_llm = DeepSeekClient(settings, purpose="sticker")
        self.memory_llm = DeepSeekClient(settings, purpose="memory")
        self.image_llm = DeepSeekClient(settings, purpose="image")
        self.thread_llm = DeepSeekClient(settings, purpose="thread")
        self.dialogue = DialogueService(settings, database, self.thread_llm)
        # 活跃焦点不跨进程恢复；持久消息归属和线程状态保留。
        self.dialogue.release()
        self.image_cache = ImageCache(settings, database, self.image_llm)
        self.reminders = reminders
        self.chat_images = ChatImageService(settings)
        self.chat_images.cached_loader = self.image_cache.load
        self.chat_images.cached_validator = self.image_cache.authorized
        self.web_search = WebSearchService(settings)
        self.stickers = StickerService(settings)
        self.sticker_indexer = StickerIndexer(settings, database, self.sticker_llm, self.stickers)
        self._ai_slots = asyncio.Semaphore(settings.ai_max_concurrency)
        self.sticker_collector = StickerCollector(
            settings, database, self.sticker_llm, self.stickers, slots=asyncio.Semaphore(1), announce=self._announce_collected,
        )
        self.sticker_collector.images.cached_loader = self.image_cache.load
        self.sticker_collector.images.cached_validator = self.image_cache.authorized
        self.memory = MemoryService(settings, database)
        self.tool_retriever = ToolRetriever(self.memory.provider, min_similarity=settings.tool_search_min_similarity)
        self.persona_selector = PersonaSelector(self.tool_retriever)
        self.memory.image_cache = self.image_cache
        self.memory_capture = MemoryCapture(settings, database, self.memory, self.memory_llm)
        self.working_memory = WorkingMemoryManager(settings, database, self.memory_capture)
        self.working_memory.media_lookup = lambda ns, ids: [self.image_cache.describe(r)
            for r in self.image_cache.rows(ns, max(ids, default=0), set(ids)) if r["source_id"] in ids]
        def receive_message(namespace, source_id):
            self.working_memory.receive(namespace, source_id)
            self.dialogue.observe(namespace, source_id)
        self.db.message_listener = receive_message
        self.pokes = PokeService(settings)
        self.voices = VoiceService(settings)
        self._confirmed_senders: dict[SendMessage, SendMessage] = {}
        self._poke_senders: dict[SendMessage, SendPoke] = {}
        self._voice_transports: dict[SendMessage, VoiceTransport] = {}
        # 冷却记录只存在于内存中，重启服务后会重置。
        self._last_ai_call: dict[AIKey, float] = {}
        self._trigger_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        # 每个发送动作串行提交；不同Agent可以在各步骤之间自然交错。
        self._group_reply_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._group_run_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        # 保存强引用和来源连接，断线或关闭时可以取消对应任务。
        self._ai_tasks: dict[asyncio.Task[None], tuple[SendMessage, AIKey]] = {}
        self._pending_ai_users: set[AIKey] = set()
        self._balance_tasks: dict[asyncio.Task[None], tuple[SendMessage, AIKey]] = {}
        self._pending_balance_users: set[AIKey] = set()
        self._status_tasks: dict[asyncio.Task[None], tuple[SendMessage, AIKey]] = {}
        self._pending_status_users: set[AIKey] = set()
        self.conversations = GroupConversations(settings, self._run_group_batch)
        def parser_active(group_id):
            state = self.conversations.states.get(group_id)
            return bool(state and (state.active(time.monotonic()) or (state.worker and not state.worker.done())))
        self.dialogue.is_active = parser_active
        self._closed = False

    async def cancel_pending(self, send_message: SendMessage | None = None, *, key: AIKey | None = None) -> None:
        """按发送连接或会话键取消后台任务；无过滤时取消全部，等待释放。"""
        confirmed = self._confirmed_senders.get(send_message) if send_message is not None else None
        if key is None or key[0] in {"group", "auto", "image"}:
            await self.sticker_collector.cancel(
                sender=confirmed or send_message, group_id=key[1] if key is not None else None,
            )
        if key is None:
            groups = [group for group, state in self.conversations.states.items()
                      if send_message is None or state.sender == send_message]
            await self.conversations.cancel(sender=send_message)
            for group in groups:
                await self.dialogue.cancel(group)
                self.dialogue.release(group)
        elif key[0] in {"group", "auto", "image"}:
            await self.conversations.cancel(group_id=key[1])
            await self.dialogue.cancel(key[1])
            self.dialogue.release(key[1])
        tasks = [
            task for task, (sender, task_key) in self._ai_tasks.items()
            if (send_message is None or sender in {send_message, confirmed})
            and (key is None or task_key == key
                 or (key[0] == "group" and task_key[0] == "group" and task_key[1] == key[1]))
        ]
        tasks.extend(
            task for task, (sender, task_key) in self._balance_tasks.items()
            if (send_message is None or sender in {send_message, confirmed})
            and (key is None or task_key == key
                 or (key[0] == "group" and task_key[0] == "group" and task_key[1] == key[1]))
        )
        tasks.extend(
            task for task, (sender, task_key) in self._status_tasks.items()
            if (send_message is None or sender in {send_message, confirmed})
            and (key is None or task_key == key
                 or (key[0] == "group" and task_key[0] == "group" and task_key[1] == key[1]))
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if send_message is not None and key is None:
            transport = self._voice_transports.pop(confirmed or send_message, None)
            if transport is not None:
                self.voices.clear_lookup(transport.get_characters)
            self._voice_transports.pop(send_message, None)
            self._confirmed_senders.pop(send_message, None)
            self._poke_senders.pop(send_message, None)
            if confirmed is not None:
                self._poke_senders.pop(confirmed, None)

    async def close(self) -> None:
        """阻止接受新请求并取消所有后台任务，在关闭模型客户端前调用。"""
        self._closed = True
        await self.conversations.close()
        await self.dialogue.cancel()
        await self.cancel_pending()
        await self.sticker_collector.close()
        await self.image_cache.close()
        await self.sticker_indexer.close()
        self.db.message_listener = None
        await self.working_memory.close()
        await self.memory_capture.close()
        await self.memory.close()
        await self.sticker_llm.close()
        await self.memory_llm.close()
        await self.image_llm.close()
        await self.thread_llm.close()
        await self.web_search.close()
        self._confirmed_senders.clear()
        self._poke_senders.clear()
        self._voice_transports.clear()
        self.voices.clear()

    def _finish_ai_task(self, task: asyncio.Task[None]) -> None:
        """任务结束后回收名额；尚未开始就被取消的任务也必须回收。"""
        entry = self._ai_tasks.pop(task, None)
        if entry is not None:
            self._pending_ai_users.discard(entry[1])
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("Unexpected background AI error", exc_info=(type(error), error, error.__traceback__))

    async def handle_event(
        self, event: dict[str, Any], send_group: SendGroup, send_private: SendPrivate | None = None,
        *, send_group_confirmed: SendGroup | None = None, send_private_confirmed: SendPrivate | None = None,
        send_poke_confirmed: SendPoke | None = None,
        voice_transport: VoiceTransport | None = None,
    ) -> None:
        """动作通知先分流；消息沿用私聊、群命令和活跃话题处理流程。"""
        if self._closed:
            return
        poke = parse_poke_event(event)
        if poke is None:
            if event.get("post_type") != "message" or event.get("message_type") not in {"group", "private"}:
                return
            if event.get("sub_type") == "notice":
                return
            user_id = int(event["user_id"])
            self_id = int(event.get("self_id", 0))
            # 忽略机器人自己的发言，避免回复触发自身业务逻辑。
            if user_id == self_id:
                return
        if send_group_confirmed is not None:
            self._confirmed_senders[send_group] = send_group_confirmed
        if send_private is not None and send_private_confirmed is not None:
            self._confirmed_senders[send_private] = send_private_confirmed
        if send_poke_confirmed is not None:
            self._poke_senders[send_group_confirmed or send_group] = send_poke_confirmed
            if send_private is not None:
                self._poke_senders[send_private_confirmed or send_private] = send_poke_confirmed
        if voice_transport is not None:
            self._voice_transports[send_group_confirmed or send_group] = voice_transport
        if poke is not None:
            await self._handle_poke(poke, send_group, send_private)
            return
        # 群收图逐张抽签，不沿用聊天看图的前三张限制；模型上下文仍最多3张。
        view = parse_message(event.get("message"), image_limit=None)
        command = view.text.strip()
        if self_id and self_id in view.mentions:
            command = re.sub(rf"@{self_id}(?!\d)", "", command).strip()
        if command == "/余额":
            private = event["message_type"] == "private"
            sender = send_private if private else send_group
            if sender is not None:
                target = user_id if private else int(event["group_id"])
                await self._queue_balance(user_id, target, sender, private=private)
            # 在聊天入库、群图收集和聚批之前分流，命令和余额回执均不进模型上下文。
            return
        if command in {"/服务器状态", "/状态"}:
            private = event["message_type"] == "private"
            sender = send_private if private else send_group
            if sender is not None:
                target = user_id if private else int(event["group_id"])
                await self._queue_status(user_id, target, sender, private=private)
            # 资源快照不进聊天历史/长期记忆，不请求模型或收集命令附图。
            return
        image_session = ChatImageSession(view.images[:3]) if view.images and self.settings.chat_image_enabled else None
        if event["message_type"] == "private":
            if send_private is not None:
                await self._handle_private(user_id, view, send_private, image_session=image_session)
            return
        group_id = int(event["group_id"])
        nickname = str((event.get("sender") or {}).get("card") or (event.get("sender") or {}).get("nickname") or user_id)
        role = str((event.get("sender") or {}).get("role", "member"))
        self.db.upsert_group_member(group_id, user_id, nickname, role)
        # 优先使用群名片，缺失时使用昵称或 QQ 号；命令文本也会被记录。
        message_id = normalize_message_id(event.get("message_id"))
        stored_text = view.text + (f" [图片{len(view.images)}张，{'附件状态见图片索引' if self.image_cache.enabled else '原图未保存'}]" if view.images else "")
        storage_id = self.db.log_message(group_id, user_id, nickname, stored_text,
                                         message_id=message_id, reply_to_message_id=view.reply_to, mentions=view.mentions)
        if view.images:
            cached_refs = self.image_cache.record(group_id, storage_id, user_id, message_id, view.images)
            view.images = cached_refs
            image_session = ChatImageSession(cached_refs) if self.settings.chat_image_enabled else None
        fresh = (isinstance(event.get("time"), (int, float))
                 and abs(time.time() - event["time"]) <= self.settings.sticker_collect_queue_wait_seconds)
        if view.images and fresh:
            self.sticker_collector.submit(
                group_id, user_id, self_id, message_id, view.images,
                self._confirmed_senders.get(send_group, send_group),
            )
        entry = GroupMessage(
            storage_id, message_id, user_id, nickname, view.text, self_id,
            mentions=view.mentions, reply_to=view.reply_to,
            images=(view.images if self.image_cache.enabled else
                    (view.images[:3] if self.settings.chat_image_enabled or self.sticker_collector.enabled else [])),
            direct=bool((self_id and self_id in view.mentions)
                        or (view.reply_to and self.db.is_bot_message(group_id, view.reply_to))),
            eligible=(not view.mentions or self_id in view.mentions)
                     and (not view.reply_to or self.db.is_bot_message(group_id, view.reply_to)),
            random_eligible=(isinstance(event.get("time"), (int, float))
                             and abs(time.time() - event["time"]) <= self.settings.random_chat_window_seconds),
            collectible=fresh,
        )

        is_admin = user_id in self.settings.admin_qq_ids or role in {"owner", "admin"}

        # 已识别的命令不再参与关键词召集或普通聊天，避免一次触发多个功能。
        if await self._handle_command(group_id, user_id, view, is_admin, send_group,
                                      image_session=image_session, entry=entry):
            return
        if await self._handle_keyword(group_id, view.text, send_group):
            return
        if entry.direct:
            entry.text = view.text.replace(f"@{self_id}", "").strip()
        if not entry.text and not entry.images and not entry.direct:
            return
        if entry.text.startswith("/") and not entry.direct:
            return
        await self._queue_group(group_id, entry, send_group)

    async def _handle_poke(self, poke: PokeView, send_group: SendGroup, send_private: SendPrivate | None) -> None:
        """戳机器人等同明确唤醒，不调用发戳接口，也不伪造可引用消息号。"""
        content = "[戳一戳动作通知：用户戳了戳机器人，并未发送文字]"
        if not poke.group_id:
            if send_private is not None:
                logger.info("Poke wake: scope=private context=%s user=%s", poke.user_id, poke.user_id)
                await self._handle_private(poke.user_id, MessageView(text=content), send_private, poke=True)
            return
        members = self.db.group_members(poke.group_id, {poke.user_id})
        nickname = members[0]["display_name"] if members else str(poke.user_id)
        if not members:
            self.db.upsert_group_member(poke.group_id, poke.user_id, nickname)
        storage_id = self.db.log_message(poke.group_id, poke.user_id, nickname, content, role="notice")
        entry = GroupMessage(
            storage_id, "", poke.user_id, nickname, content, poke.self_id,
            direct=True, random_eligible=False, poke=True,
        )
        logger.info("Poke wake: scope=group context=%s user=%s", poke.group_id, poke.user_id)
        await self._queue_group(poke.group_id, entry, send_group)

    async def _queue_balance(self, user_id: int, target: int, sender: SendMessage, *, private: bool) -> None:
        if self._closed:
            return
        if user_id not in self.settings.admin_qq_ids:
            await self._send_text(sender, target, "只有机器人管理员可以查询 API 余额。")
            return
        key = ("private" if private else "group", target, user_id)
        if key in self._pending_balance_users:
            await self._send_text(sender, target, "余额正在查询中，请稍等。")
            return
        if len(self._balance_tasks) >= 32:
            await self._send_text(sender, target, "余额查询较多，请稍后重试。")
            return
        task_sender = self._confirmed_senders.get(sender, sender)
        self._pending_balance_users.add(key)
        task = asyncio.create_task(self._run_balance(target, task_sender), name="balance-query")
        self._balance_tasks[task] = (task_sender, key)
        task.add_done_callback(self._finish_balance_task)

    async def _run_balance(self, target: int, sender: SendMessage) -> None:
        try:
            text = await self.balance.get_text()
        except BalanceUnavailable as exc:
            text = str(exc)
        # 后台任务才使用确认式发送；不经过MessageSession，因此不会写聊天历史。
        await self._send_text(sender, target, text)

    def _finish_balance_task(self, task: asyncio.Task[None]) -> None:
        entry = self._balance_tasks.pop(task, None)
        if entry is not None:
            self._pending_balance_users.discard(entry[1])
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.warning("Balance command failed: error=%s", type(error).__name__)

    async def _queue_status(self, user_id: int, target: int, sender: SendMessage, *, private: bool) -> None:
        if self._closed:
            return
        if user_id not in self.settings.admin_qq_ids:
            await self._send_text(sender, target, "只有机器人管理员可以查询服务器状态。")
            return
        key = ("private" if private else "group", target, user_id)
        if key in self._pending_status_users:
            await self._send_text(sender, target, "服务器状态正在采集中，请稍等。")
            return
        if len(self._status_tasks) >= 16:
            await self._send_text(sender, target, "状态查询较多，请稍后重试。")
            return
        task_sender = self._confirmed_senders.get(sender, sender)
        self._pending_status_users.add(key)
        task = asyncio.create_task(self._run_status(target, task_sender), name="server-status")
        self._status_tasks[task] = (task_sender, key)
        task.add_done_callback(self._finish_status_task)

    async def _run_status(self, target: int, sender: SendMessage) -> None:
        try:
            text = await self.server_status.get_text()
        except Exception as exc:
            logger.warning("Server status collection failed: error=%s", type(exc).__name__)
            text = "服务器状态采集失败，请联系部署者检查。"
        await self._send_text(sender, target, text)

    def _finish_status_task(self, task: asyncio.Task[None]) -> None:
        entry = self._status_tasks.pop(task, None)
        if entry is not None:
            self._pending_status_users.discard(entry[1])
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.warning("Server status command failed: error=%s", type(error).__name__)

    async def _queue_group(self, group_id: int, entry: GroupMessage, sender: SendGroup) -> None:
        status = await self.conversations.submit(group_id, entry, sender)
        if status == "accepted":
            self.dialogue.enqueue(group_id, entry)
        if status == "full" and entry.direct:
            await self._send_text(sender, group_id, "现在排队的消息有点多，请稍后再@我。")

    async def _run_group_batch(self, group_id: int, batch: GroupBatch, sender: SendGroup) -> AgentOutcome | None:
        async with self._group_run_locks[group_id]:
            return await self._run_group_batch_locked(group_id, batch, sender)

    async def _run_group_batch_locked(self, group_id: int, batch: GroupBatch, sender: SendGroup) -> AgentOutcome | None:
        """拿到全局名额后快照历史和本批附件，不读取下一批的用户消息。"""
        remaining = self.settings.chat_queue_wait_seconds - (time.monotonic() - batch.entries[0].received_at)
        if remaining <= 0:
            if batch.direct:
                await self._send_text(sender, group_id, "这次排队太久了，请重新@我。")
            return None
        state = self.conversations.states.get(group_id)
        if not batch.wakes_followup and (state is None or not state.active(time.monotonic())):
            return None
        selection = None
        if self.dialogue.enabled:
            if state is None or not state.active(time.monotonic()):
                self.dialogue.release(group_id)
            selection = await self.dialogue.select(group_id, batch)
            if isinstance(selection, ThreadDeferral):
                return AgentOutcome(0, 0, 0, wait_for_messages=True, retry_after=selection.retry_after)
            if selection is None:
                # 旁听/待切换不调用回复模型，也不把沉默当成退出当前交流。
                return AgentOutcome(0, 0, 0, wait_for_messages=True)
            batch = selection.batch
        remaining = self.settings.chat_queue_wait_seconds - (time.monotonic() - batch.entries[0].received_at)
        if remaining <= 0:
            return None
        try:
            await asyncio.wait_for(self._ai_slots.acquire(), timeout=max(0.01, remaining))
        except TimeoutError:
            if batch.direct:
                await self._send_text(sender, group_id, "这次排队太久了，请重新@我。")
            return None
        try:
            focus = batch.focus
            boundary = selection.original_boundary if selection else max(entry.storage_id for entry in batch.entries)
            preferred = {entry.storage_id for entry in batch.entries}
            quoted_sources = set()
            for entry in batch.entries:
                quoted = self.db.message_by_onebot_id(group_id, entry.reply_to) if entry.reply_to else None
                if quoted is not None:
                    quoted_sources.add(quoted["id"])
            image_preferred = preferred | quoted_sources | (selection.history_ids if selection else set())
            cached_rows = self.image_cache.rows(group_id, boundary, image_preferred)
            by_source = defaultdict(list)
            for row in cached_rows:
                by_source[row["source_id"]].append(self.image_cache.describe(row))
            short_context = {}
            if self.settings.working_memory_enabled or selection:
                short_context = self.working_memory.snapshot(
                    group_id, before_id=min(e.storage_id for e in batch.entries), query=focus.text,
                    roots=[e.reply_to for e in batch.entries if e.reply_to], bot_id=focus.self_id,
                    image_lookup=lambda source: by_source.get(source, []),
                    source_ids=selection.history_ids if selection else None)
                history_entries = short_context["近期往返"]
            else:
                _, history_entries = build_group_history(self.db, self.settings, group_id, batch,
                                                         image_lookup=lambda source: by_source.get(source, []))
            images: list[dict[str, str]] = []
            image_ids: dict[int, list[str]] = {}
            collection_jobs: dict[str, CollectionJob] = {}
            stable_ids = []
            image_sources = {}
            timeline_sources = {row["本地记录ID"] for row in history_entries}
            if selection:
                # 旁听thread的图片不自动进入工具候选；显式引用的证据仍可访问。
                allowed_sources = timeline_sources | preferred | quoted_sources
                if any(word in focus.text for word in ('图', '这张', '那张', '表情')):
                    # 当前请求显式谈及图片时，可把近期旁听图片作为有来源的参考候选。
                    allowed_sources.update(row['id'] for row in selection.references)
                cached_rows = [row for row in cached_rows if row['source_id'] in allowed_sources]
            if self.image_cache.enabled:
                for row in cached_rows[:self.settings.image_cache_context_limit]:
                    image_id = f"media_{row['id']}"
                    ref = self.image_cache.reference(row)
                    stable_ids.append(image_id)
                    images.append(ref)
                    image_ids.setdefault(row["source_id"], []).append(image_id)
                    image_sources[image_id] = self.image_cache.describe(row)
                    if row["source_id"] in timeline_sources and "概要" in image_sources[image_id]:
                        for field in ("概要", "图中文字", "不确定"):
                            image_sources[image_id].pop(field, None)
                        image_sources[image_id]["概要位置"] = "短期记忆近期往返或群聊时间线中此来源消息的图片资料"
                    if row["status"] in {"ready", "pending"}:
                        collection_jobs[image_id] = CollectionJob(group_id, row["user_id"], focus.self_id,
                            row["message_id"], ref, self._confirmed_senders.get(sender, sender))
            # 关闭缓存时保留原有最多3张当前批次附件路径。
            ordered = [] if self.image_cache.enabled else [focus] + [entry for entry in batch.candidates if entry is not focus]
            for entry in ordered:
                ids = []
                for image in entry.images:
                    if len(images) >= 3:
                        break
                    images.append(image)
                    image_id = f"image_{len(images)}"
                    ids.append(image_id)
                    image_sources[image_id] = {"图片编号": image_id, "消息号": entry.message_id,
                                               "发图人QQ": entry.user_id, "本地记录ID": entry.storage_id}
                    if entry.collectible and not entry.poke and entry.user_id != entry.self_id:
                        collection_jobs[image_id] = CollectionJob(
                            group_id, entry.user_id, entry.self_id, entry.message_id, dict(image),
                            self._confirmed_senders.get(sender, sender), queued_at=entry.received_at,
                        )
                image_ids[entry.storage_id] = ids
            related_ids = {focus.user_id}
            for entry in batch.entries:
                related_ids.update(entry.mentions)
            reply_rows = {
                entry.reply_to: self.db.message_by_onebot_id(group_id, entry.reply_to)
                for entry in batch.entries if entry.reply_to
            }
            related_ids.update(
                row["user_id"] for row in reply_rows.values() if row is not None and row["role"] == "user"
            )
            member_index = {member["user_id"]: member for member in self.db.group_members(group_id, related_ids)}
            # 只开放本轮可见的身份元数据，不解析正文里的QQ号，也不读取下一批成员。
            mention_targets = {item["QQ"]: item["昵称"] for item in reversed(history_entries)
                               if item["角色"] == "user"}
            mention_targets.update({member["user_id"]: member["display_name"] for member in member_index.values()})
            mention_targets.update({entry.user_id: entry.nickname for entry in batch.entries})
            bot_ids = {entry.self_id for entry in batch.entries}
            mention_targets = {qq: name for qq, name in mention_targets.items() if qq > 0 and qq not in bot_ids}
            current = []
            visible_ids = {row["消息号"] for row in history_entries} | {entry.message_id for entry in batch.entries}
            blocked = self.db.memories.blocked_sources(group_id)
            for entry in batch.entries:
                reply = reply_rows.get(entry.reply_to)
                if reply is not None and (reply["id"] > boundary or reply["id"] in blocked):
                    reply = None
                quote = None
                if reply is not None:
                    quote = {"消息号": reply["message_id"], "作者QQ": reply["user_id"],
                             "作者昵称": reply["nickname"], "角色": reply["role"]}
                    if reply["message_id"] in visible_ids:
                        quote["原文位置"] = "群聊时间线或本批新消息"
                    else:
                        quote["内容"] = reply["content"][:500]
                elif entry.reply_to:
                    quote = {"消息号": entry.reply_to, "状态": "原文未找到或不在可访问范围"}
                current.append({
                    "消息号": entry.message_id, "QQ": entry.user_id, "昵称": entry.nickname,
                    "内容": entry.text[:4000], "明确提问": entry.direct,
                    "事件类型": "戳一戳" if entry.poke else "消息",
                    "可回复": entry in batch.candidates,
                    "寻址": addressing(entry.mentions, dict(reply) if reply is not None else None,
                                       bot_id=entry.self_id, direct=entry.direct),
                    "本地记录ID": entry.storage_id,
                    "时间": (self.db.group_context_message(group_id, entry.message_id) or {}).get("created_at"),
                    "可引用": entry.message_id in batch.reply_targets and not entry.poke,
                    "提到的成员": [member_index.get(target, {"user_id": target}) for target in entry.mentions],
                    "引用消息": quote,
                    "图片编号": image_ids.get(entry.storage_id, []), "图片数量": len(entry.images),
                })
            context = ContextPacket(
                task={"唤醒消息号": [e.message_id for e in batch.entries if e.direct],
                      "当前焦点消息号": focus.message_id, "本批新消息": current},
                timeline=[] if short_context else history_entries, working=short_context,
                images=list(image_sources.values()),
                runtime={"机器人身份": {"QQ": focus.self_id, "角色": "群内参与者，不是所有消息的收件人"},
                         "相关成员索引": list(member_index.values())},
            )
            if selection:
                visible = {item['本地记录ID'] for item in [*history_entries, *current]}
                for item in [*history_entries, *current]:
                    attach_dialogue_relation(item, self.dialogue.relation_context(
                        group_id, item['本地记录ID'], boundary=boundary, visible_ids=visible))
                context.runtime["当前交流线程"] = {"编号": batch.thread_id,
                    "约束": "只回应本线程；其他交流仅供旁听，切换由程序决定"}
                context.references = [{**self.working_memory._public(group_id, row, focus.self_id),
                    "可回复": False, "说明": "旁听参考，不是本轮新请求", "内容": row['content'][:400]}
                    for row in selection.references]
                for item in context.references:
                    resources = [image['图片编号'] for image in image_sources.values()
                                 if image.get('本地记录ID') == item['本地记录ID']]
                    if resources:
                        item['图片资料（非原话）'] = [{'图片编号': resource, '概要位置': '图片目录'} for resource in resources]
                context.working['说明'] = '当前线程必要原文；历史是背景，不是本轮新授权。旁听参考不属于当前请求。'
            recall_query = memory_context_query(focus.text, context) if self.settings.memory_context_query_enabled else ''
            memories = await self.memory.retrieve(
                group_id, focus.text, user_ids={focus.user_id, *focus.mentions},
                boundary=max(entry.storage_id for entry in batch.entries),
                **({'context_query': recall_query} if recall_query else {}),
            )
            logger.info("Group context: group=%s history=%s new=%s images=%s",
                        group_id, len(history_entries), len(current), len(images))
            try:
                # 无@跟进不能因等待并发名额或模型耗时而越过活跃期时限。
                timeout = None if batch.wakes_followup else max(0.01, min(state.idle_until, state.hard_until) - time.monotonic())
                async with asyncio.timeout(timeout):
                    outcome = await self._run_ai_reply(
                        group_id, focus.user_id, focus.text, self._confirmed_senders.get(sender, sender),
                        context=context, image_session=ChatImageSession(images, image_ids=stable_ids, sources=image_sources) if images and self.settings.chat_image_enabled else None,
                        automatic=not batch.wakes_followup, batch=batch, mention_targets=mention_targets,
                        collection_jobs=collection_jobs, recalled_memories=memories,
                    )
                    if selection:
                        self.dialogue.complete(group_id, batch.thread_id, outcome)
                    return outcome
            except TimeoutError:
                if selection:
                    self.dialogue.complete(group_id, batch.thread_id, None)
                return None
        finally:
            self._ai_slots.release()

    async def _handle_private(
        self, user_id: int, view: MessageView, send_private: SendPrivate, *, image_session: ChatImageSession | None = None,
        poke: bool = False,
    ) -> None:
        """私聊普通文本直接聊天，支持帮助和锐评；从不执行群管理或模仿。"""
        text = view.text.strip()
        if not text and image_session is None and not (view.images and self.image_cache.enabled):
            await self._send_text(send_private, user_id, "请发文字给我；图片聊天需要部署者开启 CHAT_IMAGE_ENABLED，语音和视频暂不支持。")
            return
        if text == "/帮助":
            await self._send_text(send_private, user_id, PRIVATE_HELP_TEXT)
            return
        group_only = ("/名单", "/关键词", self.settings.mimic_prefix)
        if text.startswith(group_only):
            await self._send_text(send_private, user_id, "这个功能只支持群聊，请在群里使用。私聊发 /帮助 可以查看可用命令。")
            return
        prompt = text or "[用户发送了一条图片消息]"
        if text.startswith(self.settings.chat_prefix):
            prompt = text.removeprefix(self.settings.chat_prefix).strip()
            if not prompt and image_session is None:
                await self._send_text(send_private, user_id, f"直接发文字即可聊天，也可以用：{self.settings.chat_prefix} 你想说的话")
                return
        elif text.startswith(self.settings.evaluate_prefix):
            content = text.removeprefix(self.settings.evaluate_prefix).strip()
            if not content and image_session is None:
                await self._send_text(send_private, user_id, f"用法：{self.settings.evaluate_prefix} 要评价的内容")
                return
            prompt = f"请对下面这段内容或本条附图做简短、有梗但不伤人的评价：\n{content}"
        elif text.startswith("/"):
            await self._send_text(send_private, user_id, "未知私聊命令，请发送 /帮助；普通文字可以直接聊天。")
            return
        # 先读历史再写当前文本，不把当前问题重复加入上下文；按用户独立查询。
        history = None
        if not self.settings.working_memory_enabled:
            rows = self.db.recent_private_messages(user_id, self.settings.context_message_limit)
            history = [{"role": row["role"], "content": row["content"]} for row in rows]
        prompt = prompt or "[用户发送了一条图片消息]"
        stored_text = prompt + (f"\n[附带图片{len(view.images)}张，{'附件状态见图片索引' if self.image_cache.enabled else '原图未保存'}]" if view.images else "")
        private_source_id = self.db.log_private_message(user_id, "user", stored_text)
        context = ContextPacket(task={"用户请求": prompt, "事件类型": "戳一戳" if poke else "消息"})
        if view.images:
            self.image_cache.record(-user_id, private_source_id, user_id, "", view.images)
        cached_rows = self.image_cache.rows(-user_id, private_source_id, {private_source_id})
        if cached_rows:
            selected = cached_rows[:self.settings.image_cache_context_limit]
            if self.settings.chat_image_enabled:
                image_session = ChatImageSession([self.image_cache.reference(row) for row in selected],
                    image_ids=[f"media_{row['id']}" for row in selected])
            context.images = [self.image_cache.describe(row) for row in selected]
        await self._ai_reply(
            user_id, user_id, prompt, send_private,
            history=history, context=context,
            private=True, image_session=image_session, private_source_id=private_source_id,
        )

    async def _handle_command(
        self,
        group_id: int,
        user_id: int,
        view: MessageView,
        is_admin: bool,
        send_group: SendGroup,
        *,
        image_session: ChatImageSession | None = None,
        entry: GroupMessage | None = None,
    ) -> bool:
        """分发群内命令；识别并处理后返回 True，否则返回 False。"""
        text = view.text.strip()
        if text == "/帮助":
            await self._send_text(send_group, group_id, HELP_TEXT)
            return True
        if text.startswith("/名单"):
            if not is_admin:
                await self._send_text(send_group, group_id, "只有群主、管理员或机器人管理员可以维护名单。")
                return True
            await self._manage_users(group_id, text, view.mentions, send_group)
            return True
        if text.startswith("/关键词"):
            if not is_admin:
                await self._send_text(send_group, group_id, "只有管理员可以维护关键词。")
                return True
            await self._manage_keywords(group_id, text, send_group)
            return True
        if text in {"/允许模仿", "/禁止模仿"}:
            # 旧指令只做停用提示，不能让用户误以为仍然控制样本使用权限。
            await self._send_text(send_group, group_id, f"模仿授权功能已移除，这个旧指令不再生效。群内可以直接使用 {self.settings.mimic_prefix} @某人 [主题]。")
            return True
        if text.startswith(self.settings.mimic_prefix):
            await self._mimic(group_id, user_id, text, view.mentions, send_group, entry=entry)
            return True
        if text.startswith(self.settings.evaluate_prefix):
            content = text.removeprefix(self.settings.evaluate_prefix).strip()
            if not content and image_session is None:
                await self._send_text(send_group, group_id, "用法：/锐评 要评价的内容")
            else:
                if entry is not None:
                    entry.direct = True
                    entry.text = f"请对下面内容或附图做简短、有梗但不伤人的评价：{content}"
                    await self._queue_group(group_id, entry, send_group)
            return True
        if text.startswith(self.settings.chat_prefix):
            prompt = text.removeprefix(self.settings.chat_prefix).strip()
            if not prompt and image_session is None:
                await self._send_text(send_group, group_id, f"用法：{self.settings.chat_prefix} 你想说的话")
            else:
                if entry is not None:
                    entry.direct = True
                    entry.text = prompt or "[用户发送了一条图片消息]"
                    await self._queue_group(group_id, entry, send_group)
            return True
        return False

    async def _manage_users(self, group_id: int, text: str, mentions: list[int], send_group: SendGroup) -> None:
        """独立管理命名名单；旧的无名单名成员命令继续操作迁移后的“默认”名单。"""
        usage = "用法：/名单 创建|删除名单 名单名；/名单 添加|删除 名单名 @某人；/名单 查看 [名单名]。名单名为 1～32 字且不能含空白或 @。"
        parts = text.split(maxsplit=2)
        if len(parts) < 2 or parts[0] != "/名单":
            await self._send_text(send_group, group_id, usage)
            return
        action = parts[1]
        argument = parts[2].strip() if len(parts) == 3 else ""
        if action == "查看" and not argument:
            lists = self.db.list_lists(group_id)
            content = "\n".join(f'{row["name"]}（{row["member_count"]} 人）' for row in lists)
            await self._send_text(send_group, group_id, "当前群的名单：\n" + (content or "暂无，请先用 /名单 创建 名单名"))
            return
        if action not in {"创建", "删除名单", "查看", "添加", "删除"}:
            await self._send_text(send_group, group_id, usage)
            return
        if action in {"添加", "删除"}:
            # @ 消息段已被解析为 @QQ号；仅移除实际提及，不能把普通数字当作目标。
            for target in mentions:
                argument = re.sub(rf"@{target}(?!\d)", " ", argument)
            name = argument.strip() or "默认"
        else:
            name = argument
        if not name or len(name) > 32 or any(char.isspace() for char in name) or "@" in name:
            await self._send_text(send_group, group_id, usage)
            return
        if action == "创建":
            created = self.db.create_list(group_id, name)
            await self._send_text(send_group, group_id, f"已创建名单：{name}" if created else f"名单“{name}”已存在。")
            return
        if not self.db.has_list(group_id, name):
            await self._send_text(send_group, group_id, f"名单“{name}”不存在，请先发送 /名单 创建 {name}")
            return
        if action == "删除名单":
            self.db.delete_list(group_id, name)
            await self._send_text(send_group, group_id, f"已删除名单“{name}”及其成员，相关关键词已解绑但未删除。")
        elif action == "查看":
            users = self.db.list_managed_users(group_id, name)
            content = "、".join(str(row["user_id"]) for row in users) or "暂无成员"
            # 管理时只列出 QQ 号，不额外 @；真正召集由关键词触发。
            await self._send_text(send_group, group_id, f"名单“{name}”：{content}")
        elif not mentions:
            await self._send_text(send_group, group_id, f"请 @ 要操作的用户，例如：/名单 {action} {name} @某人")
        elif action == "添加":
            targets = list(dict.fromkeys(mentions))
            for target in targets:
                self.db.add_managed_user(group_id, name, target)
            await self._send_text(send_group, group_id, f"已将 {len(targets)} 人加入名单“{name}”（已有成员不会重复添加）。")
        elif action == "删除":
            removed = sum(self.db.remove_managed_user(group_id, name, target) for target in set(mentions))
            await self._send_text(send_group, group_id, f"已从名单“{name}”删除 {removed} 人。")

    async def _manage_keywords(self, group_id: int, text: str, send_group: SendGroup) -> None:
        """独立维护关键词及绑定，允许多关键词共用名单，但禁止直接覆盖绑定。"""
        usage = "用法：/关键词 添加|删除|解绑 关键词；/关键词 绑定 关键词 名单名；/关键词 查看"
        parts = text.split(maxsplit=2)
        if len(parts) < 2 or parts[0] != "/关键词":
            await self._send_text(send_group, group_id, usage)
            return
        action = parts[1]
        if action == "查看" and len(parts) == 2:
            keywords = self.db.list_keywords(group_id)
            content = "\n".join(f'{row["keyword"]} → {row["list_name"] or "未绑定"}' for row in keywords)
            await self._send_text(send_group, group_id, "关键词与名单：\n" + (content or "暂无"))
            return
        # 最多拆分两次，保留关键词内部的空格。
        if len(parts) < 3 or not parts[2].strip():
            await self._send_text(send_group, group_id, usage)
            return
        action, keyword = parts[1], parts[2].strip()
        if action == "添加":
            created = self.db.add_keyword(group_id, keyword)
            await self._send_text(send_group, group_id, f"已添加关键词：{keyword}。请绑定名单后使用。" if created else f"关键词“{keyword}”已存在，原绑定保持不变。")
        elif action == "删除":
            removed = self.db.remove_keyword(group_id, keyword)
            await self._send_text(send_group, group_id, "已删除。" if removed else "没有找到这个关键词。")
        elif action == "绑定":
            # 最后一个空白分隔的参数是名单名，关键词内部的空格仍然保留。
            binding = keyword.rsplit(maxsplit=1)
            if len(binding) != 2:
                await self._send_text(send_group, group_id, usage)
                return
            keyword, name = binding
            result = self.db.bind_keyword(group_id, keyword, name)
            replies = {
                "bound": f"已绑定：{keyword} → {name}",
                "already_bound": f"关键词“{keyword}”已经绑定名单“{name}”。",
                "missing_keyword": f"关键词不存在，请先发送 /关键词 添加 {keyword}",
                "missing_list": f"名单不存在，请先发送 /名单 创建 {name}",
                "conflict": f"关键词“{keyword}”已绑定其他名单，不能绑定多个名单。请先发送 /关键词 解绑 {keyword} 再重新绑定。",
            }
            await self._send_text(send_group, group_id, replies[result])
        elif action == "解绑":
            removed = self.db.unbind_keyword(group_id, keyword)
            await self._send_text(send_group, group_id, "已解绑，关键词和名单均保留。" if removed else "这个关键词不存在或尚未绑定名单。")
        else:
            await self._send_text(send_group, group_id, usage)

    async def _handle_keyword(self, group_id: int, text: str, send_group: SendGroup) -> bool:
        """完整匹配已绑定关键词时处理召集并返回 True，避免再触发自动聊天。"""
        # 忽略首尾空白，保留消息内部内容；关键词作为其他句子的子串时不触发。
        text = text.strip()
        if not text:
            return False
        # 按群串行发送，避免同一群的两次召集消息相互穿插；此处没有触发冷却。
        async with self._trigger_locks[group_id]:
            name = self.db.keyword_list(group_id, text)
            if name is None:
                return False
            users = self.db.list_managed_users(group_id, name)
            if not users:
                return True
            # 每条消息最多包含 20 个 @ 用户，控制单次消息大小。
            for start in range(0, len(users), 20):
                message: list[dict[str, Any]] = [text_segment(f"{name}，集合啦：")]
                for row in users[start : start + 20]:
                    message.extend([text_segment(" "), at_segment(row["user_id"])])
                await send_group(group_id, message)
            return True

    async def _mimic(
        self, group_id: int, user_id: int, text: str, mentions: list[int], send_group: SendGroup,
        *, entry: GroupMessage | None = None,
    ) -> None:
        """样本数量足够时直接模仿目标用户在本群的说话风格，无需授权。"""
        if not mentions:
            await self._send_text(send_group, group_id, f"用法：{self.settings.mimic_prefix} @某人 [主题]")
            return
        # 一次只模仿首个被 @ 的用户，至少需要 5 条本群普通文字发言。
        target = mentions[0]
        samples = self.db.style_samples(group_id, target, self.settings.style_sample_limit)
        if len(samples) < 5:
            await self._send_text(send_group, group_id, "可用样本不足，至少需要 5 条本群普通文字发言（不含指令）。")
            return
        topic = text.replace(self.settings.mimic_prefix, "", 1).replace(f"@{target}", "").strip() or "随便聊两句"
        # 每条样本截取前 300 字，减少提示词长度和模型调用成本。
        sample_text = "\n".join(f"- {sample[:300]}" for sample in samples)
        system = """根据提供的群聊文字样本，提炼语言风格并生成一句新话。
不要声称自己就是该用户，不复述隐私信息，不模仿敏感身份特征，不输出分析过程。只输出生成的话，最多 100 字。"""
        await self._ai_reply(
            group_id,
            user_id,
            f"发言样本：\n{sample_text}\n\n主题：{topic}",
            send_group,
            system=system,
            source_message=entry,
        )

    async def _ai_reply(
        self,
        context_id: int,
        user_id: int,
        prompt: str,
        send_message: SendMessage,
        *,
        history: list[dict[str, str]] | None = None,
        context: ContextPacket | None = None,
        system: str | None = None,
        private: bool = False,
        automatic: bool = False,
        image_session: ChatImageSession | None = None,
        observe_only: bool = False,
        source_message: GroupMessage | None = None,
        private_source_id: int = 0,
    ) -> None:
        """检查冷却及并发名额，然后调度后台 AI 请求，不等待模型生成。"""
        if self._closed:
            return
        scope = "image" if observe_only else ("auto" if automatic else ("private" if private else "group"))
        key = (scope, context_id, user_id)
        if automatic:
            # 最多一个自动请求；尽量留一个总名额给手动请求，不对群友发送“忙碌”提示。
            if len(self._ai_tasks) >= max(1, self.settings.ai_max_concurrency - 1):
                return
            if any(
                task_key[0] in {"auto", "image"} or (task_key[0] == "group" and task_key[1] == context_id)
                for _, task_key in self._ai_tasks.values()
            ):
                return
        if key in self._pending_ai_users:
            if not automatic:
                await self._send_text(send_message, context_id, "你的上一条 AI 请求还没完成，请稍等一下。")
            return
        # 单调时钟适合计算时间间隔，不受系统日期调整影响。
        now = time.monotonic()
        cooldown = (self.settings.sticker_collect_observe_seconds if observe_only else
                    (self.settings.auto_chat_cooldown_seconds if automatic else self.settings.ai_cooldown_seconds))
        wait = cooldown - (now - self._last_ai_call.get(key, float("-inf")))
        if wait > 0:
            if not automatic:
                await self._send_text(send_message, context_id, f"慢一点，{wait:.0f} 秒后再问我。")
            return
        if len(self._ai_tasks) >= self.settings.ai_max_concurrency or self._ai_slots.locked():
            if not automatic:
                await self._send_text(send_message, context_id, "现在 AI 请求较多，请稍后再试。")
            return
        # 在请求前记录时间，失败的请求同样计入冷却，避免连续重试。
        self._last_ai_call[key] = now
        self._pending_ai_users.add(key)
        # 注册前不 await，名额检查和占用在同一事件循环中不会被其他事件打断。
        # 只有后台Agent使用确认式发送；入口的同步命令不能等待WS响应，否则接收循环会死锁。
        task_sender = self._confirmed_senders.get(send_message, send_message)
        task = asyncio.create_task(
            self._run_limited_ai_reply(context_id, user_id, prompt, task_sender, history=history, system=system, context=context,
                               private=private, automatic=automatic, image_session=image_session,
                               observe_only=observe_only, source_message=source_message, private_source_id=private_source_id),
            name=f"ai-reply-{key[0]}-{context_id}-{user_id}",
        )
        self._ai_tasks[task] = (task_sender, key)
        task.add_done_callback(self._finish_ai_task)

    async def _run_limited_ai_reply(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("private"):
            async with self._ai_slots:
                await self._run_ai_reply(*args, **kwargs)
        else:
            async with self._group_run_locks[args[0]]:
                async with self._ai_slots:
                    await self._run_ai_reply(*args, **kwargs)

    async def _run_ai_reply(
        self,
        context_id: int,
        user_id: int,
        prompt: str,
        send_message: SendMessage,
        *,
        history: list[dict[str, str]] | None = None,
        context: ContextPacket | None = None,
        system: str | None = None,
        private: bool = False,
        automatic: bool = False,
        image_session: ChatImageSession | None = None,
        observe_only: bool = False,
        batch: GroupBatch | None = None,
        source_message: GroupMessage | None = None,
        mention_targets: dict[int, str] | None = None,
        collection_jobs: dict[str, CollectionJob] | None = None,
        private_source_id: int = 0,
        recalled_memories: list[dict] | None = None,
    ) -> AgentOutcome | None:
        """后台生成并发送回复；私聊回复入库，运行日志不记录文本、样本或凭证。"""
        started = time.monotonic()
        scope = "image" if observe_only else ("auto" if automatic else ("private" if private else "group"))
        registry = ToolRegistry()
        registry.allow_silent = automatic
        registry.batch_completion_enabled = self.settings.agent_batch_completion_enabled
        workspace = None
        sticker_session = None
        injected_memories = list(recalled_memories or [])
        if self.settings.memory_enabled and not observe_only and (batch is not None or private):
            namespace = -user_id if private else context_id
            new_source_ids = ({entry.storage_id for entry in batch.entries if not entry.poke}
                              if batch else ({private_source_id} if private_source_id else set()))
            context_ids = set()
            if batch:
                for entry in batch.entries:
                    quoted = self.db.group_context_message(namespace, entry.reply_to) if entry.reply_to else None
                    if quoted and quoted["id"] < min(new_source_ids, default=0):
                        context_ids.add(quoted["id"])
            elif private_source_id:
                with self.db._lock:
                    context_ids = {r[0] for r in self.db.connection.execute("SELECT id FROM private_messages WHERE user_id=? AND id<? ORDER BY id DESC LIMIT 4", (user_id, private_source_id))}
            context_ids -= self.db.memories.blocked_sources(namespace)
            workspace = self.memory_capture.open_workspace(namespace, new_source_ids, context_ids)
            registry.task_evidence = workspace
            if image_session is None and self.settings.chat_image_enabled and self.image_cache.enabled:
                image_session = ChatImageSession([], image_ids=[])
        try:
            context = context or ContextPacket(task={"用户请求": prompt})
            if private and self.settings.working_memory_enabled:
                boundary = private_source_id
                if not boundary:
                    with self.db._lock:
                        boundary = self.db.connection.execute("SELECT coalesce(max(id),0)+1 FROM private_messages WHERE user_id=?", (user_id,)).fetchone()[0]
                context.working = self.working_memory.snapshot(-user_id, before_id=boundary, query=prompt)
            if private and self.settings.memory_enabled:
                recall_query = memory_context_query(prompt, context, history=history) if self.settings.memory_context_query_enabled else ''
                private_memories = await self.memory.retrieve(-user_id, prompt, user_ids={user_id}, boundary=private_source_id or None,
                    **({'context_query': recall_query} if recall_query else {}))
                injected_memories.extend(private_memories)
            allowed_mentions = dict(mention_targets or {}) if not private else {}
            if source_message is not None and not private:
                allowed_mentions[source_message.user_id] = source_message.nickname
                allowed_mentions.update({member["user_id"]: member["display_name"] for member in
                                         self.db.group_members(context_id, set(source_message.mentions))})
            bot_id = batch.focus.self_id if batch else (source_message.self_id if source_message else 0)
            if (self.dialogue.enabled and batch is not None and batch.thread_id is not None
                    and not private and not observe_only):
                thread_history = ThreadHistorySession(
                    self.settings, self.dialogue.store, context_id, batch.thread_id,
                    before_id=min(entry.storage_id for entry in batch.entries), bot_id=bot_id,
                    visible_source_ids=[row['本地记录ID'] for row in context.working.get('近期往返', [])],
                    complete_source_ids=[row['本地记录ID'] for row in context.working.get('近期往返', [])
                                         if not row.get('原文已截断')],
                    current_source_ids=[entry.storage_id for entry in batch.entries], dialogue=self.dialogue)
                registry.register(READ_THREAD_HISTORY_TOOL, thread_history.execute)
                context.runtime['当前thread历史查询'] = thread_history.runtime()
            allowed_mentions = {qq: name for qq, name in allowed_mentions.items() if qq > 0 and qq != bot_id}
            preserve_layout = not private and (batch is not None or not automatic) and self._needs_multiline_layout(prompt)
            conversation = ("private", user_id) if private else ("group", context_id)
            sticker_context = {}
            def record_thread_reply(source_id: int, reply_to: str) -> None:
                if self.dialogue.enabled and batch is not None and batch.thread_id is not None:
                    # 戳一戳没有QQ消息号，仍通过本地来源绑定真实回复，不能丢失往返。
                    parent = next((entry for entry in batch.entries if reply_to and entry.message_id == reply_to), batch.focus)
                    self.dialogue.store.assign(context_id, source_id, batch.thread_id,
                                               parent=parent.storage_id, method='bot')
                    self.dialogue.store.mark(context_id, [parent.storage_id], 'resolved')
            messages = MessageSession(
                registry, self.db, send_message, context_id, private=private,
                automatic=automatic, preserve_layout=preserve_layout,
                lock=self._group_reply_locks[context_id],
                reply_targets=batch.reply_targets if batch else ({source_message.message_id} if source_message else None),
                default_reply=batch.focus.message_id if batch else (source_message.message_id if source_message else ""),
                bot_id=bot_id, mention_targets=set(allowed_mentions),
                max_chars=self.settings.group_reply_max_chars,
                max_messages=(2 if batch.direct else 1) if batch else None,
                on_group_reply=record_thread_reply,
            )
            async def send_text(arguments: dict[str, Any]) -> dict[str, Any]:
                if observe_only:
                    return {"ok": False, "error": "图片收藏已由后台独立处理，本轮观察不向群发送文字。"}
                return await messages.execute(arguments)
            send_definition = copy.deepcopy(SEND_MESSAGE_TOOL)
            if private:
                send_definition["function"]["parameters"]["properties"].pop("at_user_ids")
                send_definition["function"]["parameters"]["properties"].pop("reply_to_message_id")
            registry.register(send_definition, send_text)

            poke_sender = self._poke_senders.get(send_message)
            if (self.settings.poke_enabled and poke_sender is not None and not observe_only
                    and (batch is not None or private)):
                poke_session = PokeSession(
                    self.pokes, messages, poke_sender, default_user_id=user_id,
                    allowed_targets={user_id} if private else set(allowed_mentions),
                )
                registry.register(SEND_POKE_TOOL, poke_session.execute)

            voice_transport = self._voice_transports.get(send_message)
            if (self.settings.voice_enabled and voice_transport is not None and batch is not None
                    and not private and not observe_only):
                voice_session = VoiceSession(self.voices, messages, voice_transport)
                registry.register(LIST_VOICE_CHARACTERS_TOOL, voice_session.list_characters)
                registry.register(SEND_VOICE_TOOL, voice_session.execute)
                context.runtime["语音约束"] = {"字数上限": min(self.settings.voice_max_chars, self.settings.group_reply_max_chars),
                                              "冷却秒数": self.settings.voice_cooldown_seconds}

            if self.settings.web_search_enabled and not observe_only and (batch is not None or private):
                # 仅本批可回复消息/当前私聊文本中的URL；不授权历史、记忆或下一批链接。
                current_texts = [entry.text for entry in batch.candidates] if batch else [prompt]
                web_session = WebSession(allowed_urls=user_urls(current_texts))
                for definition in WEB_TOOLS:
                    name = definition["function"]["name"]
                    async def web_handler(arguments: dict[str, Any], tool_name: str = name) -> dict[str, Any]:
                        return await self.web_search.execute_tool(web_session, tool_name, arguments)
                    registry.register(definition, web_handler)
                context.runtime["搜索约束"] = {"搜索次数": self.settings.web_search_max_calls,
                    "每次结果数": self.settings.web_search_max_results, "need_content": self.settings.web_search_need_content,
                    "搜索正文总字数": self.settings.web_search_content_max_chars, "每页字数": self.settings.web_fetch_max_chars,
                    "额外Fetch次数（含失败，不含缓存命中）": self.settings.web_fetch_max_calls}

            async def finish(arguments: dict[str, Any]) -> dict[str, Any]:
                if not automatic and not registry.output_attempted:
                    return {"ok": False, "error": "请先通过发送工具回应当前用户，再finish；不要只在内部口头回答。"}
                if (sticker_session is not None and sticker_session.cadence_due
                        and not sticker_session.cadence_decided and not sticker_session.sent
                        and messages.count > 0 and not registry.delivery_uncertain):
                    return {"ok": False, "error": "本轮已到表情间隔，请选贴切图片send_sticker再收尾；无合适候选或语境不宜时调用skip_scheduled_sticker说明原因。不要重复发送文字。"}
                registry.wait_for_messages = False
                return {"ok": True, "finished": True, "active": False}
            registry.register(FINISH_TOOL, finish, terminal=True)
            if batch is not None:
                async def wait_for_messages(arguments: dict[str, Any]) -> dict[str, Any]:
                    result = await finish({})
                    if result.get("ok") is not True:
                        return result
                    registry.wait_for_messages = True
                    return {"ok": True, "waiting": True, "note": "本批结束，保持当前交流焦点等待新消息；切换与超时由程序控制。"}
                registry.register(WAIT_FOR_MESSAGES_TOOL, wait_for_messages, terminal=True)

            if self.settings.memory_enabled and not observe_only and (batch is not None or private):
                namespace = -user_id if private else context_id
                source_rows = self.db.memory_source_rows(namespace, workspace.source_ids if workspace else set())
                def grant_memory_image(ref_id):
                    row = self.image_cache._row(namespace, ref_id)
                    if not row or row["source_id"] > max(workspace.source_ids, default=0):
                        return
                    image_id, ref = f"media_{ref_id}", self.image_cache.reference(row)
                    if image_session is not None and image_id not in image_session.available:
                        if len(image_session.images) >= 24:
                            return
                        if not image_session.image_ids:
                            image_session.image_ids = list(image_session.available)
                        image_session.images.append(ref)
                        image_session.image_ids.append(image_id)
                        image_session.sources[image_id] = self.image_cache.describe(row)
                    if batch is not None and collection_jobs is not None:
                        collection_jobs[image_id] = CollectionJob(namespace, row["user_id"], bot_id, row["message_id"], ref, send_message)
                async def working_consolidate():
                    return await self.working_memory.archive(namespace, max(workspace.source_ids, default=0) + 1)
                register_memory_tools(registry, self.memory_capture, namespace, user_id, source_rows,
                    workspace=workspace, media_grant=grant_memory_image,
                    working_consolidate=working_consolidate)
                for item in injected_memories:
                    if type(item.get("media_ref_id")) is int:
                        grant_memory_image(item["media_ref_id"])
                    for media_id in item.get("关联图片", []):
                        grant_memory_image(int(media_id[6:]))

            if image_session is not None:
                async def get_image(arguments: dict[str, Any]) -> dict[str, Any]:
                    return await self.chat_images.execute_tool(image_session, arguments)
                registry.register(CHAT_IMAGE_TOOL, get_image)
                logger.info("Chat images attached: scope=%s context=%s user=%s count=%s", scope, context_id, user_id, len(image_session.images))

            if (batch is not None and not private and not observe_only and collection_jobs is not None
                    and self.sticker_collector.enabled):
                collection_session = CollectionSession(collection_jobs)
                async def collect_sticker(arguments: dict[str, Any]) -> dict[str, Any]:
                    result = await self.sticker_collector.execute_tool(
                        collection_session, arguments,
                        loaded_images=image_session.loaded_images if image_session is not None else None,
                    )
                    if result.get("created") is True:
                        # 审核落盘后才给确认；后续模型失败时复用现有安全回执兜底。
                        if "这图我收下了。" not in registry.receipts:
                            registry.receipts.append("这图我收下了。")
                    return result
                registry.register(COLLECT_STICKER_TOOL, collect_sticker)

            if self.settings.sticker_enabled and not observe_only and (system is None or private or automatic):
                sticker_session = StickerSession(conversation)
                cadence = self.stickers.reply_cadence(sticker_session)
                if cadence:
                    context.runtime["表情发送节奏"] = cadence
                if sticker_session.cadence_due:
                    async def skip_scheduled_sticker(arguments: dict[str, Any]) -> dict[str, Any]:
                        if not arguments["reason"].strip():
                            return {"ok": False, "error": "请说明本轮没有适合表情的原因。"}
                        sticker_session.cadence_decided = True
                        return {"ok": True, "skipped": True, "note": "本轮延后，下次实际回复时仍优先选图；不需要发送额外解释。"}
                    registry.register({"type": "function", "function": {
                        "name": "skip_scheduled_sticker", "description": "定期表情到点，但语境不适合或没有合适可用图片时延后；不向QQ发送内容。",
                        "parameters": {"type": "object", "properties": {
                            "reason": {"type": "string", "minLength": 1, "maxLength": 120}},
                            "required": ["reason"], "additionalProperties": False},
                    }}, skip_scheduled_sticker)
                sticker_context = await self.stickers.context_catalog(
                    sticker_session, query=batch.focus.text if batch is not None else prompt[:80],
                )
                if not self.stickers.auto_index_enabled:
                    context.runtime["图库标注"] = "本地自动标注关闭；人工图库按文件名和人工描述，收藏仍独立审核"
                sticker_sent = False
                async def sticker_action(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    nonlocal sticker_sent
                    if name != "send_sticker":
                        return await self.stickers.execute_tool(sticker_session, name, arguments)
                    if registry.delivery_uncertain:
                        return {"ok": False, "error": "上一条发送未确认，本轮不再发送。"}
                    if sticker_sent:
                        return {"ok": False, "error": "本轮表情已发送，最多一张；可用wait_for_messages等待后续，或finish退出跟进。"}
                    try:
                        reply_to = messages.resolve_reply(arguments)
                    except ValueError as exc:
                        return {"ok": False, "error": str(exc)}
                    sticker_args = {key: value for key, value in arguments.items() if key != "reply_to_message_id"}
                    sticker_session.cadence_decided = True
                    result = await self.stickers.execute_tool(sticker_session, name, sticker_args)
                    if result.get("ok") is not True:
                        return result
                    async def confirmed_sender(target: int, segments: list[dict[str, Any]]) -> None:
                        registry.output_attempted = True
                        registry.delivery_uncertain = True
                        messages.selected_reply = reply_to
                        quote_to = arguments.get("reply_to_message_id", "")
                        segments = ([reply_segment(quote_to)] if quote_to and not private else []) + segments
                        sent_id = await send_message(target, segments)
                        registry.delivery_uncertain = False
                        registry.outputs_sent += 1
                        messages.last_reply = reply_to
                        if not private:
                            messages.log_group_reply(f"[表情包：{sticker_session.selected_id}]",
                                                     message_id=normalize_message_id(sent_id), reply_to=reply_to)
                    async with self._group_reply_locks[context_id]:
                        submitted = await self.stickers.send_selected(sticker_session, context_id, confirmed_sender)
                    if not submitted:
                        return {"ok": False, "error": "图片已变化、被禁用或触发冷却，未发送；可以用文字。"}
                    sticker_sent = True
                    if private:
                        self.db.log_private_message(user_id, "assistant", f"[表情包：{sticker_session.selected_id}]")
                    return {"ok": True, "sent": True, "sticker_id": sticker_session.selected_id,
                            "note": "NapCat接受发送，不代表用户已读；如有wait_for_messages可等待后续，不重复输出发送确认。"}
                for definition in STICKER_TOOLS:
                    name = definition["function"]["name"]
                    if name == "send_sticker" and batch is not None:
                        definition = copy.deepcopy(definition)
                        definition["function"]["parameters"]["properties"]["reply_to_message_id"] = (
                            SEND_MESSAGE_TOOL["function"]["parameters"]["properties"]["reply_to_message_id"]
                        )
                    if name == "get_sticker_image" and not self.settings.sticker_vision_enabled:
                        continue
                    async def sticker_handler(arguments: dict[str, Any], name: str = name) -> dict[str, Any]:
                        return await sticker_action(name, arguments)
                    registry.register(definition, sticker_handler)

            if private and self.reminders is not None:
                for definition in REMINDER_TOOLS:
                    name = definition["function"]["name"]
                    async def reminder_handler(arguments: dict[str, Any], name: str = name) -> dict[str, Any]:
                        if registry.output_attempted:
                            return {"ok": False, "error": "备忘操作必须在发送确认之前完成，本轮已经发送，不再改动提醒。"}
                        result = await self.reminders.execute_tool(user_id, name, arguments)
                        if isinstance(result.get("message"), str):
                            registry.receipts.append(result["message"])
                            registry.receipt_error |= result.get("ok") is not True
                        return result
                    registry.register(definition, reminder_handler)

            trigger = "活跃跟进" if automatic else "明确提问"
            if batch is not None:
                trigger = ("戳一戳唤醒" if batch.focus.poke else "明确提问") if batch.direct else (
                    "程序随机接话" if batch.random_trigger else "活跃跟进")
            elif context.task.get("事件类型") == "戳一戳":
                trigger = "戳一戳唤醒"
            scene = PromptScene(
                kind="observe" if observe_only else ("private" if private else ("group_batch" if batch else "group")),
                trigger=trigger, preserve_layout=preserve_layout,
                max_chars=4000 if private or preserve_layout else messages.max_chars,
                max_messages=0 if observe_only else messages.limit,
            )
            context.memories = injected_memories
            context.sticker_catalog = sticker_context
            context.runtime["当前发言人QQ"] = user_id
            context.runtime["允许@成员"] = [{"QQ": qq, "昵称": name} for qq, name in sorted(allowed_mentions.items())]
            context.runtime["允许引用回复的消息号"] = sorted(messages.reply_targets) if not private else []
            context.runtime["可看图片编号"] = list(image_session.available) if image_session is not None else []
            context.runtime["可收藏图片编号"] = list(collection_jobs or {}) if "collect_sticker" in registry.bindings else []
            if image_session is not None:
                known = {item["图片编号"] for item in context.images}
                context.images.extend(
                    image_session.sources.get(image_id, {"图片编号": image_id})
                    for image_id in image_session.available if image_id not in known
                )
            if self.settings.tool_search_enabled and not observe_only and (system is None or private or automatic):
                initial_tools = ([definition["function"]["name"] for definition in STICKER_TOOLS]
                                 if sticker_session is not None and sticker_session.cadence_due else [])
                enable_tool_search(registry, self.tool_retriever, max_calls=self.settings.tool_search_max_calls,
                                   top_k=self.settings.tool_search_top_k, initial_tools=initial_tools)
            persona = system or SYSTEM_PROMPT
            if system is None and not observe_only and self.settings.persona_details_enabled:
                persona, selected_details = await self.persona_selector.select(prompt,
                    focus_background(context, history=history, limit=2, assistant_only=True))
                logger.info('Persona selected: details=%s', ','.join(selected_details) or 'core')
            outcome = await self.agent.run(prompt, registry, history=history, system=persona,
                                           context=context, scene=scene)
            if outcome.error and not registry.output_attempted and not automatic:
                await registry.execute("send_message", {"text": "AI暂时没处理完，请稍后重试。"})
            logger.info("Agent handled: scope=%s context=%s user=%s sent=%s elapsed=%.2fs",
                        scope, context_id, user_id, registry.outputs_sent, time.monotonic() - started)
            return outcome
        except Exception:
            logger.exception("Background agent failed: scope=%s context=%s user=%s", scope, context_id, user_id)
        finally:
            if sticker_session is not None:
                self.stickers.finish_reply(sticker_session, replied=messages.count > 0)
            if workspace is not None:
                try:
                    self.memory_capture.close_workspace(workspace)
                except Exception as exc:
                    logger.warning("Memory task handoff deferred: error=%s", type(exc).__name__)
            if automatic:
                self._last_ai_call[(scope, context_id, user_id)] = time.monotonic()

    async def _announce_collected(self, job: CollectionJob) -> None:
        """已经成功保存才生成人设短句；同样等待发送确认并写入自身历史。"""
        try:
            text = await self.llm.complete(
                "程序已确认：刚收到的一张群友图片已通过审核，新收藏到共享表情库。"
                "只生成一句自然的偷图成功短话，例如‘这图我拿走了’。不要再决定是否收藏，"
                "不编造画面、不@任何人，不输出工具或解释，不超过20字。",
                system=PromptBuilder.text_system(),
            )
            text = text.strip().splitlines()[0][:min(100, self.settings.group_reply_max_chars)]
            if not text or any(mark in text for mark in ("[CQ:", "[分条]", "[不插话]")):
                text = "这图我拿走了"
        except Exception:
            text = "这图我拿走了"
        registry = ToolRegistry()
        messages = MessageSession(
            registry, self.db, job.sender, job.group_id, private=False, automatic=True,
            preserve_layout=False, lock=self._group_reply_locks[job.group_id],
            reply_targets={job.message_id} if job.message_id else set(), default_reply=job.message_id,
            bot_id=job.bot_id, max_chars=self.settings.group_reply_max_chars, max_messages=1,
        )
        result = await messages.execute({"text": text})
        logger.info("Sticker collection announcement: group=%s sent=%s", job.group_id, result.get("sent", 0))

    @staticmethod
    def _needs_multiline_layout(prompt: str) -> bool:
        """根据当前请求决定是否保留排版，不让历史话题开启长文模式。"""
        return bool(re.search(
            r"详细|展开|长文|完整.{0,6}(代码|脚本|程序)|"
            r"(?:写|给|生成|提供).{0,10}(代码|脚本|程序|诗|故事|作文|文章|歌词|列表|清单|表格)|"
            r"作诗|多行|换行|分步骤|教程",
            prompt, re.IGNORECASE,
        ))

    @staticmethod
    async def _send_text(send_message: SendMessage, target_id: int, text: str) -> None:
        """通过对应发送回调向群或私聊发送纯文本。"""
        await send_message(target_id, [text_segment(text)])
