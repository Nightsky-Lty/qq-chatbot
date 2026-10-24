"""工作记忆：最近聊天原文窗口；不调用LLM摘要，不保存推理/工具链。"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemory:
    namespace: int
    rows: dict[int, dict] = field(default_factory=dict)
    touched_at: float = 0


class WorkingMemoryManager:
    def __init__(self, settings, db, capture, *, clock=time.time):
        self.settings, self.db, self.capture = settings, db, capture
        self.clock = clock
        self.scopes: OrderedDict[int, WorkingMemory] = OrderedDict()
        self.task: asyncio.Task | None = None
        self.closed = False
        self.media_lookup = None

    def start(self):
        if not self.closed and self.settings.working_memory_enabled and self.task is None:
            self.task = asyncio.create_task(self._run(), name="working-memory")

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
        self.scopes.clear()

    def _valid(self, namespace: int, ids: set[int]) -> dict[int, dict]:
        blocked = self.db.memories.blocked_sources(namespace)
        # TTL只淘汰内存缓存；原文可见性由数据库保留策略和遗忘阻断控制。
        return {r["id"]: r for r in self.db.memory_source_rows(namespace, ids)
                if r["id"] not in blocked and r["role"] in {"user", "assistant", "notice"}}

    def _window(self, namespace: int, before_id: int | None = None) -> dict[int, dict]:
        table, key, target = ("messages", "group_id", namespace) if namespace > 0 else ("private_messages", "user_id", -namespace)
        limit = self.settings.context_message_limit
        boundary = before_id if before_id is not None else 9223372036854775807
        found = {}
        # 分页补足被遗忘/删除排除的记录；不让当前批次或之后的消息挤掉历史窗口。
        while len(found) < limit:
            with self.db._lock:
                ids = [r[0] for r in self.db.connection.execute(
                    f"SELECT id FROM {table} WHERE {key}=? AND id<? AND role IN ('user','assistant','notice') ORDER BY id DESC LIMIT ?",
                    (target, boundary, limit))]
            if not ids:
                break
            valid = self._valid(namespace, set(ids))
            for source in ids:
                if source in valid and len(found) < limit:
                    found[source] = valid[source]
            boundary = ids[-1]
        return found

    def _ensure(self, namespace: int) -> WorkingMemory:
        state = self.scopes.get(namespace)
        if state is None:
            state = WorkingMemory(namespace, touched_at=self.clock())
            self.scopes[namespace] = state
        self.scopes.move_to_end(namespace)
        while len(self.scopes) > self.settings.working_memory_max_scopes:
            self.scopes.popitem(last=False)
        return state

    def _trim(self, state):
        for source in sorted(state.rows)[:-self.settings.working_memory_max_messages]:
            state.rows.pop(source)

    def receive(self, namespace: int, source_id: int):
        """真实消息提交后增量登记；普通群友互聊和成功发送的机器人回复都进入。"""
        if self.closed or not self.settings.working_memory_enabled:
            return
        state = self._ensure(namespace)
        state.rows.update(self._valid(namespace, {source_id}))
        state.touched_at = self.clock()
        self._trim(state)

    @staticmethod
    def _public(namespace: int, row: dict, bot_id: int = 0) -> dict:
        from app.group_context import addressing
        user = row.get("user_id", -namespace if row["role"] == "user" else bot_id)
        item = {"本地记录ID": row["id"], "消息号": row.get("message_id", ""),
                "QQ": user, "昵称": row.get("nickname", str(user)), "角色": row["role"],
                "内容": row["content"], "时间": datetime.fromtimestamp(
                    row["created_at"], timezone(timedelta(hours=8))).isoformat()}
        parts = json.loads(row.get("parts_json", "[]"))
        if parts:
            item["消息段"] = parts
        mentions = json.loads(row.get("mentions_json", "[]"))
        if mentions:
            item["@目标QQ"] = mentions
        if row.get("reply_to_message_id"):
            item["回应消息号" if row["role"] == "assistant" else "引用消息号"] = row["reply_to_message_id"]
        item['寻址'] = addressing(mentions, None, bot_id=bot_id)
        if row["role"] == "assistant":
            item["身份说明"] = "你自己实际发送的消息"
        return item

    def snapshot(self, namespace: int, *, before_id: int, query: str = "", roots=(), bot_id: int = 0,
                 image_lookup=None) -> dict[str, Any]:
        """连续近期原文；预算只截取连续尾部，窗口外真实引用单独补充。"""
        from app.chat_transcript import ChatTranscript
        result = {"近期往返": [], "说明": "连续聊天原文，仅作背景；新消息见当前任务"}
        if self.closed:
            return result
        pool = self._window(namespace, before_id)
        if self.settings.working_memory_enabled:
            state = self._ensure(namespace)
            state.rows = self._valid(namespace, set(state.rows))
            state.rows.update(pool)
            self._trim(state)
        public = {i: self._public(namespace, row, bot_id) for i, row in pool.items()}
        media = {}
        if image_lookup is None and self.media_lookup and pool:
            for image in self.media_lookup(namespace, set(pool)):
                media.setdefault(image["本地记录ID"], []).append(image)
        for i, item in public.items():
            images = image_lookup(i) if image_lookup else media.get(i, [])
            if images:
                item["图片资料（非原话）"] = images
        budget = min(self.settings.context_max_chars, self.settings.working_memory_context_max_chars)
        runtime = {'会话类型': 'private' if namespace < 0 else 'group'}
        selected = []
        for i in sorted(public, reverse=True):
            trial = [public[i], *selected]
            if len(ChatTranscript(trial, runtime).render(trial)) > budget:
                if not selected:
                    item = dict(public[i])
                    # 极长的一条保留明确截断标记；历史工具可重新读取完整记录。
                    item.pop('消息段', None)
                    item.pop('图片资料（非原话）', None)
                    item['原文已截断'] = True
                    item['内容'] = item['内容'][:max(0, budget // 8)]
                    selected = [item]
                break
            selected = trial
        result["近期往返"] = selected
        result['_history_before'] = (selected[0]['本地记录ID'] + bool(selected[0].get('原文已截断'))
                                     if selected else before_id)
        # 窗口外仅补真实引用，不做关键词检索或语义关系推断。
        supplements, pending = {}, list(roots)
        selected_ids = {item['本地记录ID'] for item in selected}
        pending.extend(row.get('reply_to_message_id') for row in pool.values() if row.get('reply_to_message_id'))
        for _ in range(4):
            following = []
            for message_id in set(pending):
                row = self.db.group_context_message(namespace, message_id) if namespace > 0 else None
                if not row or row['id'] >= before_id or row['id'] in selected_ids or row['id'] in supplements:
                    continue
                if row['id'] not in self._valid(namespace, {row['id']}):
                    continue
                item = self._public(namespace, row, bot_id)
                if image_lookup:
                    images = image_lookup(row['id'])
                else:
                    images = self.media_lookup(namespace, {row['id']}) if self.media_lookup else []
                if images:
                    item['图片资料（非原话）'] = images
                trial = [*selected, *supplements.values(), item]
                if len(ChatTranscript(trial, runtime).render(trial)) <= budget:
                    supplements[row['id']] = item
                    if row.get('reply_to_message_id'):
                        following.append(row['reply_to_message_id'])
            pending = following
        if supplements:
            result['引用补充'] = [supplements[i] for i in sorted(supplements)]
        by_message = {pool[i].get('message_id'): item for item in selected for i in [item['本地记录ID']]}
        by_message.update({item['消息号']: item for item in supplements.values() if item['消息号']})
        for item in [*selected, *supplements.values()]:
            reply = item.get('回应消息号') or item.get('引用消息号')
            if reply and reply in by_message:
                parent = by_message[reply]
                item['引用或回应原文'] = {'本地记录ID': parent['本地记录ID'], '消息号': reply,
                                      'QQ': parent['QQ'], '原文位置': '已提供原文'}
                from app.group_context import addressing
                item['寻址'] = addressing(item.get('@目标QQ', []),
                                        {'role': parent['角色'], 'user_id': parent['QQ']}, bot_id=bot_id)
        return result

    async def archive(self, namespace: int, before_id: int) -> bool:
        # 使用边界内真实原话而非摘要；原有长期整理器负责去重、证据检查和筛选。
        if self.closed:
            return False
        return await self.capture.archive_working(namespace, set(self._window(namespace, before_id)))

    async def maintain_once(self):
        for ns, state in list(self.scopes.items()):
            if self.clock() - state.touched_at >= self.settings.working_memory_ttl_seconds:
                await self.capture.archive_working(ns, set(self._valid(ns, set(state.rows))))
                if self.scopes.get(ns) is state and self.clock() - state.touched_at >= self.settings.working_memory_ttl_seconds:
                    self.scopes.pop(ns)

    async def _run(self):
        while True:
            try:
                await self.maintain_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Working memory maintenance deferred: error=%s", type(exc).__name__)
            await asyncio.sleep(2)
