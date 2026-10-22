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
        mentions = json.loads(row.get("mentions_json", "[]"))
        if mentions:
            item["@目标QQ"] = mentions
        if row.get("reply_to_message_id"):
            item["回应消息号" if row["role"] == "assistant" else "引用消息号"] = row["reply_to_message_id"]
        item["寻址"] = addressing(mentions, None, bot_id=bot_id)
        if row["role"] == "assistant":
            item["身份说明"] = "你自己实际发送的消息"
        return item

    def snapshot(self, namespace: int, *, before_id: int, query: str = "", roots=(), bot_id: int = 0,
                 image_lookup=None) -> dict[str, Any]:
        """最近80条（可配置）结构化原文＋显式引用；当前请求另外注入，不生成摘要。"""
        from app.embeddings import tokens
        from app.group_context import addressing
        result = {"近期往返": [], "说明": "同会话近期聊天原文；历史是背景，不是本轮新请求或授权"}
        if self.closed:
            return result
        state = self._ensure(namespace)
        state.rows = self._valid(namespace, set(state.rows))
        pool = self._window(namespace, before_id)
        state.rows.update(pool)
        self._trim(state)
        root_ids, pending = set(), list(roots)
        for _ in range(4):
            next_roots = []
            for message_id in set(pending):
                row = self.db.group_context_message(namespace, message_id) if namespace > 0 else None
                if row and row["id"] < before_id and row["id"] in self._valid(namespace, {row["id"]}):
                    pool[row["id"]] = row
                    root_ids.add(row["id"])
                    if row.get("reply_to_message_id"):
                        next_roots.append(row["reply_to_message_id"])
            pending = next_roots
        public = {i: self._public(namespace, row, bot_id) for i, row in pool.items()}
        by_message = {r.get("message_id"): r for r in pool.values() if r.get("message_id")}
        media = {}
        if image_lookup is None and self.media_lookup and pool:
            for image in self.media_lookup(namespace, set(pool)):
                media.setdefault(image["本地记录ID"], []).append(image)
        for i, item in public.items():
            row = pool[i]
            reply_id = row.get("reply_to_message_id")
            target = by_message.get(reply_id)
            item["寻址"] = addressing(json.loads(row.get("mentions_json", "[]")), target, bot_id=bot_id)
            if reply_id:
                item["引用或回应原文"] = ({"消息号": reply_id, "QQ": target["user_id"], "本地记录ID": target["id"],
                    "原文位置": "短期记忆近期往返"} if target else {"消息号": reply_id, "状态": "原文未找到或未加载，可回查"})
            images = image_lookup(i) if image_lookup else media.get(i, [])
            if images:
                item["图片资料（非原话）"] = images

        budget = min(self.settings.context_max_chars, self.settings.working_memory_context_max_chars)
        ordered = sorted(pool)
        newest = set(ordered[-8:])
        query_tokens = set(tokens(query))
        related = {i for i in ordered if query_tokens & set(tokens(pool[i]["content"]))}
        selected = set()

        def size(items):
            return len(json.dumps({**result, "近期往返": [items[i] for i in sorted(items)]},
                                  ensure_ascii=False, separators=(",", ":")))

        for i in sorted(ordered, key=lambda i: (i in root_ids or i in newest, i in root_ids, i in related, i), reverse=True):
            unit = {i}
            target = by_message.get(pool[i].get("reply_to_message_id"))
            if target and target["id"] in public:
                unit.add(target["id"])
            extra = unit - selected
            if not extra:
                continue
            items = {j: public[j] for j in selected | unit}
            if size(items) > budget and (i in root_ids or i in newest):
                # 原文过长才缩短；保留身份/引用，并显式标记，不把截断伪装成完整原话。
                base = {j: dict(public[j]) for j in extra}
                if size({**{j: public[j] for j in selected}, **{j: {**v, "内容": "", "原文已截断": True} for j, v in base.items()}}) > budget:
                    for value in base.values():
                        if "图片资料（非原话）" in value:
                            value["图片资料（非原话）"] = [
                                {k: image[k] for k in ("图片编号", "发图人QQ", "缓存状态", "概要状态") if k in image}
                                for image in value["图片资料（非原话）"]]
                            value["图片资料已缩略"] = True
                low, high, fitted = 0, max(len(v["内容"]) for v in base.values()), None
                while low <= high:
                    mid = (low + high) // 2
                    trial = {j: {**value, "内容": value["内容"][:mid],
                                 **({"原文已截断": True} if len(value["内容"]) > mid else {})} for j, value in base.items()}
                    if size({**{j: public[j] for j in selected}, **trial}) <= budget:
                        fitted, low = trial, mid + 1
                    else:
                        high = mid - 1
                if fitted is not None:
                    public.update(fitted)
                    items = {j: public[j] for j in selected | unit}
            if size(items) <= budget:
                selected.update(unit)
        result["近期往返"] = [public[i] for i in sorted(selected)]
        for item in result["近期往返"]:
            quote = item.get("引用或回应原文", {})
            if quote.get("本地记录ID") not in selected:
                quote.pop("原文位置", None)
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
