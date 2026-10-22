import asyncio
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.bot import BotService
from app.config import Settings
from app.conversation import GroupBatch, GroupMessage
from app.database import Database
from app.llm import DeepSeekClient
from app.memory import MemoryService
from app.memory_capture import MemoryCapture
from app.memory_tools import register_memory_tools
from app.tools import ToolRegistry
from app.working_memory import WorkingMemoryManager


class Summarizer:
    def __init__(self):
        self.settings = SimpleNamespace(deepseek_api_key="synthetic")
        self.inputs = []
        self.value = None

    async def extract_memories(self, instructions, data):
        self.inputs.append(copy.deepcopy(data))
        if self.value is not None:
            return self.value
        source = data["messages"][0]["本地记录ID"]
        return {"notes": [{"kind": "topic", "text": "QQ11与QQ22在讨论一起练习英语，具体安排待确认", "source_ids": [source]}]}


class WorkingMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.cfg = Settings(_env_file=None, embedding_provider="hash", working_memory_max_messages=32,
            working_memory_recent_messages=4, working_memory_compact_batch=8,
            working_memory_context_max_chars=4000, working_memory_ttl_seconds=60, working_memory_max_scopes=2)
        self.db = Database(Path(self.folder.name) / "db.sqlite")
        self.model = Summarizer()
        self.memory = MemoryService(self.cfg, self.db)
        self.capture = MemoryCapture(self.cfg, self.db, self.memory, self.model)
        self.now = [time.time()]
        self.manager = WorkingMemoryManager(self.cfg, self.db, self.model, self.capture, clock=lambda: self.now[0])
        self.db.message_listener = self.manager.receive

    async def asyncTearDown(self):
        self.db.message_listener = None
        await self.manager.close()
        await self.capture.close()
        await self.memory.close()
        self.db.close()
        self.folder.cleanup()

    def say(self, text, *, namespace=7, user=11, role="user", reply=""):
        if namespace < 0:
            return self.db.log_private_message(-namespace, role, text)
        with self.db._lock:
            message_id = str(self.db.connection.execute("SELECT coalesce(max(id),0)+1 FROM messages").fetchone()[0])
        return self.db.log_message(namespace, user, f"User{user}", text, role=role,
                                   message_id=message_id, reply_to_message_id=reply)

    def snapshot(self, namespace=7, **kwargs):
        return self.manager.snapshot(namespace, before_id=999999, bot_id=90, **kwargs)

    def test_invalid_window_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            Settings(_env_file=None, working_memory_max_messages=32,
                     working_memory_recent_messages=24, working_memory_compact_batch=24)

    async def test_worker_lifecycle_cancels_without_external_request(self):
        self.say("后台不应为少量新消息立即调用模型")
        self.manager.start()
        await asyncio.sleep(0)
        self.assertIsNotNone(self.manager.task)
        self.assertEqual(self.model.inputs, [])
        await self.manager.close()
        self.assertIsNone(self.manager.task)
        self.assertEqual(self.manager.scopes, {})
        self.manager.start()
        self.assertIsNone(self.manager.task)

    async def test_receive_untriggered_multi_user_and_real_bot_reply_without_llm(self):
        first = self.say("今晚一起练英语", user=11)
        self.say("九点吧", user=22, reply=str(first))
        self.say("我也来", user=90, role="assistant", reply=str(first))
        result = self.snapshot()
        rows = result["近期往返"]
        self.assertEqual([r["QQ"] for r in rows], [11, 22, 90])
        self.assertEqual(rows[1]["寻址"]["对象判断"], "其他群友")
        self.assertEqual(rows[-1]["身份说明"], "你自己实际发送的消息")
        self.assertEqual(self.model.inputs, [])

    async def test_scope_isolation_including_identical_group_and_private_numbers(self):
        self.say("群里的事情", namespace=7)
        self.say("私聊的事情", namespace=-7)
        self.assertNotIn("私聊的事情", json.dumps(self.snapshot(7), ensure_ascii=False))
        self.assertNotIn("群里的事情", json.dumps(self.snapshot(-7), ensure_ascii=False))

    async def test_snapshot_rejects_next_batch_including_summarized_future_sources(self):
        first = self.say("较早内容")
        boundary = self.say("本批当前请求")
        future = self.say("下一批不能提前看到")
        state = self.manager.scopes[7]
        state.notes = [{"kind": "topic", "text": "不能提前看到的摘要", "source_ids": [first, future]}]
        result = self.manager.snapshot(7, before_id=boundary)
        self.assertNotIn("本批当前请求", json.dumps(result, ensure_ascii=False))
        self.assertNotIn("不能提前看到", json.dumps(result, ensure_ascii=False))
        # 边界过滤不能把下一批原文从实际工作记忆里删掉。
        self.assertIn(future, state.rows)

    async def test_summary_context_dependencies_prevent_future_and_deleted_input_leaks(self):
        for i in range(12):
            self.say(f"摘要可读的上下文{i}")
        self.assertTrue(await self.manager.compact_once(7))
        note = self.manager.scopes[7].notes[0]
        self.assertEqual(note["source_ids"], [1])
        self.assertIn(8, note["context_source_ids"])
        self.assertEqual(self.manager.snapshot(7, before_id=5)["语境摘要"], [])
        with self.db._lock, self.db.connection:
            # 即使不是模型自报引用，读过的上下文被删也必须让摘要失效。
            self.db.connection.execute("DELETE FROM messages WHERE id=8")
        self.assertEqual(self.snapshot()["语境摘要"], [])

    async def test_compaction_is_background_bounded_and_archives_originals_not_summary(self):
        for i in range(12):
            self.say(f"一起练习英语，第{i}条", user=11 if i % 2 == 0 else 22)
        self.assertTrue(await self.manager.compact_once(7))
        state = self.manager.scopes[7]
        self.assertEqual(len(state.rows), 4)
        self.assertEqual(len(state.notes), 1)
        self.assertIn("具体安排待确认", json.dumps(self.snapshot(), ensure_ascii=False))
        job = self.db.connection.execute("SELECT * FROM memory_consolidation_jobs").fetchone()
        ids = set(json.loads(job["source_ids_json"]))
        self.assertEqual(len(ids), 8)
        self.assertTrue(all(r["content"].startswith("一起练习英语") for r in self.db.memory_source_rows(7, ids)))
        self.assertNotIn("语境摘要", job["source_ids_json"])
        self.assertFalse(await self.manager.compact_once(7))

    async def test_invalid_schema_or_unprovided_sources_keep_raw_and_back_off(self):
        for i in range(12):
            self.say(f"原话{i}")
        for value in ({"notes": [{"kind": "topic", "text": "编造的跨范围证据", "source_ids": [99999]}]},
                      {"notes": "不是数组"}, {"actions": []}):
            self.model.value = value
            self.manager.scopes[7].retry_at = 0
            self.assertFalse(await self.manager.compact_once(7))
            self.assertEqual(len(self.manager.scopes[7].rows), 12)
        calls = len(self.model.inputs)
        self.assertFalse(await self.manager.compact_once(7))
        self.assertEqual(len(self.model.inputs), calls)

    async def test_deleted_and_forgotten_provenance_invalidates_summary_and_raw(self):
        source = self.say("需要清除的消息")
        state = self.manager.scopes[7]
        state.notes = [{"kind": "topic", "text": "需要清除的摘要", "source_ids": [source]}]
        with patch.object(self.db.memories, "blocked_sources", return_value={source}):
            self.assertNotIn("需要清除", json.dumps(self.snapshot(), ensure_ascii=False))
        source = self.say("删除源记录也必须失效")
        state.notes = [{"kind": "topic", "text": "删除后不能复活的摘要", "source_ids": [source]}]
        with self.db._lock, self.db.connection:
            self.db.connection.execute("DELETE FROM messages WHERE id=?", (source,))
        self.assertNotIn("删除", json.dumps(self.snapshot(), ensure_ascii=False))

    async def test_inflight_compaction_cannot_restore_deleted_sources(self):
        for i in range(12):
            self.say(f"原始记录{i}")
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.model.extract_memories
        async def delayed(instructions, data):
            entered.set()
            await release.wait()
            return await original(instructions, data)
        self.model.extract_memories = delayed
        task = asyncio.create_task(self.manager.compact_once(7))
        await entered.wait()
        with self.db._lock, self.db.connection:
            self.db.connection.execute("DELETE FROM messages WHERE id=1")
        release.set()
        self.assertFalse(await task)
        self.assertEqual(self.manager.scopes[7].notes, [])

    async def test_capacity_ttl_and_lru_are_bounded_without_deleting_raw_database(self):
        for i in range(70):
            self.say(f"保留原始记录{i}")
        self.assertEqual(len(self.manager.scopes[7].rows), 32)
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM messages").fetchone()[0], 70)
        self.say("其他群", namespace=8)
        self.say("另一个群", namespace=9)
        self.assertEqual(len(self.manager.scopes), 2)
        self.assertNotIn(7, self.manager.scopes)
        self.now[0] += 61
        await self.manager.maintain_once()
        self.assertEqual(len(self.manager.scopes), 0)
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM messages").fetchone()[0], 72)

    async def test_cold_restore_once_then_receive_incrementally(self):
        first = self.say("第一次收到")
        manager = WorkingMemoryManager(self.cfg, self.db, self.model, self.capture)
        try:
            self.assertEqual(manager.snapshot(7, before_id=first + 1)["近期往返"][0]["内容"], "第一次收到")
            with patch.object(manager, "_ensure", wraps=manager._ensure) as ensure:
                source = self.say("第二次收到")
                manager.receive(7, source)
                result = manager.snapshot(7, before_id=source + 1)
                self.assertEqual(len(result["近期往返"]), 2)
                self.assertEqual(ensure.call_count, 2)
        finally:
            await manager.close()

    async def test_explicit_old_quote_loads_original_with_same_scope_and_boundary(self):
        root = self.say("很早但明确引用的原话")
        for i in range(40):
            self.say(f"无关消息{i}")
        current = self.say("你说的是这个吗", user=22, reply=str(root))
        result = self.manager.snapshot(7, before_id=current, roots=[str(root)])
        self.assertIn("很早但明确引用的原话", json.dumps(result, ensure_ascii=False))
        other = self.manager.snapshot(8, before_id=current, roots=[str(root)])
        self.assertNotIn("很早但明确引用的原话", json.dumps(other, ensure_ascii=False))

    async def test_budget_covers_notes_and_originals_even_when_summary_is_full(self):
        for i in range(12):
            self.say("长原文" * 300)
        state = self.manager.scopes[7]
        state.notes = [{"kind": "topic", "text": "长摘要" * 60, "source_ids": [i]}
                       for i in sorted(state.rows)[:8]]
        self.manager.settings = self.cfg.model_copy(update={"working_memory_context_max_chars": 1000})
        result = self.snapshot()
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))), 1000)

    async def test_archiving_twice_or_after_incremental_capture_does_not_duplicate_jobs(self):
        source = self.say("本人新的具体事件")
        self.assertTrue(await self.capture.archive_working(7, {source}))
        self.assertFalse(await self.capture.archive_working(7, {source}))
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM memory_consolidation_jobs").fetchone()[0], 1)
        source = self.say("旧增量已经处理")
        self.db.connection.execute("UPDATE memory_capture_state SET cursor_id=? WHERE namespace=7", (source,))
        self.db.connection.commit()
        self.assertFalse(await self.capture.archive_working(7, {source}))

    async def test_large_archive_is_batched_and_keeps_fresh_sources_distinct(self):
        sources = {self.say(f"归档的新原话{i}") for i in range(90)}
        self.assertTrue(await self.capture.archive_working(7, sources))
        jobs = self.db.connection.execute("SELECT * FROM memory_consolidation_jobs ORDER BY rowid").fetchall()
        self.assertEqual(len(jobs), 3)
        fresh = [json.loads(job["new_ids_json"]) for job in jobs]
        self.assertTrue(all(len(ids) <= self.cfg.memory_capture_batch_size for ids in fresh))
        self.assertEqual(sum(len(ids) for ids in fresh), len(sources))
        self.assertEqual(set().union(*(set(ids) for ids in fresh)), sources)
        self.assertTrue(set(json.loads(jobs[1]["source_ids_json"])) - set(fresh[1]))

    async def test_summary_has_real_image_metadata_but_not_user_authored_caption(self):
        for i in range(12):
            self.say(f"原始记录{i}")
        self.manager.media_lookup = lambda ns, ids: [{"本地记录ID": min(ids), "图片编号": "media_1",
            "发图人QQ": 11, "概要": "蓝色角色举球棒", "概要状态": "ready"}]
        self.assertTrue(await self.manager.compact_once(7))
        message = self.model.inputs[0]["messages"][0]
        self.assertEqual(message["内容"], "原始记录0")
        self.assertEqual(message["图片资料（非原话）"][0]["概要"], "蓝色角色举球棒")

    async def test_receive_during_idle_archiving_does_not_erase_new_activity(self):
        self.say("旧对话")
        self.now[0] += 61
        entered, release = asyncio.Event(), asyncio.Event()
        async def archive(ns, ids):
            entered.set()
            await release.wait()
            return True
        self.capture.archive_working = archive
        task = asyncio.create_task(self.manager.maintain_once())
        await entered.wait()
        with patch("app.database.time.time", return_value=self.now[0]):
            self.say("等待归档时又有了新消息")
        release.set()
        await task
        self.assertIn(7, self.manager.scopes)
        self.assertIn("又有了新消息", json.dumps(self.snapshot(), ensure_ascii=False))

    async def test_no_key_or_disabled_summary_uses_bounded_raw_fallback(self):
        for i in range(12):
            self.say(f"不要额外调用模型{i}")
        self.model.settings.deepseek_api_key = ""
        self.assertFalse(await self.manager.compact_once(7))
        self.assertEqual(self.model.inputs, [])
        self.assertEqual(len(self.manager.scopes[7].rows), 12)
        self.manager.settings = self.cfg.model_copy(update={"working_memory_summary_enabled": False})
        self.model.settings.deepseek_api_key = "synthetic"
        self.assertFalse(await self.manager.compact_once(7))
        self.assertEqual(self.model.inputs, [])


class WorkingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_working_tool_expands_summary_sources_without_next_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Settings(_env_file=None, embedding_provider="hash", sticker_enabled=False,
                sticker_collect_enabled=False, web_search_enabled=False, voice_enabled=False, poke_enabled=False)
            db = Database(Path(folder) / "db.sqlite")
            llm = DeepSeekClient(cfg)
            bot = BotService(cfg, db, llm)
            try:
                source = db.log_message(7, 11, "甲", "我在准备英语演讲", message_id="1")
                current = db.log_message(7, 22, "乙", "甲刚才说的是什么", message_id="2")
                future = db.log_message(7, 33, "丙", "下一批的私密测试句", message_id="3")
                state = bot.working_memory.scopes[7]
                state.notes = [{"kind": "topic", "text": "QQ11在准备英语演讲", "source_ids": [source]}]
                state.rows.pop(source)
                observed = []
                async def request(messages, temperature, definitions):
                    observed.append(copy.deepcopy(messages))
                    if len(observed) == 1:
                        return {"tool_calls": [{"id": "lookup", "function": {"name": "memory_search",
                            "arguments": '{"query":"英语","layer":"working","include_sources":true}'}}]}
                    return {"tool_calls": [{"id": "send", "function": {"name": "send_message",
                        "arguments": '{"text":"甲说在准备英语演讲","next_action":"finish"}'}}]}
                llm.request_message = request
                batch = GroupBatch([GroupMessage(current, "2", 22, "乙", "甲刚才说的是什么", 90, direct=True)])
                outcome = await bot._run_group_batch_locked(7, batch, AsyncMock(return_value="1000"))
                self.assertEqual(outcome.sent, 1)
                result = json.loads(next(m["content"] for m in observed[1] if m["role"] == "tool"))
                self.assertEqual(result["memories"][0]["来源摘录"][0]["内容"], "我在准备英语演讲")
                self.assertNotIn("下一批的私密测试句", json.dumps(observed, ensure_ascii=False))
                self.assertIn(future, state.rows)
            finally:
                await bot.close()
                await llm.close()
                db.close()

    async def test_cross_reply_continuity_and_legacy_history_is_not_used(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Settings(_env_file=None, embedding_provider="hash", memory_enabled=False,
                sticker_enabled=False, sticker_collect_enabled=False, web_search_enabled=False, voice_enabled=False, poke_enabled=False)
            db = Database(Path(folder) / "db.sqlite")
            llm = DeepSeekClient(cfg)
            bot = BotService(cfg, db, llm)
            requests = []
            async def request(messages, temperature, definitions):
                requests.append(copy.deepcopy(messages))
                return {"tool_calls": [{"id": str(len(requests)), "function": {"name": "send_message",
                    "arguments": '{"text":"我记得刚刚的讨论","next_action":"finish"}'}}]}
            llm.request_message = request
            sender = AsyncMock(return_value="1000")
            try:
                db.log_message(7, 11, "甲", "我和乙约好九点练习英语", message_id="1")
                first = db.log_message(7, 22, "乙", "你知道了吗", message_id="2")
                batch = GroupBatch([GroupMessage(first, "2", 22, "乙", "你知道了吗", 90, direct=True)])
                with patch("app.bot.build_group_history", side_effect=AssertionError("不应重新拼80条历史")):
                    await bot._run_group_batch_locked(7, batch, sender)
                    second = db.log_message(7, 11, "甲", "你刚才说了什么", message_id="3")
                    batch = GroupBatch([GroupMessage(second, "3", 11, "甲", "你刚才说了什么", 90, direct=True)])
                    await bot._run_group_batch_locked(7, batch, sender)
                first_data, second_data = [json.loads(m[1]["content"]) for m in requests]
                self.assertIn("短期记忆", first_data["背景资料"])
                self.assertNotIn("群聊时间线", first_data["背景资料"])
                rows = second_data["背景资料"]["短期记忆"]["近期往返"]
                self.assertTrue(any(r["角色"] == "assistant" and r["内容"] == "我记得刚刚的讨论" for r in rows))
                self.assertIn(7, bot.working_memory.scopes)  # finish不清空跨回复语境。
                self.assertFalse(hasattr(bot.working_memory.scopes[7], "messages"))
                source = db.log_private_message(11, "user", "我的私聊问题")
                await bot._run_ai_reply(11, 11, "我的私聊问题", sender, private=True, private_source_id=source)
                private = json.loads(requests[-1][1]["content"])
                self.assertNotIn("我和乙约好", json.dumps(private, ensure_ascii=False))
            finally:
                await bot.close()
                await llm.close()
                db.close()

    async def test_working_search_returns_short_context_not_just_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Settings(_env_file=None, embedding_provider="hash")
            db = Database(Path(folder) / "db.sqlite")
            memory = MemoryService(cfg, db)
            model = Summarizer()
            capture = MemoryCapture(cfg, db, memory, model)
            manager = WorkingMemoryManager(cfg, db, model, capture)
            db.message_listener = manager.receive
            try:
                source = db.log_private_message(11, "user", "今天在准备英语演讲")
                workspace = capture.open_workspace(-11, {source})
                registry = ToolRegistry()
                register_memory_tools(registry, capture, -11, 11, db.memory_source_rows(-11, {source}),
                    workspace=workspace, working_search=lambda q, a: manager.snapshot(-11, before_id=source + 1, query=q),
                    working_consolidate=lambda: manager.archive(-11, source + 1))
                result = await registry.execute("memory_search", {"layer": "working", "query": "英语"})
                self.assertIn("准备英语演讲", json.dumps(result, ensure_ascii=False))
                self.assertFalse((await registry.execute("memory_search", {"layer": "working", "query": "英语", "subject": 22}))["ok"])
                result = await registry.execute("memory_consolidate", {})
                self.assertEqual(result["stored"], 0)
                self.assertTrue(result["scheduled"])
                capture.close_workspace(workspace)
            finally:
                await manager.close()
                await capture.close()
                await memory.close()
                db.close()


if __name__ == "__main__":
    unittest.main()
