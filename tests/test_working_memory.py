import asyncio
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
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


class Extractor:
    def __init__(self):
        self.inputs = []

    async def extract_memories(self, instructions, data):
        self.inputs.append(copy.deepcopy(data))
        return {"actions": []}


class WorkingMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.cfg = Settings(_env_file=None, embedding_provider="hash", working_memory_max_messages=32,
            working_memory_context_max_chars=24000, working_memory_ttl_seconds=60, working_memory_max_scopes=2)
        self.db = Database(Path(self.folder.name) / "db.sqlite")
        self.model = Extractor()
        self.memory = MemoryService(self.cfg, self.db)
        self.capture = MemoryCapture(self.cfg, self.db, self.memory, self.model)
        self.now = [time.time()]
        self.manager = WorkingMemoryManager(self.cfg, self.db, self.capture, clock=lambda: self.now[0])
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

    def test_default_window_is_80_and_not_summary_mode(self):
        cfg = Settings(_env_file=None)
        self.assertEqual(cfg.context_message_limit, 80)
        self.assertEqual(cfg.working_memory_context_max_chars, 24000)
        self.assertFalse(hasattr(cfg, "working_memory_summary_enabled"))

    async def test_worker_lifecycle_cancels_without_external_request(self):
        self.say("短期记忆不应调用摘要模型")
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
        self.assertNotIn("语境摘要", result)
        self.assertFalse(hasattr(self.manager.scopes[7], "notes"))
        self.assertEqual(self.model.inputs, [])

    async def test_latest_80_are_present_even_when_cache_only_holds_32(self):
        ids = [self.say(f"原话{i}", user=11 if i % 2 else 22) for i in range(100)]
        rows = self.snapshot()["近期往返"]
        self.assertEqual([r["本地记录ID"] for r in rows], ids[-80:])
        self.assertEqual(self.model.inputs, [])

    async def test_message_limit_can_be_configured_and_long_original_is_not_summarized(self):
        self.manager.settings = self.cfg.model_copy(update={"context_message_limit": 20})
        ids = [self.say(f"最近消息{i}") for i in range(30)]
        self.assertEqual([r["本地记录ID"] for r in self.snapshot()["近期往返"]], ids[-20:])
        text = "这是完整的长原话，不能先截成一千字。" * 100
        source = self.say(text)
        row = next(r for r in self.snapshot()["近期往返"] if r["本地记录ID"] == source)
        self.assertEqual(row["内容"], text)
        self.assertNotIn("原文已截断", row)

    async def test_private_window_is_80_and_current_request_is_not_duplicated(self):
        ids = [self.say(f"私聊历史{i}", namespace=-11, role="assistant" if i % 2 else "user") for i in range(90)]
        current = self.say("本轮私聊请求", namespace=-11)
        rows = self.manager.snapshot(-11, before_id=current)["近期往返"]
        self.assertEqual([r["本地记录ID"] for r in rows], ids[-80:])
        self.assertNotIn("本轮私聊请求", json.dumps(rows, ensure_ascii=False))

    async def test_scope_isolation_including_identical_group_and_private_numbers(self):
        self.say("群里的事情", namespace=7)
        self.say("私聊的事情", namespace=-7)
        self.assertNotIn("私聊的事情", json.dumps(self.snapshot(7), ensure_ascii=False))
        self.assertNotIn("群里的事情", json.dumps(self.snapshot(-7), ensure_ascii=False))

    async def test_future_batches_do_not_push_prior_80_out_of_window(self):
        ids = [self.say(f"上一批历史{i}") for i in range(80)]
        current = self.say("本批当前请求")
        for i in range(100):
            self.say(f"下一批不能提前看到{i}")
        rows = self.manager.snapshot(7, before_id=current)["近期往返"]
        self.assertEqual([r["本地记录ID"] for r in rows], ids)
        self.assertNotIn("本批当前请求", json.dumps(rows, ensure_ascii=False))
        self.assertNotIn("下一批", json.dumps(rows, ensure_ascii=False))

    async def test_deleted_and_forgotten_sources_are_excluded_and_window_refilled(self):
        ids = [self.say(f"原始消息{i}") for i in range(100)]
        with patch.object(self.db.memories, "blocked_sources", return_value=set(ids[-20:])):
            rows = self.snapshot()["近期往返"]
            self.assertEqual([r["本地记录ID"] for r in rows], ids[:80])
        with self.db.connection:
            self.db.connection.execute("DELETE FROM messages WHERE id=?", (ids[-1],))
        self.assertNotIn(ids[-1], [r["本地记录ID"] for r in self.snapshot()["近期往返"]])

    async def test_cache_ttl_does_not_hide_low_frequency_chat_history(self):
        source = self.say("两小时前的原话仍是最近聊天")
        with self.db.connection:
            self.db.connection.execute("UPDATE messages SET created_at=? WHERE id=?", (int(self.now[0]) - 7200, source))
        self.now[0] += 61
        await self.manager.maintain_once()
        self.assertNotIn(7, self.manager.scopes)
        self.assertIn("两小时前的原话", json.dumps(self.snapshot(), ensure_ascii=False))
        self.assertEqual(self.model.inputs, [])

    async def test_capacity_ttl_and_lru_do_not_delete_raw_database(self):
        for i in range(70):
            self.say(f"保留原始记录{i}")
        self.assertEqual(len(self.manager.scopes[7].rows), 32)
        self.say("其他群", namespace=8)
        self.say("另一个群", namespace=9)
        self.assertEqual(len(self.manager.scopes), 2)
        self.assertNotIn(7, self.manager.scopes)
        self.now[0] += 61
        await self.manager.maintain_once()
        self.assertEqual(len(self.manager.scopes), 0)
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM messages").fetchone()[0], 72)

    async def test_restart_restores_raw_window_from_database(self):
        first = self.say("第一次收到")
        manager = WorkingMemoryManager(self.cfg, self.db, self.capture)
        try:
            self.assertEqual(manager.snapshot(7, before_id=first + 1)["近期往返"][0]["内容"], "第一次收到")
            source = self.say("第二次收到")
            manager.receive(7, source)
            self.assertEqual(len(manager.snapshot(7, before_id=source + 1)["近期往返"]), 2)
        finally:
            await manager.close()

    async def test_explicit_quote_outside_80_is_loaded_without_cross_scope_or_duplicate_text(self):
        root = self.say("很早但明确引用的原话")
        for i in range(90):
            self.say(f"无关消息{i}")
        current = self.say("你说的是这个吗", user=22, reply=str(root))
        result = self.manager.snapshot(7, before_id=current + 1, roots=[str(root)])
        text = json.dumps(result, ensure_ascii=False)
        self.assertEqual(text.count("很早但明确引用的原话"), 1)
        reply = next(r for r in result["近期往返"] if r["本地记录ID"] == current)
        self.assertEqual(reply["引用或回应原文"]["本地记录ID"], root)
        self.assertNotIn("内容", reply["引用或回应原文"])
        self.assertNotIn("很早但明确引用的原话", json.dumps(self.manager.snapshot(8, before_id=current, roots=[str(root)]), ensure_ascii=False))

    async def test_budget_is_exact_and_truncation_is_marked(self):
        for i in range(12):
            self.say("长原文" * 300)
        self.manager.settings = self.cfg.model_copy(update={"working_memory_context_max_chars": 1000})
        result = self.snapshot()
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))), 1000)
        self.assertTrue(result["近期往返"])
        self.assertTrue(any(r.get("原文已截断") for r in result["近期往返"]))

    async def test_images_remain_separate_metadata_not_user_speech_in_both_scopes(self):
        for ns in (7, -11):
            source = self.say("看这张图", namespace=ns)
            self.manager.media_lookup = lambda scope, ids: [{"本地记录ID": source, "图片编号": "media_1",
                "发图人QQ": 11, "概要": "蓝色角色举球棒", "概要状态": "ready"}]
            message = self.snapshot(ns)["近期往返"][0]
            self.assertEqual(message["内容"], "看这张图")
            self.assertEqual(message["图片资料（非原话）"][0]["概要"], "蓝色角色举球棒")
        self.assertEqual(self.model.inputs, [])

    async def test_oversized_image_metadata_keeps_image_id_within_budget(self):
        source = self.say("有人发图")
        self.manager.settings = self.cfg.model_copy(update={"working_memory_context_max_chars": 1000})
        result = self.snapshot(image_lookup=lambda i: [{"图片编号": "media_1", "概要": "很长" * 10000}] if i == source else [])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))), 1000)
        self.assertEqual(result["近期往返"][0]["图片资料（非原话）"][0]["图片编号"], "media_1")
        self.assertTrue(result["近期往返"][0]["图片资料已缩略"])

    async def test_archiving_twice_or_after_incremental_capture_does_not_duplicate_jobs(self):
        source = self.say("本人新的具体事件")
        self.assertTrue(await self.capture.archive_working(7, {source}))
        self.assertFalse(await self.capture.archive_working(7, {source}))
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM memory_consolidation_jobs").fetchone()[0], 1)
        source = self.say("旧增量已经处理")
        with self.db.connection:
            self.db.connection.execute("UPDATE memory_capture_state SET cursor_id=? WHERE namespace=7", (source,))
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

    async def test_explicit_consolidate_uses_real_recent_window(self):
        ids = [self.say(f"待整合原话{i}") for i in range(90)]
        self.manager.scopes.clear()
        self.assertTrue(await self.manager.archive(7, ids[-1] + 1))
        jobs = self.db.connection.execute("SELECT * FROM memory_consolidation_jobs").fetchall()
        fresh = set().union(*(set(json.loads(j["new_ids_json"])) for j in jobs))
        self.assertEqual(fresh, set(ids[-80:]))

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
        self.say("等待归档时又有了新消息")
        release.set()
        await task
        self.assertIn(7, self.manager.scopes)
        self.assertIn("又有了新消息", json.dumps(self.snapshot(), ensure_ascii=False))

    async def test_short_window_never_calls_extractor_but_long_memory_still_does(self):
        source = self.say("请记住，我喜欢红茶", namespace=-11)
        self.snapshot(-11)
        await self.manager.maintain_once()
        self.assertEqual(self.model.inputs, [])
        await self.capture.extract(-11, self.db.memory_source_rows(-11, {source}), requested_by=11)
        self.assertEqual(len(self.model.inputs), 1)


class WorkingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_prompt_contains_80_originals_and_new_long_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Settings(_env_file=None, embedding_provider="hash", sticker_enabled=False,
                sticker_collect_enabled=False, web_search_enabled=False, voice_enabled=False, poke_enabled=False)
            db = Database(Path(folder) / "db.sqlite")
            llm = DeepSeekClient(cfg)
            bot = BotService(cfg, db, llm)
            observed = []
            long_memory = [{"记忆编号": 9001, "内容": "QQ11偏好乌龙茶", "层": "abstract", "版本": 1}]
            bot.memory.retrieve = AsyncMock(return_value=long_memory)
            async def request(messages, temperature, definitions):
                observed.append(copy.deepcopy(messages))
                self.assertTrue({"memory_add", "memory_search", "memory_update", "memory_delete", "memory_forget", "memory_consolidate"}
                                <= {d["function"]["name"] for d in definitions})
                return {"tool_calls": [{"id": "send", "function": {"name": "send_message",
                    "arguments": '{"text":"我看到了","next_action":"finish"}'}}]}
            llm.request_message = request
            try:
                ids = [db.log_message(7, 11 if i % 2 else 22, "甲", f"原文{i}", message_id=str(i)) for i in range(100)]
                current = db.log_message(7, 11, "甲", "当前请求", message_id="current")
                db.log_message(7, 22, "乙", "未来批次消息", message_id="future")
                batch = GroupBatch([GroupMessage(current, "current", 11, "甲", "当前请求", 90, direct=True)])
                await bot._run_group_batch_locked(7, batch, AsyncMock(return_value="sent"))
                payload = json.loads(observed[0][1]["content"])
                working = payload["背景资料"]["短期记忆"]
                self.assertEqual([r["本地记录ID"] for r in working["近期往返"]], ids[-80:])
                self.assertNotIn("语境摘要", working)
                self.assertEqual(payload["背景资料"]["长期记忆"], long_memory)
                self.assertNotIn("未来批次消息", json.dumps(payload, ensure_ascii=False))
                self.assertNotIn("当前请求", json.dumps(working, ensure_ascii=False))
                bot.memory.retrieve.assert_awaited_once()
            finally:
                await bot.close()
                await llm.close()
                db.close()

    async def test_bot_working_tool_returns_raw_sources_without_next_batch(self):
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
                self.assertEqual(result["memories"][0]["近期往返"][0]["内容"], "我在准备英语演讲")
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
                with patch("app.bot.build_group_history", side_effect=AssertionError("不应回退旧的重复时间线组装")):
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
            model = Extractor()
            capture = MemoryCapture(cfg, db, memory, model)
            manager = WorkingMemoryManager(cfg, db, capture)
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
