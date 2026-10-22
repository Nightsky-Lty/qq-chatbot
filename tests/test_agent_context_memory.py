import asyncio
import copy
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from app.config import Settings
from app.bot import BotService
from app.conversation import GroupBatch, GroupMessage
from app.database import Database
from app.group_context import build_group_history
from app.llm import DeepSeekClient
from app.memory import MemoryService
from app.memory_capture import MemoryCapture
from app.memory_tools import register_memory_tools
from app.tools import ToolRegistry


class FakeExtractor:
    def __init__(self):
        self.actions = []
        self.inputs = []

    async def extract_memories(self, instructions, data):
        self.inputs.append(data)
        return {"actions": self.actions}


def candidate(subject, sources, **changes):
    return {"action": "upsert", "subject_user_id": subject, "memory_type": "preference",
            "content": f"QQ{subject}偏好英语纠错先给正确句子", "fact_key": "英语纠错方式",
            "source_ids": sources, "replaces_ids": [], "confidence": 0.95,
            "importance": 0.8, "expires_at": 0, **changes}


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "bot.db"
        self.db = Database(self.path)
        self.cfg = Settings(_env_file=None, embedding_provider="hash", memory_capture_delay_seconds=5)
        self.memory = MemoryService(self.cfg, self.db)
        self.extractor = FakeExtractor()
        self.capture = MemoryCapture(self.cfg, self.db, self.memory, self.extractor)

    async def asyncTearDown(self):
        await self.capture.close()
        await self.memory.close()
        self.db.close()
        self.tmp.cleanup()

    def private(self, qq, text, role="user"):
        return self.db.log_private_message(qq, role, text)

    async def store_private(self):
        record = self.private(11, "我希望英语纠错先给正确句子")
        self.extractor.actions = [candidate(11, [record])]
        await self.capture.extract(-11, self.db.memory_source_rows(-11, {record}))
        return self.db.memory_candidates(-11, {11}, 20)[0]["id"]

    async def test_private_recall_scope_and_update(self):
        old_id = await self.store_private()
        self.assertTrue(await self.memory.retrieve(-11, "英语怎么改", user_ids={11}))
        self.assertEqual(await self.memory.retrieve(11, "英语怎么改", user_ids={11}), [])
        self.assertEqual(await self.memory.retrieve(-12, "英语怎么改", user_ids={12}), [])
        record = self.private(11, "改一下，英语纠错先解释原因")
        self.extractor.actions = [candidate(11, [record], content="QQ11偏好英语纠错先解释原因", replaces_ids=[old_id])]
        await self.capture.extract(-11, self.db.memory_source_rows(-11, {record}))
        rows = self.db.memory_candidates(-11, {11}, 20)
        self.assertEqual(len(rows), 1)
        self.assertIn("先解释", rows[0]["content"])
        self.assertTrue(json.loads(rows[0]["sources_json"]))
        self.assertEqual(self.db.connection.execute("SELECT status FROM memory_records WHERE id=?", (old_id,)).fetchone()[0], "superseded")

    async def test_provenance_and_atomic_rejection(self):
        record = self.private(11, "记住我喜欢先给正确句子")
        self.extractor.actions = [candidate(11, [record]), candidate(12, [record])]
        with self.assertRaises(ValueError):
            await self.capture.extract(-11, self.db.memory_source_rows(-11, {record}))
        self.assertEqual(self.db.memory_candidates(-11, {11}, 20), [])
        self.extractor.actions = [candidate(11, [record + 99])]
        with self.assertRaises(ValueError):
            await self.capture.extract(-11, self.db.memory_source_rows(-11, {record}))

    async def test_joint_episode_needs_actual_bot_reply(self):
        user = self.db.log_message(99, 11, "A", "想练英语", message_id="100")
        wrong = self.db.log_message(99, 90, "bot", "一起练", role="assistant", reply_to_message_id="200")
        self.extractor.actions = [candidate(11, [user, wrong], memory_type="commitment", content="QQ11与机器人约定一起练英语")]
        with self.assertRaises(ValueError):
            await self.capture.extract(99, self.db.memory_source_rows(99, {user, wrong}))
        bot = self.db.log_message(99, 90, "bot", "一起练", role="assistant", reply_to_message_id="100")
        self.extractor.actions = [candidate(11, [user, bot], memory_type="commitment", content="QQ11与机器人约定一起练英语")]
        await self.capture.extract(99, self.db.memory_source_rows(99, {user, bot}))
        self.assertEqual(len(self.db.memory_candidates(99, {11}, 20)), 1)

    async def test_forget_does_not_resurrect_from_pending_capture(self):
        old_id = await self.store_private()
        self.assertTrue(self.db.forget_memory(-11, 11, old_id))
        record = self.private(11, "今天练英语")
        self.extractor.actions = [candidate(11, [record])]
        result = await self.capture.extract(-11, self.db.memory_source_rows(-11, {record}))
        self.assertEqual(result["stored"], 0)
        self.assertEqual(self.db.memory_candidates(-11, {11}, 20), [])

    async def test_expired_records_not_recalled(self):
        old_id = await self.store_private()
        self.db.connection.execute("UPDATE memory_records SET expires_at=? WHERE id=?", (int(time.time()) - 1, old_id))
        self.db.connection.commit()
        self.assertEqual(await self.memory.retrieve(-11, "英语", user_ids={11}), [])

    async def test_persistent_queue_high_watermark_and_restart(self):
        first = self.private(11, "我喜欢英语")
        self.db.connection.execute("UPDATE memory_capture_state SET due_at=0")
        self.db.connection.commit()
        job = self.db.capture_job()
        second = self.private(11, "也喜欢数学")
        self.extractor.actions = []
        await self.capture.extract(-11, self.db.memory_source_rows(-11, {first}), cursor_id=job["cursor_id"], through_id=first)
        self.db.close()
        self.db = Database(self.path)
        state = self.db.connection.execute("SELECT * FROM memory_capture_state WHERE namespace=-11").fetchone()
        self.assertEqual(state["cursor_id"], first)
        self.assertEqual(state["requested_id"], second)
        self.assertIn("sources_json", {r["name"] for r in self.db.connection.execute("PRAGMA table_info(memory_records)")})

    async def test_clear_private_data_invalidates_inflight_sources(self):
        await self.store_private()
        record = self.private(11, "我的偏好改变了")
        sources = self.db.memory_source_rows(-11, {record})
        self.db.delete_private_data(11)
        self.extractor.actions = [candidate(11, [record])]
        with self.assertRaises(ValueError):
            await self.capture.extract(-11, sources)
        self.assertEqual(self.db.memory_candidates(-11, {11}, 20), [])

    async def test_explicit_forget_is_bound_to_request_and_owner(self):
        old_id = await self.store_private()
        record = self.private(11, "别忘记我的偏好")
        registry = ToolRegistry()
        register_memory_tools(registry, self.capture, -11, 11, self.db.memory_source_rows(-11, {record}))
        self.assertFalse((await registry.execute("memory_forget", {"memory_id": old_id, "expected_version": 1}))["ok"])
        record = self.private(12, "忘掉我的偏好")
        registry = ToolRegistry()
        register_memory_tools(registry, self.capture, -12, 12, self.db.memory_source_rows(-12, {record}))
        self.assertFalse((await registry.execute("memory_forget", {"memory_id": old_id, "expected_version": 1}))["ok"])


class GroupContextTests(unittest.TestCase):
    def test_legacy_database_migrates_without_losing_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.db"
            connection = sqlite3.connect(path)
            connection.executescript("""
                CREATE TABLE memories(id INTEGER PRIMARY KEY AUTOINCREMENT, group_id INTEGER, user_id INTEGER,
                    source_user_id INTEGER,memory_type TEXT,content TEXT,vector TEXT,importance REAL,
                    confidence REAL,source_message_id TEXT,created_at INTEGER,updated_at INTEGER,last_accessed_at INTEGER,
                    UNIQUE(group_id,user_id,memory_type,content));
                INSERT INTO memories VALUES(1,7,11,11,'preference','原有偏好','{}',0.8,0.9,'100',1,1,0);
            """)
            connection.close()
            db = Database(path)
            rows = db.memory_candidates(7, {11}, 10)
            self.assertEqual(rows[0]["content"], "原有偏好")
            self.assertEqual(rows[0]["status"], "active")
            self.assertEqual(rows[0]["sources_json"], "[]")
            db.close()

    def test_timeline_keeps_identity_quotes_and_no_duplicate_bot_history(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "test.db")
            db.log_message(7, 11, "同名", "今晚打吗", message_id="1", mentions=[22])
            db.log_message(7, 22, "同名", "九点", message_id="2", reply_to_message_id="1")
            db.log_message(7, 90, "bot", "我理解错了", message_id="3", role="assistant", reply_to_message_id="2")
            focus = db.log_message(7, 11, "同名", "你刚刚理解错了", message_id="4", reply_to_message_id="3")
            batch = GroupBatch([GroupMessage(focus, "4", 11, "同名", "你刚刚理解错了", 90, direct=True, reply_to="3")])
            history, rows = build_group_history(db, Settings(_env_file=None), 7, batch)
            self.assertEqual(len(rows), len({r["本地记录ID"] for r in rows}))
            by_id = {r["消息号"]: r for r in rows}
            self.assertEqual(by_id["1"]["寻址"]["对象判断"], "其他群友")
            self.assertEqual(by_id["2"]["引用或回应原文"]["QQ"], 11)
            self.assertEqual(by_id["3"]["身份说明"], "你自己实际发送的消息")
            self.assertEqual(len(history), 1)
            self.assertIn("时间", by_id["1"])
            db.close()


class LLMRoleTests(unittest.IsolatedAsyncioTestCase):
    async def test_role_config_and_json_request_are_independent(self):
        cfg = Settings(_env_file=None, deepseek_model="chat-model", deepseek_api_key="test",
                       memory_llm_model="memory-model", sticker_llm_model="image-model")
        memory = DeepSeekClient(cfg, purpose="memory")
        sticker = DeepSeekClient(cfg, purpose="sticker")
        self.assertEqual(sticker.settings.deepseek_model, "image-model")
        self.assertEqual(cfg.deepseek_model, "chat-model")
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"actions":[]}'}}]})
        await memory.client.aclose()
        memory.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        result = await memory.extract_memories("只提取记忆", {"messages": []})
        self.assertEqual(result, {"actions": []})
        self.assertEqual(requests[0]["model"], "memory-model")
        self.assertNotIn("tools", requests[0])
        self.assertEqual(requests[0]["thinking"]["type"], "disabled")
        await memory.close()
        await sticker.close()


class ChatIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_and_private_tools_and_context(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Settings(_env_file=None, embedding_provider="hash", sticker_enabled=False,
                           sticker_collect_enabled=False, web_search_enabled=False,
                           voice_enabled=False, poke_enabled=False)
            db = Database(Path(folder) / "bot.db")
            llm = DeepSeekClient(cfg)
            service = BotService(cfg, db, llm)
            observed = []
            async def request(messages, temperature, tools):
                observed.append((copy.deepcopy(messages), tools))
                return {"tool_calls": [{"id": f"send-{len(observed)}", "type": "function",
                        "function": {"name": "finish" if messages[-1]["role"] == "tool" else "send_message",
                                     "arguments": '{}' if messages[-1]["role"] == "tool" else '{"text":"收到"}'}}]}
            llm.request_message = request
            sent = []
            async def sender(target, segments):
                sent.append((target, segments))
                return str(1000 + len(sent))
            try:
                source_id = db.log_message(7, 11, "A", "纠正你刚才的说法", message_id="10", mentions=[90])
                batch = GroupBatch([GroupMessage(source_id, "10", 11, "A", "纠正你刚才的说法", 90,
                                                direct=True, mentions=[90])])
                outcome = await service._run_group_batch_locked(7, batch, sender)
                self.assertEqual(outcome.sent, 1)
                self.assertIn("群聊结构阅读规则", observed[0][0][0]["content"])
                current = json.loads(observed[0][0][-1]["content"])
                self.assertEqual(current["运行环境"]["机器人身份"]["QQ"], 90)
                self.assertEqual(current["当前任务"]["本批新消息"][0]["寻址"]["对象判断"], "机器人")
                names = {tool["function"]["name"] for tool in observed[0][1]}
                self.assertIn("memory_search", names)
                self.assertNotIn("remember_group_memory", names)
                observed.clear()
                private_id = db.log_private_message(11, "user", "你好")
                outcome = await service._run_ai_reply(11, 11, "你好", sender, private=True, private_source_id=private_id)
                self.assertEqual(outcome.sent, 1)
                private_context = json.loads(observed[0][0][-1]["content"])
                self.assertEqual(private_context["运行环境"]["会话类型"], "private")
                self.assertEqual(private_context["当前任务"]["用户请求"], "你好")
                self.assertNotIn("群聊时间线", private_context["背景资料"])
            finally:
                await service.close()
                await llm.close()
                db.close()


if __name__ == "__main__":
    unittest.main()
