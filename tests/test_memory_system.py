import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from app.bot import BotService
from app.agent import AgentOutcome
from app.chat_images import ChatImageSession
from app.config import Settings
from app.conversation import GroupBatch, GroupMessage
from app.database import Database
from app.embeddings import HASH_MODEL_ID, sparse_vector
from app.group_context import build_group_history
from app.image_cache import ImageCache
from app.llm import DeepSeekClient
from app.memory import MemoryService
from app.memory_capture import MemoryCapture
from app.memory_retrieval import time_range
from app.memory_tools import register_memory_tools
from app.tools import ToolRegistry


class Extractor:
    def __init__(self):
        self.actions = []
        self.inputs = []

    async def extract_memories(self, instructions, data):
        self.inputs.append(data)
        return {"actions": self.actions}


class Dense:
    available = True
    model_id = "test-dense-v1"
    dimensions = 3

    async def start(self):
        return True

    async def embed_query(self, text):
        return "[1.0,0.0,0.0]"

    async def embed_documents(self, texts):
        return ["[1.0,0.0,0.0]" for _ in texts]


class MemorySystemTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Settings(_env_file=None, database_path=Path(self.tmp.name)/"db.sqlite", embedding_provider="hash",
                            image_cache_directory=Path(self.tmp.name)/"images", memory_capture_delay_seconds=5,
                            memory_source_search_enabled=False)
        self.db = Database(self.cfg.database_path)
        self.memory = MemoryService(self.cfg, self.db)
        self.model = Extractor()
        self.capture = MemoryCapture(self.cfg, self.db, self.memory, self.model)

    async def asyncTearDown(self):
        await self.capture.close()
        await self.memory.close()
        self.db.close()
        self.tmp.cleanup()

    def source(self, text, ns=7, user=11, role="user", reply=""):
        if ns > 0:
            return self.db.log_message(ns, user, str(user), text, message_id=str(time.time_ns()), role=role, reply_to_message_id=reply)
        return self.db.log_private_message(-ns, role, text)

    def action(self, source, content, kind="event", subject=11, **extra):
        return {"action": "upsert", "subject_user_id": subject, "memory_type": kind,
                "content": content, "fact_key": content[:60], "source_ids": [source], "replaces_ids": [],
                "confidence": 0.9, "importance": 0.5, "expires_at": 0, **extra}

    async def save(self, content, *, ns=7, user=11, kind="event", **extra):
        source = self.source(content, ns, user)
        self.model.actions = [self.action(source, content, kind, user, **extra)]
        await self.capture.extract(ns, self.db.memory_source_rows(ns, {source}))
        row = self.db.connection.execute("SELECT * FROM memory_records WHERE group_id=? AND content=?", (ns, content)).fetchone()
        return source, dict(row)

    async def test_old_event_not_lost_to_recent_300_and_cross_speaker_recall(self):
        _, old = await self.save("QQ22修复了火星数据库连接错误", user=22)
        # 直接写入测试夹具，避免400次模型/向量调用；旧记录必须仍能被FTS命中。
        with self.db.connection:
            self.db.connection.executemany("INSERT INTO memory_records(group_id,user_id,source_user_id,memory_type,content,created_at,updated_at,layer) VALUES(7,11,11,'event',?,?,?,'event')",
                                           [(f"普通事件{i}", int(time.time()), int(time.time())) for i in range(400)])
        rows = await self.memory.retrieve(7, "火星数据库怎么修复", user_ids={11})
        self.assertIn(old["id"], [r.get("记忆编号") for r in rows])
        self.assertEqual(await self.memory.retrieve(8, "火星数据库", user_ids={22}), [])
        self.assertEqual(await self.memory.retrieve(-22, "火星数据库", user_ids={22}), [])

    async def test_unrelated_profiles_do_not_fill_context_and_fts_escaping(self):
        await self.save("QQ11喜欢红茶", kind="preference")
        self.assertEqual(await self.memory.retrieve(7, "量子力学", user_ids={11}), [])
        self.assertIsInstance(await self.memory.retrieve(7, '红茶" OR * NOT NEAR(', user_ids={11}), list)

    async def test_personal_episode_is_normalized_without_weakening_joint_provenance(self):
        source = self.source("昨天我修好了网站", -11)
        self.model.actions = [self.action(source, "QQ11昨天修好了网站", "episode")]
        await self.capture.extract(-11, self.db.memory_source_rows(-11, {source}))
        row = self.db.memory_candidates(-11, {11}, 10)[0]
        self.assertEqual(row["memory_type"], "event")
        self.model.actions = [self.action(source, "QQ11与机器人一起修好了网站", "episode")]
        with self.assertRaises(ValueError):
            await self.capture.extract(-11, self.db.memory_source_rows(-11, {source}))

    async def test_old_quoted_evidence_is_context_not_new_authorization(self):
        old = self.source("删除我的红茶记忆", -11)
        new = self.source("今天只是问候", -11)
        workspace = self.capture.open_workspace(-11, {new}, {old})
        registry = ToolRegistry()
        register_memory_tools(registry, self.capture, -11, 11, self.db.memory_source_rows(-11, {old, new}), workspace=workspace)
        self.assertFalse((await registry.execute("memory_delete", {"memory_id": 1, "expected_version": 1}))["ok"])
        self.capture.close_workspace(workspace)
        self.model.actions = []
        await self.capture.process_task_once()
        self.assertEqual(self.model.inputs[-1]["new_source_ids"], [new])

    async def test_task_cannot_reextract_personal_fact_from_only_old_context(self):
        old = self.source("我喜欢红茶", user=11)
        new = self.source("今天聊别的", user=22)
        self.model.actions = [self.action(old, "QQ11喜欢红茶", "preference", source_ids=[old, new])]
        with self.assertRaises(ValueError):
            await self.capture.extract(7, self.db.memory_source_rows(7, {old, new}), new_source_ids={new})

    async def test_time_filter_uses_event_time_and_not_record_time(self):
        source, row = await self.save("我昨天修复了金星网站报错")
        yesterday, end = time_range("昨天")
        self.assertEqual(row["occurred_at"], yesterday)
        found = await self.memory.retrieve(7, "昨天金星网站", user_ids={11}, layer="event")
        self.assertEqual(found[0]["记忆编号"], row["id"])
        self.assertEqual(await self.memory.retrieve(7, "金星网站", user_ids={11}, after=end+1), [])

    async def test_abstract_inference_is_candidate_and_definition_can_be_active(self):
        source = self.source("我们聊天提到坐牢")
        self.model.actions = [self.action(source, "本群坐牢可能表示调试", "group_meme", 0, meaning="长时间调试")]
        await self.capture.extract(7, self.db.memory_source_rows(7, {source}))
        self.assertEqual(self.db.connection.execute("SELECT status FROM memory_records").fetchone()[0], "candidate")
        self.assertEqual(await self.memory.retrieve(7, "坐牢什么意思", user_ids={11}), [])
        source = self.source("我们这里坐牢指的是长时间调试程序")
        self.model.actions = [self.action(source, "本群坐牢表示长时间调试程序", "group_meme", 0,
                             fact_key="坐牢", concept_basis="explicit_definition", meaning="长时间调试程序", conditions="技术讨论时", aliases=["工位囚徒"])]
        await self.capture.extract(7, self.db.memory_source_rows(7, {source}))
        rows = await self.memory.retrieve(7, "今天说的坐牢什么意思", user_ids={11})
        self.assertTrue(rows)
        self.assertEqual(rows[0]["层"], "abstract")
        self.assertEqual(rows[0]["conditions"], "技术讨论时")
        alias = await self.memory.retrieve(7, "工位囚徒什么意思", user_ids={11})
        self.assertEqual(alias[0]["记忆编号"], rows[0]["记忆编号"])

    async def test_same_key_different_conditions_coexist_without_auto_overwrite(self):
        _, first = await self.save("QQ11白天喜欢喝咖啡", kind="preference", fact_key="咖啡偏好", conditions="白天")
        _, second = await self.save("QQ11晚上不喝咖啡", kind="preference", fact_key="咖啡偏好", conditions="晚上")
        self.assertEqual(self.db.memories.row(7, first["id"])["status"], "active")
        self.assertEqual(self.db.memories.row(7, second["id"])["status"], "active")
        self.db.memories.revoke(7, 11, first["id"])
        self.assertFalse(self.db.memories.blocked(7, 11, "咖啡偏好", "新证据晚上不喝咖啡", conditions="晚上"))

    async def test_identical_event_summary_on_different_days_is_not_merged(self):
        source, first = await self.save("QQ11修复了水星网站")
        next_source = self.source("第二天又修复了水星网站")
        with self.db.connection:
            self.db.connection.execute("UPDATE messages SET created_at=created_at+86400 WHERE id=?", (next_source,))
        self.model.actions = [self.action(next_source, first["content"])]
        result = await self.capture.extract(7, self.db.memory_source_rows(7, {next_source}))
        self.assertEqual(result["stored"], 1)
        rows = self.db.memory_candidates(7, {11}, 10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r["occurred_at"] for r in rows}), 2)

    async def test_forget_blocks_raw_history_reextraction_and_private_context(self):
        self.memory.settings.memory_source_search_enabled = True
        source, row = await self.save("我之前研究过银河报错", ns=-11)
        self.db.forget_memory(-11, 11, row["id"])
        self.assertEqual(await self.memory.retrieve(-11, "之前银河报错", user_ids={11}), [])
        self.assertEqual(self.db.recent_private_messages(11, 20), [])
        fresh = self.source("今天再聊银河报错", -11)
        self.model.actions = [self.action(fresh, "我之前研究过银河报错", fact_key=row["fact_key"])]
        result = await self.capture.extract(-11, self.db.memory_source_rows(-11, {fresh}))
        self.assertEqual(result["stored"], 0)
        stored = self.db.memories.row(-11, row["id"], active=False)
        self.assertNotIn("银河", stored["content"])
        self.assertEqual(stored["sources_json"], "[]")

    async def test_group_forgotten_sources_not_in_timeline_or_quotes(self):
        source, row = await self.save("不再展示的秘密偏好", kind="preference")
        old = self.db.memory_source_rows(7, {source})[0]
        self.db.forget_memory(7, 11, row["id"])
        focus = self.source("刚刚呢", reply=old["message_id"])
        batch = GroupBatch([GroupMessage(focus, "new", 11, "A", "刚刚呢", 90, direct=True, reply_to=old["message_id"])])
        history, timeline = build_group_history(self.db, self.cfg, 7, batch)
        self.assertNotIn(source, [r["本地记录ID"] for r in timeline])
        self.assertNotIn("秘密偏好", json.dumps(history, ensure_ascii=False))

    async def test_six_tools_versioned_update_delete_and_owner_binding(self):
        _, row = await self.save("QQ11喜欢红茶", kind="preference")
        source = self.source("改一下，我现在喜欢绿茶")
        registry = ToolRegistry()
        register_memory_tools(registry, self.capture, 7, 11, self.db.memory_source_rows(7, {source}))
        self.assertEqual(set(registry.bindings), {"memory_add", "memory_search", "memory_update", "memory_delete", "memory_forget", "memory_consolidate"})
        self.model.actions = [self.action(source, "QQ11现在喜欢绿茶", "preference", 11, replaces_ids=[row["id"]])]
        args = {"memory_id": row["id"], "expected_version": row["version"]}
        self.assertTrue((await registry.execute("memory_update", args))["ok"])
        self.assertEqual(self.db.memories.row(7, row["id"], active=False)["status"], "superseded")
        self.assertTrue(self.db.connection.execute("SELECT * FROM memory_revisions WHERE memory_id=?", (row["id"],)).fetchone())
        self.assertFalse((await registry.execute("memory_update", args))["ok"])
        newest = dict(self.db.connection.execute("SELECT * FROM memory_records WHERE status='active'").fetchone())
        source = self.source("删除我的绿茶偏好")
        registry = ToolRegistry()
        register_memory_tools(registry, self.capture, 7, 22, self.db.memory_source_rows(7, {source}))
        args = {"memory_id": newest["id"], "expected_version": newest["version"]}
        self.assertFalse((await registry.execute("memory_delete", args))["ok"])
        registry = ToolRegistry()
        register_memory_tools(registry, self.capture, 7, 11, self.db.memory_source_rows(7, {source}))
        self.assertTrue((await registry.execute("memory_delete", args))["ok"])
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM memory_records").fetchone()[0], 0)
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM memory_revisions").fetchone()[0], 0)
        self.assertTrue(self.db.connection.execute("SELECT * FROM memory_index_jobs").fetchone())

    async def test_task_release_durable_consolidation_no_thoughts_or_old_retrieval(self):
        source = self.source("今天修复了银河网站")
        workspace = self.capture.open_workspace(7, {source})
        self.assertFalse(hasattr(workspace, "messages"))
        workspace.search_cache["query"] = [(9999, 1)]
        original = self.db.memory_source_rows(7, {source})[0]
        reply = self.source("我们刚刚讨论了处理方法", role="assistant", reply=original["message_id"])
        self.capture.close_workspace(workspace)
        self.assertTrue(workspace.closed)
        self.assertEqual(workspace.search_cache, {})
        self.model.actions = [self.action(source, "QQ11今天修复了银河网站")]
        self.assertTrue(await self.capture.process_task_once())
        data = self.model.inputs[-1]
        self.assertEqual(set(data["new_source_ids"]), {source, reply})
        self.assertNotIn("不应入库的推理", json.dumps(data, ensure_ascii=False))
        self.assertNotIn("检索出来的旧事", json.dumps(data, ensure_ascii=False))
        job = self.db.connection.execute("SELECT * FROM memory_consolidation_jobs").fetchone()
        self.assertEqual(job["status"], "done")

    async def test_clear_data_invalidates_pending_task_and_restart_preserves_index_jobs(self):
        source = self.source("我喜欢红茶", -11)
        workspace = self.capture.open_workspace(-11, {source})
        self.capture.close_workspace(workspace)
        self.db.delete_private_data(11)
        self.assertFalse(await self.capture.process_task_once())
        _, row = await self.save("QQ22修复了银河服务", user=22)
        self.db.close()
        self.db = Database(self.cfg.database_path)
        self.memory.db = self.db
        self.memory.retriever.db = self.db
        self.capture.db = self.db
        self.assertEqual(self.db.memories.row(7, row["id"])["content"], row["content"])
        self.assertTrue(self.db.connection.execute("SELECT * FROM memory_index_jobs WHERE memory_id=?", (row["id"],)).fetchone())
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM memory_fts WHERE rowid=?", (row["id"],)).fetchone()[0], 1)

    async def test_qdrant_payload_scope_version_and_delete_all_model_indexes(self):
        self.memory.provider = Dense()
        self.memory.index.settings.memory_qdrant_url = "http://qdrant:6333"
        source, row = await self.save("QQ11处理了银河故障")
        collection = self.memory.index.collection(Dense.model_id, 3)
        old_collection = self.memory.index.collection("old-dense", 3)
        requests = []
        def handle(request):
            body = json.loads(request.content) if request.content else {}
            requests.append((request.method, request.url.path, body))
            if request.url.path == "/collections":
                return httpx.Response(200, json={"result": {"collections": [{"name": collection}, {"name": old_collection}, {"name": "unrelated"}]}})
            if request.url.path.endswith("/points/query"):
                return httpx.Response(200, json={"result": {"points": [
                    {"id": row["id"], "score": 0.99, "payload": {"namespace": 8, "version": 1}},
                    {"id": row["id"], "score": 0.99, "payload": {"namespace": 7, "version": 99}}]}})
            return httpx.Response(200, json={"result": {}})
        await self.memory.index.client.aclose()
        self.memory.index.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        await self.memory.sync_index_once()
        upsert = next(body for method, path, body in requests if path.endswith("/points") and method == "PUT")
        self.assertEqual(upsert["points"][0]["payload"]["namespace"], 7)
        self.assertNotIn("content", upsert["points"][0]["payload"])
        self.db.forget_memory(7, 11, row["id"])
        self.assertEqual(await self.memory.retrieve(7, "银河故障", user_ids={11}), [])
        await self.memory.sync_index_once()
        deletes = [path for method, path, body in requests if path.endswith("/points/delete")]
        self.assertTrue(any(collection in p for p in deletes))
        self.assertTrue(any(old_collection in p for p in deletes))
        self.assertFalse(any("unrelated" in p for p in deletes))

    async def test_index_outage_falls_back_and_outbox_survives_retry(self):
        self.memory.provider = Dense()
        self.memory.index.settings.memory_qdrant_url = "http://qdrant:6333"
        _, row = await self.save("QQ11修复了水星系统报错")
        def unavailable(request):
            raise httpx.ConnectError("synthetic unavailable")
        await self.memory.index.client.aclose()
        self.memory.index.client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
        found = await self.memory.retrieve(7, "水星系统", user_ids={11})
        self.assertEqual(found[0]["记忆编号"], row["id"])
        self.assertEqual(await self.memory.sync_index_once(), 0)
        self.assertGreater(self.db.connection.execute("SELECT attempts FROM memory_index_jobs WHERE memory_id=?", (row["id"],)).fetchone()[0], 0)

    async def test_backfill_cannot_restore_forgotten_vectors(self):
        _, row = await self.save("QQ11喜欢红茶", kind="preference")
        self.db.forget_memory(7, 11, row["id"])
        self.assertFalse(self.db.update_memory_embedding(row["id"], "[1,0,0]", "new-model", 3, row["content"]))
        self.assertEqual(self.db.memories.row(7, row["id"], active=False)["vector"], "{}")

    async def test_failed_consolidation_is_bounded_and_other_jobs_can_continue(self):
        source = self.source("原始资料")
        workspace = self.capture.open_workspace(7, {source})
        self.capture.close_workspace(workspace)
        self.model.actions = [self.action(source+999, "不真实证据")]
        for _ in range(self.cfg.memory_job_max_attempts):
            self.db.connection.execute("UPDATE memory_consolidation_jobs SET due_at=0")
            self.db.connection.commit()
            self.assertTrue(await self.capture.process_task_once())
        job = self.db.connection.execute("SELECT status FROM memory_consolidation_jobs").fetchone()
        self.assertEqual(job[0], "failed")
        self.assertFalse(await self.capture.process_task_once())

    async def test_retrieval_cache_does_not_reuse_forgotten_content(self):
        _, row = await self.save("QQ11喜欢红茶", kind="preference")
        source = self.source("之前红茶偏好是什么")
        workspace = self.capture.open_workspace(7, {source})
        registry = ToolRegistry()
        register_memory_tools(registry, self.capture, 7, 11, self.db.memory_source_rows(7, {source}), workspace=workspace)
        args = {"query": "红茶偏好"}
        self.assertTrue((await registry.execute("memory_search", args))["memories"])
        self.db.forget_memory(7, 11, row["id"])
        self.assertEqual((await registry.execute("memory_search", args))["memories"], [])
        self.capture.close_workspace(workspace)

    async def test_cancelled_agent_releases_workspace_and_queues_only_real_evidence(self):
        cfg = self.cfg.model_copy(update={"sticker_enabled": False, "sticker_collect_enabled": False,
                                           "web_search_enabled": False, "voice_enabled": False, "poke_enabled": False})
        llm = DeepSeekClient(cfg)
        service = BotService(cfg, self.db, llm)
        observed = []
        async def cancelled(prompt, registry, **kwargs):
            observed.append(registry.task_evidence)
            raise asyncio.CancelledError()
        service.agent.run = cancelled
        try:
            source = self.source("合成任务", -11)
            with self.assertRaises(asyncio.CancelledError):
                await service._run_ai_reply(11, 11, "合成任务", AsyncMock(), private=True, private_source_id=source)
            self.assertTrue(observed[0].closed)
            self.assertFalse(observed[0].source_ids)
            self.assertEqual(self.db.connection.execute("SELECT status FROM memory_consolidation_jobs").fetchone()[0], "pending")
        finally:
            await service.close()
            await llm.close()

    async def test_old_image_automatic_recall_grants_real_view_capability(self):
        cfg = self.cfg.model_copy(update={"sticker_enabled": False, "sticker_collect_enabled": False,
                                           "web_search_enabled": False, "voice_enabled": False, "poke_enabled": False})
        llm = DeepSeekClient(cfg)
        service = BotService(cfg, self.db, llm)
        service.image_cache.downloader.load = AsyncMock(return_value=("image/png", b"\x89PNG\r\n\x1a\nsynthetic"))
        service.image_cache.llm.summarize_image = AsyncMock(return_value={"summary": "蓝色角色举球棒", "text": "", "uncertain": False})
        results = []
        try:
            photo = self.source("[图片]", -11)
            ref = service.image_cache.record(-11, photo, 11, "", [{"url": "https://gchat.qpic.cn/test"}])[0]
            await service.image_cache.load(ref)
            # 使用假的图片模型Key仅令摘要任务运行，不发送外部请求。
            service.image_cache.llm.settings = service.image_cache.llm.settings.model_copy(update={"deepseek_api_key": "test"})
            await service.image_cache.summarize_once()
            await service.memory.sync_images()
            source = self.source("以前那张蓝色球棒图，看看", -11)
            async def run(prompt, registry, **kwargs):
                result = await registry.execute("get_message_image", {"image_id": f"media_{ref['cache_id']}"})
                results.append(result)
                return AgentOutcome(1, 1, 0)
            service.agent.run = run
            # 模拟该图已不在初始附件候选中；必须由自动长期召回授予访问能力。
            await service._run_ai_reply(11, 11, "以前蓝色球棒图", AsyncMock(), private=True,
                                        private_source_id=source, image_session=ChatImageSession([]))
            self.assertTrue(results[0]["ok"])
            self.assertTrue(results[0]["_image_content"])
        finally:
            await service.close()
            await llm.close()

    async def test_image_perception_vectors_scope_and_retained_event_source_expiry(self):
        image_model = SimpleNamespace(settings=SimpleNamespace(deepseek_api_key="test"), summarize_image=AsyncMock(return_value={"summary": "蓝色角色举着球棒", "text": "测试", "uncertain": False}))
        cache = ImageCache(self.cfg, self.db, image_model)
        cache.downloader.load = AsyncMock(return_value=("image/png", b"\x89PNG\r\n\x1a\nfixture"))
        self.memory.image_cache = cache
        try:
            source = self.source("今天讨论了这张球棒图")
            ref = cache.record(7, source, 11, "photo", [{"url": "https://gchat.qpic.cn/test"}])[0]
            await cache.load(ref)
            await cache.summarize_once()
            await self.memory.sync_images()
            media_id = int(ref["cache_id"])
            self.model.actions = [self.action(source, "QQ11今天讨论了一张图片", media_ids=[f"media_{media_id}"])]
            await self.capture.extract(7, self.db.memory_source_rows(7, {source}))
            rows = await self.memory.retrieve(7, "蓝色角色球棒", user_ids={11}, layer="perceptual")
            self.assertEqual(rows[0]["图片编号"], f"media_{media_id}")
            self.assertEqual(await self.memory.retrieve(8, "蓝色角色球棒", user_ids={11}), [])
            self.db.connection.execute("DELETE FROM messages WHERE id=?", (source,))
            self.db.connection.commit()
            self.assertTrue(cache.authorized(ref))
            self.db.delete_group_user_data(7, 11)
            self.assertFalse(cache.authorized(ref))
        finally:
            await cache.close()


if __name__ == "__main__":
    unittest.main()
