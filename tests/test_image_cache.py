import asyncio
import base64
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from app.bot import BotService
from app.chat_images import ChatImageService, ChatImageSession
from app.config import Settings
from app.database import Database
from app.image_cache import ImageCache
from app.llm import DeepSeekClient, LLMUnavailable


PNG = b"\x89PNG\r\n\x1a\n" + b"synthetic-image-test"
SUMMARY = {"summary": "蓝色角色举着球棒的反应图", "text": "测试配文", "uncertain": False}


class ImageCacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.cfg = Settings(_env_file=None, database_path=Path(self.folder.name)/"db.sqlite",
                            image_cache_directory=Path(self.folder.name)/"images")
        self.db = Database(self.cfg.database_path)
        self.llm = SimpleNamespace(settings=SimpleNamespace(deepseek_api_key="fake"),
                                   summarize_image=AsyncMock(return_value=SUMMARY))
        self.cache = ImageCache(self.cfg, self.db, self.llm)
        self.cache.downloader.load = AsyncMock(return_value=("image/png", PNG))

    async def asyncTearDown(self):
        await self.cache.close()
        self.db.close()
        self.folder.cleanup()

    def record(self, ns=7, *, user=11, data=None):
        source = (self.db.log_message(ns, user, "A", "[图片]", message_id=str(time.time_ns())) if ns > 0
                  else self.db.log_private_message(-ns, "user", "[图片]"))
        ref = self.cache.record(ns, source, user, str(source), [{"url": "https://gchat.qpic.cn/test"}])[0]
        if data is not None:
            self.cache.downloader.load.return_value = ("image/png", data)
        return source, ref

    async def test_hash_dedupe_summary_reuse_and_no_automatic_sticker_collection(self):
        source, first = self.record()
        await self.cache.load(first)
        self.assertTrue(await self.cache.summarize_once())
        _, second = self.record(8)
        await self.cache.load(second)
        self.assertFalse(await self.cache.summarize_once())
        self.llm.summarize_image.assert_awaited_once()
        self.assertEqual(len(list(self.cache.directory.glob("*.img"))), 1)
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM image_cache_refs").fetchone()[0], 2)
        description = self.cache.describe(self.cache.rows(7, source)[0])
        self.assertEqual(description["概要"], SUMMARY["summary"])
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM sticker_annotations").fetchone()[0], 0)
        # 同一ref重复加载不再下载；不同ref必须取得内容才能做哈希去重。
        await self.cache.load(first)
        self.assertEqual(self.cache.downloader.load.await_count, 2)

    async def test_scope_boundary_and_stable_tool_ids(self):
        source, ref = self.record()
        await self.cache.load(ref)
        self.assertEqual(self.cache.rows(8, 9999), [])
        self.assertEqual(self.cache.rows(7, source-1), [])
        with self.assertRaises(ValueError):
            await self.cache.load({**ref, "namespace": "8"})
        row = self.cache.rows(7, source)[0]
        tool = ChatImageService(self.cfg)
        tool.cached_loader = self.cache.load
        tool.cached_validator = self.cache.authorized
        image_id = f"media_{row['id']}"
        session = ChatImageSession([ref], image_ids=[image_id])
        result = await tool.execute_tool(session, {"image_id": image_id})
        self.assertTrue(result["ok"])
        self.assertEqual(session.loaded_images[image_id][1], PNG)
        self.assertFalse((await tool.execute_tool(session, {"image_id": "media_999999"}))["ok"])
        self.db.connection.execute("UPDATE image_cache_refs SET expires_at=0")
        self.db.connection.commit()
        self.assertFalse((await tool.execute_tool(session, {"image_id": image_id}))["ok"])

    async def test_restart_restores_files_sources_and_summary(self):
        source, ref = self.record()
        await self.cache.load(ref)
        await self.cache.summarize_once()
        await self.cache.close()
        self.cache = ImageCache(self.cfg, self.db, self.llm)
        self.cache.downloader.load = AsyncMock(side_effect=AssertionError("缓存不可重新下载"))
        self.assertEqual((await self.cache.load(ref))[1], PNG)
        self.assertEqual(self.cache.describe(self.cache.rows(7, source)[0])["概要"], SUMMARY["summary"])
        self.assertFalse(await self.cache.summarize_once())

    async def test_expiry_and_quota_evict_only_image_cache(self):
        self.cache.settings = self.cfg.model_copy(update={"image_cache_max_files": 1})
        source, first = self.record()
        await self.cache.load(first)
        _, second = self.record(data=PNG+b"second")
        await self.cache.load(second)
        with self.assertRaises(ValueError):
            await self.cache.load(first)
        self.assertEqual(len(list(self.cache.directory.glob("*.img"))), 1)
        unrelated = self.cache.directory / "not-a-cache-file.txt"
        unrelated.write_text("keep")
        self.db.connection.execute("UPDATE image_cache_refs SET expires_at=0")
        self.db.connection.commit()
        self.assertEqual(self.cache.rows(7, 9999), [])
        await self.cache.cleanup()
        self.assertEqual(list(self.cache.directory.glob("*.img")), [])
        self.assertTrue(unrelated.exists())

    async def test_deleted_source_and_inflight_download_cannot_resurrect(self):
        source, ref = self.record(-11)
        started, release = asyncio.Event(), asyncio.Event()
        async def download(image):
            started.set()
            await release.wait()
            return "image/png", PNG
        self.cache.downloader.load = download
        task = asyncio.create_task(self.cache.load(ref))
        await started.wait()
        self.db.delete_private_data(11)
        release.set()
        with self.assertRaises(ValueError):
            await task
        await self.cache.cleanup()
        self.assertEqual(self.cache.rows(-11, source), [])
        self.assertEqual(self.db.connection.execute("SELECT count(*) FROM image_cache_blobs").fetchone()[0], 0)

    async def test_summary_failure_is_not_faked_and_original_remains_available(self):
        source, ref = self.record()
        await self.cache.load(ref)
        self.llm.summarize_image.side_effect = LLMUnavailable("failed")
        await self.cache.summarize_once()
        description = self.cache.describe(self.cache.rows(7, source)[0])
        self.assertNotIn("概要", description)
        self.assertEqual(description["概要状态"], "error")
        self.assertEqual((await self.cache.load(ref))[1], PNG)
        self.assertFalse(await self.cache.summarize_once())  # 退避中不刷API。

    async def test_intake_queue_limit_and_checksum_validation(self):
        self.cache.settings = self.cfg.model_copy(update={"image_cache_queue_limit": 1})
        _, first = self.record()
        source, second = self.record()
        with self.assertRaises(ValueError):
            await self.cache.load(second)
        await self.cache.load(first)
        digest = hashlib.sha256(PNG).hexdigest()
        self.cache._path(digest).write_bytes(b"corrupted")
        with self.assertRaises(ValueError):
            await self.cache.load(first)
        self.assertEqual(self.cache.rows(7, source)[0]["status"], "unavailable")

    async def test_explicit_quote_can_find_older_cached_reference_and_deleted_scope_is_removed(self):
        source, first = self.record()
        await self.cache.load(first)
        _, private = self.record(-11)
        await self.cache.load(private)
        last = source
        for _ in range(205):
            last, _ = self.record()
        rows = self.cache.rows(7, last, {source})
        self.assertEqual(rows[0]["source_id"], source)
        self.db.delete_private_data(11)
        await self.cache.cleanup()
        self.assertEqual(self.cache.rows(-11, 999999), [])
        self.assertEqual((await self.cache.load(first))[1], PNG)  # 同文件其他群引用不能误删。

    async def test_background_queue_persists_and_download_runs_without_chat_wakeup(self):
        source, ref = self.record()
        self.cache.start()
        async def ready():
            while self.cache.rows(7, source)[0]["status"] != "ready":
                await asyncio.sleep(0.01)
        await asyncio.wait_for(ready(), 2)
        self.assertEqual((await self.cache.load(ref))[1], PNG)


class ImageConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_history_uses_only_current_private_image_namespace(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Settings(_env_file=None, database_path=Path(folder)/"db.sqlite", image_cache_directory=Path(folder)/"cache",
                           memory_enabled=False, sticker_enabled=False, ai_cooldown_seconds=0)
            db, observed = Database(cfg.database_path), []
            llm = DeepSeekClient(cfg)
            bot = BotService(cfg, db, llm)
            source = db.log_private_message(11, "user", "[图片]")
            own = bot.image_cache.record(-11, source, 11, "", [{"url": "https://gchat.qpic.cn/own"}])[0]
            other_source = db.log_private_message(22, "user", "[图片]")
            other = bot.image_cache.record(-22, other_source, 22, "", [{"url": "https://gchat.qpic.cn/other"}])[0]
            async def request(messages, temperature, tools):
                observed.append(messages)
                text = " ".join(message.get("content", "") for message in messages if isinstance(message.get("content"), str))
                self.assertIn(f"media_{own['cache_id']}", text)
                self.assertNotIn(f"media_{other['cache_id']}", text)
                self.assertIn("get_message_image", {tool["function"]["name"] for tool in tools})
                return {"tool_calls": [{"id": "send", "function": {"name": "send_message",
                    "arguments": json.dumps({"text": "收到", "next_action": "finish"})}}]}
            llm.request_message = request
            async def sender(target, message):
                return "300"
            try:
                await bot.handle_event({"post_type": "message", "message_type": "private", "user_id": 11,
                    "self_id": 99, "message": "刚才那张图片是什么？"}, sender, sender)
                await asyncio.gather(*list(bot._ai_tasks))
                self.assertTrue(observed)
            finally:
                await bot.close()
                await llm.close()
                db.close()

    async def test_cross_speaker_image_from_ignored_previous_batch_is_operable(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = Settings(_env_file=None, deepseek_api_key="fake", database_path=Path(folder)/"db.sqlite", image_cache_directory=Path(folder)/"cache",
                random_chat_enabled=False, chat_batch_seconds=0, ai_cooldown_seconds=0, memory_enabled=False,
                sticker_collect_probability=0, sticker_auto_index_enabled=False, sticker_prompt_limit=0)
            db, captured, sent = Database(cfg.database_path), [], []
            llm = DeepSeekClient(cfg)
            bot = BotService(cfg, db, llm)
            bot.image_cache.downloader.load = AsyncMock(return_value=("image/png", PNG))
            bot.image_llm.summarize_image = AsyncMock(return_value=SUMMARY)
            # 保留注册和真实授权链；仅替换审核落盘本体，不上传图或发真实QQ。
            bot.sticker_collector._collect = AsyncMock(return_value={"ok": True, "created": True})
            async def sender(target, segments):
                sent.append((target, segments))
                return str(200+len(sent))
            async def request(messages, temperature, tools):
                captured.append(json.loads(json.dumps(messages)))
                if not any(message["role"] == "tool" for message in messages):
                    self.assertIn("collect_sticker", {tool["function"]["name"] for tool in tools})
                    context = json.loads(messages[-1]["content"])
                    candidate = context["背景资料"]["图片目录（概要非原话）"][0]
                    self.assertEqual(candidate["发图人QQ"], 11)
                    self.assertIn(candidate["图片编号"], context["运行环境"]["可收藏图片编号"])
                    self.assertIn(SUMMARY["summary"], json.dumps(messages, ensure_ascii=False))
                    return {"tool_calls": [{"id": "collect", "function": {"name": "collect_sticker",
                            "arguments": json.dumps({"image_id": candidate["图片编号"]})}}]}
                return {"tool_calls": [{"id": "send", "function": {"name": "send_message",
                    "arguments": json.dumps({"text": "这图收好了", "next_action": "finish"})}}]}
            llm.request_message = request
            def event(user, mid, message):
                return {"post_type": "message", "message_type": "group", "group_id": 7, "user_id": user,
                        "self_id": 99, "message_id": mid, "time": time.time(), "sender": {"nickname": f"User{user}"}, "message": message}
            try:
                await bot.handle_event(event(11, "101", [{"type": "image", "data": {"url": "https://gchat.qpic.cn/test"}}]), sender)
                self.assertFalse(captured)  # 图片没唤醒机器人，但仍缓存。
                row = bot.image_cache.rows(7, 99999)[0]
                await bot.image_cache.load(bot.image_cache.reference(row))
                await bot.image_cache.summarize_once()
                await bot.handle_event(event(22, "102", [{"type": "at", "data": {"qq": "99"}},
                    {"type": "text", "data": {"text": "你把这个图片保存一下吧"}}]), sender)
                await asyncio.wait_for(bot.conversations.states[7].worker, 3)
                bot.sticker_collector._collect.assert_awaited_once()
                job = bot.sticker_collector._collect.call_args.args[0]
                self.assertEqual(job.group_id, 7)
                self.assertEqual(job.user_id, 11)
                self.assertEqual(job.message_id, "101")
                self.assertTrue(sent)
                self.assertIn("图片资料（非原话）", captured[0][1]["content"])
            finally:
                await bot.close()
                await llm.close()
                db.close()

    async def test_image_summary_client_is_independent_and_json_validated(self):
        cfg = Settings(_env_file=None, deepseek_api_key="fake", image_llm_model="image-model",
                       image_llm_base_url="https://vision.example.com", image_llm_thinking_enabled=False)
        client = DeepSeekClient(cfg, purpose="image")
        await client.client.aclose()
        requests = []
        def reply(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(SUMMARY)}}]})
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        try:
            self.assertEqual(await client.summarize_image("image/png", base64.b64encode(PNG).decode()), SUMMARY)
            self.assertEqual(requests[0]["model"], "image-model")
            self.assertNotIn("tools", requests[0])
            self.assertEqual(requests[0]["thinking"]["type"], "disabled")
            self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        finally:
            await client.close()

    async def test_qq_downloader_pins_dns_without_following_redirect(self):
        cfg = Settings(_env_file=None)
        downloader = ChatImageService(cfg)
        requests = []
        def response(request):
            requests.append(request)
            return httpx.Response(200, stream=httpx.ByteStream(PNG))
        actual_client = httpx.AsyncClient
        def factory(**kwargs):
            return actual_client(**kwargs, transport=httpx.MockTransport(response))
        async def dns(*args, **kwargs):
            return [(2, 1, 6, "", ("93.184.216.34", 443))]
        with patch("app.chat_images.httpx.AsyncClient", side_effect=factory), \
             patch.object(asyncio.get_running_loop(), "getaddrinfo", side_effect=dns):
            mime, data = await downloader.load({"url": "https://gchat.qpic.cn/test"})
        self.assertEqual(data, PNG)
        self.assertEqual(requests[0].url.host, "93.184.216.34")
        self.assertEqual(requests[0].headers["host"], "gchat.qpic.cn")
        self.assertEqual(requests[0].extensions["sni_hostname"], "gchat.qpic.cn")


if __name__ == "__main__":
    unittest.main()
