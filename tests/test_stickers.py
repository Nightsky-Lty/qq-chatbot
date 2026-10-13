"""真实工具/消息链路的离线回归，不调用DeepSeek或发送真实QQ消息。"""

import asyncio
import base64
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from app.bot import AUTO_CHAT_SYSTEM_PROMPT, PRIVATE_SYSTEM_PROMPT, BotService
from app.config import Settings
from app.database import Database
from app.llm import DeepSeekClient, LLMUnavailable
from app.main import OneBotConnection
from app.reminders import ReminderService
from app.stickers import StickerService, StickerSession


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfS0AAAAASUVORK5CYII="
)


def tool_call(name, arguments, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
    }}


def call_message(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


class StickerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "开心.png").write_bytes(PNG)
        (self.images / "晚安.gif").write_bytes(b"GIF89a" + b"\x00" * 20)
        self.settings = Settings(
            _env_file=None, deepseek_api_key="test-only", database_path=self.root / "test.db",
            sticker_directory=self.images,
        )
        self.service = StickerService(self.settings)
        self.sent = []
        self.db = Database(self.settings.database_path)
        self.llm = DeepSeekClient(self.settings)
        await self.llm.client.aclose()
        self.requests = []
        self.responses = []

        def respond(request):
            self.requests.append(json.loads(request.content))
            self.assertTrue(self.responses, "unexpected extra model request")
            response = self.responses.pop(0)
            if isinstance(response, int):
                return httpx.Response(response, json={"error": "test failure"})
            return httpx.Response(200, json={"choices": [{"message": response}]})

        self.llm.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.reminders = ReminderService(self.settings, self.db)
        self.bot = BotService(self.settings, self.db, self.llm, self.reminders)

    async def asyncTearDown(self):
        await self.bot.close()
        await self.reminders.close()
        await self.llm.close()
        self.db.close()
        self.temp.cleanup()

    async def sender(self, target, message):
        self.sent.append((target, message))

    async def select(self, service, conversation=("group", 123), sticker_id="开心.png"):
        session = StickerSession(conversation)
        listed = await service.execute_tool(session, "list_stickers", {})
        self.assertTrue(listed["ok"])
        selected = await service.execute_tool(session, "send_sticker", {"sticker_id": sticker_id})
        self.assertTrue(selected["ok"], selected)
        return session

    def sticker_responses(self, final="", duplicate=False):
        calls = [tool_call("send_sticker", {"sticker_id": "开心.png"}, "choose")]
        if duplicate:
            calls.append(calls[0])
        return [call_message(tool_call("list_stickers", {})), call_message(*calls),
                {"role": "assistant", "content": final}]

    async def test_catalog_metadata_query_pagination_and_selfie_exclusion(self):
        (self.images / "自拍.png").write_bytes(PNG)
        (self.images / "hidden.png").write_bytes(PNG)
        (self.images / "manifest.json").write_text(json.dumps({
            "开心.png": {"description": "收到夸奖很得意", "tags": ["嘿嘿"]},
            "hidden.png": {"selfie": True},
        }, ensure_ascii=False))
        session = StickerSession(("group", 1))
        result = await self.service.execute_tool(session, "list_stickers", {"query": "得意"})
        self.assertEqual([item["id"] for item in result["stickers"]], ["开心.png"])
        all_items = await self.service.execute_tool(session, "list_stickers", {})
        self.assertEqual(all_items["total"], 2)
        for index in range(21):
            (self.images / f"extra{index:02}.png").write_bytes(PNG)
        page = await self.service.execute_tool(session, "list_stickers", {})
        self.assertEqual(len(page["stickers"]), 20)
        self.assertEqual(page["next_offset"], 20)
        last = await self.service.execute_tool(session, "list_stickers", {"offset": 20})
        self.assertEqual(len(last["stickers"]), 3)
        self.assertIsNone(last["next_offset"])

    async def test_requires_list_and_rejects_paths_urls_and_recipient_override(self):
        session = StickerSession(("group", 1))
        direct = await self.service.execute_tool(session, "send_sticker", {"sticker_id": "开心.png"})
        self.assertFalse(direct["ok"])
        await self.service.execute_tool(session, "list_stickers", {})
        for arguments in ({"sticker_id": "../secret.png"}, {"sticker_id": "/tmp/secret.png"},
                          {"sticker_id": "https://example.com/img.png"},
                          {"sticker_id": "开心.png", "user_id": 999}):
            with self.subTest(arguments=arguments):
                self.assertFalse((await self.service.execute_tool(session, "send_sticker", arguments))["ok"])
        self.assertIsNone(session.selected_image)

    async def test_symlinks_invalid_images_and_oversize_files_are_rejected(self):
        outside = self.root / "outside.png"
        outside.write_bytes(PNG)
        (self.images / "link.png").symlink_to(outside)
        (self.images / "bad.png").write_text("not an image")
        (self.images / "large.png").write_bytes(b"x" * (self.settings.sticker_max_bytes + 1))
        session = StickerSession(("group", 1))
        listing = await self.service.execute_tool(session, "list_stickers", {})
        ids = [item["id"] for item in listing["stickers"]]
        self.assertNotIn("link.png", ids)
        self.assertNotIn("large.png", ids)
        bad = await self.service.execute_tool(session, "send_sticker", {"sticker_id": "bad.png"})
        self.assertFalse(bad["ok"])
        with self.assertRaises(OSError):
            self.service._read_file("link.png", self.settings.sticker_max_bytes)
        if hasattr(os, "mkfifo"):
            os.mkfifo(self.images / "pipe.png")
            with self.assertRaises(ValueError):
                self.service._read_file("pipe.png", self.settings.sticker_max_bytes)

    async def test_one_image_per_request_and_duplicate_selection_is_idempotent(self):
        session = await self.select(self.service)
        again = await self.service.execute_tool(session, "send_sticker", {"sticker_id": "开心.png"})
        self.assertTrue(again["ok"])
        second = await self.service.execute_tool(session, "send_sticker", {"sticker_id": "晚安.gif"})
        self.assertFalse(second["ok"])
        self.assertTrue(await self.service.send_selected(session, 123, self.sender))
        self.assertEqual(base64.b64decode(self.sent[0][1][0]["data"]["file"][9:]), PNG)

    async def test_concurrent_requests_cannot_bypass_shared_group_cooldown(self):
        first = await self.select(self.service)
        second = await self.select(self.service)
        results = await asyncio.gather(
            self.service.send_selected(first, 123, self.sender),
            self.service.send_selected(second, 123, self.sender),
        )
        self.assertEqual(results, [True, False])
        self.assertEqual(len(self.sent), 1)
        private = await self.select(self.service, ("private", 123))
        self.assertTrue(await self.service.send_selected(private, 123, self.sender))

    async def test_repeat_window_and_failed_send_do_not_consume_limit(self):
        session = await self.select(self.service)

        async def failed(target, message):
            raise RuntimeError("disconnected")

        with self.assertRaises(RuntimeError):
            await self.service.send_selected(session, 123, failed)
        with patch("app.stickers.monotonic", return_value=1000):
            self.assertTrue(await self.service.send_selected(session, 123, self.sender))
        with patch("app.stickers.monotonic", return_value=1031):
            another = await self.select(self.service, sticker_id="晚安.gif")
            self.assertTrue(await self.service.send_selected(another, 123, self.sender))
            self.assertIsNotNone(self.service._blocked(("group", 123), "开心.png"))
        with patch("app.stickers.monotonic", return_value=1062):
            self.assertIsNotNone(self.service._blocked(("group", 123), "开心.png"))
        with patch("app.stickers.monotonic", return_value=8201):
            self.assertIsNone(self.service._blocked(("group", 123), "开心.png"))

    async def test_empty_and_invalid_manifest_and_missing_directory(self):
        self.service.directory = self.root / "missing"
        session = StickerSession(("group", 1))
        empty = await self.service.execute_tool(session, "list_stickers", {})
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["stickers"], [])
        self.service.directory = self.images
        (self.images / "manifest.json").write_text("invalid json")
        listing = await self.service.execute_tool(session, "list_stickers", {})
        self.assertEqual(listing["total"], 2)
        for arguments in ({"offset": True}, {"offset": -1}, {"query": 1}, {"user_id": 1}):
            self.assertFalse((await self.service.execute_tool(session, "list_stickers", arguments))["ok"])

    async def test_group_image_only_reply_and_duplicate_tool_call(self):
        self.responses = self.sticker_responses(duplicate=True)
        await self.bot._run_ai_reply(123, 456, "发表情包", self.sender)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0][0], 123)
        self.assertEqual([segment["type"] for segment in self.sent[0][1]], ["image"])
        names = {tool["function"]["name"] for tool in self.requests[0]["tools"]}
        self.assertEqual(names, {"list_stickers", "send_sticker"})

    async def test_text_and_image_are_separate_messages(self):
        self.responses = self.sticker_responses(final="嘿嘿")
        await self.bot._run_ai_reply(123, 456, "发表情包", self.sender)
        self.assertEqual(len(self.sent), 2)
        self.assertEqual([seg["type"] for seg in self.sent[0][1]], ["at", "text"])
        self.assertEqual([seg["type"] for seg in self.sent[1][1]], ["image"])

    async def test_same_request_is_sent_once_even_with_limits_disabled(self):
        self.settings.sticker_cooldown_seconds = 0
        self.settings.sticker_repeat_seconds = 0
        session = await self.select(self.service)
        results = await asyncio.gather(
            self.service.send_selected(session, 123, self.sender),
            self.service.send_selected(session, 123, self.sender),
        )
        self.assertEqual(results, [True, False])
        self.assertEqual(len(self.sent), 1)

    async def test_onebot_wire_format_for_group_and_private_images(self):
        actions = []

        class WebSocket:
            async def send_json(self, payload):
                actions.append(payload)

        connection = OneBotConnection(WebSocket())
        group = await self.select(self.service, ("group", 123))
        private = await self.select(self.service, ("private", 456))
        await self.service.send_selected(group, 123, connection.send_group)
        await self.service.send_selected(private, 456, connection.send_private)
        self.assertEqual([action["action"] for action in actions], ["send_group_msg", "send_private_msg"])
        self.assertEqual(actions[0]["params"]["group_id"], 123)
        self.assertEqual(actions[1]["params"]["user_id"], 456)
        for action in actions:
            self.assertEqual(action["params"]["message"][0]["type"], "image")
            self.assertEqual(base64.b64decode(action["params"]["message"][0]["data"]["file"][9:]), PNG)

    async def test_automatic_silence_discards_queued_image(self):
        self.responses = self.sticker_responses(final="[不插话]")
        await self.bot._run_ai_reply(123, 0, "自动接话", self.sender, system=AUTO_CHAT_SYSTEM_PROMPT, automatic=True)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.bot.stickers._last_sent, {})
        self.responses = self.sticker_responses()
        await self.bot._run_ai_reply(123, 0, "自动接话", self.sender, system=AUTO_CHAT_SYSTEM_PROMPT, automatic=True)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual([seg["type"] for seg in self.sent[0][1]], ["image"])

    async def test_private_stickers_and_reminders_keep_receipts_and_owner(self):
        due = (datetime.now(timezone(timedelta(hours=8))) + timedelta(minutes=5)).isoformat()
        self.responses = [call_message(
            tool_call("list_stickers", {}, "list"),
            tool_call("create_reminder", {"content": "喝水", "due_at": due}, "reminder"),
        ), call_message(tool_call("send_sticker", {"sticker_id": "开心.png"}, "image")),
            {"role": "assistant", "content": ""}]
        await self.bot._run_ai_reply(456, 456, "提醒并发表情", self.sender, system=PRIVATE_SYSTEM_PROMPT, private=True)
        self.assertEqual(len(self.db.list_reminders(456)), 1)
        self.assertEqual(self.db.list_reminders(789), [])
        self.assertEqual(len(self.sent), 2)
        self.assertIn("已保存提醒 #", self.sent[0][1][0]["data"]["text"])
        self.assertEqual(self.sent[1][0], 456)
        self.assertEqual(self.sent[1][1][0]["type"], "image")
        self.assertEqual(self.db.recent_private_messages(456, 1)[0]["content"], "[表情包：开心.png]")

    async def test_model_failure_after_selection_still_sends_image(self):
        self.responses = self.sticker_responses()[:-1] + [503]
        await self.bot._run_ai_reply(123, 456, "发表情包", self.sender)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0][1][0]["type"], "image")

    async def test_blank_response_without_selected_image_is_an_error(self):
        self.responses = [{"role": "assistant", "content": ""}]
        with self.assertRaises(LLMUnavailable):
            await self.llm.complete("你好")

    async def test_mimic_and_disabled_stickers_have_no_sticker_tools(self):
        self.responses = [{"role": "assistant", "content": "模仿结果"}]
        await self.bot._run_ai_reply(123, 456, "模仿", self.sender, system="独立模仿提示词")
        self.assertNotIn("tools", self.requests[0])
        self.settings.sticker_enabled = False
        self.responses = [{"role": "assistant", "content": "你好"}]
        await self.bot._run_ai_reply(456, 456, "你好", self.sender, system=PRIVATE_SYSTEM_PROMPT, private=True)
        names = {item["function"]["name"] for item in self.requests[-1]["tools"]}
        self.assertEqual(names, {"create_reminder", "list_reminders", "cancel_reminder"})


if __name__ == "__main__":
    unittest.main()
