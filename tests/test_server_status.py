import asyncio
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.bot import BotService, HELP_TEXT, PRIVATE_HELP_TEXT
from app.config import Settings
from app.database import Database
from app.llm import DeepSeekClient
from app.server_status import ServerStatusService, cpu_counters, cpu_percent, memory_values


CPU_BEFORE = "cpu  100 0 20 700 100 10 10 0 50 10\ncpu0 0\ncpu1 0\n"
CPU_AFTER = "cpu  130 0 30 740 110 15 15 0 999 50\ncpu0 0\ncpu1 0\n"
MEMORY = "MemTotal: 8388608 kB\nMemAvailable: 2097152 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
Disk = namedtuple("Disk", "total used free")
GIB = 1024 ** 3


class StatusMetricTests(unittest.IsolatedAsyncioTestCase):
    def service(self):
        service = ServerStatusService(Settings(_env_file=None))
        service.SAMPLE_SECONDS = 0
        return service

    def metrics(self):
        return {"cpu": cpu_counters(CPU_AFTER), "memory": memory_values(MEMORY),
                "disk": Disk(20 * GIB, 8 * GIB, 12 * GIB), "load": (0.1, 0.2, 0.3), "uptime": 90061.0}

    def test_cpu_sampling_excludes_double_counted_guest_and_handles_reset(self):
        before, cores = cpu_counters(CPU_BEFORE)
        after, _ = cpu_counters(CPU_AFTER)
        self.assertEqual(cores, 2)
        self.assertEqual(len(before), 8)
        self.assertEqual(cpu_percent(before, after), 50)
        self.assertIsNone(cpu_percent(before, before))
        self.assertIsNone(cpu_percent(after, before))
        self.assertIsNone(cpu_percent(before, after[:4]))

    def test_format_labels_host_memory_disk_scope_and_time(self):
        service = self.service()
        text = service._format(cpu_counters(CPU_BEFORE), self.metrics())
        for expected in ("宿主机CPU：50.0%", "2逻辑核", "6.00 GiB / 8.00 GiB（75.0%）",
                         "Swap：未启用", "数据卷所在文件系统", "8.00 GiB / 20.00 GiB（40.0%）",
                         "0.10 / 0.20 / 0.30", "1天1小时1分", "北京时间"):
            self.assertIn(expected, text)
        self.assertNotIn("/host", text)
        self.assertNotIn("127.0.0.1", text)

    async def test_concurrent_queries_share_sample_and_cache(self):
        service = self.service()
        with patch.object(service, "_cpu", return_value=cpu_counters(CPU_BEFORE)) as first, \
             patch.object(service, "_collect", return_value=self.metrics()) as second:
            texts = await asyncio.gather(service.get_text(), service.get_text())
            self.assertIn("50.0%", texts[0])
            self.assertTrue(texts[1].endswith("（10秒内缓存）"))
            self.assertEqual(first.call_count, 1)
            self.assertEqual(second.call_count, 1)
            await service.get_text()
            self.assertEqual(second.call_count, 1)
            service._cached = (0, "expired")
            await service.get_text()
            self.assertEqual(second.call_count, 2)

    def test_partial_unavailable_is_not_fabricated_as_zero(self):
        service = self.service()
        text = service._format(None, {"cpu": None, "memory": {}, "disk": self.metrics()["disk"]})
        self.assertIn("宿主机CPU：暂不可用", text)
        self.assertIn("宿主机内存：暂不可用", text)
        self.assertIn("只读统计文件挂载", text)
        self.assertIn("8.00 GiB / 20.00 GiB", text)
        self.assertNotIn("CPU：0.0%", text)

    def test_collect_validates_fixed_proc_files_and_invalid_values(self):
        service = self.service()
        files = {"stat": CPU_AFTER, "meminfo": MEMORY, "loadavg": "0.1 0.2 0.3 1/999 12345",
                 "uptime": "90061 80000"}
        with patch.object(service, "_read", side_effect=lambda name: files.get(name)), \
             patch("app.server_status.shutil.disk_usage", return_value=self.metrics()["disk"]):
            metrics = service._collect()
        self.assertEqual(metrics["load"], (0.1, 0.2, 0.3))
        self.assertEqual(metrics["uptime"], 90061)
        files.update(loadavg="nan inf -1", uptime="nan")
        with patch.object(service, "_read", side_effect=lambda name: files.get(name)), \
             patch("app.server_status.shutil.disk_usage", side_effect=OSError):
            metrics = service._collect()
        self.assertNotIn("load", metrics)
        self.assertNotIn("uptime", metrics)
        self.assertNotIn("disk", metrics)

    def test_permission_failure_and_invalid_memory_do_not_leak_paths(self):
        service = self.service()
        with patch("pathlib.Path.open", side_effect=PermissionError("/private/secret")):
            self.assertIsNone(service._read("stat"))
        metrics = self.metrics()
        metrics["memory"] = {"MemTotal": 1, "MemAvailable": 2, "SwapTotal": 2, "SwapFree": 3}
        text = service._format(None, metrics)
        self.assertIn("内存：暂不可用", text)
        self.assertIn("Swap：暂不可用", text)
        self.assertNotIn("secret", text)


class StatusCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.folder.name) / "db.sqlite")
        self.cfg = Settings(_env_file=None, admin_qq_ids={11}, sticker_auto_index_enabled=False)
        self.llm = DeepSeekClient(self.cfg)
        self.llm.request_message = AsyncMock(side_effect=AssertionError("状态命令不能请求LLM"))
        self.bot = BotService(self.cfg, self.db, self.llm)
        self.bot.server_status.get_text = AsyncMock(return_value="服务器资源状态\n宿主机CPU：5.0%")
        self.sent = []
        async def sender(target, segments):
            self.sent.append((target, segments[0]["data"]["text"]))
        self.sender = sender

    async def asyncTearDown(self):
        await self.bot.close()
        await self.llm.close()
        self.db.close()
        self.folder.cleanup()

    def event(self, *, private=False, user=11, text="/服务器状态", mention=False):
        message = [{"type": "text", "data": {"text": text}}]
        if mention:
            message.insert(0, {"type": "at", "data": {"qq": "99"}})
        return {"post_type": "message", "message_type": "private" if private else "group",
                "user_id": user, "self_id": 99, "group_id": 7, "message_id": 123,
                "message": message, "sender": {"role": "owner", "nickname": "A"}}

    async def drain(self):
        tasks = list(self.bot._status_tasks)
        if tasks:
            await asyncio.gather(*tasks)
        await asyncio.sleep(0)  # 完成回调清理。

    async def test_group_mention_and_private_alias_use_command_not_llm_or_history(self):
        await self.bot.handle_event(self.event(mention=True), self.sender, self.sender)
        await self.drain()
        await self.bot.handle_event(self.event(private=True, text="/状态"), self.sender, self.sender)
        await self.drain()
        self.assertEqual([target for target, _ in self.sent], [7, 11])
        self.assertTrue(all("CPU：5.0%" in text for _, text in self.sent))
        self.assertEqual(self.db.recent_messages(7, 10), [])
        self.assertEqual(self.db.recent_private_messages(11, 10), [])
        self.llm.request_message.assert_not_called()
        self.assertFalse(self.bot._status_tasks)
        self.assertFalse(self.bot._pending_status_users)
        self.assertIn("/服务器状态", HELP_TEXT)
        self.assertIn("/服务器状态", PRIVATE_HELP_TEXT)

    async def test_group_owner_not_bot_admin_is_rejected_without_sampling(self):
        await self.bot.handle_event(self.event(user=22), self.sender, self.sender)
        self.assertIn("只有机器人管理员", self.sent[0][1])
        self.bot.server_status.get_text.assert_not_called()
        self.assertEqual(self.db.recent_messages(7, 10), [])

    async def test_pending_query_is_nonblocking_and_duplicate_is_not_resampled(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def collect():
            started.set()
            await release.wait()
            return "服务器资源状态"
        self.bot.server_status.get_text = AsyncMock(side_effect=collect)
        await asyncio.wait_for(self.bot.handle_event(self.event(), self.sender), 1)
        await started.wait()
        await self.bot.handle_event(self.event(), self.sender)
        self.assertIn("正在采集中", self.sent[0][1])
        release.set()
        await self.drain()
        self.bot.server_status.get_text.assert_awaited_once()

    async def test_disconnect_cancels_queued_reply(self):
        started = asyncio.Event()
        async def collect():
            started.set()
            await asyncio.Event().wait()
        self.bot.server_status.get_text = AsyncMock(side_effect=collect)
        await self.bot.handle_event(self.event(), self.sender)
        await started.wait()
        await self.bot.cancel_pending(self.sender)
        self.assertFalse(self.bot._status_tasks)
        self.assertFalse(self.bot._pending_status_users)
        self.assertEqual(self.sent, [])

    async def test_confirmation_sender_and_fixed_failure_message(self):
        confirmed = AsyncMock()
        self.bot.server_status.get_text = AsyncMock(side_effect=OSError("/private/secret"))
        await self.bot.handle_event(self.event(), self.sender, send_group_confirmed=confirmed)
        await self.drain()
        confirmed.assert_awaited_once()
        target, segments = confirmed.call_args.args
        self.assertEqual(target, 7)
        self.assertIn("采集失败", segments[0]["data"]["text"])
        self.assertNotIn("secret", segments[0]["data"]["text"])
        self.assertEqual(self.sent, [])


if __name__ == "__main__":
    unittest.main()
