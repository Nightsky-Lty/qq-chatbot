"""程序随机收图：有界后台队列、SHA-256去重、视觉审核和原子落盘。"""

import asyncio
import base64
import hashlib
import logging
import os
import random
import stat
import tempfile
from dataclasses import dataclass, field
from time import monotonic, time
from typing import Any, Awaitable, Callable

from app.chat_images import ChatImageService
from app.config import Settings
from app.database import Database
from app.llm import DeepSeekClient
from app.stickers import EXTENSIONS, StickerService, file_signature

logger = logging.getLogger(__name__)
Sender = Callable[[int, list[dict[str, Any]]], Awaitable[Any]]


@dataclass(slots=True)
class CollectionJob:
    group_id: int
    user_id: int
    bot_id: int
    message_id: str
    image: dict[str, str]
    sender: Sender
    queued_at: float = field(default_factory=monotonic)


Announcement = Callable[[CollectionJob], Awaitable[None]]


class StickerCollector:
    def __init__(self, settings: Settings, db: Database, llm: DeepSeekClient, stickers: StickerService,
                 *, slots: asyncio.Semaphore, announce: Announcement):
        self.settings, self.db, self.llm, self.stickers = settings, db, llm, stickers
        self.slots, self.announce = slots, announce
        self.images = ChatImageService(settings)
        self.images.max_bytes = min(settings.chat_image_max_bytes, settings.sticker_max_bytes)
        self.queue: asyncio.Queue[CollectionJob] = asyncio.Queue(maxsize=settings.sticker_collect_queue_size)
        self._task: asyncio.Task[None] | None = None
        self._current: CollectionJob | None = None
        self._closed = False
        self._last_reviewed: dict[int, float] = {}
        self._file_hashes: dict[str, tuple[tuple, str]] = {}

    @property
    def enabled(self) -> bool:
        return (not self._closed and self.settings.sticker_enabled
                and self.settings.sticker_vision_enabled and self.settings.sticker_collect_enabled
                and self.settings.sticker_collect_probability > 0)

    def submit(self, group_id: int, user_id: int, bot_id: int, message_id: str,
               images: list[dict[str, str]], sender: Sender) -> None:
        """每张图片独立抽签；队列只存引用，不在OneBot接收循环下载/等待LLM。"""
        if not self.enabled:
            return
        now = monotonic()
        self._last_reviewed = {group: at for group, at in self._last_reviewed.items()
                               if now - at < self.settings.sticker_collect_cooldown_seconds}
        if group_id in self._last_reviewed:
            return
        for ref in images:
            if random.random() >= self.settings.sticker_collect_probability:
                continue
            try:
                self.queue.put_nowait(CollectionJob(group_id, user_id, bot_id, message_id, dict(ref), sender))
            except asyncio.QueueFull:
                logger.info("Sticker collection skipped: reason=queue_full group=%s", group_id)
                break
            logger.info("Sticker collection sampled: group=%s probability=%.3f",
                        group_id, self.settings.sticker_collect_probability)
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if not self._closed and not self.queue.empty() and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._run(), name="random-sticker-collector")

    def _find_duplicate(self, content_hash: str) -> bool:
        """扫描实际文件，包含未标注/禁用的人工图片；变化文件重新计算哈希。"""
        known = {}
        duplicate = False
        annotations = self.stickers.annotations
        for prefix, directory in (("", self.stickers.directory), ("collected:", self.stickers.collected_directory)):
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                sticker_id = prefix + path.name
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= self.settings.sticker_max_bytes:
                    continue
                signature = file_signature(info)
                cached = self._file_hashes.get(sticker_id)
                indexed = annotations.get(sticker_id, {})
                if cached and cached[0] == signature:
                    digest = cached[1]
                elif indexed.get("signature") == signature and indexed.get("hash"):
                    digest = indexed["hash"]
                else:
                    data = self.stickers._read_file(sticker_id, self.settings.sticker_max_bytes)
                    if file_signature(path.lstat()) != signature:
                        raise ValueError("图库文件在去重期间变化")
                    digest = hashlib.sha256(data).hexdigest()
                known[sticker_id] = (signature, digest)
                duplicate |= digest == content_hash
        self._file_hashes = known
        return duplicate

    def _has_room(self, size: int) -> bool:
        directory = self.stickers.collected_directory
        count = total = 0
        if directory.is_dir():
            for path in directory.iterdir():
                info = path.lstat()
                if stat.S_ISREG(info.st_mode):
                    count += 1
                    total += info.st_size
        return (count < self.settings.sticker_collect_max_files
                and total + size <= self.settings.sticker_collect_max_disk_bytes)

    def _write_new(self, sticker_id: str, data: bytes) -> bool:
        directory = self.stickers.collected_directory
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        target = self.stickers.path_for_id(sticker_id)
        if target.exists() or target.is_symlink():
            if self.stickers._read_file(sticker_id, self.settings.sticker_max_bytes) != data:
                raise ValueError("已有收藏文件内容异常，请部署者检查。")
            return False
        if not self._has_room(len(data)):
            raise ValueError("收藏图库容量已满，请部署者清理，不自动删除旧图片。")
        descriptor, temporary = tempfile.mkstemp(prefix=".collect-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target, follow_symlinks=False)
        finally:
            os.unlink(temporary)
        return True

    async def _collect(self, job: CollectionJob) -> bool:
        if monotonic() - self._last_reviewed.get(job.group_id, float("-inf")) < self.settings.sticker_collect_cooldown_seconds:
            return False
        mime, data = await self.images.load(job.image)
        content_hash = hashlib.sha256(data).hexdigest()
        if await asyncio.to_thread(self._find_duplicate, content_hash):
            logger.info("Sticker collection skipped: reason=duplicate group=%s", job.group_id)
            return False
        if not await asyncio.to_thread(self._has_room, len(data)):
            logger.info("Sticker collection skipped: reason=capacity group=%s", job.group_id)
            return False
        record = self.db.get_sticker_annotation(content_hash)
        if record and (record["status"] == "quarantined"
                       or (record["status"] == "error" and record["next_attempt_at"] > time())):
            return False
        if len(await asyncio.to_thread(self.stickers._catalog, raw=True)) >= 500:
            return False
        # 尝试审核也占用冷却，连续收到不合格图片时不连续消耗API。
        self._last_reviewed[job.group_id] = monotonic()
        metadata = (record or {}).get("metadata", {})
        if not (record and record["status"] == "ready" and metadata.get("is_sticker") is True):
            try:
                metadata = await self.llm.annotate_sticker(mime, base64.b64encode(data).decode("ascii"), require_sticker=True)
            except Exception as exc:
                attempts = (record["attempts"] if record else 0) + 1
                delay = min(86400, self.settings.sticker_index_retry_seconds * 2 ** min(attempts - 1, 5))
                self.db.save_sticker_annotation(content_hash, "error", {}, attempts=attempts,
                                                next_attempt_at=int(time()) + delay)
                logger.warning("Sticker collection review failed: error=%s", type(exc).__name__)
                return False
        if (not metadata.get("safe") or metadata.get("selfie") is not False
                or metadata.get("warnings") or metadata.get("is_sticker") is not True):
            self.db.save_sticker_annotation(content_hash, "quarantined", metadata)
            logger.info("Sticker collection skipped: reason=review_rejected group=%s", job.group_id)
            return False
        extension = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}[mime]
        sticker_id = f"collected:{content_hash}.{extension}"
        self.db.save_sticker_annotation(content_hash, "ready", metadata)
        created = await asyncio.to_thread(self._write_new, sticker_id, data)
        signature = file_signature(self.stickers.path_for_id(sticker_id).stat())
        self.stickers.annotations = {**self.stickers.annotations, sticker_id: {
            "hash": content_hash, "signature": signature, "record": self.db.get_sticker_annotation(content_hash),
        }}
        logger.info("Sticker collected: group=%s created=%s source=random", job.group_id, created)
        return created

    async def _run(self) -> None:
        while not self.queue.empty() and not self._closed:
            job = self.queue.get_nowait()
            self._current = job
            try:
                async with self.slots:
                    if monotonic() - job.queued_at > self.settings.sticker_collect_queue_wait_seconds:
                        continue
                    async with asyncio.timeout(self.settings.agent_timeout_seconds):
                        if await self._collect(job):
                            await self.announce(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Sticker collection skipped: error=%s", type(exc).__name__)
            finally:
                self._current = None
                self.queue.task_done()

    async def cancel(self, *, sender: Sender | None = None, group_id: int | None = None) -> None:
        def matches(job: CollectionJob) -> bool:
            return (sender is None or job.sender == sender) and (group_id is None or job.group_id == group_id)

        kept = []
        while not self.queue.empty():
            job = self.queue.get_nowait()
            self.queue.task_done()
            if not matches(job):
                kept.append(job)
        for job in kept:
            self.queue.put_nowait(job)
        if self._current is not None and matches(self._current) and self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self._ensure_worker()

    async def close(self) -> None:
        self._closed = True
        await self.cancel()
