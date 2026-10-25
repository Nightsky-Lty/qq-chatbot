"""收藏工具与随机收图共用：SHA-256去重、视觉审核和原子落盘。"""

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

import httpx

from app.chat_images import ChatImageService
from app.config import Settings
from app.database import Database
from app.llm import DeepSeekClient
from app.stickers import EXTENSIONS, StickerService, file_signature

logger = logging.getLogger(__name__)
Sender = Callable[[int, list[dict[str, Any]]], Awaitable[Any]]

COLLECT_STICKER_TOOL = {"type": "function", "function": {
    "name": "collect_sticker",
    "description": "收藏本轮授权的当前/近期会话图片到共享图库，群聊仅同群图片，私聊仅当前用户图片。私聊在用户明确要求保存时调用；图库可在群聊和私聊复用。按来源确定指代，先去重再独立审核，真实成功前不能声称已收藏；每轮最多尝试一张。",
    "parameters": {"type": "object", "properties": {
        "image_id": {"type": "string", "minLength": 1, "maxLength": 32,
                     "description": "程序列出的本轮可收藏图片编号，如media_101或image_1；不能传未列出的编号、URL、路径或自选群号/用户号。"},
    }, "required": ["image_id"], "additionalProperties": False},
}}

COLLECT_INSTRUCTIONS = """【收藏会话表情】
群聊中用户要求或同群图片适合当表情时可调用collect_sticker；私聊在当前用户明确要求保存时调用，仅用其本人当前/近期或检索后真实授权的图片编号。不能跨群、跨私聊用户或收藏网页图。每轮一张，失败不换图绕限制。
收藏进入共享图库，可在群聊和私聊复用；私聊不随机收图，不因收到图片就自动收藏。
收藏独立审核，不必先看图；概要、图片缓存及概率抽签都不是收藏成功。仅ok=true且created=true可说新收藏成功，重复表示已有，其余如实说明。随后文字确认，本轮工具收藏不另发后台播报；收藏不是QQ账号收藏或向群发图。"""


@dataclass(slots=True)
class CollectionJob:
    """group_id为来源命名空间：群聊用正群号，私聊用负QQ号。"""
    group_id: int
    user_id: int
    bot_id: int
    message_id: str
    image: dict[str, str]
    sender: Sender
    queued_at: float = field(default_factory=monotonic)


Announcement = Callable[[CollectionJob], Awaitable[None]]


@dataclass(slots=True)
class CollectionSession:
    jobs: dict[str, CollectionJob]
    cached: dict[str, dict[str, Any]] = field(default_factory=dict)
    attempted: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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
        # 两条入口必须串行完成检查/审核/写入，防止重复审核和突破容量。
        self._collect_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return (not self._closed and self.settings.sticker_enabled
                and self.settings.sticker_vision_enabled and self.settings.sticker_collect_enabled)

    def submit(self, group_id: int, user_id: int, bot_id: int, message_id: str,
               images: list[dict[str, str]], sender: Sender) -> None:
        """每张图片独立抽签；队列只存引用，不在OneBot接收循环下载/等待LLM。"""
        if group_id <= 0 or not self.enabled or self.settings.sticker_collect_probability <= 0:
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

    def _cooling(self, group_id: int) -> bool:
        return monotonic() - self._last_reviewed.get(group_id, float("-inf")) < self.settings.sticker_collect_cooldown_seconds

    @staticmethod
    def _rejected(status: str, error: str) -> dict[str, Any]:
        return {"ok": False, "created": False, "status": status, "error": error}

    async def execute_tool(self, session: CollectionSession, arguments: dict[str, Any],
                           *, loaded_images: dict[str, tuple[str, bytes]] | None = None) -> dict[str, Any]:
        image_id = arguments.get("image_id")
        if not isinstance(image_id, str) or image_id not in session.jobs or set(arguments) != {"image_id"}:
            return self._rejected("invalid_image", "只能收藏程序列出的本轮当前会话图片编号。")
        async with session.lock:
            if image_id in session.cached:
                return session.cached[image_id]
            if session.attempted:
                return self._rejected("limit", "本轮最多尝试收藏一张，不换图绕过限制。")
            session.attempted = True
            try:
                # 聊天Agent已持有AI并发额度，不能再申请同一个额度，否则单并发会死锁。
                result = await self._collect(session.jobs[image_id], source="agent",
                                             loaded=(loaded_images or {}).get(image_id))
            except (ValueError, httpx.HTTPError, OSError, TimeoutError) as exc:
                logger.warning("Sticker collection tool failed: error=%s", type(exc).__name__)
                result = self._rejected("unavailable", "图片获取或收藏失败，可能链接过期、格式不支持或图库不可写；未确认收藏成功。")
            session.cached[image_id] = result
            return result

    async def _collect(self, job: CollectionJob, *, source: str = "random",
                       loaded: tuple[str, bytes] | None = None) -> dict[str, Any]:
        async with self._collect_lock:
            if not self.enabled:
                return self._rejected("disabled", "收藏已关闭或视觉审核不可用。")
            if ((job.group_id < 0 and job.user_id != -job.group_id)
                    or ('cache_id' in job.image and str(job.image.get('namespace')) != str(job.group_id))):
                return self._rejected('invalid_image', '图片不属于本轮当前会话，未收藏。')
            if ("cache_id" in job.image and self.images.cached_validator is not None
                    and not self.images.cached_validator(job.image)):
                return self._rejected("expired", "图片缓存已过期、已清理或来源消息已删除。")
            if monotonic() - job.queued_at > self.settings.sticker_collect_queue_wait_seconds:
                return self._rejected("expired", "当前图片已超过收藏处理时限，请重新发送。")
            # 随机入口冷却时不下载；工具仍可先确认是否已经入库，但不能绕过审核冷却。
            if source == "random" and self._cooling(job.group_id):
                return self._rejected("cooldown", "本会话图片审核冷却中，未新收藏。")
            return await self._collect_locked(job, source=source, loaded=loaded)

    async def _collect_locked(self, job: CollectionJob, *, source: str,
                              loaded: tuple[str, bytes] | None) -> dict[str, Any]:
        mime, data = loaded if loaded is not None else await self.images.load(job.image)
        if len(data) > self.images.max_bytes:
            return self._rejected("too_large", "图片超过收藏大小限制。")
        content_hash = hashlib.sha256(data).hexdigest()
        if await asyncio.to_thread(self._find_duplicate, content_hash):
            logger.info("Sticker collection skipped: reason=duplicate group=%s", job.group_id)
            return {"ok": True, "created": False, "status": "duplicate", "note": "图库已有相同内容，不重新审核或保存，不播报新收藏成功。"}
        if self._cooling(job.group_id):
            return self._rejected("cooldown", "本会话图片审核冷却中，未新收藏。")
        if not await asyncio.to_thread(self._has_room, len(data)):
            logger.info("Sticker collection skipped: reason=capacity group=%s", job.group_id)
            return self._rejected("capacity", "收藏图库容量已满，不自动删除旧图。")
        record = self.db.get_sticker_annotation(content_hash)
        if record and (record["status"] == "quarantined"
                       or (record["status"] == "error" and record["next_attempt_at"] > time())):
            return self._rejected("review_blocked", "该图片已被审核拒绝或仍在审核失败退避期，不能重复提交审核。")
        if len(await asyncio.to_thread(self.stickers._catalog, raw=True)) >= 500:
            return self._rejected("capacity", "人工和收藏图库已达到扫描数量上限。")
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
                return self._rejected("review_error", "图片审核失败，尚未收藏；暂不重复审核。")
        if (not metadata.get("safe") or metadata.get("selfie") is not False
                or metadata.get("warnings") or metadata.get("is_sticker") is not True):
            self.db.save_sticker_annotation(content_hash, "quarantined", metadata)
            logger.info("Sticker collection skipped: reason=review_rejected group=%s", job.group_id)
            return self._rejected("review_rejected", "未通过表情图及安全审核，未收藏；不保存普通照片、自拍、隐私截图或风险不明图片。")
        extension = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}[mime]
        if ("cache_id" in job.image and self.images.cached_validator is not None
                and not self.images.cached_validator(job.image)):
            return self._rejected("expired", "审核期间图片来源失效，未收藏。")
        sticker_id = f"collected:{content_hash}.{extension}"
        self.db.save_sticker_annotation(content_hash, "ready", metadata)
        write = asyncio.create_task(asyncio.to_thread(self._write_new, sticker_id, data))
        try:
            created = await asyncio.shield(write)
        except asyncio.CancelledError:
            # 写盘线程无法取消，完成前不释放收藏锁，否则另一入口可能重入。
            await write
            raise
        signature = file_signature(self.stickers.path_for_id(sticker_id).stat())
        self.stickers.annotations = {**self.stickers.annotations, sticker_id: {
            "hash": content_hash, "signature": signature, "record": self.db.get_sticker_annotation(content_hash),
        }}
        logger.info("Sticker collected: group=%s created=%s source=%s", job.group_id, created, source)
        return {"ok": True, "created": created, "status": "collected" if created else "duplicate",
                "sticker_id": sticker_id,
                "note": "审核通过并新入库，可简短确认，不额外后台播报。" if created else "相同图片已在图库，不重复播报新收藏成功。"}

    async def _run(self) -> None:
        while not self.queue.empty() and not self._closed:
            job = self.queue.get_nowait()
            self._current = job
            try:
                async with self.slots:
                    if monotonic() - job.queued_at > self.settings.sticker_collect_queue_wait_seconds:
                        continue
                    async with asyncio.timeout(self.settings.agent_timeout_seconds):
                        if (await self._collect(job)).get("created") is True:
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
