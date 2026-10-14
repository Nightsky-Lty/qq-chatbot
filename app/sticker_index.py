"""表情库后台入库：内容哈希去重、视觉标注、SQLite缓存及保守隔离。"""

import asyncio
import base64
import hashlib
import logging
import time
from typing import Any

from app.config import Settings
from app.database import Database
from app.llm import DeepSeekClient
from app.stickers import StickerService, file_signature, image_mime

logger = logging.getLogger(__name__)


class StickerIndexer:
    def __init__(self, settings: Settings, database: Database, llm: DeepSeekClient, stickers: StickerService):
        self.settings = settings
        self.db = database
        self.llm = llm
        self.stickers = stickers
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if (self.settings.sticker_enabled and self._task is None
                and (self.stickers.auto_index_enabled or self.settings.sticker_collect_enabled
                     or self.stickers.collected_directory.is_dir())):
            self._task = asyncio.create_task(self._run(), name="sticker-indexer")
            logger.info("Sticker index refresh enabled: automatic=%s poll=%ss batch=%s",
                        self.stickers.auto_index_enabled, self.settings.sticker_index_poll_seconds, self.settings.sticker_index_batch_size)

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    def _read_snapshot(self, filename: str, signature: tuple) -> tuple[str, str, str]:
        """上传与哈希来自同一份字节；读前后的签名必须与扫描一致。"""
        path = self.stickers.path_for_id(filename)
        if file_signature(path.lstat()) != signature:
            raise ValueError("image changed")
        data = self.stickers._read_file(filename, self.settings.sticker_max_bytes)
        if file_signature(path.lstat()) != signature:
            raise ValueError("image changed")
        return hashlib.sha256(data).hexdigest(), image_mime(data), base64.b64encode(data).decode("ascii")

    async def scan_once(self) -> None:
        """逐张处理，每轮限制API次数；未变化文件复用哈希，未就绪图片不进入聊天目录。"""
        items = await asyncio.to_thread(self.stickers._catalog, raw=True)
        if not self.stickers.auto_index_enabled:
            # 关闭自动识别时仍恢复已收藏、已审核的缓存，不上传新图。
            items = [item for item in items if item["id"].startswith("collected:")]
        previous = self.stickers.annotations
        known: dict[str, dict[str, Any]] = {}
        # 先移除被删除、禁用或移出扫描范围的文件，缓存记录仍按内容保留。
        active = {item["id"] for item in items}
        self.stickers.annotations = {name: value for name, value in previous.items()
                                    if name in active or (name.startswith("collected:") and self.stickers.path_for_id(name).is_file())}
        calls_used = 0
        for item in items:
            filename, signature = item["id"], item["_signature"]
            cached = previous.get(filename, {})
            mime = encoded = ""
            try:
                if cached.get("signature") == signature:
                    content_hash = cached["hash"]
                else:
                    content_hash, mime, encoded = await asyncio.to_thread(self._read_snapshot, filename, signature)
                record = self.db.get_sticker_annotation(content_hash)
                state = {"signature": signature, "hash": content_hash, "record": record}
                known[filename] = state
                # 新hash立即替换旧记录，文件替换后不能沿用旧的安全判断。
                self.stickers.annotations = {**self.stickers.annotations, filename: state}
                if record and (record["status"] != "error" or record["next_attempt_at"] > time.time()):
                    continue
                if not self.stickers.auto_index_enabled:
                    continue
                if not self.settings.deepseek_api_key or calls_used >= self.settings.sticker_index_batch_size:
                    continue
                if not encoded:
                    actual_hash, mime, encoded = await asyncio.to_thread(self._read_snapshot, filename, signature)
                    if actual_hash != content_hash:
                        continue
                calls_used += 1
                try:
                    metadata = await self.llm.annotate_sticker(mime, encoded, require_sticker=filename.startswith("collected:"))
                except Exception as exc:
                    attempts = (record["attempts"] if record else 0) + 1
                    delay = min(86400, self.settings.sticker_index_retry_seconds * 2 ** min(attempts - 1, 5))
                    self.db.save_sticker_annotation(content_hash, "error", {}, attempts=attempts,
                                                    next_attempt_at=int(time.time()) + delay)
                    logger.warning("Sticker indexing deferred: error=%s retry_in=%ss", type(exc).__name__, delay)
                else:
                    # 有风险提示、自画像/自拍或模型不确定时一律不自动使用。
                    safe = metadata["safe"] and not metadata["selfie"] and not metadata["warnings"]
                    if filename.startswith("collected:"):
                        safe = safe and metadata.get("is_sticker") is True
                    status = "ready" if safe else "quarantined"
                    self.db.save_sticker_annotation(content_hash, status, metadata)
                    logger.info("Sticker annotation stored: status=%s", status)
                state = {**state, "record": self.db.get_sticker_annotation(content_hash)}
                known[filename] = state
                self.stickers.annotations = {**self.stickers.annotations, filename: state}
            except (OSError, ValueError) as exc:
                # 文件上传尚未完成、权限不足或无效格式只影响这一张。
                logger.warning("Sticker file skipped: error=%s", type(exc).__name__)
        # 扫描期间收藏工具可能新增文件，不能用旧扫描快照抹掉它的已审核索引。
        added = {name: value for name, value in self.stickers.annotations.items()
                 if name not in active and name.startswith("collected:") and self.stickers.path_for_id(name).is_file()}
        self.stickers.annotations = {**added, **known}

    async def _run(self) -> None:
        while True:
            try:
                await self.scan_once()
            except Exception as exc:
                logger.warning("Sticker index scan failed: error=%s", type(exc).__name__)
            await asyncio.sleep(self.settings.sticker_index_poll_seconds)
