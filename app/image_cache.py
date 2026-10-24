"""有界聊天图片缓存：文件按SHA-256去重，消息授权和概要分别持久保存。"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.chat_images import ChatImageService
from app.config import Settings
from app.database import Database
from app.llm import DeepSeekClient

logger = logging.getLogger(__name__)
SCHEMA = """
CREATE TABLE IF NOT EXISTS image_cache_blobs (
 hash TEXT PRIMARY KEY, mime TEXT NOT NULL, size INTEGER NOT NULL,
 summary_json TEXT NOT NULL DEFAULT '{}', summary_state TEXT NOT NULL DEFAULT 'pending',
 attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at INTEGER NOT NULL DEFAULT 0,
 created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS image_cache_refs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, namespace INTEGER NOT NULL, source_id INTEGER NOT NULL,
 user_id INTEGER NOT NULL, message_id TEXT NOT NULL, position INTEGER NOT NULL,
 image_json TEXT NOT NULL, hash TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'pending',
 attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at INTEGER NOT NULL DEFAULT 0,
 created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
 UNIQUE(namespace,source_id,position)
);
CREATE INDEX IF NOT EXISTS idx_image_cache_refs_scope ON image_cache_refs(namespace,source_id,id);
CREATE INDEX IF NOT EXISTS idx_image_cache_refs_hash ON image_cache_refs(hash);
"""


class ImageCache:
    def __init__(self, settings: Settings, db: Database, llm: DeepSeekClient):
        self.settings, self.db, self.llm = settings, db, llm
        self.directory = settings.image_cache_directory
        self.downloader = ChatImageService(settings)
        self._io_lock = asyncio.Lock()
        self._tasks = []
        self._closed = False
        self._last_cleanup = 0.0
        # 与聊天数据库共享事务锁，避免清除聊天时后台重新插入来源记录。
        with db._lock:
            db.connection.executescript(SCHEMA)
            if "retained" not in {r["name"] for r in db.connection.execute("PRAGMA table_info(image_cache_refs)")}:
                db.connection.execute("ALTER TABLE image_cache_refs ADD COLUMN retained INTEGER NOT NULL DEFAULT 0")
                db.connection.commit()

    @property
    def enabled(self):
        return self.settings.image_cache_enabled and not self._closed

    def _valid(self, row) -> bool:
        if row["expires_at"] <= time.time():
            return False
        if row["source_id"] in self.db.memories.blocked_sources(row["namespace"]):
            return False
        ns = row["namespace"]
        table, key, target = ("messages", "group_id", ns) if ns > 0 else ("private_messages", "user_id", -ns)
        present = self.db.connection.execute(
            f"SELECT 1 FROM {table} WHERE id=? AND {key}=? AND role='user'",
            (row["source_id"], target),
        ).fetchone() is not None
        if present:
            return True
        # 原始聊天自然到期后，只有仍被有效长期事件引用的图片可继续保留。
        return bool(row["retained"] and self.db.connection.execute("""SELECT 1 FROM memory_media_links l
          JOIN memory_records m ON m.id=l.event_id WHERE l.namespace=? AND l.ref_id=? AND m.group_id=?
          AND m.status='active' AND (m.expires_at=0 OR m.expires_at>?) LIMIT 1""",
          (ns, row["id"], ns, int(time.time()))).fetchone())

    def _row(self, namespace: int, ref_id: int):
        with self.db._lock:
            row = self.db.connection.execute("SELECT * FROM image_cache_refs WHERE namespace=? AND id=?",
                                             (namespace, ref_id)).fetchone()
            return dict(row) if row is not None and self._valid(row) else None

    def authorized(self, ref: dict) -> bool:
        try:
            row = self._row(int(ref["namespace"]), int(ref["cache_id"]))
            return self.enabled and row is not None and row["status"] in {"pending", "ready"}
        except (ValueError, KeyError, TypeError):
            return False

    def record(self, namespace: int, source_id: int, user_id: int, message_id: str, images: list[dict]) -> list[dict]:
        """仅登记真实消息，最多20张/消息；不在接收循环下载或调用模型。"""
        if not self.enabled or not namespace or source_id <= 0:
            return images
        result = []
        now = int(time.time())
        with self.db._lock, self.db.connection:
            pending = self.db.connection.execute("SELECT count(*) FROM image_cache_refs WHERE status='pending'").fetchone()[0]
            for position, image in enumerate(images[:20]):
                ref = {key: str(image[key])[:cap] for key, cap in (("url", 8192), ("file_size", 20), ("file", 512)) if key in image}
                status = "pending" if pending < self.settings.image_cache_queue_limit else "unavailable"
                self.db.connection.execute(
                    """INSERT OR IGNORE INTO image_cache_refs(namespace,source_id,user_id,message_id,position,image_json,status,created_at,expires_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (namespace, source_id, user_id, message_id, position, json.dumps(ref), status, now,
                     now + self.settings.image_cache_retention_days * 86400))
                row = self.db.connection.execute("SELECT * FROM image_cache_refs WHERE namespace=? AND source_id=? AND position=?",
                                                 (namespace, source_id, position)).fetchone()
                if row and self._valid(row):
                    result.append({"cache_id": str(row["id"]), "namespace": str(namespace)})
                pending += 1
            self.db.connection.execute("""DELETE FROM image_cache_refs WHERE id IN
                (SELECT id FROM image_cache_refs ORDER BY id DESC LIMIT -1 OFFSET ?)""",
                (self.settings.image_cache_max_files * 20,))
        return result

    def rows(self, namespace: int, boundary: int, preferred: set[int] | None = None) -> list[dict]:
        if not self.enabled:
            return []
        with self.db._lock:
            rows = [dict(row) for row in self.db.connection.execute(
                """SELECT r.*,b.summary_json,b.summary_state FROM image_cache_refs r
                   LEFT JOIN image_cache_blobs b ON b.hash=r.hash
                   WHERE r.namespace=? AND r.source_id<=? AND r.expires_at>? ORDER BY r.source_id DESC,r.position LIMIT 200""",
                (namespace, boundary, int(time.time()))) if self._valid(row)]
            if preferred:
                ids = sorted(preferred)[:60]
                extra = self.db.connection.execute(
                    "SELECT r.*,b.summary_json,b.summary_state FROM image_cache_refs r LEFT JOIN image_cache_blobs b ON b.hash=r.hash "
                    + "WHERE r.namespace=? AND r.source_id<=? AND r.expires_at>? AND r.source_id IN ("
                    + ",".join("?" for _ in ids) + ")", (namespace, boundary, int(time.time()), *ids))
                rows = list({row["id"]: row for row in rows + [dict(row) for row in extra if self._valid(row)]}.values())
        rows.sort(key=lambda row: (row["source_id"] not in (preferred or set()), -row["source_id"], row["position"]))
        return rows

    @staticmethod
    def describe(row: dict) -> dict:
        result = {"图片编号": f"media_{row['id']}", "本地记录ID": row["source_id"],
                  "位置": row.get('position', 0),
                  "消息号": row["message_id"], "发图人QQ": row["user_id"],
                  "时间": datetime.fromtimestamp(row["created_at"], timezone(timedelta(hours=8))).isoformat(),
                  "缓存状态": row["status"], "概要状态": row.get("summary_state") or "pending"}
        if row.get("summary_state") == "ready":
            try:
                summary = json.loads(row["summary_json"])
                result.update({"概要": summary["summary"], "图中文字": summary["text"], "不确定": summary["uncertain"]})
            except (ValueError, KeyError, TypeError):
                result["概要状态"] = "error"
        return result

    @staticmethod
    def reference(row: dict) -> dict:
        return {"cache_id": str(row["id"]), "namespace": str(row["namespace"])}

    def retain(self, namespace: int, ref_id: int) -> bool:
        """被真实事件引用时延长保留；仍受全局文件/容量配额限制，不保证永不淘汰。"""
        row = self._row(namespace, ref_id)
        if not row or row["status"] not in {"pending", "ready"}:
            return False
        until = max(row["expires_at"], int(time.time()) + self.settings.image_memory_retention_days * 86400)
        with self.db._lock, self.db.connection:
            self.db.connection.execute("UPDATE image_cache_refs SET expires_at=?,retained=1 WHERE id=? AND namespace=?", (until, ref_id, namespace))
            self.db.connection.execute("UPDATE memory_records SET expires_at=? WHERE group_id=? AND memory_type='image' AND fact_key=?", (until, namespace, f"media_{ref_id}"))
        return True

    def _path(self, digest: str) -> Path:
        if self.directory.is_symlink():
            raise ValueError("图片缓存目录不可用。")
        if re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError("图片缓存标识无效。")
        return self.directory / (digest + ".img")

    def _read_blob(self, digest: str) -> bytes:
        path = self._path(digest)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as source:
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise ValueError("图片缓存文件不可用。")
            data = source.read(self.settings.chat_image_max_bytes + 1)
        if len(data) > self.settings.chat_image_max_bytes or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("图片缓存校验失败。")
        return data

    def _remove_blob(self, digest: str):
        self._path(digest).unlink(missing_ok=True)
        with self.db._lock, self.db.connection:
            self.db.connection.execute("UPDATE image_cache_refs SET hash='',status='evicted',image_json='{}' WHERE hash=?", (digest,))
            self.db.connection.execute("DELETE FROM image_cache_blobs WHERE hash=?", (digest,))

    def _store(self, digest: str, mime: str, data: bytes):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.is_symlink():
            raise ValueError("图片缓存目录不能是符号链接。")
        with self.db._lock:
            blobs = [dict(row) for row in self.db.connection.execute("SELECT * FROM image_cache_blobs ORDER BY created_at,hash")]
        existing = next((row for row in blobs if row["hash"] == digest), None)
        if existing:
            try:
                self._read_blob(digest)
                return
            except (OSError, ValueError):
                self._remove_blob(digest)
                blobs = [row for row in blobs if row["hash"] != digest]
        used = sum(row["size"] for row in blobs)
        while blobs and (len(blobs) >= self.settings.image_cache_max_files or used + len(data) > self.settings.image_cache_max_bytes):
            old = blobs.pop(0)
            self._remove_blob(old["hash"])
            used -= old["size"]
        if len(data) > self.settings.image_cache_max_bytes:
            raise ValueError("图片缓存容量不足。")
        descriptor, temporary = tempfile.mkstemp(prefix=".image-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
            os.replace(temporary, self._path(digest))
        finally:
            Path(temporary).unlink(missing_ok=True)
        with self.db._lock, self.db.connection:
            self.db.connection.execute("INSERT INTO image_cache_blobs(hash,mime,size,created_at) VALUES(?,?,?,?)",
                                       (digest, mime, len(data), int(time.time())))

    async def _finish_io(self, operation, *args):
        # 取消协程不能中断线程；等待写盘/数据库事务完成后才释放锁或关闭数据库。
        task = asyncio.create_task(asyncio.to_thread(operation, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await task
            except Exception:
                pass
            raise

    async def load(self, ref: dict) -> tuple[str, bytes]:
        """只能读取程序绑定的会话引用，不接受路径/URL/单独哈希授权。"""
        if not self.enabled:
            raise ValueError("图片缓存已关闭。")
        try:
            namespace, ref_id = int(ref["namespace"]), int(ref["cache_id"])
        except (ValueError, KeyError, TypeError):
            raise ValueError("图片缓存引用无效。") from None
        async with self._io_lock:
            row = self._row(namespace, ref_id)
            if not row:
                raise ValueError("图片已过期或来源消息已删除，请重新发送。")
            if row["status"] == "ready":
                try:
                    data = await asyncio.to_thread(self._read_blob, row["hash"])
                except (OSError, ValueError):
                    raise ValueError("图片缓存已清理或损坏，请重新发送。") from None
                if self._row(namespace, ref_id) is None:
                    raise ValueError("图片来源已删除。")
                from app.stickers import image_mime
                return image_mime(data), data
            if row["status"] != "pending" or row["next_attempt_at"] > time.time():
                raise ValueError("图片暂不可用、下载失败或缓存已清理，请重新发送。")
            try:
                mime, data = await self.downloader.load(json.loads(row["image_json"]))
                if self._row(namespace, ref_id) is None:
                    raise ValueError("图片来源已删除。")
                digest = hashlib.sha256(data).hexdigest()
                await self._finish_io(self._store, digest, mime, data)
                with self.db._lock, self.db.connection:
                    if self._row(namespace, ref_id) is None:
                        raise ValueError("图片来源已删除。")
                    self.db.connection.execute("UPDATE image_cache_refs SET hash=?,status='ready',image_json='{}' WHERE id=?",
                                               (digest, ref_id))
                logger.info("Image cache stored: bytes=%s", len(data))
                return mime, data
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts = row["attempts"] + 1
                with self.db._lock, self.db.connection:
                    self.db.connection.execute("UPDATE image_cache_refs SET attempts=?,next_attempt_at=?,status=? WHERE id=?",
                        (attempts, int(time.time()) + 60 * attempts, "pending" if attempts < 3 else "unavailable", ref_id))
                logger.warning("Image cache download deferred: error=%s", type(exc).__name__)
                raise ValueError("图片缓存下载失败，请稍后重试或重新发送。") from None

    async def summarize_once(self) -> bool:
        if not self.llm.settings.deepseek_api_key:
            return False
        with self.db._lock:
            blob = self.db.connection.execute(
                """SELECT b.* FROM image_cache_blobs b WHERE b.summary_state!='ready' AND b.attempts<3 AND b.next_attempt_at<=?
                   AND EXISTS(SELECT 1 FROM image_cache_refs r WHERE r.hash=b.hash AND r.expires_at>?) ORDER BY b.created_at LIMIT 1""",
                (int(time.time()), int(time.time()))).fetchone()
            blob = dict(blob) if blob else None
        if not blob:
            return False
        with self.db._lock:
            sources = self.db.connection.execute("SELECT * FROM image_cache_refs WHERE hash=?", (blob["hash"],)).fetchall()
            if not any(self._valid(row) for row in sources):
                return False
        try:
            async with self._io_lock:
                data = await asyncio.to_thread(self._read_blob, blob["hash"])
            summary = await self.llm.summarize_image(blob["mime"], base64.b64encode(data).decode("ascii"))
            with self.db._lock, self.db.connection:
                sources = self.db.connection.execute("SELECT * FROM image_cache_refs WHERE hash=?", (blob["hash"],)).fetchall()
                if any(self._valid(row) for row in sources):
                    self.db.connection.execute("UPDATE image_cache_blobs SET summary_json=?,summary_state='ready' WHERE hash=?",
                        (json.dumps(summary, ensure_ascii=False), blob["hash"]))
            logger.info("Image summary stored")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with self.db._lock, self.db.connection:
                self.db.connection.execute("UPDATE image_cache_blobs SET summary_state='error',attempts=attempts+1,next_attempt_at=? WHERE hash=?",
                                           (int(time.time()) + 300, blob["hash"]))
            logger.warning("Image summary deferred: error=%s", type(exc).__name__)
        return True

    async def cleanup(self):
        async with self._io_lock:
            with self.db._lock, self.db.connection:
                for row in self.db.connection.execute("SELECT * FROM image_cache_refs").fetchall():
                    if not self._valid(row):
                        self.db.connection.execute("DELETE FROM image_cache_refs WHERE id=?", (row["id"],))
                self.db.connection.execute("""DELETE FROM image_cache_refs WHERE id IN
                    (SELECT id FROM image_cache_refs ORDER BY id DESC LIMIT -1 OFFSET ?)""",
                    (self.settings.image_cache_max_files * 20,))
                unused = [row[0] for row in self.db.connection.execute(
                    "SELECT hash FROM image_cache_blobs WHERE NOT EXISTS(SELECT 1 FROM image_cache_refs r WHERE r.hash=image_cache_blobs.hash)")]
            for digest in unused:
                await self._finish_io(self._remove_blob, digest)
            with self.db._lock:
                blobs = [dict(row) for row in self.db.connection.execute("SELECT * FROM image_cache_blobs ORDER BY created_at,hash")]
            used = sum(row["size"] for row in blobs)
            while blobs and (len(blobs) > self.settings.image_cache_max_files or used > self.settings.image_cache_max_bytes):
                old = blobs.pop(0)
                await self._finish_io(self._remove_blob, old["hash"])
                used -= old["size"]
            # 崩溃留下的孤立文件不计入永久磁盘占用；只清理本模块的严格命名文件。
            if self.directory.is_dir() and not self.directory.is_symlink():
                with self.db._lock:
                    known = {row[0] for row in self.db.connection.execute("SELECT hash FROM image_cache_blobs")}
                for path in self.directory.iterdir():
                    if re.fullmatch(r"[a-f0-9]{64}\.img", path.name) and path.stem not in known:
                        path.unlink(missing_ok=True)
                    elif path.name.startswith(".image-") and path.is_file() and time.time() - path.stat().st_mtime > 3600:
                        path.unlink(missing_ok=True)
            self._last_cleanup = time.monotonic()

    async def _download_loop(self):
        while True:
            try:
                if time.monotonic() - self._last_cleanup > 60:
                    await self.cleanup()
                with self.db._lock:
                    row = self.db.connection.execute("SELECT * FROM image_cache_refs WHERE status='pending' AND next_attempt_at<=? ORDER BY id LIMIT 1",
                                                      (int(time.time()),)).fetchone()
                if row and self._row(row["namespace"], row["id"]):
                    await self.load(self.reference(dict(row)))
                else:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Image cache worker deferred: error=%s", type(exc).__name__)
                await asyncio.sleep(1)

    async def _summary_loop(self):
        while True:
            try:
                await self.summarize_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Image summary worker deferred: error=%s", type(exc).__name__)
            await asyncio.sleep(2)

    def start(self):
        if self.enabled and not self._tasks:
            self._tasks = [asyncio.create_task(self._download_loop(), name="image-cache-download"),
                           asyncio.create_task(self._summary_loop(), name="image-cache-summary")]
            logger.info("Image cache enabled: days=%s max_files=%s", self.settings.image_cache_retention_days, self.settings.image_cache_max_files)

    async def close(self):
        self._closed = True
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
