"""分层长期记忆服务：向量生成、混合检索与索引同步。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Iterable

from app.config import Settings
from app.database import Database
from app.memory_index import MemoryIndex
from app.memory_retrieval import HybridRetriever
from app.embeddings import (
    HASH_MODEL_ID,
    EmbeddingProvider,
    HashEmbeddingProvider,
    create_embedding_provider,
    sparse_vector,
)


logger = logging.getLogger(__name__)


# 自动写入不接受高风险隐私。用户也不应通过手动命令把这些内容变成长久提示词。
_SENSITIVE_PATTERNS = (
    re.compile(r"(?:密码|口令|验证码|支付密码|银行卡|身份证|家庭住址|详细地址)\s*[:：]?", re.I),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
)


def contains_sensitive_memory(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS)


class MemoryService:
    """保存经过约束的长期记忆，并为群聊构建可审计的 RAG 结果。"""

    def __init__(
        self, settings: Settings, database: Database, provider: EmbeddingProvider | None = None
    ):
        self.settings = settings
        self.db = database
        self.provider = provider or create_embedding_provider(settings)
        self.hash_provider = HashEmbeddingProvider()
        self._backfill_task: asyncio.Task[None] | None = None
        self.index = MemoryIndex(settings)
        self.retriever = HybridRetriever(self)
        self.image_cache = None
        self._index_task = None

    async def start(self) -> None:
        """预热向量模型；失败时保持服务可用，并自动退回特征哈希。"""
        if not self.settings.memory_enabled:
            return
        ready = await self.provider.start()
        if self._index_task is None:
            self._index_task = asyncio.create_task(self._index_loop(), name="memory-index-sync")
        if ready and self.provider.model_id != HASH_MODEL_ID:
            self._backfill_task = asyncio.create_task(
                self._backfill_embeddings(), name="memory-embedding-backfill"
            )

    async def close(self) -> None:
        if self._backfill_task is not None and not self._backfill_task.done():
            self._backfill_task.cancel()
            await asyncio.gather(self._backfill_task, return_exceptions=True)
        self._backfill_task = None
        if self._index_task:
            self._index_task.cancel()
            await asyncio.gather(self._index_task, return_exceptions=True)
        await self.index.close()

    async def _index_loop(self):
        while True:
            try:
                with self.db._lock, self.db.connection:
                    self.db.connection.execute("UPDATE memory_records SET status='archived',version=version+1 WHERE status='active' AND expires_at>0 AND expires_at<=?", (int(time.time()),))
                await self.sync_images()
                await self.sync_index_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Memory index worker deferred: error=%s", type(exc).__name__)
            await asyncio.sleep(2)

    async def sync_index_once(self):
        if not self.index.enabled or self.provider.model_id == HASH_MODEL_ID or not self.provider.available:
            return 0
        with self.db._lock:
            jobs = self.db.connection.execute("SELECT * FROM memory_index_jobs WHERE due_at<=? ORDER BY due_at,memory_id LIMIT 16", (int(time.time()),)).fetchall()
        completed = 0
        for job in jobs:
            with self.db._lock:
                found = self.db.connection.execute("SELECT * FROM memory_records WHERE id=?", (job["memory_id"],)).fetchone()
                row = dict(found) if found else None
            try:
                await self.index.sync(job["memory_id"], row, self.provider.model_id, self.provider.dimensions)
                with self.db._lock, self.db.connection:
                    self.db.connection.execute("DELETE FROM memory_index_jobs WHERE memory_id=? AND revision=?", (job["memory_id"], job["revision"]))
                completed += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                with self.db._lock, self.db.connection:
                    self.db.connection.execute("UPDATE memory_index_jobs SET attempts=attempts+1,due_at=? WHERE memory_id=? AND revision=?",
                        (int(time.time()) + min(3600, 10 * 2 ** min(job["attempts"], 8)), job["memory_id"], job["revision"]))
                logger.warning("Memory index sync deferred: error=%s", type(exc).__name__)
                self.index.retry_after = time.monotonic() + 30
                break
        return completed

    async def sync_images(self):
        cache = self.image_cache
        if cache is None or not cache.enabled:
            return
        with self.db._lock:
            rows = self.db.connection.execute("""SELECT r.*,b.summary_json FROM image_cache_refs r JOIN image_cache_blobs b ON b.hash=r.hash
              WHERE r.status='ready' AND b.summary_state='ready' AND r.expires_at>? AND NOT EXISTS
              (SELECT 1 FROM memory_records m WHERE m.group_id=r.namespace AND m.memory_type='image' AND m.fact_key='media_'||r.id)
              ORDER BY r.id LIMIT 8""", (int(time.time()),)).fetchall()
        for row in rows:
            ref = cache.reference(dict(row))
            if not cache.authorized(ref):
                continue
            summary = json.loads(row["summary_json"])
            content = f"图片media_{row['id']}概要（非原话）：{summary['summary']}\n图中文字：{summary['text']}"[:self.settings.memory_max_chars]
            if contains_sensitive_memory(content) or self.db.memories.blocked(row["namespace"], row["user_id"], f"media_{row['id']}", content, [row["source_id"]]):
                continue
            vector, model, dimensions = await self._document_embedding(content)
            sources = self.db.memory_source_rows(row["namespace"], {row["source_id"]})
            if not sources or not cache.authorized(ref):
                continue
            action = {"action": "upsert", "subject_user_id": row["user_id"], "memory_type": "image", "content": content,
                      "fact_key": f"media_{row['id']}", "source_ids": [row["source_id"]], "replaces_ids": [],
                      "sources": [{"id": row["source_id"], "user_id": row["user_id"], "role": "user", "time": row["created_at"], "content": "[图片附件]"}],
                      "vector": vector, "hash_vector": sparse_vector(content), "embedding_model": model, "embedding_dimensions": dimensions,
                      "importance": 0.3, "confidence": 0.8, "expires_at": row["expires_at"], "occurred_at": row["created_at"],
                      "evidence_kind": "image_summary", "metadata": {"media_ref_id": row["id"], "uncertain": summary["uncertain"]}}
            self.db.apply_memory_capture(row["namespace"], None, 0, [action])

    async def _document_embedding(self, content: str) -> tuple[str, str, int]:
        """生成主向量；模型不可用时本次写入使用可检索的哈希向量。"""
        if self.provider.available:
            try:
                vector = (await self.provider.embed_documents([content]))[0]
                return vector, self.provider.model_id, self.provider.dimensions
            except Exception as exc:
                logger.warning("Embedding inference failed; using hash fallback: error=%s", type(exc).__name__)
        vector = await self.hash_provider.embed_query(content)
        return vector, self.hash_provider.model_id, self.hash_provider.dimensions

    async def _backfill_embeddings(self) -> None:
        """在后台分批把旧记忆升级为当前模型，避免阻塞服务启动。"""
        updated = 0
        try:
            while self.provider.available:
                rows = self.db.memories_needing_embedding(
                    self.provider.model_id, self.settings.embedding_backfill_batch_size
                )
                if not rows:
                    break
                vectors = await self.provider.embed_documents([row["content"] for row in rows])
                if len(vectors) != len(rows):
                    raise RuntimeError("Embedding后端返回数量与输入不一致")
                for row, vector in zip(rows, vectors):
                    self.db.update_memory_embedding(
                        row["id"], vector, self.provider.model_id, self.provider.dimensions, row["content"]
                    )
                    updated += 1
                # 给在线查询留出调度机会，避免大量旧记忆回填时持续占满 CPU。
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Memory embedding backfill stopped: error=%s updated=%s", type(exc).__name__, updated)
        else:
            if updated:
                logger.info("Memory embedding backfill complete: updated=%s", updated)

    async def retrieve(
        self, group_id: int, query: str, *, user_ids: Iterable[int] = (), limit: int | None = None, **filters
    ) -> list[dict[str, Any]]:
        """统一入口：身份/时间条件、FTS、Qdrant、本地降级与来源展开。"""
        return await self.retriever.search(group_id, query, user_ids=user_ids, limit=limit, **filters)
