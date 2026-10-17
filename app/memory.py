"""群成员长期记忆：BGE/哈希向量、混合检索和受控写入。"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Iterable

from app.config import Settings
from app.database import Database
from app.embeddings import (
    HASH_MODEL_ID,
    EmbeddingProvider,
    HashEmbeddingProvider,
    create_embedding_provider,
    sparse_vector,
    tokens,
    vector_similarity,
)


logger = logging.getLogger(__name__)


MEMORY_TYPES = {"profile", "preference", "event", "relationship", "group_meme", "explicit", "episode", "commitment"}
MEMORY_TYPE_LABELS = {
    "profile": "个人资料",
    "preference": "偏好",
    "event": "经历",
    "relationship": "关系/称呼",
    "group_meme": "群梗",
    "explicit": "本人要求记住",
    "episode": "共同经历",
    "commitment": "双方约定（不代表已安排工具执行）",
}

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

    async def start(self) -> None:
        """预热向量模型；失败时保持服务可用，并自动退回特征哈希。"""
        if not self.settings.memory_enabled:
            return
        ready = await self.provider.start()
        if ready and self.provider.model_id != HASH_MODEL_ID:
            self._backfill_task = asyncio.create_task(
                self._backfill_embeddings(), name="memory-embedding-backfill"
            )

    async def close(self) -> None:
        if self._backfill_task is not None and not self._backfill_task.done():
            self._backfill_task.cancel()
            await asyncio.gather(self._backfill_task, return_exceptions=True)
        self._backfill_task = None

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
                        row["id"], vector, self.provider.model_id, self.provider.dimensions
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

    async def remember(
        self,
        group_id: int,
        user_id: int,
        memory_type: str,
        content: str,
        *,
        source_user_id: int | None = None,
        confidence: float,
        importance: float,
        source_message_id: str,
        explicit: bool = False,
    ) -> tuple[int, bool]:
        content = " ".join(content.split()).strip()
        if memory_type not in MEMORY_TYPES:
            raise ValueError("不支持的记忆类型。")
        if not 2 <= len(content) <= self.settings.memory_max_chars:
            raise ValueError(f"记忆内容需为 2～{self.settings.memory_max_chars} 字。")
        if contains_sensitive_memory(content):
            raise ValueError("这段内容可能包含敏感个人信息，不会写入长期记忆。")
        if not explicit and confidence < self.settings.memory_min_confidence:
            raise ValueError("这条信息不够确定，不写入长期记忆。")
        hash_vector = sparse_vector(content)
        vector, embedding_model, dimensions = await self._document_embedding(content)
        return self.db.save_memory(
            group_id, user_id, source_user_id if source_user_id is not None else user_id,
            memory_type, content, vector, hash_vector, embedding_model, dimensions,
            confidence=max(0.0, min(1.0, confidence)),
            importance=max(0.0, min(1.0, importance)),
            source_message_id=source_message_id,
            max_per_user=self.settings.memory_max_per_user,
        )

    async def retrieve(
        self, group_id: int, query: str, *, user_ids: Iterable[int], limit: int | None = None
    ) -> list[dict[str, Any]]:
        """先按群和参与者过滤，再按向量、重要度、新鲜度进行混合排序。"""
        if not self.settings.memory_enabled:
            return []
        targets = {int(user_id) for user_id in user_ids if int(user_id) > 0}
        rows = self.db.memory_candidates(
            group_id, targets, max(self.settings.memory_candidate_limit, self.settings.memory_retrieval_limit)
        )
        hash_query_vector = sparse_vector(query)
        dense_query_vector: str | None = None
        if query.strip() and self.provider.model_id != HASH_MODEL_ID and self.provider.available:
            try:
                dense_query_vector = await self.provider.embed_query(query)
            except Exception as exc:
                logger.warning("Query embedding failed; using hash fallback: error=%s", type(exc).__name__)
        query_terms = {token for token in tokens(query) if len(token) > 1}
        now = int(time.time())
        ranked: list[tuple[float, Any]] = []
        normalized_query = "".join(query.casefold().split())
        recall_interaction = bool(re.search(r"上次|之前|以前|约定|答应|说好|记得|回忆", query))
        for row in rows:
            use_dense = dense_query_vector is not None and row["embedding_model"] == self.provider.model_id
            similarity = vector_similarity(
                dense_query_vector if use_dense else hash_query_vector,
                row["vector"] if use_dense else (row["hash_vector"] or row["vector"]),
            )
            content_terms = {token for token in tokens(row["content"]) if len(token) > 1}
            lexical_overlap = bool(query_terms & content_terms)
            content_key = "".join(row["content"].casefold().split())
            exact = 0.15 if content_key and normalized_query and (
                content_key in normalized_query or normalized_query in content_key
            ) else 0.0
            same_person = 0.12 if row["user_id"] in targets else 0.0
            age_days = max(0.0, (now - row["updated_at"]) / 86400)
            freshness = 1.0 / (1.0 + age_days / 90.0)
            stable_for_person = (
                row["memory_type"] in {"profile", "preference", "relationship", "explicit"}
                and row["user_id"] in targets
            )
            joint_recall = (recall_interaction and row["memory_type"] in {"episode", "commitment"}
                            and row["user_id"] in targets)
            # 稳定档案始终可作为辨人背景；事件和群梗还需字面或语义相关，避免把
            # 某人的全部往事塞进每一轮上下文。
            semantically_related = use_dense and similarity >= self.settings.embedding_min_similarity
            if not stable_for_person and not joint_recall and not lexical_overlap and not exact and not semantically_related:
                continue
            score = similarity * 0.62 + row["importance"] * 0.13 + row["confidence"] * 0.08
            score += freshness * 0.05 + same_person + exact
            # 稳定档案即使和代词查询没有字面重合，也应进入当前成员上下文。
            if stable_for_person:
                score += 0.12
            if joint_recall:
                score += 0.18
            ranked.append((score, row))
        selected = sorted(ranked, key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        selected = selected[: limit or self.settings.memory_retrieval_limit]
        if selected:
            self.db.touch_memories([row["id"] for _, row in selected])
        return [
            {
                "记忆编号": row["id"],
                "成员QQ": row["user_id"] or None,
                "类型": MEMORY_TYPE_LABELS.get(row["memory_type"], row["memory_type"]),
                "内容": row["content"],
                "置信度": round(row["confidence"], 2),
                "来源消息号": row["source_message_id"],
                "记录时间": row["updated_at"],
                "有效期": row["expires_at"] or None,
                "来源摘录": [{"角色": source.get("role"), "内容": source.get("content", "")[:300],
                              "时间": source.get("time")} for source in json.loads(row["sources_json"])],
            }
            for _, row in selected
        ]


REMEMBER_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "remember_group_memory",
        "description": "把当前批次某位群友明确说出的、未来仍有用的稳定信息写入长期记忆。不保存猜测、一次性情绪、敏感信息或普通寒暄。静默执行，不需要向群里播报。",
        "parameters": {
            "type": "object",
            "properties": {
                "source_message_id": {"type": "string", "description": "支持该事实的本批新消息号"},
                "subject_user_id": {"type": "integer", "description": "事实所属群友的QQ号；群梗填0"},
                "memory_type": {
                    "type": "string",
                    "enum": ["profile", "preference", "event", "relationship", "group_meme"],
                },
                "content": {"type": "string", "minLength": 2, "maxLength": 200},
                "confidence": {"type": "number", "minimum": 0.75, "maximum": 1},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["source_message_id", "subject_user_id", "memory_type", "content", "confidence", "importance"],
            "additionalProperties": False,
        },
    },
}


MEMORY_INSTRUCTIONS = """

【长期成员记忆】
程序会提供按当前群和相关成员过滤后的长期记忆。它只是背景：仅在确实相关时自然使用，不复述记忆编号、置信度或数据库细节。
QQ号是稳定身份；昵称相同不代表同一个人，昵称变化也不代表换人。绝不把甲的资料、经历或情绪套给乙。
remember_group_memory只保存本批新消息中本人明确说出的、未来对聊天仍有帮助的信息，例如稳定资料、偏好、重要经历、固定称呼或反复使用的群梗。
普通寒暄、一次性情绪、玩笑中的虚构、推测出的性格、密码/证件/电话/住址等敏感信息一律不保存。不为显得聪明而强行记忆。
事实属于某人时subject_user_id必须等于来源消息的发言人QQ；只有不属于个人的群梗才使用0和group_meme。工具成功或发现重复都保持安静，不发送“已记住”。
"""
