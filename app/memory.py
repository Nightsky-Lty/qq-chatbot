"""群成员长期记忆：本地稀疏向量、混合检索和受控写入。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from typing import Any, Iterable

from app.config import Settings
from app.database import Database


MEMORY_TYPES = {"profile", "preference", "event", "relationship", "group_meme", "explicit"}
MEMORY_TYPE_LABELS = {
    "profile": "个人资料",
    "preference": "偏好",
    "event": "经历",
    "relationship": "关系/称呼",
    "group_meme": "群梗",
    "explicit": "本人要求记住",
}

# 自动写入不接受高风险隐私。用户也不应通过手动命令把这些内容变成长久提示词。
_SENSITIVE_PATTERNS = (
    re.compile(r"(?:密码|口令|验证码|支付密码|银行卡|身份证|家庭住址|详细地址)\s*[:：]?", re.I),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
)


def contains_sensitive_memory(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS)


def _tokens(text: str) -> list[str]:
    """生成适合中英文短聊天的词元；中文使用单字和双字片段。"""
    lowered = text.casefold()
    tokens = re.findall(r"[a-z0-9_]+", lowered)
    chunks = re.findall(r"[\u3400-\u9fff]+", lowered)
    for chunk in chunks:
        tokens.extend(chunk)
        tokens.extend(chunk[index:index + 2] for index in range(len(chunk) - 1))
    return tokens


def sparse_vector(text: str, dimensions: int = 512) -> str:
    """无外部依赖的特征哈希向量，序列化后可持久化在 SQLite。"""
    counts: Counter[int] = Counter()
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        counts[int.from_bytes(digest, "big") % dimensions] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return json.dumps({str(key): value / norm for key, value in counts.items()}, separators=(",", ":"))


def vector_similarity(left: str, right: str) -> float:
    try:
        a = json.loads(left)
        b = json.loads(right)
    except (TypeError, ValueError):
        return 0.0
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(float(value) * float(b.get(key, 0.0)) for key, value in a.items())


class MemoryService:
    """保存经过约束的长期记忆，并为群聊构建可审计的 RAG 结果。"""

    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.db = database

    def remember(
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
        return self.db.save_memory(
            group_id, user_id, source_user_id if source_user_id is not None else user_id,
            memory_type, content, sparse_vector(content),
            confidence=max(0.0, min(1.0, confidence)),
            importance=max(0.0, min(1.0, importance)),
            source_message_id=source_message_id,
            max_per_user=self.settings.memory_max_per_user,
        )

    def retrieve(
        self, group_id: int, query: str, *, user_ids: Iterable[int], limit: int | None = None
    ) -> list[dict[str, Any]]:
        """先按群和参与者过滤，再按向量、重要度、新鲜度进行混合排序。"""
        if not self.settings.memory_enabled:
            return []
        targets = {int(user_id) for user_id in user_ids if int(user_id) > 0}
        rows = self.db.memory_candidates(
            group_id, targets, max(self.settings.memory_candidate_limit, self.settings.memory_retrieval_limit)
        )
        query_vector = sparse_vector(query)
        query_terms = {token for token in _tokens(query) if len(token) > 1}
        now = int(time.time())
        ranked: list[tuple[float, Any]] = []
        normalized_query = "".join(query.casefold().split())
        for row in rows:
            similarity = vector_similarity(query_vector, row["vector"])
            content_terms = {token for token in _tokens(row["content"]) if len(token) > 1}
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
            # 事件和群梗只有与本轮存在字面/片段联系才召回，避免“认识这个人”退化成
            # 每轮都把他全部往事塞给模型；稳定档案则始终可作为辨人背景。
            if not stable_for_person and not lexical_overlap and not exact:
                continue
            score = similarity * 0.62 + row["importance"] * 0.13 + row["confidence"] * 0.08
            score += freshness * 0.05 + same_person + exact
            # 稳定档案即使和代词查询没有字面重合，也应进入当前成员上下文。
            if stable_for_person:
                score += 0.12
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
