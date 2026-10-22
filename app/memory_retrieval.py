"""四层记忆的混合检索：精确/FTS/向量，RRF融合，权威状态回查及有界证据展开。"""

import asyncio
import heapq
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone

from app.embeddings import HASH_MODEL_ID, sparse_vector, vector_similarity
from app.memory_store import match_query

logger = logging.getLogger(__name__)
BEIJING = timezone(timedelta(hours=8))


def time_range(query: str, reference_time: int | None = None):
    day = None
    now = datetime.fromtimestamp(reference_time, BEIJING) if reference_time else datetime.now(BEIJING)
    for word, delta in (("前天", 2), ("昨天", 1), ("今天", 0)):
        if word in query:
            day = now - timedelta(days=delta)
            break
    match = re.search(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", query)
    if match:
        try:
            day = datetime.strptime(match[1], "%Y-%m-%d").replace(tzinfo=BEIJING)
        except ValueError:
            pass
    if day:
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(start.timestamp()), int((start + timedelta(days=1)).timestamp()) - 1
    return None, None


class HybridRetriever:
    def __init__(self, memory):
        self.memory, self.db, self.settings = memory, memory.db, memory.settings

    def filters(self, namespace, layer=None, subject=None, after=None, before=None):
        where = ["m.group_id=?", "m.status='active'", "(m.expires_at=0 OR m.expires_at>?)"]
        args = [namespace, int(time.time())]
        if layer:
            where.append("m.layer=?")
            args.append(layer)
        if subject is not None:
            where.append("(m.user_id=? OR (m.layer='event' AND EXISTS(SELECT 1 FROM json_each(m.metadata_json,'$.participants') WHERE value=?)))")
            args.extend((subject, subject))
        if after is not None:
            where.append("m.occurred_at>=?")
            args.append(after)
        if before is not None:
            where.append("m.occurred_at<=?")
            args.append(before)
        return " AND ".join(where), args

    def eligible(self, row, boundary=None):
        metadata = json.loads(row["metadata_json"])
        sources = json.loads(row["sources_json"])
        if boundary is not None and any(s.get("id", 0) > boundary for s in sources):
            return False
        if any(s.get("id") in self.db.memories.blocked_sources(row["group_id"]) for s in sources):
            return False
        if row["layer"] == "perceptual":
            cache = self.memory.image_cache
            ref_id = metadata.get("media_ref_id")
            if cache is None or not cache.authorized({"namespace": str(row["group_id"]), "cache_id": str(ref_id)}):
                return False
        return True

    def local_vectors(self, where, args, query_vector, dense_model, limit):
        # 全范围流式打分，不先截取“最近300条”；堆只保留top-k，不加载整个库到内存。
        heap, cursor = [], 0
        with self.db._lock:
            rows = self.db.connection.execute(f"SELECT * FROM memories m WHERE {where} AND m.id>? ORDER BY m.id LIMIT 256", (*args, cursor)).fetchall()
        while rows:
            for row in rows:
                dense = dense_model and row["embedding_model"] == dense_model
                if dense_model and not dense:
                    continue
                score = vector_similarity(query_vector, row["vector"] if dense else row["hash_vector"] or row["vector"])
                item = (score, row["id"])
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            cursor = rows[-1]["id"]
            with self.db._lock:
                rows = self.db.connection.execute(f"SELECT * FROM memories m WHERE {where} AND m.id>? ORDER BY m.id LIMIT 256", (*args, cursor)).fetchall()
        return sorted(heap, reverse=True)

    async def search(self, namespace, query, *, user_ids=(), limit=None, layer=None, subject=None,
                     after=None, before=None, boundary=None, include_sources=False):
        if not self.settings.memory_enabled or not query.strip():
            return []
        if layer not in {None, "event", "abstract", "perceptual"}:
            raise ValueError("记忆类型无效")
        if layer is None and re.search(r"什么意思|什么梗|指什么|怎么理解|词义", query):
            layer = "abstract"
        if namespace < 0 and subject is not None and subject != -namespace:
            return []
        if after is None and before is None and layer != "abstract":
            after, before = time_range(query)
        if after is not None and before is not None and after > before:
            raise ValueError("时间范围无效")
        where, args = self.filters(namespace, layer, subject, after, before)
        route_limit = self.settings.memory_search_route_limit
        fts_query = match_query(query)
        with self.db._lock:
            lexical = list(self.db.connection.execute(
                f"SELECT m.id,bm25(memory_fts) AS score FROM memory_fts JOIN memories m ON m.id=memory_fts.rowid WHERE memory_fts MATCH ? AND {where} ORDER BY score LIMIT ?",
                (fts_query, *args, route_limit))) if fts_query else []
            concepts = list(self.db.connection.execute(f"SELECT m.* FROM memories m WHERE {where} AND m.layer='abstract' AND m.fact_key!=''", args))
        exact = [r["id"] for r in concepts if r["fact_key"].casefold() in query.casefold() and self.eligible(r, boundary)][:route_limit]
        expansion = " ".join(json.loads(r["metadata_json"]).get("meaning", "")[:100] for r in concepts if r["id"] in exact[:2])
        effective_query = query[:600] + (" " + expansion if expansion else "")
        dense = None
        provider, index = self.memory.provider, self.memory.index
        if provider.available and provider.model_id != HASH_MODEL_ID:
            try:
                dense = await provider.embed_query(effective_query)
            except Exception as exc:
                logger.warning("Memory query embedding fallback: error=%s", type(exc).__name__)
        semantic, scores = [], {}
        if dense is not None and index.enabled and time.monotonic() >= index.retry_after:
            try:
                async with asyncio.timeout(self.settings.memory_qdrant_timeout_seconds):
                    points = await index.search(namespace, dense, provider.model_id, provider.dimensions, route_limit,
                                                layer=layer, subject=subject, after=after, before=before)
                for p in points:
                    row = self.db.memories.row(namespace, int(p["id"]))
                    payload = p.get("payload") or {}
                    # 索引不同步、旧版本、跨空间或失效时绝不信向量结果。
                    if row and payload.get("namespace") == namespace and payload.get("version") == row["version"]:
                        semantic.append(row["id"])
                        scores[row["id"]] = float(p["score"])
            except Exception as exc:
                index.retry_after = time.monotonic() + 30
                logger.warning("Memory vector index fallback: error=%s", type(exc).__name__)
        # 也覆盖尚未同步的新记录。Qdrant正常时只补查待同步记录，故障时扫描全部范围。
        fallback_where = where
        if semantic:
            fallback_where += " AND EXISTS(SELECT 1 FROM memory_index_jobs j WHERE j.memory_id=m.id)"
        fallback_vector = dense or sparse_vector(effective_query)
        local = await asyncio.to_thread(self.local_vectors, fallback_where, args, fallback_vector,
                                        provider.model_id if dense else None, route_limit)
        for score, item_id in local:
            scores[item_id] = score
        semantic = sorted(set(semantic) | {i for _, i in local}, key=lambda i: scores[i], reverse=True)[:route_limit]
        routes = [exact, [r["id"] for r in lexical], semantic]
        ranks = {}
        for route in routes:
            for rank, item_id in enumerate(route):
                ranks[item_id] = ranks.get(item_id, 0) + 1 / (60 + rank + 1)
        selected = []
        related = set(user_ids)
        for item_id, score in ranks.items():
            row = self.db.memories.row(namespace, item_id)
            if not row or not self.eligible(row, boundary):
                continue
            participants = json.loads(row["metadata_json"]).get("participants", [])
            if (layer and row["layer"] != layer) or (subject is not None and row["user_id"] != subject
                    and not (row["layer"] == "event" and subject in participants)):
                continue
            if (after is not None and row["occurred_at"] < after) or (before is not None and row["occurred_at"] > before):
                continue
            lexical_hit = item_id in routes[1] or item_id in exact
            if not lexical_hit and scores.get(item_id, 0) < (self.settings.embedding_min_similarity if dense else 0.08):
                continue
            if row["user_id"] in related:
                score += 0.001
            # 不按模型自报置信度排成事实；也不让新鲜度盖过问题相关性。
            selected.append((score, row))
        selected.sort(key=lambda x: (x[0], x[1]["id"]), reverse=True)
        result, spent = [], 0
        max_items = limit or self.settings.memory_retrieval_limit
        for _, row in selected[:max_items]:
            public = self.public(row, include_sources=include_sources)
            with self.db._lock:
                links = self.db.connection.execute("SELECT * FROM memory_links WHERE namespace=? AND (source_id=? OR target_id=?) LIMIT 8", (namespace, row["id"], row["id"])).fetchall()
            related_memories = []
            for link in links:
                other_id = link["target_id"] if link["source_id"] == row["id"] else link["source_id"]
                other = self.db.memories.row(namespace, other_id)
                if other and self.eligible(other, boundary):
                    related_memories.append({"记忆编号": other_id, "层": other["layer"], "内容": other["content"][:100]})
                if len(related_memories) >= 2:
                    break
            if related_memories:
                public["关联记忆"] = related_memories
            size = len(json.dumps(public, ensure_ascii=False))
            if spent + size <= self.settings.memory_context_max_chars:
                result.append(public)
                spent += size
        # 只在明确回忆且结果不足时补查原话；不能从原话绕过遗忘/删除。
        if len(result) < min(3, max_items) and self.settings.memory_source_search_enabled and re.search(r"之前|以前|上次|说过|记得|回忆|昨天|前天", query):
            result.extend(self.raw_sources(namespace, fts_query, subject, after, before, boundary,
                                           max_items - len(result), self.settings.memory_context_max_chars - spent))
        self.db.touch_memories([r["记忆编号"] for r in result if type(r.get("记忆编号")) is int])
        logger.info("Memory retrieved: exact=%s lexical=%s semantic=%s selected=%s", len(exact), len(lexical), len(semantic), len(result))
        return result

    def public(self, row, *, include_sources=False):
        metadata = json.loads(row["metadata_json"])
        value = {"记忆编号": row["id"], "层": row["layer"], "成员QQ": row["user_id"] or None,
                 "内容": metadata.get("original_summary", row["content"]), "版本": row["version"], "发生时间": row["occurred_at"] or None,
                 "记录时间": row["created_at"], "依据": row["evidence_kind"], "有效期": row["expires_at"] or None}
        for field in ("conditions", "uncertain", "media_ref_id", "aliases", "participants"):
            if metadata.get(field):
                value[field] = metadata[field]
        if type(metadata.get("media_ref_id")) is int:
            value["图片编号"] = f"media_{metadata['media_ref_id']}"
        if metadata.get("media_refs"):
            value["关联图片"] = [f"media_{i}" for i in metadata["media_refs"][:8]]
        if include_sources:
            value["来源摘录"] = [{"本地记录ID": s.get("id"), "作者QQ": s.get("user_id"), "角色": s.get("role"),
                                  "内容": s.get("content", "")[:240], "时间": s.get("time")} for s in json.loads(row["sources_json"])[:3]]
        else:
            value["来源记录"] = [s.get("id") for s in json.loads(row["sources_json"])][:8]
        return {k: v for k, v in value.items() if v is not None and v != []}

    def raw_sources(self, namespace, query, subject, after, before, boundary, limit, budget):
        if not query or limit <= 0 or budget <= 0:
            return []
        with self.db._lock:
            hits = self.db.connection.execute("SELECT source_id FROM memory_source_fts WHERE memory_source_fts MATCH ? AND namespace=? ORDER BY bm25(memory_source_fts) LIMIT 60",
                                              (query, namespace)).fetchall()
        blocked = self.db.memories.blocked_sources(namespace)
        rows = self.db.memory_source_rows(namespace, {r[0] for r in hits} - blocked)
        result = []
        from app.memory import contains_sensitive_memory
        for row in rows:
            author = row["user_id"] if namespace > 0 else -namespace
            if (row["role"] not in {"user", "assistant"} or contains_sensitive_memory(row["content"])
                or row["content"].lstrip().startswith(("/", "[戳一戳", "[表情包"))
                or (subject is not None and author != subject) or (boundary is not None and row["id"] > boundary)
                or (after is not None and row["created_at"] < after) or (before is not None and row["created_at"] > before)):
                continue
            value = {"层": "source", "本地记录ID": row["id"], "成员QQ": author, "角色": row["role"],
                     "内容": row["content"][:300], "发生时间": row["created_at"], "依据": "原始聊天发言，不代表已核实的现实事实"}
            cost = len(json.dumps(value, ensure_ascii=False))
            if cost <= budget:
                result.append(value)
                budget -= cost
            if len(result) >= limit:
                break
        return result
