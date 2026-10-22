"""可重建的Qdrant索引；只有编号/范围/版本进payload，不复制原文和来源。"""

import hashlib
import json
import re
import time

import httpx

from app.config import Settings


class MemoryIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=settings.memory_qdrant_timeout_seconds, trust_env=False,
                                       headers={"api-key": settings.memory_qdrant_api_key} if settings.memory_qdrant_api_key else {})
        self.ready: set[str] = set()
        self.retry_after = 0.0

    @property
    def enabled(self):
        return bool(self.settings.memory_qdrant_url)

    def collection(self, model: str, dimensions: int) -> str:
        # 不同模型即使维度相同也不能混用语义空间。
        suffix = hashlib.sha256(model.encode()).hexdigest()[:12]
        return f"{self.settings.memory_qdrant_collection}_{dimensions}_{suffix}"

    async def request(self, method, path, **kwargs):
        response = await self.client.request(method, self.settings.memory_qdrant_url.rstrip("/") + path, **kwargs)
        response.raise_for_status()
        return response

    async def ensure(self, model: str, dimensions: int):
        collection = self.collection(model, dimensions)
        if collection not in self.ready:
            response = await self.client.get(self.settings.memory_qdrant_url.rstrip("/") + f"/collections/{collection}")
            if response.status_code == 404:
                await self.request("PUT", f"/collections/{collection}", json={"vectors": {"size": dimensions, "distance": "Cosine"},
                    "optimizers_config": {"default_segment_number": 1}})
                for field in ("namespace", "version"):
                    await self.request("PUT", f"/collections/{collection}/index?wait=true",
                                       json={"field_name": field, "field_schema": "integer"})
            else:
                response.raise_for_status()
            self.ready.add(collection)
        return collection

    async def sync(self, memory_id: int, row: dict | None, model: str, dimensions: int):
        collection = await self.ensure(model, dimensions)
        if not row or row["status"] != "active" or (row["expires_at"] and row["expires_at"] <= time.time()):
            # 模型升级后的旧语义空间也要清理；不让已遗忘向量遗留在旧collection。
            response = await self.request("GET", "/collections")
            pattern = re.compile(re.escape(self.settings.memory_qdrant_collection) + r"_\d+_[a-f0-9]{12}")
            names = {item["name"] for item in response.json()["result"]["collections"] if pattern.fullmatch(item["name"])}
            for name in names | {collection}:
                await self.request("POST", f"/collections/{name}/points/delete?wait=true", json={"points": [memory_id]})
            return
        vector = json.loads(row["vector"])
        if not isinstance(vector, list) or row["embedding_model"] != model or len(vector) != dimensions:
            return
        await self.request("PUT", f"/collections/{collection}/points?wait=true", json={"points": [{
            "id": memory_id, "vector": vector,
            "payload": {"namespace": row["group_id"], "version": row["version"], "layer": row["layer"],
                        "subject": row["user_id"], "occurred_at": row["occurred_at"],
                        "participants": json.loads(row["metadata_json"]).get("participants", [row["user_id"]])},
        }]})

    async def search(self, namespace: int, vector: str, model: str, dimensions: int, limit: int, *, layer=None, subject=None, after=None, before=None):
        collection = await self.ensure(model, dimensions)
        must = [{"key": "namespace", "match": {"value": namespace}}]
        if layer:
            must.append({"key": "layer", "match": {"value": layer}})
        if subject is not None:
            must.append({"should": [{"key": "subject", "match": {"value": subject}},
                                    {"key": "participants", "match": {"value": subject}}]})
        if after is not None or before is not None:
            must.append({"key": "occurred_at", "range": {**({"gte": after} if after is not None else {}), **({"lte": before} if before is not None else {})}})
        response = await self.request("POST", f"/collections/{collection}/points/query", json={
            "query": json.loads(vector), "filter": {"must": must}, "limit": limit,
            "with_payload": True, "with_vector": False,
        })
        return response.json()["result"]["points"]

    async def close(self):
        await self.client.aclose()
