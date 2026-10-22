"""临时SQLite/独立临时Qdrant collection验证；--live仅额外上传合成对话给记忆LLM。"""

import argparse
import asyncio
import json
import re
import tempfile
import uuid
from pathlib import Path

from app.config import Settings
from app.database import Database
from app.embeddings import HASH_MODEL_ID
from app.llm import DeepSeekClient
from app.memory import MemoryService
from app.memory_capture import MemoryCapture


class SyntheticExtractor:
    async def extract_memories(self, instructions, data):
        row = next(r for r in data["messages"] if r["role"] == "user")
        return {"actions": [{"action": "upsert", "subject_user_id": row["user_id"], "memory_type": "event",
            "content": "QQ110001昨天排查测试网站502错误，启动上游服务后恢复", "fact_key": "合成网站502故障事件",
            "source_ids": [row["id"]], "replaces_ids": [], "confidence": 0.95, "importance": 0.7, "expires_at": 0}]}


async def main(live):
    prefix = "qq_verify_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="qq-memory-check-") as folder:
        cfg = Settings().model_copy(update={"database_path": Path(folder)/"test.db", "image_cache_directory": Path(folder)/"images",
                                           "memory_qdrant_collection": prefix, "memory_source_search_enabled": False})
        db = Database(cfg.database_path)
        real = DeepSeekClient(cfg, purpose="memory")
        memory = MemoryService(cfg, db)
        capture = MemoryCapture(cfg, db, memory, real if live else SyntheticExtractor())
        collection = None
        try:
            assert await memory.provider.start(), "embedding unavailable"
            assert memory.provider.model_id != HASH_MODEL_ID, "real dense embedding required"
            assert memory.index.enabled, "Qdrant URL required"
            source = db.log_private_message(110001, "user", "这是一段合成测试：昨天我排查测试网站502错误，启动上游服务后已经恢复。这次具体经历以后排障还会用到，请记住。")
            workspace = capture.open_workspace(-110001, {source})
            assert not hasattr(workspace, "messages"), "任务证据不能承载内部推理/工具链"
            capture.close_workspace(workspace)
            assert workspace.closed and not workspace.source_ids and not workspace.search_cache
            assert await capture.process_task_once()
            active = db.memory_candidates(-110001, {110001}, 20)
            assert active, "no synthetic memory stored"
            synced = await memory.sync_index_once()
            assert synced > 0, "no point indexed"
            row = dict(active[0])
            collection = memory.index.collection(memory.provider.model_id, memory.provider.dimensions)
            points = await memory.index.search(-110001, row["vector"], memory.provider.model_id, memory.provider.dimensions, 8)
            assert any(p["id"] == row["id"] for p in points), "real vector search failed"
            assert not await memory.index.search(-110002, row["vector"], memory.provider.model_id, memory.provider.dimensions, 8)
            recall = await memory.retrieve(-110001, "之前测试网站打不开怎么恢复", user_ids={110001}, include_sources=True)
            assert recall, "hybrid recall failed"
            assert not await memory.retrieve(110001, "测试网站", user_ids={110001})
            assert db.memories.revoke(-110001, 110001, row["id"], delete=True, expected_version=row["version"])
            await memory.sync_index_once()
            points = await memory.index.search(-110001, row["vector"], memory.provider.model_id, memory.provider.dimensions, 8)
            assert all(p["id"] != row["id"] for p in points), "deleted vector still present"
            print(json.dumps({"live_memory_llm": live, "dense_embedding": True, "qdrant": True, "hybrid_recall": True,
                              "scope_isolation": True, "task_released": True, "delete_synced": True}))
        finally:
            # 只清理本次随机前缀的临时collection，绝不触碰生产qq_memory索引。
            if collection and re.fullmatch(re.escape(prefix) + r"_\d+_[a-f0-9]{12}", collection):
                await memory.index.request("DELETE", f"/collections/{collection}")
            await capture.close()
            await memory.close()
            await real.close()
            db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    asyncio.run(main(parser.parse_args().live))
