"""显式运行的真实API烟测：仅使用合成对话与临时数据库，不发送QQ消息。"""

import asyncio
import json
import tempfile
from pathlib import Path

from app.config import Settings
from app.database import Database
from app.group_context import GROUP_UNDERSTANDING
from app.llm import DeepSeekClient
from app.memory import MemoryService
from app.memory_capture import MemoryCapture


async def main():
    cfg = Settings().model_copy(update={"embedding_provider": "hash"})
    extraction_llm = DeepSeekClient(cfg, purpose="memory")
    chat_llm = DeepSeekClient(cfg)
    with tempfile.TemporaryDirectory() as folder:
        db = Database(Path(folder) / "synthetic.db")
        memory = MemoryService(cfg, db)
        capture = MemoryCapture(cfg, db, memory, extraction_llm)
        try:
            first = db.log_private_message(110001, "user", "请记住，我喜欢科幻电影，尤其是星际穿越。")
            result = await capture.extract(-110001, db.memory_source_rows(-110001, {first}), requested_by=110001)
            assert result["stored"] > 0, "未提取明确的电影偏好"
            second = db.log_private_message(110001, "user", "我现在不喜欢科幻电影了，改成喜欢悬疑片。请更新我的电影类型偏好。")
            result = await capture.extract(-110001, db.memory_source_rows(-110001, {second}), requested_by=110001)
            active = db.memory_candidates(-110001, {110001}, 20)
            assert any("悬疑" in row["content"] for row in active), "新偏好未更新"
            assert not db.memory_candidates(110001, {110001}, 20), "群/私聊隔离失败"
            user = db.log_private_message(110001, "user", "以后我发英语句子，你能先给正确写法，再解释原因吗？")
            bot = db.log_private_message(110001, "assistant", "好，你发过来，我先改句子再解释。")
            result = await capture.extract(-110001, db.memory_source_rows(-110001, {user, bot}))
            joint = [row for row in db.memory_candidates(-110001, {110001}, 20)
                     if row["memory_type"] in {"episode", "commitment"}]
            assert joint, f"没有提取双方明确的陪练约定，实际类型：{[row['memory_type'] for row in db.memory_candidates(-110001, {110001}, 20)]}"
            assert all(len(json.loads(row["sources_json"])) >= 2 for row in joint), "双向来源不完整"
            print(json.dumps({"memory_smoke": "passed", "active": len(db.memory_candidates(-110001, {110001}, 20)),
                              "joint_memories": len(joint)}, ensure_ascii=False))
            # 同一聊天模型直接读取多人结构；本调用只做合成回放评估，不加入线上工作流。
            replay = {
                "机器人QQ": 90,
                "历史": [
                    {"消息号": "1", "QQ": 11, "内容": "@22 今晚打游戏吗", "@目标QQ": [22]},
                    {"消息号": "2", "QQ": 22, "内容": "九点吧", "引用消息号": "1"},
                    {"消息号": "3", "QQ": 90, "角色": "assistant", "内容": "我不玩这个游戏", "回应消息号": "1"},
                ],
                "当前消息": {"QQ": 11, "内容": "我问的是他，不是你", "引用消息号": "3"},
                "评估任务": "仅输出JSON：original_addressee填第一条真正的收件人QQ整数；current_intent填当前是在纠正什么。不要角色表演。",
            }
            raw = await chat_llm.complete(json.dumps(replay, ensure_ascii=False), system=GROUP_UNDERSTANDING)
            judgment = json.loads(raw)
            assert judgment["original_addressee"] == 22, "群聊对象理解错误"
            print(json.dumps({"group_replay": "passed", "judgment": judgment}, ensure_ascii=False))
        finally:
            await capture.close()
            await memory.close()
            await extraction_llm.close()
            await chat_llm.close()
            db.close()


if __name__ == "__main__":
    asyncio.run(main())
