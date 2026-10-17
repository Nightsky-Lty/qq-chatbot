"""优化对比：默认离线上下文大小；--live用真实API回放合成任务，不发送QQ消息。"""

import argparse
import asyncio
import json
import subprocess
import tempfile
import time
import types
from pathlib import Path

from app.config import Settings
from app.conversation import GroupBatch, GroupMessage
from app.database import Database
from app.group_context import build_group_history


def offline(base_ref):
    source = subprocess.check_output(["git", "show", f"{base_ref}:app/group_context.py"], text=True)
    baseline = types.ModuleType("baseline_context")
    exec(compile(source, "baseline_group_context.py", "exec"), baseline.__dict__)
    with tempfile.TemporaryDirectory() as folder:
        db = Database(Path(folder) / "context.db")
        for i in range(24):
            db.log_message(7, 11 if i % 2 == 0 else 22, "群友A" if i % 2 == 0 else "群友B",
                           ("讨论游戏时间，以及具体的组队安排。" * 4) + str(i),
                           message_id=str(i + 1), reply_to_message_id=str(i) if i else "",
                           mentions=[22] if i == 0 else [])
        record = db.log_message(7, 11, "群友A", "@90 说好几点？", message_id="25", mentions=[90])
        batch = GroupBatch([GroupMessage(record, "25", 11, "群友A", "说好几点？", 90,
                                        direct=True, mentions=[90], reply_to="24")])
        cfg = Settings(_env_file=None)
        results = {}
        for name, builder in (("baseline", baseline.build_group_history), ("optimized", build_group_history)):
            history, rows = builder(db, cfg, 7, batch)
            results[name] = {"history_chars": sum(len(m["content"]) for m in history), "rows": len(rows)}
        assert results["optimized"]["rows"] == results["baseline"]["rows"], "此对比要求保留相同消息条数"
        results["character_reduction"] = round(1 - results["optimized"]["history_chars"] / results["baseline"]["history_chars"], 4)
        print(json.dumps(results, ensure_ascii=False))
        db.close()


async def live():
    from app.bot import BotService
    from app.llm import DeepSeekClient
    from app.reminders import ReminderService

    results = []
    with tempfile.TemporaryDirectory() as folder:
        for case in ("group_chat", "private_reminder"):
            for optimized in (False, True):
                cfg = Settings().model_copy(update={
                    "agent_batch_completion_enabled": optimized, "memory_enabled": False,
                    "embedding_provider": "hash", "sticker_enabled": False, "sticker_collect_enabled": False,
                    "web_search_enabled": False, "voice_enabled": False, "poke_enabled": False,
                })
                db = Database(Path(folder) / f"{case}-{optimized}.db")
                llm = DeepSeekClient(cfg)
                reminders = ReminderService(cfg, db)
                service = BotService(cfg, db, llm, reminders)
                delivered = []
                async def sender(target, segments):
                    delivered.append("".join(segment.get("data", {}).get("text", "") for segment in segments))
                    return str(1000 + len(delivered))
                try:
                    started = time.monotonic()
                    if case == "group_chat":
                        db.log_message(7, 11, "A", "@22 今晚几点组队？", message_id="1", mentions=[22])
                        db.log_message(7, 22, "B", "九点", message_id="2", reply_to_message_id="1")
                        text = "@90 A和B刚才约好了几点组队？只回答时间"
                        row = db.log_message(7, 33, "C", text, message_id="3", mentions=[90])
                        batch = GroupBatch([GroupMessage(row, "3", 33, "C", text, 90, direct=True, mentions=[90])])
                        outcome = await service._run_group_batch_locked(7, batch, sender)
                        success = any("九点" in text or "9" in text for text in delivered)
                    else:
                        text = "请设置一个五分钟后提醒我喝水的一次性提醒"
                        row = db.log_private_message(33, "user", text)
                        outcome = await service._run_ai_reply(33, 33, text, sender, private=True, private_source_id=row)
                        success = db.connection.execute("SELECT count(*) FROM reminders WHERE status='pending'").fetchone()[0] == 1 and bool(delivered)
                    results.append({"case": case, "batch_completion": optimized, "success": success,
                                    "steps": outcome.steps, "tool_calls": outcome.tool_calls,
                                    "prompt_tokens": outcome.prompt_tokens, "completion_tokens": outcome.completion_tokens,
                                    "reasoning_tokens": outcome.reasoning_tokens, "cache_hit_tokens": outcome.cache_hit_tokens,
                                    "usage_requests": outcome.usage_requests, "seconds": round(time.monotonic() - started, 2)})
                    print(json.dumps(results[-1], ensure_ascii=False), flush=True)
                    assert success and not outcome.error, f"合成任务未完成：{case}"
                finally:
                    await service.close()
                    await reminders.close()
                    await llm.close()
                    db.close()
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--base-ref", default="1074522")
    args = parser.parse_args()
    if args.live:
        asyncio.run(live())
    else:
        offline(args.base_ref)
