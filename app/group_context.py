"""群聊结构化上下文：单一时间线、真实引用链、相邻往返与不确定寻址。"""

import json
from typing import Callable
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.config import Settings
from app.conversation import GroupBatch
from app.database import Database

GROUP_UNDERSTANDING = """【群聊结构阅读规则】
先分清谁对谁说、你之前实际说了什么，再理解问答、纠正、指代、反讽和接梗。历史中assistant是你自己的发言，不把不同成员混成一人，不逐条补答历史。
“当前焦点”“可回复”仅是候选资格，不证明在找你。显式@和引用优先；无@不等于在找你或与你无关，相邻不证明同一交流。程序的“会话关系”提供父消息和依据，核对原文后使用；LLM推断不是QQ显式引用，拿不准则追问或安静等待，不把调侃按字面当作新人物。
旧相对日期按消息时间理解；引用缺失不捏造。图片概要不是发图人原话或其真实经历；图片占位不等于可访问，先核对目录，不编权限或未收到的原因。"""


def addressing(mentions: list[int], reply: dict | None, *, bot_id: int, direct: bool = False) -> dict:
    evidence = []
    if bot_id in mentions:
        evidence.append("显式@机器人")
    if reply and reply["role"] == "assistant":
        evidence.append("引用机器人消息")
    others = [qq for qq in mentions if qq != bot_id]
    if reply and reply["role"] == "user":
        others.append(reply["user_id"])
    if evidence:
        target = "机器人（同时出现其他目标时结合原文）" if others else "机器人"
    elif others:
        target = "其他群友"
    elif direct:
        target = "机器人（命令或动作唤醒）"
        evidence.append("程序明确唤醒")
    else:
        target = "对象不明；可能广播、续接或群友间交流"
    result = {"对象判断": target}
    if evidence:
        result["元数据依据"] = evidence
    if others:
        result["其他目标QQ"] = sorted(set(others))
    return result


def attach_dialogue_relation(item: dict, relation: dict | None) -> None:
    if relation is None:
        return
    item['会话关系'] = relation
    target = relation['回应消息']
    address = item.setdefault('寻址', {})
    # 显式@/引用及唤醒优先；语义关系另行展示，不篡改QQ元数据。
    if address.get('元数据依据') or address.get('其他目标QQ'):
        return
    if target['QQ'] == item['QQ']:
        address['对象判断'] = '延续本人之前的发言；具体收件人结合往返判断'
    elif target['角色'] == 'assistant':
        address['对象判断'] = '机器人（会话关系判断；依据见会话关系）'
    else:
        address['对象判断'] = '回应群友的发言（会话关系判断；依据见会话关系）'


def build_group_history(db: Database, settings: Settings, group_id: int, batch: GroupBatch,
                        image_lookup: Callable[[int], list[dict]] | None = None) -> tuple[list[dict], list[dict]]:
    boundary = min(e.storage_id for e in batch.entries)
    bot_id = batch.focus.self_id
    recent = db.group_context_rows(group_id, settings.context_message_limit, boundary)
    # 机器人历史不再单独重复注入，统一放回时间线并补其原始回应。
    old_bot = [dict(r) for r in db.recent_bot_messages(group_id, settings.context_bot_reply_limit)]
    pool = {r["id"]: r for r in recent}
    priority: dict[int, int] = {r["id"]: 1 for r in recent}
    for r in old_bot:
        full = db.group_context_message(group_id, r["message_id"]) if r["message_id"] else r
        pool[r["id"]] = full or r
        priority[r["id"]] = max(priority.get(r["id"], 0), 2)
    # 个人旧话带上前后往返；不再单独注入脱离语境的本人发言。
    for r in db.recent_user_messages(group_id, batch.focus.user_id, settings.memory_user_history_limit, before_id=boundary):
        for around in db.group_context_around(group_id, r["id"], boundary):
            pool[around["id"]] = around
            priority.setdefault(around["id"], 0)
    roots = [e.reply_to for e in batch.entries if e.reply_to]
    roots += [r["reply_to_message_id"] for r in list(pool.values()) if r["reply_to_message_id"]]
    # 有界展开引用祖先；只从同一群数据库读取，不请求外部或下一批消息。
    for _ in range(4):
        pending = []
        for message_id in set(roots):
            row = db.group_context_message(group_id, message_id)
            if row and (row["id"] < boundary or row["role"] == "assistant"):
                pool[row["id"]] = row
                priority[row["id"]] = 3
                if row["reply_to_message_id"]:
                    pending.append(row["reply_to_message_id"])
        roots = pending
    blocked = db.memories.blocked_sources(group_id)
    pool = {i: row for i, row in pool.items() if i not in blocked}
    newest = sorted(pool, reverse=True)[:8]
    for record_id in newest:
        priority[record_id] = max(priority.get(record_id, 0), 2)

    def brief(row: dict) -> dict:
        item = {"消息号": row["message_id"], "QQ": row["user_id"], "昵称": row["nickname"],
                "角色": row["role"], "内容": row["content"][:1000],
                "本地记录ID": row["id"],
                "时间": datetime.fromtimestamp(row["created_at"], timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).isoformat()}
        if image_lookup:
            images = image_lookup(row["id"])
            if images:
                item["图片资料（非原话）"] = images
        return item

    by_message = {r["message_id"]: r for r in pool.values() if r["message_id"]}
    cache = {}
    for row in pool.values():
        item = brief(row)
        mentions = json.loads(row.get("mentions_json", "[]"))
        target = by_message.get(row["reply_to_message_id"])
        if mentions:
            item["@目标QQ"] = mentions
        if row["reply_to_message_id"]:
            item["回应消息号" if row["role"] == "assistant" else "引用消息号"] = row["reply_to_message_id"]
            item["引用或回应原文"] = brief(target) if target else {"消息号": row["reply_to_message_id"], "状态": "原文未找到"}
        item["寻址"] = addressing(mentions, target, bot_id=bot_id)
        if row["role"] == "assistant":
            item["身份说明"] = "你自己实际发送的消息"
        cache[row["id"]] = item
    # 引用连接 + 相邻上下文一起选；预算不足跳过整个片段，不传单边半句。
    selected: set[int] = set()
    ordered = sorted(pool)
    units = sorted(ordered, key=lambda i: (priority.get(i, 0), i), reverse=True)
    used = 0
    for record_id in units:
        index = ordered.index(record_id)
        unit = set(ordered[max(0, index - 1):index + 2])
        frontier = [record_id]
        for _ in range(4):
            frontier = [by_message[pool[i]["reply_to_message_id"]]["id"] for i in frontier
                        if pool[i]["reply_to_message_id"] in by_message]
            unit.update(frontier)
        extra = unit - selected
        size = sum(len(json.dumps(cache[i], ensure_ascii=False, separators=(",", ":"))) for i in extra)
        if used + size <= settings.context_max_chars - 300:
            selected.update(extra)
            used += size
    timeline = [cache[i] for i in sorted(selected)]
    for record_id in selected:
        target = by_message.get(pool[record_id]["reply_to_message_id"])
        if target is not None and target["id"] in selected:
            # 正文已在同一时间线内，保留明确索引和作者，不为每次引用重复上传正文。
            cache[record_id]["引用或回应原文"] = {"消息号": target["message_id"], "QQ": target["user_id"],
                                                "本地记录ID": target["id"], "原文位置": "本时间线"}
    history = [{"role": "user", "content": "【已完成的群聊时间线，仅背景；相邻不代表同一话题】\n"
               + json.dumps({"机器人QQ": bot_id, "消息": timeline,
                             "说明": "已去重；引用指向本时间线记录。缺省@/引用表示没有该元数据，不代表在找机器人。"}, ensure_ascii=False, separators=(",", ":"))}]
    return history, timeline
