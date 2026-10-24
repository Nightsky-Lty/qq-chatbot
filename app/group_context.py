"""仅从真实@与引用元数据判断寻址，不推断语义关系。"""


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


def build_group_history(db, settings, group_id, batch, image_lookup=None):
    """为独立调用方提供与聊天主流程相同的连续原文视图。"""
    from app.working_memory import WorkingMemoryManager
    from app.chat_transcript import ChatTranscript
    snapshot = WorkingMemoryManager(settings, db, None).snapshot(
        group_id, before_id=min(e.storage_id for e in batch.entries),
        roots=[e.reply_to for e in batch.entries if e.reply_to],
        bot_id=batch.focus.self_id, image_lookup=image_lookup)
    rows = [*snapshot['近期往返'], *snapshot.get('引用补充', [])]
    rows.sort(key=lambda row: row['本地记录ID'])
    return [{'role': 'user', 'content': ChatTranscript(rows, {}).render(rows)}], rows
