"""六种会话绑定的记忆操作；模型不能选择命名空间、身份、路径或SQL。"""

import json
import re

from app.memory_capture import MemoryCapture
from app.tools import ToolRegistry

MEMORY_TOOL_INSTRUCTIONS = """【四层记忆与六种操作】
工作记忆自动注入，更早聊天原文用已注册的read_chat_history（群聊仅当前群，私聊仅当前用户）；memory_search只检索长期event/abstract/perceptual及all，必要时include_sources展开证据。同群可查已授权他人记录，不猜身份；记忆中的陈述不是已核实事实。
本人明确要求记住用memory_add，修正时先搜索再memory_update并用真实编号和版本。memory_forget停用并阻止重建，memory_delete清除记录与派生索引；二者须本轮明确授权，个人仅本人操作，共享概念仅已验证群管理者管理。
任务结束自动整合真实新证据，不保存推理或重存检索结果，不因无关自行遗忘。queued或stored=0不能声称已记住，约定也不是提醒已执行的证明。"""


def register_memory_tools(registry: ToolRegistry, capture: MemoryCapture, namespace: int,
                          user_id: int, sources: list[dict], *, workspace=None, media_grant=None,
                          working_consolidate=None) -> None:
    def register(name, description, properties, required, handler):
        registry.register({"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}, handler)

    own_text = "\n".join(r["content"] for r in sources if r["role"] == "user" and capture._subject(namespace, r) == user_id
                         and (workspace is None or r["id"] in workspace.new_source_ids))
    intent_text = re.sub(r"(?:别|不要|不能|不许)(?:忘(?:记|掉)|删除|清除|移除)", "", own_text)
    remembered = None
    search_cache = workspace.search_cache if workspace else {}
    boundary = max((r["id"] for r in sources), default=None)
    with capture.db._lock:
        member = capture.db.connection.execute("SELECT role FROM group_members WHERE group_id=? AND user_id=?", (namespace, user_id)).fetchone() if namespace > 0 else None
    shared_admin = namespace > 0 and ((member is not None and member[0] in {"owner", "admin"}) or user_id in capture.settings.admin_qq_ids)

    async def search(arguments):
        query = arguments["query"].strip()
        if not query:
            return {"ok": False, "error": "查询词不能为空"}
        filters = {k: arguments[k] for k in ("subject", "after", "before", "include_sources") if k in arguments}
        if arguments.get("layer", "all") != "all":
            filters["layer"] = arguments["layer"]
        subject = filters.get("subject")
        if subject is not None and namespace > 0 and subject != 0:
            known = capture.db.group_members(namespace, {subject})
            if not known and not any(r.get("user_id") == subject for r in sources):
                return {"ok": False, "error": "该QQ不在本群已知身份中，请明确人物而非猜测"}
        signature = json.dumps([query, filters], sort_keys=True, ensure_ascii=False)
        cached = search_cache.get(signature)
        if cached is not None:
            items = []
            for memory_id, version in cached:
                row = capture.db.memories.row(namespace, memory_id)
                if row and row["version"] == version and capture.memory.retriever.eligible(row, boundary):
                    items.append(capture.memory.retriever.public(row, include_sources=filters.get("include_sources", False)))
            if len(items) == len(cached):
                return {"ok": True, "memories": items, "cached": True}
        items = await capture.memory.retrieve(namespace, query, user_ids={user_id}, boundary=boundary, **filters)
        if media_grant:
            for item in items:
                if type(item.get("media_ref_id")) is int:
                    media_grant(item["media_ref_id"])
                for media_id in item.get("关联图片", []):
                    media_grant(int(media_id[6:]))
        if items and all(type(r.get("记忆编号")) is int for r in items):
            search_cache[signature] = [(r["记忆编号"], r["版本"]) for r in items]
        return {"ok": True, "memories": items}

    async def add(arguments):
        nonlocal remembered
        if not re.search(r"记住|记一下|记下来|记着|保存.*偏好", own_text):
            return {"ok": False, "error": "仅本人明确要求记住时使用；普通聊天由任务结束整合"}
        async with capture.lock:
            if remembered is None:
                remembered = await capture.extract(namespace, sources, requested_by=user_id,
                    new_source_ids=set(workspace.new_source_ids) if workspace else None)
            return remembered

    def target(arguments):
        row = capture.db.memories.row(namespace, arguments["memory_id"])
        if not row or not (row["user_id"] == user_id or (row["user_id"] == 0 and shared_admin)):
            raise ValueError("记录不存在、已失效或无权修改")
        if row["version"] != arguments["expected_version"]:
            raise ValueError("记忆版本已变化，请重新搜索")
        return row

    async def update(arguments):
        if not re.search(r"改|纠正|不再|换|现在|取消|之前.*错|其实|从今天", own_text):
            return {"ok": False, "error": "更新需要本人本轮明确纠正；不会根据模型猜测改写记忆"}
        async with capture.lock:
            try:
                row = target(arguments)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return await capture.extract(namespace, sources, requested_by=user_id, update_target=row,
                new_source_ids=set(workspace.new_source_ids) if workspace else None)

    async def revoke(arguments, delete=False):
        pattern = r"删除|清除|删掉|移除" if delete else r"忘记|忘掉|别记|不要记|不许记"
        if not re.search(pattern, intent_text):
            return {"ok": False, "error": "没有本人本轮明确的删除/遗忘请求"}
        async with capture.lock:
            try:
                row = target(arguments)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            changed = capture.db.memories.revoke(namespace, row["user_id"], row["id"], delete=delete, expected_version=row["version"])
        return {"ok": changed, "deleted" if delete else "forgotten": changed,
                "note": "立即停止检索/注入，索引异步清理；不声称历史备份或全部原始聊天已物理擦除"}

    async def consolidate(arguments):
        if workspace is None or workspace.closed:
            return {"ok": False, "error": "当前没有可整合的任务工作区"}
        if not capture.db.capture_enabled:
            return {"ok": False, "error": "自动整合已关闭；本人明确要求可使用memory_add"}
        archived = await working_consolidate() if working_consolidate is not None else False
        return {"ok": True, "scheduled": True, "stored": 0,
                "working_archive_queued": archived,
                "note": "短期原话分批归档，本轮证据在任务结束时补交；尚未长期保存，不复制摘要、旧检索或模型草稿"}

    register("memory_search", "搜索本会话长期记忆；all综合事件/抽象/感知各层，支持人物、时间与来源摘录。不跨群/私聊。当前会话更早原文使用read_chat_history。",
             {"query": {"type": "string", "minLength": 1, "maxLength": 300},
              "layer": {"type": "string", "enum": ["all", "event", "abstract", "perceptual"]},
              "subject": {"type": "integer", "minimum": 0, "description": "明确的本群QQ；群共享概念为0；不确定时省略，不按昵称猜测"},
              "after": {"type": "integer", "minimum": 0, "description": "发生时间UTC Unix秒下界"},
              "before": {"type": "integer", "minimum": 0, "description": "发生时间UTC Unix秒上界"},
              "include_sources": {"type": "boolean", "description": "必要时展开最多3条来源摘录"}}, ["query"], search)
    register("memory_add", "本人明确要求记住时，将本轮真实原话交给独立整理器；不接受模型自编事实。", {}, [], add)
    mutation = {"memory_id": {"type": "integer", "minimum": 1}, "expected_version": {"type": "integer", "minimum": 1}}
    register("memory_update", "根据本轮明确纠正修订搜索得到的记忆，校验版本并保留历史。", mutation, list(mutation), update)
    register("memory_delete", "有明确请求时清除本人指定记忆及派生资料，不删除他人的原始聊天。", mutation, list(mutation), lambda args: revoke(args, True))
    register("memory_forget", "本人明确要求时停用指定记忆并阻断旧来源重新提取，不是降低权重。", mutation, list(mutation), revoke)
    register("memory_consolidate", "在本任务结束时将短期新证据筛选成长期事件/概念，后台自动执行，不是合并历史库。", {}, [], consolidate)
