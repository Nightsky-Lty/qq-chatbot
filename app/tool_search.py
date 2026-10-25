"""本地embedding检索本轮已注册工具，按top-k加载；不执行业务动作。"""

import asyncio
import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass

from app.embeddings import HASH_MODEL_ID, EmbeddingProvider, sparse_vector, vector_similarity
from app.tools import ToolRegistry

logger = logging.getLogger(__name__)

TOOL_SEARCH_INSTRUCTIONS = """【工具按需加载】
当前tools仅含已加载工具。需要运行环境“可检索工具能力”中的其他功能时，先search_tools，用简短具体需求或准确工具名检索；它只加载匹配工具，不执行实际操作。依据回执，在下一步按新提供的schema调用，不能猜参数。未命中可换描述，不编造能力；原权限、回执和预算约束照常生效。"""

SEARCH_TOOLS_TOOL = {"type": "function", "function": {
    "name": "search_tools", "description": "用本地embedding按需求语义检索当前会话可用工具，加载top-k及必要依赖；不执行发图、联网或记忆写入等实际操作。准确工具名可直接定位。",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 120,
                  "description": "具体需求，如查询天气、回忆偏好、保存图片、发送表情；也可输入准确工具名。"},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 8,
                  "description": "最多匹配的工具数，不含必要依赖；默认使用本轮配置，优先描述一项具体需求。"},
    }, "required": ["query"], "additionalProperties": False},
}}

# 短任务描述放在文档前端，避免模型截断较长schema描述时丢失检索意图。
TOOL_HINTS = {
    "web_search": ("联网搜索与网页读取", "联网搜公开资料，查询天气、新闻、最新事实，核实说法，寻找网页和信息来源"),
    "web_fetch": ("联网搜索与网页读取", "打开已知链接或网址，阅读这篇网页文章正文，提取页面内容"),
    "memory_search": ("长期记忆", "检索以前记住的人物偏好、经历、群梗、约定和历史图片，回忆上次说过什么"),
    "memory_add": ("长期记忆", "按本人要求记住新的长期偏好、资料或事情，保存为长期记忆"),
    "memory_update": ("长期记忆", "本人更正已存信息，修改原来的长期记忆内容和版本"),
    "memory_forget": ("长期记忆", "本人要求忘记一条记忆，停用并防止旧证据重新记起"),
    "memory_delete": ("长期记忆", "本人要求彻底删除记忆记录、派生索引并阻止重建"),
    "memory_consolidate": ("长期记忆", "把本轮真实新证据提交后台整理为长期记忆，整合任务资料"),
    "read_chat_history": ("当前会话更早聊天原文", "向前翻阅当前群或当前用户私聊的连续聊天记录，读取更早的完整原话和往返，继续上一页历史"),
    "get_message_image": ("聊天图片查看与读字", "看看这张聊天图片、照片或截图，识别图中文字，分析画面内容"),
    "search_stickers": ("表情包检索与发送", "寻找适合当前情绪、场景或梗的表情包候选，按描述搜索图片编号"),
    "list_stickers": ("表情包检索与发送", "列出和翻页浏览表情包图库目录，看看有哪些可用表情"),
    "get_sticker_image": ("表情包检索与发送", "查看候选表情包原图，辨认图中文字和含义，不向群里发送"),
    "send_sticker": ("表情包检索与发送", "向当前聊天发一张表情包，用贴图接梗或表达开心、害羞、晚安等情绪"),
    "collect_sticker": ("会话图片收藏", "把当前群或本人私聊发来的这张图保存收藏到共享表情包库，收图、偷图、存图、存表情"),
    "send_poke": ("戳一戳", "戳一下、拍一下或敲一下当前聊天对象和允许的群友，轻轻互动或戳回去"),
    "list_voice_characters": ("群AI语音与音色", "查看可用语音音色和真实编号，选择想听的声音"),
    "send_voice": ("群AI语音与音色", "用语音回复，在群里把这句话说出来或念出来，合成短口语音频"),
    "create_reminder": ("私聊提醒", "设置定时提醒，约定未来时间通知本人做事；到时候叫我起床、吃药、喝水或开会"),
    "list_reminders": ("私聊提醒", "查看已经设置的提醒，列出待办提醒时间、事项和编号"),
    "cancel_reminder": ("私聊提醒", "取消一个已设置的提醒，以后不再发送这条定时通知"),
}
DEPENDENCIES = {
    "memory_update": ("memory_search",), "memory_forget": ("memory_search",),
    "memory_delete": ("memory_search",), "send_sticker": ("search_stickers",),
    "get_sticker_image": ("search_stickers",), "send_voice": ("list_voice_characters",),
    "cancel_reminder": ("list_reminders",),
}
BASE_TOOLS = {"send_message", "finish", "wait_for_messages", "skip_scheduled_sticker", "search_tools"}


@dataclass(frozen=True)
class ToolEntry:
    name: str
    label: str
    document: str


class ToolRetriever:
    """跨请求只缓存公开的工具说明向量；不缓存聊天查询，不依赖Qdrant。"""

    def __init__(self, provider: EmbeddingProvider, *, min_similarity: float = 0.35):
        self.provider = provider
        self.min_similarity = min_similarity
        self._vectors: OrderedDict[str, str] = OrderedDict()
        self._lock = asyncio.Lock()

    async def rank(self, query: str, entries: list[ToolEntry], top_k: int):
        # 准确工具名不是语义猜测；仍只允许定位当前会话已注册的工具。
        exact = next((entry for entry in entries if entry.name.casefold() == query.casefold()), None)
        if exact:
            return [(exact, 1.0)], "exact_name"
        documents = [entry.document for entry in entries]
        try:
            if self.provider.model_id == HASH_MODEL_ID or not await self.provider.start():
                raise RuntimeError("dense embeddings unavailable")
            async with self._lock:
                missing = list(dict.fromkeys(doc for doc in documents if doc not in self._vectors))
                if missing:
                    vectors = await self.provider.embed_documents(missing)
                    if len(vectors) != len(missing):
                        raise ValueError("invalid embedding batch")
                    self._vectors.update(zip(missing, vectors))
                vectors = [self._vectors[doc] for doc in documents]
                for doc in documents:
                    self._vectors.move_to_end(doc)
                while len(self._vectors) > 256:
                    self._vectors.popitem(last=False)
            vector = await self.provider.embed_query(query)
            engine, threshold = "embedding", self.min_similarity
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info("Tool embeddings fallback: error=%s", type(exc).__name__)
            # 降级必须在同一哈希空间内比较，不能混用BGE文档与哈希查询。
            vectors = [sparse_vector(document) for document in documents]
            vector = sparse_vector(query)
            engine, threshold = "hash_fallback", 0.08
        scored = sorted(((entry, vector_similarity(vector, doc)) for entry, doc in zip(entries, vectors)),
                        key=lambda item: (-item[1], item[0].name))
        return [(entry, score) for entry, score in scored if score >= threshold][:top_k], engine


class ToolSearch:
    def __init__(self, registry: ToolRegistry, retriever: ToolRetriever, *, max_calls: int, top_k: int):
        self.registry, self.retriever = registry, retriever
        self.max_calls, self.top_k = max_calls, top_k
        self.calls = 0
        self._cache = {}
        self.entries = []
        for name, binding in registry.bindings.items():
            if name in BASE_TOOLS or binding.terminal:
                continue
            description = binding.definition["function"].get("description", name)
            label, hint = TOOL_HINTS.get(name, (description[:60], description[:160]))
            self.entries.append(ToolEntry(name, label, hint))

    def catalogue(self) -> list[str]:
        return list(dict.fromkeys(entry.label for entry in self.entries))

    async def execute(self, arguments):
        query = arguments["query"].strip()
        if not query:
            return {"ok": False, "error": "请输入具体需求或工具名。"}
        top_k = arguments.get("top_k", self.top_k)
        key = (query.casefold(), top_k)
        if key in self._cache:
            return {**copy.deepcopy(self._cache[key]), "newly_loaded": [], "cached": True,
                    "remaining_calls": self.max_calls - self.calls}
        if self.calls >= self.max_calls:
            return {"ok": False, "error": "本轮工具检索次数已用完，请使用已加载能力或如实说明限制。"}
        self.calls += 1
        try:
            async with asyncio.timeout(5):
                matches, engine = await self.retriever.rank(query, self.entries, top_k)
        except TimeoutError:
            return {"ok": False, "error": "本地工具检索超时，未加载新工具；可稍后重试。",
                    "remaining_calls": self.max_calls - self.calls}
        names = {entry.name for entry, _ in matches}
        for name in list(names):
            names.update(dependency for dependency in DEPENDENCIES.get(name, ()) if dependency in self.registry.bindings)
        added = self.registry.load_tools(names)
        result = {"ok": True, "retrieval": engine,
                  "matches": [{"name": entry.name, "description": entry.document,
                               "score": round(score, 4)} for entry, score in matches],
                  "loaded_tools": sorted(names), "newly_loaded": sorted(added),
                  "remaining_calls": self.max_calls - self.calls,
                  "note": "匹配工具及必要依赖已加载；下一步按tools中的schema调用，尚未执行业务操作。" if names else "无达到阈值的工具，请参考能力目录换具体描述。"}
        if not names:
            result["available_capabilities"] = self.catalogue()
        logger.info("Tool search loaded: engine=%s matches=%s added=%s", engine, len(matches), len(added))
        self._cache[key] = copy.deepcopy(result)
        return result


def enable_tool_search(registry: ToolRegistry, retriever: ToolRetriever, *, max_calls: int, top_k: int, initial_tools=()) -> None:
    if registry.tool_search is not None:
        return
    search = ToolSearch(registry, retriever, max_calls=max_calls, top_k=top_k)
    if not search.entries:
        return
    registry.register(SEARCH_TOOLS_TOOL, search.execute)
    keep = (BASE_TOOLS | set(initial_tools)) & registry.bindings.keys()
    keep.update(name for name, binding in registry.bindings.items() if binding.terminal)
    registry.defer_tools(keep)
    registry.tool_search = search
