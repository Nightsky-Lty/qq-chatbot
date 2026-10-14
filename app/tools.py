"""每次请求独立的工具注册表：定义、参数校验和绑定的执行器。"""

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolBinding:
    definition: dict[str, Any]
    handler: Handler
    terminal: bool = False


class ToolRegistry:
    def __init__(self):
        self.bindings: dict[str, ToolBinding] = {}
        self.observed_images: set[str] = set()
        self.receipts: list[str] = []
        self.receipt_error = False
        self.output_attempted = False
        self.outputs_sent = 0
        self.delivery_uncertain = False
        self.allow_silent = False
        self.wait_for_messages = False
        self.topic = ""

    def register(self, definition: dict[str, Any], handler: Handler, *, terminal: bool = False) -> None:
        name = definition["function"]["name"]
        if name in self.bindings:
            raise ValueError(f"duplicate tool: {name}")
        self.bindings[name] = ToolBinding(definition, handler, terminal)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [binding.definition for binding in self.bindings.values()]

    async def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        binding = self.bindings.get(name)
        if binding is None or not isinstance(arguments, dict):
            return {"ok": False, "error": "当前会话未注册该工具或参数不是对象。"}
        schema = binding.definition["function"]["parameters"]
        properties = schema.get("properties", {})
        if (set(schema.get("required", [])) - arguments.keys()
                or (schema.get("additionalProperties") is False and arguments.keys() - properties.keys())):
            return {"ok": False, "error": "工具参数缺失或包含不允许的字段。"}
        for key, value in arguments.items():
            spec = properties.get(key, {})
            kind = spec.get("type")
            if ((kind == "string" and not isinstance(value, str))
                    or (kind == "integer" and type(value) is not int)
                    or (kind == "boolean" and type(value) is not bool)):
                return {"ok": False, "error": "工具参数类型错误。"}
            if isinstance(value, str) and not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", 100000):
                return {"ok": False, "error": "工具参数长度超出限制。"}
            if type(value) is int and not spec.get("minimum", value) <= value <= spec.get("maximum", value):
                return {"ok": False, "error": "工具参数数值超出限制。"}
        logger.info("Agent action: tool=%s", name)
        try:
            return await binding.handler(arguments)
        except Exception as exc:
            # 不输出参数、聊天数据、文件路径、API响应或签名URL。
            logger.warning("Agent tool failed: tool=%s error=%s", name, type(exc).__name__)
            return {"ok": False, "error": "工具执行失败，请根据已完成操作决定是否结束，不要重复已提交的动作。"}


FINISH_TOOL = {"type": "function", "function": {
    "name": "finish", "description": "结束处理；群话题进入低活跃度，之后不再无@接话。想继续参与群话题用wait_for_messages。私聊正常回应后结束。",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}}

WAIT_FOR_MESSAGES_TOOL = {"type": "function", "function": {
    "name": "wait_for_messages", "description": "本批处理完成，继续参与当前群话题，等待新消息而不立即请求模型。无新消息不发言；话题结束应finish。",
    "parameters": {"type": "object", "properties": {
        "topic": {"type": "string", "minLength": 1, "maxLength": 120,
                  "description": "当前单一话题的简短锚点，不记录所有历史话题。"},
    }, "required": ["topic"], "additionalProperties": False},
}}
