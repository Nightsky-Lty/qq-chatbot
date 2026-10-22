"""每次请求独立的工具注册表：定义、参数校验和绑定的执行器。"""

import logging
import copy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
logger = logging.getLogger(__name__)
OUTPUT_TOOLS = {"send_message", "send_sticker", "send_voice", "send_poke"}


@dataclass(slots=True)
class ToolBinding:
    definition: dict[str, Any]
    handler: Handler
    terminal: bool = False


class ToolRegistry:
    def __init__(self):
        self.bindings: dict[str, ToolBinding] = {}
        self.receipts: list[str] = []
        self.receipt_error = False
        self.output_attempted = False
        self.outputs_sent = 0
        self.delivery_uncertain = False
        self.allow_silent = False
        self.wait_for_messages = False
        self.topic = ""
        self.batch_completion_enabled = True
        self.terminated = False
        self.default_topic = "当前群聊"
        self.working_memory = None

    def register(self, definition: dict[str, Any], handler: Handler, *, terminal: bool = False) -> None:
        name = definition["function"]["name"]
        if name in self.bindings:
            raise ValueError(f"duplicate tool: {name}")
        self.bindings[name] = ToolBinding(definition, handler, terminal)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        return [self._definition(binding) for binding in self.bindings.values()]

    def _definition(self, binding: ToolBinding) -> dict[str, Any]:
        if not self.batch_completion_enabled or binding.definition["function"]["name"] not in OUTPUT_TOOLS:
            return binding.definition
        definition = copy.deepcopy(binding.definition)
        properties = definition["function"]["parameters"]["properties"]
        actions = ["continue"] + (["finish"] if "finish" in self.bindings else []) + (["wait"] if "wait_for_messages" in self.bindings else [])
        properties["next_action"] = {"type": "string", "enum": actions,
            "description": "最后一次发送用finish结束，或wait继续群话题；成功后程序直接收尾。省略/continue仍需后续决策。"}
        if "wait_for_messages" in self.bindings:
            properties["topic"] = {"type": "string", "minLength": 1, "maxLength": 120,
                                   "description": "next_action=wait时必填的简短话题锚点。"}
        return definition

    async def execute(self, name: str, arguments: Any) -> dict[str, Any]:
        binding = self.bindings.get(name)
        if self.terminated:
            return {"ok": False, "error": "本轮已结束，不再执行动作。"}
        if binding is None or not isinstance(arguments, dict):
            return {"ok": False, "error": "当前会话未注册该工具或参数不是对象。"}
        schema = self._definition(binding)["function"]["parameters"]
        properties = schema.get("properties", {})
        if (set(schema.get("required", [])) - arguments.keys()
                or (schema.get("additionalProperties") is False and arguments.keys() - properties.keys())):
            return {"ok": False, "error": "工具参数缺失或包含不允许的字段。"}
        for key, value in arguments.items():
            spec = properties.get(key, {})
            kind = spec.get("type")
            if "enum" in spec and value not in spec["enum"]:
                return {"ok": False, "error": "工具参数不在允许值内。"}
            if ((kind == "string" and not isinstance(value, str))
                    or (kind == "integer" and type(value) is not int)
                    or (kind == "boolean" and type(value) is not bool)):
                return {"ok": False, "error": "工具参数类型错误。"}
            if isinstance(value, str) and not spec.get("minLength", 0) <= len(value) <= spec.get("maxLength", 100000):
                return {"ok": False, "error": "工具参数长度超出限制。"}
            if type(value) is int and not spec.get("minimum", value) <= value <= spec.get("maximum", value):
                return {"ok": False, "error": "工具参数数值超出限制。"}
        handler_arguments = dict(arguments)
        next_action = "continue"
        topic = None
        if self.batch_completion_enabled and name in OUTPUT_TOOLS:
            next_action = handler_arguments.pop("next_action", "continue")
            topic = handler_arguments.pop("topic", None)
            if next_action == "wait" and ("wait_for_messages" not in self.bindings or not isinstance(topic, str) or not topic.strip()):
                return {"ok": False, "error": "wait需要提供本轮可用的等待工具和非空topic；尚未发送。"}
            if topic is not None and next_action != "wait":
                return {"ok": False, "error": "topic只用于next_action=wait；尚未发送。"}
        logger.info("Agent action: tool=%s", name)
        try:
            result = await binding.handler(handler_arguments)
            if result.get("ok") is True and binding.terminal:
                self.terminated = True
            if result.get("ok") is True and next_action != "continue":
                terminal = "wait_for_messages" if next_action == "wait" else "finish"
                completion = await self.execute(terminal, {"topic": topic.strip()} if next_action == "wait" else {})
                result = {**result, "next_action": next_action, "completed": completion.get("ok") is True}
                if completion.get("ok") is not True:
                    result["completion_error"] = completion.get("error", "收尾失败，请依据回执继续。")
            return result
        except Exception as exc:
            # 不输出参数、聊天数据、文件路径、API响应或签名URL。
            logger.warning("Agent tool failed: tool=%s error=%s", name, type(exc).__name__)
            return {"ok": False, "error": "工具执行失败，请根据已完成操作决定是否结束，不要重复已提交的动作。"}


FINISH_TOOL = {"type": "function", "function": {
    "name": "finish", "description": "结束当前处理，群话题回到低活跃度；后续明确提问或程序随机触发可重新唤醒。想继续当前话题用wait_for_messages。私聊正常回应后结束。",
    "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
}}

WAIT_FOR_MESSAGES_TOOL = {"type": "function", "function": {
    "name": "wait_for_messages", "description": "本批处理完成，继续参与当前群话题，等待新消息而不立即请求模型。无新消息不发言；话题结束应finish。",
    "parameters": {"type": "object", "properties": {
        "topic": {"type": "string", "minLength": 1, "maxLength": 120,
                  "description": "当前单一话题的简短锚点，不记录所有历史话题。"},
    }, "required": ["topic"], "additionalProperties": False},
}}
