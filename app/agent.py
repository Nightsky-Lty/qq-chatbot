"""基于函数调用的 ReAct：决策 → 工具动作 → 观察结果 → 再决策。"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.llm import DeepSeekClient, LLMUnavailable
from app.tools import ToolRegistry
from app.prompting import ContextPacket, PromptBuilder, PromptScene, program_feedback

logger = logging.getLogger(__name__)
IMAGE_TOOLS = {"get_message_image", "get_sticker_image"}


@dataclass(slots=True)
class AgentOutcome:
    steps: int
    tool_calls: int
    sent: int
    error: str | None = None
    wait_for_messages: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cache_hit_tokens: int = 0
    usage_requests: int = 0
    # 仅业务层等待解析额度时使用；不表示已调用或需要轮询回复模型。
    retry_after: float = 0


class ReactAgent:
    def __init__(self, settings: Settings, llm: DeepSeekClient):
        self.settings = settings
        self.llm = llm

    async def run(self, prompt: str, registry: ToolRegistry, *, system: str,
                  history: list[dict[str, str]] | None = None,
                  context: ContextPacket | None = None, scene: PromptScene | None = None) -> AgentOutcome:
        messages = PromptBuilder.messages(prompt, registry, persona=system, scene=scene,
                                          context=context, history=history)
        image_keys: set[str] = set()
        cached: dict[str, tuple[str, dict[str, Any]]] = {}
        steps = calls_used = 0
        error = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cache_hit_tokens": 0}
        usage_requests = 0
        try:
            async with asyncio.timeout(self.settings.agent_timeout_seconds):
                for step in range(self.settings.agent_max_steps):
                    steps = step + 1
                    message = await self.llm.request_message(messages, 0.8, registry.definitions)
                    observed_usage = message.get("_usage", {})
                    if isinstance(observed_usage, dict):
                        if "prompt_tokens" in observed_usage and "completion_tokens" in observed_usage:
                            usage_requests += 1
                        for key in usage:
                            value = observed_usage.get(key)
                            if type(value) is int and value >= 0:
                                usage[key] += value
                    calls = message.get("tool_calls") or []
                    assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
                    if isinstance(message.get("reasoning_content"), str):
                        # 思考模式带tools时必须原样回传，包括没有调用工具但还需继续的轮次。
                        # 只存在本次请求的内存messages，发送工具和日志均不会读取该字段。
                        assistant["reasoning_content"] = message["reasoning_content"]
                    if calls:
                        assistant["tool_calls"] = calls
                    messages.append(assistant)
                    if not calls:
                        send_result = None
                        content = message.get("content")
                        text = content.strip() if isinstance(content, str) else ""
                        if text in {"[不插话]", "不插话"}:
                            text = ""
                        if not registry.output_attempted and "send_message" in registry.bindings:
                            text = "\n".join(dict.fromkeys(registry.receipts)) or text
                            if text:
                                send_result = await registry.execute("send_message", {"text": text})
                        if registry.batch_completion_enabled and (registry.outputs_sent or registry.delivery_uncertain):
                            terminal = "wait_for_messages" if "wait_for_messages" in registry.bindings and not registry.delivery_uncertain else "finish"
                            completion = await registry.execute(terminal, {})
                            if completion.get("ok") is True:
                                break
                        if text and "wait_for_messages" in registry.bindings:
                            state = send_result if send_result is not None else {"note": "此前已有发送尝试，不再重复提交最终文字。"}
                            messages.append(program_feedback("兼容发送回执", state))
                            continue
                        final_result = await registry.execute("finish", {})
                        if final_result.get("ok") is True:
                            break
                        # 用真实执行状态让模型补上必要发送。
                        messages.append(program_feedback("未完成收尾", final_result))
                        continue
                    if not isinstance(calls, list) or len(calls) > self.settings.agent_max_tool_calls - calls_used:
                        raise LLMUnavailable("工具调用超出本轮限制。")
                    image_content = []
                    finished = False
                    for call in calls:
                        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                            raise LLMUnavailable("工具调用缺少有效编号。")
                        function = call.get("function") or {}
                        if not isinstance(function, dict):
                            function = {}
                        name = function.get("name")
                        if not isinstance(name, str):
                            name = ""
                        try:
                            arguments = json.loads(function.get("arguments", "{}"))
                        except (ValueError, TypeError):
                            arguments = None
                        signature = json.dumps([name, arguments], ensure_ascii=False, sort_keys=True)
                        call_id = call["id"]
                        key_field = "image_id" if name == "get_message_image" else "sticker_id"
                        key = f"{name}:{arguments.get(key_field)}" if isinstance(arguments, dict) else ""
                        if finished:
                            result = {"ok": False, "error": "结束工具之后不执行其他动作。"}
                        elif call_id in cached:
                            previous_signature, previous_result = cached[call_id]
                            result = previous_result if previous_signature == signature else {"ok": False, "error": "同一个调用编号不能更换动作或参数。"}
                        elif isinstance(name, str) and name in IMAGE_TOOLS and key not in image_keys and len(image_keys) >= 3:
                            result = {"ok": False, "error": "本轮聊天图与表情原图合计最多3张。"}
                        else:
                            result = await registry.execute(name if isinstance(name, str) else "", arguments)
                            cached[call_id] = (signature, result)
                        calls_used += 1
                        public_result = {k: v for k, v in result.items() if not k.startswith("_")}
                        messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(public_result, ensure_ascii=False, separators=(",", ":"))})
                        logger.info("Agent observation: tool=%s ok=%s", name if name in registry.bindings else "invalid", result.get("ok") is True)
                        if (name in IMAGE_TOOLS and result.get("ok") is True and key not in image_keys
                                and len(image_keys) < 3 and isinstance(result.get("_image_content"), list)):
                            image_content.extend(result["_image_content"])
                            image_keys.add(key)
                        binding = registry.bindings.get(name) if isinstance(name, str) else None
                        if registry.terminated or (binding is not None and binding.terminal and result.get("ok") is True):
                            finished = True
                    if image_content:
                        messages.append({"role": "user", "content": image_content})
                    if finished:
                        break
                else:
                    error = "Agent达到最大步骤数。"
        except (LLMUnavailable, TimeoutError) as exc:
            # 部分动作已完成时只报告状态，不重复提交消息或撤销已保存的收藏/提醒。
            error = "模型请求失败或处理超时。" if isinstance(exc, LLMUnavailable) else "Agent处理超时。"
        if registry.receipts and not registry.output_attempted and "send_message" in registry.bindings:
            await registry.execute("send_message", {"text": "\n".join(dict.fromkeys(registry.receipts))})
        logger.info("Agent finished: steps=%s tool_calls=%s images=%s sent=%s error=%s prompt_tokens=%s completion_tokens=%s reasoning_tokens=%s cache_hit_tokens=%s usage_requests=%s",
                    steps, calls_used, len(image_keys), registry.outputs_sent, error is not None, *usage.values(), usage_requests)
        return AgentOutcome(steps, calls_used, registry.outputs_sent, error=error, wait_for_messages=registry.wait_for_messages,
                            **usage, usage_requests=usage_requests)
