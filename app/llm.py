"""封装 DeepSeek 聊天请求，供聊天、锐评和风格模仿功能共用。"""

import json
from typing import Any, Awaitable, Callable

import httpx

from app.config import Settings
from app.persona import SYSTEM_PROMPT


# 完整角色卡在app/persona.py中维护，群聊、私聊及锐评复用；模仿提示词保持独立。


class LLMUnavailable(RuntimeError):
    """表示缺少 API 配置、请求失败或响应格式异常，供业务层统一处理。"""

    pass


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class DeepSeekClient:
    """复用异步 HTTP 客户端，发送非流式聊天请求并返回生成的文本。"""

    def __init__(self, settings: Settings):
        """保存配置并创建请求超时为 45 秒的 HTTP 客户端。"""
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=45)

    async def close(self) -> None:
        """服务退出时关闭客户端，释放连接资源。"""
        await self.client.aclose()

    async def complete(
        self,
        prompt: str,
        *,
        history: list[dict[str, str]] | None = None,
        system: str = SYSTEM_PROMPT,
        temperature: float = 0.8,
        tools: list[dict[str, Any]] | None = None,
        execute_tool: ToolExecutor | None = None,
    ) -> str:
        """生成回复并执行受限工具；媒体选择成功后允许只用图片回复。"""
        if not self.settings.deepseek_api_key:
            raise LLMUnavailable("尚未配置 DEEPSEEK_API_KEY")
        # 当前问题追加到最后，使模型结合前面的上下文作答。
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})
        # 每个请求最多4轮、8次工具调用；不执行任意代码，不无限跟随模型循环。
        receipts: list[str] = []
        cached: dict[str, dict[str, Any]] = {}
        calls_used = 0
        had_tool_error = False
        has_media_reply = False
        images_added: set[str] = set()
        allowed = {tool["function"]["name"] for tool in tools or []}
        for _ in range(4):
            try:
                message = await self._request_message(messages, temperature, tools)
            except LLMUnavailable:
                # 工具可能已成功写库，不能因后续模型失败让用户以为没有保存。
                if receipts or has_media_reply:
                    return "\n".join(dict.fromkeys(receipts))
                raise
            calls = message.get("tool_calls") or []
            if not calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    if receipts:
                        return "\n".join(dict.fromkeys(receipts))
                    if has_media_reply:
                        return ""
                    raise LLMUnavailable("模型没有返回可用文字。")
                if receipts:
                    # 精确时间与编号以真实工具回执为准，避免模型错误确认或遗漏。
                    if had_tool_error:
                        return "\n".join(dict.fromkeys(receipts))
                    return content.strip() + "\n\n" + "\n".join(dict.fromkeys(receipts))
                return content.strip()
            if execute_tool is None or not tools or not isinstance(calls, list) or len(calls) > 8 - calls_used:
                if receipts or has_media_reply:
                    return "\n".join(dict.fromkeys(receipts))
                raise LLMUnavailable("模型返回了不支持或过多的工具调用。")
            messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
            image_content: list[dict[str, Any]] = []
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                    if receipts or has_media_reply:
                        return "\n".join(dict.fromkeys(receipts))
                    raise LLMUnavailable("模型工具调用缺少有效编号。")
                function = call.get("function") or {}
                name = function.get("name") if isinstance(function, dict) else None
                call_id = call["id"]
                if call_id in cached:
                    result = cached[call_id]
                else:
                    try:
                        arguments = json.loads(function.get("arguments", "{}")) if isinstance(function, dict) else None
                    except (TypeError, ValueError):
                        arguments = None
                    if not isinstance(name, str) or name not in allowed or not isinstance(arguments, dict):
                        result = {"ok": False, "message": "工具或参数无效，请重新确认。"}
                    else:
                        try:
                            result = await execute_tool(name, arguments)
                        except Exception as exc:
                            if receipts or has_media_reply:
                                return "\n".join(dict.fromkeys(receipts)) + "\n后续工具操作未完成，请稍后重试。"
                            raise LLMUnavailable("工具执行失败，请稍后重试。") from exc
                    cached[call_id] = result
                    had_tool_error = had_tool_error or result.get("ok") is not True
                    if name == "send_sticker" and result.get("ok") is True and result.get("reply_without_text") is True:
                        has_media_reply = True
                    if isinstance(result.get("message"), str):
                        receipts.append(result["message"])
                calls_used += 1
                # Chat Completions的tool消息仍是文本；图片放在所有工具回执之后的user消息。
                # 避免把Base64当作工具JSON文字计费，或在多工具调用的回执之间插入user消息。
                tool_result = {key: value for key, value in result.items() if key != "_image_content"}
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(tool_result, ensure_ascii=False)})
                sticker_id = result.get("sticker_id")
                if (name == "get_sticker_image" and result.get("ok") is True
                        and isinstance(sticker_id, str) and sticker_id not in images_added
                        and isinstance(result.get("_image_content"), list)):
                    image_content.extend(result["_image_content"])
                    images_added.add(sticker_id)
            if image_content:
                messages.append({"role": "user", "content": image_content})
        if receipts or has_media_reply:
            return "\n".join(dict.fromkeys(receipts))
        raise LLMUnavailable("工具调用轮数超出限制，请稍后重新说明你的需求。")

    async def _request_message(
        self, messages: list[dict[str, Any]], temperature: float, tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """发送一次补全请求并校验响应；工具由业务层按会话提供。"""
        payload: dict[str, Any] = {
            "model": self.settings.deepseek_model, "messages": messages, "temperature": temperature,
            "thinking": {"type": "disabled"}, "max_tokens": 500, "stream": False,
        }
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto", "max_tokens": 1500})
        try:
            response = await self.client.post(
                f"{self.settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("invalid message")
            return message
        except httpx.HTTPError as exc:
            # 状态码错误带有响应体；网络连接和超时错误可能没有响应。
            response = getattr(exc, "response", None)
            detail = response.text[:300] if response is not None else str(exc)
            raise LLMUnavailable(f"DeepSeek 请求失败：{detail}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            # 将 JSON 解析或字段结构错误转换为业务层可识别的异常。
            detail = response.text[:300]
            raise LLMUnavailable(f"DeepSeek 请求失败：{detail}") from exc
