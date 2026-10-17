import copy
import json
import unittest
import asyncio
import tempfile
from pathlib import Path

import httpx

from app.agent import ReactAgent
from app.config import Settings
from app.llm import DeepSeekClient
from app.messaging import SEND_MESSAGE_TOOL, MessageSession
from app.database import Database
from app.tools import FINISH_TOOL, WAIT_FOR_MESSAGES_TOOL, ToolRegistry


def calls(*items, usage=None):
    return {"tool_calls": [{"id": str(i), "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
        for i, (name, arguments) in enumerate(items)], "_usage": usage or {}}


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def request_message(self, messages, temperature, tools):
        self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
        try:
            result = next(self.replies)
        except StopIteration:
            raise AssertionError("发生了不必要的额外模型调用")
        # 每次模型响应使用唯一工具调用ID。
        for call in result.get("tool_calls", []):
            call["id"] = f"{len(self.requests)}-{call['id']}"
        return result


def registry(*, group=True, enabled=True, fail_first=False):
    result = ToolRegistry()
    result.batch_completion_enabled = enabled
    delivered = []
    async def send(arguments):
        if fail_first and not delivered:
            delivered.append("rejected")
            return {"ok": False, "error": "参数校验失败，尚未提交"}
        # 控制参数不能泄漏到原有发送器。
        assert set(arguments) == {"text"}
        result.output_attempted = True
        result.outputs_sent += 1
        delivered.append(arguments["text"])
        return {"ok": True, "sent": 1}
    async def finish(arguments):
        if not result.output_attempted:
            return {"ok": False, "error": "先发送"}
        result.wait_for_messages = False
        return {"ok": True}
    async def wait(arguments):
        finished = await finish({})
        if not finished["ok"]:
            return finished
        result.wait_for_messages = True
        result.topic = arguments["topic"]
        return {"ok": True}
    result.register(SEND_MESSAGE_TOOL, send)
    result.register(FINISH_TOOL, finish, terminal=True)
    if group:
        result.register(WAIT_FOR_MESSAGES_TOOL, wait, terminal=True)
    return result, delivered


class EfficiencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_send_and_wait_in_one_model_call(self):
        tools, delivered = registry()
        llm = ScriptedLLM([calls(("send_message", {"text": "九点见", "next_action": "wait", "topic": "游戏时间"}))])
        outcome = await ReactAgent(Settings(_env_file=None), llm).run("几点", tools, system="群聊")
        self.assertEqual(outcome.steps, 1)
        self.assertTrue(outcome.wait_for_messages)
        self.assertEqual(outcome.topic, "游戏时间")
        self.assertEqual(delivered, ["九点见"])

    async def test_private_send_and_finish_in_one_model_call(self):
        tools, delivered = registry(group=False)
        llm = ScriptedLLM([calls(("send_message", {"text": "收到", "next_action": "finish"}))])
        outcome = await ReactAgent(Settings(_env_file=None), llm).run("你好", tools, system="私聊")
        self.assertEqual(outcome.steps, 1)
        self.assertTrue(tools.terminated)
        self.assertFalse(outcome.wait_for_messages)

    async def test_invalid_completion_is_rejected_before_sending(self):
        tools, delivered = registry()
        for arguments in ({"text": "hi", "next_action": "wait"},
                          {"text": "hi", "next_action": "invalid"},
                          {"text": "hi", "next_action": "finish", "topic": "不允许"}):
            result = await tools.execute("send_message", arguments)
            self.assertFalse(result["ok"])
        self.assertEqual(delivered, [])
        self.assertFalse(tools.terminated)

    async def test_failed_send_does_not_trigger_completion(self):
        tools, delivered = registry(fail_first=True)
        llm = ScriptedLLM([
            calls(("send_message", {"text": "too long", "next_action": "wait", "topic": "话题"})),
            calls(("send_message", {"text": "修正", "next_action": "wait", "topic": "话题"})),
        ])
        outcome = await ReactAgent(Settings(_env_file=None), llm).run("你好", tools, system="群聊")
        self.assertEqual(outcome.steps, 2)
        self.assertEqual(delivered, ["rejected", "修正"])
        self.assertTrue(outcome.wait_for_messages)

    async def test_uncertain_real_transport_never_retries_or_waits_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Database(Path(folder) / "db.sqlite")
            tools = ToolRegistry()
            attempts = []
            async def sender(target, segments):
                attempts.append(target)
                raise TimeoutError("unconfirmed")
            session = MessageSession(tools, db, sender, 11, private=True, automatic=False,
                                     preserve_layout=False, lock=asyncio.Lock())
            tools.register(SEND_MESSAGE_TOOL, session.execute)
            async def finish(arguments):
                return {"ok": tools.output_attempted}
            tools.register(FINISH_TOOL, finish, terminal=True)
            llm = ScriptedLLM([
                calls(("send_message", {"text": "你好", "next_action": "finish"})),
                calls(("send_message", {"text": "不能补发", "next_action": "finish"}), ("finish", {})),
            ])
            outcome = await ReactAgent(Settings(_env_file=None), llm).run("你好", tools, system="私聊")
            self.assertEqual(attempts, [11])
            self.assertEqual(outcome.sent, 0)
            self.assertFalse(outcome.wait_for_messages)
            db.close()

    async def test_no_actions_after_successful_fused_completion(self):
        tools, delivered = registry()
        llm = ScriptedLLM([calls(
            ("send_message", {"text": "第一句"}),
            ("send_message", {"text": "第二句", "next_action": "finish"}),
            ("send_message", {"text": "不应发送"}),
        )])
        outcome = await ReactAgent(Settings(_env_file=None), llm).run("你好", tools, system="群聊")
        self.assertEqual(delivered, ["第一句", "第二句"])
        self.assertEqual(outcome.steps, 1)
        self.assertFalse((await tools.execute("send_message", {"text": "已结束"}))["ok"])

    async def test_compatibility_plain_text_uses_program_default(self):
        tools, delivered = registry()
        tools.default_topic = "正在聊的游戏"
        llm = ScriptedLLM([{"content": "九点见"}])
        outcome = await ReactAgent(Settings(_env_file=None), llm).run("几点", tools, system="群聊")
        self.assertEqual(outcome.steps, 1)
        self.assertEqual(outcome.topic, "正在聊的游戏")
        self.assertEqual(delivered, ["九点见"])

    async def test_legacy_tool_flow_remains_available(self):
        tools, delivered = registry(enabled=False)
        definition = tools.definitions[0]["function"]["parameters"]["properties"]
        self.assertNotIn("next_action", definition)
        llm = ScriptedLLM([calls(("send_message", {"text": "收到"})), calls(("finish", {}))])
        outcome = await ReactAgent(Settings(_env_file=None), llm).run("你好", tools, system="群聊")
        self.assertEqual(outcome.steps, 2)
        self.assertEqual(delivered, ["收到"])

    async def test_actual_usage_aggregation(self):
        tools, _ = registry()
        llm = ScriptedLLM([calls(("send_message", {"text": "收到", "next_action": "finish"}),
                                usage={"prompt_tokens": 200, "completion_tokens": 40,
                                       "reasoning_tokens": 20, "cache_hit_tokens": 100})])
        outcome = await ReactAgent(Settings(_env_file=None), llm).run("你好", tools, system="群聊")
        self.assertEqual((outcome.prompt_tokens, outcome.completion_tokens, outcome.usage_requests), (200, 40, 1))
        self.assertEqual((outcome.reasoning_tokens, outcome.cache_hit_tokens), (20, 100))

    async def test_request_usage_is_private_and_reasoning_preserved(self):
        llm = DeepSeekClient(Settings(_env_file=None, deepseek_api_key="test"))
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "收到",
                                                                       "reasoning_content": "test reasoning"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_cache_hit_tokens": 80,
                          "completion_tokens_details": {"reasoning_tokens": 10}}})
        await llm.client.aclose()
        llm.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            value = await llm.request_message([{"role": "user", "content": "hi"}], 0.8, None)
            self.assertEqual(value["_usage"]["prompt_tokens"], 100)
            self.assertEqual(value["reasoning_content"], "test reasoning")
            self.assertNotIn("_usage", requests[0])
        finally:
            await llm.close()


if __name__ == "__main__":
    unittest.main()
