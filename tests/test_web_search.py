import asyncio
import copy
import json
import unittest

import httpx

from app.agent import ReactAgent
from app.config import Settings
from app.tools import FINISH_TOOL, ToolRegistry
from app.web_search import DEEPSEEK_TOOLS_URL, WEB_TOOLS, WEB_INSTRUCTIONS, WebSearchService, WebSession, user_urls


URLS = [f"https://example.com/article-{i}" for i in range(5)]


def rows(count=5, *, content=True):
    return [{"title": f"标题{i}", "url": URLS[i], "snippet": "短摘要", "summary": "相关长摘要",
             "content": "# 正文\n\n正文证据，不同于搜索摘要。" if content else "",
             "site_name": "Example", "date": "2026-10-18T00:00:00Z", "authority": 3,
             "rerank_score": 2.5} for i in range(count)]


class StubAPI:
    def __init__(self):
        self.search_rows = rows()
        self.fetch_value = {"title": "页面标题", "content": "# Fetch正文\n\n补读证据。"}
        self.requests = []
        self.statuses = {}
        self.override = {}

    async def __call__(self, request):
        self.requests.append(request)
        name = request.url.path.rsplit("/", 1)[-1]
        value = self.override.get(name, {"results": self.search_rows} if name == "search" else self.fetch_value)
        data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode()
        return httpx.Response(self.statuses.get(name, 200), headers={"content-type": "application/json"},
                              stream=httpx.ByteStream(data))


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    def setup_service(self, **settings):
        api = StubAPI()
        cfg = Settings(_env_file=None, deepseek_api_key="fake-test-key", **settings)
        service = WebSearchService(cfg, transport=httpx.MockTransport(api))
        service._request_interval = 0  # 除节流专项测试外，离线测试无需等待。
        self.addAsyncCleanup(service.close)
        return service, WebSession(), api

    async def search(self, service, session, query="测试"):
        return await service.execute_tool(session, "web_search", {"query": query})

    async def fetch(self, service, session, url=URLS[0]):
        return await service.execute_tool(session, "web_fetch", {"url": url})

    async def test_search_requests_official_api_with_reused_key_and_content(self):
        service, session, api = self.setup_service(deepseek_base_url="https://chat-proxy.example.com/beta")
        result = await self.search(service, session, "文档 site:python.org after:2025")
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "deepseek")
        self.assertEqual(result["pages_read"], 5)
        self.assertEqual(session.fetch_calls, 0)
        self.assertEqual(len(api.requests), 1)
        request = api.requests[0]
        self.assertEqual(str(request.url), DEEPSEEK_TOOLS_URL + "/search")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.headers["authorization"], "Bearer fake-test-key")
        self.assertEqual(json.loads(request.content), {
            "query": "文档 site:python.org after:2025", "count": 5, "need_content": True})
        self.assertIn("正文证据", result["results"][0]["page"]["content"])
        self.assertIn("\n\n", result["results"][0]["page"]["content"])
        self.assertEqual(result["results"][0]["date"], "2026-10-18T00:00:00Z")
        self.assertEqual(result["results"][0]["authority"], 3)

    async def test_search_content_is_cached_and_never_triggers_extra_fetch(self):
        service, session, api = self.setup_service()
        result = await self.search(service, session)
        for item in result["results"]:
            self.assertEqual(await self.fetch(service, session, item["url"]), item["page"])
        self.assertEqual(await self.search(service, session), result)
        await self.search(service, session, "换关键词")
        self.assertEqual(len(api.requests), 2)  # 两个Search，零Fetch。
        self.assertEqual(session.fetch_calls, 0)

    async def test_missing_content_is_only_fetched_when_model_requests_it(self):
        service, session, api = self.setup_service()
        api.search_rows = rows(content=False)
        result = await self.search(service, session)
        self.assertEqual(result["pages_read"], 0)
        self.assertTrue(all(item["page_status"] == "not_read" for item in result["results"]))
        self.assertIn("只有搜索摘要", result["note"])
        self.assertEqual(len(api.requests), 1)
        page = await self.fetch(service, session)
        self.assertTrue(page["ok"])
        self.assertEqual(page["source"], "deepseek_fetch")
        self.assertEqual(str(api.requests[-1].url), DEEPSEEK_TOOLS_URL + "/fetch")
        self.assertEqual(json.loads(api.requests[-1].content), {"url": URLS[0]})
        self.assertEqual(session.fetch_calls, 1)
        self.assertEqual(page, await self.fetch(service, session))
        self.assertEqual(len(api.requests), 2)

    async def test_direct_user_link_does_not_search(self):
        service, session, api = self.setup_service()
        session.allowed_urls = user_urls([f"帮我读一下 {URLS[0]}。"])
        self.assertTrue((await self.fetch(service, session))["ok"])
        self.assertEqual(session.search_calls, 0)
        self.assertEqual(len(api.requests), 1)
        self.assertTrue(str(api.requests[0].url).endswith("/fetch"))

    async def test_search_and_fetch_budgets_are_independent(self):
        service, session, api = self.setup_service(web_fetch_max_calls=1, web_search_max_calls=1)
        api.search_rows = rows(content=False)
        await self.search(service, session)
        self.assertFalse((await self.search(service, session, "其他问题"))["ok"])
        self.assertTrue((await self.fetch(service, session))["ok"])
        self.assertFalse((await self.fetch(service, session, URLS[1]))["ok"])
        self.assertTrue((await self.fetch(service, session))["ok"])  # 缓存不受已用完额度限制。
        self.assertEqual(session.search_calls, 1)
        self.assertEqual(session.fetch_calls, 1)
        self.assertEqual(len(api.requests), 2)

    async def test_body_caps_cache_overflow_without_marking_it_read(self):
        service, session, api = self.setup_service(web_fetch_max_chars=1000, web_search_content_max_chars=1500)
        for item in api.search_rows:
            item["content"] = "正文" * 1000
        result = await self.search(service, session)
        self.assertEqual([item["page_status"] for item in result["results"]],
                         ["read", "read", "cached", "cached", "cached"])
        self.assertEqual(sum(len(item.get("page", {}).get("content", "")) for item in result["results"]), 1500)
        self.assertTrue(result["results"][0]["page"]["truncated"])
        self.assertTrue(result["results"][1]["page"]["truncated"])
        page = await self.fetch(service, session, URLS[2])
        self.assertEqual(len(page["content"]), 1000)
        self.assertEqual(session.fetch_calls, 0)
        self.assertEqual(len(api.requests), 1)

    async def test_need_content_off_and_session_isolation(self):
        service, session, api = self.setup_service(web_search_need_content=False)
        api.search_rows = rows(content=False)
        result = await self.search(service, session)
        self.assertFalse(json.loads(api.requests[0].content)["need_content"])
        self.assertEqual(result["pages_read"], 0)
        self.assertFalse((await self.fetch(service, WebSession()))["ok"])
        self.assertEqual(len(api.requests), 1)
        self.assertTrue((await self.fetch(service, session))["ok"])

    async def test_fetch_error_and_empty_body_are_cached_not_faked_as_read(self):
        for status, content in [(404, "error-body-with-secret"), (200, "")]:
            service, session, api = self.setup_service()
            session.allowed_urls.add(URLS[0])
            api.statuses["fetch"] = status
            api.fetch_value["content"] = content
            result = await self.fetch(service, session)
            self.assertFalse(result["ok"])
            self.assertNotIn("content", result)
            self.assertNotIn("secret", json.dumps(result))
            self.assertEqual(await self.fetch(service, session), result)
            self.assertEqual(session.fetch_calls, 1)
            self.assertEqual(len(api.requests), 1)

    async def test_auth_rate_limit_and_server_errors_do_not_leak_or_retry(self):
        for status in (400, 401, 402, 403, 429, 500, 504):
            service, session, api = self.setup_service()
            api.statuses["search"] = status
            api.override["search"] = {"error": {"message": "secret-and-private-url"}}
            result = await self.search(service, session)
            self.assertFalse(result["ok"])
            self.assertIn(str(status), result["error"])
            self.assertNotIn("secret", result["error"])
            self.assertEqual(result, await self.search(service, session))
            self.assertEqual(len(api.requests), 1)

    async def test_rejects_unsafe_and_out_of_scope_urls_without_api_request(self):
        service, session, api = self.setup_service()
        unsafe = ["http://127.0.0.1/", "http://169.254.169.254/", "http://10.0.0.1/",
                  "http://localhost/", "http://foo.local/", "file:///etc/passwd",
                  "https://user:password@example.com/", "https://example.com:8443/", "http://2130706433/"]
        for url in unsafe:
            session.allowed_urls.add(url)  # 即使误授权，仍有语法和字面地址校验。
            self.assertFalse((await self.fetch(service, session, url))["ok"])
        self.assertFalse((await self.fetch(service, session, "https://other.example.com/"))["ok"])
        self.assertEqual(session.fetch_calls, 0)
        self.assertEqual(api.requests, [])

    async def test_result_validation_dedupe_caps_and_unsafe_links_filtered(self):
        service, session, api = self.setup_service(web_search_max_results=2)
        api.search_rows = [None, {"title": "bad", "url": "http://127.0.0.1/"},
                           {"title": "missing url"}, rows()[0], rows()[0], rows()[1], rows()[2]]
        api.search_rows[-1]["content"] = {"unexpected": "object"}
        result = await self.search(service, session)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(session.allowed_urls, {URLS[0], URLS[1]})
        self.assertNotIn(URLS[2], session.pages)

    async def test_invalid_json_shapes_and_empty_search(self):
        for value in (b"not-json-secret", [], {}, {"results": {}}):
            service, session, api = self.setup_service()
            api.override["search"] = value
            result = await self.search(service, session)
            self.assertFalse(result["ok"])
            self.assertNotIn("secret", json.dumps(result))
        service, session, api = self.setup_service()
        api.search_rows = []
        result = await self.search(service, session)
        self.assertTrue(result["ok"])
        self.assertEqual(result["results"], [])
        self.assertEqual(session.fetch_calls, 0)

    async def test_response_byte_limit_before_and_during_stream(self):
        for advertised in (False, True):
            service, session, _ = self.setup_service(web_max_response_bytes=65536)
            await service.client.aclose()
            async def oversized(request):
                headers = {"content-type": "application/json"}
                if advertised:
                    headers["content-length"] = "70000"
                return httpx.Response(200, headers=headers, stream=httpx.ByteStream(b"x" * 70000))
            service.client = httpx.AsyncClient(transport=httpx.MockTransport(oversized))
            result = await self.search(service, session)
            self.assertFalse(result["ok"])
            self.assertIn("大小限制", result["error"])

    async def test_api_redirect_never_forwards_authorization(self):
        service, session, api = self.setup_service()
        await service.client.aclose()
        async def redirect(request):
            api.requests.append(request)
            return httpx.Response(302, headers={"location": "https://evil.example.com/steal"})
        service.client = httpx.AsyncClient(transport=httpx.MockTransport(redirect), follow_redirects=False)
        result = await self.search(service, session)
        self.assertFalse(result["ok"])
        self.assertEqual(len(api.requests), 1)
        self.assertEqual(api.requests[0].url.host, "api.deepseek.com")

    async def test_disabled_missing_key_and_query_validation_do_not_request(self):
        service, session, api = self.setup_service(web_search_enabled=False)
        self.assertFalse((await self.search(service, session))["ok"])
        service.settings = service.settings.model_copy(update={"web_search_enabled": True, "deepseek_api_key": ""})
        self.assertFalse((await self.search(service, session))["ok"])
        for query in (" ", "字" * 129):
            self.assertFalse((await self.search(service, session, query))["ok"])
        self.assertEqual(api.requests, [])

    async def test_concurrent_duplicates_share_cache_and_cancellation_propagates(self):
        service, session, api = self.setup_service()
        left, right = await asyncio.gather(self.search(service, session), self.search(service, session))
        self.assertEqual(left, right)
        self.assertEqual(len(api.requests), 1)
        await service.client.aclose()
        started = asyncio.Event()
        async def blocked(request):
            started.set()
            await asyncio.Event().wait()
        service.client = httpx.AsyncClient(transport=httpx.MockTransport(blocked))
        task = asyncio.create_task(self.search(service, WebSession()))
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_timeout_includes_queue_wait(self):
        service, session, _ = self.setup_service()
        service.settings = service.settings.model_copy(update={"web_timeout_seconds": 0.02})
        await service._slots.acquire()
        await service._slots.acquire()
        result = await self.search(service, session)
        self.assertFalse(result["ok"])
        self.assertIn("超时", result["error"])
        service._slots.release()
        service._slots.release()

    async def test_per_endpoint_pacing(self):
        service, _, _ = self.setup_service()
        service._request_interval = 0.02
        times = []
        async def paced(name):
            await service._pace(name)
            times.append((name, asyncio.get_running_loop().time()))
        await asyncio.gather(paced("search"), paced("search"), paced("fetch"))
        search_times = [t for name, t in times if name == "search"]
        self.assertGreaterEqual(search_times[1] - search_times[0], 0.015)

    async def test_agent_observes_body_after_single_search_tool_call(self):
        service, session, api = self.setup_service()
        registry = ToolRegistry()
        for definition in WEB_TOOLS:
            name = definition["function"]["name"]
            async def handler(arguments, tool_name=name):
                return await service.execute_tool(session, tool_name, arguments)
            registry.register(definition, handler)
        async def finish(arguments):
            return {"ok": True}
        registry.register(FINISH_TOOL, finish, terminal=True)
        class Model:
            requests = []
            async def request_message(self, messages, temperature, tools):
                self.requests.append(copy.deepcopy(messages))
                if len(self.requests) == 1:
                    name, args = "web_search", {"query": "测试"}
                else:
                    observation = json.loads(messages[-1]["content"])
                    assert "正文证据" in observation["results"][0]["page"]["content"]
                    name, args = "finish", {}
                return {"tool_calls": [{"id": str(len(self.requests)), "function": {
                    "name": name, "arguments": json.dumps(args)}}]}
        result = await ReactAgent(service.settings, Model()).run("测试", registry, system=WEB_INSTRUCTIONS)
        self.assertIsNone(result.error)
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(api.requests), 1)
        self.assertEqual(session.fetch_calls, 0)


if __name__ == "__main__":
    unittest.main()
