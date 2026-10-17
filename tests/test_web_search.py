import asyncio
import copy
import json
import unittest
from unittest.mock import patch

import httpx

from app.agent import ReactAgent
from app.config import Settings
from app.safe_web import SafeWebClient, WebPage, WebUnavailable
from app.tools import FINISH_TOOL, ToolRegistry
from app.web_search import BING_SEARCH_URL, WEB_TOOLS, WEB_INSTRUCTIONS, WebSearchService, WebSession


URLS = [f"https://example.com/article-{i}" for i in range(5)]


def search_html(urls=URLS):
    return "".join(f'<li class="b_algo"><h2><a href="{url}">标题{i}</a></h2>'
                   f'<p>搜索摘要{i}</p></li>' for i, url in enumerate(urls))


class FakeWeb:
    def __init__(self, urls=URLS, failures=()):
        self.urls, self.failures, self.calls = urls, failures, []
        self.inflight = self.peak = 0

    async def get(self, url):
        self.calls.append(url)
        if url.startswith(BING_SEARCH_URL + "?"):
            return WebPage(url, search_html(self.urls), "text/html")
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            # 让两个读取协程实际重叠，而非仅检查gather调用次数。
            await asyncio.sleep(0)
            if url in self.failures:
                raise WebUnavailable("网页返回HTTP 403，可能受限。")
            return WebPage(url, '<head><title>正文标题</title></head><nav>导航噪音</nav>'
                           '<article><p>正文证据，不同于搜索摘要。</p></article>', "text/html")
        finally:
            self.inflight -= 1


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    def setup_service(self, **settings):
        service = WebSearchService(Settings(_env_file=None, **settings))
        service.http = FakeWeb()
        return service, WebSession()

    async def test_search_automatically_reads_two_pages_in_parallel(self):
        service, session = self.setup_service()
        result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["pages_read"], 2)
        self.assertEqual([r["page_status"] for r in result["results"]],
                         ["read", "read", "not_read", "not_read", "not_read"])
        self.assertEqual(service.http.peak, 2)
        self.assertEqual(session.fetch_calls, 2)
        self.assertIn("正文证据", result["results"][0]["page"]["content"])
        self.assertNotIn("导航噪音", result["results"][0]["page"]["content"])
        self.assertIn("搜索摘要", result["results"][0]["snippet"])

    async def test_search_and_manual_reads_reuse_success_and_failure_cache(self):
        service, session = self.setup_service()
        service.http.failures = {URLS[1]}
        result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["results"][1]["page_status"], "failed")
        self.assertNotIn("content", result["results"][1]["page"])
        self.assertEqual(result, await service.execute_tool(session, "web_search", {"query": "测试"}))
        for item in result["results"][:2]:
            self.assertEqual(item["page"], await service.execute_tool(session, "web_fetch", {"url": item["url"]}))
        self.assertEqual(len(service.http.calls), 3)
        # 换关键词搜到相同页面，同样复用读取缓存。
        again = await service.execute_tool(session, "web_search", {"query": "其他关键词"})
        self.assertEqual(again["pages_read"], 1)
        self.assertEqual(session.fetch_calls, 2)
        self.assertEqual(len(service.http.calls), 4)

    async def test_all_page_failures_preserve_search_and_disclose_snippets_only(self):
        service, session = self.setup_service()
        service.http.failures = set(URLS)
        result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["pages_read"], 0)
        self.assertIn("只有搜索摘要", result["note"])
        self.assertEqual(session.fetch_calls, 2)
        self.assertTrue(all(r["page_status"] == "failed" for r in result["results"][:2]))
        # 初次自动读失败不耗尽默认总额度，允许手动补读其他来源。
        service.http.failures = ()
        self.assertTrue((await service.execute_tool(session, "web_fetch", {"url": URLS[2]}))["ok"])

    async def test_automatic_and_manual_reads_share_budget_without_overrun(self):
        service, session = self.setup_service(web_fetch_max_calls=3)
        await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertTrue((await service.execute_tool(session, "web_fetch", {"url": URLS[2]}))["ok"])
        self.assertFalse((await service.execute_tool(session, "web_fetch", {"url": URLS[3]}))["ok"])
        service.http.urls = [URLS[3], URLS[4]]
        result = await service.execute_tool(session, "web_search", {"query": "新问题"})
        self.assertEqual(result["pages_read"], 0)
        self.assertTrue(all(r["page_status"] == "not_read" for r in result["results"]))
        self.assertTrue(all("额度" in r["page_note"] for r in result["results"]))
        self.assertEqual(session.fetch_calls, 3)
        self.assertNotIn(URLS[3], service.http.calls)

    async def test_remaining_single_slot_and_disable_switch(self):
        service, session = self.setup_service(web_fetch_max_calls=1)
        result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(session.fetch_calls, 1)
        service, session = self.setup_service(web_search_auto_fetch_pages=0)
        result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertEqual(result["pages_read"], 0)
        self.assertEqual(session.fetch_calls, 0)
        self.assertEqual(len(service.http.calls), 1)
        self.assertTrue((await service.execute_tool(session, "web_fetch", {"url": URLS[0]}))["ok"])

    async def test_result_scope_and_response_length_preserved(self):
        service, session = self.setup_service(web_fetch_max_chars=1000)
        original = service.http.get
        async def long_page(url):
            if url.startswith(BING_SEARCH_URL + "?"):
                return await original(url)
            return WebPage(url, "正文" * 1000, "text/plain")
        service.http.get = long_page
        result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertEqual(len(result["results"][0]["page"]["content"]), 1000)
        self.assertTrue(result["results"][0]["page"]["truncated"])
        denied = await service.execute_tool(session, "web_fetch", {"url": "https://other.example.com/"})
        self.assertFalse(denied["ok"])
        self.assertEqual(session.fetch_calls, 2)

    async def test_search_failure_does_not_fetch_and_is_cached(self):
        service, session = self.setup_service()
        async def captcha(url):
            service.http.calls.append(url)
            return WebPage(url, '<div id="b_captcha">验证码</div>', "text/html")
        service.http.get = captcha
        result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertFalse(result["ok"])
        self.assertEqual(result, await service.execute_tool(session, "web_search", {"query": "测试"}))
        self.assertEqual(session.fetch_calls, 0)
        self.assertEqual(len(service.http.calls), 1)

    async def test_single_result_empty_body_and_concurrent_duplicate_search(self):
        service, session = self.setup_service()
        service.http.urls = [URLS[0]]
        left, right = await asyncio.gather(*(
            service.execute_tool(session, "web_search", {"query": "同一问题"}) for _ in range(2)))
        self.assertEqual(left, right)
        self.assertEqual(left["pages_read"], 1)
        self.assertEqual(session.search_calls, 1)
        self.assertEqual(session.fetch_calls, 1)
        service, session = self.setup_service()
        original = service.http.get
        async def empty_page(url):
            if url.startswith(BING_SEARCH_URL + "?"):
                return await original(url)
            return WebPage(url, '<script>js_only()</script>', "text/html")
        service.http.get = empty_page
        result = await service.execute_tool(session, "web_search", {"query": "空页面"})
        self.assertEqual(result["pages_read"], 0)
        self.assertTrue(all(r["page_status"] == "failed" for r in result["results"][:2]))

    async def test_auto_read_keeps_real_safe_client_dns_and_redirect_guards(self):
        service, session = self.setup_service()
        requests = []
        async def transport(request):
            requests.append(request)
            if request.url.path == "/search":
                return httpx.Response(200, stream=httpx.ByteStream(search_html([
                    "https://blocked.example.com/page", "https://redirect.example.com/page"]).encode()),
                    headers={"content-type": "text/html"})
            return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})
        async def dns(host, port, **kwargs):
            address = "127.0.0.1" if host == "blocked.example.com" else "93.184.216.34"
            return [(2, 1, 6, "", (address, port))]
        service.http = SafeWebClient(service.settings, transport=httpx.MockTransport(transport))
        # 只替换真实事件循环的DNS方法，网络始终由MockTransport截获。
        with patch.object(asyncio.get_running_loop(), "getaddrinfo", side_effect=dns):
            result = await service.execute_tool(session, "web_search", {"query": "测试"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["pages_read"], 0)
        self.assertTrue(all(r["page_status"] == "failed" for r in result["results"]))
        self.assertEqual(len(requests), 2)  # 搜索页、重定向入口；私网从未请求。

    async def test_agent_observes_body_after_single_search_tool_call(self):
        service, session = self.setup_service()
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
        model = Model()
        result = await ReactAgent(service.settings, model).run("测试", registry, system=WEB_INSTRUCTIONS)
        self.assertIsNone(result.error)
        self.assertEqual(result.steps, 2)
        self.assertEqual(session.fetch_calls, 2)


if __name__ == "__main__":
    unittest.main()
