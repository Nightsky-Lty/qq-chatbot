"""真实DeepSeek Search/Fetch测试：复用Key，按成功API请求收费。

不调用聊天LLM、不写数据库、不发送QQ。加--verify-fetch额外测试1次Fetch。
在项目根目录运行：PYTHONPATH=. python3 scripts/verify_web_search.py --verify-fetch
"""

import argparse
import asyncio
import json

from app.config import Settings
from app.web_search import WebSearchService, WebSession


async def main(query, verify_fetch=False):
    service = WebSearchService(Settings())
    try:
        session = WebSession()
        result = await service.execute_tool(session, "web_search", {"query": query})
        summary = {
            "search_ok": result.get("ok") is True,
            "pages_read": result.get("pages_read", 0),
            "search_fetch_calls": session.fetch_calls,
            "results": [{"page_status": row["page_status"],
                         "content_chars": len(row.get("page", {}).get("content", ""))}
                        for row in result.get("results", [])],
        }
        if not summary["search_ok"]:
            summary["error"] = result.get("error", "")
        for row in result.get("results", []):
            if row["page_status"] == "read":
                cached = await service.execute_tool(session, "web_fetch", {"url": row["url"]})
                summary["cache_ok"] = cached.get("ok") is True and session.fetch_calls == 0
                break
        if verify_fetch:
            url = "https://api-docs.deepseek.com/zh-cn/guides/search/"
            fetch_session = WebSession(allowed_urls={url})
            page = await service.execute_tool(fetch_session, "web_fetch", {"url": url})
            summary["fetch_ok"] = page.get("ok") is True
            summary["fetch_content_chars"] = len(page.get("content", ""))
            if not summary["fetch_ok"]:
                summary["fetch_error"] = page.get("error", "")
        print(json.dumps(summary, ensure_ascii=False))
    finally:
        await service.close()
    if (not summary["search_ok"] or summary["pages_read"] < 1
            or not summary.get("cache_ok") or (verify_fetch and not summary["fetch_ok"])):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="Python asyncio 官方文档")
    parser.add_argument("--verify-fetch", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.query, args.verify_fetch))
