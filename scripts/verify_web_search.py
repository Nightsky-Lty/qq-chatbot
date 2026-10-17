"""真实网络冒烟测试：只检索公开关键词，不调用LLM、不写数据库、不发送QQ。

在项目根目录运行：PYTHONPATH=. python3 scripts/verify_web_search.py
"""

import argparse
import asyncio
import json

from app.config import Settings
from app.web_search import WebSearchService, WebSession


async def main(query):
    service = WebSearchService(Settings(_env_file=None))
    session = WebSession()
    result = await service.execute_tool(session, "web_search", {"query": query})
    summary = {
        "search_ok": result.get("ok") is True,
        "pages_read": result.get("pages_read", 0),
        "fetch_calls": session.fetch_calls,
        "results": [{"page_status": row["page_status"],
                     "content_chars": len(row.get("page", {}).get("content", "")),
                     "error": row.get("page", {}).get("error", "")}
                    for row in result.get("results", [])],
    }
    if not summary["search_ok"]:
        summary["error"] = result.get("error", "")
    print(json.dumps(summary, ensure_ascii=False))
    if not summary["search_ok"] or summary["pages_read"] < 1:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="Python asyncio 官方文档")
    asyncio.run(main(parser.parse_args().query))
