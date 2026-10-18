"""验证宿主机统计挂载和真实快照，不发QQ、不写数据库、不调用LLM。"""

import asyncio

from app.config import Settings
from app.server_status import ServerStatusService


async def main():
    service = ServerStatusService(Settings())
    text = await service.get_text()
    print(text)
    if "暂不可用" in text or "超时" in text or not text.startswith("服务器资源状态"):
        raise SystemExit(1)
    cached = await service.get_text()
    if not cached.endswith("（10秒内缓存）"):
        raise SystemExit(1)
    print("snapshot_cache_verified")


if __name__ == "__main__":
    asyncio.run(main())
