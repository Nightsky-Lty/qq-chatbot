"""临时库与合成PNG验证缓存、去重、概要及重启；--live调用1次真实视觉API。"""

import argparse
import asyncio
import json
import struct
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.database import Database
from app.image_cache import ImageCache
from app.llm import DeepSeekClient


def sample_png():
    # 纯测试夹具：蓝底白色圆形，不涉及真实聊天图片。
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    raw = bytearray()
    for y in range(64):
        raw.append(0)
        for x in range(64):
            raw.extend((255, 255, 255) if (x-32)**2 + (y-32)**2 < 16**2 else (30, 100, 220))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


async def main(live):
    with tempfile.TemporaryDirectory(prefix="qq-image-cache-check-") as folder:
        cfg = Settings().model_copy(update={"database_path": Path(folder)/"test.db",
                                           "image_cache_directory": Path(folder)/"images", "image_cache_enabled": True})
        db = Database(cfg.database_path)
        real = DeepSeekClient(cfg, purpose="image")
        class Fake:
            settings = SimpleNamespace(deepseek_api_key="test")
            async def summarize_image(self, mime, encoded):
                return {"summary": "蓝底白色圆形", "text": "", "uncertain": False}
        model = real if live else Fake()
        cache = ImageCache(cfg, db, model)
        async def download(ref):
            return "image/png", sample_png()
        cache.downloader.load = download
        try:
            refs = []
            for group in (7, 8):
                source = db.log_message(group, 11, "合成测试", "[合成图片]", message_id=str(group))
                refs.append(cache.record(group, source, 11, str(group), [{"url": "https://gchat.qpic.cn/synthetic"}])[0])
                await cache.load(refs[-1])
            await cache.summarize_once()
            status = cache.rows(7, 999999)[0]
            summary_ok = status["summary_state"] == "ready"
            dedupe_ok = len(list(cfg.image_cache_directory.glob("*.img"))) == 1
            await cache.close()
            cache = ImageCache(cfg, db, model)
            restart_ok = (await cache.load(refs[0]))[1] == sample_png()
            isolated = not cache.authorized({**refs[0], "namespace": "8"})
            print(json.dumps({"live": live, "summary_ok": summary_ok, "dedupe_ok": dedupe_ok,
                              "restart_ok": restart_ok, "scope_isolation_ok": isolated}, ensure_ascii=False))
            if not all((summary_ok, dedupe_ok, restart_ok, isolated)):
                raise SystemExit(1)
        finally:
            await cache.close()
            await real.close()
            db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    asyncio.run(main(parser.parse_args().live))
