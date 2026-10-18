"""固定格式的Linux宿主机资源快照，不执行命令、不联网、不读取进程信息。"""

import asyncio
import math
import re
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import Settings


def cpu_counters(text: str) -> tuple[tuple[int, ...], int]:
    lines = text.splitlines()
    first = next(line.split()[1:] for line in lines if line.startswith("cpu "))
    values = tuple(int(value) for value in first[:8])
    if len(values) < 4 or any(value < 0 for value in values):
        raise ValueError("invalid cpu counters")
    # guest/guest_nice已包含在user/nice里，不能重复加入总时间。
    return values, sum(bool(re.match(r"^cpu\d+\s", line)) for line in lines)


def cpu_percent(before: tuple[int, ...], after: tuple[int, ...]) -> float | None:
    if len(before) != len(after) or any(right < left for left, right in zip(before, after)):
        return None
    total = sum(after) - sum(before)
    idle = after[3] - before[3]
    if len(after) > 4:
        idle += after[4] - before[4]  # iowait不计为正在执行CPU任务。
    if total <= 0:
        return None
    return min(100.0, max(0.0, (total - idle) * 100 / total))


def memory_values(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+)\s+kB\s*", line)
        if match:
            values[match[1]] = int(match[2]) * 1024
    return values


def gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def duration(seconds: float) -> str:
    minutes = int(seconds) // 60
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    return f"{days}天{hours}小时{minutes}分"


class ServerStatusService:
    CACHE_SECONDS = 10
    SAMPLE_SECONDS = 0.3

    def __init__(self, settings: Settings):
        self.proc_path = settings.server_status_proc_path
        self.disk_path = settings.database_path.parent
        self.started_at = time.monotonic()
        self._lock = asyncio.Lock()
        self._cached: tuple[float, str] | None = None

    def _read(self, name: str) -> str | None:
        # 仅调用方写死的四个统计文件，不接收用户路径，不读取整个/proc。
        try:
            with (self.proc_path / name).open(encoding="ascii") as source:
                text = source.read(1024 * 1024 + 1)
            return text if len(text) <= 1024 * 1024 else None
        except (OSError, UnicodeError):
            return None

    def _cpu(self) -> tuple[tuple[int, ...], int] | None:
        try:
            return cpu_counters(self._read("stat") or "")
        except (ValueError, StopIteration):
            return None

    def _collect(self) -> dict:
        result = {"cpu": self._cpu(), "memory": memory_values(self._read("meminfo") or "")}
        try:
            load = tuple(float(value) for value in (self._read("loadavg") or "").split()[:3])
            if len(load) == 3 and all(math.isfinite(value) and value >= 0 for value in load):
                result["load"] = load
        except ValueError:
            pass
        try:
            uptime = float((self._read("uptime") or "").split()[0])
            if math.isfinite(uptime) and uptime >= 0:
                result["uptime"] = uptime
        except (ValueError, IndexError):
            pass
        try:
            result["disk"] = shutil.disk_usage(self.disk_path)
        except OSError:
            pass
        return result

    async def get_text(self) -> str:
        async with self._lock:
            if self._cached and time.monotonic() - self._cached[0] < self.CACHE_SECONDS:
                return self._cached[1] + "\n（10秒内缓存）"
            try:
                async with asyncio.timeout(3):
                    before = await asyncio.to_thread(self._cpu)
                    if before is not None:
                        await asyncio.sleep(self.SAMPLE_SECONDS)
                    metrics = await asyncio.to_thread(self._collect)
                    text = self._format(before, metrics)
            except TimeoutError:
                return "服务器状态采集超时，请稍后重试。"
            self._cached = (time.monotonic(), text)
            return text

    def _format(self, before: tuple[tuple[int, ...], int] | None, metrics: dict) -> str:
        lines = ["服务器资源状态"]
        after = metrics.get("cpu")
        percent = cpu_percent(before[0], after[0]) if before and after else None
        cores = f"，{after[1]}逻辑核" if after and after[1] else ""
        lines.append(f"宿主机CPU：{percent:.1f}%{cores}（约0.3秒采样）" if percent is not None
                     else "宿主机CPU：暂不可用")
        memory = metrics["memory"]
        total, available = memory.get("MemTotal", 0), memory.get("MemAvailable")
        if total > 0 and available is not None and 0 <= available <= total:
            used = total - available
            lines.append(f"宿主机内存：{gib(used)} / {gib(total)}（{used * 100 / total:.1f}%）")
        else:
            lines.append("宿主机内存：暂不可用")
        swap, free = memory.get("SwapTotal"), memory.get("SwapFree")
        if swap == 0:
            lines.append("宿主机Swap：未启用")
        elif swap and free is not None and 0 <= free <= swap:
            lines.append(f"宿主机Swap：{gib(swap - free)} / {gib(swap)}")
        else:
            lines.append("宿主机Swap：暂不可用")
        disk = metrics.get("disk")
        if disk and disk.total > 0:
            lines.append(f"磁盘（数据卷所在文件系统）：{gib(disk.used)} / {gib(disk.total)}（{disk.used * 100 / disk.total:.1f}%）")
            lines.append(f"磁盘可用：{gib(disk.free)}")
        else:
            lines.append("磁盘（数据卷所在文件系统）：暂不可用")
        load = metrics.get("load")
        lines.append("宿主机负载（1/5/15分钟）：" + (" / ".join(f"{value:.2f}" for value in load) if load else "暂不可用"))
        lines.append("服务器运行：" + (duration(metrics["uptime"]) if "uptime" in metrics else "暂不可用"))
        lines.append("Bot进程运行：" + duration(time.monotonic() - self.started_at))
        if after is None or not memory:
            lines.append("部分宿主机统计缺失，请检查只读统计文件挂载。")
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"采集时间：{stamp}（北京时间）")
        return "\n".join(lines)
