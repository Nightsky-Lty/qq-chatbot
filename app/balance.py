"""直接查询DeepSeek账户余额；固定格式输出，不经过聊天模型或聊天数据库。"""

import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.config import Settings


class BalanceUnavailable(RuntimeError):
    """只包含可安全向用户展示的错误，不包含Key、URL或接口原始响应。"""


class BalanceService:
    CACHE_SECONDS = 30
    ERROR_CACHE_SECONDS = 5

    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings, self.client = settings, client
        self._lock = asyncio.Lock()
        self._cached: tuple[float, str] | None = None
        self._error: tuple[float, str] | None = None
        self._credentials = (settings.deepseek_base_url, settings.deepseek_api_key)

    async def get_text(self) -> str:
        """并发查询共享30秒缓存；短暂缓存失败，避免同时重试挤占连接。"""
        async with self._lock:
            credentials = (self.settings.deepseek_base_url, self.settings.deepseek_api_key)
            if credentials != self._credentials:
                self._cached = self._error = None
                self._credentials = credentials
            now = time.monotonic()
            if self._cached is not None and now - self._cached[0] < self.CACHE_SECONDS:
                return self._cached[1] + "\n（30秒内缓存）"
            if self._error is not None and now - self._error[0] < self.ERROR_CACHE_SECONDS:
                raise BalanceUnavailable(self._error[1])
            try:
                text = await self._fetch()
            except BalanceUnavailable as exc:
                self._error = (time.monotonic(), str(exc))
                raise
            self._cached = (time.monotonic(), text)
            self._error = None
            return text

    async def _fetch(self) -> str:
        if not self.settings.deepseek_api_key:
            raise BalanceUnavailable("尚未配置 DeepSeek API Key，请联系部署者。")
        try:
            async with asyncio.timeout(5):
                response = await self.client.get(
                    f"{self.settings.deepseek_base_url.rstrip('/')}/user/balance",
                    headers={"Authorization": f"Bearer {self.settings.deepseek_api_key}"},
                    timeout=5, follow_redirects=False,
                )
                if response.status_code in {401, 403}:
                    raise BalanceUnavailable("余额查询鉴权失败，请检查部署的 API Key。")
                if response.status_code in {404, 405, 501}:
                    raise BalanceUnavailable("当前配置的 API 服务不支持余额查询。")
                if response.status_code == 429:
                    raise BalanceUnavailable("余额查询过于频繁，请稍后重试。")
                if response.status_code != 200:
                    raise BalanceUnavailable("余额服务暂时不可用，请稍后重试。")
                return self._format(response.json())
        except (TimeoutError, httpx.TimeoutException):
            raise BalanceUnavailable("余额查询超时，请稍后重试。") from None
        except (httpx.HTTPError, httpx.InvalidURL):
            raise BalanceUnavailable("余额查询网络异常，请稍后重试。") from None
        except (ValueError, KeyError, TypeError):
            raise BalanceUnavailable("余额接口返回格式异常，请联系部署者检查。") from None

    @staticmethod
    def _amount(value: Any) -> str:
        # 金额保留接口精度，不经float转换；限制输入格式防止远端文本进入回复。
        if not isinstance(value, str) or re.fullmatch(r"-?\d{1,18}(?:\.\d{1,12})?", value) is None:
            raise ValueError("invalid amount")
        whole, _, fraction = value.partition(".")
        return whole + "." + fraction.rstrip("0").ljust(2, "0")

    @classmethod
    def _format(cls, data: Any) -> str:
        if not isinstance(data, dict) or type(data.get("is_available")) is not bool:
            raise ValueError("invalid balance")
        balances = data.get("balance_infos")
        if not isinstance(balances, list) or len(balances) > 2:
            raise ValueError("invalid balances")
        lines = ["DeepSeek 账户余额"]
        currencies = set()
        for info in balances:
            if not isinstance(info, dict):
                raise ValueError("invalid balance entry")
            currency = info.get("currency")
            if currency not in {"CNY", "USD"} or currency in currencies:
                raise ValueError("invalid currency")
            currencies.add(currency)
            symbol = "¥" if currency == "CNY" else "$"
            lines.extend([
                f"币种：{currency}",
                f"可用余额：{symbol}{cls._amount(info['total_balance'])}",
                f"充值余额：{symbol}{cls._amount(info['topped_up_balance'])}",
                f"赠金余额：{symbol}{cls._amount(info['granted_balance'])}",
            ])
        if not balances:
            lines.append("暂无余额明细")
        lines.append("账户状态：" + ("有余额可供调用" if data["is_available"] else "无余额可供调用"))
        queried_at = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"查询时间：{queried_at}（北京时间）")
        return "\n".join(lines)
