"""只读公网HTTP抓取：逐跳校验、DNS固定、TLS验证和响应体上限。"""

import asyncio
import ipaddress
import re
import socket
import zlib
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from app.config import Settings


class WebUnavailable(ValueError):
    """可安全交给模型的固定失败原因，不包含底层响应、凭证或内部地址。"""


def public_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if not address.is_global or address.is_multicast:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        # 不允许IPv4转换/隧道地址绕过私网IPv4检查。
        if address.ipv4_mapped or address.sixtofour or address.teredo:
            return False
        if address in ipaddress.ip_network("64:ff9b::/96"):
            return False
    return True


def public_url(value: str) -> str:
    """仅作语法/字面地址校验；真正发请求前还必须检查所有DNS解析结果。"""
    if not isinstance(value, str) or not value or len(value) > 4096 or re.search(r"[\x00-\x20\x7f\\]", value):
        raise WebUnavailable("网页地址格式无效。")
    try:
        parts = urlsplit(value)
        parsed = httpx.URL(value)
        host = parsed.raw_host.decode("ascii").rstrip(".")
        port = parts.port
    except (ValueError, UnicodeError, httpx.InvalidURL) as exc:
        raise WebUnavailable("网页地址格式无效。") from exc
    if (parts.scheme not in {"http", "https"} or not host or "%" in host
            or parts.username is not None or parts.password is not None
            or port not in {None, 80 if parts.scheme == "http" else 443}):
        raise WebUnavailable("只允许无凭证、使用标准端口的HTTP/HTTPS公开网页。")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        try:
            socket.inet_aton(host)
        except OSError:
            pass
        else:
            raise WebUnavailable("不支持缩写、整数或其他非标准IP地址。")
        if ("." not in host or "%" in host or host.endswith(
                (".localhost", ".local", ".internal", ".lan", ".home", ".invalid", ".test"))):
            raise WebUnavailable("禁止读取本机、内网或特殊用途地址。")
    else:
        if not public_address(host):
            raise WebUnavailable("禁止读取本机、内网或特殊用途地址。")
    return str(parsed.copy_with(host=host, fragment=None))


@dataclass(frozen=True, slots=True)
class WebPage:
    url: str
    text: str
    content_type: str


class SafeWebClient:
    def __init__(self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None):
        self.timeout = settings.web_timeout_seconds
        self.max_bytes = settings.web_max_response_bytes
        self.transport = transport
        self._slots = asyncio.Semaphore(2)

    async def _resolve(self, host: str, port: int) -> list[str]:
        try:
            answers = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise WebUnavailable("网页域名解析失败。") from exc
        addresses = list(dict.fromkeys(answer[4][0] for answer in answers))
        if not addresses or any(not public_address(address) for address in addresses):
            raise WebUnavailable("网页域名解析到非公网地址，已拒绝访问。")
        return sorted(addresses, key=lambda address: ":" in address)

    async def get(self, url: str) -> WebPage:
        try:
            # 超时覆盖并发等待、DNS、跳转和完整下载，不只约束单次socket读取。
            async with asyncio.timeout(self.timeout):
                async with self._slots:
                    return await self._get(url)
        except (httpx.HTTPError, httpx.InvalidURL, OSError, TimeoutError) as exc:
            raise WebUnavailable("网页请求失败或超时，不能视为已读取。") from exc

    async def _get(self, url: str) -> WebPage:
        current = public_url(url)
        for hop in range(4):
            original = httpx.URL(current)
            host = original.raw_host.decode("ascii")
            addresses = await self._resolve(host, original.port or (443 if original.scheme == "https" else 80))
            # URL实际连接已经校验过的IP，避免连接阶段再次DNS解析（DNS rebinding）。
            # SNI仍用原域名，HTTPS证书验证保持开启；Host也保留原域名。
            pinned = original.copy_with(host=addresses[0])
            headers = {
                "Host": f"[{host}]" if ":" in host else host,
                "User-Agent": "Mozilla/5.0 (compatible; QQChatbot/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
                "Accept-Encoding": "identity",
            }
            # 每跳独立客户端，不继承环境代理、Cookie、认证或其他模型API配置。
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False, trust_env=False,
                                         transport=self.transport) as client:
                async with client.stream("GET", pinned, headers=headers, extensions={"sni_hostname": host}) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location", "")
                        if not location or hop == 3:
                            raise WebUnavailable("网页跳转次数过多或跳转地址缺失。")
                        current = public_url(urljoin(current, location))
                        continue
                    if response.status_code != 200:
                        raise WebUnavailable(f"网页返回HTTP {response.status_code}，可能受限、需要登录或暂时不可用。")
                    encoding_header = response.headers.get("content-encoding", "identity").strip().lower()
                    if encoding_header not in {"identity", "gzip", "deflate"}:
                        raise WebUnavailable("网页使用不支持的压缩格式。")
                    decoder = (zlib.decompressobj(16 + zlib.MAX_WBITS if encoding_header == "gzip" else zlib.MAX_WBITS)
                               if encoding_header != "identity" else None)
                    mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if mime not in {"text/html", "application/xhtml+xml", "text/plain"}:
                        raise WebUnavailable("只读取HTML或纯文本网页，不下载图片、PDF或其他文件。")
                    size = response.headers.get("content-length", "")
                    if size.isdigit() and int(size) > self.max_bytes:
                        raise WebUnavailable("网页超过下载大小限制。")
                    data = bytearray()
                    wire_size = 0
                    async for chunk in response.aiter_raw(chunk_size=16384):
                        wire_size += len(chunk)
                        if wire_size > self.max_bytes:
                            raise WebUnavailable("网页超过下载大小限制。")
                        if decoder is not None:
                            try:
                                # 同时限制压缩前/后的字节，避免小响应解压成巨大正文。
                                chunk = decoder.decompress(chunk, self.max_bytes - len(data) + 1)
                            except zlib.error as exc:
                                raise WebUnavailable("网页压缩响应损坏。") from exc
                        if len(data) + len(chunk) > self.max_bytes:
                            raise WebUnavailable("网页超过解压后大小限制。")
                        data.extend(chunk)
                    if decoder is not None and (not decoder.eof or decoder.unused_data):
                        raise WebUnavailable("网页压缩响应不完整或含额外压缩数据。")
                    # 尊重响应字符集；旧中文网页未声明时补查HTML编码，再退回UTF-8。
                    encoding = response.charset_encoding
                    if not encoding:
                        match = re.search(rb"charset\s*=\s*[\"']?([a-zA-Z0-9_-]+)", bytes(data[:4096]), re.I)
                        encoding = match[1].decode("ascii") if match else "utf-8"
                    if encoding.lower() in {"gb2312", "gbk"}:
                        encoding = "gb18030"
                    try:
                        content = data.decode(encoding, errors="replace")
                    except LookupError:
                        content = data.decode("utf-8", errors="replace")
                    return WebPage(current, content, mime)
        raise WebUnavailable("网页跳转次数过多。")
