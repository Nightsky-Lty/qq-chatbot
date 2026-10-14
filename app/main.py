"""FastAPI 服务入口：管理应用资源，通过反向 WebSocket 接入 OneBot 事件。"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.bot import BotService
from app.config import Settings, get_settings
from app.database import Database
from app.llm import DeepSeekClient
from app.reminders import ReminderService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# HTTP 请求日志会包含 QQ 图片链接中的临时签名，不输出完整 URL。
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class OneBotSendError(RuntimeError):
    """OneBot 拒绝发送、没有及时确认或连接断开；提醒会保留并重试。"""


class OneBotConnection:
    """封装同一 OneBot 连接上的群聊和私聊发送，共用锁及请求序号。"""

    def __init__(self, websocket: WebSocket):
        """保存连接，并建立发送锁和请求序号。"""
        self.websocket = websocket
        self.lock = asyncio.Lock()
        self.sequence = 0
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    async def send_group(self, group_id: int, message: list[dict[str, Any]]) -> None:
        """向 NapCat 发送群消息调用；这里只提交请求，不等待发送结果确认。"""
        await self._send_action("send_group_msg", {"group_id": group_id, "message": message})

    async def send_private(self, user_id: int, message: list[dict[str, Any]]) -> None:
        """向指定 QQ 发送私聊消息；使用私聊动作，不向群发送或附加 @。"""
        await self._send_action("send_private_msg", {"user_id": user_id, "message": message})

    async def send_private_confirmed(self, user_id: int, message: list[dict[str, Any]]) -> None:
        """提醒使用确认式发送，NapCat 返回成功后才允许标记提醒完成。"""
        await self._send_action("send_private_msg", {"user_id": user_id, "message": message}, confirmed=True)

    def handle_response(self, response: dict[str, Any]) -> None:
        """用 echo 唤醒等待中的发送；普通回复失败只记状态码，不打印消息内容。"""
        echo = response.get("echo")
        future = self.pending.get(echo) if isinstance(echo, str) else None
        if future is not None and not future.done():
            future.set_result(response)
        if response.get("status") == "failed":
            logger.warning("OneBot action failed: retcode=%s", response.get("retcode"))

    def close_pending(self) -> None:
        """断开连接时唤醒所有确认等待，避免旧连接的提醒任务挂起。"""
        for future in self.pending.values():
            if not future.done():
                future.set_exception(OneBotSendError("OneBot连接已断开"))
                # 发送阶段尚未进入 await 的 Future 也不会产生未消费异常。
                future.exception()

    async def _send_action(self, action: str, params: dict[str, Any], *, confirmed: bool = False) -> None:
        """串行提交 OneBot 动作，群聊和私聊共用序号，防止并发写入。"""
        # 同一连接上的发送操作串行执行，避免并发写入。
        echo = ""
        future = None
        try:
            async with self.lock:
                self.sequence += 1
                echo = f"send-{self.sequence}"
                if confirmed:
                    future = asyncio.get_running_loop().create_future()
                    self.pending[echo] = future
                await self.websocket.send_json({"action": action, "params": params, "echo": echo})
            if future is not None:
                try:
                    response = await asyncio.wait_for(future, timeout=15)
                except TimeoutError as exc:
                    raise OneBotSendError("NapCat没有及时确认发送") from exc
                if response.get("status") != "ok" or response.get("retcode") != 0:
                    raise OneBotSendError(f"NapCat发送失败，状态码{response.get('retcode')}")
        finally:
            self.pending.pop(echo, None)
            if future is not None and not future.done():
                future.cancel()


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建应用，可显式传入配置；未传入时使用缓存的环境配置。"""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """启动时初始化资源并清理过期消息，关闭时释放客户端和数据库。"""
        db = Database(cfg.database_path)
        purged = db.purge_old_messages(cfg.message_retention_days)
        if purged:
            logger.info("Purged %s expired records", purged)
        llm = DeepSeekClient(cfg)
        reminders = ReminderService(cfg, db)
        app.state.bot = BotService(cfg, db, llm, reminders)
        app.state.db = db
        app.state.llm = llm
        app.state.reminders = reminders
        reminders.start()
        app.state.bot.sticker_indexer.start()
        # yield 前是启动逻辑，yield 后是应用关闭时执行的清理逻辑。
        try:
            yield
        finally:
            # 先结束后台请求，再关闭它们使用的客户端和数据库。
            await app.state.bot.close()
            await reminders.close()
            await llm.close()
            db.close()

    app = FastAPI(title="QQ DeepSeek Bot", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """报告 HTTP 服务可响应，不代表 QQ 已连接或 DeepSeek 接口可用。"""
        return {"status": "ok"}

    @app.websocket("/onebot/v11/ws")
    async def onebot_websocket(websocket: WebSocket) -> None:
        """接收 NapCat 发起的反向连接，认证后将事件交给机器人业务层。"""
        # 优先使用 Authorization 请求头，兼容查询参数中的 access_token。
        authorization = websocket.headers.get("authorization", "")
        query_token = websocket.query_params.get("access_token", "")
        supplied_token = authorization.removeprefix("Bearer ").strip() or query_token
        # 仅在配置了 Token 时进行校验；空配置会允许连接。
        if cfg.onebot_access_token and supplied_token != cfg.onebot_access_token:
            await websocket.close(code=1008, reason="invalid access token")
            return
        await websocket.accept()
        connection = OneBotConnection(websocket)
        app.state.reminders.set_sender(connection.send_private_confirmed)
        logger.info("OneBot reverse WebSocket connected")
        try:
            while True:
                event = await websocket.receive_json()
                if "post_type" not in event:
                    connection.handle_response(event)
                    continue
                try:
                    # 接收和配置操作保持有序；模型请求由业务层后台执行，不阻塞收消息。
                    await app.state.bot.handle_event(event, connection.send_group, connection.send_private)
                except Exception:
                    # 单条业务事件异常只记录日志，继续处理连接上的后续事件。
                    logger.exception("Failed to handle OneBot event")
        except WebSocketDisconnect:
            logger.info("OneBot reverse WebSocket disconnected")
        finally:
            app.state.reminders.clear_sender(connection.send_private_confirmed)
            connection.close_pending()
            # 不让旧连接的任务在重连后继续向已关闭的 WebSocket 发送回复。
            await app.state.bot.cancel_pending(connection.send_group)
            await app.state.bot.cancel_pending(connection.send_private)

    return app


# uvicorn app.main:app 通过这个模块级对象启动服务。
app = create_app()
