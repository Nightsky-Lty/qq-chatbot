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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class OneBotConnection:
    """封装单个 OneBot 连接上的群消息发送操作。"""

    def __init__(self, websocket: WebSocket):
        """保存连接，并建立发送锁和请求序号。"""
        self.websocket = websocket
        self.lock = asyncio.Lock()
        self.sequence = 0

    async def send_group(self, group_id: int, message: list[dict[str, Any]]) -> None:
        """向 NapCat 发送群消息调用；这里只提交请求，不等待发送结果确认。"""
        # 同一连接上的发送操作串行执行，避免并发写入。
        async with self.lock:
            self.sequence += 1
            await self.websocket.send_json(
                {
                    "action": "send_group_msg",
                    "params": {"group_id": group_id, "message": message},
                    # echo 用于标识请求，NapCat 的调用响应会携带该字段。
                    "echo": f"send-{self.sequence}",
                }
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建应用，可显式传入配置；未传入时使用缓存的环境配置。"""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """启动时初始化资源并清理过期消息，关闭时释放客户端和数据库。"""
        db = Database(cfg.database_path)
        purged = db.purge_old_messages(cfg.message_retention_days)
        if purged:
            logger.info("Purged %s expired messages", purged)
        llm = DeepSeekClient(cfg)
        app.state.bot = BotService(cfg, db, llm)
        app.state.db = db
        app.state.llm = llm
        # yield 前是启动逻辑，yield 后是应用关闭时执行的清理逻辑。
        yield
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
        logger.info("OneBot reverse WebSocket connected")
        try:
            while True:
                event = await websocket.receive_json()
                if "post_type" not in event:
                    # API 调用响应没有 post_type，当前不处理这些响应。
                    continue
                try:
                    # 当前连接逐条处理事件；模型请求完成前不会读取下一条事件。
                    await app.state.bot.handle_event(event, connection.send_group)
                except Exception:
                    # 单条业务事件异常只记录日志，继续处理连接上的后续事件。
                    logger.exception("Failed to handle OneBot event")
        except WebSocketDisconnect:
            logger.info("OneBot reverse WebSocket disconnected")

    return app


# uvicorn app.main:app 通过这个模块级对象启动服务。
app = create_app()
