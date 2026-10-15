"""统一读取配置：普通值来自 config.env，私密值来自 .env 或进程环境。"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """机器人配置及默认值，环境变量名称与字段名称对应，不区分大小写。"""

    # 本地运行时以 .env 覆盖仓库默认值；Compose 只从 .env 显式注入敏感字段，
    # 因而服务器旧 .env 中残留的普通参数不会阻止 config.env 随 git pull 更新。
    model_config = SettingsConfigDict(env_file=("config.env", ".env"), extra="ignore")

    # DeepSeek 请求的认证信息、接口地址和模型名称。
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_thinking_enabled: bool = True
    deepseek_reasoning_effort: Literal["low", "high", "max"] = "low"
    # 开启思考时包括推理与最终输出，避免旧500/1500上限截断工具调用。
    deepseek_thinking_max_tokens: int = Field(default=4096, ge=512, le=16384)
    # OneBot 连接认证与额外管理员；群主、群管理员的权限在业务层判断。
    onebot_access_token: str = ""
    # 跳过环境变量来源的自动 JSON 解码，由下面的校验器兼容两种输入格式。
    admin_qq_ids: Annotated[set[int], NoDecode] = Field(default_factory=set)
    database_path: Path = Path("data/bot.db")
    # bot_name 暂未参与提示词或消息生成；命令前缀由业务层读取。
    bot_name: str = "小群友"
    chat_prefix: str = "/问"
    evaluate_prefix: str = "/锐评"
    mimic_prefix: str = "/模仿"
    # 冷却单位为秒；上下文和模仿样本限制按消息条数计算。
    ai_cooldown_seconds: int = Field(default=8, ge=0, le=600)
    # 限制模型运行总数；群话题有界排队，私聊/模仿满额时提示重试。
    ai_max_concurrency: int = Field(default=3, ge=1, le=16)
    agent_max_steps: int = Field(default=8, ge=2, le=16)
    agent_max_tool_calls: int = Field(default=16, ge=4, le=32)
    agent_timeout_seconds: int = Field(default=120, ge=20, le=300)
    # 无Key的Bing网页搜索；公网只读抓取，共用全局并发上限2。
    web_search_enabled: bool = True
    web_search_max_results: int = Field(default=5, ge=1, le=8)
    web_search_max_calls: int = Field(default=2, ge=1, le=4)
    web_fetch_max_calls: int = Field(default=2, ge=1, le=4)
    web_timeout_seconds: int = Field(default=15, ge=3, le=30)
    web_max_response_bytes: int = Field(default=1024 * 1024, ge=65536, le=5 * 1024 * 1024)
    web_fetch_max_chars: int = Field(default=6000, ge=1000, le=12000)
    context_message_limit: int = Field(default=80, ge=1, le=200)
    context_max_chars: int = Field(default=24000, ge=1000, le=100000)
    context_bot_reply_limit: int = Field(default=12, ge=0, le=50)
    # 群成员长期记忆：SQLite 持久化，使用本地向量做混合 RAG，不依赖额外服务。
    memory_enabled: bool = True
    memory_auto_capture_enabled: bool = True
    memory_retrieval_limit: int = Field(default=8, ge=1, le=30)
    memory_candidate_limit: int = Field(default=300, ge=20, le=2000)
    memory_user_history_limit: int = Field(default=12, ge=0, le=50)
    memory_max_per_user: int = Field(default=200, ge=10, le=2000)
    memory_max_chars: int = Field(default=200, ge=20, le=500)
    memory_min_confidence: float = Field(default=0.8, ge=0.5, le=1.0)
    # bge_onnx 使用只读挂载的本地 INT8 模型；加载或推理失败时自动退回哈希向量。
    embedding_provider: Literal["hash", "bge_onnx"] = "hash"
    embedding_model_path: Path = Path("/app/models/bge-small-zh-v1.5")
    embedding_model_file: str = "model_quantized.onnx"
    embedding_max_length: int = Field(default=128, ge=16, le=512)
    embedding_threads: int = Field(default=2, ge=1, le=4)
    embedding_backfill_batch_size: int = Field(default=16, ge=1, le=64)
    embedding_min_similarity: float = Field(default=0.35, ge=0, le=1)
    embedding_query_prefix: str = "为这个句子生成表示以用于检索相关文章："
    chat_batch_seconds: float = Field(default=3, ge=0, le=10)
    chat_batch_max_messages: int = Field(default=30, ge=3, le=100)
    chat_max_pending_groups: int = Field(default=32, ge=1, le=100)
    chat_queue_wait_seconds: int = Field(default=120, ge=10, le=600)
    chat_active_idle_seconds: int = Field(default=300, ge=10, le=3600)
    chat_active_max_seconds: int = Field(default=1800, ge=30, le=7200)
    group_reply_max_chars: int = Field(default=100, ge=10, le=500)
    # 主动戳一戳由LLM决定；每轮最多一次，同一群/私聊共用跨轮冷却。
    poke_enabled: bool = True
    poke_cooldown_seconds: int = Field(default=60, ge=1, le=3600)
    # 最近不同发言人数决定随机唤醒概率；活跃话题沿用原调度，不重复抽签。
    random_chat_enabled: bool = True
    random_chat_window_seconds: int = Field(default=120, ge=10, le=3600)
    random_chat_min_probability: float = Field(default=0.01, ge=0, le=1)
    random_chat_probability_per_user: float = Field(default=0.02, ge=0, le=1)
    random_chat_max_probability: float = Field(default=0.20, ge=0, le=1)
    random_chat_cooldown_seconds: int = Field(default=60, ge=1, le=3600)
    # 兼容旧 .env 的字段；三人自动接话入口已由群话题状态机替代，不再读取人数配置。
    auto_chat_enabled: bool = False
    auto_chat_min_users: int = Field(default=3, ge=3)
    auto_chat_window_seconds: int = Field(default=180, ge=1)
    auto_chat_cooldown_seconds: int = Field(default=30, ge=1)
    # 提醒由本地后台调度，到期通知不依赖模型再次生成。
    reminder_poll_seconds: int = Field(default=5, ge=1, le=60)
    reminder_max_pending: int = Field(default=50, ge=1, le=100)
    # 来图只有模型调用工具才下载；独立于本地表情包视觉开关。
    chat_image_enabled: bool = True
    chat_image_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=5 * 1024 * 1024)
    # 表情包仅从本地目录读取；每次回复最多一张，发送限制按群或私聊会话计算。
    sticker_enabled: bool = True
    sticker_vision_enabled: bool = True
    # 入库识别在独立后台任务中执行；关闭视觉时同时停用自动识别。
    sticker_auto_index_enabled: bool = True
    sticker_index_poll_seconds: int = Field(default=60, ge=10, le=3600)
    sticker_index_batch_size: int = Field(default=5, ge=1, le=20)
    sticker_index_retry_seconds: int = Field(default=3600, ge=60, le=86400)
    # 最多3张，按单图最高10MiB计算，内联Base64仍在DeepSeek 48MiB请求体限制以内。
    sticker_preview_limit: int = Field(default=3, ge=1, le=3)
    # 仅预览文字元数据，不调用视觉或额外模型；0关闭预览，仍可按需搜索。
    sticker_prompt_limit: int = Field(default=10, ge=0, le=20)
    sticker_prompt_rotate_seconds: int = Field(default=3600, ge=60, le=86400)
    sticker_directory: Path = Path("app/sticker_images")
    sticker_collect_enabled: bool = True
    sticker_collect_probability: float = Field(default=0.20, ge=0, le=1)
    sticker_collect_queue_size: int = Field(default=32, ge=1, le=200)
    sticker_collect_queue_wait_seconds: int = Field(default=120, ge=10, le=600)
    # 旧图片观察配置仅兼容读取，随机入口不依赖话题；概率0仅停随机，保留收藏工具。
    sticker_collect_auto_enabled: bool = False
    sticker_collected_directory: Path = Path("data/collected_stickers")
    sticker_collect_observe_seconds: int = Field(default=300, ge=30, le=3600)
    sticker_collect_cooldown_seconds: int = Field(default=60, ge=0)
    sticker_collect_max_files: int = Field(default=200, ge=1, le=1000)
    sticker_collect_max_disk_bytes: int = Field(default=200 * 1024 * 1024, ge=1024 * 1024, le=1024 * 1024 * 1024)
    sticker_cooldown_seconds: int = Field(default=30, ge=0)
    sticker_repeat_seconds: int = Field(default=7200, ge=0)
    sticker_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=10 * 1024 * 1024)
    style_sample_limit: int = 60
    # 保留期单位为天，过期消息在服务启动时清理；非正数表示不自动清理。
    message_retention_days: int = 30

    @field_validator("admin_qq_ids", mode="before")
    @classmethod
    def parse_admin_ids(cls, value: object) -> object:
        """兼容逗号分隔 QQ 号和 JSON 数组，统一转换为整数集合。"""
        if isinstance(value, str):
            if value.strip().startswith("["):
                return {int(part) for part in json.loads(value)}
            return {int(part.strip()) for part in value.split(",") if part.strip()}
        return value


@lru_cache
def get_settings() -> Settings:
    """缓存配置对象，避免重复读取配置文件；配置修改后需要重启服务。"""
    return Settings()
