"""统一读取和管理机器人配置，配置来源包括环境变量和项目根目录的 .env。"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """机器人配置及默认值，环境变量名称与字段名称对应，不区分大小写。"""

    # 允许 .env 中存在额外字段，便于在共享配置文件中保留其他服务的配置。
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DeepSeek 请求的认证信息、接口地址和模型名称。
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
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
    ai_cooldown_seconds: int = 8
    # 限制后台请求总数，不建立无上限队列；满额时请用户稍后重试。
    ai_max_concurrency: int = Field(default=3, ge=1, le=16)
    context_message_limit: int = Field(default=40, ge=1)
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
    """缓存配置对象，避免重复读取 .env；配置修改后需要重启服务。"""
    return Settings()
