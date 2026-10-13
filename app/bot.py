"""群机器人业务逻辑：命令处理、关键词召集、聊天、锐评和授权风格模仿。"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from app.config import Settings
from app.database import Database
from app.llm import DeepSeekClient, LLMUnavailable
from app.models import MessageView, at_segment, parse_message, text_segment

logger = logging.getLogger(__name__)
# 发送回调由连接层提供：业务层只传入群号和消息段，不直接操作 WebSocket。
SendGroup = Callable[[int, list[dict[str, Any]]], Awaitable[None]]


HELP_TEXT = """可用命令：
/名单 创建 名单名｜删除名单 名单名
/名单 添加 名单名 @某人｜删除 名单名 @某人
/名单 查看 [名单名]
/关键词 添加 词｜删除 词｜查看
/关键词 绑定 词 名单名｜解绑 词
只有完整发送已绑定的关键词，才会 @ 对应名单；改绑需先解绑。
/问 内容（或直接 @机器人）
/锐评 内容
/允许模仿｜/禁止模仿（只能为自己授权）
/删除我的数据
/模仿 @某人 主题
/帮助"""


class BotService:
    """协调配置、数据库和模型客户端，处理每个群的机器人功能。"""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        llm: DeepSeekClient,
    ):
        """保存依赖，并初始化按用户计时的冷却记录和按群分配的发送锁。"""
        self.settings = settings
        self.db = database
        self.llm = llm
        # 冷却记录只存在于内存中，重启服务后会重置。
        self._last_ai_call: dict[tuple[int, int], float] = defaultdict(float)
        self._trigger_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def handle_event(self, event: dict[str, Any], send_group: SendGroup) -> None:
        """过滤事件、保存消息，再依次处理命令、关键词和 @ 机器人聊天。"""
        if event.get("post_type") != "message" or event.get("message_type") != "group":
            return
        if event.get("sub_type") == "notice":
            return
        group_id = int(event["group_id"])
        user_id = int(event["user_id"])
        self_id = int(event.get("self_id", 0))
        # 忽略机器人自己的发言，避免回复触发自身业务逻辑。
        if user_id == self_id:
            return
        view = parse_message(event.get("message"))
        nickname = str((event.get("sender") or {}).get("card") or (event.get("sender") or {}).get("nickname") or user_id)
        # 优先使用群名片，缺失时使用昵称或 QQ 号；命令文本也会被记录。
        self.db.log_message(group_id, user_id, nickname, view.text)

        role = str((event.get("sender") or {}).get("role", "member"))
        is_admin = user_id in self.settings.admin_qq_ids or role in {"owner", "admin"}

        # 已识别的命令不再参与关键词召集或普通聊天，避免一次触发多个功能。
        if await self._handle_command(group_id, user_id, nickname, view, is_admin, send_group):
            return
        await self._handle_keyword(group_id, view.text, send_group)
        if self_id and self_id in view.mentions:
            prompt = view.text.replace(f"@{self_id}", "").strip()
            if prompt:
                await self._chat(group_id, user_id, prompt, send_group)

    async def _handle_command(
        self,
        group_id: int,
        user_id: int,
        nickname: str,
        view: MessageView,
        is_admin: bool,
        send_group: SendGroup,
    ) -> bool:
        """分发群内命令；识别并处理后返回 True，否则返回 False。"""
        text = view.text.strip()
        if text == "/帮助":
            await self._send_text(send_group, group_id, HELP_TEXT)
            return True
        if text.startswith("/名单"):
            if not is_admin:
                await self._send_text(send_group, group_id, "只有群主、管理员或机器人管理员可以维护名单。")
                return True
            await self._manage_users(group_id, text, view.mentions, send_group)
            return True
        if text.startswith("/关键词"):
            if not is_admin:
                await self._send_text(send_group, group_id, "只有管理员可以维护关键词。")
                return True
            await self._manage_keywords(group_id, text, send_group)
            return True
        if text == "/允许模仿":
            # 使用发送者自己的 QQ 号授权，无法通过 @ 替他人授权。
            self.db.set_mimic_consent(group_id, user_id, True)
            await self._send_text(send_group, group_id, f"已记录 {nickname} 的授权。之后的新发言可作为风格样本。")
            return True
        if text == "/禁止模仿":
            self.db.set_mimic_consent(group_id, user_id, False)
            await self._send_text(send_group, group_id, f"已关闭 {nickname} 的模仿授权。")
            return True
        if text == "/删除我的数据":
            deleted = self.db.delete_user_messages(group_id, user_id)
            await self._send_text(send_group, group_id, f"已删除你在本群保存的 {deleted} 条消息，并关闭模仿授权。")
            return True
        if text.startswith(self.settings.mimic_prefix):
            await self._mimic(group_id, user_id, text, view.mentions, send_group)
            return True
        if text.startswith(self.settings.evaluate_prefix):
            content = text.removeprefix(self.settings.evaluate_prefix).strip()
            if not content:
                await self._send_text(send_group, group_id, "用法：/锐评 要评价的内容")
            else:
                await self._ai_reply(
                    group_id,
                    user_id,
                    f"请对下面这段群聊内容做简短、有梗但不伤人的评价：\n{content}",
                    send_group,
                )
            return True
        if text.startswith(self.settings.chat_prefix):
            prompt = text.removeprefix(self.settings.chat_prefix).strip()
            if not prompt:
                await self._send_text(send_group, group_id, f"用法：{self.settings.chat_prefix} 你想说的话")
            else:
                await self._chat(group_id, user_id, prompt, send_group)
            return True
        return False

    async def _manage_users(self, group_id: int, text: str, mentions: list[int], send_group: SendGroup) -> None:
        """独立管理命名名单；旧的无名单名成员命令继续操作迁移后的“默认”名单。"""
        usage = "用法：/名单 创建|删除名单 名单名；/名单 添加|删除 名单名 @某人；/名单 查看 [名单名]。名单名为 1～32 字且不能含空白或 @。"
        parts = text.split(maxsplit=2)
        if len(parts) < 2 or parts[0] != "/名单":
            await self._send_text(send_group, group_id, usage)
            return
        action = parts[1]
        argument = parts[2].strip() if len(parts) == 3 else ""
        if action == "查看" and not argument:
            lists = self.db.list_lists(group_id)
            content = "\n".join(f'{row["name"]}（{row["member_count"]} 人）' for row in lists)
            await self._send_text(send_group, group_id, "当前群的名单：\n" + (content or "暂无，请先用 /名单 创建 名单名"))
            return
        if action not in {"创建", "删除名单", "查看", "添加", "删除"}:
            await self._send_text(send_group, group_id, usage)
            return
        if action in {"添加", "删除"}:
            # @ 消息段已被解析为 @QQ号；仅移除实际提及，不能把普通数字当作目标。
            for target in mentions:
                argument = re.sub(rf"@{target}(?!\d)", " ", argument)
            name = argument.strip() or "默认"
        else:
            name = argument
        if not name or len(name) > 32 or any(char.isspace() for char in name) or "@" in name:
            await self._send_text(send_group, group_id, usage)
            return
        if action == "创建":
            created = self.db.create_list(group_id, name)
            await self._send_text(send_group, group_id, f"已创建名单：{name}" if created else f"名单“{name}”已存在。")
            return
        if not self.db.has_list(group_id, name):
            await self._send_text(send_group, group_id, f"名单“{name}”不存在，请先发送 /名单 创建 {name}")
            return
        if action == "删除名单":
            self.db.delete_list(group_id, name)
            await self._send_text(send_group, group_id, f"已删除名单“{name}”及其成员，相关关键词已解绑但未删除。")
        elif action == "查看":
            users = self.db.list_managed_users(group_id, name)
            content = "、".join(str(row["user_id"]) for row in users) or "暂无成员"
            # 管理时只列出 QQ 号，不额外 @；真正召集由关键词触发。
            await self._send_text(send_group, group_id, f"名单“{name}”：{content}")
        elif not mentions:
            await self._send_text(send_group, group_id, f"请 @ 要操作的用户，例如：/名单 {action} {name} @某人")
        elif action == "添加":
            targets = list(dict.fromkeys(mentions))
            for target in targets:
                self.db.add_managed_user(group_id, name, target)
            await self._send_text(send_group, group_id, f"已将 {len(targets)} 人加入名单“{name}”（已有成员不会重复添加）。")
        elif action == "删除":
            removed = sum(self.db.remove_managed_user(group_id, name, target) for target in set(mentions))
            await self._send_text(send_group, group_id, f"已从名单“{name}”删除 {removed} 人。")

    async def _manage_keywords(self, group_id: int, text: str, send_group: SendGroup) -> None:
        """独立维护关键词及绑定，允许多关键词共用名单，但禁止直接覆盖绑定。"""
        usage = "用法：/关键词 添加|删除|解绑 关键词；/关键词 绑定 关键词 名单名；/关键词 查看"
        parts = text.split(maxsplit=2)
        if len(parts) < 2 or parts[0] != "/关键词":
            await self._send_text(send_group, group_id, usage)
            return
        action = parts[1]
        if action == "查看" and len(parts) == 2:
            keywords = self.db.list_keywords(group_id)
            content = "\n".join(f'{row["keyword"]} → {row["list_name"] or "未绑定"}' for row in keywords)
            await self._send_text(send_group, group_id, "关键词与名单：\n" + (content or "暂无"))
            return
        # 最多拆分两次，保留关键词内部的空格。
        if len(parts) < 3 or not parts[2].strip():
            await self._send_text(send_group, group_id, usage)
            return
        action, keyword = parts[1], parts[2].strip()
        if action == "添加":
            created = self.db.add_keyword(group_id, keyword)
            await self._send_text(send_group, group_id, f"已添加关键词：{keyword}。请绑定名单后使用。" if created else f"关键词“{keyword}”已存在，原绑定保持不变。")
        elif action == "删除":
            removed = self.db.remove_keyword(group_id, keyword)
            await self._send_text(send_group, group_id, "已删除。" if removed else "没有找到这个关键词。")
        elif action == "绑定":
            # 最后一个空白分隔的参数是名单名，关键词内部的空格仍然保留。
            binding = keyword.rsplit(maxsplit=1)
            if len(binding) != 2:
                await self._send_text(send_group, group_id, usage)
                return
            keyword, name = binding
            result = self.db.bind_keyword(group_id, keyword, name)
            replies = {
                "bound": f"已绑定：{keyword} → {name}",
                "already_bound": f"关键词“{keyword}”已经绑定名单“{name}”。",
                "missing_keyword": f"关键词不存在，请先发送 /关键词 添加 {keyword}",
                "missing_list": f"名单不存在，请先发送 /名单 创建 {name}",
                "conflict": f"关键词“{keyword}”已绑定其他名单，不能绑定多个名单。请先发送 /关键词 解绑 {keyword} 再重新绑定。",
            }
            await self._send_text(send_group, group_id, replies[result])
        elif action == "解绑":
            removed = self.db.unbind_keyword(group_id, keyword)
            await self._send_text(send_group, group_id, "已解绑，关键词和名单均保留。" if removed else "这个关键词不存在或尚未绑定名单。")
        else:
            await self._send_text(send_group, group_id, usage)

    async def _handle_keyword(self, group_id: int, text: str, send_group: SendGroup) -> None:
        """消息文本与已绑定关键词完全相同时，只召集对应名单的成员。"""
        # 忽略首尾空白，保留消息内部内容；关键词作为其他句子的子串时不触发。
        text = text.strip()
        if not text:
            return
        # 按群串行发送，避免同一群的两次召集消息相互穿插；此处没有触发冷却。
        async with self._trigger_locks[group_id]:
            name = self.db.keyword_list(group_id, text)
            if name is None:
                return
            users = self.db.list_managed_users(group_id, name)
            if not users:
                return
            # 每条消息最多包含 20 个 @ 用户，控制单次消息大小。
            for start in range(0, len(users), 20):
                message: list[dict[str, Any]] = [text_segment(f"{name}，集合啦：")]
                for row in users[start : start + 20]:
                    message.extend([text_segment(" "), at_segment(row["user_id"])])
                await send_group(group_id, message)

    async def _chat(self, group_id: int, user_id: int, prompt: str, send_group: SendGroup) -> None:
        """读取最近群消息，附上发送者昵称作为上下文，再请求模型回复。"""
        rows = self.db.recent_messages(group_id, self.settings.context_message_limit)
        # 当前消息已在入口保存，因此跳过最后一条，避免和 prompt 重复。
        history = [
            {"role": "user", "content": f'{row["nickname"]}: {row["content"]}'}
            for row in rows[:-1]
        ]
        await self._ai_reply(group_id, user_id, prompt, send_group, history=history)

    async def _mimic(
        self, group_id: int, user_id: int, text: str, mentions: list[int], send_group: SendGroup
    ) -> None:
        """确认目标用户授权和样本数量后，用其发言样本生成指定主题的新话。"""
        if not mentions:
            await self._send_text(send_group, group_id, f"用法：{self.settings.mimic_prefix} @某人 [主题]")
            return
        # 一次只模仿首个被 @ 的用户，至少需要 5 条授权期间保存的消息。
        target = mentions[0]
        if not self.db.has_mimic_consent(group_id, target):
            await self._send_text(send_group, group_id, "对方尚未发送 /允许模仿，不能模仿。")
            return
        samples = self.db.style_samples(group_id, target, self.settings.style_sample_limit)
        if len(samples) < 5:
            await self._send_text(send_group, group_id, "可用样本不足，至少需要 5 条发言。")
            return
        topic = text.replace(self.settings.mimic_prefix, "", 1).replace(f"@{target}", "").strip() or "随便聊两句"
        # 每条样本截取前 300 字，减少提示词长度和模型调用成本。
        sample_text = "\n".join(f"- {sample[:300]}" for sample in samples)
        system = """根据用户主动授权的聊天样本，提炼语言风格并生成一句新话。
不要声称自己就是该用户，不复述隐私信息，不模仿敏感身份特征，不输出分析过程。只输出生成的话，最多 100 字。"""
        await self._ai_reply(
            group_id,
            user_id,
            f"发言样本：\n{sample_text}\n\n主题：{topic}",
            send_group,
            system=system,
        )

    async def _ai_reply(
        self,
        group_id: int,
        user_id: int,
        prompt: str,
        send_group: SendGroup,
        *,
        history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> None:
        """统一执行 AI 调用：检查冷却、生成回复、处理异常并 @ 提问者。"""
        key = (group_id, user_id)
        # 单调时钟适合计算时间间隔，不受系统日期调整影响。
        now = time.monotonic()
        wait = self.settings.ai_cooldown_seconds - (now - self._last_ai_call[key])
        if wait > 0:
            await self._send_text(send_group, group_id, f"慢一点，{wait:.0f} 秒后再问我。")
            return
        # 在请求前记录时间，失败的请求同样计入冷却，避免连续重试。
        self._last_ai_call[key] = now
        try:
            kwargs: dict[str, Any] = {"history": history}
            if system is not None:
                kwargs["system"] = system
            answer = await self.llm.complete(prompt, **kwargs)
        except LLMUnavailable as exc:
            logger.warning("LLM unavailable: %s", exc)
            await self._send_text(send_group, group_id, f"AI 暂时不可用：{exc}")
            return
        await send_group(group_id, [at_segment(user_id), text_segment(f" {answer}")])

    @staticmethod
    async def _send_text(send_group: SendGroup, group_id: int, text: str) -> None:
        """通过发送回调向群内发送一条纯文本消息。"""
        await send_group(group_id, [text_segment(text)])
