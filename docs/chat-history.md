# 群聊原文历史查询

普通群聊Agent注册`read_chat_history`，由工具检索器按需加载。它读取当前群、本轮消息边界之前的原文，包含群友及机器人发言，不分类、不调用额外LLM，不改变发送目标或附件权限。

首次传`{}`，默认从上下文中连续已注入历史之前读取；历史被截断时，从该条之后开始，以重新获取完整记录。远端引用不影响起点，避免跳过中间消息。

| 参数 | 含义 |
| --- | --- |
| limit | 默认10，范围1～20 |
| cursor | 上页next_cursor，原样传回，仅本轮当前群有效 |
| before_source_id | 已提供的本地记录ID，读取它之前的原文；不是QQ消息号，不可与cursor混用 |

回执有`ok`、简单JSON聊天数组`messages`、`count`、`has_more`、`next_cursor`、`next_before_source_id`、`remaining_calls`与`budget_exhausted`。每页按旧到新排列，严格不超过limit。游标为程序生成的随机值，不能跨群或任务复用。

默认`CHAT_HISTORY_READ_MAX_CALLS=3`、`CHAT_HISTORY_READ_MAX_CHARS=12000`、`CHAT_HISTORY_READ_TOTAL_CHARS=24000`。优先保留完整消息；派生媒体过大时可省略，原话不悄悄跳过或截断。剩余额度连一条也放不下时返回预算不足且不推进，不代表没有历史。

每次回查来源存在性和遗忘阻断，不缓存原文回执。工具中的图片编号只是历史说明，查看仍需获得本轮目录或记忆工具授权；原文不扩大@、引用、URL或附件操作范围。
