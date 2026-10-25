# 角色卡来源与加载

当前角色卡采用 [shirachino/astrbot-persona-prompts](https://github.com/shirachino/astrbot-persona-prompts) 的 DeepSeek 群友版本，内容适配在 `app/prompts/persona.txt`，由 `app/persona.py` 以UTF-8读取。`PERSONA_PROMPT`与`SYSTEM_PROMPT`为同一份完整文本，群聊、私聊和锐评每次固定注入；专用模仿使用自己的提示词。修改角色卡后重建bot容器生效。

来源为 `deepseek/大肥鱼群友提示词 Beta.docx`，固定于提交 `869d6793b93a756a410f00da44852f8b774e48f1`。作者README说明可按需填写游戏和爱好，角色卡及仓库声明以作者发布内容为准。此为社区二创形象，不代表官方角色设定。

当前人设偏普通群友、懂梗、偶尔吐槽和轻微傲娇；认真求助或情绪低落时简短认真。保留兴趣比例和7组对话示例，游戏偏好为喜欢原神和英雄联盟（LOL），讨厌鸣潮、王者荣耀和瓦罗兰特，对其他游戏无感；示例不提供虚构餐馆、交易或共同经历。AI身份如实说明，拟人经历、主观看法与真实事实区分，互动和权限遵守程序规则。

发送字数、分条、网址/代码例外、工具调用、会话隔离、真实回执与群聊/私聊参与规则由 `app/prompting.py` 及实际执行器统一负责。角色卡不声明未注册的AstrBot插件或工具。

## 来源校验

- 源DOCX SHA-256：`63997ccd1c4e606162c61c1474df271de452ff0fbd0a9cae89cf13f41b4eda18`
- 提取文本 SHA-256：`8532acbb228e9e99a25aba4c76f07d613d0a3651d1380fda65b56c1ed5a7e7fc`
- 当前角色卡 SHA-256：`d5aebad4aacd421a0f5b4f96f16bd4eec31bd8101de87b27fb9356c008749245`
