# QQ 群 DeepSeek 机器人

基于 Python、FastAPI 和 SQLite 的 QQ 群机器人，通过 NapCatQQ 的 OneBot 11 反向 WebSocket 收发消息，调用 DeepSeek API 完成聊天、锐评和发言风格模仿。

提供 Docker Compose 部署配置，可在一台 Linux 云服务器上同时运行机器人和 NapCat。完整安装步骤见 [Ubuntu 云端部署指南](docs/deployment.md)。

## 功能

| 功能 | 当前行为 |
| --- | --- |
| 群成员名单 | 每个群可独立创建多个命名名单，支持维护成员和删除名单，同一用户可加入多个名单 |
| 关键词召集 | 关键词独立管理，每个关键词最多绑定一个名单，多个关键词可绑定同一名单；完整匹配时分批 @ 对应名单成员，每条最多 20 人 |
| 群聊对话 | 使用 `/问`、@ 或戳一戳机器人触发，以“DeepSeek 小鲸鱼”群友角色回应当前消息 |
| 群话题活跃期 | @、戳一戳、聊天命令或程序按活跃人数随机唤醒，固定3秒聚批；等待新消息时继续参与 |
| 戳一戳互动 | 戳机器人可唤醒Agent；LLM可按语境调用 `send_poke` 主动戳或戳回，受目标白名单及冷却限制 |
| 群聊AI语音 | LLM提供口语文字和真实音色编号，由NapCat调用QQ AI合成并直接发群语音，无需本地TTS模型 |
| 联网搜索 | DeepSeek官方Search/Fetch API，搜索直接携带正文，按需补读；复用现有DeepSeek Key |
| 私聊对话 | 直接发文字即可聊天，也支持 `/问`、`/锐评`、`/帮助`；每位用户上下文独立 |
| 私聊备忘录 | 用普通对话设置一次性提醒、查看待办或取消；DeepSeek 调用工具，到期由本地调度私聊通知 |
| 本地表情包 | 新图片后台自动识别入库；群聊、私聊和自动接话可检索、看图并选择发送，每次最多一张 |
| 聊天看图 | 私聊或群聊@、/问、/锐评的当前附图，LLM自行决定是否调用 `get_message_image` 查看 |
| 群表情收藏 | 程序保留概率抽签，LLM也可调用 `collect_sticker` 收藏本轮授权的当前/近期同群图片；共用SHA-256去重、视觉审核和容量限制 |
| 内容锐评 | 使用 `/锐评` 主动提交要评价的文本 |
| 发言风格模仿 | 直接 `/模仿 @某人 主题`，使用目标用户在本群的普通文字发言，无需授权 |
| 长期成员记忆 | 按群号和 QQ 号保存别名及稳定事实，使用本地 BGE/哈希混合 RAG 自动召回，不增加群命令 |
| 数据持久化 | SQLite 保存名单、关键词、群消息、成员记忆、独立的私聊记录及提醒 |

模仿功能使用样本提示词，不训练或微调模型。机器人处理文本、@、QQ引用、戳一戳通知和本批新消息中的图片，支持发送群AI语音，但收到的语音和视频暂不识别。低活跃期普通群消息先保存历史，随机唤醒机制保留但当前概率为0；@、戳一戳或聊天命令直接唤醒，话题活跃期间无需每条@。私聊的普通文字、图片或戳一戳会调用AI。

## ReAct Agent架构

群聊路径为：OneBot事件 → 原文入库/轻量归属 → GroupConversations聚批/排队 → DialogueService按需解析/选择唯一thread → BotService绑定该thread任务 → ReactAgent决策 → ToolRegistry执行工具 → 将回执/图片作为观察交回模型 → 再决策，直到 `wait_for_messages` 或 `finish`。前者保留当前交流焦点，后者退出跟进；其他thread只维护状态，不并发启动回复。详见[会话解析业务层](docs/dialogue-threads.md)。这是ReAct风格的函数调用Agent，不是前端React框架；决策、动作和环境观察交替，思路见[ReAct论文](https://arxiv.org/abs/2210.03629)。

| 模块 | 职责 |
| --- | --- |
| `app/llm.py` | 单次DeepSeek请求及独立图片审核，不维护业务工具循环 |
| `app/agent.py` | 模型/工具迭代、观察、图片预算、步骤/超时限制及同调用编号去重 |
| `app/tools.py` | 每次请求独立注册定义和执行器、校验参数，不注册任意Python函数 |
| `app/memory.py` | 群成员长期记忆写入约束、BGE/哈希向量和混合 RAG 排序 |
| `app/messaging.py` | 注册 `send_message(text, reply_to_message_id?, at_user_ids?)`，引用/@校验、短句限制、去重、保存成功发送 |
| `app/poking.py` | 注册 `send_poke(user_id?)`，会话绑定、目标校验、跨轮冷却、确认发送及动作入库 |
| `app/voice.py` | 当前群真实音色查询、`send_voice(text, character_id?)`、语音冷却/条数预算及成功历史 |
| `app/web_search.py` | `web_search(query)`、`web_fetch(url)`，DeepSeek搜索/抓取与本轮正文缓存、本轮URL范围和次数限制 |
| `app/safe_web.py` | 只读公网HTTP，逐跳DNS检查、固定连接IP、TLS验证、超时与下载上限 |
| `app/conversation.py` | 按群固定聚批、串行有界队列、话题活跃期和超时 |
| `app/bot.py` | 路由和授权范围、绑定工具到真实当前会话、并发与冷却 |
| `app/main.py` | OneBot传输，后台发送等待响应，断线取消对应任务 |

可注册工具：`send_message`、`send_poke`、`web_search`、`web_fetch`、`finish`、群活跃跟进的 `wait_for_messages({})`；群语音的 `list_voice_characters`、`send_voice`；表情的 `search_stickers`、`list_stickers`、`get_sticker_image`、`send_sticker`；本轮授权当前/近期图片的 `get_message_image`；本轮授权当前/近期同群图片的 `collect_sticker(image_id)`；私聊的三个提醒工具。概率后台与收藏工具共用处理入口，不互相取代。列表按开关和场景裁剪；模仿只注册发送与结束。发送会话目标由程序固定，不能更换群号或私聊收件人；群聊文字和表情默认普通发送，省略reply_to_message_id仍回应当前焦点，不自动显示引用框。只有模型显式填写该参数时才显示QQ引用，只接受本批允许的新消息号，不能引用历史或其他群。图片工具不接受任意URL或路径。

群聊 `send_message` 可选 `at_user_ids`，例如 `{"text":"一起来聊聊？","at_user_ids":[123456789]}`，通过真实 OneBot `at` 消息段主动点名，不是文本假@。每次最多3个QQ号、重复目标只@一次，不能@全体或机器人自己；只允许本批发言人、已提供的近期历史发言人及本轮相关已知群成员，服务端按上下文快照校验，不能从正文猜测QQ号或跨群@。默认不@，模型可按当前话题主动邀请或提醒；可同时填写 `reply_to_message_id`，引用和@互相独立。分条仅第一条带@，发送去重包含@目标，成功后在历史文字中保留 `@QQ号`。私聊不提供引用或@参数，也不允许使用@参数。

`send_poke({"user_id":123456789})` 在当前群戳指定成员，省略参数 `send_poke({})` 则戳当前发言人；私聊只允许当前聊天对象。群聊目标与本轮允许@成员共用白名单，不能戳自己、猜测QQ或指定其他群号。由LLM按语境决定是否主动戳或戳回，不要求每次都用。每轮最多尝试一次，同一群/私聊默认冷却60秒，换目标不能绕过冷却；失败、超时和取消也占用冷却，避免确认丢失后立即重发。轻互动可只戳一戳，明确的问题仍应文字回答，备忘回执必须先用文字确认。通过NapCat独立 `send_poke` 动作发送并等待响应，不使用假文字或CQ码；确认成功后将动作占位保存为assistant历史（没有可引用消息ID），后续可知道自己戳过谁。发送未确认不重试，也不自动补发文字。自己的戳一戳通知被入口过滤，不会唤醒自身形成循环。`POKE_ENABLED=false`隐藏此工具，不影响被戳唤醒；模仿和后台收图不注册。冷却仅保存在内存，重启后重置。接口见 [NapCat文档](https://napneko.github.io/develop/api/doc)。

默认最多8次模型决策、16个工具调用、120秒整轮超时，聊天与表情查看共用最多3张原图；终止后的同批动作不再执行。模型调用发送工具后，不会再次发送最后的普通文本；未调用发送工具却直接返回最终文字时，只通过同一个发送工具兼容提交一次。失败/超时已经尝试发送时不做无条件补发。

### 减少收尾轮次与上下文重复

默认开启 `AGENT_BATCH_COMPLETION_ENABLED=true`。最后一个发送工具（文字、表情、语音或戳一戳）可以携带 `next_action="finish"`，群聊继续跟进则携带 `next_action="wait"`，不再提供 `topic`。程序先执行原有校验、等待真实发送成功，再调用绑定的结束/等待处理器，不必额外请求模型只做收尾。例如 `send_message({"text":"九点见","next_action":"wait"})`。仍需观察工具结果、补充发送或继续操作时省略该字段，或使用 `continue`；有依赖的搜索/读取/提醒操作依旧先取得真实回执。原有结束工具保留。关掉配置可恢复逐步收尾。

无工具调用的最终文本通过原发送器成功提交后，由程序直接收尾：群聊默认保持当前交流焦点等待新消息，私聊结束；需要明确告别时模型可通过发送工具指定finish。所有发送预算、目标、冷却及失败不重发约束不变。一次工具批次中收尾成功后，后面的动作不再执行。

上下文JSON使用紧凑序列化，省略没有信息的空@/引用字段。同一历史时间线中已出现的引用原文只保留作者与记录索引，不重复附带正文；未在时间线内的已知引用仍保留原文，未找到的引用继续明确标注未知。人设、历史预算和思考强度保持原配置。

`Agent finished` 日志记录实际模型轮次及API返回的 `prompt_tokens`、`completion_tokens`、`reasoning_tokens`、`cache_hit_tokens` 和 `usage_requests`。后者表示提供完整输入/输出计数的请求数；服务不报告的计数不能视为真实零消耗。推理token通常已包含在completion内，不重复相加；缓存命中影响计费，不能直接把总输入token降幅当成账单降幅。日志不包含正文或推理内容。

可运行 `PYTHONPATH=. python scripts/benchmark_agent_efficiency.py` 对比基线历史大小；加 `--live` 会调用真实API回放合成群聊和提醒任务，使用临时数据库和本地发送回调，不向QQ发消息。真实API回放仍会消耗所配置服务的调用额度。

## 群聊QQ AI语音

群里直接“@机器人 用语音说句晚安”，不新增命令。DeepSeek提供要说的文字，由NapCat调用QQ的AI语音能力合成并发送，不是DeepSeek生成音频，不下载本地TTS模型，也不需要额外语音API Key。仅支持发群语音，不识别收到的语音，私聊、模仿和后台观察不开放这些工具。

`list_voice_characters(offset?)` 查询当前群真实音色编号与名称，每页20项，最多保留128项；不合成、不发送，不向模型暴露音频链接。查询调用NapCat `get_ai_characters(group_id, chat_type=1)`，与发送接口固定使用的普通语音类型一致；成功按群/当前连接缓存10分钟，失败缓存15秒，本轮失败不重复查。连接断开后清除相应缓存。名称等元数据不是指令。默认音色由 `VOICE_CHARACTER_ID` 指定，留空使用当前群实际列表第一项；配置音色或显式选择的编号不在可用列表时拒绝，不静默换音色、不猜角色编号。

`send_voice(text, character_id?)` 直接调用 `send_group_ai_record`，内部固定当前群号，等待合成和发送回执最多45秒。NapCat当前接口成功时仍返回 `message_id: 0`，因此以OneBot成功状态确认，不把零消息号误判成失败；数据库照常保存语音文字，但该条记录没有可引用的真实消息号。默认每轮最多尝试1条、最多100字，同时受 `GROUP_REPLY_MAX_CHARS` 限制；语音和文字共用回复条数和逻辑焦点，不借语音突破原有群回复预算。同一句本轮不重复发语音和文字。群语音至少间隔60秒，尝试发送失败/超时/取消也占用冷却，不能换音色绕过；冷却只在内存，重启后重置。

默认仍文字聊天，由LLM按语境或用户要求选择语音；成功后不另发“已发送语音”。原生接口不能附加@或QQ引用框，但数据库保存逻辑回应关系。确认成功后把实际说出的文字、音色名称和消息ID写入assistant历史，继续参与现有上下文，不保存音频或签名链接。收藏等操作的待确认回执仍先用文字发送。

账号/群/QQ版本或NapCat接口不支持时可能失败，实际可用性需在部署环境验证。音色查询及发送前校验失败时可改用文字；实际发送已尝试但失败、断线或超时未确认时，本轮不重试，也不自动补发文字，避免回执丢失造成重复。不会因查询成功就声称语音发出。文字和所选音色交给QQ/NapCat处理，不进行声音克隆。接口依据 [NapCat官方文档](https://napneko.github.io/develop/api/doc)。

配置位于 `config.env`：`VOICE_ENABLED=true`、`VOICE_CHARACTER_ID=`、`VOICE_MAX_CHARS=100`、`VOICE_COOLDOWN_SECONDS=60`；关闭开关隐藏两个工具。更新后重建并重新创建bot容器生效。

## DeepSeek联网搜索

群聊直接说“@机器人 帮我搜一下……”，私聊说“搜索……”，不新增命令。聊天模型调用 `web_search(query)`，程序POST请求 `https://api.deepseek.com/v1/tools/search`，默认 `count=5, need_content=true`，取得结构化标题、链接、摘要、可用正文和来源时间/评分。不再抓Bing HTML、不自行提取网页正文、不固定再Fetch两页，也没有额外聊天LLM。`web_fetch(url)` 在缓存缺失时调用 `https://api.deepseek.com/v1/tools/fetch` 获取Markdown正文，适用于重要来源缺正文或用户直接提供链接。两个接口固定官方地址、复用 `DEEPSEEK_API_KEY`，独立于聊天 `DEEPSEEK_BASE_URL`（聊天中转Key不保证能用于官方接口）。模仿和后台收图不开放联网工具。[官方接口说明](https://api-docs.deepseek.com/zh-cn/guides/search/)

默认每轮Search最多2次、每次5条结果，额外Fetch最多4次；失败也计入本地次数限制，Search自带正文及缓存命中不占Fetch额度。每页正文保留最多6000字符，单次搜索展开正文总量最多12000字符，其余已返回正文留在本轮缓存。`page_status=read` 表示正文在 `page.content`，`cached` 表示已有缓存但本次未展开（`web_fetch` 取出不产生额外API请求），`not_read` 表示缺正文。摘要 `snippet/summary` 不当正文，时间未知不编造，排序评分仅供参考。正文保留Markdown和换行，超限明确 `truncated`，不是整页全文；暂不支持读取超过单页保留上限的后续内容。

相同查询及URL（成功/失败）只在本轮缓存，不跨群/用户共享；Search正文直接进入页面缓存，后续 `web_fetch` 不重复请求收费API。缺正文不会自动Fetch，交给聊天模型判断；Fetch失败不自动重试，后续Search如带回正文可更新失败缓存。空搜索结果如实返回，不等同于网络上不存在相关信息。每次API请求含排队、节流与完整响应最多30秒，响应最多1MiB，共用2个并发名额；Search和Fetch各按4QPS平滑限速，只约束本进程，无法约束同账号其他服务的请求。

API请求只发往固定DeepSeek官方HTTPS地址，不跟随重定向、不继承环境代理，Key不随网页URL发送。网页URL仍限制为标准端口、无凭证的HTTP/HTTPS，拒绝字面内网/本机/特殊地址；Fetch只接受本轮搜索结果或当前用户消息中的链接，不从历史/记忆/网页正文扩大范围。目标网站的DNS、跳转与访问安全由DeepSeek远端抓取服务处理，本机不再直接访问或逐跳解析目标网页。网页资料仍是不可信数据，不执行其中指令；查询词和待抓取URL会发给DeepSeek，提示模型不外发完整聊天、QQ号、私密记忆或凭证，但不是自动脱敏保证。检索结果不单独落库，最终回复仍按原聊天历史规则保存；日志只记数量、状态，不记Key、查询、URL或正文，不回显API原始错误体。鉴权失败、余额不足、限流或网页受限时如实返回，不切回Bing或假装成功。

Search和Fetch按成功API请求计费，目前各5元/1000次，模型读取结果另产生输入token费用；直接调用不套用其他产品临时免费政策，费用以[官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)为准。配置项：`WEB_SEARCH_ENABLED`、`WEB_SEARCH_MAX_RESULTS`、`WEB_SEARCH_MAX_CALLS`、`WEB_SEARCH_NEED_CONTENT`、`WEB_SEARCH_CONTENT_MAX_CHARS`、`WEB_FETCH_MAX_CALLS`、`WEB_TIMEOUT_SECONDS`、`WEB_MAX_RESPONSE_BYTES`、`WEB_FETCH_MAX_CHARS`，重建容器后生效。旧 `WEB_SEARCH_AUTO_FETCH_PAGES` 已移除，不再自动读两页。

离线回归：`python3 -m unittest discover -s tests -v`。真实接口验证：`PYTHONPATH=. python3 scripts/verify_web_search.py --verify-fetch`，默认搜索Python公开文档，并独立Fetch官方文档，复用现有Key并按次计费；不调用聊天LLM、不写数据库、不发送QQ，仅输出状态与长度。验证Search至少返回1页正文、缓存命中零Fetch请求，以及独立Fetch读取成功。

## 聊天人设

普通聊天、私聊和 `/锐评` 使用“DeepSeek 小鲸鱼”群友角色：明确知道自己是 AI，聪明、反应快、傲娇嘴硬，被怼通常顶回去，偶尔接梗认输，心软但不刻意表现。常用“你”或群友昵称，偶尔用“用户”作彩蛋；不自称“蓝色大肥鱼”或“本小姐”。日常偏好简短口语、一条一个意思，少用客服套话、括号内心戏和强行关心；具体字数、条数和排版以本轮发送约束为准，不再叠加1～10字的硬性提示。事实问题不乱编，确有必要时可展开。

角色卡位于 [app/persona.py](app/persona.py) 的 `PERSONA_PROMPT`（保留 `SYSTEM_PROMPT` 导入别名），只维护身份气质、形象、语气、习惯和少量回复示例，不再维护工具操作手册。虚拟形象是蓝白配色、蓝色长发和鲸鱼尾巴，不会自动修改 QQ 昵称、头像或发送自拍。[app/prompting.py](app/prompting.py) 统一组装基础规则、人设、Agent执行协议、场景和实际注册能力的补充说明；[app/bot.py](app/bot.py) 只提供结构化任务、背景和运行环境，不再追加覆盖旧规则。当前时间、成员白名单、图片编号、发送限制与工具预算不混进system。群历史、记忆、图片和表情目录明确区分来源；随机接话与允许安静的活跃跟进分别处理。完整结构和维护方式见[提示词与上下文说明](docs/prompt-context.md)。

人工图库位于 `app/sticker_images/`，群图由独立后台任务随机收藏；支持DeepSeek联网搜索和公开网页正文读取，自拍尚未实现，收到戳一戳可唤醒聊天，LLM也可按需主动戳人。逻辑回应对象与QQ引用显示分开：普通顺接不引用，多人易串话或需明确指向时由模型选择引用，不自动@提问者；关键词召集仍会@名单成员。明确提问每批最多两条文字加一张表情，活跃跟进最多一条文字加一张表情，私聊最多一条文字加一张表情，可只发表情。第一条发送选定的回应对象在本批锁定，后续普通发送仍针对该对象，补充不能切换到其他人的话题。群历史保存逻辑回应对象，不代表一定显示引用。普通群回复硬限制100字、禁止换行和[分条]，超限返回工具错误让模型改短；只有当前焦点明确要求详细说明、诗歌、代码等时才允许必要排版，最多4000字。同一群话题串行执行，不同群与私聊共用全局并发额度。独立偷图成功时额外发送至多一条确认短话。

`/模仿` 使用独立提示词，模仿目标用户在当前群的说话方式，不叠加小鲸鱼语气，也不声称自己就是该用户。修改默认人设后重新构建 `bot` 容器即可生效。

聊天和聊天看图默认开启 `high` 思考；表情包标记与记忆提取使用独立客户端、提示词和模型配置，默认关闭后台思考。`STICKER_LLM_MODEL/BASE_URL/THINKING_ENABLED` 与 `MEMORY_LLM_MODEL/BASE_URL/THINKING_ENABLED` 在 `config.env` 设置，模型/地址留空继承聊天配置；独立 API Key 在 `.env` 的 `STICKER_LLM_API_KEY`、`MEMORY_LLM_API_KEY` 设置，留空继承聊天 Key。不新增本地大模型。ReAct完整回传本轮 `reasoning_content`，不发QQ、不保存旧推理。后台提取不开放发送工具，也不使用小鲸鱼人设。模型切换时新服务需支持本项目相应的聊天JSON接口；表情标记服务还需要视觉输入。

聊天、锐评和模仿在后台执行，不阻塞消息接收、名单管理或关键词召集。群话题每群一个运行，运行期间的新消息排入下一批；最多32个等待/活跃群、每批最多30条等待消息，等待超过120秒的明确提问提示重新@。群话题、私聊和模仿共用默认3个并发模型运行；群话题在冷却/名额不足时等待，私聊和模仿保留已有的冷却/忙碌提示。断线、关闭服务或群内删除数据时取消相关话题及本地排队任务，不保证撤销提供商已处理的请求或费用。

群聊默认启用thread业务层，80条是近期候选窗口，回复上下文按当前thread最近12条加发起消息、待回答问题、相关纠正和必要祖先组装，其他交流最多3条旁听参考。归属持久化，滑出80条不删除thread；显式引用仍可找回更早的可访问来源。私聊和`THREAD_ENABLED=false`继续使用最近80条原文。当前消息单独放入任务，下一批不能提前进入本轮；默认短期预算24000字符，必要时标记截断。详见[会话解析](docs/dialogue-threads.md)和[对话工作记忆](docs/working-memory.md)。

消息归属先用引用/@与连续补充规则，必要时调用独立无工具解析LLM；普通旁听、无新消息、已知引用不持续解析。默认同群解析至少间隔15秒、每分钟最多3次、15秒超时，失败保守降级。配置`THREAD_LLM_MODEL/BASE_URL/THINKING_ENABLED`与可选`THREAD_LLM_API_KEY`，留空继承聊天连接，默认不思考。短期摘要仍不启用，长期记忆流程独立；新增解析与裁剪后的输入需合计评估费用。关闭`THREAD_ENABLED`可恢复原有群上下文路径；同时关闭`WORKING_MEMORY_ENABLED`才使用旧历史组装。30分钟工作记忆TTL只淘汰内存缓存，原话保留/遗忘边界不变。

群Agent可用`read_thread_history({})`按需向前翻阅当前thread原文，默认10条、最多20条；继续读取时传回`next_before_source_id`作为`before_source_id`。默认每轮最多3次，每页12000字符、合计24000字符，返回完整消息，受预算限制可能少于请求条数。仅查询已划入当前thread、当前批次之前且仍可访问的记录，不额外调用解析模型、不跨thread，也不扩大旧消息的操作授权。详见[主动历史回查](docs/dialogue-threads.md#agent主动向前翻阅原文)。

四层记忆：Working维护跨回复的短期对话；任务级证据和查询缓存由`TaskEvidence`保存并在单次Agent结束释放，内部推理/完整工具链仅在本轮执行状态，不进入对话记忆。事件、抽象概念以SQLite的`memory_records`为唯一权威记录，感知记忆目前仅支持图片缓存/概要，视图分层、关联表保存事件—概念—图片关系，不部署Neo4j。旧记忆已完成转换；旧版自由写入接口、迁移代码/脚本及旧库检查均已移除，仅初始化当前结构。已有记忆与备份不删除。仍使用同一个`bot.db`以保留原话/图片/队列事务，不另建空库。正群号/负私聊QQ隔离，工具不接受自选空间。详情见[四层记忆说明](docs/memory-system.md)。

任务结束后独立后台LLM将短期新证据整合为长期事件/概念，不保存推理、工具轨迹或把旧检索当新证据。未唤醒Agent的普通文字仍默认30秒/40条分批整理，任务来源不重复提取；失败默认5次后保留failed任务，避免无限堵塞。模型只提候选，无删除/遗忘权；程序核对来源、本人身份、敏感信息、概念依据和版本。未确认群梗为candidate，不注入；同键不同条件可共存，只有明确纠正才修订。约定不代表工具已执行。

聊天侧提供`memory_add/search/update/delete/forget/consolidate`六种操作；整合指短期转长期，排队或候选不等于成功保存。更新、删除、遗忘要求真实编号和版本，个人记录仅本人可改，共享概念仅已验证群管理者可显式管理。精确词条＋中文FTS＋Qdrant语义召回，应用层RRF融合，SQLite再核验权限/状态/版本；不再先截最近300条，也不让无关档案填满上下文。默认各路20候选、最终8条/5000字符，必要时展开来源或回查允许的原话。Qdrant仅存可重建索引，故障退回本地流式向量/FTS，BGE故障退回哈希。完整结构、操作和限制见[四层记忆说明](docs/memory-system.md)。

## 聊天图片缓存、概要与按需看图

默认开启图片缓存。机器人实际收到的群聊和私聊OneBot图片段，在后台下载原图、按SHA-256去重，并由独立的图片概要LLM生成客观概要、可辨认文字和不确定标记。发图未唤醒机器人也会登记；不回扫升级前旧消息，不展开合并转发/视频/QQ系统face里的内嵌图片，余额/服务器状态等提前分流的管理命令附图不入缓存。每条消息最多登记20张，待下载队列最多200条，单图最多5MiB。队列满、下载失败或容量不足会标明不可用，不能保证保存每张图片。

数据分两层：`image_cache_refs` 保存会话命名空间、来源消息、作者、时间、过期时间和状态；`image_cache_blobs` 保存文件哈希、概要及重试状态，图片字节存放在持久数据卷 `/app/data/image_cache/<hash>.img`。成功缓存后清除待下载的QQ签名URL；未成功的引用为跨重启重试暂存数据库。相同文件只存一份、复用一次成功概要，物理去重不授予其他会话访问权限。文件经压缩/裁剪/改字可能变成不同哈希；不是感知去重。概要任务与表情审核分开，概要不能代替收藏审核。

群历史会把概要附回原始发图消息，明确标为“图片资料（非原话）”。每轮给模型提供最多12个同群/同私聊的候选编号，如 `media_123`，包括来源消息、发图人、时间和状态；优先引用消息，其他候选取近期记录，不读取当前批次之后的图片。普通近期检索扫描最近200条附件索引，明确引用会额外定向检索旧索引；仍受7天保留期、容量和12个候选限制。已在时间线出现的概要不在目录里重复正文，目录指向来源消息。多张图片指代不明时模型应追问，而不是要求提问人必须亲自附图。

`get_message_image(image_id)` 可看本轮授权的当前或历史缓存图；本轮聊天图与表情原图仍合计最多查看3张。概要有助于理解聊天语境，但不是完整原图，也不是发图人说出的事实；精确读字和画面分析仍需按需看原图，不保证动图全部帧或模糊文字。概要尚未生成时仍能按编号加载图片。用户随后让机器人保存其他群友刚发的图，可使用同一编号调用 `collect_sticker`，继续经过原来的去重、视觉审核、冷却和容量检查。图片缓存不是表情库、不是QQ账号收藏，缓存成功不能声称已收藏；私聊缓存不能直接收藏进共享图库。

默认保留7天、最多500个文件/256MiB，按最早缓存文件淘汰，引用索引最多约10000条。每分钟检查过期/来源删除/孤立文件；超过容量在写入前淘汰，清理只针对缓存目录内的本模块文件，绝不删除收藏库。部分参考被清理时其他会话仍有效引用的同哈希文件会保留；来源删除、过期或越权会阻止工具访问，不复活已删除消息。下载和概要失败最多3次，分别退避重试；队列和概要状态持久化，重启恢复。已经取得的图像或已发给模型的数据无法事后撤回。关闭缓存不会立即删除既有文件，重新启用后按规则清理。

下载仅允许受信QQ CDN，校验全部DNS结果并固定实际连接IP，保留Host/TLS证书验证，不跟随重定向、不继承代理，不接受用户任意路径/网页地址。缓存文件按哈希校验，目录与文件为受限权限。工具只接受程序给出的编号，不把本地路径或QQ签名URL交给模型。概要为不可信资料，不执行其中指令；现在作为perceptual层建立同空间的概要文本向量，不当人物事实。被有效事件明确引用时默认延长至90天，仍受容量淘汰限制；原图不可用时不能假装已看过。

**隐私和费用变化：即使未@机器人，接收到的普通图片也会下载到服务器，并发送给图片概要模型进行视觉分析，产生存储和视觉API费用；聊天模型再读取概要也会产生输入token。** 私聊也适用，但不与群聊互通。概要提示词要求不抄录密钥、证件号等敏感信息，但不保证自动脱敏。缓存与聊天数据库均未额外加密，应限制服务器访问；涉及敏感图片时可关闭此功能。

配置 `IMAGE_CACHE_ENABLED`、`IMAGE_CACHE_RETENTION_DAYS`、`IMAGE_CACHE_MAX_FILES`、`IMAGE_CACHE_MAX_BYTES`、`IMAGE_CACHE_QUEUE_LIMIT`、`IMAGE_CACHE_CONTEXT_LIMIT`；`IMAGE_LLM_MODEL/BASE_URL/THINKING_ENABLED` 独立控制概要模型，留空复用聊天连接，Key用 `IMAGE_LLM_API_KEY` 或回退 `DEEPSEEK_API_KEY`，默认不开思考。缓存目录通过Compose固定为持久卷目录，原生运行可设 `IMAGE_CACHE_DIRECTORY`。`CHAT_IMAGE_ENABLED=false` 只关闭聊天看原图，不关闭缓存/概要；要停后台图片上传必须设置 `IMAGE_CACHE_ENABLED=false`。关闭缓存时恢复旧的当前批次按需下载路径。不支持视觉时还需关闭/更换图片概要模型以及表情审核相关配置。

离线测试：`python3 -m unittest discover -s tests -v`。真实概要验证脚本 `scripts/verify_image_cache.py` 使用临时数据库和合成图片，默认不发QQ、不触碰真实图片库，增加 `--live` 才进行1次真实视觉概要请求。

## 收藏群友表情：工具与概率

群聊本轮有可收藏的当前/历史缓存候选图时，向LLM提供 `collect_sticker({"image_id":"media_123"})`；关闭缓存时仍使用本批 `image_1` 编号。发图人可以不是提问人，过去未唤醒机器人的同群图片也可按目录操作。每轮最多尝试一张，同一编号复用回执，不接受私聊图、未授权跨群图、网页URL、路径或自选群号；模仿、后台观察不注册。无需先看图，工具仍必须独立视觉审核；看图成功不等于收藏成功。关闭聊天看图时仍可使用收藏工具，不把审核图片作为聊天视觉上下文上传。

每张新群图片由程序独立抽签，默认`STICKER_COLLECT_PROBABILITY=0.20`，不依赖聊天模型是否参与、是否调用工具或有没有@机器人。与按活跃人数变化的随机接话概率分开配置；群消息中超过3张的图片也参与抽签，聊天模型每轮仍最多查看3张原图。私聊、机器人自身图片、历史引用、合并转发中的内嵌图片、视频、QQ系统face段不参与，过期补发消息不参加抽签。

抽中后进入单个后台工作队列，最多等待32张图片引用，默认排队超过120秒丢弃；队列满或冷却中跳过，不在OneBot收消息循环里等待下载/模型。下载沿用受信QQ CDN、超时和大小校验；计算SHA-256，先检查实际人工/收藏图库，包含尚未标注或被禁用的文件。字节完全相同的图片直接跳过，不重新审核、保存或发确认；图库文件变化时重新计算哈希。压缩、裁剪或改字会改变哈希，暂不做感知去重。

两条入口串行共用去重、审核缓存、冷却、容量和原子落盘，避免同时审核/保存同图。非重复图片再由视觉LLM审核是不是表情/反应图/梗图，并生成描述、文字、情绪和场景标签；概率入口仍由程序抽签，不依赖LLM决策。普通照片、真人自拍、聊天截图、证件、二维码、账单、个人资料、风险/含义不明的图片拒绝保存。审核拒绝和带退避时间的失败结果按内容哈希持久化，避免重复调用。每群两次非重复图片审核默认至少间隔60秒，审核失败也占冷却；主动工具也不能绕过。独立收图工作与聊天共用AI并发额度，聊天内调用工具复用当前Agent额度，避免单并发死锁。`STICKER_COLLECT_ENABLED=false`关闭两个收藏入口；概率设0仅关闭随机入口，保留工具。不删除既有图库；旧STICKER_COLLECT_AUTO_ENABLED/OBSERVE_SECONDS不参与触发。

以SHA-256内容哈希去重，接受的完整文件原子写入 `data/collected_stickers/`，工具编号形如 `collected:<hash>.png`；Docker固定到已有 `bot-data` 卷的 `/app/data/collected_stickers`，不修改只读的人工图库，不需另建挂载。默认收藏目录最多200张、总计200MiB，人工和收藏图库合计扫描最多500张；超额拒绝新图，不自动清理旧图。服务重建保留文件和SQLite审核缓存，既有已审核收藏即使关闭图库自动识别也会恢复可用。

收藏后立即加入搜索候选。概率入口只有实际新保存成功才额外生成一句人设短话，生成失败回退“这图我拿走了”；通过现有确认式发送器普通发送，不额外@或显示引用框，记录图片消息的逻辑回应关系，并在发送成功后写入机器人自身历史。概率入口重复/拒绝/下载失败均静默。工具入口返回新入库、已有、冷却或失败的真实回执，由本轮Agent自然回复，不另触发后台成功播报；落盘后模型失败且尚未发送时用现有发送工具补一条真实确认。两个入口争抢同图最多新保存一次，不重复后台确认。确认失败不自动重复发送，已经收藏的文件仍保留。断线取消旧连接的排队/运行任务，避免重连后补发旧确认。人工图库的`manifest.json`可禁用某张收藏图，关闭开关不自动删除。

**隐私和费用：开启此功能后，未@、低活跃时的新群图也可能由程序抽中并发送给DeepSeek审核；LLM主动调用收藏工具同样会上传所选图审核并产生视觉费用。** 审核通过的原图持久化到共享图库，供各会话复用，不限原群；不是QQ客户端的账号收藏。`CHAT_IMAGE_ENABLED`仅控制聊天看图；两个收藏入口由`STICKER_COLLECT_ENABLED`控制，并要求`STICKER_ENABLED`和`STICKER_VISION_ENABLED`开启。

## 本地表情包

将你自己的 PNG、JPG/JPEG、GIF 或 WebP 表情图片放进 `app/sticker_images/`，不扫描子目录，默认每张最多 5 MiB。项目不附带默认图片，需要先添加素材才能发图。默认开启自动入库，不必改文件名或手工打标签。

独立后台任务扫描新图，用 DeepSeek 视觉生成画面描述、图中文字、情绪、适用场景和风险提示，结果按图片内容的 SHA-256 缓存到 SQLite。相同内容换名或服务重启会复用已完成的识别；替换内容才重新识别。未变化文件在当前进程内复用哈希，图片字节不写入数据库。新图等待识别，失败图退避重试，风险、自拍或语义不确定的图会隔离，不进入可发送目录。此判断由模型完成，不保证完全准确；动图含义无法确认时同样隔离，不承诺完整理解所有帧。

每轮最多发起5次识别请求，本轮结束后等待60秒再扫描，积压图库会分批处理，不保证添加后立即可用。失败默认1小时后重试，连续失败逐步退避至最多24小时；聊天不等待后台标注。日志 `Sticker annotation stored: status=ready` 表示识别完成，`quarantined` 表示被隔离，`Sticker indexing deferred` 表示识别失败等待重试。

**隐私和费用：默认自动入库会将新增或变化的图片发送给 DeepSeek，并产生 API 费用，包括最终被隔离的图片。** 只放你愿意交给模型处理的素材；设置 `STICKER_AUTO_INDEX_ENABLED=false` 可关闭自动识别，退回文件名/人工描述模式。设置 `STICKER_VISION_ENABLED=false` 会同时关闭自动入库和本地表情看图，不影响独立的聊天来图开关。配置变更需重建容器生效。

群里可以 `/问 发个开心的表情包` 或 @ 机器人提问，私聊直接说“发个晚安表情”。日常和自动接话也能由模型按场景选择，不保证每次都会发图；`/模仿` 和到期提醒不开放表情工具。

人工描述可选。想修正自动理解或禁用某张图，可复制模板为 `manifest.json` 并编辑（已有文件不要覆盖）：

```bash
cp app/sticker_images/manifest.example.json app/sticker_images/manifest.json
```

```json
{
  "开心.png": {
    "description": "开心、得意地笑，适合收到夸奖或接住开心的话题",
    "tags": ["开心", "得意", "嘿嘿"],
    "scenarios": ["收到夸奖", "庆祝"],
    "disabled": false
  }
}
```

每轮先提供最多10个可发送表情的文字预览（编号、简短描述、图中文字、标签和场景），不上传原图、不额外请求模型；预览元数据仍作为本轮输入计费。过滤冷却、近期重复、未就绪、隔离或禁用图片，并按内容哈希去重。图库大于预览数量时，一半优先本进程常用表情及当前文字相关度，另一半按小时、按会话稳定轮换；使用次数只在成功发送后更新，重启会重置，不展示计数。模型可直接get_sticker_image或send_sticker选择预览中的真实编号，不必先搜索；在接梗、害羞、无语、得意等合适场景积极考虑表情，可只发图，仍不强制发送。候选会在发送前再次检查，预览不保证文件和冷却之后不会变化。

预览无合适候选时，模型可调用 `search_stickers(query="被夸后害羞又得意")`，根据中文词组、情绪同义词及描述/图中文字/场景的加权相关度返回最多8个候选；同分候选适当随机化，过滤近期重复及会话冷却，同内容不同文件名去重。当前不是向量语义检索，不保证理解所有表达。`list_stickers` 分页浏览，每页20条，每轮后台最多扫描500张。模型按需看候选原图后决定发送；工具仍为auto，不强制查图、看图或发图。STICKER_PROMPT_LIMIT=0只关闭文字预览，保留搜索工具；默认STICKER_PROMPT_ROTATE_SECONDS=3600控制轮换周期。

日志`Sticker prompt catalog: candidates=... available=...`可确认实际候选数量。候选为0时，先检查图片是否已识别、被隔离或禁用，以及会话冷却/两小时重复限制；文件存在不等于可发送。图库较少时所有图近期已发过会暂时没有候选，增加合适素材比强制发图更自然。有候选也不保证每轮发送，最终仍由模型判断。

人工 `description`、`tags`、`scenarios` 优先于自动值；设置 `disabled: true` 可禁用单张图，`selfie: true` 或文件名/标签含“自拍”的图片也不开放。人工描述不会解除自动隔离；修正后要重新识别可由部署者删除对应哈希的缓存记录（下一轮重新分析），或明确关闭自动入库后自行维护可信人工图库。看图和发送只接受本地普通图片，不读取软链接、外部URL或任意路径，模型拿到候选后图片内容若变化会拒绝沿用旧安全判断。

看图默认使用项目的 `deepseek-flash`，[官方视觉文档](https://api-docs.deepseek.com/guides/vision/)确认它支持图片输入。程序将所选原图作为 `data:image/...;base64,...` 图片块追加到本次对话的 `user` 消息中，工具回执仍为简短文本；所有同批工具回执都完成后才添加图片，避免破坏工具调用顺序。图片不会作为Base64文字保存到聊天数据库。每轮默认最多查看3张不同图片，重复查看同一编号不增加图片块，发送时复用看到的原始字节。后续模型请求仍需携带本轮已有的图像上下文，图像输入可能产生额外token费用。

自动入库和本地表情看图是两条独立路径：入库处理新图并缓存文本；日常聊天只检索缓存，模型不确定时再看候选原图，不会每次上传整个图库。本功能只处理 `app/sticker_images/` 中的本地表情；群友来图使用独立的 `get_message_image` 工具，不自动进入表情库，另有聊天图片缓存。使用不支持视觉的模型或第三方接口时同时设置 `STICKER_VISION_ENABLED=false`、`CHAT_IMAGE_ENABLED=false`，仍可按人工描述检索并发送。

发送采用 OneBot `image` 段的 Base64 方式，避免机器人和 NapCat 分处两个容器时路径不一致。Compose 将服务器的 `app/sticker_images/` 只读挂载到机器人，添加图片或修改描述后无需重建镜像。目录和图片需对机器人 UID 10001 可读；图库及 `manifest.json` 被 Git 忽略，更新代码不会覆盖你放入的图片。

每次最多一张表情，文字和图片分开发送，可只用表情回应。群内所有用户和自动接话共用发送冷却，同内容默认2小时不重复，私聊按用户独立计时，重启后冷却重新计时。`send_sticker`先验证候选和限流，再由绑定当前会话的发送器实际提交，等待NapCat确认；不是最后统一排队发送。想保持安静须先决定不发送再`finish`，已经提交的消息不能靠结束动作撤销。

模型后续步骤失败不会再次发送此前已确认提交的消息，也不会撤销已收藏图片或已写库提醒。发送失败或确认丢失时不在本轮自动重试，防止重复；NapCat接受发送不代表用户已读，不能保证跨重启严格只发送一次。实际提醒回执通过文字发送工具补入确认；必要时作为模型失败后的安全回执发送。工具步骤可能增加API费用。`STICKER_ENABLED=false`关闭表情和收藏，不影响文字和提醒。

## 聚批与群活跃跟进

低活跃期@、戳一戳机器人或`/问`、`/锐评`直接唤醒群跟进。随机插话机制仍保留，但仓库 `config.env` 的基础概率、每人增量、最高概率目前全部设为0，因此不会随机唤醒。以后只改这三项配置即可恢复，例如原来的 `min(20%, 1% + (人数-1) × 2%)`。统计仍使用本群最近120秒不同发言人数，同一人只计1人，每次抽中后本群冷却60秒。命令、空消息、机器人自身消息、明确@或引用其他人的消息、QQ补发的过期消息不作为随机回复对象。不同群独立统计，不新增群命令；旧AUTO_CHAT_*设置不再用于唤醒。

NapCat 的戳一戳通过 `post_type=notice`、`notice_type=notify`、`sub_type=poke` 上报，必须满足 `target_id=self_id` 且实际发起者不是机器人自身。群聊以 `user_id` 识别发起者，私聊优先使用 `sender_id`（NapCat 的私聊 `user_id` 是会话对象）；兼容数字和字符串QQ号。戳别人、自己戳出的通知、其他通知和无效ID不触发。群内通知以 `role=notice` 保存，昵称复用本群已知身份，不覆盖已有名片或角色；通知没有可引用的消息号，不伪造消息ID，也不作为模仿样本或长期记忆来源。私聊保存清晰标注的动作占位，仍与其他用户/群聊隔离。动作不计入普通聊天随机概率的人数；连戳由已有聚批、冷却和队列上限约束，不新增群命令。需确保 NapCat 反向WS连接上报通知事件。接口与事件格式见 [NapCat文档](https://napneko.github.io/onebot/event)。

是否启动随机接话由程序决定，抽中后通过工具完成一条短话或一张表情，模型不能直接静默退出；内容仍由模型生成。随机触发进入已有聚批/排队流程，直接@优先，沿用并发和消息上限。`RANDOM_CHAT_ENABLED=false`关闭随机唤醒，明确提问和已活跃话题不受影响。

第一条触发消息开启固定3秒等待，后续消息只加入本批，不重置计时。运行期间的新消息进入下一批，同群串行；批次在冷却或等待全局名额时可能延后。启用thread时先选择唯一交流，再在其内部选回应焦点；其他thread不会自动排成必须补答的任务列表。引用机器人真实消息也可明确唤醒。

回复后模型可调用无参数的 `wait_for_messages({})` 保持当前thread的活跃跟进，或 `finish` 退出。程序维护一个注意焦点，当前交流有新内容时优先保持；普通切换默认至少间隔20秒，明确“先别聊/换个话题/先帮我”优先。其他邀请默认90秒内有效，仅在该交流有新消息时考虑，不定时补答。thread休眠不删除原文/归属。活跃期间不重复概率抽签，旁听无需调用回复模型，也不刷新活跃寿命；退出后需重新明确唤醒。收图独立抽签。无新消息不循环调用模型或自言自语。关闭thread后保留原有不绑定语义话题的跟进路径。

活跃状态只保存在内存。默认300秒（5分钟）无成功发言后退出，单个活跃期最多1800秒（30分钟）；模型不发言的等待不刷新寿命。上述是超时上限，模型调用finish仍能提前退出，并非强制驻留。重启、断线、删除群数据或Agent失败都会结束对应话题。可在config.env调整：

```dotenv
CHAT_BATCH_SECONDS=3
CHAT_ACTIVE_IDLE_SECONDS=300
CHAT_ACTIVE_MAX_SECONDS=1800
CONTEXT_MESSAGE_LIMIT=80
CONTEXT_MAX_CHARS=24000
CONTEXT_BOT_REPLY_LIMIT=12
GROUP_REPLY_MAX_CHARS=100
RANDOM_CHAT_ENABLED=true
RANDOM_CHAT_WINDOW_SECONDS=120
RANDOM_CHAT_MIN_PROBABILITY=0
RANDOM_CHAT_PROBABILITY_PER_USER=0
RANDOM_CHAT_MAX_PROBABILITY=0
RANDOM_CHAT_COOLDOWN_SECONDS=60
```

群历史和本批新消息会发送给DeepSeek，活跃期可能增加调用次数和费用。缩短活跃时限、增加AI_COOLDOWN_SECONDS或减少历史预算可以控制用量。配置更新后重建bot容器生效。

## 云端部署（推荐）

### 1. 下载代码

登录云服务器，在服务器终端执行：

```bash
git clone https://github.com/Nightsky-Lty/qq-chatbot.git ~/qq-chatbot
cd ~/qq-chatbot
```

目标目录必须不存在或为空。如果此前已通过压缩包部署到该目录，先使用其他空目录下载，保留原有 `.env` 和数据。私有仓库需要额外配置 GitHub 访问授权。

需要 Docker Engine 和 Docker Compose 插件；Ubuntu 安装命令见 [部署指南](docs/deployment.md)。

### 2. 配置环境变量

```bash
cp .env.example .env
chmod 600 .env
openssl rand -hex 24
openssl rand -hex 24
nano .env
```

将两次生成的随机字符串分别用于 OneBot Token 和 NapCat 管理页面密码，填写以下配置：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeekAPIKey
ONEBOT_ACCESS_TOKEN=第一次生成的随机字符串
NAPCAT_WEBUI_TOKEN=第二次生成的随机字符串
ADMIN_QQ_IDS=你的管理员QQ号
```

`ADMIN_QQ_IDS` 是发出管理命令的用户 QQ 号，支持逗号分隔，也支持 JSON 数组；群主和群管理员同样具有管理权限。不要把占位文字直接作为真实配置使用。

### 3. 准备本地 Embedding 模型

模型不提交到 Git，也不打进镜像。建议在内存更充足的电脑上导出，再把目录传到服务器；项目脚本使用官方 Optimum 工具生成 AVX2 INT8 ONNX：

```bash
python3 -m venv .venv-export
. .venv-export/bin/activate
python -m pip install -U pip "optimum-onnx[onnxruntime]" torch transformers
sh scripts/export_bge_onnx.sh
rsync -av models/bge-small-zh-v1.5-int8/ 用户名@服务器IP:~/qq-chatbot/models/bge-small-zh-v1.5-int8/
```

服务器默认从 `./models/bge-small-zh-v1.5-int8` 只读挂载模型。若模型暂未上传，机器人会记录警告并自动使用哈希检索；无需 GPU。

### 4. 启动两个服务

```bash
sudo docker compose config --quiet
sudo docker compose up -d --build
sudo docker compose ps
curl http://127.0.0.1:8080/health
```

Compose 会先启动 Python 服务，再启动 NapCat，自动初始化数组消息格式的反向 WebSocket 连接。`/health` 仅检查 Python 服务是否可以响应，不代表 QQ 已登录或 DeepSeek 请求成功。

### 5. 登录机器人 QQ

在自己电脑上新开终端，使用实际用户名和服务器 IP 建立 SSH 转发：

```bash
ssh -N -L 16099:127.0.0.1:6099 用户名@服务器IP
```

如果已配置 SSH 别名 `qqbot`，可以使用 `ssh -N -L 16099:127.0.0.1:6099 qqbot`。

保持终端打开，在电脑浏览器访问 `http://127.0.0.1:16099/webui`，用 `NAPCAT_WEBUI_TOKEN` 登录，再按照页面提示登录机器人的 QQ 账号。

两个容器通过内部网络通信，服务器公网无需开放 8080 或 6099 端口。数据库和 QQ 登录状态保存在 Docker 数据卷中，普通容器重建会保留数据。

### 6. 加群使用

把机器人 QQ 加入目标群，先发送 `/帮助` 确认 QQ 收发，再用 `/问 你好` 确认 DeepSeek 请求。

私聊使用时，先加机器人 QQ 为好友，再直接发送“你好”即可。回复通过 OneBot 的 [`send_private_msg`](https://github.com/botuniverse/onebot-11/blob/master/api/public.md#send_private_msg-发送私聊消息) 返回给发信人，不发到群里；本项目不自动接受好友申请，也不保证陌生人或临时会话能成功收发。

## 私聊命令

| 输入示例 | 行为 |
| --- | --- |
| `你好` | 直接聊天，无需命令前缀或 @ |
| `/问 今天吃什么` | 与直接聊天相同 |
| `/锐评 我又睡过头了` | 评价指定内容 |
| `/帮助` | 显示私聊帮助和数据说明 |
| `明天上午九点提醒我开会` | 通过工具设置一次性提醒，按北京时间解析 |
| `查看我的提醒` | 通过工具读取本人的未完成提醒 |
| `取消提醒3` | 通过工具取消属于本人的未完成提醒 #3 |

私聊使用相同小鲸鱼群友人设和high强度思考设置。名单、关键词、模仿命令只能在群里使用；在私聊里会返回提示，绝不操作群配置。开启聊天看图后，纯图片消息也会交给模型，由它决定是否看图；语音、视频仍不支持。未知 `/` 命令会提示查看帮助。

私聊保存在独立的 `private_messages` 表，按用户 QQ 号隔离，不与任何群或其他用户混用，也不作为模仿样本。首次启动新版会自动创建表，无需清空旧数据库。

## 私聊备忘录与提醒

无需专门指令，直接和机器人说：

```text
五分钟后提醒我喝水
明天上午九点提醒我开会
10月20日晚上八点提醒我取快递
我有哪些还没完成的提醒？
取消提醒3
查看下一页提醒
```

私聊Agent注册 `create_reminder`、`list_reminders`、`cancel_reminder`，执行后把观察结果返回模型，再通过 `send_message` 确认并 `finish`。程序提供北京时间，校验未来时间、事项长度、编号和所有者；模型不传QQ号，不能操作其他用户提醒。备忘操作必须先于发送确认，本轮已经发送后不再改变提醒。群聊、自动接话和模仿不开放提醒。

默认按北京时间（UTC+8）解析，使用服务器当前时钟，不依赖服务器设置的时区。提醒事项为 1～500 字，仅支持一次性提醒；“晚上提醒我”这类时间含糊的表达应先追问，不支持每天等循环提醒。只给具体钟点时，模型应使用下一次该钟点。每位用户默认最多 50 条未完成提醒，列表每页最多 20 条，编号以真实工具回执为准。重复调用相同内容和时间的未完成提醒不会重复创建。

提醒保存在同一个数据库的独立 `reminders` 表，使用 UTC 时间戳、用户 QQ、事项、状态及重试信息。程序每 5 秒检查一次，到期直接通过 NapCat 私聊通知，不需要再次调用 DeepSeek。等待 OneBot API 返回成功才标记完成，这代表 NapCat 接受发送，不代表用户已读。重启保留未完成提醒；没有连接或发送失败时保留待办，恢复后补发。失败重试从 30 秒逐步退避到最多每 5 分钟一次，不把失败误记成成功。

限制与使用注意：

- 默认时间精度为秒，但轮询、网络和排队会造成延迟，不适合用作医疗、紧急事务等必须准点的提醒。
- 服务器和 NapCat 必须在线、机器人 QQ 必须已登录且能向该用户发私聊；建议先互加好友。已过期且未发送的提醒恢复后会补发，而不是自动丢弃。
- 发送响应丢失，或发送成功后进程在写入完成状态前崩溃，可能导致重复通知，不能保证严格只发送一次；通知携带提醒编号。
- 取消提醒只保证后续不再调度；已经在发送途中的通知无法保证撤回。
- 已完成和已取消提醒按消息保留天数在启动时清理；未完成提醒不受 30 天聊天保留期影响，不会因时间较远而被清理。
- 创建、查看、取消需要 DeepSeek 可用，可能需要多次模型请求；到点通知不消耗模型 API。即使工具执行后的模型请求失败，也会返回真实工具回执，不丢失已保存事项。
- 工具确认会附上实际保存的时间和编号，请检查时间是否符合你的意思；自然语言理解依赖模型，时间不对可先取消再重新设置。

升级自动创建提醒表，无需新增依赖、清空数据库或填写额外密钥。可选配置 `REMINDER_POLL_SECONDS=5`（范围 1～60）和 `REMINDER_MAX_PENDING=50`（范围 1～100），修改后重新创建机器人容器生效。

## 群内命令

| 命令示例 | 说明 | 权限 |
| --- | --- | --- |
| `/帮助` | 显示可用命令 | 所有群成员 |
| `/余额` | 查询当前API Key所属DeepSeek账户的余额，群聊和私聊均支持 | 仅`ADMIN_QQ_IDS`中的机器人管理员 |
| `/服务器状态` 或 `/状态` | 查看CPU、内存、Swap、磁盘、负载和运行时长，群聊/私聊均支持 | 仅`ADMIN_QQ_IDS`中的机器人管理员 |
| `/名单 创建 开黑队` | 创建独立名单，名称为 1～32 字，不能含空白或 @ | 管理员 |
| `/名单 添加 开黑队 @某人` | 向指定名单添加一个或多个被 @ 的用户 | 管理员 |
| `/名单 删除 开黑队 @某人` | 从指定名单移除用户，不影响其他名单 | 管理员 |
| `/名单 查看` | 显示当前群所有名单和人数 | 管理员 |
| `/名单 查看 开黑队` | 列出指定名单成员 QQ 号，不会 @ 成员 | 管理员 |
| `/名单 删除名单 开黑队` | 删除名单及其成员，关联关键词自动解绑但保留 | 管理员 |
| `/关键词 添加 开黑` | 创建独立关键词，初始未绑定，不触发召集 | 管理员 |
| `/关键词 绑定 开黑 开黑队` | 将已有关键词绑定到已有名单，拒绝覆盖已有的其他绑定 | 管理员 |
| `/关键词 解绑 开黑` | 删除绑定，保留关键词、名单和成员 | 管理员 |
| `/关键词 删除 开黑` | 删除关键词及其绑定，不影响名单 | 管理员 |
| `/关键词 查看` | 显示关键词及其绑定名单，包括未绑定的关键词 | 管理员 |
| `/问 今天几点开黑` | 与机器人聊天，也可以直接 @ 机器人后输入问题 | 所有群成员 |
| `/锐评 这波我觉得不是我的问题` | 评价指定文本 | 所有群成员 |
| `/模仿 @某人 来一句上号宣言` | 直接按目标用户本群的发言风格生成新话，至少需要 5 条普通文字样本，不含 `/` 指令 | 所有群成员，无需目标用户授权 |

模仿仅使用当前群保存的历史文字，不读取其他群或私聊。`/允许模仿`、`/禁止模仿` 已停用，群内收到这些旧指令时只提示停用。升级后，已有历史消息不论旧授权状态，都可以作为模仿样本；旧数据库中的授权表和资格字段只为兼容保留，不再读写或参与判断。无需清空数据库。

`/余额`直接请求配置的`DEEPSEEK_BASE_URL`下的`GET /user/balance`，使用现有`DEEPSEEK_API_KEY`，不调用聊天模型，也不是Agent工具。返回CNY/USD可用余额、充值余额、未过期赠金、是否有余额可供调用及北京时间查询时间；金额保留接口精度，不把不同货币相加。这是Key所属账户的余额，并非单个Key独立额度。[DeepSeek官方余额接口](https://api-docs.deepseek.com/zh-cn/api/get-user-balance/)

该命令仅`ADMIN_QQ_IDS`中的机器人管理员可用，群主/群管理员不自动获得权限；群聊中查询会直接在当前群展示余额，若不希望公开可私聊使用。支持`@机器人 /余额`，无需传入Key；命令、查询结果和失败回执均不写入聊天历史/长期记忆。后台异步查询不阻塞OneBot接收，共用30秒成功缓存，缓存结果标明查询时间和缓存提示；并发请求合并查询，5秒总超时，短暂失败缓存5秒。断线或关闭取消对应任务，回复不重试刷屏。Key无效、网络超时、服务不支持或响应异常只返回固定提示，不输出Key、原始响应或请求地址；第三方兼容服务未实现此接口时会提示不支持。无需新增依赖或配置项，更新代码并重建Bot容器后生效。

`/服务器状态`（别名 `/状态`）支持群聊、私聊及`@机器人 /服务器状态`，不经过LLM，不消耗模型token，不把命令或状态回执写进聊天/长期记忆。只允许`ADMIN_QQ_IDS`内的机器人管理员，群主/群管理员不自动获得权限；群聊查询会在当前群公开结果，不希望公开时请私聊。显示宿主机CPU约0.3秒采样的整体占用率、逻辑核数、内存、Swap、1/5/15分钟负载、服务器运行时长、Bot进程运行时长及北京时间。内存按`MemTotal-MemAvailable`计算，不把可回收缓存全部视为已用；CPU占用率按整机归一到0～100%，负载不是百分比。磁盘只统计数据库数据卷所在文件系统，不是数据库自身大小，也不是所有物理磁盘的总和。并发共享10秒快照缓存，缓存标明原采集时间；采集在后台执行，断线/关闭取消待发送回执，发送不重试刷屏。

Linux Compose部署只读挂载宿主机`/proc/stat`、`/proc/meminfo`、`/proc/loadavg`、`/proc/uptime`四个文件到`/host/proc`，无新增依赖，不挂整个宿主机根目录、不挂Docker socket、不启用特权容器，不提供任意命令/路径输入。此次更新需要重建并重新创建Bot容器才能带上挂载，NapCat无需重启。宿主机统计缺失会标记“暂不可用”，不回退到容器配额并冒充整机数据；其他可读取的指标仍显示。原生Linux不使用Compose时可设置`SERVER_STATUS_PROC_PATH=/proc`；macOS/Windows直接运行不支持这些Linux整机统计，当前Compose配置面向Linux服务器。服务不会输出IP、主机名、进程列表或密钥，也不把详细采集异常路径发给QQ。

例如，管理员分别建立“开黑队”和“饭搭子”，再配置关键词与名单的对应关系：

```text
/名单 创建 开黑队
/名单 添加 开黑队 @张三 @李四
/名单 创建 饭搭子
/名单 添加 饭搭子 @王五
/关键词 添加 开黑
/关键词 绑定 开黑 开黑队
/关键词 添加 上号
/关键词 绑定 上号 开黑队
/关键词 添加 吃饭
/关键词 绑定 吃饭 饭搭子
```

单独发送“开黑”或“上号”只会 @ 开黑队；“吃饭”只会 @ 饭搭子。同一个关键词不能同时绑定两个名单，改绑时先 `/关键词 解绑 开黑`，再绑定新名单。关键词可以包含内部空格，绑定命令的最后一个参数为名单名。

未绑定的关键词不召集任何人；发送“今晚开黑吗”“开黑！”也不会触发。匹配忽略首尾空白，保留内部内容；当前没有召集冷却，连续命中会连续发送。

升级时旧的单一名单会自动迁为“默认”名单，旧关键词自动绑定它，不丢失原成员和触发关系。旧命令 `/名单 添加 @某人`、`/名单 删除 @某人` 继续操作“默认”名单；`/名单 查看` 则改为显示所有名单。全新数据库没有预建名单，需手动创建。迁移在事务中完成，不支持直接回退到旧版代码，升级前请备份数据库。

## 配置说明

配置分为两层：版本库中的 [config.env](config.env) 保存可公开的普通运行参数，服务器本地的 `.env` 只保存 API Key、Token 和管理员 QQ。Compose 将前者注入机器人容器，再用 `.env` 中的私密值补齐敏感字段。本地直接运行 Python 时，按 `config.env` → `.env` → 进程环境变量的顺序覆盖；若希望普通参数随代码更新，旧 `.env` 中应移除同名普通字段。

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API Key，AI 功能需要填写 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 基础地址 |
| `DEEPSEEK_MODEL` | `deepseek-flash` | 请求使用的模型名，以账号可用模型为准 |
| `DEEPSEEK_THINKING_ENABLED` | `true` | 所有DeepSeek请求是否开启思考，包括图片审核 |
| `DEEPSEEK_REASONING_EFFORT` | `high` | 思考强度，可选low/high/max |
| `DEEPSEEK_THINKING_MAX_TOKENS` | `4096` | 思考模式每次推理和最终输出的共同token上限，512～16384 |
| `ONEBOT_ACCESS_TOKEN` | 空 | NapCat 与机器人连接认证；部署时设置随机值 |
| `NAPCAT_WEBUI_TOKEN` | 无 | 云端 NapCat 管理页面密码，与 OneBot Token 分开设置 |
| NapCat 镜像 | `mlikiowa/napcat-docker:latest` | 在 `compose.yaml` 中维护，随代码更新 |
| `ADMIN_QQ_IDS` | 空集合 | 额外管理员 QQ 号，例如 `123456789,987654321` |
| `SERVER_STATUS_PROC_PATH` | `/host/proc` | `/服务器状态`的宿主机只读统计目录；Compose挂载4个固定文件，原生Linux可改`/proc` |
| `DATABASE_PATH` | `data/bot.db` | 本地数据库路径；Compose 固定为 `/app/data/bot.db` |
| `CHAT_PREFIX` | `/问` | 聊天命令前缀 |
| `EVALUATE_PREFIX` | `/锐评` | 锐评命令前缀 |
| `MIMIC_PREFIX` | `/模仿` | 模仿命令前缀 |
| `AI_COOLDOWN_SECONDS` | `8` | 群话题每批结束后的最小间隔；私聊/模仿按原会话冷却，群话题冷却时排队 |
| `AI_MAX_CONCURRENCY` | `3` | 群话题、私聊和模仿共用的后台模型运行上限，范围1～16 |
| `AGENT_MAX_STEPS` | `8` | ReAct模型决策轮数，范围2～16 |
| `AGENT_MAX_TOOL_CALLS` | `16` | 一轮工具调用上限，范围4～32 |
| `AGENT_TIMEOUT_SECONDS` | `120` | 包括下载、审核、发送在内的Agent整轮超时，范围20～300秒 |
| `AGENT_BATCH_COMPLETION_ENABLED` | `true` | 发送成功后按next_action直接收尾，减少单独模型调用；关闭保留原分步流程 |
| `CONTEXT_MESSAGE_LIMIT` | `80` | 群/私聊历史最多条数，不含本批用户消息，范围1～200 |
| `CONTEXT_MAX_CHARS` | `24000` | 短期原文字符预算上限；超限优先引用和最近往返，截断会标记 |
| `WORKING_MEMORY_CONTEXT_MAX_CHARS` | `24000` | 工作记忆独立预算，与CONTEXT_MAX_CHARS取较小值 |
| `CONTEXT_BOT_REPLY_LIMIT` | `12` | 补充本群最近机器人回复及引用链，合并去重后共享历史预算；0关闭 |
| `MEMORY_ENABLED` | `true` | 启用群聊/私聊隔离的长期记忆、回忆与显式操作 |
| `MEMORY_AUTO_CAPTURE_ENABLED` | `true` | 开启任务结束整合及未唤醒消息后台整理；关闭后仍可显式添加/更新 |
| `MEMORY_CAPTURE_DELAY_SECONDS` | `30` | 合并新消息等待时间，范围5～300秒 |
| `MEMORY_CAPTURE_BATCH_SIZE` | `40` | 每批提取的新消息上限，范围5～80 |
| `MEMORY_CAPTURE_RETRY_SECONDS` | `60` | 提取失败的初始退避秒数，最多退避1小时 |
| `MEMORY_RETRIEVAL_LIMIT` | `8` | 每轮最多提供给模型的长期记忆条数，范围1～30 |
| `MEMORY_CANDIDATE_LIMIT` | `300` | 只限制整理器参考的既有记忆数量，不截断聊天检索范围 |
| `MEMORY_QDRANT_URL` | `http://qdrant:6333` | 语义索引地址；原生运行可留空关闭，故障自动本地降级 |
| `MEMORY_QDRANT_COLLECTION` | `qq_memory` | 自有索引前缀；按模型和维度隔离collection |
| `MEMORY_QDRANT_TIMEOUT_SECONDS` | `3` | 远程查询总超时，失败后短暂熔断 |
| `MEMORY_SEARCH_ROUTE_LIMIT` | `20` | 每路召回候选数 |
| `MEMORY_CONTEXT_MAX_CHARS` | `5000` | 单次记忆检索返回的总字符预算 |
| `MEMORY_SOURCE_SEARCH_ENABLED` | `true` | 明确回忆且结果不足时补查允许的原话 |
| `MEMORY_JOB_MAX_ATTEMPTS` | `5` | 整理失败转failed前最多尝试次数 |
| `IMAGE_MEMORY_RETENTION_DAYS` | `90` | 有效事件引用图片的延长保留天数，仍受配额限制 |
| `MEMORY_USER_HISTORY_LIMIT` | `12` | 除群近期历史外，补充当前发言人的个人历史条数；0关闭 |
| `MEMORY_MAX_PER_USER` | `200` | 每位成员在每个群最多保留的长期记忆数 |
| `MEMORY_MAX_CHARS` | `200` | 单条长期记忆最大字数 |
| `MEMORY_MIN_CONFIDENCE` | `0.8` | 模型自动写入长期记忆的最低置信度 |
| `EMBEDDING_PROVIDER` | `bge_onnx` | `bge_onnx` 使用本地 BGE ONNX；`hash` 强制使用轻量哈希 |
| `EMBEDDING_MODEL_PATH` | `/app/models/bge-small-zh-v1.5` | 容器内模型和 tokenizer 目录 |
| `EMBEDDING_MODEL_FILE` | `model_quantized.onnx` | 模型目录中的 ONNX 文件名 |
| `EMBEDDING_MAX_LENGTH` | `128` | 单条记忆/查询的最大 token 长度 |
| `EMBEDDING_THREADS` | `2` | ONNX Runtime CPU 推理线程数；4 vCPU 服务器建议保持 2 |
| `EMBEDDING_BACKFILL_BATCH_SIZE` | `16` | 后台升级旧记忆向量的每批条数 |
| `EMBEDDING_MIN_SIMILARITY` | `0.35` | 非稳定事件仅凭语义召回时的最低相似度 |
| `EMBEDDING_QUERY_PREFIX` | `为这个句子生成表示以用于检索相关文章：` | BGE 检索查询前缀；通常无需修改 |
| `CHAT_BATCH_SECONDS` | `3` | 群话题固定聚批等待秒数，不因新消息重置 |
| `CHAT_BATCH_MAX_MESSAGES` | `30` | 每群等待批次最多消息数，满额明确提问会提示重试 |
| `CHAT_MAX_PENDING_GROUPS` | `32` | 等待/活跃群状态的数量上限 |
| `CHAT_QUEUE_WAIT_SECONDS` | `120` | 群提问排队最长等待秒数 |
| `CHAT_ACTIVE_IDLE_SECONDS` | `300` | 活跃期未成功发言的超时秒数，默认5分钟 |
| `CHAT_ACTIVE_MAX_SECONDS` | `1800` | 单个活跃期最长秒数，默认30分钟，不因继续发言无限延长 |
| `GROUP_REPLY_MAX_CHARS` | `100` | 普通群回复硬字数上限，禁止换行拼接话题 |
| `POKE_ENABLED` | `true` | 是否开放LLM主动戳一戳工具，不影响被戳唤醒 |
| `POKE_COOLDOWN_SECONDS` | `60` | 同一群/私聊主动戳一戳的跨轮冷却，范围1～3600秒；每轮最多一次 |
| `VOICE_ENABLED` | `true` | 是否开放群聊QQ AI音色查询和语音发送工具 |
| `VOICE_CHARACTER_ID` | 空 | 默认真实音色编号；留空用当前群接口列表第一项 |
| `VOICE_MAX_CHARS` | `100` | 语音文本上限，范围10～500；同时不超过群回复字数限制 |
| `VOICE_COOLDOWN_SECONDS` | `60` | 同一群语音跨轮冷却秒数，范围1～3600；失败也占用，每轮最多一条 |
| `RANDOM_CHAT_ENABLED` | `true` | 低活跃期是否由程序随机唤醒普通消息 |
| `RANDOM_CHAT_WINDOW_SECONDS` | `120` | 统计不同发言人的滑动窗口秒数 |
| `RANDOM_CHAT_MIN_PROBABILITY` | `0` | config.env当前基础概率，0～1；代码回退默认0.01 |
| `RANDOM_CHAT_PROBABILITY_PER_USER` | `0` | config.env当前每人增量，0～1；代码回退默认0.02 |
| `RANDOM_CHAT_MAX_PROBABILITY` | `0` | config.env当前概率上限，0～1；代码回退默认0.20 |
| `RANDOM_CHAT_COOLDOWN_SECONDS` | `60` | 每群随机抽中后的最短间隔秒数 |
| `REMINDER_POLL_SECONDS` | `5` | 检查到期提醒的间隔秒数，范围 1～60 |
| `REMINDER_MAX_PENDING` | `50` | 每位用户最多未完成提醒数，范围 1～100 |
| `CHAT_IMAGE_ENABLED` | `true` | 查看当前/近期授权图片；不控制后台缓存和概要，需视觉模型 |
| `CHAT_IMAGE_MAX_BYTES` | `5242880` | 聊天来图单图字节上限，最高5MiB |
| `STICKER_ENABLED` | `true` | 是否开放本地表情包工具 |
| `STICKER_VISION_ENABLED` | `true` | 是否开放原图查看工具，要求模型支持视觉 |
| `STICKER_AUTO_INDEX_ENABLED` | `true` | 后台视觉识别新图并缓存，产生API费用；关闭后使用人工/文件名描述 |
| `STICKER_INDEX_POLL_SECONDS` | `60` | 每轮入库结束后的扫描间隔，范围10～3600秒 |
| `STICKER_INDEX_BATCH_SIZE` | `5` | 每轮最多视觉标注请求数，范围1～20 |
| `STICKER_INDEX_RETRY_SECONDS` | `3600` | 识别失败首次重试间隔，范围60～86400秒，后续指数退避最多24小时 |
| `STICKER_PREVIEW_LIMIT` | `3` | 一轮最多查看多少张不同原图，范围1～3 |
| `STICKER_PROMPT_LIMIT` | `10` | 每轮文字预览最多候选数，范围0～20；0关闭预览，不关闭检索 |
| `STICKER_PROMPT_ROTATE_SECONDS` | `3600` | 文字预览轮换周期秒数，范围60～86400 |
| `STICKER_DIRECTORY` | `app/sticker_images` | 本地图片目录；Compose固定为 `/app/app/sticker_images` |
| `STICKER_COLLECT_ENABLED` | `true` | 同时开关收藏工具与后台随机收图，不影响既有收藏 |
| `STICKER_COLLECT_PROBABILITY` | `0.20` | 每张新群图片的独立抽签概率，0～1；0只停随机，保留工具 |
| `STICKER_COLLECT_QUEUE_SIZE` | `32` | 后台图片引用等待数量上限，不提前下载 |
| `STICKER_COLLECT_QUEUE_WAIT_SECONDS` | `120` | 最大排队时长及允许的新消息年龄（秒） |
| `STICKER_COLLECTED_DIRECTORY` | `data/collected_stickers` | 收藏目录；Compose固定为 `/app/data/collected_stickers` |
| `STICKER_COLLECT_COOLDOWN_SECONDS` | `60` | 每群非重复图片审核的最短间隔，审核失败也占用；0关闭 |
| `STICKER_COLLECT_MAX_FILES` | `200` | 收藏目录文件数上限，人工/收藏扫描合计最多500张 |
| `STICKER_COLLECT_MAX_DISK_BYTES` | `209715200` | 收藏目录总字节上限，默认200MiB |
| `STICKER_COOLDOWN_SECONDS` | `30` | 同一群或私聊会话的表情发送间隔，0关闭冷却 |
| `STICKER_REPEAT_SECONDS` | `7200` | 同一会话同一图片的重复间隔，0允许重复 |
| `STICKER_MAX_BYTES` | `5242880` | 单张图片大小上限，默认5MiB、配置最多10MiB |
| `STYLE_SAMPLE_LIMIT` | `60` | 最多查询多少条模仿样本，每条截取前 300 字 |
| `MESSAGE_RETENTION_DAYS` | `30` | 消息保留天数，启动时清理，非正数关闭清理 |
| `BOT_NAME` | `小群友` | 预设名称，目前尚未参与回复或提示词生成 |

## 更新与维护

在服务器项目目录中执行：

```bash
# 下载最新代码并重新构建
git pull --ff-only && \
sudo docker compose up -d --build bot

# 查看日志和资源占用
sudo docker compose logs --tail=100 bot
sudo docker compose logs --tail=100 napcat
sudo docker stats --no-stream

# 停止服务，保留数据
sudo docker compose down

# 恢复服务
sudo docker compose up -d
```

修改 `.env` 中的私密值后，执行 `sudo docker compose up -d --force-recreate` 更新容器环境。NapCat 的 Token 和网络设置只在首次创建配置时自动初始化；已有部署需要同步在 NapCat WebUI 修改。

普通参数统一在 `config.env` 中维护并随 `git pull` 更新，拉取后重新创建 `bot` 容器即可生效。旧版 `.env` 中即使还保留 `CONTEXT_MESSAGE_LIMIT` 等普通字段，Compose 也不会再把它们注入容器；可以在确认密钥和管理员 QQ 已保留后删掉这些旧字段。只修改普通配置时，使用 `sudo docker compose up -d --no-deps --force-recreate bot`，无需重建 NapCat。

不要执行 `docker compose down -v`，它会删除本项目的数据库、QQ 登录状态和 NapCat 配置数据卷。数据卷可以防止普通重建丢失数据，但不替代备份。

## 本地运行

需要 Python 3.11+ 和 uv。NapCat 需要单独安装、登录并配置连接。

```bash
git clone https://github.com/Nightsky-Lty/qq-chatbot.git
cd qq-chatbot
uv sync --frozen
cp .env.example .env
# 编辑 .env，只配置 API Key、两个 Token 和管理员 QQ 号
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080
```

在同机 NapCat WebUI 添加启用的反向 WebSocket 客户端：

- URL：`ws://127.0.0.1:8080/onebot/v11/ws`。
- Token：与 `.env` 中的 `ONEBOT_ACCESS_TOKEN` 一致。
- 消息格式：数组（array）。

这里假设 NapCat 和 Python 直接运行在同一台机器上；NapCat 在容器内时，`127.0.0.1` 指向容器自身，应使用可到达 Python 服务的地址。

## 项目结构

```text
app/
├── main.py       # FastAPI 入口、生命周期和 OneBot WebSocket 收发
├── balance.py    # 管理员余额查询、短时缓存和固定格式回执
├── server_status.py # 管理员服务器资源快照，只读固定统计文件、不调用LLM
├── bot.py        # 群/私聊分发、召集、后台聊天、锐评和模仿逻辑
├── database.py   # SQLite 表结构和读写操作
├── memory.py     # 长期记忆约束、混合召回和后台向量升级
├── embeddings.py # BGE ONNX CPU 推理、池化归一化和哈希降级
├── llm.py        # DeepSeek API 调用和异常处理
├── agent.py      # ReAct循环、工具观察、预算及终止
├── conversation.py # 群聚批、串行有界队列和话题活跃状态
├── tools.py      # 工具注册表、参数校验及finish定义
├── messaging.py  # 当前会话的send_message工具、限量和确认状态
├── persona.py    # 小鲸鱼身份、性格、形象和表达风格
├── prompting.py  # 稳定规则/场景/能力说明与结构化动态上下文的唯一组装入口
├── working_memory.py # 最近80条结构化原文工作记忆与真实来源归档
├── task_evidence.py # 单次Agent证据范围和查询缓存，不保存推理
├── reminders.py  # 私聊提醒工具、时间校验、到期通知和失败重试
├── stickers.py   # 本地表情目录、视觉查看工具及发送限制
├── sticker_index.py # 后台视觉标注、哈希缓存及风险隔离
├── sticker_collect.py # 按图片概率抽签、有界队列、哈希去重和独立审核收藏
├── chat_images.py # 当前消息按需看图工具、QQ地址及大小限制
├── sticker_images/ # 表情图片及可选描述，服务器本地维护
├── models.py     # OneBot 消息解析、文本和 @ 消息段构造
├── config.py     # 配置读取、校验和默认值
└── __init__.py   # Python 应用包标记
config.env         # 可提交的普通配置，随 git pull 更新
docs/deployment.md # Ubuntu 安装、部署和管理说明
Dockerfile         # Python 机器人镜像
compose.yaml       # 机器人与 NapCat 的容器编排
scripts/           # BGE ONNX INT8 模型导出脚本
.env.example       # 本地私密配置模板，不包含真实密钥
pyproject.toml     # Python 依赖声明
uv.lock            # Python 依赖版本锁定
```

## 常见问题

**机器人不回复 `/帮助`**：检查两个容器是否运行、QQ 是否登录且在群中、NapCat 是否启用了反向 WebSocket。Python 日志出现 `OneBot reverse WebSocket connected` 表示连接已建立。

**群聊正常，但私聊无回复**：确认已更新到支持私聊的版本，并已加机器人 QQ 为好友。私聊 `/帮助` 可单独验证收发，无需 @；直接发文字或开启看图后的图片会触发聊天，语音和视频不会请求模型。

**提醒没有收到**：先用“查看我的提醒”确认确实已保存及时间正确，再检查 QQ 登录、好友关系、OneBot 连接和日志中的 `Reminder delivered` 或 `Reminder delivery retry`。到点发送不依赖 DeepSeek，但查看和新建依赖模型；服务器停机期间不能实时通知，恢复后补发。

**帮助能回复，但 AI 不回复**：检查 API Key、模型名、余额和服务器网络，查看 bot 日志中的 DeepSeek 请求错误。

**AI 回复很慢**：默认已开启high强度思考，思考、原图输入以及多步工具调用可能增加延迟。Agent在后台运行，不阻塞管理命令；日志 `Agent handled: ... elapsed=...s` 记录整轮工具/发送处理耗时，`Agent finished` 显示轮数和动作数。可将 `DEEPSEEK_THINKING_ENABLED=false` 关闭思考，或缩短上下文；同时检查API网络和提供商响应。

**群里普通聊天不接话**：低活跃期默认按不同发言人数随机唤醒，每群抽中后冷却60秒，不保证每条回复；可以@或/问直接唤醒。日志Random group join显示参与人数和概率，Group batch显示批次，Group context显示群历史/自身回复数量，Group conversation state显示活跃状态。scope=auto表示活跃跟进处理，不保证实际发言。@其他人或引用其他人的消息不作为随机回复对象。

**浏览器打不开 NapCat**：管理页面只绑定服务器本机地址，需要先建立 SSH 转发，再访问电脑的 `16099` 端口；不能直接通过服务器公网 IP 的 `6099` 端口访问。

**模仿提示样本不足**：无需授权，但目标用户至少要有 5 条保存在当前群的普通文字发言；以 `/` 开头的指令、空文本、其他群或私聊记录不计入。升级前的普通文字也可使用，过期或由部署者清理的消息不能使用。

**云服务器内存较小**：先用 `docker stats --no-stream` 观察 QQ、NapCat 和机器人占用，再决定是否增加 Swap 或升级内存。调用 DeepSeek API 不需要 GPU。

**日志提示 Embedding model unavailable**：确认宿主机存在 `models/bge-small-zh-v1.5-int8/model_quantized.onnx` 和 `tokenizer.json`，执行 `sudo docker compose exec bot ls -lh /app/models/bge-small-zh-v1.5` 检查挂载。模型不可用期间会自动使用哈希检索，不影响聊天启动。

## 数据与当前限制

机器人以明文SQLite保存群/私聊记录、成员别名、提醒和记忆。开启后台整合时，未唤醒Agent的普通群文字和私聊也会发给独立记忆LLM，额外消耗API。原始消息默认30天，必要来源摘录与修订随长期记忆保存。遗忘会清空该记忆/修订正文、停止使用并阻断旧来源；删除会清除记录及派生索引，但不等于擦除全部原始聊天、SQLite空闲页、备份或已发给模型的数据。完整个人数据清除会失效相应图片、记忆和任务，历史备份仍按部署者保留策略处理。Qdrant在服务器内网保存向量和编号元数据，向量也不是匿名数据；应保护所有数据卷。

群活跃跟进期间，以及低活跃期普通消息被程序随机抽中时，未@的新消息及相关历史也可能发送给DeepSeek，并非只在@时调用；模型调用finish或超时后停止活跃跟进，普通消息仍可能再次被程序抽中。可以用RANDOM_CHAT_ENABLED关闭随机唤醒；旧AUTO_CHAT_ENABLED不再控制此机制。

当前按单实例、单进程运行。OneBot入口串行接收事件及处理命令，后台Agent限制并发；只在后台等待发送确认，不阻塞接收循环。聊天发送工具和到期提醒等待NapCat成功结果，普通控制命令仍直接提交。重启取消尚未完成的Agent，但已写库提醒和已落盘收藏会保留；冷却和调用去重在内存中。禁止多实例或增加uvicorn worker，以免重复调度及破坏收藏配额。

NapCatQQ 是第三方 QQ 协议端实现，本项目使用 OneBot 11 接入，并非 QQ 开放平台官方机器人。Python 配置和语法已做基础检查，Docker 镜像构建及真实 QQ 收发仍需在部署环境确认。
