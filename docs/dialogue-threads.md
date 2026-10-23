# 群会话解析与单焦点参与

默认开启 `THREAD_ENABLED=true`。原文仍在现有 SQLite `messages` 表；会话业务层先判断消息归属，再选择一个 thread 组装上下文交给回复 Agent。同群仍串行，其他 thread 不启动并发回复，也不产生必须逐个补答的任务清单。私聊与模仿路径保持原有行为。

## 消息归属

原文提交后，在同一SQLite数据库记录解析状态。非活跃期间只登记未解析状态和确定的引用关系，不启动语义解析，也不为每条消息创建临时thread。@、戳一戳、显式命令或程序随机唤醒后，首次解析会补最多12条邻近待定前文。

活跃期包括正在回复和`wait_for_messages`后的跟进。新群消息进入独立的解析队列，包括其他参与者和旁听消息；队列每群上限为`CHAT_BATCH_MAX_MESSAGES × 4`，超出部分仍保留原文与未解析状态。默认聚批1秒、满20条提前提交，每群一个解析任务，全局最多2个解析请求，不占回复Agent的并发额度。没有待处理消息不请求模型；退出、断线、关闭服务时取消对应解析任务。回复仍受独立的8秒冷却与单thread参与约束。

同群引用连接已有thread；同作者8秒内相邻、无引用/@且非问句的补充可暂定连接，其他活跃消息使用临时归属。@用于判断对象，不能单独合并所有找机器人的问题。解析LLM结合往返、时间、参与者和语义判断已有thread或新交流；待定短句可以保持不确定。确认发送的文字、表情、语音和戳一戳通过本地来源ID绑定真实回应对象，没有QQ消息号也保留关系。

每次最多50个解析目标、5个候选thread，动态输入最多14000字符，输出最多3000 token、15秒超时。超限先缩短并标注，再减少候选或目标；未处理目标继续保持未解析。解析结果逐项验证群、来源、候选thread及向前的parent关系，不执行消息或昵称中的指令。

默认同群最短间隔1秒、每分钟最多60次，实时与历史补解析共用同群锁、全局并发和次数额度。限额不足保留活跃新消息等待，不调用回复模型轮询，也不刷新活跃寿命；排队受120秒期限与活跃超时控制。失败状态保存尝试次数和30秒退避时间，同一来源最多3次尝试；失败不会自动回扫整段历史。不确定是已处理状态，不因缺少确定thread反复请求模型。

## 注意力与切换

每群一个 `active_thread_id`。本批当前thread有可回复消息时优先保持；其他thread维护记录但不触发回复Agent。无当前新话时可选择新的明确邀请，默认焦点至少保持20秒；“先别聊”“换个话题”“先帮我”等明确切换表达优先。邀请默认90秒内有效，并且仅在对应thread再次有新消息时考虑，不定时补答积压问题。

切换只在群串行锁内、一次Agent运行结束后的边界发生。`wait_for_messages` 保留当前交流，成功发送才刷新原有群活跃期限；旁听沉默不刷新期限。`finish`、失败、断线或群数据清除释放焦点。重启释放活跃注意力，但保留消息归属；下次明确唤醒可按引用/解析恢复交流。30分钟无活动的thread退出常规近期候选，显式引用仍可找回。

## 回复上下文

当前任务只包含选定thread的本批新消息，引用目标、当前URL授权和发送焦点也限定到该批。默认历史为本thread最近12条，再补发起消息、待回答问题、有限的相关原文/纠正及至多4层关系祖先。必要旧消息可以在最近80条全群窗口之外；检索时重新核对原文存在、遗忘阻断和批次边界。不能恢复已删除来源。

其他交流最多提供3条、每条正文最多400字符的 `旁听参考（非当前请求）`，标为不可回复，不成为新操作授权。图片目录通常仅含当前thread与显式引用来源；当前请求明确谈及图片时，可提供近期旁听图片作有来源的候选，仍执行图片有效期/同群授权/独立审核。当前thread原文回查使用`read_thread_history`。长期记忆按相关性检索，`memory_search`支持`all/event/abstract/perceptual`，另有新增、更新、删除、遗忘和整合五个操作，均受会话权限约束。

原文使用现有短期字符预算（两个配置的较小值，默认24000），超限仍明确标注截断。条数和字符不等于计费token；实际成本包括新增解析调用，需观察日志，不能保证总费用必然下降。

## Agent主动向前翻阅原文

启用thread的群回复Agent提供`read_thread_history`，关闭长期记忆或工作记忆缓存仍可使用；私聊、模仿、观察和未绑定thread的运行不注册。工具绑定当前群、当前thread和本批开始前的来源边界，不接受模型自选群/thread。

首次可传`{}`，默认从连续可见的近期往返之前读取10条，`limit`支持1～20，仅统计返回的当前thread消息。发起消息或远端引用不会把默认位置拉到最老；可见历史之间存在未分类空洞时，也不会跳过空洞。`before_source_id`可定位已知当前thread来源，不包含该条，不是QQ消息号。

续读优先传`{"cursor":"上次返回的next_cursor"}`。凭证仅在本次Agent运行有效，绑定会话与时间边界，记录群时间线的扫描进度；空页也能继续。`cursor`与`before_source_id`不能同时提供。`next_before_source_id`仍提供已返回消息的定位编号，但无法表达空页进度。

已分类区间使用thread索引，完整分类但无关的区间直接跳过。遇到未分类区间时，每块最多扫描50条本群原始消息，并带入有限邻近证据和已知thread锚点，调用解析LLM后只返回当前thread原文。块按消息顺序切分，不固定按分钟切分；不同块可以属于同一交流。仅主题相似不能把独立往事强行归入当前thread。分类状态持久化，跨查询和重启复用。

每次翻页最多扫描2块、进行2次解析请求，总等待最多15秒。解析失败、预算不足或未分类空洞未处理完时，不越过该位置推进游标；`classification_pending`、`classification_failed`和`retry_after_seconds`说明待处理状态，`scan_budget_exhausted`说明扫描预算用完。`has_more=true`表示已知还有当前thread记录，`null`表示尚不能确认，`false`表示已完成该方向查询且无更早可返回记录；空页不等于读完。`next_cursor`用于继续，不把未完成说成不存在。

返回`messages`按旧到新排列，保留完整原文、作者、角色、北京时间、@、引用/逻辑回应和关系类型。每轮最多3次工具调用，每页最多12000字符，累计最多24000字符，按完整JSON计算；不截断或跳过装不下的长原文。`budget_exhausted`说明本轮读取或字符额度不足。

工具可补充历史归属，但不切换注意焦点，不把旧消息当作任务新证据，不扩大QQ引用、@、URL或图片操作授权。每页重新核验来源、遗忘阻断和群/thread边界；删除或遗忘的原文不恢复，后到消息不提前注入。解析调用和回复模型读取回传原文均会产生API费用。

## 持久化与配置

- `dialogue_threads`：群范围、状态和待回答问题来源ID。
- `dialogue_assignments`：消息来源ID、thread、回应来源、规则/模型依据类型、关系及邀请标记。
- `dialogue_attention`：唯一当前焦点、切换时间、最后解析尝试时间/边界。
- `dialogue_parse_state`：每条来源的pending/resolved/uncertain/failed状态、尝试次数、更新时间和可重试时间；通过来源ID关联原文，区间完成情况由逐条状态确定，不用单个高水位掩盖空洞。

不复制聊天正文、模型摘要或推理。消息删除级联清除归属；使用前剔除被遗忘来源并清理空thread。保留策略按原有消息数据库执行，离开近期候选不等于删除thread。当前仍为单实例/单进程设计。

`THREAD_LLM_MODEL/BASE_URL/THINKING_ENABLED` 在 `config.env` 中配置；默认模型/地址继承聊天连接，思考关闭。独立密钥使用 `.env` 的 `THREAD_LLM_API_KEY`，留空继承聊天Key，Compose已显式注入。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| THREAD_ENABLED | true | 关闭后恢复全群原文上下文路径 |
| THREAD_HISTORY_LIMIT | 12 | 当前thread最近历史条数，不含必要补充 |
| THREAD_REFERENCE_LIMIT | 3 | 旁听参考条数 |
| THREAD_CANDIDATE_LIMIT | 5 | 解析候选thread数 |
| THREAD_PARSE_MAX_MESSAGES | 50 | 每次最多待解析消息数 |
| THREAD_PARSE_BATCH_SECONDS | 1 | 活跃消息聚批秒数 |
| THREAD_PARSE_BATCH_MESSAGES | 20 | 满批提前提交条数 |
| THREAD_PARSE_CONCURRENCY | 2 | 全局解析并发 |
| THREAD_PARSE_MAX_CHARS | 14000 | 解析动态输入字符预算 |
| THREAD_PARSE_TIMEOUT_SECONDS | 15 | 解析超时 |
| THREAD_PARSE_MIN_INTERVAL_SECONDS | 1 | 同群最小请求间隔 |
| THREAD_PARSE_CALLS_PER_MINUTE | 60 | 每进程同群每分钟上限 |
| THREAD_DORMANT_SECONDS | 1800 | 常规候选休眠时间 |
| THREAD_ATTENTION_HOLD_SECONDS | 20 | 普通切换前的焦点保持时间 |
| THREAD_INVITATION_TTL_SECONDS | 90 | 邀请新鲜度 |
| THREAD_HISTORY_READ_MAX_CALLS | 3 | 每轮历史翻页调用上限 |
| THREAD_HISTORY_READ_MAX_CHARS | 12000 | 每页工具结果字符预算 |
| THREAD_HISTORY_READ_TOTAL_CHARS | 24000 | 每轮成功页回传字符总预算 |
| THREAD_HISTORY_SCAN_MESSAGES | 50 | 每块物理扫描消息上限 |
| THREAD_HISTORY_SCAN_BLOCKS | 2 | 每次翻页扫描块数上限 |
| THREAD_HISTORY_PARSE_CALLS | 2 | 每次翻页解析请求上限 |
| THREAD_HISTORY_TIMEOUT_SECONDS | 15 | 每次翻页总等待上限 |

日志 `Thread parsed` 记录解析目标、候选数和返回的token计数；`Thread parse fallback` 只记异常类型；`Thread attention` 记录保持/选择结果，不写原文、密钥或推理。

## 验证方式

现有测试包含原文窗口与长期记忆验证。部分工作记忆集成测试与当前thread行为不一致，全量测试尚未全部通过。新业务验证用例按项目约定在仓库外临时执行，不新增回归测试文件；真实多人聊天的隐式归属质量仍需使用观察与回放评估。
