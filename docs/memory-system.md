# 四层记忆与混合检索

## 存储与边界

- Working：默认最近80条结构化聊天原文，不做短期摘要；当前请求单独注入，保留多人身份、引用、图片资料，受24000字符预算约束。30分钟TTL只控制内存缓存，原文窗口可从SQLite恢复，不因单次finish清空；任务证据/查询缓存仍由`TaskEvidence`维护。详见[对话工作记忆](working-memory.md)。
- Event：SQLite `memory_records.layer='event'`，通过`memory_events`视图查看。event/episode/commitment保留参与来源、原话摘录、发生/记录时间、版本、有效期、陈述归属；共同经历/约定需要真实双方发言和群内回应关系，约定不是工具成功证明。
- Abstract：`memory_records.layer='abstract'`与`memory_concepts`视图。profile/preference/relationship/explicit等子类型；group_meme保存条件、别名、含义、支持事件。明确解释或至少三个已提供事件支持可生效，其余为candidate，不自动注入。计数及模型自报置信度均不等于客观正确。
- Perceptual：`image_cache_blobs/refs`仍是图片资源权威记录，文件在`/app/data/image_cache`按SHA-256去重。概要以`memory_records.layer='perceptual'`建立可检索引用，不把概要当发图人的发言/经历。只实现图片，未实现音频、视频或视觉embedding。

`memory_records`是长期记忆唯一权威表，事件/抽象/感知按layer分层；`memory_events/memory_concepts/memory_percepts`是视图。`memory_type`仅是层内子类型（偏好、约定、群梗等），不是旧系统的独立入口。`memory_links`表达derived_from/revises，`memory_media_links`连接事件和图片。未部署Neo4j。

## 初始化与已有数据

旧记忆已在部署版本 `52c13a5` 完成转换。当前版本仅幂等创建新存储结构，不再包含旧库检测、拒绝启动、导入、转换、自动迁移备份或离线迁移脚本；不会自动读取或清理旧`memories`表。仅支持当前数据库，全新部署从空的`memory_records`开始。

仍使用现有SQLite `bot.db`，以保留聊天原话、图片引用、整合游标/队列和记忆的关联事务。已转换的记忆、来源、版本、状态、遗忘阻断和既有备份不因清理迁移代码而变动。所有正常写入继续通过证据校验与`MemoryStore.apply`；缺少来源摘录时统一说明来源不足，不再依赖迁移标记判断。

已有升级前备份由部署者按保留策略管理，不自动删除，也不要直接用于当前版本；历史转换代码可在Git版本`52c13a5`中查阅。备份含完整敏感数据，不上传Git。运行中的数据库和图片目录仍应定期备份。

事件明确引用的图片默认延长至90天，可由`IMAGE_MEMORY_RETENTION_DAYS`调整；普通图片仍默认7天。仍受全局500文件/256MiB配额约束，FIFO容量淘汰可早于保留期。原始消息自然到期后，只要有有效事件引用，保留图仍可访问；本人数据清除会失效对应引用，不因保留而复活。物理文件去重不授予跨会话权限。

## 短期到长期的整合

短期上下文改用原文窗口，不再进行语境压缩；长期整合仍使用原来的持久队列、独立提取模型与证据校验。空闲缓存淘汰及显式整合可提交真实来源，在整理锁内排除游标已处理和已归属任务的来源；缓存淘汰不删除SQLite原话，原话入库时已有的增量队列仍兜底。

任务开始即保存最小的新证据ID任务标记，状态running，以便崩溃后不丢掉已接收证据；并不持久化工作区/推理。结束收集本批真实新消息和确实发出的对应机器人回复，将任务改为pending，然后释放工作区。独立记忆LLM只根据这些证据提出候选，程序检查来源、本人身份、敏感信息、概念依据、版本、图片范围、有效期后原子提交。

`memory_consolidation_jobs`持久重试；启动将中断的running任务转为pending，不恢复推理，也不自动重新发送QQ消息。失败指数退避，默认5次后failed，不无限堵塞新记忆。后台仍对未唤醒Agent的普通新消息以30秒/40条批次整理；已分配任务的来源不重复作为新证据。原始消息被清除后不能靠在途任务重建。结果有stored/candidates/duplicates/updated，不把排队或候选说成已存有效记忆。

## 六个Agent工具

| 工具 | 行为 |
| --- | --- |
| memory_add() | 本人本轮明确要求记住，将真实原话交给独立提取器；不接收自由编写的事实 |
| memory_search(query, layer?, subject?, after?, before?, include_sources?) | 检索本会话，人物明确时可过滤，时间为Unix秒；默认紧凑摘要，必要时最多3条来源 |
| memory_update(memory_id, expected_version) | 本轮明确纠正，校验原版本并新增修订、保留旧版本；同键不同条件不自动覆盖 |
| memory_delete(memory_id, expected_version) | 有明确请求时清除指定记录、修订正文及依赖的派生记忆，排队删除向量，阻断旧来源重建 |
| memory_forget(memory_id, expected_version) | 停止使用，清空正文/来源/向量及修订正文，保留最小摘要指纹和来源ID阻断；不只是降权 |
| memory_consolidate() | 声明任务结束时短期转长期；自动执行，不是任意合并/重写历史库 |

空间、请求者由程序绑定，不接受模型提供namespace/路径/SQL。个人记忆只有本人可修改；共享群概念仅已验证的群管理者/配置管理员可显式管理，管理权限不延伸到任何私聊。普通后台整理器不拥有delete/forget权限。

删除/遗忘会阻断关联源消息进入后续原话回查和聊天上下文，可能保守地排除同一条消息中的其他信息。阻断依靠来源ID、正文摘要和带条件的语义键，并不能保证识别任意改写的全新发言；第一版不支持自动恢复已遗忘事实。原始QQ聊天与历史备份不因删除一条记忆自动擦除，SQLite空闲页/既已发给模型的数据也不能承诺安全擦除。完整个人资料清除仍使用现有数据清除功能，并按部署者备份保留策略处理历史备份。

## 检索

1. 当前批次焦点/私聊请求作为查询，不将整段群聊拼成向量。相关成员是软排序线索，只有明确人物过滤才硬限制；默认不会只搜当前发言人的资料。
2. 术语先匹配词条，关键词用FTS5（预生成中文单字/双字片段和英文词，查询词转义），向量用本地BGE生成、Qdrant检索；图片第一版只搜索概要文本向量。
3. 各路默认20候选，应用层RRF融合，不直接相加BM25与余弦分数，不用模型自报置信度当正确率。不再先截取最近300条才比较语义。
4. SQLite再次检查空间、版本、状态、期限、已遗忘来源及本批边界。Qdrant回传跨空间/旧版本也不采信。低相关内容不为填满数量入选。
5. 只展开一层、最多两个有效关联摘要；默认最多8条、5000字符上下文。`include_sources=true`才展开来源。必要的历史图编号动态加入本轮看图/收藏授权，仍受看图最多3张与独立收藏审核限制。
6. 明确回忆且结果不足时才用源消息FTS补查，仍遵守权限、期限、来源阻断和预算；无可靠结果返回空，不编造。常见今天/昨天/前天/ISO日期按北京时间解析；其他复杂日期可由工具显式提供边界，人物歧义由聊天模型追问。

任务内查询缓存只存候选ID/版本，再次读取仍回查状态。Qdrant失败短暂熔断，使用SQLite FTS和本地流式向量top-k降级；BGE失败再用哈希向量，语义效果会下降。没有额外前置LLM查询改写或LLM重排调用。

## Qdrant与同步

Compose固定`qdrant/qdrant:v1.19.2`，可通过`.env`的`QDRANT_IMAGE`换可信镜像站同版本；仅内部网络，无宿主机6333/6334映射，数据卷`qdrant-data`，内存上限512MiB、CPU1核。可选`MEMORY_QDRANT_API_KEY`同时用于服务和客户端。

`MEMORY_QDRANT_URL=http://qdrant:6333`；原生运行可置空关闭远程索引。collection名称为`MEMORY_QDRANT_COLLECTION_维度_模型摘要`，不同模型不混向量空间。payload只包含编号、范围、层次、版本、人物与时间，不复制聊天正文。即便如此向量也不是匿名化数据，仍须保护数据卷。

权威记录变化和`memory_index_jobs`在同一SQLite事务写入，后台幂等upsert/delete，只有任务修订号仍相同才确认完成；失败退避、不丢任务。删除会清理同一自有前缀下的旧模型collection点位，不能仅删当前模型。索引暂时不同步不影响SQLite立即阻断内容。SQLite保留本地向量作为降级/重建来源。

备份权威SQLite与图片目录。Qdrant卷可备份，但不是唯一事实来源；索引丢失时可用下面方式重新排队，然后重启bot或等待worker：

```sql
INSERT OR REPLACE INTO memory_index_jobs(memory_id,revision)
SELECT id,version FROM memory_records;
```

不要执行`docker compose down -v`，它会删除所有持久卷。更新推荐`docker compose up -d qdrant`，构建/测试后`docker compose up -d --no-deps bot`，不重启NapCat。

## 验证

`python3 -m unittest discover -s tests -q`覆盖新库初始化、现有数据重启保持、跨空间/跨人物、超过300条旧事件、中文词条、日期、概念候选、条件共存、版本/权限、原话兜底与遗忘、任务交接/重启、Qdrant协议和旧索引删除、感知检索与保留。

`scripts/verify_memory_system.py`在临时数据库和独立临时collection中验证真实BGE/Qdrant；`--live`额外调用一次真实记忆LLM，输入仅合成对话，不读真实聊天、不发送QQ。临时collection由脚本清理。

当前限制：仍是单实例/单进程；语义抽取、群梗推断不是事实核验；复杂事件时间和多人归属可能需要追问。没有自动图推理、音视频处理、视觉embedding或专用重排模型。
