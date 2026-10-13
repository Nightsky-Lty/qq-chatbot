# QQ 群 DeepSeek 机器人

基于 Python、FastAPI 和 SQLite 的 QQ 群机器人，通过 NapCatQQ 的 OneBot 11 反向 WebSocket 收发消息，调用 DeepSeek API 完成聊天、锐评和发言风格模仿。

提供 Docker Compose 部署配置，可在一台 Linux 云服务器上同时运行机器人和 NapCat。完整安装步骤见 [Ubuntu 云端部署指南](docs/deployment.md)。

## 功能

| 功能 | 当前行为 |
| --- | --- |
| 群成员名单 | 每个群可独立创建多个命名名单，支持维护成员和删除名单，同一用户可加入多个名单 |
| 关键词召集 | 关键词独立管理，每个关键词最多绑定一个名单，多个关键词可绑定同一名单；完整匹配时分批 @ 对应名单成员，每条最多 20 人 |
| 群聊对话 | 使用 `/问` 或 @ 机器人触发，以蓝色鲸鱼娘女仆形象参考最近群消息回复 |
| 内容锐评 | 使用 `/锐评` 主动提交要评价的文本 |
| 发言风格模仿 | 用户本人授权后，以授权期间保存的发言作为提示词样本生成新话 |
| 个人消息删除 | 删除本人在当前群保存的消息，并撤销模仿授权 |
| 数据持久化 | SQLite 保存名单、关键词、群消息和授权记录 |

模仿功能使用样本提示词，不训练或微调模型。机器人当前处理文本和 @ 消息，不解析图片、语音和视频。普通聊天不会自动调用 AI；必须使用命令或 @ 机器人。

## 聊天人设

普通聊天和 `/锐评` 使用“大肥鱼”风格的蓝色鲸鱼娘女仆人设：贪吃、偶尔摸鱼、有一点傲娇，喜欢接梗和自嘲，认真提问时优先把问题回答清楚。日常回复以短句为主，少量使用“主人”、颜文字或动作描写。

提示词位于 [app/llm.py](app/llm.py) 的 `SYSTEM_PROMPT`，参考社区的 [鲸鱼娘陪伴项目](https://github.com/AiChiJiMaoRen/dsh-dafeiyu-chan) 和 [角色设定](https://github.com/YunYueSama/codex-deepseek-pet/blob/main/docs/%E8%A7%92%E8%89%B2%E8%AE%BE%E5%AE%9A.md) 编写原创群聊表达。这里是文字聊天角色，不会自动修改 QQ 昵称、头像或发送角色图片。

`/模仿` 使用独立提示词，继续模仿授权用户的说话方式，不叠加女仆语气。修改默认人设后重新构建 `bot` 容器即可生效。

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

### 3. 启动两个服务

```bash
sudo docker compose config --quiet
sudo docker compose up -d --build
sudo docker compose ps
curl http://127.0.0.1:8080/health
```

Compose 会先启动 Python 服务，再启动 NapCat，自动初始化数组消息格式的反向 WebSocket 连接。`/health` 仅检查 Python 服务是否可以响应，不代表 QQ 已登录或 DeepSeek 请求成功。

### 4. 登录机器人 QQ

在自己电脑上新开终端，使用实际用户名和服务器 IP 建立 SSH 转发：

```bash
ssh -N -L 16099:127.0.0.1:6099 用户名@服务器IP
```

如果已配置 SSH 别名 `qqbot`，可以使用 `ssh -N -L 16099:127.0.0.1:6099 qqbot`。

保持终端打开，在电脑浏览器访问 `http://127.0.0.1:16099/webui`，用 `NAPCAT_WEBUI_TOKEN` 登录，再按照页面提示登录机器人的 QQ 账号。

两个容器通过内部网络通信，服务器公网无需开放 8080 或 6099 端口。数据库和 QQ 登录状态保存在 Docker 数据卷中，普通容器重建会保留数据。

### 5. 加群使用

把机器人 QQ 加入目标群，先发送 `/帮助` 确认 QQ 收发，再用 `/问 你好` 确认 DeepSeek 请求。

## 群内命令

| 命令示例 | 说明 | 权限 |
| --- | --- | --- |
| `/帮助` | 显示可用命令 | 所有群成员 |
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
| `/允许模仿` | 授权本人之后的新发言成为模仿样本 | 仅本人 |
| `/禁止模仿` | 撤销本人当前的模仿授权，历史消息仍会保留 | 仅本人 |
| `/模仿 @某人 来一句上号宣言` | 按目标用户的授权样本生成新话，至少需要 5 条样本 | 所有群成员，目标用户须授权 |
| `/删除我的数据` | 删除本人在当前群的消息和授权，不删除召集名单 | 仅本人 |

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

配置模板为 [.env.example](.env.example)。本地运行从 `.env` 和环境变量读取配置；Compose 显式将对应配置传入容器。

| 配置项 | 默认值 | 用途 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API Key，AI 功能需要填写 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 基础地址 |
| `DEEPSEEK_MODEL` | `deepseek-flash` | 请求使用的模型名，以账号可用模型为准 |
| `ONEBOT_ACCESS_TOKEN` | 空 | NapCat 与机器人连接认证；部署时设置随机值 |
| `NAPCAT_WEBUI_TOKEN` | 无 | 云端 NapCat 管理页面密码，与 OneBot Token 分开设置 |
| `NAPCAT_IMAGE` | `mlikiowa/napcat-docker:latest` | Compose 使用的 NapCat 镜像，可指定版本标签 |
| `ADMIN_QQ_IDS` | 空集合 | 额外管理员 QQ 号，例如 `123456789,987654321` |
| `DATABASE_PATH` | `data/bot.db` | 本地数据库路径；Compose 固定为 `/app/data/bot.db` |
| `CHAT_PREFIX` | `/问` | 聊天命令前缀 |
| `EVALUATE_PREFIX` | `/锐评` | 锐评命令前缀 |
| `MIMIC_PREFIX` | `/模仿` | 模仿命令前缀 |
| `AI_COOLDOWN_SECONDS` | `8` | 同一用户在同一群的 AI 调用冷却秒数 |
| `CONTEXT_MESSAGE_LIMIT` | `20` | 聊天上下文查询条数，当前问题单独传入 |
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

修改 `.env` 后，执行 `sudo docker compose up -d --force-recreate` 更新容器环境。NapCat 的 Token 和网络设置只在首次创建配置时自动初始化；已有部署需要同步在 NapCat WebUI 修改。

不要执行 `docker compose down -v`，它会删除本项目的数据库、QQ 登录状态和 NapCat 配置数据卷。数据卷可以防止普通重建丢失数据，但不替代备份。

## 本地运行

需要 Python 3.11+ 和 uv。NapCat 需要单独安装、登录并配置连接。

```bash
git clone https://github.com/Nightsky-Lty/qq-chatbot.git
cd qq-chatbot
uv sync --frozen
cp .env.example .env
# 编辑 .env，至少配置 API Key、OneBot Token 和管理员 QQ 号
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
├── bot.py        # 命令分发、召集、聊天、锐评和模仿逻辑
├── database.py   # SQLite 表结构和读写操作
├── llm.py        # DeepSeek API 调用和异常处理
├── models.py     # OneBot 消息解析、文本和 @ 消息段构造
├── config.py     # 配置读取、校验和默认值
└── __init__.py   # Python 应用包标记
docs/deployment.md # Ubuntu 安装、部署和管理说明
Dockerfile         # Python 机器人镜像
compose.yaml       # 机器人与 NapCat 的容器编排
.env.example       # 环境配置模板，不包含真实密钥
pyproject.toml     # Python 依赖声明
uv.lock            # Python 依赖版本锁定
```

## 常见问题

**机器人不回复 `/帮助`**：检查两个容器是否运行、QQ 是否登录且在群中、NapCat 是否启用了反向 WebSocket。Python 日志出现 `OneBot reverse WebSocket connected` 表示连接已建立。

**帮助能回复，但 AI 不回复**：检查 API Key、模型名、余额和服务器网络，查看 bot 日志中的 DeepSeek 请求错误。

**浏览器打不开 NapCat**：管理页面只绑定服务器本机地址，需要先建立 SSH 转发，再访问电脑的 `16099` 端口；不能直接通过服务器公网 IP 的 `6099` 端口访问。

**模仿提示没有授权或样本不足**：目标用户必须先发送 `/允许模仿`，并在之后至少发送 5 条可保存文本。

**云服务器内存较小**：先用 `docker stats --no-stream` 观察 QQ、NapCat 和机器人占用，再决定是否增加 Swap 或升级内存。调用 DeepSeek API 不需要 GPU。

## 数据与当前限制

机器人会保存收到的群聊文本和命令，用于聊天上下文；聊天上下文及模仿样本会随相应请求发送给 DeepSeek。部署前应向群成员说明这一行为，用户可用 `/删除我的数据` 删除本人在该群的消息。

当前按单实例、单进程运行；同一 WebSocket 连接逐条处理消息，模型请求期间后续事件会等待。消息发送接口目前只提交请求，不跟踪 NapCat 返回的发送结果。

NapCatQQ 是第三方 QQ 协议端实现，本项目使用 OneBot 11 接入，并非 QQ 开放平台官方机器人。Python 配置和语法已做基础检查，Docker 镜像构建及真实 QQ 收发仍需在部署环境确认。
