# Ubuntu 云端部署

适用于全新 Ubuntu 22.04 / 24.04 / 26.04 云服务器。需要一个可用的 SSH 账号、DeepSeek API Key，以及用于机器人的 QQ 账号。

本项目的 Compose 同时运行 Python 机器人和 NapCat；机器人监听容器内的 8080 端口，NapCat 通过容器服务名 `bot` 发起反向 WebSocket 连接。数据库、QQ 登录状态、NapCat 配置和插件各自放在持久化数据卷中。

以下命令中的 `服务器IP` 和 `用户名` 需要替换成自己的值。Ubuntu 常见登录用户是 `ubuntu`，也可能是 `root`，以云厂商控制台为准；所有云端命令都在 SSH 登录后执行。

## 1. 从 GitHub 下载项目

在你的电脑上登录服务器：

```bash
ssh 用户名@服务器IP
```

如果使用 SSH 密钥，给 ssh 添加 `-i /你的私钥路径`，或者直接使用已配置的 SSH 别名（例如 `ssh qqbot`）。不要把私钥复制到项目目录或发到聊天里。

登录服务器后安装 Git 并下载项目：

```bash
sudo apt-get update
sudo apt-get install -y git
git clone https://github.com/Nightsky-Lty/qq-chatbot.git ~/qq-chatbot
cd ~/qq-chatbot
```

公开仓库可以直接下载；私有仓库需要 GitHub 访问授权，云服务器登录密钥不会自动获得 GitHub 访问权限。

目标目录必须不存在或为空。如果此前已经解压项目到 `~/qq-chatbot`，请使用另一个空目录（例如 `~/qq-chatbot-git`）下载，再把原有 `.env` 复制过去；不要删除已有数据。Compose 的项目名固定为 `qq-chatbot`，同一台服务器上的默认数据卷可以继续复用；不要在两个目录中同时运行同一部署。

## 2. 安装 Docker 和 Compose

确认系统版本：

```bash
cat /etc/os-release
```

下面按 [Docker 官方 Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/) 配置软件源。适用于尚未安装 Docker 的新服务器：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker compose version
```

如果已有 Docker 或其他容器服务，请按官方文档检查软件包兼容性，不要直接删除已有服务。

## 3. 填写配置

在服务器项目目录执行：

```bash
cp .env.example .env
chmod 600 .env
openssl rand -hex 24
openssl rand -hex 24
nano .env
```

两次 openssl 输出分别用于 `ONEBOT_ACCESS_TOKEN` 和 `NAPCAT_WEBUI_TOKEN`，填写配置时用真实值替换以下占位文字：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeekKey
ONEBOT_ACCESS_TOKEN=第一次生成的随机字符串
NAPCAT_WEBUI_TOKEN=第二次生成的随机字符串
ADMIN_QQ_IDS=你的管理员QQ号
```

管理员是你日常操作机器人的 QQ 号，不一定是机器人登录的 QQ 号；可以逗号分隔多个人。`nano` 保存按 Ctrl+O、回车，退出按 Ctrl+X。Key 和 Token 只填写在服务器的 `.env` 中。

Compose 会自动读取这些配置，并为 NapCat 设置反向 WebSocket URL。数据库路径在容器中固定为 `/app/data/bot.db`，由数据卷保存。

## 4. 启动

```bash
sudo docker compose config --quiet
sudo docker compose up -d --build
sudo docker compose ps
curl http://127.0.0.1:8080/health
```

第一次构建需要下载 Python、uv、NapCat 镜像和 Python 依赖。机器人容器健康检查通过后，NapCat 才会启动。健康检查返回 `{"status":"ok"}` 仅代表 Python 服务已启动，不代表 QQ 已登录。

镜像拉取失败时，先查看错误涉及 Docker Hub、GHCR 还是 PyPI，再处理对应网络问题；Docker 镜像下载由 Docker daemon 负责，不会自动使用终端的代理设置。NapCat 官方镜像也可以用 `NAPCAT_IMAGE` 选择明确的版本标签。

## 5. 在电脑上打开 NapCat，扫码登录

在电脑上新开一个终端，建立 SSH 转发并保持该终端打开：

```bash
ssh -N -L 16099:127.0.0.1:6099 用户名@服务器IP
```

浏览器打开 `http://127.0.0.1:16099/webui`，用 `.env` 的 `NAPCAT_WEBUI_TOKEN` 登录，按页面提示登录机器人的 QQ 账号。扫码账号应是准备运行机器人的 QQ 账号。

Compose 的管理端口绑定在服务器的 `127.0.0.1`，所以不需要在云防火墙开放 6099 或 8080；只需保留 SSH 访问。NapCat 和机器人通过容器内网通信，不需要域名。

首次启动会用 NapCat 官方镜像的反向 WebSocket 模板自动创建配置，消息格式为数组。按 [NapCat Docker 官方说明](https://github.com/NapNeko/NapCat-Docker)，环境变量只初始化缺失配置；已有配置需在 WebUI 中修改。

若登录后机器人未连接，在 NapCat WebUI 的 OneBot 11 网络配置检查：

- 类型：反向 WebSocket 客户端，启用。
- URL：`ws://bot:8080/onebot/v11/ws`。
- Token：与 `.env` 的 `ONEBOT_ACCESS_TOKEN` 相同。
- 消息格式：数组（array）。
- 重连：启用，间隔约 30 秒。

自动创建的 URL 会带 `?access_token=...`，当前机器人支持这种认证方式；手动配置可以用上面不带查询参数的 URL 和独立 Token 字段。不要同时保留两个启用的反向连接，以免事件重复上报。

## 6. 进群确认功能

将机器人 QQ 加入目标群，用管理员 QQ 发送：

```text
/帮助
/名单 添加 @某人
/关键词 添加 开黑
/问 你好
```

`/帮助` 验证 QQ 收发链路，`/问` 验证 DeepSeek 请求。机器人账号无需成为群管理员；维护名单的发令用户必须是群主、群管理员，或配置在 `ADMIN_QQ_IDS` 中。

查看日志：

```bash
sudo docker compose logs --tail=100 bot
sudo docker compose logs --tail=100 napcat
```

Python 日志出现 `OneBot reverse WebSocket connected` 表示 NapCat 已连接。日志可能包含身份、Token、消息内容或接口错误，分享前应遮盖这些信息。

## 日常管理

```bash
# 修改 .env 后重新创建容器，使新配置生效
sudo docker compose up -d --force-recreate

# 修改代码后重新构建机器人镜像
git pull --ff-only
sudo docker compose up -d --build bot

# 查看资源占用
sudo docker stats --no-stream

# 停止服务，保留数据卷；恢复使用 docker compose up -d
sudo docker compose down
```

Docker 开机自启和 `restart: unless-stopped` 会在服务器重启后重新启动容器；如果主动停止了服务，需要手动恢复。QQ 登录异常时仍需按页面提示重新验证。

不要使用 `docker compose down -v`：它会删除本项目的数据库、QQ 登录状态和 NapCat 配置。普通重建会保留数据，但数据卷不等于异地备份，长期运行应另行备份数据库和配置。

若服务器只有 2 GB 内存，先观察两个容器的实际占用。进程异常退出或内存不足时先检查日志和资源状态，再考虑添加 Swap 或升级内存。
