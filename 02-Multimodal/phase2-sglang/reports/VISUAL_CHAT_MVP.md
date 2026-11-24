# 目见 · 多用户视觉助手

目见是阶段 2 的浏览器聊天界面，后端使用单张 RTX 5090 上的 SGLang 0.5.23 和 Qwen2.5-VL-7B-Instruct。支持独立浏览器身份、多个会话、SQLite 历史、每轮最多四张图片、同图追问、流式响应和停止生成。浏览器历史列表及附件由当前 Cookie 所属会话隔离。会话使用独立 Radix Cache 命名空间，成功生成后保存本轮历史，下一轮使用实际历史回复。固定中档视觉预算，每个对话最多 10 轮，按 8K 服务端上下文预检。

## 调度

沿用小规模持续负载对照中合并 TTFT P95 最低的估计未缓存 Token 策略。使用本地 tokenizer 估计完整 prompt Token，将图像占位换成图像预算对应的视觉 Token 估计，并扣除上一轮可复用的会话 prompt 长度。校准仿射变换给出毫秒代价，等待每 1 ms 扣除 0.35 ms 的代价。等待满 100 ms 进入优先组，组内按引擎入队时间排序。引擎直接按该规则重排等待队列。页面分别显示首字延迟、估计未缓存 Token 和服务端实际前缀命中。[小规模对照](CHARTQA_PREFILL_SIGNALS.md)

## 单卡验收

- 浏览器上传合成柱状图，读出 20、50、35、65；同图追问 Q4 与 Q1 的差值，结果为 45。追问实际命中前缀 444 Token。
- 从两个独立 Cookie 并发发起单图和双图请求，结果分别为 65 和 20。第一位用户继续追问得到 45，实际命中前缀 396 Token。两位用户交叉读取会话均返回 404，历史分别保留两轮和一轮。
- 面向内网验收的进程绑定回环地址。当前浏览器身份对应独立 Cookie，可使用不同浏览器配置文件测试多用户。

## 部署

在 `phase2-sglang` 下部署。先按 SGLang 部署说明设置 CUDA 路径、选择一张空闲 GPU、配置 `PYTHON_BIN` 和 `MODEL_PATH`、准备固定版本的隔离运行时。[运行时说明](CHARTQA_SESSION_COST.md#复现)

在服务端第一个终端启动 SGLang。

```bash
SESSION_TRACE=1 SCHEDULE_COMPARE=1 \
  SCHEDULE_RULES="$PWD/configs/chat-scheduling.json" \
  SGLANG_CONFIG=configs/sglang-chat-server.json SGLANG_VLM_CACHE_SIZE_MB=12 \
  RUN_LOG_DIR="$PWD/results/chat-engine" bash scripts/start_sglang_smoke.sh
```

在第二个终端使用相同的 Python 和模型路径启动 Web 服务。默认 Web 端口为 7863，默认 SGLang 端口为 30002。可用 `CHAT_DB` 指定 SQLite 文件，用 `CHAT_PORT` 指定 Web 端口。

```bash
bash scripts/start_visual_chat.sh
```

从客户机使用 SSH 端口转发，浏览器打开 `http://127.0.0.1:7863/`。

```bash
ssh -N -L 127.0.0.1:7863:127.0.0.1:7863 -p <SSH_PORT> <USER>@<GPU_HOST>
```

## 临时公网 HTTPS 验收

公网演示使用单独的 Web 进程和 SQLite 文件，复用回环地址上的 SGLang。设置 `CHAT_AUTH_FILE` 启用浏览器 HTTP Basic 认证。凭据文件为一行 `用户名:随机密码`，权限设为 `0600`，保存在 Git 忽略的运行目录。示例为 GPU 服务器上的两个终端。

```bash
umask 077
mkdir -p results/chat-public
test -s results/chat-public/credentials || "$PYTHON_BIN" -c \
  'from secrets import token_urlsafe; print("demo:"+token_urlsafe(21))' > results/chat-public/credentials

CHAT_PORT=7864 CHAT_AUTH_FILE="$PWD/results/chat-public/credentials" \
  CHAT_DB="$PWD/results/chat-public/chat.sqlite3" bash scripts/start_visual_chat.sh
```

```bash
cloudflared tunnel --protocol http2 --no-autoupdate --url http://127.0.0.1:7864
```

浏览器用 NDJSON 接收流式输出，原有 API 客户端仍可使用 SSE。Cloudflare Quick Tunnel 地址与进程生命周期绑定，重新启动后由日志提供新地址。[Cloudflare Quick Tunnel 说明](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)

## 固定 HTTPS 入口（ngrok 免费开发域名）

ngrok 免费账户分配一个固定 HTTPS 开发域名。Agent 重启可继续使用同一地址，Agent 与 GPU 服务在线时可访问。首次浏览器访问会先显示 ngrok 提示，再进入演示账号登录。[固定域名说明](https://ngrok.com/docs/gateway/domains) · [免费版访问提示](https://ngrok.com/pricing)

沿用受登录保护的公网 Web 进程 `127.0.0.1:7864` 及 `results/chat-public/chat.sqlite3`。GPU 服务器上的 ngrok Agent 默认位于 `results/chat-public/ngrok`。账户授权后在 `results/chat-public/ngrok.yml` 写入 v3 Agent 配置，权限设为 `0600`。Agent 凭据与演示账号是两套独立凭据。

```yaml
version: 3
agent:
  authtoken: YOUR_NGROK_AGENT_TOKEN
  web_addr: false
```

```bash
CHAT_PUBLIC_URL=https://YOUR_ASSIGNED_NAME.ngrok-free.app bash scripts/start_visual_chat_ngrok.sh
```

启动脚本检查本机 Web 入口确实要求登录，仅向该端口转发，并关闭 HTTP 请求检查。切换后从公网验证匿名访问返回 401，独立 Cookie 无法互读会话，图表问答及追问逐段返回。接着重启 Agent，使用同一 URL 和 Cookie 验证历史与追问。浏览器从旧临时域名初次进入固定域名时会建立新的浏览器身份。

2026-11-24 固定入口已启用。[打开「目见」](https://lethargic-aeration-elevate.ngrok-free.dev/)

在同一公网 HTTPS 地址上创建两个独立 Cookie 会话，重启 ngrok Agent 后用原 Cookie 继续对话。匿名页面及 API 访问返回 401，使用演示凭据返回 200。两个会话的历史列表各自独立，跨会话读取返回 404。图表追问答出差值 45，服务端报告命中 398 个前缀 Token。另一会话可回忆自己的文字上下文。长回复实测收到 65 次 NDJSON 增量，客户端在 0.552 秒收到首个增量，于 1.186 秒收到完成事件。凭据文件仅保存在运行目录和用户本机。

## 自动启动与故障恢复

使用 Supervisor 4.3.0 管理固定公网入口的 SGLang、Web 和 ngrok。启动时读取独立的运行目录 `results/chat-public`，其中保存演示密码、ngrok Agent 配置、固定域名和历史 SQLite。控制套接字权限为 `0600`，控制入口只在本机。ngrok 等待本地模型健康检查返回 200，且 Web 入口对匿名访问返回 401，再发布固定地址。[Supervisor 配置文档](https://supervisord.org/configuration.html)

运行中 Supervisor 会自动拉起退出的进程。SGLang 的独立进程组会在主进程退出后被清理，避免残留 worker 继续占用显存。健康检查在启动宽限期后每 10 秒探测本机 SGLang 和 Web，连续三次失败才发起重启。ngrok 的连接由 Agent 自行重连，进程退出时由 Supervisor 拉起。

首次部署可以用 GPU 环境的 Python 安装隔离的 Supervisor。使用宿主提供的 Python 创建目录时可以跳过 `ensurepip`。

```bash
umask 077
"$PYTHON_BIN" -m venv --without-pip results/chat-public/supervisor-venv
"$PYTHON_BIN" -m pip --python "$PWD/results/chat-public/supervisor-venv/bin/python" install supervisor==4.3.0
```

在 `results/chat-public/stack.env` 通过 `export` 指定 GPU 环境的 `PYTHON_BIN`、`MODEL_PATH`、`CUDA_VISIBLE_DEVICES` 和所需 CUDA 变量，固定 HTTPS URL 单独写入 `results/chat-public/ngrok-url`。迁移前停止同端口上手动启动的旧进程，再运行以下命令。

```bash
bash scripts/visual_chat_stack.sh start
bash scripts/visual_chat_stack.sh ctl status
bash scripts/visual_chat_stack.sh ctl shutdown
bash scripts/visual_chat_stack.sh start
```

当前宿主是由 tini 管理的托管 Pod，启动命令由算力平台控制，Pod 内没有可启用服务的 systemd 实例。服务级运行管理由现有 Supervisor 实例负责。若需要 Pod 重建后无人值守启动，在工作区的启动钩子里执行项目的 `scripts/visual_chat_stack.sh start`，或者将 `scripts/visual_chat_stack.sh run` 配置为由平台运行的专用前台进程。

2026-11-24 整套服务重启验收通过。旧 Supervisor 退出，SGLang 和公网 Web 端口关闭，随后由新 Supervisor 统一拉起三个服务。公网恢复到同一个 HTTPS 地址。匿名请求返回 401，演示账号登录成功。重启前创建的两个会话使用原 Cookie 恢复各自的历史，双向跨会话读取返回 404。模型重启后首次图表追问从持久化历史重建上下文，正确回答差值 45，GPU 前缀缓存命中为 0。另一会话也正确复述自己的文字上下文。长回复实测 65 个 NDJSON 增量，首段 0.569 秒，完成 1.199 秒。

故障注入验收通过。暂停 Web 进程，三次本机探测失败后由健康检查发起重启。强制结束 ngrok Agent 和 SGLang 主进程，各由 Supervisor 自动拉起，SGLang 的旧 worker 组被清理。三个新进程全部运行后，从公网再次验证登录和一条流式问答。整套重启记录中，从旧 ngrok 下线到同一地址重新上线约 53 秒。
