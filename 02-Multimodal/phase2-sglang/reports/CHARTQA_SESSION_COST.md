# 多轮图表追问的缓存与请求代价拆解

## 实测结果

2026-11-24 11:31（北京时间），完成 96 条计量请求、24 条预热和 60 对消息哈希校验。每轮三个客户端请求的生命周期均重叠，发送起点最大间隔 2.62 ms。全部 120 条请求自然结束，无重排回收。96 条计量请求中，每组均有 44/48 道的数值与指定答案完全一致，两组对应文本逐对相同。失分集中在一条跨图求平均的第四轮回复，模型反复输出 `75`，参考值为 `55`。

表中每个轮次、每种条件有 12 条计量请求。未命中组实际命中 Token 始终为 0，命中组首次请求同样为 0。所有正命中在 API usage 和内部请求 Trace 中一致。

| 轮次 | 实际视觉 Token | 命中组实际前缀 Token | 命中前缀覆盖的视觉 Token | 客户端 TTFT P50（命中 / 未命中，ms） | 客户端 TTFT P90（命中 / 未命中，ms） |
|---|---:|---:|---:|---:|---:|
| 1 · 首张图 | 777 | 0 | 0 | 203.2 / 200.9 | 209.8 / 202.9 |
| 2 · 同图追问 | 777 | 816–818 | 777 | 110.7 / 207.0 | 119.2 / 209.6 |
| 3 · 追加新图 | 1,554 | 844–847 | 777 | 248.8 / 386.7 | 267.2 / 399.4 |
| 4 · 继续追问 | 1,554 | 1,654–1,664 | 1,554 | 156.8 / 398.8 | 176.3 / 419.2 |

下表是各阶段 P50，单元格顺序为命中组 / 未命中组，单位 ms。

| 轮次 | API 收到至派发 | 图像预处理 | 派发至首次 Forward | Scheduler 排队 | Prefill 墙钟窗口 | Prefill 后到首输出 | 首输出后至完成 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.4 / 30.5 | 26.8 / 23.6 | 36.2 / 34.6 | 0.34 / 0.33 | 114.4 / 108.1 | 0.84 / 0.75 | 27.4 / 27.5 |
| 2 | 34.5 / 30.4 | 26.6 / 23.5 | 36.0 / 34.6 | 0.34 / 0.33 | 23.4 / 124.9 | 0.82 / 0.82 | 26.7 / 26.7 |
| 3 | 57.4 / 52.4 | 43.4 / 39.2 | 51.0 / 50.4 | 0.38 / 0.35 | 110.6 / 223.3 | 0.89 / 0.87 | 32.9 / 34.0 |
| 4 | 53.5 / 63.4 | 40.6 / 48.5 | 51.0 / 49.9 | 0.39 / 0.36 | 23.8 / 232.2 | 0.89 / 0.93 | 31.8 / 35.5 |

第 2 和第 4 轮即使已匹配全部历史视觉占位符，API 仍要处理本轮重传的图像。可见的 TTFT 差异主要与 Prefill 窗口缩短相伴。第 3 轮只覆盖历史图的 777 个视觉占位符，仍有第二张图的输入需要纳入新的前缀。

视觉编码在预热中执行了两次，CUDA 批窗口为 420.9 ms 和 26.4 ms，首次观测包含冷启动影响。96 条计量请求均未再次调用 ViT，确认两组都使用已预热的进程内视觉 embedding 状态。配对实验衡量的是这一视觉状态下的 **KV 前缀复用**。视觉输入准备的 CUDA 批窗口仍单独保存在批日志。首次运行因重复安装观测钩子漏记了预处理计时，修复安装逻辑并重跑后才归档最终结果。

[统计汇总](chartqa-session-cost/summary.json) · [逐请求记录](chartqa-session-cost/requests.jsonl) · [GPU 视觉批次](chartqa-session-cost/vision_batches.jsonl) · [实验清单](chartqa-session-cost/manifest.json)

## 负载与配对

单张 RTX 5090，SGLang 0.5.23、Qwen2.5-VL-7B-Instruct BF16，FCFS，运行请求上限 4。三个会话用同一轮屏障并发发出请求，每个会话四轮。第 1 轮发一张图，第 2 轮同图追问，第 3 轮追加第二张图，第 4 轮继续追问。图表来自阶段 1 的自制样例，固定中档 602,112 像素上限，保持方向和图像版本。输入含当前轮和全部历史消息。历史助手短答由配置固定，每轮仍执行真实推理。

两组使用同一进程，启用 Radix Cache。命中组在一个会话内保持 `cache_salt` 不变，未命中组给每个请求独立盐值。不同会话、轮次组、重复实验使用独立命名空间。相同会话与位置的消息 SHA-256 必须匹配。各组先跑一次完整预热，再做四次配对重复，轮换两组执行顺序。客户端同一轮发三个请求，待该轮完成后发下一轮。

本实验隔离语言模型前缀 KV 的可复用状态。图像预处理缓存设为 0，全局和多模态前缀缓存开关关闭，两组保持同一进程内的视觉 embedding 缓存（SGLang 0.5.23 默认 100 MiB）。GPU 计时单独检查视觉编码是否真正执行。

## 请求级计量

| 指标 | 口径 |
|---|---|
| 实际命中 Token | SGLang 最终输出 `meta_info.cached_tokens`，有 API usage 时交叉核对 |
| 实际视觉 Token | 处理器网格合并值、图像占位符区间长度与占位符数逐图一致 |
| 前缀覆盖视觉 Token | 实际命中的前缀长度与图像占位符区间的交集 |
| 多模态预处理 | Qwen-VL 图像处理器异步调用的墙钟耗时，在写出 Trace 前停止计时 |
| API 处理、派发到首次 Forward、排队 | API 收到请求至派发完成、派发至 Scheduler 首次 Forward、Scheduler 自报 `queue_time`。排队字段单列，派发至 Forward 还包含其他间隔 |
| Prefill | SGLang 首次 Forward 到 Prefill 完成的墙钟窗口，包含经过该窗口的批处理与多模态工作 |
| 视觉 CUDA 时间 | 多模态输入准备及实际 ViT Forward 的 CUDA event，按批次存一行，并联接涉及的请求 ID。批次时间可共享 |
| 首输出与生成收尾 | Prefill 完成到首个服务端输出、首个服务端输出到请求完成 |
| 客户端 TTFT | 从同机客户端发送请求到首段非空流式正文，包含请求序列化和传输 |

阶段耗时可能嵌套或并发，优先逐请求比较同一字段。CUDA event 与服务端窗口分别报告。

## 复现

使用[部署说明](DEPLOYMENT_SMOKE.md)中的隔离运行环境、模型权重与可用的 CUDA 工具链。在一张空闲卡启动实验实例，端口为 `127.0.0.1:30001`。如同一张卡还在运行阶段 2 默认实例，应先在该卡释放模型占用。

```bash
cd 02-Multimodal/phase2-sglang
export PYTHON_BIN="$(command -v python3.12)"
export MODEL_PATH=/absolute/path/to/Qwen2.5-VL-7B-Instruct
export CUDA_VISIBLE_DEVICES=1
SESSION_TRACE=1 SGLANG_CONFIG=configs/sglang-session-server.json \
  RUN_LOG_DIR="$PWD/results/session-server" bash scripts/start_sglang_smoke.sh
```

在同机另一终端复放并汇总。服务器分目录部署时，把样例路径替换为阶段 1 的 `examples/` 绝对路径。

```bash
cd 02-Multimodal/phase2-sglang
export PYTHONPATH="$PWD/.runtime:$PWD"
out="results/chartqa-session-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m benchmark.chartqa_session_replay \
  --config configs/chartqa-same-image-session.json \
  --fixtures ../phase1-assistant/examples --output "$out"
python3.12 -m analysis.chartqa_session_cost --run "$out"
```

[服务器配置](../configs/sglang-session-server.json) · [会话工作负载](../configs/chartqa-same-image-session.json)
