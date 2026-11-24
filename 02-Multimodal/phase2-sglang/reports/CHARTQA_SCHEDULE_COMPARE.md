# 单卡多轮视觉请求的四种调度对照

## 实测结果

2026-11-24 11:46–11:49（北京时间），4 种策略共完成 432 条计量及 144 条预热，全部正常完成，无截断或回收。客户端和 API 实际接收顺序均与轨迹相同。每组 108 条计量请求中 45 条实际命中前缀。36 个固定会话轮次中，9 个轮次的实际命中长度在重复之间浮动 1 Token，命中与否始终一致。对应轮次的回答文本相同。

| 策略 | 含排队 TTFT P50 / P95（ms） | 排队 P50 / P95（ms） | 整轨迹请求/s | 轮内请求/s | 输出 Token/s |
|---|---:|---:|---:|---:|---:|
| FCFS | 241.3 / 547.3 | 92.3 / 331.6 | 3.962 | 14.00 | 15.41 |
| SGLang LPM | 233.3 / 534.1 | 40.4 / 300.7 | 3.963 | 14.00 | 15.41 |
| 代价优先 | 230.8 / 509.0 | 45.4 / 252.0 | 3.955 | 13.55 | 15.38 |
| 代价优先 + 老化 | 227.5 / 542.4 | 56.2 / 304.0 | 3.955 | 13.76 | 15.38 |

### 长请求与饥饿观察

长请求的冻结预测阈值为 100.26 ms，每组有 27 条。四组均全部完成，且没有长请求达到 500 ms 排队观察阈值。有限次短突发只支持本次观察，持续高负载的饥饿风险待另测。

| 策略 | 长请求排队 P95 / 最大（ms） | 长请求 TTFT P95（ms） | 最多被多少个后到请求先执行 |
|---|---:|---:|---:|
| FCFS | 352.4 / 352.5 | 650.6 | 2 |
| SGLang LPM | 349.6 / 357.6 | 647.9 | 4 |
| 代价优先 | 441.2 / 472.8 | 714.4 | 5 |
| 代价优先 + 老化 | 347.2 / 350.0 | 639.3 | 5 |

代价优先将整体 TTFT P95 从 547.3 ms 降到 509.0 ms，长请求排队 P95 则比 FCFS 增加 88.8 ms。老化把长请求排队 P95 从 441.2 ms 降到 347.2 ms，整体 TTFT P95 回升到 542.4 ms。整轨迹吞吐接近，轮内吞吐也未显示代价优先的吞吐提升。

### 各用户排队时间

每格为 P50 / P95，单位 ms。

| 用户 | FCFS | LPM | 代价优先 | 代价优先 + 老化 |
|---|---:|---:|---:|---:|
| validation-1-0 | 99.2 / 215.8 | 141.1 / 165.9 | 48.1 / 131.1 | 77.3 / 117.3 |
| validation-1-1 | 0.4 / 133.9 | 0.6 / 348.3 | 0.5 / 461.3 | 0.5 / 339.7 |
| validation-1-2 | 130.0 / 335.0 | 39.4 / 106.4 | 45.8 / 184.1 | 81.6 / 155.6 |
| validation-3-0 | 159.3 / 352.4 | 175.0 / 353.3 | 208.8 / 240.3 | 235.0 / 349.9 |
| validation-3-1 | 0.6 / 153.0 | 0.7 / 190.6 | 0.7 / 266.4 | 12.0 / 126.5 |
| validation-3-2 | 21.5 / 156.2 | 13.4 / 98.0 | 12.9 / 99.0 | 6.7 / 112.7 |
| validation-5-0 | 139.7 / 326.9 | 25.4 / 103.3 | 26.1 / 106.2 | 53.4 / 104.6 |
| validation-5-1 | 142.5 / 187.2 | 126.5 / 217.7 | 29.6 / 143.2 | 30.2 / 128.4 |
| validation-5-2 | 129.9 / 228.9 | 124.2 / 232.3 | 77.2 / 149.1 | 58.9 / 278.8 |

[分策略与逐用户统计](chartqa-schedule-compare/summary.json) · [到达轨迹与运行清单](chartqa-schedule-compare/manifest.json) · [逐请求记录](chartqa-schedule-compare/requests.jsonl.gz)

## 对照设计

复用 [Prefill 预测实验](CHARTQA_PREFILL_PREDICTION.md)的 9 个验证会话与训练完成的模型系数。每个会话 3–5 轮，包含同图追问和追加新图。固定中档 602,112 像素与历史助手短答。每次复放 36 条请求，按轮在 2,200 ms 间隔发出，每轮请求间隔 12 ms。每个策略预热一次、正式重复三次，轮转策略顺序。运行器验证对应请求的消息哈希、客户端发送顺序及每个会话前一轮先结束。保持 3 个运行请求上限，为短突发留下引擎排队压力。试跑中首个冷启动波次耗时约 1.74 s，超过原定 1.3 s 间隔，故统一延长轮间隔后重新执行四组。

单张 RTX 5090、SGLang 0.5.23、同一进程和 Radix Cache 配置。会话缓存命名空间的选择在四组间相同，试验和策略使用独立盐。实际命中数由服务端报出。视觉 embedding 缓存在同一次实验内共享。

| 策略 | 待处理队列规则 |
|---|---|
| FCFS | 调用 SGLang 的先到先服务策略 |
| SGLang LPM | 调用 SGLang 原生最长前缀匹配策略 |
| 代价优先 | 按派发前预测的 Prefill 毫秒升序，同分按加入引擎队列时间 |
| 代价优先加等待老化 | 预测毫秒减去 0.35 × 等待毫秒；已等待 220 ms 的请求优先于其余请求，超时组按入队先后排列 |

实验钩子只接管带实验前缀的单引擎请求，预测值由客户端用冻结模型计算，在请求 ID 中携带。220 ms 是优先级阈值；运行中的批次继续执行。模型、缓存和批处理上限固定。

## 指标与复现

TTFT 从计划到达时刻计至客户端收到首个非空流式正文，包含客户端派发滞后和引擎排队。用户排队时间为客户端派发滞后加 SGLang 自报的调度队列耗时；两者原始值均保存。另记录服务端接收到首次 Forward 的时间。长请求定义为轨迹中预测 Prefill 代价最高四分位，请求若排队达到 500 ms 则记一次观察告警，同时报告最长排队、被后来请求先执行的最大次数和完成情况。吞吐以首个计划到达至最后完成为主，另报告排除固定轮间空档后的轮内吞吐。

```bash
cd 02-Multimodal/phase2-sglang
# CUDA、隔离依赖与 MODEL_PATH 同多轮缓存实验
SESSION_TRACE=1 SCHEDULE_COMPARE=1 \
  SCHEDULE_RULES="$PWD/configs/chartqa-schedule-compare.json" \
  SGLANG_CONFIG=configs/sglang-schedule-server.json \
  RUN_LOG_DIR="$PWD/results/schedule-server" bash scripts/start_sglang_smoke.sh
```

同机另一个终端运行。

```bash
cd 02-Multimodal/phase2-sglang
export PYTHONPATH="$PWD/.runtime:$PWD"
out="results/schedule-compare-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m benchmark.chartqa_schedule_compare \
  --fixtures ../phase1-assistant/data/processed/ablation \
  --model-path /absolute/path/to/Qwen2.5-VL-7B-Instruct --output "$out"
python3.12 -m analysis.chartqa_schedule_compare --run "$out"
```

[多轮服务部署](CHARTQA_SESSION_COST.md#复现) · [实验配置](../configs/chartqa-schedule-compare.json)
