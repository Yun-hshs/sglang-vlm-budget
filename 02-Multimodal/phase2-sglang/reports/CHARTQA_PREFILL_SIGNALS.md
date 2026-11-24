# 多轮持续负载下的 Prefill 代价与调度对照

## 实测结果

在单张 RTX 5090 上用 SGLang 0.5.23 运行 Qwen2.5-VL-7B-Instruct，固定中档 602,112 像素。新工作负载按完整会话模板划分，校准和验证各 9 个模板、36 个独立会话、6 张独立 ChartQA 图片。两组图像 SHA-256 无交集，也与上一轮预测实验的图像无交集。每个会话 3–5 轮，覆盖同图追问及追加新图。每条轨迹 144 请求，在前三轮连续窗口内每 64 ms 计划到达一条，约 15.63 请求/s。

### Prefill 误差与同队列排序

校准和验证各用一条 FCFS 探针轨迹，分别预热一次、计量两次，各 288 条计量请求。预测目标是 SGLang 记录的首次模型 Forward 至 Prefill 完成的墙钟耗时。每条请求第一次进入调度快照时的特征用于误差计算。排序以同一验证快照中的两个请求为一对，去掉真实耗时差不超过 10 ms 的请求对，每对只计第一次，预测平手计半分。88 个共享队列快照产生 1,294 对。

| 代价信号 | MAE（ms） | RMSE（ms） | P90 绝对误差（ms） | 同队列排序一致率 |
|---|---:|---:|---:|---:|
| 估计未缓存 Token | 42.33 | 52.66 | 73.14 | 73.61% |
| 冻结 Ridge | 40.52 | 53.69 | 91.54 | 74.85% |
| 本地校准模型 | **32.04** | **43.99** | 78.97 | 74.54% |
| SGLang 原生 SPF 的未缓存 Token 分数 | 40.06 | 53.85 | 83.98 | 60.05% |

本地校准模型的 MAE 比冻结 Ridge 低 20.9%，排序一致率相差 0.31 个百分点。估计未缓存 Token 的 P90 绝对误差最低。验证探针有 108 条实际前缀命中、76 条请求的视觉缓存缺失特征大于零。当前模型的两个批负载特征在同一个快照中对所有候选请求取值相同。作为线性加项，它们可以校准不同批负载下的耗时，同队列请求的先后由各自的请求特征决定。模型输入保存在[验证预测明细](chartqa-prefill-signals/validation-predictions.jsonl.gz)，系数和快照统计见[离线汇总](chartqa-prefill-signals/model.json)。

### 线上调度与用户等待

2026-11-24 14:23–14:26（北京时间），四组各完成 432 条计量请求，合计 1,728 条。另有 576 条预热请求。每组 162 次实际前缀命中，所有计量请求均正常结束。TTFT P95 和最差用户基于全部三轮。逐轮 P95 单列，便于核对波动。

| 策略 | 含排队 TTFT P50 / P95（ms） | 三轮 TTFT P95（ms） | 最差用户排队 P95（ms） | 全部用户单次最长排队（ms） | 整轨迹请求/s |
|---|---:|---:|---:|---:|---:|
| 估计未缓存 Token + 老化 | 219 / **1,800** | 1,795 / 1,784 / 1,828 | **1,786** | 1,799 | 12.25 |
| 冻结 Ridge + 老化 | 233 / 1,816 | 1,789 / 1,822 / 1,824 | 1,805 | 1,805 | 12.25 |
| 本地校准模型 + 老化 | 502 / 2,477 | 2,712 / 1,804 / 1,823 | 2,244 | 2,652 | 11.93 |
| SGLang 原生 SPF | **162** / 2,409 | 2,309 / 2,323 / 2,282 | 3,180 | 3,250 | 12.27 |

未缓存 Token 和冻结 Ridge 的 TTFT P95 逐轮范围重叠，且两组最差用户均为 `validation-5-2-r02`。原生 SPF 的中位数最低，最差用户 `validation-3-0-r01` 的等待 P95 约 3.18 秒。该组 9 条请求因等待同会话前一轮而延迟派发，发生 51 个 API 实际接收逆序对。

本地模型第一轮图像预处理 P95 为 196 ms，最长 1,176 ms，另外两轮 P95 为 21 和 26 ms。21 条同会话依赖延迟均出现在第一轮。保留该轮数据。本地模型第二、三轮的 TTFT P95 与未缓存 Token 和 Ridge 接近。这组记录支持本地模型降低 Prefill 毫秒误差，线上尾延迟是否改善仍需独立重复测量。

[逐用户等待 CSV](chartqa-prefill-signals/online/user_wait.csv) · [总表与逐轮明细](chartqa-prefill-signals/online/summary.json) · [同一计划到达轨迹](chartqa-prefill-signals/online/manifest.json) · [运行时与原生 SPF 日志](chartqa-prefill-signals/runtime-check.txt)

## 策略与计量口径

- **估计未缓存 Token**使用估计输入 Token 减去会话可复用的上轮 prompt 长度。仅在校准组对这个计数拟合正斜率的仿射变换，将单位换为毫秒，共用等待老化系数。排序诊断使用原始计数。
- **冻结 Ridge**沿用之前实验的输入长度、图像、历史轮数和估计前缀特征及已有系数。运行清单记录原模型摘要的 SHA-256。
- **本地校准模型**使用估计未缓存 Token \(U\)、估计可复用前缀 \(C\)、注意力计算量代理 \(U(C+U/2)\)、当前视觉 embedding 缓存中缺失且在估计 KV 前缀之外的视觉 Token 数、当前运行批请求数和等待队列长度。沿用标准化 Ridge 拟合，正则系数 1.0。视觉缓存通过只读逐图哈希查询取得，因此这个特征表示决策时的缺失估计。
- **原生 SPF**直接切换到 SGLang 的 `shortest-prefill-first`，保留其实际 Radix 前缀匹配、重复前缀降权和 Prefill 分块规则。线上使用原生策略。离线基于 FCFS 探针内 SGLang 查到的同一时刻前缀长度，按原生未缓存 Token 分数诊断排序。误差列只为该分数做校准组的毫秒换算，排序列只比较原生代价分数。

三种自定义策略采用相同老化规则。等待 1 ms 减少 0.35 ms 预测代价，调度等待满 100 ms 的请求进入优先组，按引擎入队顺序排列。原生 SPF 保留自身策略。每种策略预热一次、正式重复三次，每轮轮换执行顺序。每个策略、每轮用独立 KV 缓存盐，跨轮复用计划和视觉缓存配置保持一致。视觉 embedding 缓存固定 12 MiB，以在同一小轨迹同时观察缓存命中与缺失。旧实验使用了不同的视觉缓存容量。

TTFT 从计划到达至收到首个非空正文 Token，包含同会话依赖、HTTP 和引擎等待。用户等待为计划到达至实际派发的滞后加服务端自报队列时间。最差用户取每位用户请求等待 P95 的最大值。每位用户有 3–5 轮 × 3 次测量，逐用户表同时保留最大单次等待。同一会话按顺序派发，因此可能发生实际 API 接收逆序。线上比较前逐项核对计划到达、请求内容哈希、缓存类别和冻结 Ridge 预测。

## 复现

使用已固定的 SGLang 运行时及 CUDA 配置。[部署说明](CHARTQA_SESSION_COST.md#复现)在 `phase2-sglang` 下设置本地模型路径、仅一张 GPU、隔离 Python 环境和新的日志目录。服务与请求回放在同一主机执行。

```bash
export MODEL_PATH=/absolute/path/to/Qwen2.5-VL-7B-Instruct
export PYTHON_BIN=/absolute/path/to/compatible/python3.12
export CUDA_VISIBLE_DEVICES=1
export RUN_LOG_DIR="$PWD/results/prefill-signals-server"
export PREFILL_QUEUE_TRACE="$RUN_LOG_DIR/queue.jsonl"
export PREFILL_CALIBRATOR_PATH="$PWD/results/prefill-signals-model.json"

SESSION_TRACE=1 SCHEDULE_COMPARE=1 SGLANG_VLM_CACHE_SIZE_MB=12 \
  SCHEDULE_RULES="$PWD/configs/chartqa-schedule-prefill-signals.json" \
  SGLANG_CONFIG=configs/sglang-schedule-server.json bash scripts/start_sglang_smoke.sh
```

在同机另一终端设置相同的路径变量与解释器。从已处理的阶段 1 ChartQA 数据构建工作负载，校准，再回放验证轨迹。

```bash
set -euo pipefail
export PYTHONPATH="$PWD/.runtime:$PWD"
fixtures=../phase1-assistant/data/processed/ablation
workload=results/prefill-signals-workload.json

"$PYTHON_BIN" -m benchmark.build_prefill_sessions \
  --config configs/chartqa-prefill-signals-workload.json --dataset "$fixtures/test.jsonl" \
  --exclude-manifest reports/chartqa-prefill-prediction/manifest.json --output "$workload"

common=(--workload "$workload" --fixtures "$fixtures" --model-path "$MODEL_PATH"
        --vision-trace "$RUN_LOG_DIR/vision.jsonl" --session-trace "$RUN_LOG_DIR/session.jsonl")
for split in train validation; do
  "$PYTHON_BIN" -m benchmark.chartqa_schedule_compare "${common[@]}" \
    --config configs/chartqa-prefill-local-probe.json --split "$split" \
    --output "results/prefill-signals-probe-$split"
done

validation=results/prefill-signals-probe-validation
"$PYTHON_BIN" -m analysis.chartqa_prefill_signals --workload "$workload" \
  --train results/prefill-signals-probe-train --validation "$validation" \
  --queue-trace "$PREFILL_QUEUE_TRACE" --output "$PREFILL_CALIBRATOR_PATH"

online=results/prefill-signals-compare
"$PYTHON_BIN" -m benchmark.chartqa_schedule_compare "${common[@]}" \
  --config configs/chartqa-schedule-prefill-signals.json --split validation \
  --calibrator "$PREFILL_CALIBRATOR_PATH" --reference-trace "$validation/manifest.json" \
  --output "$online"
"$PYTHON_BIN" -m analysis.chartqa_schedule_compare --run "$online"
```

[工作负载](chartqa-prefill-signals/workload.json) · [校准探针](chartqa-prefill-signals/probe-train/) · [验证探针](chartqa-prefill-signals/probe-validation/) · [引擎队列快照](chartqa-prefill-signals/queue.jsonl.gz) · [线上逐请求及配置](chartqa-prefill-signals/online/)
