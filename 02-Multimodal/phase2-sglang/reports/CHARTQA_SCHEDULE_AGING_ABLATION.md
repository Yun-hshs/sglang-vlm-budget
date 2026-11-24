# 持续负载下的老化阈值消融

## 实测结果

2026-11-24 14:07–14:12（北京时间），完成 3,240 条计量请求和 1,080 条预热。每组 648 条计量，均正常结束。每组 243 条实际命中前缀，预期命中均发生。五组计划提交完全相同。纯代价优先有 12 条请求因等待同一会话前一轮而延迟提交，最长约 1.38 秒，导致 73 个实际 API 接收逆序对。其余四组未发生该类延迟或接收逆序。依赖等待计入 TTFT 和排队。

| 策略 | 含排队 TTFT P95（ms） | 长请求排队 P95 / 最长（ms） | 长请求排队 ≥500 ms | 连续窗口完成请求/s | 整轨迹请求/s |
|---|---:|---:|---:|---:|---:|
| FCFS | 2,004 | 1,805 / 1,922 | 93/162 | 14.20 | 13.179 |
| 纯代价优先 | 3,036 | 3,700 / 4,195 | 75/162 | 14.88 | 13.180 |
| 老化 100 ms | 1,976 | 1,804 / 1,868 | 93/162 | 14.20 | 13.178 |
| 老化 220 ms | 2,018 | 1,816 / 2,009 | 95/162 | 14.20 | 13.179 |
| 老化 350 ms | 1,958 | 1,779 / 1,878 | 93/162 | 14.20 | 13.178 |

连续窗口前三轮按 16.67 请求/s 提交。整轨迹输出吞吐为 51.25–51.26 Token/s。纯代价优先降低中位数（154 ms），却把长请求最长等待推到 4.20 秒，最多被后到请求超越 51 次。三种老化阈值都把最长等待压到约 2 秒以内，长请求最多被超越 1、2、2 次。纯代价优先超过 500 ms 的长请求更少，严重延迟集中在少部分请求，需与最长等待及逐用户分布一起看。

350 ms 的合并 TTFT P95 最低，100 ms 的最长等待最短。三轮 TTFT P95 的范围分别为 100 ms 组 1,965–2,011 ms、220 ms 组 1,984–2,073 ms、350 ms 组 1,908–2,005 ms，范围重叠。FCFS 的结果也接近三个老化组。这条轨迹支持将老化与纯代价优先区分，阈值的相对排序需要在另一条负载轨迹验证。

### 逐用户等待

| 策略 | 54 个用户排队 P95 的中位数（ms） | 最差用户 P95（ms） | 用户 validation-5-2-r05 的 P95（ms） |
|---|---:|---:|---:|
| FCFS | 1,678 | 2,005 | 1,879 |
| 纯代价优先 | 188 | 4,156 | 4,156 |
| 老化 100 ms | 1,674 | 1,982 | 1,864 |
| 老化 220 ms | 1,730 | 2,022 | 1,960 |
| 老化 350 ms | 1,657 | 1,989 | 1,862 |

[54 用户等待 CSV](chartqa-schedule-aging-ablation/user_wait.csv) · [逐轮与总汇总](chartqa-schedule-aging-ablation/summary.json) · [到达轨迹与运行清单](chartqa-schedule-aging-ablation/manifest.json) · [逐请求记录](chartqa-schedule-aging-ablation/requests.jsonl.gz)

## 对照与复现

单张 RTX 5090、固定中档、SGLang 0.5.23、运行请求上限 3。沿用[持续负载对照](CHARTQA_SCHEDULE_SUSTAINED.md)的 54 个会话、216 条计划到达及请求内容。运行入口验证参考清单 SHA-256，并逐项对照计划到达、请求内容、缓存命名空间类别、冻结预测器哈希及预测值。计划提交每 60 ms 一条，前三轮组成 9.72 秒连续窗口。多轮客户端等上一轮完成后发出下一轮，依赖等待计入 TTFT。每种策略预热一次、正式跑三次，轮换策略顺序。

比较 FCFS、纯代价优先和三个老化阈值。老化系数固定为每等 1 ms 减少 0.35 ms 预测代价。达到各自的 100、220 或 350 ms 阈值后进入优先的逾期队列，组内按引擎入队时间排列。阈值改变优先级，已经运行的批次继续执行。长请求沿用同一轨迹的预测代价最高四分位，报告总排队至少 500 ms 的次数。排队口径为客户端派发滞后（包含同会话依赖等待）加引擎自报队列耗时。吞吐同时报告连续窗口与整轨迹（含排空）。

按[调度对照部署说明](CHARTQA_SCHEDULE_COMPARE.md#指标与复现)配置隔离运行时、模型与 GPU。启动时把 `SCHEDULE_RULES` 指向 `"$PWD/configs/chartqa-schedule-aging-ablation.json"`，其余使用同一服务器配置。在同一主机另一个终端运行以下命令。

```bash
cd 02-Multimodal/phase2-sglang
export PYTHONPATH="$PWD/.runtime:$PWD"
out="results/schedule-aging-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m benchmark.chartqa_schedule_compare \
  --config configs/chartqa-schedule-aging-ablation.json \
  --reference-trace reports/chartqa-schedule-sustained/manifest.json \
  --fixtures ../phase1-assistant/data/processed/ablation \
  --model-path /absolute/path/to/Qwen2.5-VL-7B-Instruct --output "$out"
python3.12 -m analysis.chartqa_schedule_compare --run "$out"
```

运行器默认从 `results/schedule-server` 读取 Trace。若启动使用其他目录，分别传入 `--vision-trace` 与 `--session-trace`。

## 开源 Prefill 方案与下一步

| 项目 | 可复用内容 | 本项目适用方式 |
|---|---|---|
| SGLang 0.5.23 `shortest-prefill-first` | 用实际前缀匹配后的未缓存 Token 数作为调度代价，并为短请求预留部分 Prefill chunk 预算 | 可以在固定版本上做原生对照。另加相同派发前估计下的“按未缓存 Token 排序”基线，以区分简单代理、引擎内实时缓存信息和 chunk 调整的收益 |
| Microsoft Vidur | 可复用的算子 Profiling 和执行时间建模，开源配置支持线性回归与随机森林 | 借鉴 profiling 方法，在 5090 上测本地模型和运行时。公开 profiling 数据需核对硬件、模型、批形状及视觉编码覆盖 |
| Azure ModServe | 多模态分阶段耗时 profiling 与随机森林拟合工具 | 优先参考视觉编码与语言阶段分别拟合的做法，在现有单卡部署上重新测量 |
| Sarathi-Serve | Chunked Prefill 和依 Profiling 选取混合批 Token 预算 | 作为调度与批处理对照。实验应固定预测器与公平性条件，再改变 chunk 参数 |

参考 [SGLang 0.5.23 原生策略源码](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_policy.py) · [Vidur Profiling](https://github.com/microsoft/vidur/blob/main/docs/profiling.md) · [Vidur 预测器配置](https://github.com/microsoft/vidur/blob/main/vidur/config/config.py) · [ModServe 实验说明](https://github.com/Azure/ModServe/blob/main/docs/experiments.md) · [Sarathi-Serve 论文](https://arxiv.org/abs/2403.02310)

优先用一批新会话轨迹比较三种代价信号：未缓存 Token 数、当前冻结 Ridge、加入本地校准的轻量模型。可分别建模未缓存 Token 数、命中前缀长度、二者的乘积、未复用的视觉网格和同时 Decode 的批负载。所有特征只使用决策时可取得的信息，执行后的命中数留给校验。按会话与负载轨迹划分校准和验证，分别报告 Prefill MAE、同队列排序、用户 TTFT P95 与最差用户等待。调度阈值也要在新轨迹复测。原生 `hrrn` 采用已等待的 Prefill Token 进度与未缓存 Token 数的比值，适合作为另一个公平性基线。[SGLang 0.5.23 调度源码](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/schedule_policy.py)
