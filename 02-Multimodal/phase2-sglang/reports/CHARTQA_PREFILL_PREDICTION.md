# 多轮视觉会话的 Prefill 代价预测

## 实测结果

2026-11-24 11:36–11:38（北京时间），单张 RTX 5090 完成 576 条计量和 192 条预热。15 个完整训练会话产生 360 条计量请求，9 个完整验证会话产生 216 条。验证集实际 KV 命中 81 条、未命中 135 条。所有重复、缓存条件和轮次沿用所属会话的划分。训练实际使用 6 张图、验证使用 3 张图，两组图像文件哈希无交集。

| 派发前特征 | 验证 MAE（ms） | RMSE（ms） | 绝对误差 P90（ms） | Spearman | 全验证集轻重排序 | 同批并发轻重排序 |
|---|---:|---:|---:|---:|---:|---:|
| 输入 Token | 33.69 | 42.90 | 72.52 | 0.330 | 63.07% | 71.43% |
| + 视觉、历史 | 33.39 | 42.35 | 72.82 | 0.333 | 63.48% | 71.43% |
| + 缓存估计 | 20.33 | 25.74 | 38.18 | 0.834 | 88.74% | 75.97% |

排序正确率忽略真实耗时差 ≤ 10 ms 的请求对，预测并列记半分。全验证集有 17,987 对，同批并发评价只比较同一次屏障发出的请求，共 154 对。请求对包含同一会话的重复观测。

| 派发前特征 | 实际命中 MAE（81 条，ms） | 实际未命中 MAE（135 条，ms） |
|---|---:|---:|
| 输入 Token | 39.82 | 30.00 |
| + 视觉、历史 | 40.53 | 29.11 |
| + 缓存估计 | 15.17 | 23.42 |

仅加入视觉与历史的 MAE 改善约 0.3 ms。继续加入缓存估计后，相对输入基线 MAE 降低约 39.7%，同批并发排序提高 4.55 个百分点。固定中档与已知的缓存命名空间使本次环境较可控。真实流量下的缓存驱逐、前缀变化和未知到达过程仍待验证。本实验未修改 FCFS，也未测调度收益。

[统计与模型系数](chartqa-prefill-prediction/summary.json) · [验证预测](chartqa-prefill-prediction/predictions.jsonl) · [样本与运行清单](chartqa-prefill-prediction/manifest.json) · [逐请求记录](chartqa-prefill-prediction/requests.jsonl.gz)

## 数据与口径

按固定种子在 ChartQA 测试导出中分面积选图，预留互斥图池，再生成包含 24 个会话的可复放负载。每组 3 个并发会话，每个会话 3–5 轮，覆盖同图追问、追加新图和最多 3 张历史图。请求的历史范围为 0–4 轮，估算输入为 173–1,955 Token。真实 ChartQA 图表和有图题目用于复放，历史短答固定，无图追问题和长短提示由脚本生成。每条件做一轮预热及三轮正式重复。

三个模型使用同一个 Ridge 回归与预设正则系数，标准化和拟合只用训练会话。总 Token 在请求发出前用本地模型模板、分词器和图像估算网格求得；视觉/历史组再加入视觉 Token、本轮新增视觉 Token、图数和历史轮数。缓存估计使用相同命名空间内上一已完成请求的预测输入长度，新命名空间或首轮取 0。采集器在发出 API 请求前生成这些特征。服务端命中数和视觉 Token 只用于审计及事后分组。

验证中总输入与视觉 Token 估算均与实测一致，前缀估计对实际命中数的 MAE 为 1.32 Token。这一结果基于可控的缓存命名空间和已完成的历史。Prefill 标签为服务端首次 Forward 到 Prefill 完成的墙钟窗口，包含批处理效应。[多轮阶段计量口径](CHARTQA_SESSION_COST.md)

## 复现

按[多轮实例启动说明](CHARTQA_SESSION_COST.md#复现)启动单卡 SGLang 实验服务。下面在服务所在主机的另一终端执行。

```bash
cd 02-Multimodal/phase2-sglang
export PHASE1_DIR="$(cd ../phase1-assistant && pwd)"
export PYTHONPATH="$PWD/.runtime:$PWD"
data="$PHASE1_DIR/data/processed/ablation"
out="results/prefill-prediction-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m benchmark.build_prefill_sessions \
  --config configs/chartqa-prefill-prediction.json --dataset "$data/test.jsonl" \
  --output "$out-workload.json"
python3.12 -m benchmark.chartqa_session_replay \
  --config "$out-workload.json" --fixtures "$data" \
  --model-path /absolute/path/to/Qwen2.5-VL-7B-Instruct --output "$out"
python3.12 -m analysis.chartqa_prefill_prediction --run "$out"
```

分目录部署时把 `PHASE1_DIR` 设为阶段 1 绝对路径。阶段 1 Demo 保持关闭。
