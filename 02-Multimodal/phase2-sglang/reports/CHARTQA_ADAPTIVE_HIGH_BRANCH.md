# ChartQA 高档长图分支专项验证

## 实测结果

2026-11-24 10:48–10:49（北京时间），完成 270 条计量请求和 60 条预热，无请求失败或输出截断。每道题在同一策略下的 10 轮回答一致，准确率按独立题数计算，TTFT 和 Token 汇总分别使用每策略的 40 条达门槛请求和 50 条近门槛请求。

| 图像组 | 策略 | 准确率 | 平均实际视觉 Token | TTFT P50 / P90（ms） |
|---|---|---:|---:|---:|
| 达高档门槛 | 改进规则 | 3/4（75%） | 1,273.5 | 148.1 / 155.4 |
| 达高档门槛 | 固定中档 | 3/4（75%） | 769.5 | 97.4 / 123.7 |
| 达高档门槛 | 固定高档 | 3/4（75%） | 1,273.5 | 147.2 / 196.3 |
| 接近高度门槛 | 改进规则 | 4/5（80%） | 264.0 | 64.1 / 67.3 |
| 接近高度门槛 | 固定中档 | 4/5（80%） | 760.0 | 122.7 / 124.0 |
| 接近高度门槛 | 固定高档 | 5/5（100%） | 1,253.2 | 193.2 / 196.2 |

达门槛组的四题全部自动选高档，三种策略的逐题答案一致。改进规则比固定中档每题平均多用 504 个视觉 Token，P50 多 50.7 ms。近门槛五题全部自动选低档，固定高档多得 1 分，来自百分号输出格式的变化。两组样本均来自 augmented，高档组只有 3 张独立图，结果只描述这批图。

### 错例与格式差异

- **1908，达门槛组。** 标注为 `France`，三种策略都答 `Montenegro`。人工复核发现，图表横轴为每 10 万人病例数，`Montenegro` 在图中居首。题目 “Which country is the worst affected in Europe?” 未在题干中明确口径，正式计分沿用原标注。
- **1468，近门槛组。** 标注为 `0.3`，图表显示 `0.3%`。改进规则和固定中档均答 `0.3%`，评分器将其解析为 0.003。固定高档答 `0.3` 并得分。
- **1441，近门槛组。** 三档均正确答 `Netherlands`。改进规则低档、固定中档、固定高档的视觉 Token 分别为 264、760、1,248。

[样本和调度清单](chartqa-adaptive-high-branch/manifest.json) · [分组与逐轮汇总](chartqa-adaptive-high-branch/summary.json) · [相对中档的配对记录](chartqa-adaptive-high-branch/paired-vs-medium.json) · [相对高档的配对记录](chartqa-adaptive-high-branch/paired-vs-high.json) · [逐请求记录](chartqa-adaptive-high-branch/requests.jsonl.gz)

## 样本与冻结条件

在与前两次对照相同版本的 ChartQA 官方 test 导出中，校验开发集与留出集清单，排除两批共 256 张图的全部题目。运行入口检查预算预处理、评分代码和两个排除清单的 SHA-256。沿用冻结的 `adaptive_v2`、固定中档 602,112 像素上限、固定高档 1,003,520 像素上限。 [冻结配置](../configs/chartqa-adaptive-high-branch.json)

| 组 | 定义 | 候选 | 纳入计量 |
|---|---|---|---|
| 达高档门槛 | 高度 > 1,600，高宽比 > 1.35，且面积 > 1,003,520 像素 | 3 图、4 题 | 3 图、4 题 |
| 接近高度门槛 | 高度 1,440–1,600（含端点），其余两项已过高档门槛 | 6 图、6 题 | 5 图、5 题 |

高档组保留全部合格图的关联题目，有两题共用同一张图。近门槛组按 `SHA-256(种子:样本 ID)` 排序，种子为 `2026112403`，按图像哈希去重后固定抽 5 图。最终样本共 8 张图、9 道题，均来自 augmented。复核最终样本与前两批 256 图的 PNG 哈希及解码 RGB 像素哈希，两个口径的交集均为零。

## 测量与复现

单张 RTX 5090 上复用当前 SGLang 0.5.23 和 Qwen2.5-VL-7B-Instruct BF16 基座，关闭多模态预处理缓存、全局多模态缓存和 Radix Cache。并发为 1，每轮以固定种子打乱三种策略和题目顺序。固定档每轮选两组的最大图各做一次预热，自适应档每轮按实际采用的预算各预热一条。预热单独记录，10 轮使用同一批九题。

准确率使用 ChartQA relaxed accuracy。TTFT 从本机提交流式请求计到首个非空正文事件，图片预处理在计时前完成。每条服务端实际视觉 Token 通过网格合并、占位符区间与实际占位符核对。[ChartQA 论文](https://aclanthology.org/2022.findings-acl.177/)

```bash
cd 02-Multimodal/phase2-sglang
export PHASE1_DIR="$(cd ../phase1-assistant && pwd)"
export PYTHONPATH="$PWD/.runtime:$PWD:$PHASE1_DIR"
out="results/chartqa-adaptive-high-branch-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m benchmark.chartqa_budget_pilot \
  --dataset "$PHASE1_DIR/data/processed/ablation/test.jsonl" \
  --config configs/chartqa-adaptive-high-branch.json \
  --exclude-manifest reports/chartqa-adaptive-v2/manifest.json \
  --exclude-manifest reports/chartqa-adaptive-holdout/manifest.json \
  --trace results/server/vision.jsonl --output "$out"
for reference in medium high; do
  python3.12 -m analysis.chartqa_budget_pairs --reference "$reference" \
    --manifest "$out/manifest.json" --requests "$out/requests.jsonl" \
    --output "$out/paired-vs-$reference.json"
done
```

服务和 Trace 钩子见[部署说明](DEPLOYMENT_SMOKE.md)。数据导出重建见[阶段 1 数据准备](../../phase1-assistant/docs/ABLATION.md)。分开部署时将 `PHASE1_DIR` 改为对应的绝对路径。阶段 1 Demo 保持关闭。
