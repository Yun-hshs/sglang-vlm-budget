# ChartQA 128 题自适应预算改进对照

## 实测结果

最终运行于 2026-11-24 10:17–10:19（北京时间）完成。三个策略共 1,152 条计量与 21 条预热，无请求失败或输出截断。每档同一道题的三轮回答一致，准确率按 128 道独立题统计，视觉 Token 和 TTFT 每档使用 384 条计量请求。

| 策略 | Human | Augmented | 总准确率 | 平均实际视觉 Token（范围） | TTFT P50 / P90（ms） |
|---|---:|---:|---:|---:|---:|
| 旧自适应 | 43/64 | 56/64 | 99/128（77.34%） | 255.3（84–1,296） | 62.6 / 65.1 |
| 改进自适应 | 48/64 | 57/64 | 105/128（82.03%） | 340.8（84–1,296） | 63.6 / 102.1 |
| 固定中档 | 48/64 | 61/64 | 109/128（85.16%） | 581.2（84–792） | 97.4 / 124.5 |

- 旧规则分配 125 道低档、3 道高档。新规则分配 92 道低档、35 道中档、1 道高档，实际视觉 Token 有 33 题发生变化。逐题比较，7 题从错变对、1 题从对变错，净增 6 题。P50 增约 1.1 ms，P90 增约 37.1 ms。
- 相对固定中档，新规则平均使用的视觉 Token 少 41.4%，TTFT P50 短约 34.7%。固定中档答对而新规则答错 6 题，新规则额外答对 2 题，净少 4 题。两组的正确数差异来自 augmented 子集。
- 本次旧规则和固定中档的全部 128 道预测及逐题视觉 Token 与前次对照一致。本表的延迟全部来自本次重测。

[样本与调度清单](chartqa-adaptive-v2/manifest.json) · [分组与逐轮汇总](chartqa-adaptive-v2/summary.json) · [相对旧规则的逐题变化](chartqa-adaptive-v2/paired-vs-adaptive.json) · [相对固定中档的逐题变化](chartqa-adaptive-v2/paired-vs-medium.json) · [逐请求记录](chartqa-adaptive-v2/requests.jsonl.gz)

## 路由规则

改进规则 `adaptive_v2` 只读取原始问题与方向校正后的图像尺寸。分配优先级如下。

| 优先级 | 条件 | 档位 |
|---|---|---|
| 1 | 高度 > 1,600、高宽比 > 1.35、面积 > 1,003,520 像素，三项同时满足 | 高档，1,003,520 像素 |
| 2 | 问题命中差值、平均、求和、聚合、比较多个系列等提示，或明确要求读字和 OCR | 中档，602,112 像素 |
| 3 | 其余问题 | 低档，200,704 像素 |

英文提示按整词或词组匹配，忽略大小写，`ocr` 无法命中 `democrats`。中文使用明确词组。图像面积、高度和高宽比是密集长图的代理特征。旧规则 `adaptive` 保留原样供重放对照。两条规则使用题干，固定短答后缀和参考答案不参与路由。

试跑时检查到 `How much more did …` 这类省略 `difference` 的差值问法未命中，补充了这类词组后重新运行三组。报告采用最终规则的完整运行记录。这 128 题用于规则开发和回归检查。

## 代表性对照

编号是固定测试分片的行号后缀。预测列括号内给出服务端实际视觉 Token。

| ID / 子集 | 参考答案 | 旧规则 | 改进规则 | 固定中档 | 复核 |
|---|---|---|---|---|---|
| 458 / human | 71 | 71（660） | 71（266） | 71（660） | `democrats` 不再误触发 OCR。答案一致，新规则从高档回到低档。 |
| 887 / human | 40 | 14（247） | 40（580） | 40（580） | 差值问法触发中档，正确读出阿根廷与巴西的「Good」比例差为 40 个百分点。 |
| 1063 / human | 35.2 | 34.2（247） | 38.2（580） | 38.2（580） | 实际五柱均值是 35.2。旧答案误差约 2.8%，在容差内，新答案超出 5%。 |
| 2307 / augmented | 25 | 25%（247） | 25（580） | 25（580） | `How much more …` 触发中档。图上读数为 25%，标注用百分点数值，旧答案因百分号解析失分。 |
| 1457 / augmented | 6.3 | 6.3%（247） | 6.3%（247） | 6.3（580） | 单值读取仍分到低档。图上读数是 6.3%，旧规则和新规则均因百分号格式失分。 |
| 1461 / augmented | Oklahoma | Oklahoma（1,296） | Oklahoma（1,296） | Oklahoma（756） | 高度 1,796、宽度 800 的图触发高档。固定中档也答对，当前案例中高档多用 540 个 Token。 |

## 测量与复现

数据、模型、短答提示和评分复用[五档对照](CHARTQA_BUDGET_128.md)。脚本检查本次样本 ID、顺序、尺寸和图像 SHA-256 与前次清单逐项相同。Human 与 augmented 各 64 道。每轮打乱策略和题目顺序，三组在同一轮使用相同题目顺序，请求并发为 1。共跑 3 轮。各自适应规则按选中的每档用一张该档原图面积最大的图片预热。固定中档用两个子集中原图面积最大的题各预热一次。各组预热样本保存在清单，排除在统计之外。

三组使用同一台 RTX 5090 上现有的 SGLang 0.5.23 BF16 服务，Radix Cache 与图像预处理缓存关闭。实际视觉 Token 取服务端网格合并结果，与占位符区间及实际 Token 数逐请求核对。TTFT 从本机发送请求计到首个非空流式文本，含序列化与服务等待，图像预处理在计时前完成。质量使用原短答的 ChartQA relaxed accuracy。

按[部署说明](DEPLOYMENT_SMOKE.md)启动带 Trace 的 SGLang。复用阶段 1 校验过的 `data/processed/ablation/test.jsonl` 与相邻 `images/`，并保留前次运行的样本清单。[实验配置](../configs/chartqa-adaptive-v2.json) · [预算实现](../preprocessing/image_budget.py)

```bash
cd 02-Multimodal/phase2-sglang
export PHASE1_DIR="$(cd ../phase1-assistant && pwd)"
# 独立部署时使用阶段 1 项目目录的绝对路径。
export PYTHONPATH="$PWD/.runtime:$PWD:$PHASE1_DIR"
out="results/chartqa-adaptive-v2-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m benchmark.chartqa_budget_pilot \
  --dataset "$PHASE1_DIR/data/processed/ablation/test.jsonl" \
  --config configs/chartqa-adaptive-v2.json \
  --sample-manifest reports/chartqa-budget-128/manifest.json \
  --trace results/server/vision.jsonl --output "$out"

for reference in adaptive medium; do
  python3.12 -m analysis.chartqa_budget_pairs --reference "$reference" \
    --manifest "$out/manifest.json" --requests "$out/requests.jsonl" \
    --output "$out/paired-vs-$reference.json"
done
```

阶段 1 Gradio 保持关闭。Demo 启动脚本与训练产物保留。
