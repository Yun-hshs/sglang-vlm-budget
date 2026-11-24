# ChartQA 独立图像留出集验证

## 实测结果

2026-11-24 10:40–10:42（北京时间）完成 1,152 条计量请求和 21 条预热，无失败或输出截断。各档每道题的三轮回答一致，准确率按 128 道独立题统计。视觉 Token 与 TTFT 每档基于 384 条计量请求。

| 策略 | Human | Augmented | 总准确率 | 平均视觉 Token（范围） | TTFT P50 / P90（ms） |
|---|---:|---:|---:|---:|---:|
| 旧规则 | 46/64（71.88%） | 58/64（90.63%） | 104/128（81.25%） | 253.8（91–1,272） | 62.4 / 64.8 |
| 改进规则 | 50/64（78.13%） | 58/64（90.63%） | 108/128（84.38%） | 316.4（91–1,272） | 62.6 / 97.2 |
| 固定中档 | 51/64（79.69%） | 62/64（96.88%） | 113/128（88.28%） | 564.4（91–780） | 96.9 / 108.9 |

| 策略 | Human 平均视觉 Token | Human TTFT P50 / P90（ms） | Augmented 平均视觉 Token | Augmented TTFT P50 / P90（ms） |
|---|---:|---:|---:|---:|
| 旧规则 | 243.4 | 62.3 / 65.4 | 264.1 | 62.5 / 64.1 |
| 改进规则 | 337.4 | 62.7 / 99.8 | 295.3 | 62.6 / 95.9 |
| 固定中档 | 521.6 | 96.7 / 104.6 | 607.2 | 97.0 / 123.0 |

- 改进规则较旧规则修正 4 题，无新增失分，全部来自 human。与开发集的净增 6 题方向一致。留出集改进版选低档 97 题、中档 30 题、高档 1 题。整体 P50 仅增加约 0.2 ms，P90 从 64.8 增至 97.2 ms。
- 改进规则比固定中档平均少用 43.9% 的视觉 Token，TTFT P50 短 35.3%，净少答对 5 题。逐题比较，固定中档独对 6 题，改进规则独对 1 题。
- 新图验证覆盖同一 ChartQA test 导出的 128 道分层题，高档分支仅覆盖 1 张图。结果描述这批图像和当前单卡串行部署。

[样本与调度清单](chartqa-adaptive-holdout/manifest.json) · [分组与逐轮汇总](chartqa-adaptive-holdout/summary.json) · [相对旧规则的逐题变化](chartqa-adaptive-holdout/paired-vs-adaptive.json) · [相对固定中档的逐题变化](chartqa-adaptive-holdout/paired-vs-medium.json) · [逐请求记录](chartqa-adaptive-holdout/requests.jsonl.gz)

## 样本与冻结条件

从与开发集相同版本的 ChartQA 官方 test 导出再取 128 道题，human 与 augmented 各 64 道，每张图只取一题。先用 SHA-256 校验开发集清单，排除其中 128 张图的所有题目，再应用新种子 `2026112402`。按图像面积分层，层内按 `SHA-256(种子:样本 ID)` 排序，跨子集去重。样本选择使用图像尺寸、图像哈希、子集和样本 ID。最终核查两批的 PNG 文件 SHA-256 和解码后尺寸及 RGB 像素的 SHA-256，两个口径的交集均为零。

| 子集 | ≤ 602,112 像素 | 602,112–1,003,520 像素（左开右闭） | 满足高档几何规则的长图 |
|---|---:|---:|---:|
| Human | 61 | 3 | 0 |
| Augmented | 55 | 8 | 1 |

排除后 human 只剩 3 张中面积图，且无图达到大面积上限以上。Augmented 有 4 张未见图满足改进规则的高档几何门槛，固定抽取 1 张以覆盖该分支。最初按面积抽样时选中的大图没有达到高档高度门槛。在查看评分前调整了大图抽样约束，然后重新跑完三组。正式汇总使用最终清单的运行记录。

预算实现及评分代码的 SHA-256 在抽样前固定，运行入口重新校验。旧规则 `adaptive`、改进规则 `adaptive_v2`、固定中档 `medium` 沿用[前次三组对照](CHARTQA_ADAPTIVE_V2.md)的配置；本次改动只涉及抽样与运行前校验。配置保留开发集清单、数据导出和源代码的哈希。[冻结配置](../configs/chartqa-adaptive-holdout.json)

## 典型错例与边界

编号为固定测试分片的行号后缀。预测列括号内为服务端实际视觉 Token。

| ID / 子集 | 参考答案 | 旧规则 | 改进规则 | 固定中档 | 复核 |
|---|---|---|---|---|---|
| 1033 / human | 1.19 | 0.95（247） | 1.23（580） | 1.23（580） | “Add” 触发中档。图上德国最后三个月为 0.38、0.40、0.41，合计 1.19。中档的 1.23 落在 5% 容差内。 |
| 1575 / augmented | 377.72 | 325.01（252） | 325.01（252） | 377.72（638） | 查询马拉维 2019 年人均 GDP。两版规则都将单值查找分配到低档，答成同一条折线上的其他标注值。 |
| 2316 / augmented | 9.48 | 9.48%（247） | 9.48%（247） | 9.48（580） | 原图 2020 年数据点标为 9.48%。标准评分将百分号数值除以 100；两版低档均因与标注格式不同而失分。 |
| 1481 / augmented | 375 | 375（252） | 375（252） | 8（775） | 问题给出八家医院并询问病例数。固定中档输出问题中的医院数量，两个低档答案与参考值一致。 |
| 1483 / augmented | New York | New York（1,272） | New York（1,272） | New York（779） | 唯一覆盖改进规则高档分支的图，高度 1,766。固定中档也答对，提档多用 493 个 Token。 |

## 测量与复现

沿用单张 RTX 5090 上的 SGLang 0.5.23、Qwen2.5-VL-7B-Instruct BF16 服务，串行测量。每轮固定种子打乱策略和题目顺序，三组使用相同题目顺序，共三轮。自适应组覆盖实际使用的每个档位各预热一次，固定中档用两个子集的最大图各预热一次。预热单独记录。服务端通过请求 ID 核对网格合并 Token、实际图像占位符及区间长度。TTFT 为从本机发送流式请求到第一段非空正文的耗时。图片预处理在请求计时前完成。准确率使用阶段 1 的 ChartQA relaxed accuracy，保留原始短答和标准计分。[ChartQA 论文](https://aclanthology.org/2022.findings-acl.177/)

```bash
cd 02-Multimodal/phase2-sglang
export PHASE1_DIR="$(cd ../phase1-assistant && pwd)"
export PYTHONPATH="$PWD/.runtime:$PWD:$PHASE1_DIR"
out="results/chartqa-adaptive-holdout-$(date -u +%Y%m%dT%H%M%SZ)"
python3.12 -m benchmark.chartqa_budget_pilot \
  --dataset "$PHASE1_DIR/data/processed/ablation/test.jsonl" \
  --config configs/chartqa-adaptive-holdout.json \
  --exclude-manifest reports/chartqa-adaptive-v2/manifest.json \
  --trace results/server/vision.jsonl --output "$out"
for reference in adaptive medium; do
  python3.12 -m analysis.chartqa_budget_pairs --reference "$reference" \
    --manifest "$out/manifest.json" --requests "$out/requests.jsonl" \
    --output "$out/paired-vs-$reference.json"
done
```

服务环境与 Trace 钩子见[部署说明](DEPLOYMENT_SMOKE.md)，清洗数据的重建见[阶段 1 数据准备](../../phase1-assistant/docs/ABLATION.md)。分开部署时，`PHASE1_DIR` 指向阶段 1 绝对路径。阶段 1 Demo 保持关闭。
