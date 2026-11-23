# 阶段 1 · ChartQA LoRA 秩消融

基座为固定版本的 Qwen2.5-VL-7B-Instruct，训练与推理均在同一张 RTX 5090 上执行。框架沿用 [单卡基座指南](BASELINE.md) 和 [LoRA 冒烟指南](LORA.md) 中的 Transformers 与 PEFT。

## 实验设计

- **数据**。按固定种子从三个训练分片抽取 256 条，human 和 augmented 各 128 条。剔除与官方验证、测试图像哈希完全一致的训练图，以及训练集中重复的图文问答。验证集从官方 val 里取 128 条分层样本。官方 test 保留 2,500 道题，包括重复使用图表的不同问题。预处理产出分片哈希、样本 ID、过滤计数和清单哈希。检测范围为解码规范化后的相同图片，近重复和基座预训练污染未核实。
- **训练**。rank 8 和 16 共用相同训练清单、固定种子和顺序、梯度裁剪、学习率、一个 epoch、BF16 基座和语言注意力目标层。LoRA alpha 分别为 16 和 32，保持 `alpha / rank = 2`。第 128 和 256 步分别保存 adapter。
- **选点**。每个 rank 独立比较两个 checkpoint 在同一份验证子集上的总正确数。相同时选较早步数。锁定 `selection.json` 后才执行完整测试。
- **全量测试**。基座、rank 8 和 rank 16 共用模型版本、图片预算 200,704 像素、左填充、批大小 8、贪心解码、24 个生成 token 上限、以及同一 ChartQA relaxed accuracy 实现。使用 SDPA 自动选择后端。本轮的图像预算和批处理与前次 8 条样本的基座冒烟不同，应使用本次重跑的基座对照。分组结果用 human 和 augmented 的各自题数作为分母。

## 复现

在已安装 Python/PyTorch 和 `requirements-lora.txt` 的 Linux GPU 环境，进入 `02-Multimodal/phase1-assistant` 后执行：

```bash
bash scripts/fetch_chartqa_ablation.sh
python -m vl_assistant.ablation_data --raw data/raw --output data/processed/ablation
CUDA_VISIBLE_DEVICES=0 python -m vl_assistant.ablation \
  --data data/processed/ablation --output runs/ablation/full \
  --model-path checkpoints/Qwen2.5-VL-7B-Instruct
```

服务器不能直接访问下载端点时，可在可访问端运行下载脚本，把校验通过的五个 Parquet 文件复制到服务器 `data/raw/`。数据准备会重新校验，不分发原始图片或模型。

评估逐题写入 JSONL。中断后使用同一命令和输出目录恢复，配置、清单和 adapter 哈希有变化时会拒绝续跑。若训练在完整写出 `train/summary.json` 前中断，保留原日志，再为该次训练使用新输出目录，当前脚本不恢复优化器状态。

每组在 `rank-*/train/` 中保存训练配置对应的 adapter 和日志。最终 `test/` 保存逐题输出，`results.md` 保存按组分数和错误案例，`errors.json` 可用样本 ID 找回原始图。本轮代码目录只收录必要的结果文件。

## 训练实测

两组均完成 256 步，训练日志中的样本 ID 顺序逐项一致，损失均为有限值。rank 8 有 5,046,272 个可训练参数，训练及保存用时约 177.87 秒。rank 16 有 10,092,544 个可训练参数，用时约 181.86 秒。计时不含加载基座。

[训练 rank 8](reports/ablation/train-rank-8.json) · [训练 rank 16](reports/ablation/train-rank-16.json) · [数据审计](reports/ablation/data-audit.json)

## 验证选点与全量测试

两组各在同一份 128 条验证清单上选点。rank 8 的第 128、256 步分别答对 110、114 题，选择第 256 步。rank 16 分别答对 112、110 题，选择第 128 步。测试加载的权重 SHA-256 与各自入选 checkpoint 一致。

| 模型 | Human（1,250） | Augmented（1,250） | 合计（2,500） |
|---|---:|---:|---:|
| 基座 | 671/1250（53.68%） | 535/1250（42.80%） | 1206/2500（48.24%） |
| LoRA rank 8 · step 256 | 934/1250（74.72%） | 1159/1250（92.72%） | 2093/2500（83.72%） |
| LoRA rank 16 · step 128 | 949/1250（75.92%） | 1153/1250（92.24%） | 2102/2500（84.08%） |

7,500 条测试记录均成功生成。复核时逐项匹配了测试清单的 ID、问题、图像路径和参考答案，并用同一评分实现重算。rank 16 较 rank 8 在 Human 多答对 15 题，在 Augmented 少答对 6 题，合计多答对 9 题。这个差值来自单个种子及各 rank 独立选出的 checkpoint。它描述本轮结果，无法将差异归因给 rank 本身。

两组相对基座的改进包含答案表达方式的适配。基座有 383 条被判错的预测带百分号，去掉百分号后落入无百分号标注的数值容差。两组 LoRA 在本轮均无这种方向的格式失分。反方向也可能退化：Human 样本 28 的标注是 `0.6`，基座 `60%` 按规则正确，rank 8 输出 `60` 被判错。视觉推理方面，Human 样本 0 的柱状图计数从 `13` 改为正确的 `14`。Augmented 样本 1255 的市场公司数标注为 `23`，基座和 rank 8 均答 `28`。百分号诊断单独列示，正式分数按固定规则计算。

[验证选点与 adapter 哈希](reports/ablation/selection.json) · [测试结果和分组错例](reports/ablation/results.md) · [结构化错例](reports/ablation/errors.json) · [百分号格式诊断](reports/ablation/format-diagnostic.json)

全部逐题预测和 adapter 保存在服务器项目目录的 `runs/ablation/full/`。汇总文件和可重跑脚本也已同步到该项目目录。
