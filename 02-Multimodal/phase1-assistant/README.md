# 阶段 1 · 基础作业

图表与短视频问答助手以 Qwen2.5-VL-7B-Instruct 为基座，使用 Transformers 与 PEFT 训练 ChartQA LoRA，并在单张 RTX 5090 上运行 Gradio。支持高清单图、有序多图和短视频时序问答。默认加载 rank 16 第 128 步 adapter，可切换基座对照。

[正式实验报告](reports/final-report.md) · [在线 Demo](https://dc9fa8a96886798c5a.gradio.live) · [课程提交说明](../SUBMISSION.md)

## 运行与复现

准备兼容 CUDA 版 PyTorch、模型和 adapter 后，从仓库根目录执行：

```bash
cd 02-Multimodal/phase1-assistant
python -m pip install -r requirements-demo.txt
SHARE=1 bash scripts/start_demo.sh
```

| 指南 | 内容 |
|---|---|
| [部署 Demo](docs/DEMO.md) | 环境、权重、公开地址、点击自制样例及上线验收 |
| [准备数据与基座推理](docs/BASELINE.md) | 数据下载校验、样本清洗、单卡基座和媒体冒烟 |
| [复现 LoRA 冒烟](docs/LORA.md) | 小样本训练、保存、重载和逐题对照 |
| [复现正式消融](docs/ABLATION.md) | 两种 rank、验证选点、完整 ChartQA test 和错例分析 |

## 代码与证据

| 目录 | 内容 |
|---|---|
| `vl_assistant/` | 数据清洗、推理、LoRA 训练、批量评测和 Gradio |
| `configs/` | 固定模型版本、训练与评测配置、Demo 设置 |
| `scripts/`、`tests/` | 下载校验、启动脚本和单元测试 |
| `examples/` | 自制图表与短视频样例 |
| `docs/` | 环境、部署与实验复现 |
| `reports/` | 正式报告、训练审计、实测分数、错例和 Demo 截图 |

训练使用固定抽取的 256 条 ChartQA 样本。两组 LoRA 使用相同顺序，分别用固定 128 条验证样本选点，再与基座在同一套 2,500 题上对比。总分为 48.24%、83.72% 和 84.08%。样本来源、架构图、完整子集成绩及误差口径见[报告](reports/final-report.md)。

运行时数据、权重与日志分别写入 `data/`、`checkpoints/` 和 `runs/`。公开仓库提供版本、哈希和复现方法。
