# 方向 02 · 多模态视觉语言助手

基础作业实现了图表与短视频问答，支持高清单图、有序多图和短视频时间采样。基于 Qwen2.5-VL-7B-Instruct 完成 ChartQA LoRA 训练、验证选点、全量评测和单卡 Gradio 部署。

| 阶段 | 交付范围 | 当前状态 |
|---|---|---|
| **[阶段 1 · 基础作业](02-Multimodal/phase1-assistant/README.md)** | 数据清洗、LoRA 训练与消融、多模态推理、交互 Demo、实验报告 | 已实测并发布，课程 PR 待确认 |
| [阶段 2 · 探索优化](02-Multimodal/phase2-sglang/README.md) | SGLang 部署与视觉 Token 预算研究 | 保留原型、实验方案和研究稿，GPU 实验未运行 |

[课程报告](02-Multimodal/phase1-assistant/reports/final-report.md) · [在线 Demo](https://dc9fa8a96886798c5a.gradio.live) · [一键启动与复现](02-Multimodal/phase1-assistant/docs/DEMO.md) · [课程提交说明](02-Multimodal/SUBMISSION.md)

## 基础作业实测

测试使用官方 ChartQA 全部 2,500 题，三组均无推理失败。

| 模型 | Human | Augmented | 合计 |
|---|---:|---:|---:|
| 基座 | 53.68% | 42.80% | 48.24% |
| LoRA rank 8 · step 256 | 74.72% | 92.72% | 83.72% |
| LoRA rank 16 · step 128 | 75.92% | 92.24% | 84.08% |

两组 checkpoint 都由验证集选出。得分变化包含答案格式适配，具体错例与解释见正式报告。

## 仓库导航

```text
02-Multimodal/
├── phase1-assistant/   基础作业代码、部署指南与已验证结果
├── phase2-sglang/      探索优化的原型、计划与研究材料
├── course-delivery/   匿名报告模板与课程 PR 草稿
└── SUBMISSION.md      2026 方向 02 提交规则与证据索引
```

正式课程交付引用 [phase1-v1.0.0 固定版本](https://github.com/Yun-hshs/sglang-vlm-budget/tree/phase1-v1.0.0)。在线 Demo 为临时分享地址，过期后可按部署指南重启。
