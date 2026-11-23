# 阶段 1 · ChartQA 单卡 LoRA 指令微调

这次使用 Transformers 和 PEFT 直接实现小规模 SFT。基座沿用已经核验的 Qwen2.5-VL-7B-Instruct，版本为 `cc594898137f460bfe9f0759e9844b3ce807cfb5`。训练只读 `train.jsonl`。评估沿用上一步同一份 `test.jsonl`，包含 4 条 human 和 4 条 augmented。 [PEFT LoRA 文档](https://huggingface.co/docs/peft/en/package_reference/lora)

## 运行

先按 [基座指南](BASELINE.md) 准备 Python/PyTorch、模型权重和 ChartQA 清洗清单。此次实测复用服务器已有环境，PEFT 为 0.20.0。在新环境中进入 `02-Multimodal/phase1-assistant` 后执行：

```bash
python -m pip install -r requirements-lora.txt
CUDA_VISIBLE_DEVICES=0 \
  MODEL_PATH=checkpoints/Qwen2.5-VL-7B-Instruct \
  DATA_DIR=data/processed/smoke \
  RUN_DIR=runs/lora-smoke \
  bash scripts/run_lora_smoke.sh
```

脚本按顺序启动四个独立进程：训练并保存 adapter，跑基座评估，重载 adapter 跑同一组评估，最后按样本 ID 计算差异。若需要使用现有环境，可额外设置 `PYTHON_BIN`。重复执行请使用新的 `RUN_DIR`。脚本强制关闭 PyTorch native JIT，推理入口要求仅有一张 GPU 可见。

## 训练和对比口径

- `configs/lora-smoke.json` 固定 rank 8、alpha 16、学习率 1e-4、1 epoch、随机种子 42。每步一条样本，AdamW 更新，梯度范数裁剪为 1.0。模型以 BF16 加载，开启梯度检查点，训练时使用 SDPA Math。
- LoRA 注入语言模型的 `q_proj`、`k_proj`、`v_proj` 和 `o_proj`。基座、视觉编码器和视觉投影保持冻结。训练校验图像哈希和聊天模板的 token 前缀，将用户问题和图像 token 的标签掩码设为 `-100`，损失只覆盖参考答案及助手回合结尾。
- 训练图像上限是 200,704 像素。两组评估共用基座配置的 1,605,632 像素上限、相同提示、确定性解码及 ChartQA relaxed accuracy。评估对照脚本检查模型版本、测试清单哈希、配置和逐题参考答案。
- 权重保存到 `train/adapter/adapter_model.safetensors`。同目录记录基座版本及权重 SHA-256。重新加载前核验两者。完整预测、训练日志和权重保存在运行目录，代码目录只保留汇总和复现材料。

只重载 adapter 的命令：

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_DISABLE_NATIVE_JIT=1 \
  python -m vl_assistant.infer \
    --model-path checkpoints/Qwen2.5-VL-7B-Instruct \
    --adapter runs/lora-smoke/train/adapter \
    --manifest data/processed/smoke/test.jsonl \
    --output runs/lora-reloaded
```

## 单卡实测

物理 GPU 0 为 RTX 5090。8 条训练样本各训练一步，8 步均有有限损失及有效监督 token。共训练 5,046,272 个参数，覆盖 112 个语言注意力投影层。训练结束后确认 LoRA B 矩阵已偏离零初始化，保存的 adapter 权重约 19.3 MiB。训练循环及保存共约 6.29 秒（不含模型加载），PyTorch 峰值分配为 16,654 MiB。

| 同一 ChartQA 测试清单 | 基座 | 重载 LoRA 后 |
|---|---:|---:|
| Human | 3/4 | 3/4 |
| Augmented | 2/4 | 2/4 |
| 合计 | 5/8（62.5%） | 5/8（62.5%） |
| 推理失败 | 0/8 | 0/8 |

两组的 8 条最终输出文本完全一致。这次运行验证了训练、保存、独立进程重载和配对评估链路。8 条训练和 8 条测试样本不足以估计微调收益。正式实验需扩大训练集，在验证集选定配置，并在固定的完整测试集上做秩消融。原有 3 项单元测试通过。

[训练配置与指标](../reports/lora-smoke/train-summary.json) · [逐步训练指标](../reports/lora-smoke/train-steps.jsonl) · [配对评估汇总](../reports/lora-smoke/comparison-summary.json)

服务器运行目录为 `runs/lora-smoke/`，包含 adapter、原始逐题预测以及 `comparison.json`。
