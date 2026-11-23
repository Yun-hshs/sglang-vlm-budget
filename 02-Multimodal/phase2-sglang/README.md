# 基于 SGLang 的多模态视觉问答系统设计与视觉 Token 预算优化研究

这是阶段 2 的研究存档。课程验收以[阶段 1 实施方案](../phase1-assistant/README.md)为准。

**状态：实现已编写，测试和真实推理均未运行。** 2026-11-11，用户要求暂不测试，后续提供 GPU 服务器。当前为 Apple Silicon Mac，无 NVIDIA GPU；没有下载模型权重和公开评测数据。报告没有加速比、准确率或 GPU 显存测量值。

## 目标与范围

实现图像问答、描述、文档/图表理解与多轮对话；通过客户端输入尺寸预算研究 RQ1（视觉输入规模对性能的影响）与 RQ2（质量与效率权衡）。模型采用 Qwen2.5-VL-7B-Instruct，服务采用 SGLang OpenAI-compatible API。理解能力由真实模型提供，当前没有伪造答案的演示后端。RQ3 缓存研究仅作为扩展，不修改 SGLang 源码。

## 目录

`frontend/` 界面；`api/` 流式客户端；`preprocessing/` 图像预算；`benchmark/` 请求与显存采样；`evaluation/` EM、VQA consensus、ANLS；`analysis/` 分位数与图表；`configs/` 实验配置；`scripts/` 部署复现；`tests/` 待执行的 CPU 测试；`reports/` 报告与验收记录；`slides/` 答辩源材料。

## 安装（后续服务器阶段执行）

推荐 Linux、NVIDIA CUDA GPU、Python 3.11。硬件是否可容纳 7B BF16、上下文和并发请求需实测，不承诺固定显存门槛。Mac 可用于编辑和远程客户端。

```bash
cd 02-Multimodal/phase2-sglang
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖版本为本次准备环境中的版本；跨平台安装和 Python 3.11 兼容性尚未执行验证。客户端依赖不包含 SGLang。为避免 PyTorch/CUDA 依赖冲突，服务器单独建环境，依据 GPU 驱动和 SGLang 官方安装说明选择匹配的发行版本并固定版本。不得用未固定的 latest 作为论文实验环境。

```bash
# 在独立的 Linux GPU 服务环境，先选择并记录经核实的发行版本
export SGLANG_VERSION='<exact-release>'
pip install "sglang[all]==${SGLANG_VERSION}"
export MODEL_REVISION='<Qwen 模型仓库不可变 commit SHA>'
# 编辑 configs/model.yaml，填入相同的版本和 revision
bash scripts/start_server.sh
```

启动脚本使用官方 `sglang.launch_server` 入口、`--model-path`、`--revision`、`--dtype`，这些参数组合尚未在真实 GPU 上运行。接入时先查该固定版本的 `python -m sglang.launch_server --help`，再启动；不得把本地编写脚本当作启动验证。默认只监听 127.0.0.1，通过 SSH 隧道访问；密钥只放环境变量。

```bash
# Mac 连接远端服务；按实际服务器替换地址，不写进版本库
ssh -N -L 30000:127.0.0.1:30000 user@your-gpu-server
# 客户端终端
export VLM_BASE_URL=http://127.0.0.1:30000/v1
export VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
python -m frontend.app
```

在终端显示的 Gradio 本地地址上传图片提问。首轮必须有图，后续可继续追问或加入新图；最多 8 张新增图，10 轮后清空。预算仅应用于本轮新增图；修改预算不会重写历史图片。所有历史图片随上下文发送，长会话可能超过模型上下文或显存，届时应清空会话。

## 输入预算

| 策略 | 客户端像素上限 | 用途 |
|---|---:|---|
| baseline | 不主动缩小 | 保留原始方向校正后的 RGB 图，服务端仍用模型默认处理器 |
| low | 200704 | 256 × 28² 像素 |
| medium | 602112 | 768 × 28² 像素 |
| high | 1003520 | 1280 × 28² 像素 |
| adaptive | low / high | 问题包含 OCR/文档/图表词，或高纵向分辨率时选 high，否则 low |

上述数字为**像素预算**，256/768/1280 不是实测视觉 Token。保持宽高比，只下采样、不放大，统一 PNG 编码以避免策略间 JPEG 损失差异；baseline 同样转换格式，因此不是原文件字节直传。模型仍可能按 patch 网格二次 resize。当前 `visual_tokens_actual=null`；应在 SGLang 实际图像处理器处记录 `image_grid_thw` 与 `spatial_merge_size`，按该版本实现核算，视频帧不能沿用静态图公式。不要将总 prompt_tokens 当视觉 Token。

## 数据准备

见 [data/README.md](data/README.md)。使用获得授权的 VQAv2 / TextVQA / DocVQA 本地数据构建 JSONL。每一行须包含 id、images（相对于 JSONL 的本地路径）、question、answers、task、metric。VQA consensus 必须有 10 个标注。多图实验需要具有有效参考答案的真实多图题目，脚本不会复制单图伪造多图任务。

```bash
python scripts/prepare_dataset.py --input /path/to/full.jsonl --output data/subset.jsonl --count 100 --seed 20261111
```

同一个子集跨策略复用。抽样器保存源哈希、样本 ID 和种子；正式研究建议每个任务单独准备 100–300 个样本，避免混用 EM、VQA 和 ANLS 得出统一“准确率”。数据集和图片默认不提交。

## 测试与实验（尚未执行）

收到 GPU 后，先执行 CPU 测试，再进行单请求端到端冒烟和完整实验。当前不启用 GitHub Actions。scripts/manual_validation.yml.example 保留手动验证配置示例，后续获得 workflow 上传权限且用户决定开始测试时再启用。

```bash
python -m pytest -q
bash scripts/capture_environment.sh
# 必须先填写 configs/model.yaml 中版本占位字段
export DATASET=/absolute/path/to/subset.jsonl
bash scripts/run_experiments.sh
# 在 GPU 服务主机单独终端采样，不是 Mac 端
python -m benchmark.gpu_memory --output results/gpu-memory.csv
```

正式实验以 `configs/benchmark.yaml` 为准，默认五策略、四档图数、四档请求并发、三轮重复；组顺序用固定种子打乱，各组预热三请求。无对应图数的真实样本会记录在 skipped.json；预热失败立即停止。测试必须使用独立服务，避免其他流量干扰。

`warm` 指保留服务缓存，不代表实际命中；`cold` 每请求调用 `/flush_cache`，强制并发=1，不保证清除了所有视觉/处理缓存。冷缓存须独立配置文件，结果不与热缓存混合。flush API 和范围要在固定 SGLang 版本确认。

每次运行生成 manifest.json、requests.jsonl、skipped.json 和分析目录。记录数据/图像 SHA256、解码参数、版本与请求失败信息。失败计入成功率；输出 length 截断由 finish_reason 标记，解释质量时需报告截断比例。

## 指标口径

- TTFT：客户端发送请求至首个非空文本内容事件，包含网络、排队、服务处理；不是纯 Prefill。
- E2E：请求开始到 SSE 完成，另记含图像处理的 client_total_s。
- TPOT：首末内容事件时间差 / (usage.completion_tokens−1)，仅在有可靠 usage 且流中有多个内容到达时计算。流可能合并 Token，因此是客户端估计，不是 GPU Decode kernel 时间。
- 吞吐：成功请求的输出 Token 总量 / 整组墙钟时间（含失败等待），不把请求吞吐当 Token 吞吐。
- 显存：nvidia-smi 的设备级采样，不是进程 allocator peak，可能漏掉短时峰值；真实峰值需服务端 profiler。
- 质量：项目简化规范化器下 EM、VQA leave-one-out consensus、ANLS；正式分数应使用各数据集官方 evaluator 复核。

分析脚本按策略、图数、并发和缓存分组。**应将不同任务分别运行并分析**，避免混合不同评分量纲。TTFT/TPOT/E2E 给出 P50/P90/P99，Token 不存在时留空，无原始实测数据时拒绝画图。真实视觉 Token 数据接入后才画 Token–Latency。任务分组、bootstrap 置信区间和 profiler 阶段分解属于后续研究增强。

## 常见问题

连接失败：确认隧道、端口和服务健康状态。显存不足：降低并发/图数/输入尺寸，模型切换必须建立独立实验分组。没有 usage：保留文本和耗时，不用字符数或 SSE 事件数冒充 Token 数。OCR 质量差：先比较 high 与 baseline，预算降低可能损害细节。不能安装指定依赖：建立已验证环境锁文件并在报告记录变更，不静默升级。

## 参考与许可

- [SGLang 多模态支持列表](https://github.com/sgl-project/sglang/blob/main/docs/docs/supported-models/multimodal_language_models.mdx)：列出 Qwen-VL，仍需固定版本实测。
- [Qwen2.5-VL 模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)：模型 Apache-2.0，模型使用遵循原始许可。
- [Qwen 处理器配置](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json)：patch_size=14、merge_size=2，不等于所有输入的 Token 数固定。
- [Qwen2.5-VL 技术报告](https://arxiv.org/abs/2502.13923)。

链接查阅日期为 2026-11-11。项目代码采用 MIT；模型与数据集许可独立，不能以代码许可替代。课程目录创建前为空，没有可核对的评分细则。
