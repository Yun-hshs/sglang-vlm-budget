# 阶段 2 · SGLang 多模态服务与调度

**目见**是运行在单张 RTX 5090 上的多用户视觉助手，使用 SGLang 0.5.23 和 Qwen2.5-VL-7B-Instruct。支持多图上传、流式输出、同图追问和隔离的持久化会话。

- **代价感知调度**。采用固定中档视觉预算，以估计未缓存 Token 排序，加入 100 ms 等待老化。小规模持续负载实测中，该策略的含排队 TTFT P95 为 1.80 秒，四组中最低。[调度实测](reports/CHARTQA_PREFILL_SIGNALS.md)
- **部署与恢复**。固定 HTTPS 入口启用登录，SGLang、Web 和 ngrok 由 Supervisor 管理。整套服务重启后验证了同域名登录、历史恢复、会话隔离和流式回答。[试用入口](https://lethargic-aeration-elevate.ngrok-free.dev/) · [部署与验收](reports/VISUAL_CHAT_MVP.md)
- **研究与复现**。归档了 ChartQA 视觉预算、自适应分配、缓存代价、Prefill 预测与等待老化的小规模对照，包含脚本、配置和实测记录。[实验索引](results/STATUS.md) · [研究报告](reports/final_report.md)

[阶段 1 · 基础作业](../phase1-assistant/README.md)记录 ChartQA 微调、评测和课程 Demo。

## 目录

`frontend/` 界面；`api/` 流式客户端；`preprocessing/` 图像预算；`benchmark/` 请求与显存采样；`evaluation/` EM、VQA consensus、ANLS；`analysis/` 分位数与图表；`configs/` 配置和依赖版本；`deployment/` 固定版本服务启动与冒烟；`scripts/` 部署复现；`tests/` CPU 测试；`reports/` 报告与验收记录；`slides/` 答辩源材料。

## 实验客户端与 Gradio

多用户 Web 的部署见上方验收报告。实验服务端使用[单卡部署复现说明](reports/DEPLOYMENT_SMOKE.md)，客户端依赖单独安装。

```bash
cd 02-Multimodal/phase2-sglang
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

本节的实验服务和 Gradio 默认监听回环地址。跨主机使用实验客户端时通过 SSH 转发。

```bash
# Mac 连接远端服务；按实际服务器替换地址，不写进版本库
ssh -N -L 30000:127.0.0.1:30000 user@your-gpu-server
# 客户端终端
export VLM_BASE_URL=http://127.0.0.1:30000/v1
export VLM_MODEL=qwen25vl-7b
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
| adaptive_v2 | low / medium / high | 差值、聚合、比较与显式读字选 medium，满足面积和纵向尺寸门槛的长图选 high |

上述数字为**像素预算**。客户端保持宽高比，只下采样，统一 PNG 编码。模型可能按 patch 网格再次调整尺寸。静态图验收的实际视觉 Token 来自服务端 `image_grid_thw`、`spatial_merge_size` 与占位符的交叉验证，客户端正式实验记录中的 `visual_tokens_actual` 等待接入该观测。视频需要单独验证计数口径。

## 数据准备

见 [data/README.md](data/README.md)。使用获得授权的 VQAv2 / TextVQA / DocVQA 本地数据构建 JSONL。每一行须包含 id、images（相对于 JSONL 的本地路径）、question、answers、task、metric。VQA consensus 必须有 10 个标注。多图实验需要具有有效参考答案的真实多图题目，脚本不会复制单图伪造多图任务。

```bash
python scripts/prepare_dataset.py --input /path/to/full.jsonl --output data/subset.jsonl --count 100 --seed 20261111
```

同一个子集跨策略复用。抽样器保存源哈希、样本 ID 和种子；正式研究建议每个任务单独准备 100–300 个样本，避免混用 EM、VQA 和 ANLS 得出统一“准确率”。数据集和图片默认不提交。

## 正式预算实验（尚未执行）

正式实验使用单独的服务配置和固定样本。两条部署冒烟请求的记录见[部署验收报告](reports/DEPLOYMENT_SMOKE.md)。

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
