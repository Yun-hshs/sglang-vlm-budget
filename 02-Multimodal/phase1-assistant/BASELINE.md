# 阶段 1 · 数据准备与单卡基座推理

本阶段模型是 `Qwen/Qwen2.5-VL-7B-Instruct`，固定版本 `cc594898137f460bfe9f0759e9844b3ce807cfb5`。使用 PyTorch SDPA 的 Math 后端、BF16 和单张 RTX 5090。真实下载与运行路径由调用者指定。模型运行不依赖 SGLang。 [模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

## 环境

在 Linux、Python 3.12、支持 Blackwell 的 NVIDIA 驱动和 FFmpeg（含 libx264）环境执行。先安装经过 GPU 环境验证的 PyTorch CUDA 版本，再安装其余项目依赖。此次复用服务器已有 PyTorch 环境，未修改其依赖。

```bash
cd 02-Multimodal/phase1-assistant
python3.12 -m venv .venv
source .venv/bin/activate
pip install 'torch==2.13.0' 'torchvision==0.28.0' --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-baseline.txt
python -m pytest -q
```

## 准备 ChartQA

使用固定 Hugging Face 镜像 revision。小样本只需要一个训练分片和验证、测试分片。下载脚本验证上游公布的 SHA-256。服务器若无法访问该端点，可在可访问的机器下载，再复制这三个文件到 `data/raw/`。不随代码分发原始媒体。 [数据集](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) · [ChartQA 源数据](https://github.com/vis-nlp/ChartQA)

```bash
bash scripts/fetch_chartqa_smoke.sh
python -m vl_assistant.data --raw data/raw --output data/processed/smoke --per-subset 4
python scripts/make_cases.py data/cases/smoke
```

每个切分按上游分片和行顺序取两类各 4 条有效样本。清洗检查问题和答案、图像可解码性、重复问答以及已入选样本的跨切分相同图像。输出保留分片与行号，图像路径绑定内容哈希，送入模型前重新核验。审计计数包含已满配额而跳过的行。此流程用于验证工程链路，未做全量或近重复泄漏检测。结论只适用于这组冒烟样本。

## 单卡生成

脚本默认只暴露物理 GPU 0。可用 `CUDA_VISIBLE_DEVICES=1` 切换单卡。多卡可见时推理入口会报错。启动脚本关闭 PyTorch 2.13 的 native JIT，可在未安装 Python 开发头文件的服务器上运行稳定的基线。支持在首次运行时自动从官方 HF 版本下载，或提供经校验的本地目录。下载外部模型前预留约 17 GB 权重空间。

```bash
# 路线 A：自动从 Hugging Face 下载
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh

# 路线 B：ModelScope 下载，核对权重 SHA-256 与配置 Git blob ID
bash scripts/fetch_modelscope.sh checkpoints/Qwen2.5-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=checkpoints/Qwen2.5-VL-7B-Instruct \
  RUN_DIR=runs/baseline-mirror bash scripts/run_baseline.sh
```

两个入口均先做 ChartQA test 样本，再做原创验收样例。后者包含 1920×1080 图表、两个图表的差值题、4 秒双色视频时序题。视频通过 PyAV 保留取帧时间，以原生视频张量送入 Qwen 处理器。默认最多解码 30 秒视频、取 16 帧。图像由模型处理器按配置做网格对齐缩放，原始尺寸和实际网格均写入日志。输入集之中若有上限或解码错误，逐条记录并返回非零退出状态。

每组输出 `predictions.jsonl` 与 `summary.json`。后者记录模型、配置、样本清单哈希、各组正确数和失败数。ChartQA 评分采用论文规则，数值容差为 5%，百分比转数值，真值为 0 时要求数值等于 0。文本忽略首尾空白和大小写后精确匹配，支持多个参考答案。失败纳入分母。短视频等原创样例单独记录，不纳入 ChartQA 分数。 [ChartQA 论文 §5.1](https://aclanthology.org/2022.findings-acl.177/)

`generation_s` 为 GPU 同步包围的生成耗时，包含视觉编码和生成阶段，不含输入预处理、不作为服务延迟。显存为 PyTorch peak allocated，非整卡占用。

## 单卡实测记录

2026-11-23，使用物理 GPU 0 的一张 RTX 5090，PyTorch 2.13.0+cu130、Transformers 5.12.1、BF16 和 SDPA Math。五个权重分片和九个配置文件核对通过。3 个单元测试通过。24 条抽样记录均逐条核对原始 Parquet 行的问答、子集、尺寸和媒体哈希。

| 验证项 | 成功生成 | 正确 |
|---|---:|---:|
| ChartQA human | 4/4 | 3/4 |
| ChartQA augmented | 4/4 | 2/4 |
| ChartQA 合计 | 8/8 | 5/8（62.5%） |
| 自制高清图、多图与短视频 | 3/3 | 3/3 |

三个原创用例分别回答 `40`、`30` 和 `Red`。四秒视频实际取 16 帧。ChartQA 与媒体用例的 PyTorch 峰值分配分别为 16,838 MiB 和 25,060 MiB。本次只验证了两张高清图同请求的情形。批量高分辨率输入仍需按显存调整预算。

[数据审计](reports/baseline-smoke/data-audit.json) · [ChartQA 汇总](reports/baseline-smoke/chartqa-summary.json) · [原创媒体汇总](reports/baseline-smoke/media-summary.json) · [原创媒体逐题输出](reports/baseline-smoke/media-predictions.jsonl)

ChartQA 原始逐题预测保存在运行服务器的 `runs/baseline-verified/chartqa/predictions.jsonl`。8 条测试样本仅供链路验收，正式结果需在全量固定测试集上重跑。
