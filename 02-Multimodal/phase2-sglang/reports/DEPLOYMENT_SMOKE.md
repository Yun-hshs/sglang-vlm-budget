# SGLang 单卡部署与视觉 Token 验收

## 实测结果

2026-11-23 17:07（北京时间），两条请求通过 `/v1/chat/completions` 流式返回，均以 `stop` 结束。输入均为 1920 × 1080 的阶段 1 图表样例，客户端保留尺寸。

| 请求 | 参考答案 | 实际回答 | 逐图视觉 Token | 视觉 Token 合计 | API prompt_tokens |
|---|---:|---:|---|---:|---:|
| 单图读数，`day1.png` 的苹果销量 | 40 | 40 | 2,691 | 2,691 | 2,726 |
| 双图差值，`day2.png` 减 `day1.png` 的苹果销量 | 30 | 30 | 2,691 + 2,691 | 5,382 | 5,440 |

两张图的服务端网格均为 `[1, 78, 138]`，空间合并系数为 2，单图计算为 `1 × 39 × 69 = 2,691`。单图的占位符区间为 `[15, 2705]`，双图额外区间为 `[2708, 5398]`，下标从 0 开始且包含首尾。每个区间的长度和实际图像占位符数量均为 2,691。服务端处理器的输入 ID 总数与 API 的 `prompt_tokens` 逐请求相等。[逐请求 JSON 记录](deployment-smoke.json)

结束后，阶段 2 的 `/health`、阶段 1 Gradio 的本机和公开 `/config` 均返回 HTTP 200。两张 GPU 的进程归属分别为阶段 1（GPU 0）和 SGLang（GPU 1）。本次范围为部署和两条功能请求，正式视觉预算、并发与质量评估登记为待运行。[阶段 1 Gradio Demo](https://dc9fa8a96886798c5a.gradio.live) · [实验状态](../results/STATUS.md)

## 部署与复现

阶段 2 通过独立进程在物理 GPU 1 启动 SGLang，监听 `127.0.0.1:30000`。模型使用 Qwen2.5-VL-7B-Instruct 的 BF16 基座权重，张量并行度为 1，上下文上限为 8192，显存静态分配比例为 0.75，最大运行请求数为 4。保留模型处理器的默认图像预算，关闭 Radix Cache，图像预处理缓存大小设为 0。模型 revision 和服务名称见[启动配置](../configs/sglang-smoke.json)。

运行环境为 Ubuntu 24.04.4 x86-64、Python 3.12.3、RTX 5090（32607 MiB）、NVIDIA 驱动 595.58.03。SGLang 0.5.23、Torch 2.14.1、Transformers 5.19.0、sglang-kernel 0.4.10。[Linux 依赖版本快照](../configs/sglang-linux-x86_64-py312.lock.txt)

首次启动时，Triton 编译因宿主缺少 `Python.h` 报错。验收使用项目目录内解包的 Ubuntu `libpython3.12-dev` 头文件。启动脚本将隔离环境的 `bin` 加入 `PATH`，使 SGLang JIT 能调用依赖锁中的 Ninja。语言模型注意力后端显式设为 Triton，JIT 使用宿主的 CUDA 13.0 工具链。已有对应 Python 开发头文件的机器可跳过解包步骤。

```bash
# 从仓库根目录进入阶段 2，在 Linux GPU 主机执行
cd 02-Multimodal/phase2-sglang
export PYTHON_BIN="$(command -v python3.12)"
"$PYTHON_BIN" -m pip install --target .runtime -r configs/sglang-linux-x86_64-py312.lock.txt

# 仅在缺少 Python.h 的 Ubuntu 24.04 x86-64 主机执行
mkdir -p .python-dev
curl -fsSL https://archive.ubuntu.com/ubuntu/pool/main/p/python3.12/libpython3.12-dev_3.12.3-1ubuntu0.17_amd64.deb -o .python-dev/headers.deb
printf '%s  %s\n' c123ddab7763e45199e21b4ac8545e705f20a0315b9ada4c236b52fb088c4e16 .python-dev/headers.deb | sha256sum -c
dpkg-deb -x .python-dev/headers.deb .python-dev
export CPATH="$PWD/.python-dev/usr/include/python3.12:$PWD/.python-dev/usr/include${CPATH:+:$CPATH}"

# MODEL_PATH 使用上面配置中 revision 对应的本地完整模型快照
export MODEL_PATH=/absolute/path/to/Qwen2.5-VL-7B-Instruct
export CUDA_VISIBLE_DEVICES=1  # 用 nvidia-smi 确认后改为空闲物理卡编号
export CUDA_HOME=/usr/local/cuda  # 指向本机匹配的 CUDA 编译工具链
bash scripts/start_sglang_smoke.sh
```

服务就绪后，在同一主机的另一个终端进入阶段 2 目录。按部署位置修改图片路径，使用阶段 1 的原始 `day1.png`、`day2.png`。

```bash
curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30000/v1/models
PYTHONPATH="$PWD/.runtime:$PWD" python3.12 -m deployment.smoke_requests \
  --fixtures ../phase1-assistant/examples \
  --output results/server/smoke-replay.json
```

从其他机器调用 API 可以建立 SSH 转发，例如 `ssh -N -L 30000:127.0.0.1:30000 -p SSH_PORT user@gpu-host`。读取服务端 trace 的验收脚本在服务主机运行。阶段 1 Gradio 进程可按其[部署说明](../../phase1-assistant/docs/DEMO.md)独立启动。

## 计数口径

观测代码只针对带 `stage2-mm-` 请求 ID 的静态图请求，记录 Qwen-VL 服务端处理器返回的 `image_grid_thw`、`spatial_merge_size`、图像占位符偏移、实际 `<|image_pad|>` 数量和完整输入 ID 数。按 `t × (h / merge) × (w / merge)` 计算每张图的合并后 Token，并用含首尾端点的占位符区间和实际占位符数交叉验证。OpenAI API 的 `usage.prompt_tokens` 单列核对，它包含图像和文本等完整输入。原始观测文件默认写入本机 `results/server/vision.jsonl`。JSON 中的客户端耗时来自两次独立请求，首个图像请求可能包含运行时编译开销，正式性能比较仍需预热和重复测量。[SGLang 0.5.23 的 Qwen-VL 处理器](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/multimodal/processors/qwen_vl.py)
