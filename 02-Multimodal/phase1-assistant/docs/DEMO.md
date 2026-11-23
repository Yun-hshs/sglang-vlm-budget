# 阶段 1 · 单卡 Gradio Demo

Demo 使用单张 RTX 5090 运行 Qwen2.5-VL-7B-Instruct BF16。默认加载 ChartQA 验证集选出的 **LoRA rank 16 · step 128**，启动时核对基座 revision、adapter 元数据和权重 SHA-256。在页面选择「基座对照」可用相同的输入和解码参数重新提问。两种模式共享同一份 GPU 模型权重。

## 一键启动

按 [基座](BASELINE.md) 和 [LoRA 消融](ABLATION.md) 指南准备环境与权重。先安装与 GPU 匹配的 CUDA 版 PyTorch 和 TorchVision，再从仓库根目录执行：

```bash
cd 02-Multimodal/phase1-assistant
python -m pip install -r requirements-demo.txt
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=checkpoints/Qwen2.5-VL-7B-Instruct \
  SHARE=1 bash scripts/start_demo.sh
```

启动脚本默认用 GPU 0 和回环端口 7861。设置 `SHARE=1` 后，从日志中读取当次启动的 `Running on public URL`，访问该 HTTPS 地址即可打开页面。Gradio 分享地址为临时链接，服务停机或隧道过期后需重新启动并取新地址。

如果已将 Gradio 装入项目的独立依赖目录，可通过环境变量复用兼容的 PyTorch 环境：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.demo-deps \
  PYTHON_BIN=/path/to/cuda-venv/bin/python \
  MODEL_PATH=checkpoints/Qwen2.5-VL-7B-Instruct \
  SHARE=1 bash scripts/start_demo.sh
```

公开仓库不含模型或 adapter 文件。将基座放进 `checkpoints/Qwen2.5-VL-7B-Instruct` 后，脚本会自动使用该目录。adapter 可按 [消融指南](ABLATION.md) 复现。复现得到的 adapter 可能有不同的文件哈希。若使用自己复现的 rank 16 第 128 步产物，可明确传入该产物及其训练时写入的哈希：

```bash
export ADAPTER_PATH=runs/ablation/full/rank-16/train/step-128/adapter
export ADAPTER_SHA256="$(python -c 'import json,os,pathlib; print(json.loads((pathlib.Path(os.environ["ADAPTER_PATH"])/"provenance.json").read_text())["sha256"])')"
SHARE=1 bash scripts/start_demo.sh
```

页面的「自制验收样例」可直接填入素材与问题。选中高清图、双图或短视频样例后，点击「生成回答」即可复现。样例是项目自制素材。

若只需要内网或 SSH 隧道访问，可去掉 `SHARE=1`，将远端 7861 端口转发到本机后打开本机端口。通过 `PORT` 可另选端口。adapter 默认路径是 `runs/ablation/full/rank-16/train/step-128/adapter`，可用 `ADAPTER_PATH` 指定同一份校验通过的副本。当前在线进程的日志为 `runs/demo/server.log`。

## 使用与边界

- 上传一至四张图片，或一个短视频，输入问题后生成回答。多图按上传顺序编号。高清图像按实际像素预算送入模型。单张输入上限为 1,605,632 个处理像素，上传数量增加时平分约 321 万像素的总预算。
- 支持 PNG、JPEG、WebP 图片，以及 MP4、MOV、WebM 视频。单文件最大 25 MiB，短视频最长 30 秒，最多采样 16 帧。处理用到视频画面与时间信息。
- 页面逐段显示解码文本。对于数字或一个词的答案，通常只出现一次正文更新。服务按单卡串行生成，最多排队六个请求。临时上传使用 Gradio 定期清理。

## 公开链路实测

用实际公开 HTTPS 地址调用 Gradio 的上传和生成接口。以下均为 2026-11-23 在服务器上的真实模型结果。

| 输入 | 模式 | 实测 |
|---|---|---|
| 1920×1080 高清单图 | LoRA rank 16 · step 128 | 苹果销量 `40` |
| 两张 1920×1080 图 | LoRA rank 16 · step 128 | 图 2 比图 1 多 `30` |
| 4 秒短视频 | LoRA rank 16 · step 128 | 首个颜色 `red`，实际采样 16 帧 |
| 高清单图描述 | 基座对照 | 正确描述苹果 40、梨 20，客户端收到 18 次不同的正文更新 |

[验收记录](../reports/demo/public-smoke.json)。复测可使用仓库内的自制样例，将当前公开地址作为第一个参数。

```bash
python scripts/check_demo.py https://YOUR-DEMO.gradio.live examples
```

公开首页、前端资源和 API 均已通过 HTTP 检查。如果内置小窗显示空白，可使用独立浏览器。实测截图通过 Chrome 和服务器的 SSH 本地转发获取。若客户端系统代理导致连接超时，可给验收命令加 `--direct`。在公开地址失效时，也可用相同脚本检查本机或 SSH 转发的地址。
