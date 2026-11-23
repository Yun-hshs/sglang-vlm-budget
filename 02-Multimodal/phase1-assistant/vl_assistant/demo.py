"""Serve the selected ChartQA adapter and its base model on one GPU."""
import argparse
from contextlib import nullcontext
import json
import logging
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import perf_counter

import av
import gradio as gr
from PIL import Image
import torch
from transformers import StoppingCriteria, TextIteratorStreamer

from .adapter import attach_adapter
from .infer import Baseline

ADAPTED = "LoRA rank 16 · step 128"
BASE = "基座对照"
IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".webp"]
VIDEO_TYPES = [".mp4", ".mov", ".webm"]
LOG = logging.getLogger(__name__)


class StopOnCancel(StoppingCriteria):
    def __init__(self, stop):
        self.stop = stop

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop.is_set()


class Demo:
    def __init__(self, baseline_config, settings, model_path, adapter_path):
        self.settings = settings
        self.assistant = Baseline(
            {**baseline_config, "max_new_tokens": settings["max_new_tokens"]}, model_path
        )
        attach_adapter(self.assistant, adapter_path, settings["adapter_sha256"])
        self.lock = Lock()

    def validate(self, images, video, question, mode):
        question = (question or "").strip()
        if mode not in (ADAPTED, BASE):
            raise ValueError("请选择有效的模型。")
        if not 1 <= len(question) <= self.settings["max_question_chars"]:
            raise ValueError(f"请输入 1–{self.settings['max_question_chars']} 个字符的问题。")
        images = [Path(image) for image in images or []]
        video = Path(video) if video else None
        if bool(images) == bool(video):
            raise ValueError("请上传图片或一个短视频。")
        if len(images) > self.settings["max_images"]:
            raise ValueError(f"每次最多上传 {self.settings['max_images']} 张图片。")
        paths = [video] if video else images
        for path in paths:
            if path.suffix.lower() not in (VIDEO_TYPES if video else IMAGE_TYPES):
                raise ValueError("媒体格式不支持。")
            if path.stat().st_size > self.settings["max_upload_mib"] * 2**20:
                raise ValueError(f"单个文件不能超过 {self.settings['max_upload_mib']} MiB。")
        for path in images:
            with Image.open(path) as source:
                if source.width * source.height > self.settings["max_source_image_pixels"]:
                    raise ValueError("图片解码尺寸不能超过 2500 万像素。")
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("实际图片编码须为 PNG、JPEG 或 WebP。")
        if video:
            with av.open(str(video)) as container:
                if not container.streams.video:
                    raise ValueError("文件没有视频轨道。")
                stream = container.streams.video[0]
                if stream.width * stream.height > 4096 * 2304:
                    raise ValueError("视频分辨率不能超过 4096 × 2304。")
                if stream.average_rate and stream.average_rate > 60:
                    raise ValueError("视频帧率不能超过 60 fps。")
        return images, video, question

    def generate(self, images, video, question, mode):
        try:
            images, video, question = self.validate(images, video, question, mode)
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            message = str(error) if isinstance(error, ValueError) else "无法读取媒体，请更换文件。"
            raise gr.Error(message) from None

        start = perf_counter()
        yield "", f"{mode} · 正在处理媒体"
        with self.lock:
            config = self.assistant.config
            max_pixels = config["max_pixels"]
            try:
                if images:
                    config["max_pixels"] = min(
                        max_pixels, self.settings["max_total_image_pixels"] // len(images)
                    )
                inputs, timestamps, sizes = self.assistant.prepare_inputs(question, images, video)
                inputs = inputs.to("cuda:0")
            except ValueError as error:
                raise gr.Error(str(error)) from None
            except (OSError, RuntimeError, Image.DecompressionBombError):
                LOG.exception("Media preparation failed")
                raise gr.Error("媒体处理失败，请尝试较小的文件。") from None
            finally:
                config["max_pixels"] = max_pixels
            streamer = TextIteratorStreamer(
                self.assistant.processor.tokenizer, skip_prompt=True, timeout=90,
                skip_special_tokens=True,
            )
            stop = Event()
            failures = Queue()

            def worker():
                try:
                    mode_context = self.assistant.model.disable_adapter() if mode == BASE else nullcontext()
                    with torch.inference_mode(), mode_context:
                        self.assistant.model.generate(
                            **inputs, do_sample=False, streamer=streamer,
                            stopping_criteria=[StopOnCancel(stop)],
                            max_new_tokens=config["max_new_tokens"],
                        )
                except Exception as error:
                    failures.put(error)
                    streamer.on_finalized_text("", stream_end=True)

            thread = Thread(target=worker, daemon=True)
            thread.start()
            answer = ""
            media_label = (f"视频采样 {len(timestamps)} 帧"
                           if video else " / ".join(f"{w}×{h}" for w, h in sizes))
            try:
                for chunk in streamer:
                    if chunk:
                        answer += chunk
                        yield answer, f"{mode} · {media_label} · 正在生成"
            except Empty:
                raise gr.Error("生成超时，请缩短问题或减少输入媒体。") from None
            finally:
                stop.set()
                thread.join()
            if not failures.empty():
                error = failures.get_nowait()
                LOG.error("Model generation failed", exc_info=error)
                if isinstance(error, torch.OutOfMemoryError):
                    del inputs
                    torch.cuda.empty_cache()
                raise gr.Error("生成失败，请尝试减少图片数量或缩短视频。") from None
            if not answer.strip():
                raise gr.Error("模型未生成有效文本，请换一种提问方式。")
            yield answer, f"{mode} · {media_label} · 完成（{perf_counter() - start:.1f} 秒）"


def build(demo):
    with gr.Blocks(title="图表与视频问答", delete_cache=(1800, 1800), analytics_enabled=False) as page:
        gr.Markdown("# 图表与视频问答\nQwen2.5-VL-7B-Instruct")
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio([ADAPTED, BASE], value=ADAPTED, label="模型")
                images = gr.File(label="图片 · 最多 4 张，按上传顺序编号",
                                 file_count="multiple", file_types=IMAGE_TYPES)
                video = gr.File(label="短视频 · 最长 30 秒，最多取 16 帧",
                                file_types=VIDEO_TYPES)
                question = gr.Textbox(label="问题", lines=3,
                                      placeholder="例如：图 2 比图 1 多了多少？")
                submit = gr.Button("生成回答", variant="primary")
            with gr.Column(scale=1):
                answer = gr.Textbox(label="回答", lines=10, interactive=False)
                status = gr.Textbox(label="推理状态", interactive=False)
        samples = Path("examples").resolve()
        gr.Examples(
            examples=[
                [[str(samples / "day1.png")], None, "How many units of apples were sold?", ADAPTED],
                [[str(samples / "day1.png"), str(samples / "day2.png")], None,
                 "How many more apple units are sold in Image 2 than in Image 1?", ADAPTED],
                [None, str(samples / "red-blue.mp4"),
                 "Which color fills the screen first, red or blue?", ADAPTED],
            ],
            inputs=[images, video, question, mode],
            cache_examples=False,
            label="自制验收样例",
        )
        submit.click(demo.generate, [images, video, question, mode], [answer, status],
                     api_name="answer", concurrency_limit=1)
    return page.queue(max_size=6, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--adapter-sha256", help="Expected checksum of an explicitly supplied adapter")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    settings = json.loads(Path("configs/demo.json").read_text())
    if args.adapter_sha256 is not None:
        if args.adapter is None:
            parser.error("--adapter-sha256 requires --adapter")
        settings["adapter_sha256"] = args.adapter_sha256
    baseline = json.loads(Path("configs/baseline.json").read_text())
    demo = Demo(baseline, settings, args.model_path, args.adapter or Path(settings["adapter"]))
    build(demo).launch(
        server_name="127.0.0.1", server_port=args.port, share=args.share,
        max_file_size=settings["max_upload_mib"] * 2**20, show_error=False,
        blocked_paths=[str(Path(folder).resolve()) for folder in
                       ("checkpoints", "runs/ablation", "data/raw", "data/processed")],
        enable_monitoring=False, footer_links=[], theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
