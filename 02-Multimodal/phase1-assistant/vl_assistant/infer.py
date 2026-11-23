"""Single-GPU BF16 inference for ChartQA and image/video acceptance cases."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import av
from PIL import Image, ImageOps
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.video_utils import VideoMetadata

from .data import file_sha256
from .scoring import relaxed_correct, summarize


def sample_video(path, max_seconds, max_frames):
    frames, times = [], []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("File has no video stream")
        stream = container.streams.video[0]
        duration = (float(stream.duration * stream.time_base) if stream.duration
                    else float(container.duration or 0) / av.time_base)
        if not 0 < duration <= max_seconds:
            raise ValueError(f"Video duration must be known and within {max_seconds}s")
        origin = float((stream.start_time or 0) * stream.time_base)
        interval = duration / max_frames
        for frame in container.decode(video=0):
            if frame.time is None:
                continue
            timestamp = max(0.0, frame.time - origin)
            if timestamp >= len(frames) * interval:
                frames.append(frame.to_image())
                times.append(round(timestamp, 4))
            if len(frames) == max_frames:
                break
    if len(frames) < 2:
        raise ValueError("Video must provide at least two sampled frames")
    metadata = VideoMetadata(total_num_frames=len(frames), fps=len(frames) / duration)
    return frames, times, metadata


class Baseline:
    def __init__(self, config, model_path=None):
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
        if config["dtype"] != "bfloat16" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This baseline requires a single BF16-capable CUDA GPU")
        self.config = config
        source = str(model_path) if model_path else config["model"]
        options = {"local_files_only": True} if model_path else {"revision": config["revision"]}
        self.processor = AutoProcessor.from_pretrained(source, **options)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa", **options,
        ).eval()

    def prepare_inputs(self, question, images=(), video=None, response=None):
        if not question.strip() or bool(images) == bool(video) or len(images) > 4:
            raise ValueError("Provide a question and one to four images or one video")
        content, media, times, kwargs = [], [], [], {}
        for index, path in enumerate(images, start=1):
            with Image.open(path) as source:
                media.append(ImageOps.exif_transpose(source).convert("RGB"))
            content.extend([{"type": "text", "text": f"Image {index}: "}, {"type": "image"}])
        if video:
            frames, times, metadata = sample_video(
                video, self.config["max_video_seconds"], self.config["max_video_frames"]
            )
            content.append({"type": "video"})
            kwargs["videos"] = [frames]
            kwargs["videos_kwargs"] = {"video_metadata": [metadata],
                                       "do_sample_frames": False,
                                       "size": {"shortest_edge": 3136,
                                                "longest_edge": self.config["video_max_pixels"]}}
        if media:
            kwargs["images"] = media
            kwargs["images_kwargs"] = {"size": {"shortest_edge": 3136,
                                                "longest_edge": self.config["max_pixels"]}}
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        if response is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=response is None
        )
        inputs = self.processor(text=prompt, return_tensors="pt", **kwargs)
        return inputs, times, [list(image.size) for image in media]

    @torch.inference_mode()
    def answer(self, question, images=(), video=None):
        inputs, times, image_sizes = self.prepare_inputs(question, images, video)
        inputs = inputs.to("cuda:0")
        torch.cuda.synchronize()
        start = perf_counter()
        # The generic CUDA math backend keeps this baseline independent of Triton JIT.
        with sdpa_kernel(SDPBackend.MATH):
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.config["max_new_tokens"]
            )
        torch.cuda.synchronize()
        generated = output[0, inputs["input_ids"].shape[1]:]
        result = {"prediction": self.processor.decode(generated, skip_special_tokens=True).strip(),
                  "generation_s": round(perf_counter() - start, 4),
                  "output_tokens": generated.numel(), "frame_timestamps_s": times,
                  "image_sizes": image_sizes}
        for key in ("image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
            if key in inputs:
                value = inputs[key]
                result[key] = value.tolist() if hasattr(value, "tolist") else value
        return result


def evaluate(model, source, output, chartqa, adapter=None):
    samples = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError("Empty manifest")
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w") as raw:
        for sample in samples:
            result = {key: sample[key] for key in ("id", "subset", "question", "answers")}
            try:
                question = sample["question"]
                if chartqa:
                    image = source.parent / sample["image"]
                    if file_sha256(image) != sample["sha256"]:
                        raise ValueError("Manifest image hash mismatch")
                    images, video = [image], None
                    question += "\n" + model.config["chartqa_suffix"]
                else:
                    images = [source.parent / path for path in sample.get("images", [])]
                    video = source.parent / sample["video"] if "video" in sample else None
                result.update(model.answer(question, images, video))
                result["status"] = "ok"
                result["correct"] = relaxed_correct(result["prediction"], sample["answers"])
            except (ValueError, OSError, RuntimeError) as error:
                result.update(status="error", correct=False, error=str(error))
            rows.append(result)
            raw.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw.flush()
            print(result["id"], result.get("prediction", result.get("error")), flush=True)
    metadata = {"model": model.config["model"], "revision": model.config["revision"],
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "cuda": torch.version.cuda, "attention": "sdpa_math", "config": model.config,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "manifest_sha256": file_sha256(source), "adapter": adapter, "scores": summarize(rows)}
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter", type=Path, help="Load a verified Phase 1 LoRA adapter")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", action="store_true", help="Use the acceptance case format")
    args = parser.parse_args()
    model = Baseline(json.loads(args.config.read_text()), args.model_path)
    adapter = None
    if args.adapter:
        from .adapter import attach_adapter

        adapter = attach_adapter(model, args.adapter)
    evaluate(model, args.manifest, args.output, chartqa=not args.cases, adapter=adapter)


if __name__ == "__main__":
    main()
