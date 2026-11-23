"""Evaluate a complete ChartQA manifest in batches, with restartable predictions."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image

from .data import file_sha256
from .infer import Baseline
from .scoring import relaxed_correct, summarize


def predict(assistant, manifest, samples):
    prompts, images = [], []
    for sample in samples:
        path = manifest.parent / sample["image"]
        if file_sha256(path) != sample["sha256"]:
            raise ValueError(f"Image hash mismatch: {sample['id']}")
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
        content = [{"type": "text", "text": "Image 1: "}, {"type": "image"},
                   {"type": "text", "text": sample["question"] + "\n" + assistant.config["chartqa_suffix"]}]
        prompts.append(assistant.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
    inputs = assistant.processor(
        text=prompts, images=images, padding=True, return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": 3136,
                                "longest_edge": assistant.config["max_pixels"]}},
    ).to("cuda:0")
    with torch.inference_mode():
        result = assistant.model.generate(**inputs, do_sample=False,
                                          max_new_tokens=assistant.config["max_new_tokens"])
    return assistant.processor.batch_decode(result[:, inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True)


def predict_with_retry(assistant, manifest, samples):
    try:
        return predict(assistant, manifest, samples)
    except torch.OutOfMemoryError:
        if len(samples) == 1:
            raise
    torch.cuda.empty_cache()
    middle = len(samples) // 2
    return (predict_with_retry(assistant, manifest, samples[:middle])
            + predict_with_retry(assistant, manifest, samples[middle:]))


def evaluate(manifest, model_path, adapter_path, output, config, batch_size):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    samples = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Manifest is empty or contains duplicate IDs")
    adapter = None
    if adapter_path:
        adapter = json.loads((adapter_path / "provenance.json").read_text())
        if any(adapter[key] != config[key] for key in ("model", "revision")):
            raise ValueError("Adapter base model or revision mismatch")
        if adapter["sha256"] != file_sha256(adapter_path / "adapter_model.safetensors"):
            raise ValueError("Adapter checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    signature = {"manifest_sha256": file_sha256(manifest), "config": config,
                 "batch_size": batch_size, "adapter": adapter, "attention": "sdpa_auto"}
    signature_file = output / "settings.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text()) != signature:
            raise ValueError("Run settings changed. Use a new output directory.")
    else:
        signature_file.write_text(json.dumps(signature, indent=2) + "\n")

    predictions = output / "predictions.jsonl"
    rows = []
    if predictions.exists():
        with predictions.open("rb+") as stream:
            while raw := stream.readline():
                if not raw.endswith(b"\n"):
                    stream.seek(-len(raw), 1)
                    stream.truncate()
                    break
                rows.append(json.loads(raw))
        if [row["id"] for row in rows] != [sample["id"] for sample in samples[:len(rows)]]:
            raise ValueError("Existing predictions do not match manifest order")

    if len(rows) < len(samples):
        assistant = Baseline(config, model_path)
        assistant.processor.tokenizer.padding_side = "left"
        if adapter_path:
            from peft import PeftModel
            assistant.model = PeftModel.from_pretrained(
                assistant.model, str(adapter_path), is_trainable=False, local_files_only=True
            ).eval()
        start = perf_counter()
        with predictions.open("a", encoding="utf-8", buffering=1) as stream:
            for offset in range(len(rows), len(samples), batch_size):
                batch = samples[offset:offset + batch_size]
                answers = predict_with_retry(assistant, manifest, batch)
                for sample, answer in zip(batch, answers, strict=True):
                    row = {key: sample[key] for key in ("id", "subset", "question", "answers", "image")}
                    row.update(prediction=answer.strip(), status="ok",
                               correct=relaxed_correct(answer, sample["answers"]))
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                if len(rows) >= len(samples) or (offset // batch_size) % 8 == 0:
                    print(f"{output.name}: {len(rows)}/{len(samples)} ({perf_counter() - start:.1f}s)",
                          flush=True)
    if len(rows) != len(samples):
        raise RuntimeError("Refusing to report scores for a partial manifest")
    report = {**signature, "model": config["model"], "revision": config["revision"],
              "scores": summarize(rows)}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["scores"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ablation.json"))
    args = parser.parse_args()
    evaluation = json.loads(args.config.read_text())["evaluation"]
    baseline = json.loads(Path("configs/baseline.json").read_text())
    batch_size = evaluation.pop("batch_size")
    evaluate(args.manifest, args.model_path, args.adapter, args.output,
             {**baseline, **evaluation}, batch_size)
