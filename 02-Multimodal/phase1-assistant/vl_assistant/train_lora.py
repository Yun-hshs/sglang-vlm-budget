"""Run single-GPU answer-only LoRA SFT on the prepared ChartQA train manifest."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import re
from time import perf_counter

from peft import LoraConfig, TaskType, get_peft_model
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import file_sha256
from .infer import Baseline

LANGUAGE_PROJECTION = r".*\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def make_training_batch(assistant, image, question, answer):
    prompt, _, _ = assistant.prepare_inputs(question, [image])
    batch, _, _ = assistant.prepare_inputs(question, [image], response=answer)
    prefix_len = prompt["input_ids"].shape[-1]
    tokens = batch["input_ids"]
    if tokens.shape[-1] <= prefix_len or not torch.equal(tokens[:, :prefix_len], prompt["input_ids"]):
        raise ValueError("Chat template does not preserve the generation prompt prefix")
    # Supervise the answer and its closing turn marker, excluding image and user tokens.
    labels = tokens.clone()
    labels[:, :prefix_len] = -100
    batch["labels"] = labels
    return batch


def save_adapter(model, destination, baseline_config):
    model.save_pretrained(destination, safe_serialization=True)
    weights = destination / "adapter_model.safetensors"
    provenance = {"model": baseline_config["model"], "revision": baseline_config["revision"],
                  "sha256": file_sha256(weights)}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def train(config, baseline_config, model_path, manifest, output):
    samples = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not samples or any(s["split"] != "train" or not s["answers"] for s in samples):
        raise ValueError("Expected a nonempty ChartQA train manifest with reference answers")
    if config["epochs"] < 1:
        raise ValueError("At least one training epoch is required")
    for sample in samples:
        if file_sha256(manifest.parent / sample["image"]) != sample["sha256"]:
            raise ValueError(f"Image hash mismatch for {sample['id']}")

    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    output.mkdir(parents=True, exist_ok=False)
    assistant = Baseline({**baseline_config, "max_pixels": config["train_max_pixels"]}, model_path)
    torch.cuda.reset_peak_memory_stats()
    targets = [name for name, module in assistant.model.named_modules()
               if isinstance(module, torch.nn.Linear) and re.fullmatch(LANGUAGE_PROJECTION, name)]
    if not targets:
        raise RuntimeError("No matching language attention projections in the pretrained model")

    model = get_peft_model(assistant.model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config["rank"], lora_alpha=config["alpha"],
        lora_dropout=config["dropout"], target_modules=targets, bias="none",
        revision=baseline_config["revision"],
    ))
    # Record the published model ID instead of the path of the server-local checkpoint.
    model.peft_config["default"].base_model_name_or_path = baseline_config["model"]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    if len(trainable) != 2 * len(targets) or any(
        ".lora_A." not in name and ".lora_B." not in name for name in trainable
    ):
        raise RuntimeError("Expected only LoRA A/B matrices to be trainable")
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=config["learning_rate"], weight_decay=0)
    checkpoints = set(config.get("checkpoint_steps", []))
    total_steps = len(samples) * config["epochs"]
    if any(step < 1 or step > total_steps for step in checkpoints):
        raise ValueError("Checkpoint steps must be within the training schedule")
    losses = []
    start = perf_counter()
    with (output / "steps.jsonl").open("w") as log:
        for epoch in range(config["epochs"]):
            order = list(samples)
            rng.shuffle(order)
            for sample in order:
                question = sample["question"] + "\n" + assistant.config["chartqa_suffix"]
                batch = make_training_batch(assistant, manifest.parent / sample["image"],
                                            question, sample["answers"][0]).to("cuda:0")
                optimizer.zero_grad(set_to_none=True)
                with sdpa_kernel(SDPBackend.MATH):
                    loss = model(**batch, use_cache=False).loss
                    loss_value = loss.item()
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f"Nonfinite training loss for {sample['id']}")
                    loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    list(trainable.values()), config["gradient_clip_norm"], error_if_nonfinite=True
                ).item()
                optimizer.step()
                losses.append(loss_value)
                entry = {"step": len(losses), "epoch": epoch + 1, "sample_id": sample["id"],
                         "loss": loss_value, "grad_norm": norm,
                         "supervised_tokens": int((batch["labels"] != -100).sum().item())}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
                if len(losses) in checkpoints and len(losses) != total_steps:
                    save_adapter(model, output / f"step-{len(losses)}" / "adapter", baseline_config)
    if not any(t.detach().count_nonzero().item() for name, t in trainable.items() if ".lora_B." in name):
        raise RuntimeError("No LoRA B weights changed from zero initialization")
    adapter = output / "adapter"
    provenance = save_adapter(model, adapter, baseline_config)
    weights = adapter / "adapter_model.safetensors"
    stats = {
        **provenance, "adapter_bytes": weights.stat().st_size, "training": config,
        "train_manifest_sha256": file_sha256(manifest), "train_samples": len(samples),
        "training_steps": len(losses), "trainable_params": sum(p.numel() for p in trainable.values()),
        "target_layers": len(targets), "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
        "mean_training_loss": sum(losses) / len(losses),
        "training_wall_s": round(perf_counter() - start, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
    }
    (output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora-smoke.json"))
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), json.loads(args.baseline_config.read_text()),
          args.model_path, args.manifest, args.output)
