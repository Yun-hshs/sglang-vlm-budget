"""Train both ranks on one GPU, choose checkpoints on validation, then run full test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import file_sha256
from .compare import compare


def run_module(module, *args):
    subprocess.run([sys.executable, "-m", f"vl_assistant.{module}", *map(str, args)], check=True)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def run(data, output, model_path, plan_path):
    plan = json.loads(plan_path.read_text())
    audit = json.loads((data / "audit.json").read_text())
    if audit["export_counts"]["test"] != {"human": 1250, "augmented": 1250}:
        raise ValueError("The complete ChartQA test set is required")
    for split, expected in audit["manifest_sha256"].items():
        if file_sha256(data / f"{split}.jsonl") != expected:
            raise ValueError(f"Data manifest changed: {split}")

    output.mkdir(parents=True, exist_ok=True)
    model_args = ("--model-path", model_path) if model_path else ()
    signature = {"plan": plan, "data": audit["manifest_sha256"],
                 "baseline_sha256": file_sha256(Path("configs/baseline.json"))}
    signature_path = output / "experiment.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Experiment settings changed. Use a fresh output directory.")
    write_json(signature_path, signature)

    checkpoints = plan["training"]["checkpoint_steps"]
    selections = {}
    for rank in plan["training_ranks"]:
        rank_root = output / f"rank-{rank}"
        rank_root.mkdir(exist_ok=True)
        config_path = rank_root / "training-config.json"
        write_json(config_path, {**plan["training"], "rank": rank,
                                 "alpha": plan["alpha_per_rank"] * rank})
        training = rank_root / "train"
        if not (training / "summary.json").exists():
            run_module("train_lora", *model_args, "--config", config_path,
                       "--manifest", data / "train.jsonl", "--output", training)
        summary = json.loads((training / "summary.json").read_text())
        candidates = []
        for step in checkpoints:
            adapter = (training if step == summary["training_steps"]
                       else training / f"step-{step}") / "adapter"
            val = rank_root / "validation" / f"step-{step}"
            run_module("ablation_eval", *model_args, "--manifest", data / "val.jsonl",
                       "--output", val, "--adapter", adapter, "--config", plan_path)
            val_summary = json.loads((val / "summary.json").read_text())
            candidates.append({"step": step, "adapter": str(adapter.relative_to(output)),
                               "sha256": val_summary["adapter"]["sha256"],
                               "scores": val_summary["scores"]})
        best = max(candidates, key=lambda row: (row["scores"]["overall"]["correct"], -row["step"]))
        selections[str(rank)] = {"selected": best, "candidates": candidates}
    selection_path = output / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selections:
        raise ValueError("Checkpoint selection differs from the locked selection")
    write_json(selection_path, selections)

    base = output / "test" / "base"
    run_module("ablation_eval", *model_args, "--manifest", data / "test.jsonl",
               "--output", base, "--config", plan_path)
    for rank, decision in selections.items():
        test = output / "test" / f"rank-{rank}"
        run_module("ablation_eval", *model_args, "--manifest", data / "test.jsonl",
                   "--output", test, "--adapter", output / decision["selected"]["adapter"],
                   "--config", plan_path)
        write_json(output / f"comparison-rank-{rank}.json", compare(base, test))
    run_module("ablation_report", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/ablation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/ablation.json"))
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["TORCH_DISABLE_NATIVE_JIT"] = "1"
    run(args.data, args.output, args.model_path, args.config)
