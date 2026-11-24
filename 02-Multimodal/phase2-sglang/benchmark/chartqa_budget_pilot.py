"""Run the paired ChartQA visual budget pilot on the trace-enabled local SGLang server."""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from inspect import getfile
from pathlib import Path
from statistics import median, quantiles
from uuid import uuid4

import httpx

from api.client import VLMClient, user_message
from deployment.smoke_requests import verify_trace
from preprocessing.image_budget import ADAPTIVE_RULES, BUDGETS, prepare_image, select_budget
from vl_assistant.scoring import relaxed_correct


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pixel_area(sample):
    return sample["width"] * sample["height"]


def select_geometry_sample(rows, config, excluded_images):
    spec = config["geometry_sample"]
    high, near = [], {}
    for row in sorted(rows, key=lambda sample: sha256(f"{config['seed']}:{sample['id']}".encode())):
        if row["sha256"] in excluded_images:
            continue
        width, height = row["width"], row["height"]
        if select_budget("adaptive_v2", width, height, "") == "high":
            high.append({**row, "geometry_group": "qualifies_high"})
        elif (
            height >= spec["near_min_height"] and height / width > spec["min_aspect"]
            and pixel_area(row) > spec["min_area"]
        ):
            near.setdefault(row["sha256"], {**row, "geometry_group": "near_height"})
    if not high or len(near) < spec["near_image_count"]:
        raise ValueError("The geometry groups have too few eligible images")
    return high + list(near.values())[:spec["near_image_count"]]


def load_sample(dataset_path, config, excluded_images=()):
    source = dataset_path.read_bytes()
    dataset = config["dataset"]
    if sha256(source) != dataset["jsonl_sha256"]:
        raise ValueError("The ChartQA JSONL checksum differs from the configured test export")
    rows = [json.loads(line) for line in source.splitlines()]
    if any(row["split"] != dataset["split"] for row in rows):
        raise ValueError("Unexpected split in test export")

    if "geometry_sample" in config:
        chosen = select_geometry_sample(rows, config, excluded_images)
    else:
        chosen = select_stratified_sample(rows, config, excluded_images)
    for row in chosen:
        if sha256((dataset_path.parent / row["image"]).read_bytes()) != row["sha256"]:
            raise ValueError(f"Chart file checksum mismatch for {row['id']}")
    return sorted(chosen, key=lambda row: row["id"])


def select_stratified_sample(rows, config, excluded_images):
    # Reserve scarce large charts first, picking one question per distinct image.
    strata = config["size_strata"]
    limits = [0] + [stratum["max_pixels"] for stratum in strata]
    chosen = []
    used_images = set(excluded_images)
    for index in reversed(range(len(strata))):
        minimum, maximum = limits[index:index + 2]
        quota = strata[index]["questions_per_subset"]
        for subset in config["subsets"]:
            count = quota[subset] if isinstance(quota, dict) else quota
            if count == 0:
                continue
            candidates = [
                row for row in rows
                if row["subset"] == subset and pixel_area(row) > minimum
                and (maximum is None or pixel_area(row) <= maximum)
                and (
                    not strata[index].get("require_high_geometry")
                    or select_budget("adaptive_v2", row["width"], row["height"], "") == "high"
                )
            ]
            ranked = sorted(candidates, key=lambda row: sha256(f"{config['seed']}:{row['id']}".encode()))
            group = {}
            for row in ranked:
                digest = row["sha256"]
                if digest not in used_images:
                    group[digest] = row
                    used_images.add(digest)
                if len(group) == count:
                    break
            if len(group) != count:
                raise ValueError(f"Too few distinct images for {subset} in pixel stratum {index}")
            chosen.extend(group.values())

    return chosen


def summarize(rows):
    ok = [row for row in rows if row["status"] == "ok"]
    tokens = [row["server_trace"]["visual_tokens"] for row in ok]
    ttft = [row["completion"]["ttft_s"] * 1000 for row in ok]
    correct = sum(row.get("correct", False) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "relaxed_accuracy": correct / len(rows),
        "errors": len(rows) - len(ok),
        "truncated": sum(row["completion"]["finish_reason"] == "length" for row in ok),
        "mean_visual_tokens": sum(tokens) / len(tokens) if tokens else None,
        "visual_tokens_p50": median(tokens) if tokens else None,
        "visual_tokens_range": [min(tokens), max(tokens)] if tokens else None,
        "ttft_p50_ms": median(ttft) if ttft else None,
        "ttft_p90_ms": quantiles(ttft, n=10, method="inclusive")[8] if len(ttft) > 1 else None,
    }


def request(client, sample, strategy, repeat, warmup, context):
    args, config, run_id = context
    request_id = f"stage2-mm-chartqa-{run_id}-{repeat}-{strategy}-{uuid4().hex}"
    prompt = f"{sample['question']}\n{config['prompt_suffix']}"
    row = {
        "id": sample["id"], "subset": sample["subset"], "strategy": strategy,
        "repeat": repeat, "warmup": warmup, "request_id": request_id, "status": "error",
        "answers": sample["answers"], "question": sample["question"],
    }
    if "geometry_group" in sample:
        row["geometry_group"] = sample["geometry_group"]
    try:
        image = prepare_image(args.dataset.parent / sample["image"], strategy, sample["question"])
        row["image"] = image.metadata
        trace_offset = args.trace.stat().st_size if args.trace.exists() else 0
        result = client.complete([user_message(prompt, [image])], config["max_tokens"], request_id)
        row["completion"] = result.to_dict()
        row["server_trace"] = verify_trace(args.trace, request_id, 1, result.prompt_tokens, trace_offset)
        row["image"]["visual_tokens_actual"] = row["server_trace"]["visual_tokens"]
        row["image"].pop("token_note")
        row["correct"] = relaxed_correct(result.text, sample["answers"])
        row["status"] = "ok"
    except (ValueError, OSError, RuntimeError, KeyError, httpx.HTTPError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run(args):
    config = json.loads(args.config.read_text())
    limits = {"baseline": None, **BUDGETS, **dict.fromkeys(ADAPTIVE_RULES)}
    strategies = config["strategies"]
    if not strategies or any(name not in limits or cap != limits[name] for name, cap in strategies.items()):
        raise ValueError("Configured strategy limits differ from the image preprocessor")
    frozen_sources = {
        "image_budget": sha256(Path(getfile(prepare_image)).read_bytes()),
        "chartqa_scoring": sha256(Path(getfile(relaxed_correct)).read_bytes()),
    }
    if "source_sha256" in config and config["source_sha256"] != frozen_sources:
        raise ValueError("Frozen budget or scoring source checksum differs from the configuration")
    exclusion_hashes = config.get("exclude_manifest_sha256", [])
    if isinstance(exclusion_hashes, str):
        exclusion_hashes = [exclusion_hashes]
    if len(exclusion_hashes) != len(args.exclude_manifest):
        raise ValueError("An exclusion manifest and its configured checksum are required together")
    excluded_images = set()
    for manifest_path, exclusion_sha in zip(args.exclude_manifest, exclusion_hashes, strict=True):
        exclusion_data = manifest_path.read_bytes()
        if sha256(exclusion_data) != exclusion_sha:
            raise ValueError("Exclusion manifest checksum differs from the configuration")
        manifest_images = {row["sha256"] for row in json.loads(exclusion_data)["samples"]}
        if not manifest_images:
            raise ValueError("Exclusion manifest contains no charts")
        excluded_images.update(manifest_images)
    samples = load_sample(args.dataset, config, excluded_images)
    sample_manifest = [
        {
            key: row[key]
            for key in ("id", "subset", "image", "sha256", "width", "height", "geometry_group")
            if key in row
        }
        for row in samples
    ]
    if args.sample_manifest:
        prior_samples = json.loads(args.sample_manifest.read_text())["samples"]
        if sample_manifest != prior_samples:
            raise ValueError("Sample IDs, order, or fingerprints differ from the reference run")
    warmup_field = "geometry_group" if "geometry_sample" in config else "subset"
    active_subsets = {sample["subset"] for sample in samples}
    if warmup_field == "geometry_group":
        warmup_groups = sorted({sample[warmup_field] for sample in samples})
    else:
        warmup_groups = [subset for subset in config["subsets"] if subset in active_subsets]
    fixed_warmups = [
        max((sample for sample in samples if sample[warmup_field] == group), key=pixel_area)
        for group in warmup_groups
    ]
    warmups = dict.fromkeys(strategies, fixed_warmups)
    largest_first = sorted(samples, key=pixel_area, reverse=True)
    for strategy in ADAPTIVE_RULES.keys() & strategies.keys():
        by_budget = {}
        for sample in largest_first:
            budget = select_budget(strategy, sample["width"], sample["height"], sample["question"])
            by_budget.setdefault(budget, sample)
        warmups[strategy] = list(by_budget.values())
    rng = random.Random(config["seed"])
    schedule = []
    for _ in range(config["repeats"]):
        strategy_order = list(strategies)
        sequence = [sample["id"] for sample in samples]
        rng.shuffle(strategy_order)
        rng.shuffle(sequence)
        schedule.append({"strategies": strategy_order, "sample_ids": sequence})

    run_id = uuid4().hex[:12]
    args.output.mkdir(parents=True, exist_ok=False)
    with VLMClient(base_url=args.base_url, model=config["server"]["served_model_name"]) as client:
        response = client.http.get(client.base_url.removesuffix("/v1") + "/server_info")
        response.raise_for_status()
        observed = response.json()
        server = {key: observed.get(key) for key in config["server"]}
        if server != config["server"]:
            raise ValueError(f"Unexpected serving configuration: {server}")
        manifest = {
            "run_id": run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "server": server,
            "samples": sample_manifest,
            "reference_manifest_sha256": (
                sha256(args.sample_manifest.read_bytes()) if args.sample_manifest else None
            ),
            "source_sha256": frozen_sources,
            "excluded_images_count": len(excluded_images),
            "warmup_sample_ids": {
                strategy: [row["id"] for row in warmup_rows]
                for strategy, warmup_rows in warmups.items()
            },
            "schedule": schedule,
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        by_id = {sample["id"]: sample for sample in samples}
        measured = {strategy: [] for strategy in config["strategies"]}
        context = (args, config, run_id)
        with (args.output / "requests.jsonl").open("w", buffering=1) as handle:
            for repeat, order in enumerate(schedule, start=1):
                for strategy in order["strategies"]:
                    rows = [(sample, True) for sample in warmups[strategy]]
                    rows.extend((by_id[sample_id], False) for sample_id in order["sample_ids"])
                    for sample, warmup in rows:
                        row = request(client, sample, strategy, repeat, warmup, context)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        if warmup and row["status"] != "ok":
                            raise RuntimeError(f"Warmup failed for {strategy}: {row['error']}")
                        if not warmup:
                            measured[strategy].append(row)
                    print(f"Round {repeat}: {strategy} complete", flush=True)

    summary = {}
    for strategy, rows in measured.items():
        summary[strategy] = {
            "overall": summarize(rows),
            "subsets": {
                subset: summarize([row for row in rows if row["subset"] == subset])
                for subset in config["subsets"] if subset in active_subsets
            },
            "repeats": {
                repeat: summarize([row for row in rows if row["repeat"] == repeat])
                for repeat in range(1, config["repeats"] + 1)
            },
        }
        if "geometry_sample" in config:
            summary[strategy]["geometry_groups"] = {
                group: summarize([row for row in rows if row["geometry_group"] == group])
                for group in warmup_groups
            }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(group["overall"]["errors"] for group in summary.values()):
        raise RuntimeError("At least one measured request failed; inspect requests.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/chartqa-budget-pilot.json"))
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--trace", type=Path, default=Path("results/server/vision.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    run(parser.parse_args())
