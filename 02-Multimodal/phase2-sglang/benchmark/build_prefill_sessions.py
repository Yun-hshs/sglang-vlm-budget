"""Create deterministic, image-disjoint replay workloads from the existing ChartQA export."""
import argparse
import json
from pathlib import Path

from benchmark.chartqa_session_replay import digest


NOTES = (
    "Use only the chart attached to the current question. Keep the units as the chart presents them.",
    "Treat the chart labels and legend as the reference for the current review. Pay attention to axis units.",
    "This is part of a multi-chart review. Keep earlier charts in their original order and answer from the requested chart.",
    "Labels can be similar across series, so check the requested category, the date, and the relevant legend before giving the result.",
)


def build(config, raw, excluded=()):
    if digest(raw) != config["dataset_sha256"]:
        raise ValueError("ChartQA export checksum mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    rows.sort(key=lambda r: digest(f"{config['seed']}:{r['id']}".encode()))
    buckets = [[], [], []]
    seen = set(excluded)
    for row in rows:
        if row["sha256"] in seen:
            continue
        size = row["width"] * row["height"]
        bucket = buckets[0 if size < 250_000 else 1 if size < 600_000 else 2]
        if len(bucket) < 4:
            bucket.append(row)
            seen.add(row["sha256"])
    if any(len(bucket) != 4 for bucket in buckets):
        raise ValueError("Insufficient source charts")
    pools = {
        "train": [r for bucket in buckets for r in bucket[:2]],
        "validation": [r for bucket in buckets for r in bucket[2:]],
    }
    sessions = []
    fixtures = {}
    for cohort, (split, length) in enumerate(config["cohorts"]):
        pool = pools[split]
        for member in range(3):
            turns = []
            for idx in range(length):
                notes = " ".join(NOTES[:(cohort + member + idx) % 5])
                if idx in (0, 2) or (idx == 4 and member % 2 == 0):
                    slot = idx // 2 if config.get("spread_chart_pool") else idx
                    chart = pool[(cohort + member * 2 + slot) % len(pool)]
                    fixtures[chart["image"]] = chart["sha256"]
                    question = f"{notes} Using the latest chart: {chart['question']}"
                    kind = "initial" if idx == 0 else "append_image"
                    images = [chart["image"]]
                    answer = chart["answers"][0]
                else:
                    question = f"{notes} Looking at the same chart, repeat your previous short answer."
                    kind = "same_image"
                    images = []
                    answer = turns[-1]["answer"]
                turns.append({"kind": kind, "images": images, "question": question.strip(), "answer": answer})
            sessions.append({"id": f"{split}-{cohort}-{member}", "cohort": cohort, "split": split, "turns": turns})
    return {
        "model": "qwen25vl-7b", "fixtures": fixtures, "sessions": sessions,
        "repeats": config["repeats"], "max_tokens": config["max_tokens"],
        "prediction_protocol": config,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    excluded = ()
    if args.exclude_manifest or config.get("exclude_manifest_sha256"):
        if not args.exclude_manifest:
            parser.error("A pinned exclusion manifest is required")
        source = args.exclude_manifest.read_bytes()
        if digest(source) != config.get("exclude_manifest_sha256"):
            raise ValueError("Exclusion manifest checksum mismatch")
        excluded = json.loads(source)["config"]["fixtures"].values()
    workload = build(config, args.dataset.read_bytes(), excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(workload, indent=2, ensure_ascii=False) + "\n")
