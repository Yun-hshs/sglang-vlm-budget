"""Summarize request stages and CUDA batches for the visual session replay."""
import argparse
import json
from collections import defaultdict
from statistics import median, quantiles
from pathlib import Path


def latency(values):
    if not values:
        return None
    values = sorted(values)
    return {
        "p50_ms": median(values),
        "p90_ms": quantiles(values, n=10, method="inclusive")[8] if len(values) > 1 else values[0],
    }


def metrics(rows, batches):
    batch_ids = {batch_id for row in rows for batch_id in row["vision_batch_ids"]}
    joined = [batches[batch_id] for batch_id in sorted(batch_ids)]
    vision = [batch["vision_forward_cuda_ms"] for batch in joined if batch["vision_forward_calls"]]
    stages = {
        key: latency([r["stages_s"][key] * 1000 for r in rows])
        for key in rows[0]["stages_s"]
    }
    stages["dispatch_to_forward"] = latency([
        (r["runtime"]["forward_entry_time"] - r["runtime"]["api_server_dispatch_finish_ts"]) * 1000
        for r in rows
    ])
    return {
        "requests": len(rows), "exact_numeric_answers": sum(r["client"]["text"].strip() == r["reference"] for r in rows),
        "prompt_tokens": sorted({r["runtime"]["prompt_tokens"] for r in rows}),
        "cached_tokens": sorted({r["runtime"]["cached_tokens"] for r in rows}),
        "visual_tokens": sorted({r["visual"]["visual_tokens"] for r in rows}),
        "cached_visual_prefix_tokens": sorted({r["cached_visual_prefix_tokens"] for r in rows}),
        "client_ttft": latency([r["client"]["ttft_s"] * 1000 for r in rows]),
        "stages": stages, "mm_batches": len(joined),
        "mm_batch_cuda": latency([batch["mm_preparation_cuda_ms"] for batch in joined]),
        "vision_forward_batches": len(vision), "vision_forward_batch_cuda": latency(vision),
    }


def summarize(rows, batch_rows):
    batches = {row["batch_id"]: row for row in batch_rows}
    grouped = defaultdict(dict)
    for r in rows:
        key = (r["warmup"], r["turn"], r["round"], r["session"])
        if r["mode"] in grouped[key]:
            raise ValueError(f"Duplicate replay request: {key}")
        grouped[key][r["mode"]] = r
    by_turn = defaultdict(lambda: defaultdict(list))
    deltas = defaultdict(list)
    for (warmup, turn, _, _), by_mode in grouped.items():
        hit, miss = (by_mode[x] for x in ("hit", "miss"))
        for key in ("messages_sha256", "image_count", "kind"):
            if hit[key] != miss[key]:
                raise ValueError(f"Mismatch between paired requests: {key}")
        if hit["visual"]["visual_tokens"] != miss["visual"]["visual_tokens"]:
            raise ValueError("Paired visual tokens differ")
        if not warmup:
            by_turn[turn]["hit"].append(hit)
            by_turn[turn]["miss"].append(miss)
            deltas[turn].append((hit["client"]["ttft_s"] - miss["client"]["ttft_s"]) * 1000)
    summary = {}
    for turn, modes in sorted(by_turn.items()):
        summary[turn] = {
            **{mode: metrics(records, batches) for mode, records in modes.items()},
            "paired_ttft_delta_hit_minus_miss": latency(deltas[turn]),
        }
    return {
        "measured_by_turn": summary,
        "warmup_batches_with_vision_forward": [
            b for b in batch_rows
            if b["vision_forward_calls"] and any(
                r["warmup"] and b["batch_id"] in r["vision_batch_ids"] for r in rows
            )
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    with (args.run / "requests.jsonl").open() as reqs, (args.run / "vision_batches.jsonl").open() as vision:
        summary = summarize(list(map(json.loads, reqs)), list(map(json.loads, vision)))
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
