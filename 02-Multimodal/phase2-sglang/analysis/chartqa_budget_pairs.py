"""Audit paired ChartQA measurements and report per-question budget changes."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from vl_assistant.scoring import relaxed_correct


def analyze(manifest, rows, reference_strategy="baseline"):
    config = manifest["config"]
    samples = {row["id"]: row for row in manifest["samples"]}
    repeats = range(1, config["repeats"] + 1)
    repeat_keys = set(repeats)
    by_strategy = {strategy: {sample: {} for sample in samples} for strategy in config["strategies"]}
    warmup_sample_ids = manifest["warmup_sample_ids"]
    if isinstance(warmup_sample_ids, list):  # The 32-question pilot shared warmup IDs across strategies.
        warmup_sample_ids = dict.fromkeys(config["strategies"], warmup_sample_ids)
    expected_warmups = Counter(
        (strategy, sample, repeat)
        for strategy, warmups in warmup_sample_ids.items()
        for sample in warmups
        for repeat in repeats
    )
    seen_warmups = Counter()
    request_ids = set()
    for row in rows:
        rid = row["request_id"]
        if rid in request_ids or row["status"] != "ok":
            raise ValueError(f"Duplicate or failed request: {rid}")
        request_ids.add(rid)
        sample, strategy, repeat = row["id"], row["strategy"], row["repeat"]
        trace = row["server_trace"]
        image, = trace["images"]
        t, h, w = image["grid_thw"]
        start, end = image["offset"]
        merge = trace["spatial_merge_size"]
        if (
            row["subset"] != samples[sample]["subset"] or rid != trace["request_id"]
            or row.get("geometry_group") != samples[sample].get("geometry_group")
            or h % merge or w % merge
            or not 0 <= start <= end < trace["processor_prompt_tokens"]
            or len({
                t * (h // merge) * (w // merge), end - start + 1, image["image_pad_tokens"],
                image["offset_tokens"], image["merged_tokens"], trace["visual_tokens"],
                trace["prompt_image_pad_tokens"], row["image"]["visual_tokens_actual"],
            }) != 1
            or row["completion"]["prompt_tokens"] != trace["processor_prompt_tokens"]
            or row["correct"] != relaxed_correct(row["completion"]["text"], row["answers"])
        ):
            raise ValueError(f"Inconsistent trace or score: {rid}")
        if row["warmup"]:
            seen_warmups[strategy, sample, repeat] += 1
            continue
        records = by_strategy[strategy][sample]
        if repeat in records:
            raise ValueError(f"Duplicate measurement: {sample}, {strategy}, round {repeat}")
        records[repeat] = row
    if seen_warmups != expected_warmups:
        raise ValueError("Warmups differ from the manifest")

    reference = by_strategy[reference_strategy]
    output = {}
    for strategy, samples_by_id in by_strategy.items():
        switches = Counter()
        stable_answers = 0
        disagreements = []
        token_changes = 0
        for sample, records in samples_by_id.items():
            if set(records) != repeat_keys or set(reference[sample]) != repeat_keys:
                raise ValueError(f"Missing measurements for {sample}, {strategy}")
            predictions = [records[repeat]["completion"]["text"] for repeat in repeats]
            stable_answers += len(set(predictions)) == 1
            selected = {r["image"]["selected_budget"] for r in records.values()}
            if len(selected) != 1:
                raise ValueError(f"Budget selection changed across rounds: {sample}, {strategy}")
            switches[next(iter(selected))] += 1
            base = reference[sample]
            scores = [records[repeat]["correct"] for repeat in repeats]
            base_scores = [base[repeat]["correct"] for repeat in repeats]
            tokens = [records[repeat]["server_trace"]["visual_tokens"] for repeat in repeats]
            base_tokens = [base[repeat]["server_trace"]["visual_tokens"] for repeat in repeats]
            token_changes += tokens != base_tokens
            if scores != base_scores:
                disagreements.append({
                    "id": sample, "subset": records[1]["subset"],
                    "reference": records[1]["answers"], "selected_budget": next(iter(selected)),
                    f"{reference_strategy}_answers": [base[repeat]["completion"]["text"] for repeat in repeats],
                    "answers": predictions, f"{reference_strategy}_correct": base_scores, "correct": scores,
                    f"{reference_strategy}_visual_tokens": base_tokens, "visual_tokens": tokens,
                })
        output[strategy] = {
            "n_questions": len(samples),
            "stable_answers": stable_answers,
            f"changed_visual_tokens_vs_{reference_strategy}": token_changes,
            "selected_budgets": dict(switches),
            f"score_changes_vs_{reference_strategy}": disagreements,
        }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="baseline", help="Strategy used for paired differences")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    open_requests = gzip.open if args.requests.suffix == ".gz" else open
    with open_requests(args.requests, "rt", encoding="utf-8") as handle:
        result = analyze(manifest, map(json.loads, handle), args.reference)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
