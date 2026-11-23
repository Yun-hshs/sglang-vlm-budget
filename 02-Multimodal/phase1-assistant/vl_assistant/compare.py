"""Compare paired ChartQA predictions from the same manifest and model settings."""
import argparse
import json
from pathlib import Path

from .scoring import summarize


def read_run(directory):
    summary = json.loads((directory / "summary.json").read_text())
    rows = {}
    for line in (directory / "predictions.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        rows[row["id"]] = row
    return summary, rows


def compare(base_directory, lora_directory):
    base_summary, base = read_run(base_directory)
    lora_summary, lora = read_run(lora_directory)
    for key in ("model", "revision", "config", "attention", "manifest_sha256"):
        if base_summary[key] != lora_summary[key]:
            raise ValueError(f"Run mismatch: {key}")
    if base_summary.get("adapter") or not lora_summary.get("adapter"):
        raise ValueError("Expected an unmodified base run and a reloaded adapter run")
    if not base or base.keys() != lora.keys():
        raise ValueError("Evaluation samples do not match")

    paired = []
    for sample_id, baseline in base.items():
        adapted = lora[sample_id]
        for key in ("subset", "question", "answers"):
            if baseline[key] != adapted[key]:
                raise ValueError(f"Sample mismatch: {sample_id} ({key})")
        paired.append({
            "id": sample_id, "subset": baseline["subset"],
            "base_prediction": baseline.get("prediction"), "lora_prediction": adapted.get("prediction"),
            "base_correct": baseline["correct"], "lora_correct": adapted["correct"],
        })
    base_scores, lora_scores = summarize(list(base.values())), summarize(list(lora.values()))
    return {
        "manifest_sha256": base_summary["manifest_sha256"],
        "adapter_sha256": lora_summary["adapter"]["sha256"],
        "scores": {group: {"base": base_scores[group], "lora": lora_scores[group]}
                   for group in base_scores},
        "paired": {
            "improved": sum(not row["base_correct"] and row["lora_correct"] for row in paired),
            "regressed": sum(row["base_correct"] and not row["lora_correct"] for row in paired),
            "changed_prediction": sum(row["base_prediction"] != row["lora_prediction"] for row in paired),
        },
        "rows": paired,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("lora", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.base, args.lora)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
