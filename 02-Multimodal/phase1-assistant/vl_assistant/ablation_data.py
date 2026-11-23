"""Prepare a fixed ChartQA training/validation sample and the intact official test set."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from pyarrow import concat_tables
from pyarrow.parquet import ParquetFile

from .data import REVISION, SUBSETS, file_sha256, normalize

SHARDS = {
    "train": {
        "train-00000-of-00003-49492f364babfa44.parquet": "169979f8a8c64ba93ac1be22916bf9b49c9c2e4c57f5c0104e6f671f2aacd83a",
        "train-00001-of-00003-7302bae5e425bbc7.parquet": "a8819291729c807f163b93dcb2dc1fe60044d56412a0c0626cb7312df7392140",
        "train-00002-of-00003-194c9400785577a2.parquet": "528133aec54c7e948f40b8af56b083b98f9b7d1fdcdd1cd9e375f3df984f7ae4",
    },
    "val": {
        "val-00000-of-00001-0f11003c77497969.parquet": "de58d649aaac8343e6a3fa5ab362a0458d5b5374f4112a95fea5d1d9253dd86e",
    },
    "test": {
        "test-00000-of-00001-e2cd0b7a0f9eb20d.parquet": "efcf31333368c485eb569ec7bd4cc3694bd0e8e1a06c503124b51d27c924491d",
    },
}


def prepare(raw, output, config):
    for files in SHARDS.values():
        for name, digest in files.items():
            if file_sha256(raw / name) != digest:
                raise ValueError(f"Pinned shard hash mismatch: {name}")

    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    rng = random.Random(config["dataset_seed"])
    audit = {"source": "HuggingFaceM4/ChartQA", "revision": REVISION, "shards": SHARDS,
             "source_counts": {}, "export_counts": {}, "skipped_training": Counter(),
             "manifest_sha256": {}}
    evaluation_images = set()
    validation_images = set()
    training_questions = set()
    for split in ("test", "val", "train"):
        # Keep shard and row identities even when sampling across shards.
        tables = [ParquetFile(raw / name).read() for name in SHARDS[split]]
        origins = [(Path(name).stem, row) for name, table in zip(SHARDS[split], tables)
                   for row in range(table.num_rows)]
        table = concat_tables(tables)
        labels = table["human_or_machine"].to_pylist()
        audit["source_counts"][split] = dict(Counter(SUBSETS[label] for label in labels))
        indices = list(range(len(origins)))
        if split == "train":
            rng.shuffle(indices)
        counts = Counter()
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as manifest:
            for index in indices:
                subset = SUBSETS[labels[index]]
                if split == "train" and counts[subset] >= config["train_per_subset"]:
                    continue
                question, answers, subset, size, png = normalize(table.slice(index, 1).to_pylist()[0])
                digest = hashlib.sha256(png).hexdigest()
                if split == "test":
                    evaluation_images.add(digest)
                elif split == "val":
                    validation_images.add(digest)
                    # Exclude test-image overlaps from the checkpoint selection pool.
                    if digest in evaluation_images:
                        continue
                else:
                    key = digest, " ".join(question.casefold().split())
                    if digest in evaluation_images:
                        audit["skipped_training"]["image_in_test_or_validation"] += 1
                        continue
                    if key in training_questions:
                        audit["skipped_training"]["duplicate_image_question"] += 1
                        continue
                    training_questions.add(key)
                filename = Path("images") / f"{digest}.png"
                if not (output / filename).exists():
                    (output / filename).write_bytes(png)
                shard, row = origins[index]
                item = {"id": f"{shard}:{row}", "split": split, "subset": subset,
                        "question": question, "answers": answers, "image": filename.as_posix(),
                        "sha256": digest, "width": size[0], "height": size[1]}
                manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[subset] += 1
                if split == "train" and sum(counts.values()) == 2 * config["train_per_subset"]:
                    break
        if split == "val":
            evaluation_images.update(validation_images)
        audit["export_counts"][split] = dict(counts)

    validation = output / "val.jsonl"
    pool = [json.loads(line) for line in validation.read_text().splitlines()]
    selected = []
    for subset in SUBSETS.values():
        selected.extend(rng.sample([row for row in pool if row["subset"] == subset],
                                   config["validation_per_subset"]))
    validation.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    audit["validation_selection"] = dict(Counter(row["subset"] for row in selected))
    if audit["export_counts"]["test"] != {"human": 1250, "augmented": 1250}:
        raise ValueError("Incomplete ChartQA test set")
    if audit["export_counts"]["train"] != dict.fromkeys(SUBSETS.values(), config["train_per_subset"]):
        raise ValueError("Training quota not met")
    for split in SHARDS:
        audit["manifest_sha256"][split] = file_sha256(output / f"{split}.jsonl")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["export_counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ablation.json"))
    args = parser.parse_args()
    prepare(args.raw, args.output, json.loads(args.config.read_text()))
