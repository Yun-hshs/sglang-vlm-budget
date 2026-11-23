"""Build a small deterministic, split-preserving ChartQA manifest from parquet."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps
from pyarrow.parquet import ParquetFile

REVISION = "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5"
SUBSETS = {0: "human", 1: "augmented"}


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize(row):
    question = row["query"]
    labels = row["label"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("empty_question")
    if not isinstance(labels, list):
        raise ValueError("invalid_labels")
    answers = list(dict.fromkeys(label.strip() for label in labels
                                 if isinstance(label, str) and label.strip()))
    if not answers:
        raise ValueError("empty_answer")
    if row["human_or_machine"] not in SUBSETS:
        raise ValueError("unknown_subset")
    media = row["image"]
    if not media or not media.get("bytes"):
        raise ValueError("missing_image_bytes")
    with Image.open(io.BytesIO(media["bytes"])) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("image_too_large")
        image = ImageOps.exif_transpose(source).convert("RGB")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    return question.strip(), answers, SUBSETS[row["human_or_machine"]], image.size, payload.getvalue()


def prepare(raw, output, per_subset, splits):
    if per_subset < 1:
        raise ValueError("--per-subset must be positive")
    inputs = {split: sorted(raw.glob(f"{split}-*.parquet")) for split in splits}
    for split, files in inputs.items():
        if not files:
            raise FileNotFoundError(f"No {split} parquet shard in {raw}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    image_owner = {}
    seen = set()
    audit = {"source": "HuggingFaceM4/ChartQA", "revision": REVISION,
             "selection": "First valid rows per subset in shard and row order",
             "splits": {}}
    # Prioritize evaluation when exact duplicates occur across selected splits.
    for split in ("test", "val", "train"):
        if split not in inputs:
            continue
        counts = Counter()
        rejects = Counter()
        examined = 0
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as manifest:
            for shard in inputs[split]:
                parquet = ParquetFile(shard)
                row_index = 0
                for batch in parquet.iter_batches(batch_size=32,
                        columns=["image", "query", "label", "human_or_machine"]):
                    for row in batch.to_pylist():
                        if all(counts[name] >= per_subset for name in SUBSETS.values()):
                            break
                        row_index += 1
                        examined += 1
                        subset = SUBSETS.get(row["human_or_machine"])
                        if subset and counts[subset] >= per_subset:
                            rejects["subset_quota"] += 1
                            continue
                        try:
                            question, answers, subset, size, payload = normalize(row)
                        except (ValueError, OSError, Image.DecompressionBombError) as error:
                            reason = str(error) if isinstance(error, ValueError) else "decode_failed"
                            rejects[reason] += 1
                            continue
                        digest = hashlib.sha256(payload).hexdigest()
                        if digest in image_owner and image_owner[digest] != split:
                            rejects["cross_split_image"] += 1
                            continue
                        key = (digest, " ".join(question.casefold().split()))
                        if key in seen:
                            rejects["duplicate_question"] += 1
                            continue
                        seen.add(key)
                        image_owner[digest] = split
                        relative = Path("images") / f"{digest}.png"
                        (output / relative).write_bytes(payload)
                        item = {"id": f"{shard.stem}:{row_index - 1}", "split": split,
                                "subset": subset, "question": question, "answers": answers,
                                "image": relative.as_posix(), "sha256": digest,
                                "width": size[0], "height": size[1]}
                        manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
                        counts[subset] += 1
                    if all(counts[name] >= per_subset for name in SUBSETS.values()):
                        break
                if all(counts[name] >= per_subset for name in SUBSETS.values()):
                    break
        audit["splits"][split] = {"examined": examined, "selected": dict(counts),
                                   "rejected_or_skipped": dict(rejects),
                                   "inputs": {s.name: file_sha256(s) for s in inputs[split]}}
    audit["complete"] = all(values["selected"].get(name, 0) == per_subset
                            for values in audit["splits"].values() for name in SUBSETS.values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["complete"]:
        raise ValueError("Some subset quotas were not met; see audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-subset", type=int, default=4)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"],
                        default=["train", "val", "test"])
    args = parser.parse_args()
    prepare(args.raw, args.output, args.per_subset, args.splits)
