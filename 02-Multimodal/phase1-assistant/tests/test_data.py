import io
import json

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from vl_assistant.data import file_sha256, prepare


def chart_row(color, subset):
    image = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(image, format="PNG")
    return {"image": {"bytes": image.getvalue()}, "query": " What color? ",
            "label": [color, color], "human_or_machine": subset}


def test_prepare_rejects_cross_split_duplicate_and_preserves_alignment(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    test = [chart_row("red", 0), chart_row("blue", 1)]
    train = [chart_row("red", 0), chart_row("green", 0), chart_row("yellow", 1)]
    for split, rows in [("test", test), ("train", train)]:
        pq.write_table(pa.Table.from_pylist(rows), raw / f"{split}-sample.parquet")

    output = tmp_path / "processed"
    prepare(raw, output, 1, ["train", "test"])
    audit = json.loads((output / "audit.json").read_text())
    assert audit["splits"]["train"]["rejected_or_skipped"] == {"cross_split_image": 1}
    row = json.loads((output / "train.jsonl").read_text().splitlines()[0])
    assert row["id"] == "train-sample:1"
    assert row["question"] == "What color?"
    assert row["answers"] == ["green"]
    assert file_sha256(output / row["image"]) == row["sha256"]
