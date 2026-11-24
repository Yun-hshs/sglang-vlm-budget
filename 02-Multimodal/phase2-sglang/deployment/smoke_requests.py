"""Run the two image smoke requests and verify the matching processor traces."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from api.client import VLMClient, user_message
from preprocessing.image_budget import prepare_image


CASES = (
    ("single", ("day1.png",), "How many units of apples were sold? Answer with the number only.", 40),
    (
        "double", ("day1.png", "day2.png"),
        "Image 1 and Image 2 are shown in order. How many more apple units are sold in Image 2 "
        "than in Image 1? Answer with the number only.",
        30,
    ),
)


def verify_trace(trace_path, request_id, image_count, prompt_tokens, trace_offset=0):
    with trace_path.open("rb") as handle:
        handle.seek(trace_offset)
        events = map(json.loads, handle)
        matches = [event for event in events if event["request_id"] == request_id]
    if len(matches) != 1:
        raise ValueError(f"{request_id}: expected one processor trace, got {len(matches)}")
    event = matches[0]
    images = event["images"]
    if len(images) != image_count:
        raise ValueError(f"{request_id}: unexpected number of traced images")
    previous_end = -1
    for image in images:
        start, end = image["offset"]
        if (
            not previous_end < start <= end < event["processor_prompt_tokens"]
            or not 0 < image["merged_tokens"] == image["offset_tokens"] == image["image_pad_tokens"]
        ):
            raise ValueError(f"{request_id}: inconsistent per-image counts or offsets: {image}")
        previous_end = end
    if not sum(image["merged_tokens"] for image in images) == event["visual_tokens"] == event["prompt_image_pad_tokens"]:
        raise ValueError(f"{request_id}: visual-token totals disagree")
    if event["processor_prompt_tokens"] != prompt_tokens:
        raise ValueError(f"{request_id}: processor and API prompt-token totals disagree")
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=Path("results/server/vision.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="qwen25vl-7b")
    args = parser.parse_args()

    results = {}
    with VLMClient(base_url=args.base_url, model=args.model, timeout=300) as client:
        for name, files, question, expected in CASES:
            paths = [args.fixtures / filename for filename in files]
            images = [prepare_image(path, "baseline") for path in paths]
            request_id = f"stage2-mm-{name}-{uuid4().hex}"
            answer = client.complete([user_message(question, images)], max_tokens=24, request_id=request_id)
            print(f"{request_id} -> {answer.text!r} (prompt_tokens={answer.prompt_tokens})", flush=True)
            if not re.fullmatch(rf"{expected}(?:\s+units?)?[.!]?", answer.text.strip(), re.IGNORECASE):
                raise ValueError(f"{name}: expected {expected}, got {answer.text!r}")
            trace = verify_trace(args.trace, request_id, len(paths), answer.prompt_tokens)
            results[name] = {
                "question": question,
                "expected": expected,
                "inputs": [
                    {
                        "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "width": image.metadata["width"],
                        "height": image.metadata["height"],
                    }
                    for path, image in zip(paths, images, strict=True)
                ],
                "completion": answer.to_dict(),
                "server_trace": trace,
            }
    record = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "image_budget": "native_default",
        "status": "PASS",
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
