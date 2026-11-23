"""Upload original fixtures to a running Gradio Demo and check streamed answers."""
import argparse
import json
from pathlib import Path
import re
from time import perf_counter

from gradio_client import Client, handle_file

ADAPTED = "LoRA rank 16 · step 128"
BASE = "基座对照"


def check(url, fixtures, trust_env=True):
    client = Client(url, verbose=False, httpx_kwargs={"timeout": 120, "trust_env": trust_env})
    cases = [
        ("hd-image", ["day1.png"], None, ADAPTED,
         "How many units of apples were sold? Give the value.", r"\b40\b"),
        ("two-images", ["day1.png", "day2.png"], None, ADAPTED,
         "How many more apple units are sold in Image 2 than in Image 1?", r"\b30\b"),
        ("video", [], "red-blue.mp4", ADAPTED,
         "Which color fills the screen first, red or blue? Answer with one color.", r"\bred\b"),
        ("base-stream", ["day1.png"], None, BASE,
         "In two full English sentences, describe the chart and report the values for both fruits.",
         r"\b40\b"),
    ]
    results = []
    for name, images, video, mode, question, pattern in cases:
        started = perf_counter()
        job = client.submit(
            [handle_file(str(fixtures / path)) for path in images] or None,
            handle_file(str(fixtures / video)) if video else None,
            question, mode, api_name="/answer",
        )
        chunks = []
        for answer, status in job:
            if answer.strip() and (not chunks or chunks[-1] != answer):
                chunks.append(answer)
        answer, status = job.result()
        if not re.search(pattern, answer, flags=re.IGNORECASE) or "完成" not in status:
            raise RuntimeError(f"{name}: {answer!r} ({status})")
        if mode not in status:
            raise RuntimeError(f"{name}: incorrect model mode in status")
        if name == "base-stream" and (len(chunks) < 2 or not answer.startswith(chunks[0])):
            raise RuntimeError(f"{name}: expected distinct incremental text updates")
        result = {"case": name, "mode": mode, "answer": answer, "status": status,
                  "distinct_text_updates": len(chunks), "client_wall_s": round(perf_counter() - started, 2)}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--direct", action="store_true", help="Ignore client proxy settings")
    args = parser.parse_args()
    results = check(args.url, args.fixtures, trust_env=not args.direct)
    if args.output:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
