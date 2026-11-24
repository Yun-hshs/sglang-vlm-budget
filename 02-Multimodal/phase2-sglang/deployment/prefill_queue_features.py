"""Decision-time scheduler snapshots and the calibrated local score."""
import json
import os
from pathlib import Path


TRACE = os.getenv("PREFILL_QUEUE_TRACE")
SEEN = set()
FIT = None
FEATURES = ("uncached", "prefix", "attention_work", "visual_miss", "running", "pending")


def snapshot(queue, parts, running_batch):
    from sglang.srt.managers import mm_schedule

    cache = mm_schedule.embedding_cache
    running = len(running_batch.reqs) if running_batch is not None else 0
    rows = {}
    for request, fields in zip(queue, parts, strict=True):
        prefix, total = map(int, fields[-2:])
        uncached = max(1, total - prefix)
        mm = request.multimodal_inputs
        items = mm.mm_items if mm is not None else ()
        visual_miss = sum(
            max(0, end + 1 - max(start, prefix))
            for item in items if cache is None or not cache.has(item.hash)
            for start, end in (item.offsets or ())
        )
        rows[request.rid] = dict(zip(FEATURES, (
            uncached, prefix, uncached * (prefix + uncached / 2), visual_miss, running, len(queue),
        ), strict=True))
    return rows


def record(rows, queue):
    if TRACE and rows and any(rid not in SEEN for rid in rows):
        SEEN.update(rows)
        native = {r.rid: max(1, len(r.origin_input_ids) + len(r.output_ids) - r.num_matched_prefix_tokens)
                  for r in queue}
        payload = (json.dumps({"features": rows, "native": native}) + "\n").encode()
        descriptor = os.open(TRACE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)


def predict(values, fit, floor=0):
    prediction = fit["intercept_ms"] + sum(
        (values[name] - mu) / scale * weight
        for name, mu, scale, weight in zip(
            fit["features"], fit["centers"], fit["scales"], fit["standardized_weights"], strict=True
        )
    )
    return max(floor, prediction) if floor is not None else prediction


def local_scores(rows):
    global FIT
    if FIT is None:
        FIT = json.loads(Path(os.environ["PREFILL_CALIBRATOR_PATH"]).read_text())["models"]["local"]["fit"]
    return {rid: predict(values, FIT) for rid, values in rows.items()}
