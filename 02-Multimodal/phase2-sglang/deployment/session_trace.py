"""Store only timings and token metadata for the version-pinned replay experiment."""
import json
import os
from contextvars import ContextVar
from queue import SimpleQueue
from threading import Thread
from uuid import uuid4


PREFIX = "stage2-mm-session-"
TRACE_PATH = os.environ.get("SGLANG_SESSION_TRACE_JSONL")
VISION_EVENTS = ContextVar("vision_events", default=None)
PENDING = None


def write_event(event):
    line = (json.dumps(event) + "\n").encode()
    fd = os.open(TRACE_PATH, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def emit_processor_trace(request, elapsed):
    rid = getattr(request, "rid", None)
    if TRACE_PATH and isinstance(rid, str) and rid.startswith(PREFIX):
        write_event({"kind": "processor", "request_id": rid, "mm_preprocess_s": elapsed})


def emit_runtime_trace(rid, meta):
    if TRACE_PATH:
        keys = (
            "prompt_tokens", "cached_tokens", "completion_tokens", "num_retractions",
            "queue_time", "api_server_dispatch_finish_ts", "request_received_ts",
            "forward_entry_time", "prefill_finished_time", "request_finished_ts",
            "first_token_latency", "e2e_latency", "finish_reason",
        )
        write_event({"kind": "request", "request_id": rid, **{key: meta.get(key) for key in keys}})


def drain_events(pending):
    while True:
        metadata, entire_stage, vision = pending.get()
        try:
            start, end = entire_stage
            end.synchronize()
            write_event({
                **metadata,
                "mm_preparation_cuda_ms": start.elapsed_time(end),
                "vision_forward_cuda_ms": sum(a.elapsed_time(b) for a, b in vision),
                "vision_forward_calls": len(vision),
            })
        except Exception as exc:
            write_event({**metadata, "error": f"{type(exc).__name__}: {exc}"})


def cuda_event():
    import torch
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def timed_mm_preparation(prepare, input_ids, forward_batch, *args, **kwargs):
    rids = forward_batch.rids or ()
    mm_inputs = forward_batch.mm_inputs or ()
    if not TRACE_PATH or not any(rid.startswith(PREFIX) for rid in rids) or not mm_inputs:
        return prepare(input_ids, forward_batch, *args, **kwargs)
    active = [rid for rid, mm in zip(rids, mm_inputs, strict=True) if mm is not None]
    if not active:
        return prepare(input_ids, forward_batch, *args, **kwargs)
    vision = []
    token = VISION_EVENTS.set(vision)
    started = cuda_event()
    try:
        result = prepare(input_ids, forward_batch, *args, **kwargs)
        ended = cuda_event()
    finally:
        VISION_EVENTS.reset(token)
    global PENDING
    if PENDING is None:
        PENDING = SimpleQueue()
        Thread(target=drain_events, args=(PENDING,), daemon=True).start()
    PENDING.put(({
        "kind": "vision_batch", "batch_id": uuid4().hex,
        "active_requests": active, "batch_requests": list(rids),
    }, (started, ended), vision))
    return result


def timed_vision_forward(forward, *args, **kwargs):
    events = VISION_EVENTS.get()
    if events is None:
        return forward(*args, **kwargs)
    start = cuda_event()
    result = forward(*args, **kwargs)
    events.append((start, cuda_event()))
    return result
