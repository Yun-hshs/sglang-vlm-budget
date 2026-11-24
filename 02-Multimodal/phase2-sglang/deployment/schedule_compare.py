"""Benchmark policies with client estimates or live scheduler features."""
import json
import os
from pathlib import Path
from time import perf_counter


RULES = json.loads(Path(os.environ["SCHEDULE_RULES"]).read_text()) if "SCHEDULE_RULES" in os.environ else None
PREFIX = "stage2-mm-session-sched:"


def arrange_experiment_queue(policy, queue, lpm, fcfs, running_batch=None, spf=None):
    if not RULES or not queue:
        return
    if any(not r.rid.startswith(PREFIX) for r in queue):
        policy.policy = fcfs
        return
    parts = [r.rid.removeprefix(PREFIX).split(":") for r in queue]
    modes = {p[0] for p in parts}
    if len(modes) != 1:
        raise RuntimeError("Overlapping benchmark strategies in the waiting queue")
    mode = modes.pop()
    policy.policy = lpm if mode == "lpm" else spf if mode == "spf" else fcfs
    thresholds = RULES.get("aging_thresholds_ms", {"aging": RULES["overdue_ms"]})
    probe = RULES.get("prefill_predictor_experiment", False)
    if mode not in {"fcfs", "lpm", "cost", "probe", "spf", *thresholds}:
        raise ValueError(f"Unknown experimental policy {mode}")
    rows = None
    if probe:
        from deployment.prefill_queue_features import snapshot, local_scores
        rows = snapshot(queue, parts, running_batch)
    if mode in {"fcfs", "lpm", "probe", "spf"}:
        return rows
    costs = {r.rid: int(p[1]) / 1000 for r, p in zip(queue, parts, strict=True)}
    if mode == "local":
        costs = local_scores(rows)
    now = perf_counter()

    def key(req):
        cost = costs[req.rid]
        since = req.time_stats.wait_queue_entry_time
        if mode == "cost":
            return (cost, since)
        wait_ms = (now - since) * 1000 if since else 0
        if wait_ms >= thresholds[mode]:
            return (0, since)
        return (1, cost - RULES["aging_rate"] * wait_ms, since)

    queue.sort(key=key)
    return rows
