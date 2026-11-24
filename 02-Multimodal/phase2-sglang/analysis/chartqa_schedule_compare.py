"""Summarize paired scheduling traces, including per-user waiting and long requests."""
import argparse
import collections
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def distribution(values):
    return {name: float(np.quantile(values, q)) for name, q in (("p50", .5), ("p95", .95), ("max", 1))}


def backlog(rows, window_ms):
    """Reconstruct waiting occupancy from request timing, within the constant-rate window."""
    origin = min(r["runtime"]["request_received_ts"] for r in rows)
    edges = collections.defaultdict(int)
    edges[0] = edges[window_ms] = 0
    for row in rows:
        end = (row["runtime"]["forward_entry_time"] - origin) * 1000
        wait = row["stages_s"]["scheduler_queue"] * 1000
        if wait < 5:
            continue
        start = max(0, end - wait)
        end = min(window_ms, end)
        if end > start:
            edges[start] += 1
            edges[end] -= 1
    depth = peak = occupied = area = previous = 0
    for now in sorted(edges):
        dt = now - previous
        occupied += dt if depth else 0
        area += dt * depth
        depth += edges[now]
        peak = max(peak, depth)
        previous = now
    return {
        "occupied_fraction": occupied / window_ms,
        "mean_waiting": area / window_ms,
        "peak_waiting": peak,
    }


def analyze(manifest, raw):
    config = manifest["config"]
    events = manifest["arrivals"]
    expected = {(e["session"], e["turn"]): e for e in events}
    threshold = float(np.quantile([e["cost_ms"] for e in events], config["long_quantile"]))
    all_trials = collections.defaultdict(list)
    for row in raw:
        event = expected[row["session"], row["turn"]]
        if any(row[key] != event[key] for key in event):
            raise ValueError("Replay request differs from frozen arrival trace")
        all_trials[row["policy"], row["trial"]].append(row)
    results = {}
    for policy in config["policies"]:
        rows = []
        seconds = wave_seconds = 0
        inversions = 0
        trial_backlog = []
        core_completed = 0
        for trial in range(config["repeats"] + 1):
            ordered = all_trials[policy, trial]
            if [(r["session"], r["turn"]) for r in ordered] != list(expected):
                raise ValueError("A trial does not contain the same arrival order")
            if trial == 0:
                continue
            ingress = [r["runtime"]["request_received_ts"] for r in ordered]
            inversions += sum(a > b for i,a in enumerate(ingress) for b in ingress[i+1:])
            if "session_replicas" in config:
                core_ms = 3 * config["turn_interval_ms"]
                trial_backlog.append(backlog(ordered, core_ms))
                core_completed += sum(r["completed_ms"] <= core_ms for r in ordered)
            seconds += max(r["completed_ms"] for r in ordered) / 1000
            waves = collections.defaultdict(list)
            for row in ordered:
                client_wait = row["sent_ms"] - row["scheduled_ms"]
                queue_ms = row["stages_s"]["scheduler_queue"] * 1000
                ttft = row["client"]["ttft_s"] * 1000 + client_wait
                net_and_api_ms = client_wait + (
                    row["runtime"]["forward_entry_time"] - row["runtime"]["request_received_ts"]
                ) * 1000
                later_first = sum(
                    other["scheduled_ms"] > row["scheduled_ms"]
                    and other["runtime"]["forward_entry_time"] < row["runtime"]["forward_entry_time"]
                    for other in ordered
                )
                entry = {
                    **row, "arrival_to_first_text_ms": ttft,
                    "queue_wait_ms": queue_ms + client_wait, "scheduler_queue_ms": queue_ms,
                    "arrival_to_forward_est_ms": net_and_api_ms, "later_first_forward": later_first,
                }
                waves[row["turn"]].append(entry)
                rows.append(entry)
            wave_seconds += sum(max(r["completed_ms"] for r in wave) - min(r["scheduled_ms"] for r in wave)
                                for wave in waves.values()) / 1000
        users = collections.defaultdict(list)
        measured_trials = collections.defaultdict(list)
        for row in rows:
            users[row["session"]].append(row)
            measured_trials[row["trial"]].append(row)
        long = [r for r in rows if r["cost_ms"] >= threshold]
        output_tokens = sum(r["runtime"]["completion_tokens"] for r in rows)
        results[policy] = {
            "requests": len(rows),
            "actual_prefix_hits": sum(r["runtime"]["cached_tokens"] > 0 for r in rows),
            "expected_hits_with_zero_match": sum(
                r["mode"] == "hit" and r["turn"] > 1 and r["runtime"]["cached_tokens"] == 0
                for r in rows
            ),
            "ttft_ms": distribution([r["arrival_to_first_text_ms"] for r in rows]),
            "queue_ms": distribution([r["queue_wait_ms"] for r in rows]),
            "scheduler_queue_ms": distribution([r["scheduler_queue_ms"] for r in rows]),
            "arrival_to_forward_est_ms": distribution([r["arrival_to_forward_est_ms"] for r in rows]),
            "ingress_order_inversions": inversions,
            "dependency_delayed_requests": sum(r.get("dependency_wait_ms", 0) > 0 for r in rows),
            "dependency_wait_max_ms": max(r.get("dependency_wait_ms", 0) for r in rows),
            "requests_per_second": len(rows) / seconds,
            "output_tokens_per_second": output_tokens / seconds,
            "active_wave_requests_per_second": None if trial_backlog else len(rows) / wave_seconds,
            "constant_rate_backlog_by_trial": trial_backlog,
            "constant_rate_requests_per_second": (
                core_completed * 1000 / (len(trial_backlog) * core_ms) if trial_backlog else None
            ),
            "user_wait_ms": {
                user: {"n": len(items), "queue": distribution([r["queue_wait_ms"] for r in items])}
                for user, items in sorted(users.items())
            },
            "measured_trials": {
                trial: {
                    "ttft_ms": distribution([r["arrival_to_first_text_ms"] for r in items]),
                    "long_queue_max_ms": max(r["queue_wait_ms"] for r in items if r["cost_ms"] >= threshold),
                    "long_over_watch": sum(
                        r["queue_wait_ms"] >= config["starvation_watch_ms"]
                        for r in items if r["cost_ms"] >= threshold
                    ),
                    "requests_per_second": len(items) * 1000 / max(r["completed_ms"] for r in items),
                }
                for trial, items in sorted(measured_trials.items())
            },
            "long": {
                "n": len(long), "queue_ms": distribution([r["queue_wait_ms"] for r in long]),
                "engine_queue_ms": distribution([r["scheduler_queue_ms"] for r in long]),
                "ttft_ms": distribution([r["arrival_to_first_text_ms"] for r in long]),
                "max_overtakes": max(r["later_first_forward"] for r in long),
                "over_watch": sum(r["queue_wait_ms"] >= config["starvation_watch_ms"] for r in long),
                "engine_over_watch": sum(r["scheduler_queue_ms"] >= config["starvation_watch_ms"] for r in long),
            },
            "finish_reasons": dict(collections.Counter(r["client"]["finish_reason"] for r in rows)),
        }
    return {"long_cost_threshold_ms": threshold, "policies": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    path = parser.parse_args().run
    manifest = json.loads((path / "manifest.json").read_text())
    requests = path / "requests.jsonl"
    handle = requests.open() if requests.is_file() else gzip.open(requests.with_suffix(".jsonl.gz"), "rt")
    with handle:
        summary = analyze(manifest, map(json.loads, handle))
    (path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if "session_replicas" in manifest["config"]:
        with (path / "user_wait.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["policy", "session", "requests", "queue_p50_ms", "queue_p95_ms", "queue_max_ms"])
            for name, policy in summary["policies"].items():
                for session, user in policy["user_wait_ms"].items():
                    q = user["queue"]
                    writer.writerow([name, session, user["n"], *(f"{q[k]:.3f}" for k in ("p50", "p95", "max"))])
