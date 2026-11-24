"""Replay a fixed multi-session arrival trace against four single-engine policies."""
import argparse
import collections
import json
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter, sleep
from uuid import uuid4

from api.client import VLMClient
from benchmark.chartqa_session_replay import build_prompts, digest, join, read_traces
from deployment.prefill_queue_features import predict


def estimate(prompt, namespace_reusable, fit):
    f = prompt["features"]
    values = {**f, "prefix_tokens_est": f["previous_prompt_tokens_est"] if namespace_reusable else 0}
    adjustment = sum((values[k] - center) / scale * w for k, center, scale, w in zip(
        fit["features"], fit["centers"], fit["scales"], fit["standardized_weights"], strict=True
    ))
    return max(0, fit["intercept_ms"] + adjustment)


def arrivals(workload, prompts, config, fit):
    events = []
    sessions = sorted(workload["sessions"], key=lambda s: s["id"])
    reuse = {session["id"] for session in sessions[::2]}
    sustained_slots = {}
    if "session_replicas" in config:
        stable_order = sorted((s["id"] for s in sessions), key=lambda name: digest(f"2026112406:{name}".encode()))
        sustained_slots = {session: i for i, session in enumerate(stable_order)}
    for turn in range(1, max(len(s["turns"]) for s in sessions) + 1):
        eligible = [session["id"] for session in sessions if len(session["turns"]) >= turn]
        if sustained_slots:
            eligible.sort(key=sustained_slots.__getitem__)
        else:
            random.Random(2026112405 + turn).shuffle(eligible)
        for position, session in enumerate(eligible):
            position = sustained_slots.get(session, position)
            prompt = prompts[session, turn]
            hit = session in reuse
            event = {
                "session": session, "turn": turn,
                "scheduled_ms": (turn - 1) * config["turn_interval_ms"] + position * config["arrival_spacing_ms"],
                "mode": "hit" if hit else "miss", "cost_ms": estimate(prompt, hit, fit),
                "messages_sha256": digest(json.dumps(prompt["messages"], sort_keys=True).encode()),
            }
            if config.get("prefill_predictor_experiment"):
                features = prompt["features"]
                event["prefix_est"] = features["previous_prompt_tokens_est"] if hit else 0
                event["total_est"] = features["input_tokens_est"]
            events.append(event)
    return events


def issue(client, event, prompt, policy, trial, run_id, max_tokens, origin, previous=None, token_fit=None):
    cost = event["cost_ms"]
    if policy == "tokens":
        cost = predict({"uncached": max(1, event["total_est"] - event["prefix_est"])}, token_fit, floor=None)
    encoded_cost = round(cost * 1000)
    rid = f"stage2-mm-session-sched:{policy}:{encoded_cost}:{run_id}:{trial}:{event['session']}:{event['turn']}"
    if "prefix_est" in event:
        rid += f":{event['prefix_est']}:{event['total_est']}"
    salt = f"{run_id}:{trial}:{policy}:{event['session']}"
    if event["mode"] == "miss":
        salt += f":{event['turn']}"
    dependency_wait_ms = 0
    if previous is not None and not previous.done():
        before = perf_counter()
        previous.result()
        dependency_wait_ms = (perf_counter() - before) * 1000
    elif previous is not None:
        previous.result()
    sent = perf_counter()
    response = client.complete(prompt["messages"], max_tokens, rid, cache_salt=salt)
    finished = perf_counter()
    return {
        **event, **{name: prompt[name] for name in ("kind", "image_count", "new_images", "reference")},
        "request_id": rid, "policy": policy, "trial": trial, "warmup": trial == 0,
        "sent_ms": (sent - origin) * 1000, "completed_ms": (finished - origin) * 1000,
        "dependency_wait_ms": dependency_wait_ms,
        "client": response.to_dict(),
    }


def run(args):
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text())
    manifest = json.loads((args.predictor / "manifest.json").read_text())
    predictor_summary = (args.predictor / "summary.json").read_bytes()
    predictor = json.loads(predictor_summary)
    workload_bytes = args.workload.read_bytes() if args.workload else None
    if workload_bytes:
        workload = json.loads(workload_bytes)
        workload["sessions"] = [s for s in workload["sessions"] if s["split"] == args.split]
    else:
        workload = {**manifest["config"], "sessions": [
            s for s in manifest["config"]["sessions"] if s["id"] in predictor["validation_sessions"]
        ]}
    calibrator = args.calibrator.read_bytes() if args.calibrator else None
    token_fit = json.loads(calibrator)["models"]["tokens"]["fit"] if calibrator else None
    if "tokens" in config["policies"] and not token_fit:
        raise ValueError("An affine token calibration is required for aging in milliseconds")
    if "session_replicas" in config:
        workload["sessions"] = [
            {**session, "id": f"{session['id']}-r{replica:02}"}
            for session in workload["sessions"]
            for replica in range(config["session_replicas"])
        ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompts = build_prompts(workload, args.fixtures, tokenizer)
    events = arrivals(workload, prompts, config, predictor["models"]["visual_history_cache"]["fit"])
    if args.reference_trace or "reference_trace_sha256" in config:
        if args.reference_trace is None:
            raise ValueError("A pinned reference trace is required")
        reference = args.reference_trace.read_bytes()
        if "reference_trace_sha256" in config and digest(reference) != config["reference_trace_sha256"]:
            raise ValueError("Reference trace checksum mismatch")
        original = json.loads(reference)
        if (
            original["arrivals"] != events
            or original["predictor_sha256"] != digest(predictor_summary)
            or workload_bytes and (
                original.get("workload_sha256"), original.get("split")
            ) != (digest(workload_bytes), args.split)
        ):
            raise ValueError("Replay differs from the sustained arrival trace or frozen predictor")
    offsets = [p.stat().st_size if p.exists() else 0 for p in (args.vision_trace, args.session_trace)]
    with VLMClient(base_url=args.base_url, model=workload["model"]) as client:
        check = client.http.get(args.base_url.removesuffix("/v1") + "/server_info")
        check.raise_for_status()
        live = check.json()
    keys = ("revision", "version", "tp_size", "max_running_requests", "schedule_policy",
            "disable_radix_cache", "enable_cache_report", "mm_preprocess_cache_size_mb",
            "enable_prefix_mm_cache", "enable_mm_global_cache")
    settings = {k: live[k] for k in keys}
    expected = {**manifest["server"], "max_running_requests": 3}
    if any(settings[k] != expected[k] for k in keys):
        raise ValueError(f"Unexpected live server settings: {settings}")

    run_id = uuid4().hex[:10]
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        clients = {
            s["id"]: stack.enter_context(VLMClient(base_url=args.base_url, model=workload["model"]))
            for s in workload["sessions"]
        }
        session_gate = "session_replicas" in config
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients) * (2 if session_gate else 1)))
        with (args.output / "client.jsonl").open("w", buffering=1) as log:
            for trial in range(config["repeats"] + 1):
                policies = config["policies"]
                for policy in policies[trial:] + policies[:trial]:
                    start = perf_counter()
                    pending = []
                    previous = {}
                    for event in events:
                        delay = start + event["scheduled_ms"] / 1000 - perf_counter()
                        if delay > 0:
                            sleep(delay)
                        future = executor.submit(
                            issue, clients[event["session"]], event, prompts[event["session"], event["turn"]],
                            policy, trial, run_id, workload["max_tokens"], start,
                            previous.get(event["session"]), token_fit,
                        )
                        pending.append(future)
                        if session_gate:
                            previous[event["session"]] = future
                    output = [task.result() for task in pending]
                    if not session_gate and any(a["sent_ms"] > b["sent_ms"] for a, b in zip(output, output[1:])):
                        raise RuntimeError("Client send order differs from the arrival trace")
                    completed = collections.defaultdict(float)
                    for row in output:
                        if row["sent_ms"] < completed[row["session"]]:
                            raise RuntimeError("A session turn arrived before its preceding turn completed")
                        completed[row["session"]] = row["completed_ms"]
                        log.write(json.dumps(row) + "\n")
                    rows.extend(output)
                    print(f"{trial=} {policy=} completed {len(output)} requests", flush=True)
    visuals, stages, batches = read_traces(args, offsets, rows, run_id)
    if any("error" in batch for batch in batches) or any(len(v) != 1 for v in visuals.values()):
        raise ValueError("Invalid processor or CUDA trace")
    output = [
        join(r, visuals[r["request_id"]][0], stages[r["request_id"]], batches,
             allow_expected_hit_miss="session_replicas" in config)
        for r in rows
    ]
    (args.output / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in output))
    (args.output / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "server": settings, "config": config, "arrivals": events,
        "predictor_sha256": digest(predictor_summary),
        **({"workload_sha256": digest(workload_bytes), "split": args.split} if workload_bytes else {}),
        **({"calibrator_sha256": digest(calibrator)} if calibrator else {}),
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/chartqa-schedule-compare.json"))
    parser.add_argument("--reference-trace", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--predictor", type=Path, default=Path("reports/chartqa-prefill-prediction"))
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-trace", type=Path, default=Path("results/schedule-server/vision.jsonl"))
    parser.add_argument("--session-trace", type=Path, default=Path("results/schedule-server/session.jsonl"))
    parser.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    run(parser.parse_args())
