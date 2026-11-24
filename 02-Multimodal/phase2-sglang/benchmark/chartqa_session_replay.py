"""Replay identical image sessions with isolated versus reusable radix-cache keys."""
import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier
from time import perf_counter, sleep
from uuid import uuid4

from api.client import VLMClient, user_message
from preprocessing.image_budget import prepare_image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build_prompts(config, fixture_path, tokenizer=None):
    images = {}
    for name, expected in config["fixtures"].items():
        path = fixture_path / name
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Unexpected fixture digest: {name}")
        images[name] = prepare_image(path, "medium")
    prompts = {}
    for session in config["sessions"]:
        history = []
        image_count = 0
        vision_est = 0
        previous = 0
        for index, turn in enumerate(session["turns"], start=1):
            media = [images[name] for name in turn["images"]]
            message = user_message(f"{turn['question']} Answer with only the number.", media)
            history.append(message)
            image_count += len(media)
            prompt = {
                "messages": list(history), "image_count": image_count,
                "new_images": len(media), "kind": turn["kind"], "reference": turn["answer"],
            }
            if tokenizer is not None:
                new_vision = sum(
                    max(1, round(im.metadata["width"] / 28)) * max(1, round(im.metadata["height"] / 28))
                    for im in media
                )
                vision_est += new_vision
                rendered = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                if rendered.count("<|image_pad|>") != image_count:
                    raise ValueError("Local chat template does not match image placeholders")
                input_est = len(tokenizer.encode(rendered, add_special_tokens=False)) - image_count + vision_est
                prompt["features"] = {
                    "input_tokens_est": input_est, "visual_tokens_est": vision_est,
                    "new_visual_tokens_est": new_vision, "previous_prompt_tokens_est": previous,
                    "history_turns": index - 1, "image_count": image_count,
                }
                previous = input_est
            prompts[session["id"], index] = prompt
            history.append({"role": "assistant", "content": turn["answer"]})
    return prompts


def replay(client, prompt, session, number, mode, round_id, run_id, max_tokens, gate):
    rid = f"stage2-mm-session-{run_id}-{round_id}-{mode}-{session}-{number}"
    salt = f"{run_id}:{round_id}:{mode}:{session}"
    if mode == "miss":
        salt += f":{number}"
    if "features" in prompt:
        features = dict(prompt["features"])
        features["prefix_tokens_est"] = features["previous_prompt_tokens_est"] if mode == "hit" else 0
        features["uncached_tokens_est"] = features["input_tokens_est"] - features["prefix_tokens_est"]
        prompt = {**prompt, "features": features}
    content = json.dumps(prompt["messages"], sort_keys=True, ensure_ascii=False).encode()
    gate.wait(timeout=30)
    started = perf_counter()
    result = client.complete(prompt["messages"], max_tokens, rid, cache_salt=salt)
    return {
        **{k: v for k, v in prompt.items() if k != "messages"},
        "request_id": rid, "session": session, "turn": number, "round": round_id,
        "mode": mode, "warmup": round_id == 0, "messages_sha256": digest(content),
        "client_start_seconds": started, "client": result.to_dict(),
    }


def read_from(path, offset):
    if path.is_file():
        with path.open("rb") as handle:
            handle.seek(offset)
            for raw in handle:
                if raw.endswith(b"\n"):
                    yield json.loads(raw)


def read_traces(args, offsets, rows, run_id):
    ids = {row["request_id"] for row in rows}
    for _ in range(40):
        visuals = defaultdict(list)
        for trace in read_from(args.vision_trace, offsets[0]):
            if trace["request_id"] in ids:
                visuals[trace["request_id"]].append(trace)
        stages = defaultdict(dict)
        batches = []
        for trace in read_from(args.session_trace, offsets[1]):
            kind = trace["kind"]
            if kind == "vision_batch":
                if any(rid in ids for rid in trace["active_requests"]):
                    batches.append(trace)
            elif trace["request_id"] in ids:
                rid = trace["request_id"]
                if kind in stages[rid]:
                    raise ValueError(f"Duplicate {kind} trace for {rid}")
                stages[rid][kind] = trace
        if ids == set(visuals) == set(stages) and all(len(v) == 2 for v in stages.values()):
            return visuals, stages, batches
        sleep(0.25)
    raise RuntimeError(f"Incomplete server traces for {run_id}")


def join(row, visual, stages, batches, allow_expected_hit_miss=False):
    rid = row["request_id"]
    proc = stages["processor"]
    server = stages["request"]
    client = row["client"]
    cached = server["cached_tokens"]
    ids = visual["images"]
    if (
        len(ids) != row["image_count"]
        or any(not (im["image_pad_tokens"] == im["offset_tokens"] == im["merged_tokens"]) for im in ids)
        or sum(im["merged_tokens"] for im in ids) != visual["visual_tokens"]
        or visual["visual_tokens"] != visual["prompt_image_pad_tokens"]
        or not (visual["processor_prompt_tokens"] == client["prompt_tokens"] == server["prompt_tokens"])
        or not (isinstance(cached, int) and 0 <= cached < server["prompt_tokens"])
        or server["num_retractions"] != 0
        or (client["cached_tokens"] is not None and client["cached_tokens"] != cached)
    ):
        raise ValueError(f"Unexpected visual/cache trace for {rid}: {server}")
    expected_hit = row["mode"] == "hit" and row["turn"] > 1
    if (cached and not expected_hit) or (expected_hit and not cached and not allow_expected_hit_miss):
        raise ValueError(f"Cache experiment failed for {rid}: cached_tokens={cached}")
    matched_vision = sum(max(0, min(end + 1, cached) - start) for start, end in (im["offset"] for im in ids))
    batch_ids = [batch["batch_id"] for batch in batches if rid in batch["active_requests"]]
    if matched_vision < visual["visual_tokens"] and not batch_ids:
        raise ValueError(f"Missing multimodal batch trace for {rid}")
    stages_s = {
        "api_before_dispatch": server["api_server_dispatch_finish_ts"] - server["request_received_ts"],
        "mm_preprocess": proc["mm_preprocess_s"],
        "scheduler_queue": server["queue_time"],
        "prefill_wall": server["prefill_finished_time"] - server["forward_entry_time"],
        "prefill_to_first_output": (
            server["request_received_ts"] + server["first_token_latency"] - server["prefill_finished_time"]
        ),
        "server_ttft": server["first_token_latency"],
        "decode_and_finish": server["e2e_latency"] - server["first_token_latency"],
    }
    return {
        **row, "runtime": server, "stages_s": stages_s, "visual": visual,
        "cached_visual_prefix_tokens": matched_vision, "vision_batch_ids": batch_ids,
    }


def run(args):
    config = json.loads(args.config.read_text())
    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompts = build_prompts(config, args.fixtures, tokenizer)
    groups = defaultdict(list)
    for session in config["sessions"]:
        groups[session.get("cohort", "default")].append(session)
    for group in groups.values():
        counts = {len(s["turns"]) for s in group}
        if len(counts) != 1 or not 2 <= len(group) <= 4 or not 3 <= next(iter(counts)) <= 5:
            raise ValueError("Each cohort requires 2–4 sessions with the same number of turns (3–5)")
    sessions = [s["id"] for s in config["sessions"]]
    if len(set(sessions)) != len(sessions):
        raise ValueError("Duplicate session IDs")
    offsets = [p.stat().st_size if p.exists() else 0 for p in (args.vision_trace, args.session_trace)]
    run_id = uuid4().hex[:10]
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with ExitStack() as stack:
        clients = {
            session: stack.enter_context(VLMClient(base_url=args.base_url, model=config["model"]))
            for session in sessions
        }
        first_client = clients[sessions[0]]
        response = first_client.http.get(args.base_url.removesuffix("/v1") + "/server_info")
        response.raise_for_status()
        observed = response.json()
        settings = {
            "version": "0.5.23", "served_model_name": config["model"], "tp_size": 1,
            "schedule_policy": "fcfs", "disable_radix_cache": False, "enable_cache_report": True,
            "mm_preprocess_cache_size_mb": 0, "enable_prefix_mm_cache": False,
            "enable_mm_global_cache": False,
        }
        if any(observed.get(k) != v for k, v in settings.items()):
            raise ValueError("Live serving configuration differs from the expected single-GPU replay setup")
        fields = ("revision", "context_length", "attention_backend", "max_running_requests")
        settings.update({k: observed[k] for k in fields})
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=max(map(len, groups.values()))))
        with (args.output / "client.jsonl").open("w", buffering=1) as raw:
            for r in range(config["repeats"] + 1):
                order = ("hit", "miss") if r % 2 == 0 else ("miss", "hit")
                for group in groups.values():
                    for mode, turn in itertools.product(order, range(1, len(group[0]["turns"]) + 1)):
                        gate = Barrier(len(group))
                        tasks = [
                            executor.submit(
                                replay, clients[session], prompts[session, turn], session, turn, mode, r,
                                run_id, config["max_tokens"], gate
                            )
                            for session in (s["id"] for s in group)
                        ]
                        for task in tasks:
                            row = task.result()
                            rows.append(row)
                            raw.write(json.dumps(row) + "\n")
                print(f"Completed paired replay {r}", flush=True)
    visuals, stages, batches = read_traces(args, offsets, rows, run_id)
    if any("error" in batch for batch in batches) or any(len(v) != 1 for v in visuals.values()):
        raise ValueError("Invalid visual batch or processor trace")
    output = [join(row, visuals[row["request_id"]][0], stages[row["request_id"]], batches) for row in rows]
    paired = defaultdict(dict)
    for row in output:
        paired[row["round"], row["session"], row["turn"]][row["mode"]] = row["messages_sha256"]
    if any(set(pair) != {"hit", "miss"} or pair["hit"] != pair["miss"] for pair in paired.values()):
        raise ValueError("Paired inputs differ or are missing")
    for name, records in (("requests.jsonl", output), ("vision_batches.jsonl", batches)):
        (args.output / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    (args.output / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "config": config, "server": settings,
        "signed_input_pairs": len(paired), "requests": len(output), "batch_traces": len(batches),
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, help="Local tokenizer for pre-dispatch estimates")
    parser.add_argument("--config", type=Path, default=Path("configs/chartqa-same-image-session.json"))
    parser.add_argument("--vision-trace", type=Path, default=Path("results/session-server/vision.jsonl"))
    parser.add_argument("--session-trace", type=Path, default=Path("results/session-server/session.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30001/v1")
    run(parser.parse_args())
