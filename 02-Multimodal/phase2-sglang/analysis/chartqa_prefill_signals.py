"""Calibrate pre-dispatch service-time scores and evaluate pairs coexisting in a real queue."""
import argparse
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.chartqa_prefill_prediction import errors, fit
from benchmark.chartqa_session_replay import digest
from deployment.prefill_queue_features import FEATURES, predict


def read_jsonl(path):
    handle = gzip.open(path, "rt") if path.suffix == ".gz" else path.open()
    with handle:
        yield from map(json.loads, handle)


def load_probe(path, workload_sha, split):
    manifest = json.loads((path / "manifest.json").read_text())
    if (manifest["workload_sha256"], manifest["split"]) != (workload_sha, split):
        raise ValueError(f"Probe does not match workload and split: {path}")
    source = path / "requests.jsonl"
    if not source.is_file():
        source = source.with_suffix(".jsonl.gz")
    return {row["request_id"]: row for row in read_jsonl(source) if not row["warmup"]}


def feature_summary(rows):
    columns = FEATURES + ("native",)
    return {
        key: {"nonzero": sum(row["features"][key] != 0 for row in rows.values()),
              "max": max(row["features"][key] for row in rows.values())}
        for key in columns
    }


def analyze(workload_path, train_path, valid_path, trace):
    raw = workload_path.read_bytes()
    workload = json.loads(raw)
    charts = {
        split: {workload["fixtures"][image]
                for session in workload["sessions"] if session["split"] == split
                for turn in session["turns"] for image in turn["images"]}
        for split in ("train", "validation")
    }
    if charts["train"] & charts["validation"]:
        raise ValueError("Calibration and validation share charts")
    train = load_probe(train_path, digest(raw), "train")
    valid = load_probe(valid_path, digest(raw), "validation")
    sessions = [{row["session"] for row in rows.values()} for rows in (train, valid)]
    if not train or not valid or sessions[0] & sessions[1]:
        raise ValueError("Calibration and validation must have separate complete sessions")
    for path, rows in ((train_path, train), (valid_path, valid)):
        manifest = json.loads((path / "manifest.json").read_text())
        expected = {(e["session"], e["turn"], trial)
                    for e in manifest["arrivals"] for trial in range(1, manifest["config"]["repeats"] + 1)}
        actual = {(r["session"], r["turn"], r["trial"]) for r in rows.values()}
        if expected != actual or len(actual) != len(rows):
            raise ValueError(f"Missing or repeated measured turns: {path}")
    requests = train | valid
    queue_snapshots = []
    for snap in read_jsonl(trace):
        cohort = {}
        for rid, features in snap["features"].items():
            if rid not in requests:
                continue
            values = {**features, "native": snap["native"][rid]}
            requests[rid].setdefault("features", values)
            if rid in valid:
                cohort[rid] = values
        if len(cohort) >= 2:
            queue_snapshots.append(cohort)
    if any("features" not in row for row in requests.values()):
        raise ValueError("First scheduler snapshot is missing for a measured request")

    protocol = workload["prediction_protocol"]
    target = np.array([row["stages_s"]["prefill_wall"] * 1000 for row in valid.values()])
    parameters = {
        name: fit(list(train.values()), list(valid.values()), columns, protocol["ridge_alpha"])[1]
        for name, columns in (("tokens", ("uncached",)), ("local", FEATURES), ("spf", ("native",)))
    }
    if parameters["tokens"]["standardized_weights"][0] <= 0:
        raise ValueError("Token calibration must preserve ascending token-count order")

    def scores(rid, features, ranking=False):
        return {
            "tokens": features["uncached"] if ranking else predict(features, parameters["tokens"], floor=None),
            "ridge": valid[rid]["cost_ms"],
            "local": predict(features, parameters["local"]),
            "spf": features["native"] if ranking else predict(features, parameters["spf"]),
        }

    prediction_rows = [
        {"request_id": rid, "prefill_wall_ms": float(actual), "features": row["features"],
         "predictions_ms": scores(rid, row["features"])}
        for (rid, row), actual in zip(valid.items(), target, strict=True)
    ]
    gaps = dict(zip(valid, target, strict=True))
    seen = set()
    points = Counter()
    for queue in queue_snapshots:
        for i, j in itertools.combinations(sorted(queue), 2):
            truth = gaps[i] - gaps[j]
            if (i, j) in seen or abs(truth) <= protocol["ranking_gap_ms"]:
                continue
            seen.add((i, j))
            lhs, rhs = scores(i, queue[i], ranking=True), scores(j, queue[j], ranking=True)
            for name, value in lhs.items():
                gap = value - rhs[name]
                points[name] += 0.5 if abs(gap) < 1e-9 else float(truth * gap > 0)
    if not seen:
        raise ValueError("No comparable pairs in shared queues")

    result = {
        "workload_sha256": digest(raw),
        "calibration_requests": len(train), "validation_requests": len(valid),
        "calibration_sessions": sorted(sessions[0]), "validation_sessions": sorted(sessions[1]),
        "images_by_split": {split: len(values) for split, values in charts.items()},
        "features": {"train": feature_summary(train), "validation": feature_summary(valid)},
        "validation_actual_prefix_hits": sum(row["runtime"]["cached_tokens"] > 0 for row in valid.values()),
        "shared_queue_snapshots": len(queue_snapshots), "pair_min_gap_ms": protocol["ranking_gap_ms"],
        "models": {},
    }
    for name in prediction_rows[0]["predictions_ms"]:
        estimates = np.array([row["predictions_ms"][name] for row in prediction_rows])
        result["models"][name] = {
            **errors(target, estimates),
            "same_queue_rank": {"pairs": len(seen), "concordance": points[name] / len(seen)},
            **({"fit": parameters[name]} if name in parameters else {}),
        }
    return result, prediction_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("workload", "train", "validation", "queue-trace", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    summary, predictions = analyze(args.workload, args.train, args.validation, args.queue_trace)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    args.output.with_suffix(".predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
