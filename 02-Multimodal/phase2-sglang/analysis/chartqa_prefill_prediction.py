"""Fit pre-dispatch Prefill predictors and evaluate held-out session rankings."""
import argparse
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


INPUT = ("input_tokens_est",)
VISUAL_HISTORY = INPUT + (
    "visual_tokens_est", "new_visual_tokens_est", "image_count", "history_turns",
)
FEATURES = {
    "input_only": INPUT,
    "visual_history": VISUAL_HISTORY,
    "visual_history_cache": VISUAL_HISTORY + ("prefix_tokens_est",),
}


def rank(values):
    ordered = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        result[ordered[start:end]] = (start + end - 1) / 2
        start = end
    return result


def concordance(target, prediction, pairs, minimum_gap):
    n = 0
    points = 0.0
    for i, j in pairs:
        gap = target[i] - target[j]
        if abs(gap) <= minimum_gap:
            continue
        estimated = prediction[i] - prediction[j]
        n += 1
        points += 0.5 if abs(estimated) < 1e-9 else float(estimated * gap > 0)
    return {"pairs": n, "concordance": points / n if n else None}


def errors(target, prediction):
    residual = prediction - target
    return {
        "n": len(target), "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual ** 2))),
        "p90_abs_error_ms": float(np.quantile(np.abs(residual), 0.9)),
    }


def fit(train, valid, names, alpha):
    x = np.array([[r["features"][key] for key in names] for r in train])
    x_val = np.array([[r["features"][key] for key in names] for r in valid])
    y = np.array([r["stages_s"]["prefill_wall"] * 1000 for r in train])
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1)
    design = (x - center) / scale
    weights = np.linalg.solve(design.T @ design + alpha * np.eye(len(names)), design.T @ (y - y.mean()))
    prediction = np.maximum(0, (x_val - center) / scale @ weights + y.mean())
    return prediction, {
        "features": list(names), "centers": center.tolist(), "scales": scale.tolist(),
        "standardized_weights": weights.tolist(), "intercept_ms": float(y.mean()),
    }


def run(path):
    manifest = json.loads((path / "manifest.json").read_text())
    workload = manifest["config"]
    config = workload["prediction_protocol"]
    sessions = {s["id"]: s for s in workload["sessions"]}
    split_images = defaultdict(set)
    for session in sessions.values():
        split_images[session["split"]].update(
            workload["fixtures"][name] for turn in session["turns"] for name in turn["images"]
        )
    if split_images["train"] & split_images["validation"]:
        raise ValueError("Training and validation images overlap")

    requests = path / "requests.jsonl"
    handle = requests.open() if requests.is_file() else gzip.open(requests.with_suffix(".jsonl.gz"), "rt")
    with handle:
        rows = [r for line in handle if not (r := json.loads(line))["warmup"]]
    train, valid = ([r for r in rows if sessions[r["session"]]["split"] == split]
                    for split in ("train", "validation"))
    if not train or not valid or len(train) + len(valid) != len(rows):
        raise ValueError("Incomplete session split")
    target = np.array([r["stages_s"]["prefill_wall"] * 1000 for r in valid])
    actual_hit = np.array([bool(r["runtime"]["cached_tokens"]) for r in valid])
    cohorts = defaultdict(list)
    for i, row in enumerate(valid):
        cohorts[sessions[row["session"]]["cohort"], row["round"], row["mode"], row["turn"]].append(i)
    models = {}
    prediction_rows = {r["request_id"]: {"request_id": r["request_id"], "actual_prefill_ms": float(t)}
                       for r, t in zip(valid, target, strict=True)}
    for name, names in FEATURES.items():
        prediction, parameters = fit(train, valid, names, config["ridge_alpha"])
        models[name] = {
            **errors(target, prediction), "fit": parameters,
            "by_actual_cache": {
                "hit": errors(target[actual_hit], prediction[actual_hit]),
                "miss": errors(target[~actual_hit], prediction[~actual_hit]),
            },
            "spearman": float(np.corrcoef(rank(target), rank(prediction))[0, 1]),
            "all_pairs": concordance(target, prediction, itertools.combinations(range(len(valid)), 2),
                                     config["ranking_gap_ms"]),
            "concurrent_pairs": concordance(target, prediction,
                                            (pair for members in cohorts.values() for pair in itertools.combinations(members, 2)),
                                            config["ranking_gap_ms"]),
        }
        for row, estimate in zip(valid, prediction, strict=True):
            prediction_rows[row["request_id"]][name] = float(estimate)
    result = {
        "train_sessions": sorted({r["session"] for r in train}),
        "validation_sessions": sorted({r["session"] for r in valid}),
        "train_requests": len(train), "validation_requests": len(valid),
        "images_by_split": {split: len(images) for split, images in split_images.items()},
        "estimator_checks_validation": {
            f"{feature}_mae": float(np.mean([abs(r["features"][feature] - actual(r)) for r in valid]))
            for feature, actual in (
                ("input_tokens_est", lambda r: r["runtime"]["prompt_tokens"]),
                ("visual_tokens_est", lambda r: r["visual"]["visual_tokens"]),
                ("prefix_tokens_est", lambda r: r["runtime"]["cached_tokens"]),
            )
        },
        "models": models,
    }
    (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prediction_rows.values()))
    (path / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    run(parser.parse_args().run)
