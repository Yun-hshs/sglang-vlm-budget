#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET:?Set local JSONL dataset path}"
RUN_DIR="results/run-$(date -u +%Y%m%dT%H%M%SZ)"
python -m benchmark.benchmark_latency --dataset "$DATASET" --output "$RUN_DIR"
python -m analysis.plot_results --input "$RUN_DIR/requests.jsonl" --output "$RUN_DIR/analysis"
