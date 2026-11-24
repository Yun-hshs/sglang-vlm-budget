#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Use the verified local Qwen2.5-VL checkpoint}"
: "${CUDA_VISIBLE_DEVICES:?Select one idle physical GPU explicitly}"
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to a single physical GPU index." >&2
    exit 1
fi
p="${PYTHON_BIN:-python3}"
runtime="${SGLANG_RUNTIME_DIR:-$PWD/.runtime}"
log_dir="${RUN_LOG_DIR:-$PWD/results/server}"
mkdir -p "$log_dir"
"$p" -m deployment.install_trace "$runtime"
if [[ ${SESSION_TRACE:-0} == 1 ]]; then
    "$p" -m deployment.install_session_trace "$runtime"
    export SGLANG_SESSION_TRACE_JSONL="$log_dir/session.jsonl"
fi
if [[ ${SCHEDULE_COMPARE:-0} == 1 ]]; then
    "$p" -m deployment.install_schedule_compare "$runtime"
fi
export PATH="$runtime/bin:$PATH"
export PYTHONPATH="$runtime:$PWD"
export TORCH_DISABLE_NATIVE_JIT=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$log_dir/triton-cache}"
export SGLANG_MM_TRACE_JSONL="$log_dir/vision.jsonl"
export PYTHONUNBUFFERED=1

exec "$p" -m deployment.launch_smoke "${SGLANG_CONFIG:-configs/sglang-smoke.json}" "$MODEL_PATH"
