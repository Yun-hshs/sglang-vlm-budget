#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=${PYTHON_BIN:-python}
model_path=${MODEL_PATH:-}
run_dir=${RUN_DIR:-runs/baseline-smoke}
data_dir=${DATA_DIR:-data/processed/smoke}
case_dir=${CASE_DIR:-data/cases/smoke}

# An explicit single card is required even on a multi-GPU server.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# PyTorch 2.13 can JIT a Triton rotary matmul and require Python.h.
export TORCH_DISABLE_NATIVE_JIT=1
model_args=()
if [[ -n "$model_path" ]]; then
    model_args=(--model-path "$model_path")
fi
if [[ -e "$run_dir/chartqa" || -e "$run_dir/media" ]]; then
    echo "Output already exists. Set RUN_DIR to a fresh directory." >&2
    exit 1
fi
"$python_bin" -m vl_assistant.infer "${model_args[@]}" \
    --manifest "$data_dir/test.jsonl" --output "$run_dir/chartqa"
"$python_bin" -m vl_assistant.infer "${model_args[@]}" \
    --manifest "$case_dir/cases.jsonl" --output "$run_dir/media" --cases
