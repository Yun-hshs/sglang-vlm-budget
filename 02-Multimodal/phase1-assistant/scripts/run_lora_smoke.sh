#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin=${PYTHON_BIN:-python}
data_dir=${DATA_DIR:-data/processed/smoke}
run_dir=${RUN_DIR:-runs/lora-smoke}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_DISABLE_NATIVE_JIT=1
model_args=()
if [[ -n "${MODEL_PATH:-}" ]]; then
    model_args=(--model-path "$MODEL_PATH")
fi
if [[ -e "$run_dir" ]]; then
    echo "Output exists. Set RUN_DIR to a fresh directory." >&2
    exit 1
fi
"$python_bin" -m vl_assistant.train_lora "${model_args[@]}" \
    --manifest "$data_dir/train.jsonl" --output "$run_dir/train"
"$python_bin" -m vl_assistant.infer "${model_args[@]}" \
    --manifest "$data_dir/test.jsonl" --output "$run_dir/base"
"$python_bin" -m vl_assistant.infer "${model_args[@]}" \
    --manifest "$data_dir/test.jsonl" --output "$run_dir/lora" --adapter "$run_dir/train/adapter"
"$python_bin" -m vl_assistant.compare "$run_dir/base" "$run_dir/lora" "$run_dir/comparison.json"
