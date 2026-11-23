#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_DISABLE_NATIVE_JIT=1
export GRADIO_TEMP_DIR=${GRADIO_TEMP_DIR:-"$PWD/runs/demo/uploads"}
export PYTHONUNBUFFERED=1
args=(--port "${PORT:-7861}")
model_path=${MODEL_PATH:-checkpoints/Qwen2.5-VL-7B-Instruct}
if [[ -n "${MODEL_PATH:-}" || -d "$model_path" ]]; then args+=(--model-path "$model_path"); fi
if [[ -n "${ADAPTER_PATH:-}" ]]; then args+=(--adapter "$ADAPTER_PATH"); fi
if [[ -n "${ADAPTER_SHA256:-}" ]]; then args+=(--adapter-sha256 "$ADAPTER_SHA256"); fi
if [[ "${SHARE:-0}" == 1 ]]; then args+=(--share); fi
exec "${PYTHON_BIN:-python}" -m vl_assistant.demo "${args[@]}"
