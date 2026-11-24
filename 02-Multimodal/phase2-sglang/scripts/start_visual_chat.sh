#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen2.5-VL-7B-Instruct checkpoint}"
export PYTHONPATH="${SGLANG_RUNTIME_DIR:-$PWD/.runtime}:$PWD"
exec "${PYTHON_BIN:-python3}" -m uvicorn frontend.chat_app:app --host 127.0.0.1 --port "${CHAT_PORT:-7863}"
