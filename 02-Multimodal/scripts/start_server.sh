#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_REVISION:?Set immutable model commit SHA}"
: "${SGLANG_VERSION:?Set exact installed SGLang version}"
python - "$SGLANG_VERSION" <<'CHECK'
import importlib.metadata,sys
actual=importlib.metadata.version('sglang')
if actual != sys.argv[1]:raise SystemExit(f'SGLang mismatch: {actual}')
CHECK
mkdir -p results/server
python -m pip freeze > results/server/environment.txt
nvidia-smi -q > results/server/gpu.txt
python -m sglang.launch_server \
 --model-path "${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" \
 --revision "$MODEL_REVISION" --dtype bfloat16 \
 --host 127.0.0.1 --port "${PORT:-30000}" \
 "${@}" 2>&1 | tee results/server/server.log
