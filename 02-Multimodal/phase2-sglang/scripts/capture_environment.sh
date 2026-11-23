#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/environment
python -m pip freeze > results/environment/pip-freeze.txt
uname -a > results/environment/platform.txt
if command -v nvidia-smi >/dev/null; then nvidia-smi -q > results/environment/nvidia-smi.txt; fi
git rev-parse HEAD > results/environment/git-revision.txt
