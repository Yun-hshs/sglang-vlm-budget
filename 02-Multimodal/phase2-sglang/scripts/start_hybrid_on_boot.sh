#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077

public_dir="${CHAT_PUBLIC_DIR:-$PWD/results/chat-public}"
test -r "${CHAT_STACK_ENV:-$public_dir/stack.env}"
test -x "${CHAT_SUPERVISOR_BIN:-$public_dir/supervisor-venv/bin}/supervisord"

# Return after handing off to Supervisor so the platform hook does not wait for model warm-up.
exec bash scripts/visual_chat_stack.sh start
