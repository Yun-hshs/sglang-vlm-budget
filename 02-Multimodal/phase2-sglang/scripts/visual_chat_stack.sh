#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
export CHAT_STACK_ROOT="$PWD"
export CHAT_PUBLIC_DIR="${CHAT_PUBLIC_DIR:-$PWD/results/chat-public}"
env_file="${CHAT_STACK_ENV:-$CHAT_PUBLIC_DIR/stack.env}"
if [[ -r "$env_file" ]]; then
    source "$env_file"
fi
export CHAT_STACK_RUN="${CHAT_STACK_RUN:-$CHAT_PUBLIC_DIR/supervisor}"
export CHAT_SUPERVISOR_BIN="${CHAT_SUPERVISOR_BIN:-$CHAT_PUBLIC_DIR/supervisor-venv/bin}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ "$PYTHON_BIN" != /* ]]; then
    export PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi
export CHAT_STACK_CONFIG="$PWD/configs/visual-chat-supervisor.ini"
ctl=("$CHAT_SUPERVISOR_BIN/supervisorctl" -c "$CHAT_STACK_CONFIG")
supervisor=("$CHAT_SUPERVISOR_BIN/supervisord" -c "$CHAT_STACK_CONFIG")

case "${1:-}" in
    start|run)
        : "${MODEL_PATH:?Set MODEL_PATH to the local Qwen2.5-VL checkpoint}"
        : "${CUDA_VISIBLE_DEVICES:?Select one physical GPU}"
        test -s "$CHAT_PUBLIC_DIR/credentials"
        test -s "${NGROK_CONFIG:-$CHAT_PUBLIC_DIR/ngrok.yml}"
        test -s "$CHAT_PUBLIC_DIR/ngrok-url"
        mkdir -p "$CHAT_STACK_RUN"
        if [[ $1 == run ]]; then
            exec "${supervisor[@]}" -n
        fi
        if ! "${ctl[@]}" pid >/dev/null 2>&1; then
            exec "${supervisor[@]}"
        fi
        ;;
    ctl)
        shift
        exec "${ctl[@]}" "$@"
        ;;
    *)
        echo "Usage: $0 {start|run|ctl COMMAND...}" >&2
        exit 2
        ;;
esac
