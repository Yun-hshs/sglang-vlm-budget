#!/usr/bin/env bash
set -euo pipefail
umask 077
case "$1" in
    engine)
        export SESSION_TRACE=1 SCHEDULE_COMPARE=1
        export SCHEDULE_RULES="$PWD/configs/chat-scheduling.json"
        export SGLANG_CONFIG=configs/sglang-chat-server.json SGLANG_VLM_CACHE_SIZE_MB=12
        export RUN_LOG_DIR="$PWD/results/chat-engine"
        exec "$PYTHON_BIN" -m deployment.visual_chat_engine
        ;;
    web)
        export CHAT_PORT=7864 CHAT_AUTH_FILE="$CHAT_PUBLIC_DIR/credentials"
        export CHAT_DB="$CHAT_PUBLIC_DIR/chat.sqlite3"
        export VLM_BASE_URL=http://127.0.0.1:30002/v1
        exec bash scripts/start_visual_chat.sh
        ;;
    ngrok)
        export CHAT_PUBLIC_URL="$(< "$CHAT_PUBLIC_DIR/ngrok-url")"
        until curl -fsS --max-time 3 http://127.0.0.1:30002/health >/dev/null 2>&1 &&
            [[ "$(curl -sS --max-time 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:7864/ 2>/dev/null)" == 401 ]]; do
            sleep 2
        done
        exec bash scripts/start_visual_chat_ngrok.sh
        ;;
    *)
        echo "Unknown component: $1" >&2
        exit 2
        ;;
esac
