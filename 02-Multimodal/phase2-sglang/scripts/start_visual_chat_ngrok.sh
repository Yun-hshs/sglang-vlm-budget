#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

public_dir="${CHAT_PUBLIC_DIR:-$PWD/results/chat-public}"
ngrok="${NGROK_BIN:-$public_dir/ngrok}"
config="${NGROK_CONFIG:-$public_dir/ngrok.yml}"
origin=http://127.0.0.1:7864
: "${CHAT_PUBLIC_URL:?Set CHAT_PUBLIC_URL to the fixed HTTPS development URL in the ngrok dashboard}"
if [[ ! "$CHAT_PUBLIC_URL" =~ ^https://[a-z0-9]([a-z0-9-]*[a-z0-9])?\.ngrok-free\.(app|dev)/?$ ]]; then
    echo "Expected the HTTPS URL of the account's assigned ngrok-free.app or ngrok-free.dev domain." >&2
    exit 2
fi
if [[ ! -x "$ngrok" || ! -r "$config" ]]; then
    echo "Install the ngrok agent and configure its token in $config (mode 0600)." >&2
    exit 2
fi
status=$(curl -sS --max-time 3 -o /dev/null -w '%{http_code}' "$origin/")
if [[ "$status" != 401 ]]; then
    echo "Expected HTTP 401 from the protected Web origin on port 7864. Got $status." >&2
    exit 3
fi
"$ngrok" config check --config "$config"
exec "$ngrok" http "$origin" --config "$config" \
    --url "${CHAT_PUBLIC_URL%/}" --inspect=false --log=stdout
