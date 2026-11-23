#!/usr/bin/env bash
set -euo pipefail
destination=${1:?Usage: fetch_modelscope.sh CHECKPOINT_DIR}
mkdir -p "$destination"
config_dir="$(cd "$(dirname "$0")/../configs" && pwd)"
base=https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct/resolve/master
cd "$destination"
for config in config.json chat_template.json generation_config.json merges.txt model.safetensors.index.json preprocessor_config.json tokenizer.json tokenizer_config.json vocab.json; do
    if [[ ! -s "$config" ]]; then
        curl -fsSL --retry 3 --connect-timeout 20 "$base/$config" -o "$config.part"
        mv "$config.part" "$config"
    fi
done
python3 - "$config_dir/model-config-blobs.tsv" <<'VERIFY'
import hashlib
from pathlib import Path
import sys

for line in Path(sys.argv[1]).read_text().splitlines():
    expected, name = line.split("\t")
    contents = Path(name).read_bytes()
    digest = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
    if digest != expected:
        raise SystemExit(f"{name}: differs from pinned Hugging Face revision")
    print(f"{name}: OK")
VERIFY
download() {
    local name="$1" sha="$2"
    if [[ ! -f "$name" ]]; then
        curl -fsSL --retry 4 --connect-timeout 20 -C - "$base/$name" -o "$name.part"
        printf '%s  %s\n' "$sha" "$name.part" | sha256sum --check
        mv "$name.part" "$name"
    else
        printf '%s  %s\n' "$sha" "$name" | sha256sum --check
    fi
}
export base
export -f download
python3 -c 'import json,sys; [print(x["file"],x["sha256"]) for x in json.load(open(sys.argv[1]))]' "$config_dir/model-shards.json" |
    xargs -n2 -P5 bash -c 'set -eo pipefail; download "$@"' _
