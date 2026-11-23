#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
cd data/raw
base=https://huggingface.co/datasets/HuggingFaceM4/ChartQA/resolve/b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5/data
while read -r sha file; do
    if [[ ! -f "$file" ]]; then
        curl -fsSL --retry 2 "$base/$file" -o "$file.part"
        printf '%s  %s\n' "$sha" "$file.part" | sha256sum --check
        mv "$file.part" "$file"
    else
        printf '%s  %s\n' "$sha" "$file" | sha256sum --check
    fi
done <<'SOURCES'
169979f8a8c64ba93ac1be22916bf9b49c9c2e4c57f5c0104e6f671f2aacd83a train-00000-of-00003-49492f364babfa44.parquet
de58d649aaac8343e6a3fa5ab362a0458d5b5374f4112a95fea5d1d9253dd86e val-00000-of-00001-0f11003c77497969.parquet
efcf31333368c485eb569ec7bd4cc3694bd0e8e1a06c503124b51d27c924491d test-00000-of-00001-e2cd0b7a0f9eb20d.parquet
SOURCES
