#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/fetch_chartqa_smoke.sh
cd data/raw
base=https://huggingface.co/datasets/HuggingFaceM4/ChartQA/resolve/b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5/data
while read -r sha file; do
    if [[ ! -f "$file" ]]; then
        curl -fsSL --retry 2 "$base/$file" -o "$file.part"
        printf '%s  %s\n' "$sha" "$file.part" | sha256sum --check
        mv "$file.part" "$file"
    fi
    printf '%s  %s\n' "$sha" "$file" | sha256sum --check
done <<'SOURCES'
a8819291729c807f163b93dcb2dc1fe60044d56412a0c0626cb7312df7392140 train-00001-of-00003-7302bae5e425bbc7.parquet
528133aec54c7e948f40b8af56b083b98f9b7d1fdcdd1cd9e375f3df984f7ae4 train-00002-of-00003-194c9400785577a2.parquet
SOURCES
