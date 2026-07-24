#!/bin/bash
# Overlay FarSkip's vendored SGLang files onto a container's sglang workspace.
# Copies every file under <SRC> into <DEST>, preserving layout.
# Usage: update_sglang_workspace_files.sh <SRC_DIR> <DEST_DIR>
#
# Invariant: each vendored file must overwrite an existing base file in <DEST>
# (the point of an overlay is to patch files sglang already imports). A file with
# no matching base is a hard error -- it usually means it is landing in the wrong
# path -- unless it is an intentional addition: a file in ALLOW_NEW (new arch /
# new bench tool) or any *_original.py (inert pre-patch diff reference).
set -euo pipefail

SRC=${1:?usage: update_sglang_workspace_files.sh <SRC_DIR> <DEST_DIR>}
DEST=${2:?usage: update_sglang_workspace_files.sh <SRC_DIR> <DEST_DIR>}
[ -d "$SRC" ]  || { echo "ERROR: source dir not found: $SRC"; exit 1; }
[ -d "$DEST" ] || { echo "ERROR: dest dir not found:   $DEST"; exit 1; }

# Vendored files that legitimately have no stock base in <DEST>.
ALLOW_NEW=(
    "sglang/srt/models/instella_moe.py"   # new InstellaMoEForCausalLM arch
    "sglang/multi_bench_one_batch.py"     # new AMD multi-batch bench tool
)

is_allowed_new() {
    local rel="$1" a
    [[ "$rel" == *_original.py ]] && return 0
    for a in "${ALLOW_NEW[@]}"; do [ "$rel" = "$a" ] && return 0; done
    return 1
}

count=0
orphans=()
while IFS= read -r -d '' f; do
    rel="${f#"$SRC"/}"
    if [ ! -e "$DEST/$rel" ] && ! is_allowed_new "$rel"; then
        orphans+=("$rel")
        continue
    fi
    mkdir -p "$DEST/$(dirname "$rel")"
    cp "$f" "$DEST/$rel"
    count=$((count + 1))
done < <(find "$SRC" -type f -not -path '*/__pycache__/*' -not -name '*.pyc' -print0)

if [ "${#orphans[@]}" -gt 0 ]; then
    echo "ERROR: ${#orphans[@]} vendored file(s) have no matching base in $DEST"
    echo "       (wrong path, or renamed upstream? add to ALLOW_NEW if intentional):"
    printf '    - %s\n' "${orphans[@]}"
    exit 1
fi

echo "Overlaid $count files: $SRC -> $DEST"
