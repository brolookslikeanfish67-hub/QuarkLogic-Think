#!/bin/bash
# Clean poisoned aiter JIT build state (ROCm): orphaned batons (hang at graph
# capture) and incomplete module dirs with no .so (scheduler dies on launch).
# Complete modules are left alone. Skipped if a live sglang/train process runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

log_step "aiter" "clean stale JIT build state"

AITER_BUILD_DIR="${AITER_BUILD_DIR:-/app/aiter/aiter/jit/build}"
if [ ! -d "${AITER_BUILD_DIR}" ]; then
    log_ok "no aiter build dir at ${AITER_BUILD_DIR}; nothing to clean"
    exit 0
fi
if pgrep -f 'train_async\.py|MegatronTrainRayActor|RolloutManager|SGLangEngine|sglang.*[Ss]cheduler' >/dev/null 2>&1; then
    log_warn "live sglang/train process on this node — SKIPPING aiter cleanup (safety)"
    exit 0
fi

cleaned=0
# orphaned baton locks
while IFS= read -r lk; do
    [ -n "${lk}" ] || continue
    rm -f "${lk}" && log_ok "removed orphaned baton: ${lk}" && cleaned=$((cleaned + 1))
done < <(find "${AITER_BUILD_DIR}" -maxdepth 1 -name 'lock_module_*' 2>/dev/null)
# incomplete module dirs (no compiled .so)
for d in "${AITER_BUILD_DIR}"/module_*; do
    [ -d "${d}" ] || continue
    if [ -z "$(find "${d}" -name '*.so' -print -quit 2>/dev/null)" ]; then
        rm -rf "${d}" && log_ok "removed incomplete module (no .so): ${d}" && cleaned=$((cleaned + 1))
    fi
done
if [ "${cleaned}" -eq 0 ]; then
    log_ok "aiter JIT cache clean (no stale batons / incomplete modules)"
fi
