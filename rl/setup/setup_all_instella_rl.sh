#!/bin/bash
# RL setup orchestrator: applies all FarSkip patches (training + inference side).
# Run on every Ray node (the SGLang overlay edits the container-local sglang).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/common.sh"

bash "${HERE}/aiter_baton_clean.sh"
bash "${HERE}/install_python_deps.sh"

# --- SGLang overlay (inference side) ---
log_step "sglang" "overlay FarSkip files onto ${SGLANG_PYTHON}"
bash "${REPO_ROOT}/inference/scripts/update_sglang_workspace_files.sh" \
    "${REPO_ROOT}/inference/sglang/python" "${SGLANG_PYTHON}"
# Drop stale compiled kernels so overlaid triton block sizes take effect.
rm -rf /root/.triton/cache/* /tmp/torchinductor_root/triton/* 2>/dev/null || true

# --- Training side ---
bash "${HERE}/primus_instella_patch.sh"
bash "${HERE}/mbridge_gate_proj_patch.sh"
bash "${HERE}/miles_compat_patch.sh"
bash "${HERE}/torch_memory_saver.sh"

python3 "${REPO_ROOT}/inference/scripts/verify_sglang_overlay.py"
python3 "${HERE}/verify_patch.py"

log_step "done" "RL setup complete"
