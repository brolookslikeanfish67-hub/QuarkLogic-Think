#!/bin/bash
# torch_memory_saver (ROCm branch) build. Provides the hook-mode preload .so
# required by the RL weight-update broadcast — without it the broadcast hangs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TMS_DIR="${TMS_DIR:-/tmp/torch_memory_saver}"
TMS_SO="${TMS_DIR}/torch_memory_saver_hook_mode_preload.abi3.so"

log_step "tms" "torch_memory_saver at ${TMS_DIR}"

if [ -f "${TMS_SO}" ]; then
    log_ok "already built"
    exit 0
fi

rm -rf "${TMS_DIR}"
git clone https://github.com/zyzshishui/torch_memory_saver.git "${TMS_DIR}"
git -C "${TMS_DIR}" checkout -b rocm origin/rocm
pip install -e "${TMS_DIR}"
(cd "${TMS_DIR}" && python3 setup.py build_ext --inplace)

if [ ! -f "${TMS_SO}" ]; then
    log_err "build finished but ${TMS_SO} missing (RL broadcast will hang without it)"
    exit 1
fi
log_ok "built ${TMS_SO}"
