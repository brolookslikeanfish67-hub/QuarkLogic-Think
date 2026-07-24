#!/bin/bash
# Megatron-LM FarSkip patches (training side): force-reset the in-repo submodule
# to its pin for a clean base, overlay patches, install megatron-core editable.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

log_step "megatron" "FarSkip patches on ${MEGATRON_SUBMODULE}"

require_dir "${PRIMUS_ROOT}"
require_file "${MEGATRON_PATCHES}/megatron/core/transformer/transformer_layer.py"

# When REPO_ROOT is a git checkout, reset the submodule to its pin for a clean
# base. On a plain file-sync deploy (no .git) the Megatron tree is shipped
# pre-checked-out at the pin, so skip the git reset instead of failing.
if git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" submodule update --init --force "${MEGATRON_SUBMODULE}"
    log_ok "submodule at pin $(git -C "${MEGATRON_SUBMODULE}" rev-parse --short HEAD)"
else
    require_file "${MEGATRON_SUBMODULE}/megatron/core/__init__.py"
    log_warn "no git repo at ${REPO_ROOT}; using pre-shipped Megatron tree as-is"
fi

cp -r "${MEGATRON_PATCHES}/." "${MEGATRON_SUBMODULE}/"
log_ok "FarSkip patches overlaid"

pip install -e "${MEGATRON_SUBMODULE}" 2>&1 | tail -3
log_ok "megatron-core installed ($(pip show megatron-core 2>/dev/null | grep Version))"
