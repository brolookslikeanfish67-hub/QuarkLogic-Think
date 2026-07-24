# Shared config + helpers for the rl/setup/* patch scripts.
# Sourced, never executed directly. Do not `set -e` here (it would leak into
# whatever sources this); each step script sets its own `set -euo pipefail`.

# ---- Repo root (works regardless of caller cwd) ----
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "${_COMMON_DIR}" rev-parse --show-toplevel 2>/dev/null || (cd "${_COMMON_DIR}/../.." && pwd))}"

# ---- Default paths (all env-overridable) ----
# Training side: `training/` is the vendored Primus-Instella (farskip branch),
# with Megatron-LM as an in-repo submodule pinned to the commit the FarSkip
# patch tree was authored against. We use that pin as-is — no external checkout.
PRIMUS_ROOT="${PRIMUS_ROOT:-${REPO_ROOT}/training}"
MEGATRON_SUBMODULE="${MEGATRON_SUBMODULE:-${PRIMUS_ROOT}/third_party/Megatron-LM}"
MEGATRON_PATCHES="${MEGATRON_PATCHES:-${PRIMUS_ROOT}/patches/megatron-lm}"

# Inference side (container-local sglang + external FarSkip overlay sources).
MILES_DIR="${MILES_DIR:-${REPO_ROOT}/rl/miles}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/app/sglang/python}"

export REPO_ROOT PRIMUS_ROOT MEGATRON_SUBMODULE MEGATRON_PATCHES MILES_DIR SGLANG_PYTHON

# ---- Logging helpers ----
log_info() { printf '  %s\n' "$*"; }
log_ok()   { printf '  \342\234\223 %s\n' "$*"; }
log_warn() { printf '  \342\232\240 %s\n' "$*"; }
log_err()  { printf '  \342\234\227 %s\n' "$*" >&2; }
log_step() { printf '\n[%s] %s\n' "${1}" "${2}"; }

# ---- Assertions ----
require_file() { [ -f "$1" ] || { log_err "missing file: $1"; return 1; }; }
require_dir()  { [ -d "$1" ] || { log_err "missing dir:  $1"; return 1; }; }
