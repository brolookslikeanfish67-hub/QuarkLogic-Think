#!/bin/bash
# Miles ROCm/API compat shims (rl/miles submodule): honor HIP_VISIBLE_DEVICES and
# default missing sglang_*_parallel_size args to 1. Idempotent; fails loud on drift.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

log_step "miles" "compatibility patches on ${MILES_DIR}"

python3 - "${MILES_DIR}" <<'PY'
import pathlib
import sys

miles = pathlib.Path(sys.argv[1])

# relative file -> list of (anchor, replacement)
PATCHES = {
    "miles/backends/sglang_utils/sglang_engine.py": [
        ('cvd = os.environ.get("CUDA_VISIBLE_DEVICES")',
         'cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")'),
    ],
    "miles/backends/sglang_utils/arguments.py": [
        ('args.sglang_dp_size = args.sglang_data_parallel_size',
         'args.sglang_dp_size = getattr(args, "sglang_data_parallel_size", 1) or 1'),
        ('args.sglang_pp_size = args.sglang_pipeline_parallel_size',
         'args.sglang_pp_size = getattr(args, "sglang_pipeline_parallel_size", 1) or 1'),
        ('args.sglang_ep_size = args.sglang_expert_parallel_size',
         'args.sglang_ep_size = getattr(args, "sglang_expert_parallel_size", 1) or 1'),
    ],
}

for rel, repls in PATCHES.items():
    path = miles / rel
    if not path.is_file():
        sys.exit(f"  x missing file: {path}")
    src = path.read_text()
    for old, new in repls:
        if new in src:          # already patched -> idempotent
            continue
        if old not in src:      # neither form present -> anchor drifted, fail loud
            sys.exit(f"  x anchor not found in {rel}: {old!r}")
        src = src.replace(old, new, 1)
    path.write_text(src)
    print(f"  v {rel}: applied/verified")
PY
