#!/bin/bash
# Python runtime deps (session + IFEval + IFBench) and NLTK corpora.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

log_step "deps" "python runtime dependencies + nltk corpora"

# pip install is idempotent (already-satisfied packages no-op), so one line does it.
pip install wandb omegaconf pylatexenc langdetect immutabledict nltk emoji syllapy pydantic-settings 2>&1 | tail -3
log_ok "pip deps installed"

# nltk corpora are not pip packages; a CDN failure warns but must not abort.
if python3 -c "import nltk; [nltk.download(r, quiet=True) for r in ('punkt_tab','punkt')]" 2>/dev/null; then
    log_ok "nltk punkt corpora available"
else
    log_warn "nltk corpus download failed (IFEval tokenization may degrade)"
fi
