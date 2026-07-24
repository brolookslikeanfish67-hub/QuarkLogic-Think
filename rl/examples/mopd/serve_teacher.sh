#!/bin/bash
# Serve ONE Instella-MoE teacher for OPD behind a stable
# endpoint. Entrypoint for a long-lived serving workload (e.g. a k8s Deployment on
# a ClusterIP Service): installs the FarSkip SGLang overlay (idempotent), then
# execs an SGLang server exposing /generate with return_logprob (used by
# two_teacher_reward.py). Deploy two — the IF-RL teacher and the DPO/SFT anchor —
# and point the OPD job at each via IF_TEACHER_HOST / GENERAL_TEACHER_HOST.
#
# Env:
#   MODEL            (required) HF/sglang model dir (gated-attention Instella)
#   PORT             server/service port (default 8000)
#   TP / EP          tensor / expert parallel (default 8 / 8 — full node)
#   DP               dp-attention size (default 1 = OFF; DP>1 deadlocks the
#                    prefill-only scoring path — see the inline note below)
#   MEM_FRACTION     sglang static mem fraction (default 0.85)
#   CONTEXT_LENGTH   max context (default 32768 = model's real length)
#   SERVED_NAME      OpenAI served-model-name (default "instella")
#   SGLANG_PYTHON    container sglang root (default /app/sglang/python)
#   SKIP_SETUP       1 to skip the FarSkip overlay install (default 0)
#   FARSKIP_SETUP    overlay installer (default rl/setup/setup_all_instella_rl.sh)
#
# k8s liveness probe (set in the Deployment spec, NOT here): probe /health with
# timeoutSeconds>=10, NOT the kubelet default of 1s. sglang's health handler has a
# hard >=1s sleep floor, so a 1s-timeout probe fails fleet-wide under a scoring
# burst -> synchronized teacher restarts. SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
# (set below) makes /health instant; use /health_generate (timeoutSeconds>=30) only
# if you also want real scheduler-hang detection.
set -ex

: "${MODEL:?MODEL is required (path to the Instella HF/sglang teacher dir)}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
EP="${EP:-8}"
DP="${DP:-1}"   # dp-attention OFF by default; prefill-only scoring deadlocks with DP>1
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"   # model's real length; 36864 over-extends (latent OOB)
# Burst-taming knobs: bound concurrent prefill so the aiter MoE kernels
# don't OOB ("Memory access fault by GPU ... Reason: Unknown") under a
# synchronized rollout scoring burst.
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
SERVED_NAME="${SERVED_NAME:-instella}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/app/sglang/python}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
FARSKIP_SETUP="${FARSKIP_SETUP:-${SCRIPT_DIR}/../../setup/setup_all_instella_rl.sh}"

# ---- FarSkip / SGLang runtime env (required for the gated-attention arch) ----
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export ORIG_MAX_POS_EMB=4096
export FARSKIP_REFERENCE_DECODER_LAYER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_HEALTH_CHECK_TIMEOUT=600
# Make GET /health cheap (event-loop-alive only, no generation) so k8s liveness
# probes return instantly under a scoring burst — see the k8s note in the header.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export PYTHONPATH="${SGLANG_PYTHON}:${PYTHONPATH:-}"

# ---- Step 1: install the FarSkip overlay if missing (idempotent) -----------
DSV2="${SGLANG_PYTHON}/sglang/srt/models/deepseek_v2.py"
overlay_present () { grep -q "DeepseekV2ReferenceFarSkipDecoderLayer" "${DSV2}" 2>/dev/null; }
if [ "${SKIP_SETUP}" != "1" ] && ! overlay_present; then
    echo "[serve_teacher] FarSkip overlay missing -> running ${FARSKIP_SETUP}"
    bash "${FARSKIP_SETUP}"
fi
overlay_present || { echo "[serve_teacher] ERROR: FarSkip overlay still missing in ${SGLANG_PYTHON}" >&2; exit 1; }
echo "[serve_teacher] FarSkip overlay present; serving ${MODEL} on :${PORT} (TP=${TP} EP=${EP} DP=${DP})"

# ---- Step 2: exec the SGLang server (this process IS the deployment) -------
DP_ARGS=()
if [ "${DP}" -gt 1 ]; then
    DP_ARGS=(--dp-size "${DP}" --enable-dp-attention --enable-dp-lm-head)
fi

exec python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tokenizer-path "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 --port "${PORT}" \
    --tp "${TP}" --ep "${EP}" "${DP_ARGS[@]}" \
    --trust-remote-code \
    --attention-backend triton \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --disable-custom-all-reduce \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --context-length "${CONTEXT_LENGTH}"
