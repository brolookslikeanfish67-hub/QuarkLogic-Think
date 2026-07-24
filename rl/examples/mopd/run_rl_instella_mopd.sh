#!/bin/bash
# Instella-MoE — Two-Teacher On-Policy Distillation (MOPD).
# Student (DPO model) does on-policy rollouts; each is scored by the domain's
# teacher SGLang server (token logprobs). advantage = teacher_logprob - student.
#   IF-domain prompts -> IF-RL teacher ; everything else -> DPO teacher (anchor).
# Routing is per-prompt via sample.metadata["domain"] (two_teacher_reward.py).
# Recipe/env split out of the launcher:
#   prepare_rl_dolci.py  -> build the domain-tagged mixed prompt set (OPD_DATA)
#   serve_teacher.sh     -> how to serve ONE Instella teacher (reused below)
#   eval_config_mopd.yaml -> eval datasets/decoding
# FILL IN (cluster-specific): teacher GPU/host placement, Ray cluster, OPD_DATA.
set -ex
export PYTHONUNBUFFERED=1

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)

# In-repo Megatron-LM (FarSkip-patched by rl/setup; training/ is vendored Primus-Instella).
REPO_ROOT=$(cd -- "${MILES_ROOT}/.." &>/dev/null && pwd)
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/training/third_party/Megatron-LM}"
# rlsys docker (rlsys/miles:rocm7-mi300-sglang0.5.9-te2.10.0-dev-307b5e86): container-local sglang.
SGLANG_PYTHON="${SGLANG_PYTHON:-/app/sglang/python}"
export PYTHONPATH="${MEGATRON_PATH}:${MILES_ROOT}:${SGLANG_PYTHON}:${PYTHONPATH:-}"

# Student = DPO model (HF for sglang+tokenizer, Megatron torch_dist for --load).
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/dpo_student_hf}"
HF_SGLANG="${HF_SGLANG:-/path/to/dpo_student_hf_sglang}"
MEGATRON_CKPT="${MEGATRON_CKPT:-/path/to/dpo_student_megatron}"

# Domain-tagged mixed prompt data (build via prepare_rl_dolci.py).
OPD_DATA="${OPD_DATA:-/path/to/opd_mixed_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output/instella_mopd_run}"
mkdir -p "${OUTPUT_DIR}"

# --- FarSkip / SGLang / Ray env (needed by both teachers and student rollout) ---
export ORIG_MAX_POS_EMB=4096
export FARSKIP_REFERENCE_DECODER_LAYER=1
export SGLANG_MOE_PADDING=1 SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0 SGLANG_USE_AITER=1 SGLANG_USE_ROCM700A=1
export DEPRECATED_MEGATRON_COMPATIBLE=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=3600 RCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 TORCH_NCCL_ENABLE_MONITORING=0
export RAY_grpc_client_keepalive_time_ms=86400000 RAY_grpc_client_keepalive_timeout_ms=86400000
export RAY_grpc_keepalive_timeout_ms=86400000 RAY_gcs_rpc_server_reconnect_timeout_s=86400
export SGLANG_HEALTH_CHECK_TIMEOUT=600 SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=32768 SGLANG_REQ_RUNNING_TIMEOUT=1800
export PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:-/path/to/libbnxt_re.tar.gz}"
# Teacher-scoring retry window: survives a teacher container restart (~10 min
# model reload): 30 attempts w/ backoff capped at 60s (see two_teacher_reward.py).
export OPD_TEACHER_RETRIES=30
# Eval-logger reward scale: miles' eval logger divides eval/<dataset> by this, so
# 10 makes eval metrics log as [0,1] accuracy. Does NOT change training rewards
# (reward_model.py hard-codes its own REWARD_SCALE=10 for magnitude).
export REWARD_SCALE=10
# IFBench/IFEval eval verifier (rm_type=ifbench) needs nltk punkt_tab.
export NLTK_DATA="${NLTK_DATA:-/path/to/nltk_data}"
python3 -c "import nltk; nltk.download('punkt_tab', download_dir='${NLTK_DATA}', quiet=True)" 2>/dev/null || true

# --- wandb ------------------------------------------------------------------
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/path/to/wandb.key}"
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "${WANDB_KEY_FILE}" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
fi
export WANDB_RUN_ID="${WANDB_RUN_ID:-}"
export WANDB_RESUME="${WANDB_RESUME:-}"
if [ -n "${WANDB_RUN_ID}" ]; then
    WANDB_RESUME_JSON="\"WANDB_RUN_ID\": \"${WANDB_RUN_ID}\", \"WANDB_RESUME\": \"${WANDB_RESUME:-allow}\","
else
    WANDB_RESUME_JSON=""
fi

# --- Teacher servers  <<< FILL IN host/GPU placement >>> --------------------
# Each Instella-MoE teacher is its own SGLang server; they only need to be
# network-reachable from the reward actors. Defaults assume two free GPU groups
# on the LOCAL node; point HOST at a remote to use an already-running teacher.
HF_SGLANG_DEFAULT="${HF_SGLANG}"
GENERAL_TEACHER_MODEL="${GENERAL_TEACHER_MODEL:-$HF_SGLANG_DEFAULT}"   # DPO anchor
# IF-RL teacher: use the *_hf_sglang variant (InstellaMoEForCausalLM, no auto_map).
IF_TEACHER_MODEL="${IF_TEACHER_MODEL:-/path/to/if_rl_teacher_hf_sglang}"
IF_TEACHER_HOST="${IF_TEACHER_HOST:-127.0.0.1}"
IF_TEACHER_PORT="${IF_TEACHER_PORT:-13141}"
IF_TEACHER_GPUS="${IF_TEACHER_GPUS:-0,1,2,3}"
GENERAL_TEACHER_HOST="${GENERAL_TEACHER_HOST:-127.0.0.1}"
GENERAL_TEACHER_PORT="${GENERAL_TEACHER_PORT:-13142}"
GENERAL_TEACHER_GPUS="${GENERAL_TEACHER_GPUS:-4,5,6,7}"
TEACHER_TP="${TEACHER_TP:-4}"

export IF_TEACHER_URL="http://${IF_TEACHER_HOST}:${IF_TEACHER_PORT}/generate"
export GENERAL_TEACHER_URL="http://${GENERAL_TEACHER_HOST}:${GENERAL_TEACHER_PORT}/generate"
# Warm backup teachers (comma-separated /generate URLs): reward_func fails over
# to these when the primary errors/hangs. Empty = disabled.
export IF_TEACHER_BACKUP_URLS="${IF_TEACHER_BACKUP_URLS:-}"
export GENERAL_TEACHER_BACKUP_URLS="${GENERAL_TEACHER_BACKUP_URLS:-}"
# Cap concurrent teacher scoring per reward actor (<=0 = off). DP=1 teachers have
# no dp-attention barrier, so the cap only throttles; set >0 to re-enable.
export OPD_TEACHER_MAX_CONCURRENCY="${OPD_TEACHER_MAX_CONCURRENCY:-0}"
export OPD_IF_DOMAINS="${OPD_IF_DOMAINS:-if,ifeval,ifbench,instruction_following}"

# Serve ONE teacher via the shared serve_teacher.sh (single source of truth for
# the Instella teacher serve config; SKIP_SETUP=1 assumes the RL overlay is
# already installed, as the student rollout requires it too).
launch_teacher () {  # name model gpus port
    local name="$1" model="$2" gpus="$3" port="$4"
    local log="${OUTPUT_DIR}/teacher_${name}.log"
    echo "[teacher:${name}] launching on GPUs ${gpus} port ${port} model=${model}"
    HIP_VISIBLE_DEVICES="${gpus}" CUDA_VISIBLE_DEVICES="${gpus}" \
    MODEL="${model}" PORT="${port}" TP="${TEACHER_TP}" EP="${TEACHER_TP}" \
    SGLANG_PYTHON="${SGLANG_PYTHON}" SKIP_SETUP=1 \
        bash "${SCRIPT_DIR}/serve_teacher.sh" > "${log}" 2>&1 &
    echo $!
}

wait_teacher () {  # host port
    local host="$1" port="$2" waited=0
    until curl -sf "http://${host}:${port}/health_generate" >/dev/null 2>&1; do
        sleep 10; waited=$((waited+10))
        [ "${waited}" -gt 900 ] && { echo "[teacher] ${host}:${port} not ready after 900s"; exit 1; }
        echo "[teacher] waiting for ${host}:${port} (${waited}s)"
    done
    echo "[teacher] ${host}:${port} healthy"
}

# Launch teachers locally only if their host is localhost; else assume remote.
if [ "${IF_TEACHER_HOST}" = "127.0.0.1" ]; then
    IF_PID=$(launch_teacher ifrl "${IF_TEACHER_MODEL}" "${IF_TEACHER_GPUS}" "${IF_TEACHER_PORT}")
fi
if [ "${GENERAL_TEACHER_HOST}" = "127.0.0.1" ]; then
    GEN_PID=$(launch_teacher dpo "${GENERAL_TEACHER_MODEL}" "${GENERAL_TEACHER_GPUS}" "${GENERAL_TEACHER_PORT}")
fi
wait_teacher "${IF_TEACHER_HOST}" "${IF_TEACHER_PORT}"
wait_teacher "${GENERAL_TEACHER_HOST}" "${GENERAL_TEACHER_PORT}"
echo "[teachers] IF=${IF_TEACHER_URL}  GENERAL=${GENERAL_TEACHER_URL}"

# --- Model architecture (Instella-MoE 16B FarSkip MoE + MLA) --------------------
# MODEL_ARGS + MOE_LAYER_FREQ from the shared model def (miles convention).
source "${SCRIPT_DIR}/../../scripts/models/instella-moe.sh"
MODEL_ARGS+=(
    # Frozen MoE load balancing for distillation: override the sourced aux-loss
    # coeff to 0 and freeze the expert-bias balancer, so the router does not drift
    # during OPD (R3 already replays the rollout routing; preserve, not rebalance).
    --moe-aux-loss-coeff 0
    --moe-router-bias-update-rate 0
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${HF_CHECKPOINT}
)

# --- Checkpoints ------------------------------------------------------------
# FRESH (default): fine-tune from the DPO base ckpt at iter 0, no optim/RNG.
# RESUME (RESUME=1): continue THIS run from its own latest ckpt (load optim+RNG+
# iteration); used by the teacher-address cutover relaunch. Caller should also
# export WANDB_RUN_ID (+ WANDB_RESUME=allow) and may set START_ROLLOUT_ID.
if [ "${RESUME:-0}" = "1" ]; then
    CKPT_ARGS=(
        --load "${RESUME_LOAD:-${OUTPUT_DIR}/checkpoints}" --hf-checkpoint ${HF_SGLANG}
        --save ${OUTPUT_DIR}/checkpoints
        --save-interval 5
    )
    [ -n "${START_ROLLOUT_ID:-}" ] && CKPT_ARGS+=(--start-rollout-id ${START_ROLLOUT_ID})
    # Eval-only resume (NUM_ROLLOUT=0): load WEIGHTS ONLY. The saved optim/scheduler
    # records lr_decay_steps from the real num_rollout, which won't match the
    # floored eval-only scheduler and trips Megatron's load_state_dict assert.
    [ "${NUM_ROLLOUT:-200}" = "0" ] && CKPT_ARGS+=(--no-load-optim --no-load-rng)
    export WANDB_RESUME="${WANDB_RESUME:-allow}"
    echo "[ckpt] RESUME mode: load=${RESUME_LOAD:-${OUTPUT_DIR}/checkpoints}" \
         "start-rollout-id=${START_ROLLOUT_ID:-<auto>} wandb_run_id=${WANDB_RUN_ID:-<unset!>}"
else
    CKPT_ARGS=(
        --load ${MEGATRON_CKPT} --hf-checkpoint ${HF_SGLANG}
        --finetune --no-load-optim --no-load-rng
        --save ${OUTPUT_DIR}/checkpoints --start-rollout-id 0
        # Save every 5 steps and KEEP ALL checkpoints (--save-retain-interval unset
        # => Megatron never prunes). Do NOT set INSTELLA_KEEP_LAST_N_CKPT.
        --save-interval 5
    )
fi

# --- Rollout (128 prompts x 4 samples = 512 responses/update) ---------------
ROLLOUT_ARGS=(
    --prompt-data ${OPD_DATA}
    --input-key prompt --label-key label --metadata-key metadata
    --apply-chat-template --rollout-shuffle
    --num-rollout ${NUM_ROLLOUT:-200} --rollout-batch-size 128 --n-samples-per-prompt 4
    --rollout-max-response-len 16384 --rollout-temperature 1.0 --rollout-top-p 1.0
    --global-batch-size 512 --balance-data
)

# Two-teacher OPD reward + post-process (routes per sample.metadata["domain"]).
# rm-url is the fallback teacher if a per-domain URL env var is unset.
RM_ARGS=(
    --custom-rm-path examples.mopd.two_teacher_reward.reward_func
    --custom-reward-post-process-path examples.mopd.two_teacher_reward.post_process_rewards
    --rm-url ${GENERAL_TEACHER_URL}
)

# MOPD objective: advantage = teacher_logprob - student_logprob (reverse-KL on the
# student-sampled token) with truncated importance weighting for the
# SGLang(infer)-vs-Megatron(train) mismatch. No KL term -> no --ref-load needed.
OPD_ARGS=(
    --advantage-estimator on_policy_distillation
    --kl-coef 0 --entropy-coef 0.00
    --use-tis --tis-clip 2.0 --tis-clip-low 0.5
)

OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-6 --lr-decay-style constant
    --lr-warmup-iters 30 --lr-warmup-init 1e-7
    --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95 --clip-grad 1.0
)

# seq-length / max-position-embeddings / bf16 come from instella-moe.sh.
PERF_ARGS=(
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1
    --context-parallel-size 1 --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu 8192
    --distributed-timeout-minutes 480
)

# Student rollout SGLang engine (separate from the two teacher servers above).
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8 --sglang-expert-parallel-size 8
    --sglang-enable-dp-attention --sglang-dp-size 8 --sglang-enable-dp-lm-head
    --sglang-disable-custom-all-reduce --sglang-disable-radix-cache
    --sglang-disable-shared-experts-fusion --sglang-attention-backend triton
    --sglang-watchdog-timeout 600 --sglang-context-length 36864
)

# Train/infer MoE consistency for the STUDENT: replay the SGLang rollout's expert
# assignments in the Megatron forward so log pi^train matches the sampled path
# (complementary to TIS). Teachers are frozen scorers and need no replay.
MOE_IMPROVEMENTS=(
    --use-miles-router
    --use-rollout-routing-replay
)

MISC_ARGS=(
    --attention-dropout 0.0 --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32 --no-gradient-accumulation-fusion
    --no-check-for-nan-in-loss-and-grad
    --update-weight-buffer-size $((4 * 1024 * 1024 * 1024))
    --use-fault-tolerance --rollout-health-check-timeout 600
    --rollout-health-check-interval 120 --miles-router-health-check-failure-threshold 10
    --dump-details ${OUTPUT_DIR}/dump_details --make-vocab-size-divisible-by 1
)

# --- Eval (AIME 24/25 + IFEval + IFBench) -----------------------------------
# Eval scoring uses the rm_type verifiers via two_teacher_reward's eval dispatch
# (the teacher-logprob reward is used only for training samples tagged domain).
AIME2024_EVAL="${AIME2024_EVAL:-/path/to/aime-2024-olmes.jsonl}"
AIME2025_EVAL="${AIME2025_EVAL:-/path/to/aime-2025-olmes.jsonl}"
IFEVAL_EVAL="${IFEVAL_EVAL:-/path/to/ifeval_eval.jsonl}"
IFBENCH_EVAL="${IFBENCH_EVAL:-/path/to/IFBench_eval.jsonl}"

EVAL_CONFIG="${OUTPUT_DIR}/eval_config_mopd.yaml"
sed -e "s#\${AIME2024_EVAL}#${AIME2024_EVAL}#g" -e "s#\${AIME2025_EVAL}#${AIME2025_EVAL}#g" \
    -e "s#\${IFEVAL_EVAL}#${IFEVAL_EVAL}#g" -e "s#\${IFBENCH_EVAL}#${IFBENCH_EVAL}#g" \
    "${SCRIPT_DIR}/eval_config_mopd.yaml" > "${EVAL_CONFIG}"

EVAL_ARGS=(
    --eval-interval 10
    --eval-config ${EVAL_CONFIG}
    --eval-function-path examples.if_rl.eval_with_flush.generate_rollout
    # NO --log-passrate: it assumes scalar pass/fail rewards and crashes on OPD's
    # non-scalar teacher-logprob reward. Eval metrics come from the custom logger.
    --custom-eval-rollout-log-function-path
        examples.if_rl.custom_rollout_logger.log_eval_rollout_data
)

WANDB_GROUP="${WANDB_GROUP:-$(basename "${OUTPUT_DIR}")}"
WANDB_ARGS=(
    --use-wandb
    --wandb-team "${WANDB_TEAM}"
    --wandb-project "${WANDB_PROJECT:-instella-mopd}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-mode "${WANDB_MODE:-online}"
    --wandb-dir "${OUTPUT_DIR}/wandb"
)

# --- Ray cluster + GPU layout: 3-NODE (1 train EP=8 + 2 rollout = 2 engines) --
# Teachers run as SEPARATE serving deployments (not in this cluster).
NUM_NODES="${NUM_NODES:-3}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-8}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

MASTER_ADDR="${MASTER_ADDR:-$(ray list nodes --format json 2>/dev/null | python3 -c "import sys,json;n=json.load(sys.stdin);h=[x for x in n if x.get('is_head_node')];print(h[0]['node_ip'] if h else '')" 2>/dev/null)}"
[ -z "${MASTER_ADDR}" ] && { echo "[ERROR] no Ray head; start a cluster first (see RAY_SETUP.md)"; exit 1; }
export RAY_ADDRESS="${MASTER_ADDR}:6379"
DASHBOARD_ADDRESS="http://${MASTER_ADDR}:8265"

RUNTIME_ENV_JSON=$(cat <<ENVEOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_PATH}:${MILES_ROOT}:${SGLANG_PYTHON}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "HIP_VISIBLE_DEVICES": "${HIP_VISIBLE_DEVICES}",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
    "FARSKIP_REFERENCE_DECODER_LAYER": "1",
    "ORIG_MAX_POS_EMB": "4096",
    "SGLANG_MOE_PADDING": "1",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "1",
    "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "600",
    "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "32768",
    "SGLANG_REQ_RUNNING_TIMEOUT": "1800",
    "NCCL_TIMEOUT": "3600",
    "RCCL_TIMEOUT": "3600",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "7200",
    "TORCH_NCCL_ENABLE_MONITORING": "0",
    "RAY_memory_monitor_refresh_ms": "0",
    "PATH_TO_BNXT_TAR_PACKAGE": "${PATH_TO_BNXT_TAR_PACKAGE}",
    "DEPRECATED_MEGATRON_COMPATIBLE": "1",
    "IF_TEACHER_URL": "${IF_TEACHER_URL}",
    "GENERAL_TEACHER_URL": "${GENERAL_TEACHER_URL}",
    "IF_TEACHER_BACKUP_URLS": "${IF_TEACHER_BACKUP_URLS}",
    "GENERAL_TEACHER_BACKUP_URLS": "${GENERAL_TEACHER_BACKUP_URLS}",
    "OPD_IF_DOMAINS": "${OPD_IF_DOMAINS}",
    "OPD_TEACHER_RETRIES": "30",
    "OPD_TEACHER_MAX_CONCURRENCY": "${OPD_TEACHER_MAX_CONCURRENCY}",
    "NLTK_DATA": "${NLTK_DATA}",
    "REWARD_SCALE": "10",
    ${WANDB_RESUME_JSON}
    "WANDB_API_KEY": "${WANDB_API_KEY:-}"
  }
}
ENVEOF
)

# --- Launch -----------------------------------------------------------------
cd "${MILES_ROOT}"
ray job submit --address="${DASHBOARD_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ${MILES_ROOT}/train.py \
    --actor-num-nodes ${ACTOR_NUM_NODES} \
    --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \
    --rollout-num-gpus ${ROLLOUT_GPUS} \
    --num-gpus-per-node ${GPUS_PER_NODE} \
    --update-weights-interval 1 \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
    "${RM_ARGS[@]}" "${OPD_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MOE_IMPROVEMENTS[@]}" \
    "${EVAL_ARGS[@]}" "${WANDB_ARGS[@]}" "${MISC_ARGS[@]}"

# --- cleanup teachers on exit -----------------------------------------------
[ -n "${IF_PID:-}" ]  && kill -9 "${IF_PID}"  2>/dev/null || true
[ -n "${GEN_PID:-}" ] && kill -9 "${GEN_PID}" 2>/dev/null || true
