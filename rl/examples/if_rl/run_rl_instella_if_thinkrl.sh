#!/bin/bash
# Instella MoE IF RL
# 3 nodes / 24 GPUs: 8 train (EP=8) + 16 rollout (2 engines TP=8/EP=8/DP=8).
# Recipe/env split out of the launcher:
#   prepare_ifeval_olmes.py     -> OLMES-align the IFEval eval file
#   eval_config_if_thinkrl.yaml -> eval datasets/decoding
#   mis.yaml                    -> train-infer mismatch (MIS) config
set -ex
export PYTHONUNBUFFERED=1

# --- Cluster layout ---------------------------------------------------------
NUM_NODES=${NUM_NODES:-3}
GPUS_PER_NODE=8
ACTOR_NUM_NODES=1
ACTOR_GPUS_PER_NODE=8
ROLLOUT_GPUS=16

# --- ROCm / SGLang / Ray env ------------------------------------------------
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
unset CUDA_VISIBLE_DEVICES 2>/dev/null || true
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true

export ORIG_MAX_POS_EMB=4096
export FARSKIP_REFERENCE_DECODER_LAYER=1
export SGLANG_MOE_PADDING=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1

export NCCL_TIMEOUT=3600
export RCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_ENABLE_MONITORING=0
export RAY_grpc_client_keepalive_time_ms=86400000
export RAY_grpc_client_keepalive_timeout_ms=86400000
export RAY_grpc_keepalive_timeout_ms=86400000
export RAY_gcs_rpc_server_reconnect_timeout_s=86400

export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=32768
export SGLANG_REQ_RUNNING_TIMEOUT=1800

export REWARD_SCALE=10
export BATCHED_REWARD_CONCURRENCY=${BATCHED_REWARD_CONCURRENCY:-128}
export DEPRECATED_MEGATRON_COMPATIBLE=1
export PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:-/path/to/libbnxt_re.tar.gz}"

# IFBench/IFEvalG sentence tokenization needs NLTK punkt.
export NLTK_DATA="${NLTK_DATA:-${HOME}/nltk_data}"
python3 -c "import nltk; nltk.download('punkt_tab', download_dir='${NLTK_DATA}', quiet=True)" 2>/dev/null || true

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)

# In-repo Megatron-LM (FarSkip-patched by rl/setup; training/ is vendored based on Primus).
REPO_ROOT=$(cd -- "${MILES_ROOT}/../.." &>/dev/null && pwd)
MEGATRON_PATH="${MEGATRON_PATH:-${REPO_ROOT}/training/third_party/Megatron-LM}"
# rlsys docker (rlsys/miles:rocm7-mi300-sglang0.5.9-te2.10.0-dev-307b5e86): container-local sglang.
SGLANG_PYTHON="${SGLANG_PYTHON:-/app/sglang/python}"
export PYTHONPATH="${MILES_ROOT}:${MEGATRON_PATH}:${SGLANG_PYTHON}:${PYTHONPATH:-}"

# Init from the curated SFT checkpoint (HF for sglang+tokenizer, Megatron for --load).
HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/sft_checkpoint_hf}"
HF_SGLANG="${HF_SGLANG:-/path/to/sft_checkpoint_hf}"
MEGATRON_CKPT="${MEGATRON_CKPT:-/path/to/sft_checkpoint_megatron_ep8}"

DATA_DIR="${DATA_DIR:-/path/to/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output/if_thinkrl}"
mkdir -p "${OUTPUT_DIR}"

# --- Training data (Dolci-Think-RL-7B IF RLVR Mixture) ----------------------
# Build via: python ../mopd/prepare_rl_dolci.py --if-only --output <path>
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/dolci_think_rl_if.jsonl}"
[ -f "${TRAIN_DATA}" ] || { echo "[ERROR] Training data not found at ${TRAIN_DATA}"; exit 1; }

# --- Eval data + config -----------------------------------------------------
IFEVAL_EVAL="${IFEVAL_EVAL:-${DATA_DIR}/ifeval/ifeval_eval.jsonl}"
IFBENCH_EVAL="${IFBENCH_EVAL:-${DATA_DIR}/IFBench/IFBench_eval.jsonl}"
[ -f "${IFEVAL_EVAL}" ]  || { echo "[ERROR] missing eval file: ${IFEVAL_EVAL}" >&2; exit 1; }
[ -f "${IFBENCH_EVAL}" ] || { echo "[ERROR] missing eval file: ${IFBENCH_EVAL}" >&2; exit 1; }

# OLMES-align IFEval (strip system prompt); IFBench is already user-only.
IFEVAL_EVAL_OLMES="${OUTPUT_DIR}/ifeval_eval_olmes.jsonl"
python3 "${SCRIPT_DIR}/prepare_ifeval_olmes.py" "${IFEVAL_EVAL}" "${IFEVAL_EVAL_OLMES}"
[ -s "${IFEVAL_EVAL_OLMES}" ] || { echo "[ERROR] failed to build OLMES-aligned IFEval file" >&2; exit 1; }
IFEVAL_EVAL="${IFEVAL_EVAL_OLMES}"

EVAL_CONFIG="${OUTPUT_DIR}/eval_config_if_thinkrl.yaml"
IFEVAL_EVAL="${IFEVAL_EVAL}" IFBENCH_EVAL="${IFBENCH_EVAL}" \
    sed -e "s#\${IFEVAL_EVAL}#${IFEVAL_EVAL}#g" -e "s#\${IFBENCH_EVAL}#${IFBENCH_EVAL}#g" \
    "${SCRIPT_DIR}/eval_config_if_thinkrl.yaml" > "${EVAL_CONFIG}"

# --- Ray cluster ------------------------------------------------------------
if [ -z "${MASTER_ADDR:-}" ]; then
    MASTER_ADDR=$(ray list nodes --format json 2>/dev/null | python3 -c "
import sys, json
nodes = json.load(sys.stdin)
head = [n for n in nodes if n.get('is_head_node')]
print(head[0]['node_ip'] if head else '')
" 2>/dev/null)
fi
[ -n "${MASTER_ADDR}" ] || { echo "[ERROR] Cannot determine Ray head node IP"; exit 1; }

DASHBOARD_PORT=$(ps aux | grep -oP '(?<=--dashboard-port[= ])\d+' | head -1)
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
GCS_ADDRESS="${MASTER_ADDR}:6379"
DASHBOARD_ADDRESS="http://${MASTER_ADDR}:${DASHBOARD_PORT}"
ray status

# --- torch_memory_saver -----------------------------------------------------
TMS_SO=$(find /tmp/torch_memory_saver -maxdepth 1 -name '*preload*.so' 2>/dev/null | head -1)
if [ -z "${TMS_SO}" ]; then
    TMS_SO=$(python3 -c "from torch_memory_saver.utils import get_binary_path_from_package; print(get_binary_path_from_package('torch_memory_saver_hook_mode_preload'))" 2>/dev/null || true)
fi

# --- Checkpoint + overlay sanity checks -------------------------------------
[ -d "${HF_SGLANG}" ] || { echo "[ERROR] HF checkpoint not found: ${HF_SGLANG}"; exit 1; }
[ -f "${MEGATRON_CKPT}/latest_checkpointed_iteration.txt" ] || { echo "[ERROR] Megatron checkpoint not found: ${MEGATRON_CKPT}"; exit 1; }

DSV2="${SGLANG_PYTHON}/sglang/srt/models/deepseek_v2.py"
WLDR="${SGLANG_PYTHON}/sglang/srt/models/deepseek_common/deepseek_weight_loader.py"
if ! grep -q "class DeepseekV2ReferenceFarSkipDecoderLayer" "$DSV2" 2>/dev/null \
   || ! grep -q 'if "self_attn" in name' "$WLDR" 2>/dev/null; then
    echo "[ERROR] FarSkip overlay missing in $SGLANG_PYTHON; re-run: bash rl/setup/setup_all_instella_rl.sh"
    exit 1
fi

# --- Model architecture (Instella-MoE (16B FarSkip MoE + MLA)) --------------------
# MODEL_ARGS + MOE_LAYER_FREQ come from the shared model def (miles convention).
source "${SCRIPT_DIR}/../../scripts/models/instella-moe.sh"
MODEL_ARGS+=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${HF_CHECKPOINT}
)

# --- Checkpoints ------------------------------------------------------------
# Fresh start (no LOAD_CKPT): --finetune loads weights only (iteration reset,
# fresh RL optimizer). Resume: set LOAD_CKPT + START_ROLLOUT_ID and drop --finetune.
CKPT_ARGS=(
    --load ${LOAD_CKPT:-${MEGATRON_CKPT}}
    --hf-checkpoint ${HF_SGLANG}
    --save ${OUTPUT_DIR}/checkpoints
    --save-interval 25
    --save-retain-interval 100
)
if [ -z "${LOAD_CKPT:-}" ]; then
    CKPT_ARGS+=(--finetune)
fi

# --- Rollout (64 prompts x 8 samples = 512; one rollout per grad step) -------
ROLLOUT_ARGS=(
    --prompt-data ${TRAIN_DATA}
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --rollout-shuffle

    --custom-rm-path examples.if_rl.reward_model.batched_reward

    --num-rollout ${NUM_ROLLOUT:-1400}
    --start-rollout-id ${START_ROLLOUT_ID:-0}
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --rollout-max-response-len 16384
    --rollout-temperature 1.0

    --global-batch-size 512
    --balance-data
)

# Active sampling: drop zero-reward-variance groups (no GRPO signal).
ACTIVE_SAMPLING_ARGS=(
    --over-sampling-batch-size 128
    --dynamic-sampling-filter-path
        examples.if_rl.dynamic_sampling_filter.check_reward_nonzero_std_and_retirement
)

# Partial rollout: recycle long-tail decodes; offpolicy tokens get loss_mask=0.
PARTIAL_ROLLOUT_ARGS=(
    --partial-rollout
    --mask-offpolicy-in-partial-rollout
)

# --- GRPO (OLMo3 Think-RL: centered adv, clip_higher 0.272, beta 0, token loss) ---
GRPO_ARGS=(
    --advantage-estimator grpo
    --disable-grpo-std-normalization
    --kl-coef 0
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.272
    --use-tis
    --calculate-per-token-loss
)

# MIS (train-infer mismatch): bounds come from mis.yaml, not --tis-clip.
CUSTOM_ARGS=(
    --custom-config-path ${SCRIPT_DIR}/mis.yaml
    --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --clip-grad 1.0
)

# --- Parallelism / performance (train side: 8 GPUs, EP=8) -------------------
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1

    # DO NOT add --overlap-grad-reduce / --overlap-param-gather: the RL trainer
    # installs a custom no_sync_func, which Megatron's distrib_optimizer asserts
    # must be None when overlap is set.
    --use-distributed-optimizer

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --balance-by-flops

    --distributed-timeout-minutes 480
)

# --- SGLang (2 engines, each TP=8/EP=8/DP=8 with DP attention) --------------
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-expert-parallel-size 8
    --sglang-enable-dp-attention
    --sglang-dp-size 8
    --sglang-enable-dp-lm-head
    --sglang-disable-custom-all-reduce
    --sglang-disable-radix-cache
    --sglang-disable-shared-experts-fusion
    --sglang-attention-backend triton
    --sglang-watchdog-timeout 600
    --sglang-context-length 36864

    --sglang-mem-fraction-static 0.78
    --sglang-chunked-prefill-size 4096
    --sglang-max-running-requests 512
    --sglang-cuda-graph-max-bs 512
    --sglang-decode-log-interval 1000
)

# --- Eval (IFEval + IFBench every 50 rollouts) ------------------------------
EVAL_ARGS=(
    --eval-interval 50
    --eval-config ${EVAL_CONFIG}
    --eval-function-path examples.if_rl.eval_with_flush.generate_rollout
    --log-passrate
    --custom-rollout-log-function-path
        examples.if_rl.custom_rollout_logger.log_rollout_data
    --custom-eval-rollout-log-function-path
        examples.if_rl.custom_rollout_logger.log_eval_rollout_data
)

# --- Wandb ------------------------------------------------------------------
WANDB_RUN_NAME="${WANDB_RUN_NAME:-instella-16b-if-dolci-thinkrl-3node-async1-mis-rwgate-partial}"
WANDB_ARGS=(
    --use-wandb
    --wandb-team "${WANDB_TEAM}"
    --wandb-project "${WANDB_PROJECT:-instella-think}"
    --wandb-group ${WANDB_RUN_NAME}
    --wandb-mode ${WANDB_MODE:-online}
    --wandb-dir ${OUTPUT_DIR}/wandb
)

MOE_IMPROVEMENTS=(
    --use-miles-router
    --use-rollout-routing-replay
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --no-gradient-accumulation-fusion
    --no-check-for-nan-in-loss-and-grad
    --update-weight-buffer-size $((4 * 1024 * 1024 * 1024))
    --use-fault-tolerance
    --rollout-health-check-timeout 600
    --miles-router-health-check-failure-threshold 10
    --rollout-health-check-interval 120
    --pause-generation-mode in_place
    --make-vocab-size-divisible-by 1
    # DO NOT add --manual-gc: RL steps are huge/slow, GC never runs between them,
    # dead GPU-tensor refs fragment reserved memory -> OOM. Leave auto-GC on.
)

# --- Runtime env JSON for Ray multinode workers -----------------------------
LD_PRELOAD_JSON=""
[ -n "${TMS_SO}" ] && LD_PRELOAD_JSON="\"LD_PRELOAD\": \"${TMS_SO}\","

WANDB_KEY_JSON=""
[ -n "${WANDB_API_KEY:-}" ] && WANDB_KEY_JSON="\"WANDB_API_KEY\": \"${WANDB_API_KEY}\","

# Opt-in: dump training rollouts to disk (only when INSTELLA_TRAIN_DUMP_DIR set).
TRAIN_DUMP_JSON=""
[ -n "${INSTELLA_TRAIN_DUMP_DIR:-}" ] && TRAIN_DUMP_JSON="\"INSTELLA_TRAIN_DUMP_DIR\": \"${INSTELLA_TRAIN_DUMP_DIR}\", \"INSTELLA_TRAIN_DUMP_EVERY\": \"${INSTELLA_TRAIN_DUMP_EVERY:-1}\", \"INSTELLA_TRAIN_DUMP_LOGPROBS\": \"${INSTELLA_TRAIN_DUMP_LOGPROBS:-0}\","

RUNTIME_ENV_JSON=$(cat <<ENVEOF
{
  "env_vars": {
    ${LD_PRELOAD_JSON}
    ${WANDB_KEY_JSON}
    ${TRAIN_DUMP_JSON}
    "PYTHONPATH": "${MILES_ROOT}:${MEGATRON_PATH}:${SGLANG_PYTHON}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "HIP_VISIBLE_DEVICES": "${HIP_VISIBLE_DEVICES}",
    "ORIG_MAX_POS_EMB": "4096",
    "FARSKIP_REFERENCE_DECODER_LAYER": "1",
    "SGLANG_MOE_PADDING": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_ROCM_FUSED_DECODE_MLA": "0",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "1",
    "NCCL_TIMEOUT": "3600",
    "RCCL_TIMEOUT": "3600",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "7200",
    "TORCH_NCCL_ENABLE_MONITORING": "0",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "600",
    "SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "32768",
    "SGLANG_REQ_RUNNING_TIMEOUT": "1800",
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
    "RAY_memory_monitor_refresh_ms": "0",
    "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
    "REWARD_SCALE": "10",
    "INSTELLA_REQUIRE_THINK": "${INSTELLA_REQUIRE_THINK:-1}",
    "BATCHED_REWARD_CONCURRENCY": "${BATCHED_REWARD_CONCURRENCY}",
    "DEPRECATED_MEGATRON_COMPATIBLE": "1",
    "NLTK_DATA": "${NLTK_DATA}",
    "PATH_TO_BNXT_TAR_PACKAGE": "${PATH_TO_BNXT_TAR_PACKAGE}"
  }
}
ENVEOF
)

# --- Launch -----------------------------------------------------------------
cd "${MILES_ROOT}"
export RAY_ADDRESS="${GCS_ADDRESS}"

ray job submit --address="${DASHBOARD_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ${MILES_ROOT}/train_async.py \
    --actor-num-nodes ${ACTOR_NUM_NODES} \
    --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \
    --rollout-num-gpus ${ROLLOUT_GPUS} \
    --num-gpus-per-node ${GPUS_PER_NODE} \
    --update-weights-interval 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ACTIVE_SAMPLING_ARGS[@]}" \
    "${PARTIAL_ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${MOE_IMPROVEMENTS[@]}" \
    "${MISC_ARGS[@]}"
