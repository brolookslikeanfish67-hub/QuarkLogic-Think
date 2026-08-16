#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Usage:
#   EXP=…pretrain.yaml bash examples/run_instella.sh                    # pretrain (default)
#   EXP=…sft.yaml       bash examples/run_instella.sh --task sft
#   EXP=…dpo.yaml       bash examples/run_instella.sh --task dpo --compute-ref-logprobs
#   EXP=…dpo.yaml       bash examples/run_instella.sh --task dpo
#
###############################################################################

print_usage() {
cat << EOF
Usage: bash $(basename "$0") [--task <pretrain|docmask|sft|dpo>] [--help] [--compute-ref-logprobs]

Unified Instella launcher. Selects the entrypoint by --task (default: pretrain):

    pretrain    primus/cli/main.py train pretrain --config \$EXP    (MegatronPretrainTrainer)
    docmask     examples/megatron/pretrain_docmask.py --exp \$EXP  (MegatronDocMaskTrainer)
    sft         examples/megatron/sft_train.py --exp \$EXP                (requires sft_config block)
    dpo         examples/megatron/dpo_train.py --exp \$EXP                (requires dpo_config block)

Options:
    --task <name>               Which training task to launch (default: pretrain).
                                May also be set via the TASK environment variable.
    --compute-ref-logprobs    (dpo only) Run a forward-only pass to compute
                                reference log-probabilities and save them as an
                                IndexedDataset. Run this before DPO training.

Environment variables (must set before running):

    EXP                         # Path to experiment config file (required)
    TASK                        # Alternative to --task
    NNODES=1                    # Number of nodes (default: 1)
    NODE_RANK=0                 # Current node rank (default: 0)
    GPUS_PER_NODE=8             # Number of GPUs per node (default: 8)
    MASTER_ADDR=localhost       # Master node address (default: localhost)
    MASTER_PORT=1234            # Master node port (default: 1234)
    PRIMUS_HIPBLASLT_TUNING_STAGE=0  # HipBLASLt tuning stage: 0/1/2/3 (default: 0)

HipBLASLt tuning stages:
    1: Dump GEMM shapes
    2: Offline tuning
    3: Use tuned config

Examples:

    EXP=examples/megatron/configs/instella_16B_pretrain.yaml bash examples/run_instella.sh --task pretrain
    EXP=examples/megatron/configs/instella_16B_sft.yaml      bash examples/run_instella.sh --task sft

    # Two-step DPO workflow:
    EXP=examples/megatron/configs/instella_16B_dpo.yaml bash examples/run_instella.sh --task dpo --compute-ref-logprobs
    EXP=examples/megatron/configs/instella_16B_dpo.yaml bash examples/run_instella.sh --task dpo

EOF
}

# -------------------- Argument Parsing --------------------
TASK="${TASK:-pretrain}"
COMPUTE_REF_LOGPROBS=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            print_usage
            exit 0
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        --compute-ref-logprobs)
            COMPUTE_REF_LOGPROBS=1
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

HOSTNAME=$(hostname)

LOG_INFO() {
    if [ "$*" = "" ]; then
        echo ""
    else
        echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
    fi
}

LOG_INFO_RANK0() {
    if [ "${NODE_RANK:-0}" -eq 0 ]; then
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
        fi
    fi
}

LOG_ERROR() {
    echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*" >&2
}

# -------------------- Task Validation --------------------
case "$TASK" in
    pretrain|docmask|sft|dpo)
        ;;
    *)
        LOG_ERROR "Unknown --task '$TASK'. Choose one of: pretrain, docmask, sft, dpo."
        exit 1
        ;;
esac

if [[ "$COMPUTE_REF_LOGPROBS" -eq 1 && "$TASK" != "dpo" ]]; then
    LOG_ERROR "--compute-ref-logprobs is only valid with --task dpo."
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-1234}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

LOG_INFO_RANK0 "========== Training cluster info =========="
LOG_INFO_RANK0 "TASK: $TASK"
LOG_INFO_RANK0 "MASTER_ADDR: $MASTER_ADDR"
LOG_INFO_RANK0 "MASTER_PORT: $MASTER_PORT"
LOG_INFO_RANK0 "NNODES: $NNODES"
LOG_INFO_RANK0 "NODE_RANK: $NODE_RANK"
LOG_INFO_RANK0 "GPUS_PER_NODE: $GPUS_PER_NODE"
LOG_INFO_RANK0 ""

PRIMUS_PATH=$(realpath "$(dirname "$0")/..")
export DATA_PATH="${DATA_PATH:-${PRIMUS_PATH}/data}"
export HF_HOME="${HF_HOME:-${DATA_PATH}/huggingface}"

if [[ "${SKIP_PIP_INSTALL:-0}" == "1" ]]; then
    LOG_INFO_RANK0 "SKIP_PIP_INSTALL=1, skipping 'pip install -r requirements.txt'."
else
    pip install -r "$PRIMUS_PATH/requirements.txt" --quiet
fi

# -------------------- EXP Check --------------------
if [ -z "${EXP:-}" ]; then
    LOG_ERROR "EXP must be specified (e.g., examples/megatron/exp_pretrain.yaml)."
    exit 1
fi

if [ ! -f "${EXP}" ]; then
    LOG_ERROR "The specified EXP file does not exist: ${EXP}"
    exit 1
fi

TRAIN_LOG="${TRAIN_LOG:-output/log_torchrun_${TASK}_$(basename "$EXP" .yaml).txt}"
mkdir -p "$(dirname "$TRAIN_LOG")"

LOG_INFO_RANK0 "========== Training info =========="
LOG_INFO_RANK0 "EXP: $EXP"
LOG_INFO_RANK0 "TRAIN_LOG: $TRAIN_LOG"
LOG_INFO_RANK0 "PRIMUS_PATH: $PRIMUS_PATH"
LOG_INFO_RANK0 "DATA_PATH: $DATA_PATH"
LOG_INFO_RANK0 "HF_HOME: $HF_HOME"
LOG_INFO_RANK0 ""

# -------------------- NCCL and Communication Setup --------------------
HIP_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
export HIP_VISIBLE_DEVICES

export NCCL_DEBUG="${NCCL_DEBUG:-}"
export NCCL_CHECKS_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0

if [ -z "${NCCL_IB_HCA:-}" ]; then
    NCCL_IB_HCA=$(bash "${PRIMUS_PATH}/examples/scripts/get_nccl_ib_hca.sh")
fi
export NCCL_IB_HCA

if [ -z "${IP_INTERFACE:-}" ]; then
    IP_INTERFACE=$(bash "${PRIMUS_PATH}/examples/scripts/get_ip_interface.sh")
fi
export IP_INTERFACE

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}"

LOG_INFO_RANK0 "========== NCCL and Network Settings =========="
LOG_INFO_RANK0 "NCCL_DEBUG: $NCCL_DEBUG"
LOG_INFO_RANK0 "NCCL_CHECKS_DISABLE: $NCCL_CHECKS_DISABLE"
LOG_INFO_RANK0 "NCCL_IB_GID_INDEX: $NCCL_IB_GID_INDEX"
LOG_INFO_RANK0 "NCCL_CROSS_NIC: $NCCL_CROSS_NIC"
LOG_INFO "NCCL_IB_HCA: $NCCL_IB_HCA"
LOG_INFO "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
LOG_INFO "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
LOG_INFO ""

# ----------------- AMD-specific GPU optimizations -----------------
export HSA_ENABLE_SDMA=1
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0
export RCCL_MSCCLPP_FORCE_ENABLE=0
export RCCL_MSCCLPP_THRESHOLD=$((1 * 1024 * 1024 * 1024))
export MSCCLPP_DISABLE_CHANNEL_CACHE=FALSE
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0

LOG_INFO_RANK0 "========== AMD-specific GPU optimizations =========="
LOG_INFO_RANK0 "HSA_ENABLE_SDMA: $HSA_ENABLE_SDMA"
LOG_INFO_RANK0 "HSA_NO_SCRATCH_RECLAIM: $HSA_NO_SCRATCH_RECLAIM"
LOG_INFO_RANK0 "RCCL_MSCCL_ENABLE: $RCCL_MSCCL_ENABLE"
LOG_INFO_RANK0 "RCCL_MSCCLPP_ENABLE: $RCCL_MSCCLPP_ENABLE"
LOG_INFO_RANK0 "RCCL_MSCCLPP_FORCE_ENABLE: $RCCL_MSCCLPP_FORCE_ENABLE"
LOG_INFO_RANK0 "RCCL_MSCCLPP_THRESHOLD: $RCCL_MSCCLPP_THRESHOLD"
LOG_INFO_RANK0 "MSCCLPP_DISABLE_CHANNEL_CACHE: $MSCCLPP_DISABLE_CHANNEL_CACHE"
LOG_INFO_RANK0 "TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: $TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"
LOG_INFO_RANK0 ""

# ----------------- Performance tuning -----------------
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_HIGH_PRIORITY=1
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
export NCCL_P2P_NET_CHUNKSIZE="${NCCL_P2P_NET_CHUNKSIZE:-524288}"

export NVTE_USE_CAST_TRANSPOSE_TRITON=1
export NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
export NVTE_CK_USES_BWD_V3="${NVTE_CK_USES_BWD_V3:-0}"
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
export NVTE_FUSED_ATTN_LOG_CONFIG=0
export PATCH_TE_FLASH_ATTN="${PATCH_TE_FLASH_ATTN:-0}"

LOG_INFO_RANK0 "========== Performance tuning =========="
LOG_INFO_RANK0 "GPU_MAX_HW_QUEUES: $GPU_MAX_HW_QUEUES"
LOG_INFO_RANK0 "CUDA_DEVICE_MAX_CONNECTIONS: $CUDA_DEVICE_MAX_CONNECTIONS"
LOG_INFO_RANK0 "TORCH_NCCL_HIGH_PRIORITY: $TORCH_NCCL_HIGH_PRIORITY"
LOG_INFO_RANK0 "NCCL_PXN_DISABLE: $NCCL_PXN_DISABLE"
LOG_INFO_RANK0 "NCCL_P2P_NET_CHUNKSIZE: $NCCL_P2P_NET_CHUNKSIZE"
LOG_INFO_RANK0 "NVTE_CK_USES_BWD_V3: $NVTE_CK_USES_BWD_V3"
LOG_INFO_RANK0 "NVTE_USE_CAST_TRANSPOSE_TRITON: $NVTE_USE_CAST_TRANSPOSE_TRITON"
LOG_INFO_RANK0 "NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE: $NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE"

if [[ "$PATCH_TE_FLASH_ATTN" == "1" ]]; then
    LOG_INFO_RANK0 "Patching _flash_attn_max_version in attention.py..."
    sed -i 's/_flash_attn_max_version = PkgVersion(\".*\")/_flash_attn_max_version = PkgVersion(\"3.0.0.post1\")/' \
        /opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformer_engine/pytorch/attention.py
    LOG_INFO_RANK0 "Patch complete."
fi
LOG_INFO_RANK0 ""

# ----------------- Rebuild bnxt -----------------
export REBUILD_BNXT="${REBUILD_BNXT:-0}"
export PATH_TO_BNXT_TAR_PACKAGE="${PATH_TO_BNXT_TAR_PACKAGE:-}"

if [[ "$REBUILD_BNXT" == "1" && -f "$PATH_TO_BNXT_TAR_PACKAGE" ]]; then
    LOG_INFO "Rebuilding bnxt from $PATH_TO_BNXT_TAR_PACKAGE ..."
    tar xzf "${PATH_TO_BNXT_TAR_PACKAGE}" -C /tmp/
    mv /tmp/libbnxt_re-* /tmp/libbnxt
    mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox
    cd /tmp/libbnxt/ && ./autogen.sh && ./configure
    make -C /tmp/libbnxt clean all install
    echo '/usr/local/lib' > /etc/ld.so.conf.d/libbnxt_re.conf
    ldconfig
    cp -f /tmp/libbnxt/bnxt_re.driver /etc/libibverbs.d/
    cd "${PRIMUS_PATH}"
    LOG_INFO "Rebuilding libbnxt done."
else
    LOG_INFO "Skip bnxt rebuild. REBUILD_BNXT=$REBUILD_BNXT, PATH_TO_BNXT_TAR_PACKAGE=$PATH_TO_BNXT_TAR_PACKAGE"
fi

# -------------------- HipBLASLt Tuning --------------------
handle_hipblaslt_tuning() {
    local STAGE="${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}"
    local MODEL_NAME
    MODEL_NAME=$(basename "$EXP" .yaml)
    local TUNE_LOG_PATH="${PRIMUS_PATH}/output/tune_hipblaslt/${MODEL_NAME}"
    local RESULT_FILE="tune_hipblas_gemm_results.txt"

    mkdir -p "$TUNE_LOG_PATH"

    case "$STAGE" in
        0)
            export TE_HIPBLASLT_TUNING_RUN_COUNT="${TE_HIPBLASLT_TUNING_RUN_COUNT:-10}"
            export TE_HIPBLASLT_TUNING_ALGO_COUNT="${TE_HIPBLASLT_TUNING_ALGO_COUNT:-50}"
            ;;
        1)
            [[ "${TE_HIPBLASLT_TUNING:-0}" == "1" ]] && { LOG_ERROR "Disable TE_HIPBLASLT_TUNING for shape dump"; exit 1; }
            mkdir -p "$TUNE_LOG_PATH/gemm_shape"
            export HIPBLASLT_LOG_MASK=32
            export HIPBLASLT_LOG_FILE="${HIPBLASLT_LOG_FILE:-$TUNE_LOG_PATH/gemm_shape/dump_hipblaslt_gemm_shape_${NODE_RANK}.txt}"
            unset HIPBLASLT_TUNING_OVERRIDE_FILE
            ;;
        2)
            mkdir -p "$TUNE_LOG_PATH/gemm_tune"
            python "${PRIMUS_PATH}/examples/offline_tune/offline_tune_gemm.py" \
                --dump-shape-path-or-file "$TUNE_LOG_PATH/gemm_shape" \
                --tune-result-path "$TUNE_LOG_PATH/gemm_tune/$RESULT_FILE" \
                --num-devices 8
            LOG_ERROR "GEMM tuning finished. Set PRIMUS_HIPBLASLT_TUNING_STAGE=3 and re-run training."
            exit 0
            ;;
        3)
            local TUNE_FILE="$TUNE_LOG_PATH/gemm_tune/$RESULT_FILE"
            [[ ! -f "$TUNE_FILE" ]] && { LOG_ERROR "Missing tuning result: $TUNE_FILE"; exit 1; }
            export HIPBLASLT_TUNING_OVERRIDE_FILE="$TUNE_FILE"
            ;;
    esac

    if [ "${NODE_RANK:-0}" -eq 0 ]; then
        LOG_INFO "========== Training tuning info =========="
        LOG_INFO "TE_HIPBLASLT_TUNING: ${TE_HIPBLASLT_TUNING:-}"
        LOG_INFO "TE_HIPBLASLT_TUNING_RUN_COUNT: ${TE_HIPBLASLT_TUNING_RUN_COUNT:-}"
        LOG_INFO "TE_HIPBLASLT_TUNING_ALGO_COUNT: ${TE_HIPBLASLT_TUNING_ALGO_COUNT:-}"
        LOG_INFO "PRIMUS_HIPBLASLT_TUNING_STAGE: ${STAGE}"
        LOG_INFO "HIPBLASLT_LOG_MASK: ${HIPBLASLT_LOG_MASK:-}"
        LOG_INFO "HIPBLASLT_LOG_FILE: ${HIPBLASLT_LOG_FILE:-}"
        LOG_INFO "HIPBLASLT_TUNING_OVERRIDE_FILE: ${HIPBLASLT_TUNING_OVERRIDE_FILE:-}"
        if [ "$STAGE" -eq 1 ]; then
            LOG_INFO "Dump HipBLASLt shapes, make sure train_iters is set to a very small value."
        fi
        LOG_INFO ""
    fi
}

handle_hipblaslt_tuning

# -------------------- Python Path Setup --------------------
setup_pythonpath() {
    local site_packages
    site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    export PYTHONPATH="${site_packages}:${PRIMUS_PATH}:${PYTHONPATH:-}"
}

setup_pythonpath

run_prepare_experiment() {
    PRIMUS_PATCH_ARGS_FILE=$(mktemp /tmp/primus_patch_args.XXXXXX.yaml)
    trap 'rm -f "$PRIMUS_PATCH_ARGS_FILE"' EXIT

    local SCRIPT="$PRIMUS_PATH/examples/scripts/prepare_experiment.py"

    local BACKEND_ARG=()
    if [[ -n "${BACKEND_PATH:-}" ]]; then
        BACKEND_ARG=(--backend_path "$BACKEND_PATH")
    fi

    if ! python3 "$SCRIPT" \
        --config "$EXP" \
        --data_path "$DATA_PATH" \
        --patch_args "$PRIMUS_PATCH_ARGS_FILE" \
        "${BACKEND_ARG[@]}"; then
        LOG_ERROR "$SCRIPT failed, aborting."
        exit 1
    fi
}

run_prepare_experiment

# ---------- Parse optional patch args ----------
TRAIN_EXTRA_ARGS=""
TORCHRUN_EXTRA_ARGS=""

if [[ -f "$PRIMUS_PATCH_ARGS_FILE" ]]; then
    LOG_INFO_RANK0 "Loading patch args from $PRIMUS_PATCH_ARGS_FILE"
    source_yaml_args() {
        local file=$1
        local key=$2
        local collect=0
        local args=""
        while IFS= read -r line; do
            if [[ $collect -eq 0 && $line == "$key:"* ]]; then
                args="${line#*:}"
                collect=1
                continue
            fi
            if [[ $collect -eq 1 ]]; then
                if [[ $line =~ ^[[:space:]] ]]; then
                    args="${args} ${line}"
                else
                    break
                fi
            fi
        done < "$file"
        echo "$args"
    }

    TRAIN_EXTRA_ARGS=$(source_yaml_args "$PRIMUS_PATCH_ARGS_FILE" train_args)
    TORCHRUN_EXTRA_ARGS=$(source_yaml_args "$PRIMUS_PATCH_ARGS_FILE" torchrun_args)

    if [[ -n "$TRAIN_EXTRA_ARGS" ]]; then
        LOG_INFO_RANK0 "Patched TRAIN args: $TRAIN_EXTRA_ARGS"
    fi

    if [[ -n "$TORCHRUN_EXTRA_ARGS" ]]; then
        LOG_INFO_RANK0 "Patched TORCHRUN args: $TORCHRUN_EXTRA_ARGS"
    fi
else
    LOG_INFO_RANK0 "No patch args file found at $PRIMUS_PATCH_ARGS_FILE, skipping patch args."
fi

# -------------------- Launch Training --------------------
DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

if [[ "$COMPUTE_REF_LOGPROBS" -eq 1 ]]; then
    _TMP_EXP=$(mktemp /tmp/dpo_ref_logprobs_XXXXXX.yaml)
    sed 's/compute_ref_logprobs: false/compute_ref_logprobs: true/' "$EXP" > "$_TMP_EXP"
    EXP="$_TMP_EXP"
    trap 'rm -f "$_TMP_EXP"' EXIT
    LOG_INFO "compute_ref_logprobs mode: using temp config $_TMP_EXP"
fi

case "$TASK" in
    pretrain)
        ENTRY="primus/cli/main.py train pretrain --config $EXP"
        ;;
    docmask)
        ENTRY="examples/megatron/pretrain_docmask.py --exp $EXP"
        ;;
    sft)
        ENTRY="examples/megatron/sft_train.py --exp $EXP"
        ;;
    dpo)
        ENTRY="examples/megatron/dpo_train.py --exp $EXP"
        ;;
esac

CMD="torchrun ${DISTRIBUTED_ARGS[*]} $TORCHRUN_EXTRA_ARGS $ENTRY $TRAIN_EXTRA_ARGS $*"

LOG_INFO "Launching distributed training with command: $CMD"

eval "$CMD" 2>&1 | tee "$TRAIN_LOG"
exit_code=${PIPESTATUS[0]}

if [ "${PRIMUS_HIPBLASLT_TUNING_STAGE:-0}" -eq 1 ]; then
    LOG_INFO "[PRIMUS_HIPBLASLT_TUNING_STAGE-1]: HipBlasLT gemm shape dump is finished, " \
           "please set PRIMUS_HIPBLASLT_TUNING_STAGE to 2, " \
           "and tune the gemm with a single node."
fi

LOG_INFO "torchrun exited with code $exit_code"

if [[ $exit_code -ne 0 ]]; then
    if [[ $exit_code -ge 128 ]]; then
        signal=$((exit_code - 128))
        LOG_ERROR "torchrun crashed due to signal $signal"
    else
        LOG_ERROR "torchrun exited with code $exit_code"
    fi
fi

exit "$exit_code"
