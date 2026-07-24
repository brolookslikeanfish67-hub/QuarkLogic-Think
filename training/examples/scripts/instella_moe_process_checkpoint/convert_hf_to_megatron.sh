#!/bin/bash
# Convert an Instella MoE HF checkpoint
# -> parallelism-agnostic Megatron torch_dist (written at TP=PP=EP=1, reshardable).
#
# Usage:
#   bash convert_hf_to_megatron.sh <HF_CHECKPOINT_DIR> <OUTPUT_DIR>
#   HF_CHECKPOINT=/path/hf OUTPUT_DIR=/path/out bash convert_hf_to_megatron.sh
#
# Optional env overrides:
#   NGPU=1                # GPUs/processes
#   MTP_NUM_LAYERS=0      set 1 to also build an MTP layer (needs MTP weights in HF)
#   MASTER_PORT=12355
#   YaRN RoPE (a computed buffer, invisible to a weight round-trip; set to match the
#   source HF config rope_scaling if it differs from the Megatron defaults):
#     YARN_ORIGINAL_MAX_POS=4096  YARN_BETA_FAST=32  YARN_BETA_SLOW=1
set -euo pipefail

HF_CHECKPOINT="${1:-${HF_CHECKPOINT:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-}}"

if [ -z "${HF_CHECKPOINT}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: bash $0 <HF_CHECKPOINT_DIR> <OUTPUT_DIR>" >&2
    exit 1
fi
if [ ! -d "${HF_CHECKPOINT}" ]; then
    echo "ERROR: HF checkpoint dir not found: ${HF_CHECKPOINT}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# This script lives at training/examples/scripts/instella_moe_process_checkpoint/, so the
# Primus-Instella repo root is four levels up.
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"

FWD_CONVERTER="${FWD_CONVERTER:-${SCRIPT_DIR}/convert_hf_to_megatron.py}"
if [ ! -f "${FWD_CONVERTER}" ]; then
    echo "ERROR: forward converter not found: ${FWD_CONVERTER}" >&2
    exit 1
fi
# MILES_ROOT provides miles + miles_plugins imports (and locates the model-args script).
MILES_ROOT="${MILES_ROOT:-${REPO_ROOT}/rl/miles}"
if [ ! -d "${MILES_ROOT}/miles" ]; then
    echo "ERROR: miles package not found under ${MILES_ROOT}." >&2
    echo "       Set MILES_ROOT to your Primus-Instella rl/miles checkout." >&2
    exit 1
fi

# megatron.training.* lives in the full Megatron-LM tree (pip megatron-core only
# ships megatron.core), so point PYTHONPATH at the Megatron-LM submodule.
MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-${REPO_ROOT}/training/third_party/Megatron-LM}"
if [ ! -f "${MEGATRON_LM_PATH}/megatron/training/arguments.py" ]; then
    echo "ERROR: megatron.training not found under ${MEGATRON_LM_PATH}" >&2
    echo "       Set MEGATRON_LM_PATH to your Megatron-LM checkout root." >&2
    exit 1
fi
export PYTHONPATH="${MEGATRON_LM_PATH}:${MILES_ROOT}:${PYTHONPATH:-}"

NGPU="${NGPU:-1}"
MTP_NUM_LAYERS="${MTP_NUM_LAYERS:-0}"
MASTER_PORT="${MASTER_PORT:-12355}"

# YaRN RoPE params (see header); defaults match the Instella HF/Megatron defaults.
YARN_ORIGINAL_MAX_POS="${YARN_ORIGINAL_MAX_POS:-4096}"
YARN_BETA_FAST="${YARN_BETA_FAST:-32}"
YARN_BETA_SLOW="${YARN_BETA_SLOW:-1}"

# Model architecture args (single source of truth): the model-args script defines
# the MODEL_ARGS=(...) array (miles scripts/models/ convention). Staged under
# Primus-Instella rl/ for now; point MODEL_ARGS_SCRIPT elsewhere if it moves.
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${REPO_ROOT}/rl/scripts/models/instella-moe.sh}"
if [ ! -f "${MODEL_ARGS_SCRIPT}" ]; then
    echo "ERROR: model-args script not found: ${MODEL_ARGS_SCRIPT}" >&2
    echo "       Set MODEL_ARGS_SCRIPT to your instella-moe.sh." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${MODEL_ARGS_SCRIPT}"

# Tokenizer comes from the HF checkpoint being converted.
MODEL_ARGS+=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${HF_CHECKPOINT}"
)

if [ "${MTP_NUM_LAYERS}" != "0" ]; then
    MODEL_ARGS+=(--mtp-num-layers "${MTP_NUM_LAYERS}")
fi

echo "============================================================"
echo "Instella MoE HF -> Megatron torch_dist"
echo "  HF_CHECKPOINT : ${HF_CHECKPOINT}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  NGPU          : ${NGPU}"
echo "  MTP_NUM_LAYERS: ${MTP_NUM_LAYERS}"
echo "  MILES_ROOT    : ${MILES_ROOT}"
echo "  MEGATRON_LM   : ${MEGATRON_LM_PATH}"
echo "  YaRN          : original_max_pos=${YARN_ORIGINAL_MAX_POS} beta_fast=${YARN_BETA_FAST} beta_slow=${YARN_BETA_SLOW}"
echo "============================================================"

export MILES_ROOT MEGATRON_LM_PATH
torchrun --nproc-per-node="${NGPU}" --master-port="${MASTER_PORT}" \
    "${FWD_CONVERTER}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${OUTPUT_DIR}" \
    --original-max-position-embeddings "${YARN_ORIGINAL_MAX_POS}" \
    --beta-fast "${YARN_BETA_FAST}" \
    --beta-slow "${YARN_BETA_SLOW}"

echo ""
echo "DONE. torch_dist checkpoint written under: ${OUTPUT_DIR}"
echo "  (release checkpoint dir: ${OUTPUT_DIR}/release ; tracker: latest_checkpointed_iteration.txt)"
