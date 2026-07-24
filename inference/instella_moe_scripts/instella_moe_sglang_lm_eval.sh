#!/bin/bash
set -x

# unset in case already set by container
unset WORLD_SIZE
unset RANK
unset MASTER_ADDR
unset LOCAL_RANK
unset MASTER_PORT
unset PET_MASTER_ADDR
unset PET_NPROC_PER_NODE

export SGLANG_ROCM_FUSED_DECODE_MLA="0"
export SGLANG_USE_AITER=1
export GPU_MAX_HW_QUEUES="8"

export MODE="FARSKIP_REFERENCE"
export MODEL_PATH=/path/to/hf-instella-model # amd/Instella-MoE-16B-A3B-Base

bash "$(dirname "$0")/../scripts/update_sglang_workspace_files.sh" "$(dirname "$0")/../sglang/python" /app/sglang/python
export PYTHONPATH="/app/sglang/python:$PYTHONPATH"
pip show lm-eval 2>/dev/null | grep -q "Version: 0.4.9" || pip install lm-eval==0.4.9
export HF_ALLOW_CODE_EVAL="1"

if [[ "$MODE" == "FARSKIP" ]]; then
    export FARSKIP_OVERLAPPED_DECODER_LAYER=1
elif [[ "$MODE" == "FARSKIP_REFERENCE" ]]; then
    export FARSKIP_REFERENCE_DECODER_LAYER=1
elif [[ "$MODE" == "OFF" ]]; then
    unset FARSKIP_OVERLAPPED_DECODER_LAYER FARSKIP_REFERENCE_DECODER_LAYER
else
    echo "Please set MODE to FARSKIP, FARSKIP_REFERENCE, or OFF" && exit 1
fi

export EP_SIZE=8
export DP_SIZE=8
export TP_SIZE=$EP_SIZE
export BATCH_SIZE=64 # with dp-attention this is the batch-size per GPU
export MAX_GEN_TOKENS=1024

#task_list="gsm8k_cot,arc_challenge,mmlu" # full set incl. mmlu (57 subtasks, much slower)
task_list="gsm8k_cot,arc_challenge"


model_args="pretrained=${MODEL_PATH}"
model_args+=",dp_size=${DP_SIZE}"
model_args+=",tp_size=${TP_SIZE}"
model_args+=",ep_size=${EP_SIZE}"
model_args+=",enable_dp_attention=true"
model_args+=",dtype=bfloat16"
model_args+=",cuda_graph_max_bs=${BATCH_SIZE}"
model_args+=",disable_shared_experts_fusion=true"
model_args+=",disable_radix_cache=true"
model_args+=",attention_backend=triton"
model_args+=",mem_fraction_static=0.8"
model_args+=",trust_remote_code=True"

lm_eval --model sglang \
    --model_args "$model_args" \
    --tasks $task_list \
    --gen_kwargs temperature=0,do_sample=False,max_gen_toks=$MAX_GEN_TOKENS \
    --batch_size $BATCH_SIZE --trust_remote_code --confirm_run_unsafe_code
