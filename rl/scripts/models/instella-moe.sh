# Instella 16B FarSkip (gated-attention MoE, DeepSeek-V3-style) model args.
#
# Follows the miles scripts/models/<family>.sh convention: defines MOE_LAYER_FREQ
# and the MODEL_ARGS=(...) array, sourced by callers. Staged under Primus-Instella
# rl/ for open-sourcing; matches the miles layout so moving it later is a plain mv.
#
# FarSkip deltas vs vanilla DeepSeek-V3 MoE: --gated-attention (per-layer self_attn
# gate projection) and --use-simple-farskip-layer (FarSkip layer spec).

# MoE layer frequency: layer 0 dense, layers 1..26 MoE (first_k_dense_replace=1).
NLAYERS="${MODEL_ARGS_NUM_LAYERS:-27}"
FIRST_K_DENSE_REPLACE=1
arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# Padded to a multiple that matches the HF config vocab_size (tokenizer base is 128000).
PADDED_VOCAB_SIZE="${PADDED_VOCAB_SIZE:-128896}"

# instella-moe
MODEL_ARGS=(
    --disable-bias-linear
    --num-layers "$NLAYERS"
    --hidden-size 2048
    --ffn-hidden-size 10944
    --num-attention-heads 16
    --kv-channels 128
    --normalization RMSNorm
    --position-embedding-type rope
    --no-position-embedding
    --norm-epsilon 1e-6
    --swiglu
    --untie-embeddings-and-output-weights
    --padded-vocab-size "$PADDED_VOCAB_SIZE"

    # --- MLA (multi-latent attention) + FarSkip gated attention ---
    --multi-latent-attention
    --gated-attention
    --kv-lora-rank 512
    --qk-head-dim 96
    --qk-pos-emb-head-dim 32
    --v-head-dim 128
    --qk-layernorm
    --rotary-scaling-factor 40
    --rotary-base 8000000
    --mscale 1.0
    --mscale-all-dim 1.0
    --attention-softmax-in-fp32
    --no-rope-fusion

    --use-simple-farskip-layer

    # --- MoE ---
    --num-experts 64
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-ffn-hidden-size 1408
    --moe-router-topk 6
    --moe-shared-expert-intermediate-size 2816
    --moe-router-pre-softmax
    --moe-router-score-function sigmoid
    --moe-router-load-balancing-type seq_aux_loss
    --moe-token-dispatcher-type alltoall
    --moe-aux-loss-coeff 0.001
    --moe-grouped-gemm
    --moe-router-topk-scaling-factor 2.5
    --moe-router-dtype fp32
    --moe-router-enable-expert-bias

    # --- runtime / dtype (kept here so the doc-style one-liner works as-is) ---
    --seq-length 32768
    --max-position-embeddings 32768
    --bf16
)
