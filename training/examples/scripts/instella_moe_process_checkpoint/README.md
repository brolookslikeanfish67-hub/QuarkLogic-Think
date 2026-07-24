# Instella MoE Checkpoint Processing

## Checkpoint conversion
Convert Instella MoE checkpoints
between HuggingFace and Megatron `torch_dist` formats.

| Script | Direction | Runtime |
| --- | --- | --- |
| `convert_hf_to_megatron.sh` (+ `.py`) | HF → Megatron `torch_dist` | GPU (`torchrun`) |
| `convert_megatron_to_hf.py` | Megatron `torch_dist` → HF | CPU-only |

The Megatron side is written parallelism-agnostic (TP=PP=EP=1) and can be resharded
to any TP/PP/EP layout at load time via Megatron distributed checkpointing.

## Prerequisites

Run from a checkout of this repo with submodules initialized. The scripts locate
`rl/miles` and `training/third_party/Megatron-LM` relative to the repo root; override
`MILES_ROOT` / `MEGATRON_LM_PATH` if your layout differs. The model architecture is
defined once in `rl/scripts/models/instella-moe.sh` (override with
`MODEL_ARGS_SCRIPT`).

## HF → Megatron

```bash
bash convert_hf_to_megatron.sh /path/to/hf_checkpoint /path/to/megatron_out
```

Optional env overrides:

```bash
NGPU=1                # GPUs / processes
MTP_NUM_LAYERS=0      # set 1 to also build an MTP layer (needs MTP weights in HF)
# YaRN RoPE (a computed buffer, invisible to a weight round-trip). Set these to
# match the source HF config's rope_scaling if it differs from the defaults below:
YARN_ORIGINAL_MAX_POS=4096
YARN_BETA_FAST=32
YARN_BETA_SLOW=1
```

Output: `/path/to/megatron_out/release/` (torch_dist shards + `common.pt`).

## Megatron → HF

```bash
python convert_megatron_to_hf.py \
    --input-dir  /path/to/megatron_out/release \
    --output-dir /path/to/hf_out \
    --origin-hf-dir /path/to/hf_checkpoint   # copies tokenizer/config/modeling files
```

`--input-dir` must point at the dir containing `common.pt`. The HF embedding / lm_head
keep the checkpoint's padded vocab by default; pass `--vocab-size N` to trim.

## Checkpoint Merging
At the end of Mid-training we run model merging. For this we use `training/examples/scripts/instella_moe_process_checkpoint/merge_hf_ckpts.py`, which loads each checkpoint, accumulates the floating-point weights in `float32` for numerical stability, and casts back to the requested dtype before saving. All checkpoints must share the same architecture and parameter names/shapes.
Example Usage:
```bash
python training/examples/scripts/instella_moe_process_checkpoint/merge_hf_ckpts.py \
    --model_class AutoModelForCausalLM \
    --out_dir /path/to/midtrain_merged \
    --checkpoints /path/to/midtrain_v1 /path/to/midtrain_v2 /path/to/midtrain_v3 \
    --weights 1.0 1.0 1.0 \
    --dtype float32 \
    --save_safetensors \
    --trust_remote_code \
    --copy_tokenizer
```
`--weights` are optional and are normalized to sum to 1 (uniform averaging is used if omitted).
