#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Merge (average) weights from multiple Hugging Face model checkpoints.

Example:
  python merge_hf_ckpts.py \
    --model_class AutoModelForCausalLM \
    --out_dir merged_model \
    --checkpoints /path/to/ckpt1 /path/to/ckpt2 \
    --weights 0.5 0.5 \
    --dtype float32 \
    --save_safetensors \
    --trust_remote_code \
    --copy_tokenizer

Notes:
- All checkpoints must share the same architecture and parameter names/shapes.
- Floating-point tensors are accumulated in float32 for numerical stability,
  then cast back to the requested --dtype before saving.
- Non-floating tensors (e.g. integer buffers) are taken from the first checkpoint.
"""

import argparse
import json
import os
from typing import List, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

MODEL_CLASS_MAP = {
    "AutoModel": AutoModel,
    "AutoModelForCausalLM": AutoModelForCausalLM,
    "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
    "AutoModelForMaskedLM": AutoModelForMaskedLM,
}

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _normalize_weights(ws: Optional[List[float]], n: int) -> List[float]:
    if ws is None:
        return [1.0 / n] * n
    if len(ws) != n:
        raise ValueError(f"--weights length ({len(ws)}) must match #checkpoints ({n})")
    s = float(sum(ws))
    if s == 0:
        raise ValueError("Sum of --weights must be > 0")
    return [float(w) / s for w in ws]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Average the weights of multiple Hugging Face checkpoints."
    )
    ap.add_argument("--checkpoints", nargs="+", required=True, help="Paths or HF ids")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--model_class",
        default="AutoModelForCausalLM",
        choices=sorted(MODEL_CLASS_MAP.keys()),
        help="Which AutoModel* class to load",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=sorted(DTYPE_MAP.keys()),
        help="Load/merge dtype (recommended float32 for stability)",
    )
    ap.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional merge weights (same count as checkpoints). If omitted: uniform average.",
    )
    ap.add_argument(
        "--copy_tokenizer",
        action="store_true",
        help="Also copy tokenizer files from the first checkpoint (if available).",
    )
    ap.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Save merged model in safetensors format (if supported).",
    )
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading checkpoints.",
    )
    return ap.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    ckpts = args.checkpoints
    weights = _normalize_weights(args.weights, len(ckpts))

    os.makedirs(args.out_dir, exist_ok=True)

    model_cls = MODEL_CLASS_MAP[args.model_class]
    dtype = DTYPE_MAP[args.dtype]

    # Load config from the first checkpoint (assumed shared across all).
    config = AutoConfig.from_pretrained(ckpts[0], trust_remote_code=args.trust_remote_code)

    print(f"Loading base (accumulator) from: {ckpts[0]}")
    acc = model_cls.from_pretrained(
        ckpts[0],
        config=config,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    acc_sd = acc.state_dict()

    # Accumulate float tensors in float32 for numerical stability; keep the
    # non-float tensors as-is from the first checkpoint.
    acc_fp32 = {}
    acc_nonfloat = {}
    for k, v in acc_sd.items():
        if torch.is_floating_point(v):
            acc_fp32[k] = v.detach().to(torch.float32) * weights[0]
        else:
            acc_nonfloat[k] = v.detach().clone()

    for i in range(1, len(ckpts)):
        ckpt = ckpts[i]
        w = weights[i]
        print(f"Loading and merging: {ckpt} (weight={w:.6f})")

        config_i = AutoConfig.from_pretrained(ckpt, trust_remote_code=args.trust_remote_code)
        m = model_cls.from_pretrained(
            ckpt,
            config=config_i,
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=args.trust_remote_code,
        )
        sd = m.state_dict()

        if set(sd.keys()) != set(acc_sd.keys()):
            missing = set(acc_sd.keys()) - set(sd.keys())
            extra = set(sd.keys()) - set(acc_sd.keys())
            raise ValueError(
                f"State dict keys mismatch for {ckpt}.\n"
                f"Missing: {sorted(missing)[:20]}...\n"
                f"Extra: {sorted(extra)[:20]}..."
            )

        for k, v in sd.items():
            if torch.is_floating_point(v):
                if k not in acc_fp32:
                    raise RuntimeError(f"Internal error: float key {k} missing from accumulator.")
                if v.shape != acc_fp32[k].shape:
                    raise ValueError(f"Shape mismatch for {k}: {v.shape} vs {acc_fp32[k].shape}")
                acc_fp32[k].add_(v.detach().to(torch.float32), alpha=w)
            # Non-float tensors are kept from the first checkpoint.

        del m, sd

    # Rebuild the final state dict in the requested dtype.
    final_sd = {}
    for k in acc_sd.keys():
        if k in acc_fp32:
            final_sd[k] = acc_fp32[k].to(dtype)
        else:
            final_sd[k] = acc_nonfloat[k]

    acc.load_state_dict(final_sd, strict=True)

    print(f"Saving merged model to: {args.out_dir}")
    acc.save_pretrained(args.out_dir, safe_serialization=args.save_safetensors)
    config.save_pretrained(args.out_dir)

    meta = {
        "checkpoints": ckpts,
        "normalized_weights": weights,
        "model_class": args.model_class,
        "merge_dtype": args.dtype,
    }
    with open(os.path.join(args.out_dir, "merge_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.copy_tokenizer:
        try:
            tok = AutoTokenizer.from_pretrained(ckpts[0], trust_remote_code=args.trust_remote_code)
            tok.save_pretrained(args.out_dir)
            print("Tokenizer copied from first checkpoint.")
        except Exception as e:
            print(f"Tokenizer copy failed (continuing): {e}")

    print("Done.")


if __name__ == "__main__":
    main()
