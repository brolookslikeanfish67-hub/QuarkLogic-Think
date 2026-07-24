#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Instella MoE HF -> Megatron torch_dist.

Loads HF weights into a Megatron model via mbridge and saves a (reshardable)
torch_dist checkpoint. Architecture is supplied via ${MODEL_ARGS[@]}; weights via
--hf-checkpoint. Adds explicit CLI control over the YaRN RoPE params that are
otherwise only Megatron TransformerConfig defaults (and thus not overridable and
invisible to a weight round-trip): --original-max-position-embeddings / --beta-fast
/ --beta-slow. Their names match TransformerConfig fields, so they auto-map via
core_transformer_config_from_args and are recorded in common.pt.

Usage:
  torchrun --nproc-per-node 1 convert_hf_to_megatron.py \\
      ${MODEL_ARGS[@]} --hf-checkpoint <hf_dir> --save <megatron_out>
"""
import gc
import os
import shutil
import sys


def _ensure_paths():
    """Make miles + the full Megatron-LM tree importable when run standalone.

    megatron.training.* lives in the Megatron-LM source tree (the pip
    `megatron-core` package only ships megatron.core), so we add the submodule.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # This file lives at training/examples/scripts/instella_moe_process_checkpoint/, so the
    # repo root (which holds rl/miles) is four levels up.
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    miles_candidates = [
        os.environ.get("MILES_ROOT", ""),
        os.path.join(repo_root, "rl", "miles"),
    ]
    miles_root = next((c for c in miles_candidates if c and os.path.isdir(os.path.join(c, "miles"))), None)
    if miles_root is None:
        raise ImportError("Could not locate the miles package; set MILES_ROOT.")

    meg_candidates = [
        os.environ.get("MEGATRON_LM_PATH", ""),
        os.path.join(miles_root, "..", "..", "training", "third_party", "Megatron-LM"),
    ]
    meg_root = next(
        (c for c in meg_candidates if c and os.path.isfile(os.path.join(c, "megatron", "training", "arguments.py"))),
        None,
    )
    if meg_root is None:
        raise ImportError("Could not locate the Megatron-LM source tree; set MEGATRON_LM_PATH.")

    # Megatron-LM first so its full `megatron` package (incl. megatron.training)
    # takes precedence over any megatron-core-only install.
    for p in (os.path.abspath(meg_root), os.path.abspath(miles_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_paths()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.training.arguments import parse_args, validate_args  # noqa: E402
from megatron.training.checkpointing import (  # noqa: E402
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    save_checkpoint,
)
from megatron.training.training import get_model  # noqa: E402

import miles_plugins.mbridge  # noqa: F401,E402
from mbridge import AutoBridge  # noqa: E402
from miles.backends.megatron_utils.arguments import set_default_megatron_args  # noqa: E402
from miles.backends.megatron_utils.initialize import init  # noqa: E402
from miles.backends.megatron_utils.model_provider import get_model_provider_func  # noqa: E402
from miles.utils.logging_utils import configure_logger  # noqa: E402
from miles.utils.memory_utils import print_memory  # noqa: E402
from miles_plugins.models.hf_attention import _load_hf_config  # noqa: E402


def add_conversion_args(parser):
    group = parser.add_argument_group(title="hf->megatron conversion")
    group.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    group.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="Method to convert megatron weights to HF weights for SGLang.",
    )
    try:
        group.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass

    # --- YaRN RoPE overrides (names MUST match TransformerConfig fields so that
    #     core_transformer_config_from_args() auto-maps them into the config) ---
    group.add_argument(
        "--original-max-position-embeddings",
        type=int,
        default=4096,
        help="YaRN pre-scaling context length (HF rope_scaling.original_max_position_embeddings).",
    )
    group.add_argument(
        "--beta-fast", type=float, default=32,
        help="YaRN beta_fast (HF rope_scaling.beta_fast).",
    )
    group.add_argument(
        "--beta-slow", type=float, default=1,
        help="YaRN beta_slow (HF rope_scaling.beta_slow).",
    )
    return parser


def get_args():
    args = parse_args(add_conversion_args)
    args = set_default_megatron_args(args)

    # Pass megatron validate_args with conversion-friendly defaults.
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be <= number of layers {args.num_layers}. "
        "Use fewer GPUs (--nproc-per-node) for this conversion."
    )

    def ceildiv(a, b):
        return -(a // -b)

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)
            if args.decoder_last_pipeline_num_layers > 0:
                break
            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline parallel size for {args.num_layers} layers "
                    f"and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, "
        f"decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )
    print(
        f"[YaRN] original_max_position_embeddings={args.original_max_position_embeddings} "
        f"beta_fast={args.beta_fast} beta_slow={args.beta_slow} "
        f"rotary_scaling_factor={getattr(args, 'rotary_scaling_factor', None)} "
        f"rotary_base={getattr(args, 'rotary_base', None)}"
    )

    validate_args(args)
    return args


def main():
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from miles.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger()

    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    args = get_args()
    init(args)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    hf_model_path = args.hf_checkpoint
    try:
        bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    except (ValueError, KeyError):
        # Fallback for configs whose model_type is unknown to installed transformers.
        bridge = AutoBridge.from_config(_load_hf_config(hf_model_path))

    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)

    dist.barrier()

    # Must be the LAST step (after the barrier): higher-level scripts treat the
    # "release" tracker as the success signal.
    if dist.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
