#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Modification Copyright© 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron torch_dist -> HuggingFace safetensors (reverse of the forward script).

CPU-only: streams the distributed checkpoint (no_dist=True) and rewrites Megatron
tensor names to HF via miles' deepseek_v3 mapping. FarSkip's
self_attention.linear_gate_proj.weight -> self_attn.gate_proj.weight is handled by
miles natively (with a defensive fallback here for un-patched miles).

Usage:
  python convert_megatron_to_hf.py \\
      --input-dir <megatron_out>/release --output-dir <hf_out> \\
      --origin-hf-dir <original_hf>      # copies tokenizer/config/modeling files

--input-dir must contain common.pt + the distcp shards. By default the HF
embedding/lm_head keep the checkpoint's padded_vocab_size (matching the HF config);
pass --vocab-size N to trim to a smaller true vocab.
"""
import argparse
import json
import os
import pickle
import re
import shutil
import sys
import time
import types

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override


def _ensure_miles_on_path():
    """Make `import miles...` work when run as a standalone script."""
    try:
        import miles  # noqa: F401

        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    # This file lives at training/examples/scripts/instella_moe_process_checkpoint/, so the
    # repo root (which holds rl/miles) is four levels up.
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    candidates = [
        os.environ.get("MILES_ROOT", ""),
        os.path.join(repo_root, "rl", "miles"),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "miles")):
            sys.path.insert(0, c)
            return
    raise ImportError(
        "Could not locate the miles package. Set MILES_ROOT to your "
        "Primus-Instella rl/miles checkout."
    )


_ensure_miles_on_path()

# The mxfp8 quantizer needs flashinfer / a newer sglang that is unavailable on
# ROCm. We only do bf16 conversion (no quant), so stub it to avoid the import.
_STUB = "miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_mxfp8"
_stub_mod = types.ModuleType(_STUB)


def _quantize_params_mxfp8(*args, **kwargs):  # pragma: no cover - not used for bf16
    raise NotImplementedError("mxfp8 quantization is stubbed out for bf16 conversion")


_stub_mod.quantize_params_mxfp8 = _quantize_params_mxfp8
sys.modules.setdefault(_STUB, _stub_mod)

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf as _convert_to_hf  # noqa: E402
from miles.backends.megatron_utils.megatron_to_hf import remove_padding  # noqa: E402

# FarSkip gated-attention projection. miles' patched converter maps this
# natively; keep a defensive fallback so an un-patched miles still works.
_GATE_RE = re.compile(
    r"module\.module\.decoder\.layers\.(\d+)\.self_attention\.linear_gate_proj\.weight"
)


def convert_to_hf(args, model_name, name, param):
    m = _GATE_RE.match(name)
    if m:
        # Plain nn.Linear(hidden_size, num_heads * v_head_dim); 1:1 name remap.
        return [(f"model.layers.{m.group(1)}.self_attn.gate_proj.weight", param)]
    return _convert_to_hf(args, model_name, name, param)


class UnpicklerWrapper(pickle.Unpickler):
    """common.pt / .metadata reference megatron/glm classes we don't want to
    (and can't) import here; swap them for a harmless dummy."""

    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return
    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts, (
            f"{name}: leading dim {param.shape[0]} != num_experts {num_experts}"
        )
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            yield expert_name, param[expert_id]
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield name, param
        return
    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers, (
            f"{name}: leading dim {param.shape[0]} != num_layers {num_layers}"
        )
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            yield from get_expert_param(args, layer_name, param[layer_id])
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None):
    # for miles update_weight compatibility (unused here, but convert_to_hf reads it)
    args.sglang_enable_ep_moe = False
    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    n_tensors = 0
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        for converted_name, converted_param in convert_to_hf(args, model_name, name, param):
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param.contiguous()
            current_size += tensor_size
            total_size += tensor_size
            n_tensors += 1

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}
    num_files = len(modeltensors)
    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i + 1:05d}-of-{num_files:05d}.safetensors"
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"{index_filepath} saved ({n_tensors} tensors, {total_size / 1e9:.2f} GB).")

    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i + 1:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        safetensors.torch.save_file(tensors, os.path.join(output_dir, filename))
        print(f"{filename} saved in {time.time() - t:.2f} sec.")


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            continue
        shutil.copy(origin_filename, os.path.join(output_dir, filename))
        print(f"copied {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Megatron torch_dist dir containing common.pt (e.g. .../release).")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write the HF checkpoint.")
    parser.add_argument("--origin-hf-dir", type=str, default=None,
                        help="Original HF dir to copy tokenizer/config/modeling files from.")
    parser.add_argument("--model-name", type=str, default="deepseekv3",
                        help="miles converter key; deepseekv3 covers DeepSeek-V3/Instella FarSkip.")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output dir if it exists.")
    parser.add_argument("--chunk-size", type=int, default=5 * 1024**3, help="Max bytes per safetensors shard.")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Trim embedding/lm_head to this vocab; default keeps the padded vocab.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    common_pt = os.path.join(args.input_dir, "common.pt")
    if not os.path.isfile(common_pt):
        raise FileNotFoundError(
            f"{common_pt} not found. --input-dir must point at the torch_dist dir "
            f"(e.g. <megatron_out>/release)."
        )

    print(f"loading model from {args.input_dir}")
    t = time.time()
    megatron_args = torch.load(common_pt, weights_only=False)["args"]
    # Keep the padded vocab (matches the HF config and reference checkpoints);
    # otherwise remove_padding would trim embed/lm_head to the true vocab.
    padded_vocab = getattr(megatron_args, "padded_vocab_size", None)
    if padded_vocab is not None:
        megatron_args.vocab_size = padded_vocab

    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(args.input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"model loaded in {time.time() - t:.2f} sec.")

    save_tensors(megatron_args, args.model_name, state_dict, args.output_dir, args.chunk_size, args.vocab_size)

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
    print("DONE")


if __name__ == "__main__":
    main()
