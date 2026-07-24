#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Validate that the FarSkip SGLang overlay landed correctly.

Standalone and inference-only: run this right after
update_sglang_workspace_files.sh, with no dependency on RL / Megatron. Reads the
overlaid sglang root from SGLANG_PYTHON (default /app/sglang/python). Exit 0 if
every check passes, 1 otherwise.
"""
import importlib
import os
import sys

SGLANG = os.environ.get("SGLANG_PYTHON", "/app/sglang/python")

ok = True


def check(label, fn):
    global ok
    try:
        fn()
        print("OK   " + label)
    except Exception as e:
        print("FAIL " + label + ": " + str(e))
        ok = False


def _read(rel):
    return open(os.path.join(SGLANG, rel)).read()


def farskip_decoder():
    m = importlib.import_module("sglang.srt.models.deepseek_v2")
    m.DeepseekV2ReferenceFarSkipDecoderLayer


def instella_arch():
    # The arch file must be importable and register InstellaMoEForCausalLM, else
    # sglang silently falls back to TransformersForCausalLM (no FarSkip).
    m = importlib.import_module("sglang.srt.models.instella_moe")
    assert m.EntryClass.__name__ == "InstellaMoEForCausalLM"


def weight_loader_safety():
    s = _read("sglang/srt/models/deepseek_common/deepseek_weight_loader.py")
    assert 'weight_loader = getattr(param, "weight_loader", None)' in s
    assert "weight_loader = param.weight_loader" not in s


def unquant_aiter_moe():
    s = _read("sglang/srt/layers/quantization/unquant.py")
    for needle in (
        "def unshuffle_weight(",
        "def _make_aiter_weight_loader(",
        'if not getattr(layer.w13_weight, "is_shuffled", False):',
        "layer.w13_weight.data.copy_(",
        "layer.w13_weight.weight_loader = _make_aiter_weight_loader(",
    ):
        assert needle in s, needle


def shuffle_invertible():
    import torch
    from aiter.ops.shuffle import shuffle_weight
    from sglang.srt.layers.quantization.unquant import unshuffle_weight

    x = torch.randn(8, 2816, 2048, dtype=torch.bfloat16)
    assert torch.equal(x, unshuffle_weight(shuffle_weight(x.clone(), (16, 16)), (16, 16)))


def kv_splits():
    assert "SGLANG_FORCE_KV_SPLITS" in _read("sglang/srt/layers/attention/triton_backend.py")


for mod in (
    "sglang.srt.models.deepseek_v2",
    "sglang.srt.layers.communicator",
    "sglang.srt.layers.moe.topk",
):
    check("import " + mod, lambda mod=mod: importlib.import_module(mod))
check("farskip reference decoder", farskip_decoder)
check("instella_moe arch (InstellaMoEForCausalLM)", instella_arch)
check("0c weight_loader safety", weight_loader_safety)
check("0d unquant aiter moe", unquant_aiter_moe)
check("unshuffle(shuffle(x))==x", shuffle_invertible)
check("dp-attn kv-splits", kv_splits)

sys.exit(0 if ok else 1)
