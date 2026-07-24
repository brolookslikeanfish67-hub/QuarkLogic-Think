#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""RL setup verification, training side: megatron farskip layers + python deps.
The sglang overlay is verified separately by inference/scripts/verify_sglang_overlay.py."""
import importlib
import os
import sys

ok = True
MEGATRON = os.environ.get("MEGATRON_SUBMODULE")
if MEGATRON:
    sys.path.insert(0, MEGATRON)


def check(label, fn):
    global ok
    try:
        fn()
        print("OK   " + label)
    except Exception as e:
        print("FAIL " + label + ": " + str(e))
        ok = False


def megatron():
    from megatron.core.transformer.transformer_layer import (
        OverlappedFarSkipTransformerLayer,
        SimpleFarSkipTransformerLayer,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    get_gpt_layer_local_spec(use_simple_farskip_layer=True)


# --- Training side ---
check("megatron farskip layers", megatron)

# --- Python deps ---
deps = [
    "wandb", "omegaconf", "pylatexenc", "langdetect", "immutabledict",
    "nltk", "emoji", "syllapy", "pydantic_settings",
]
missing = [m for m in deps if not importlib.util.find_spec(m)]
if missing:
    print("FAIL python deps: missing " + ", ".join(missing))
    ok = False
else:
    print("OK   python deps")

sys.exit(0 if ok else 1)
