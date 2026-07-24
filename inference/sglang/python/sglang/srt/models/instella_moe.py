###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.utils import get_bool_env_var


def _rank0_print(msg: str) -> None:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(f"[InstellaMoEForCausalLM] {msg}", flush=True)


class InstellaMoEForCausalLM(DeepseekV2ForCausalLM):
    """Instella-MoE architecture incorporating Gated MLA Attention and FarSkip-Collective in overlay"""

    def __init__(self, *args, **kwargs):
        assert get_bool_env_var("FARSKIP_OVERLAPPED_DECODER_LAYER", "0") or get_bool_env_var(
            "FARSKIP_REFERENCE_DECODER_LAYER", "0"
        ), "InstellaMoEForCausalLM requires one of FARSKIP_OVERLAPPED_DECODER_LAYER=1 or FARSKIP_REFERENCE_DECODER_LAYER=1"
        _rank0_print(
            "resolved InstellaMoEForCausalLM\n"
            "initializing model"
        )
        super().__init__(*args, **kwargs)


EntryClass = InstellaMoEForCausalLM
