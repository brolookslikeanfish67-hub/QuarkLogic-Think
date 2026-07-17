###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

"""Pre-training with cross-document attention masking via cu_seqlens.

Drop-in replacement for pretrain.py that adds proper document masking
using packed_seq_params (cu_seqlens) when reset_position_ids is enabled.
When reset_position_ids is False (default), this behaves identically to
the standard pretrain.py.

How it works:
    1. GPTDataset (Megatron) packs multiple documents into each sequence,
       separated by EOD tokens. When reset_position_ids=true, position IDs
       restart at 0 at each EOD boundary.
    2. This script detects those position resets and builds cu_seqlens
       (cumulative sequence lengths) marking document boundaries.
    3. cu_seqlens is passed as packed_seq_params to the model, which flows
       to Flash Attention's varlen kernel for per-document causal masking.
    4. Result: tokens in different documents CANNOT attend to each other.

Requirements:
    - context_parallel_size: 1   (cu_seqlens cannot cross CP rank boundaries)
    - reset_position_ids: true   (enables document boundary detection)
    - micro_batch_size: 1        (Flash Attention varlen requires mbs=1)
    - create_attention_mask_in_dataloader: false  (saves ~4.3GB at 64k seq_length)

Usage:
    Use run_instella.sh with --task docmask:
        EXP=examples/megatron/configs/long-context/your_config.yaml \
        bash ./examples/run_instella.sh --task docmask

    Or run directly (for testing):
        EXP=examples/megatron/configs/long-context/instella_16B_long_64k_docmask.yaml
        torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 \\
            --master_addr localhost --master_port 1234 \\
            examples/megatron/pretrain_docmask.py --exp $EXP
"""

import os
import sys
from functools import partial

# Make `primus` and the vendored Megatron-LM importable before the megatron
# imports below, so this entrypoint works whether launched via run_instella.sh
# or torchrun'd directly. setup_backend_path resolves third_party/Megatron-LM
# (and honors BACKEND_PATH / --backend_path).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from primus.pretrain import setup_backend_path

setup_backend_path("megatron", verbose=False)

import torch
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers
from megatron.training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank

from primus.core.launcher.initialize import log_init
from primus.core.launcher.parser import parse_args
from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

stimer = StragglerDetector()


def _build_cu_seqlens_from_position_ids(position_ids: torch.Tensor):
    """Build cu_seqlens from position IDs that reset to 0 at document boundaries.

    Args:
        position_ids: Tensor of shape [batch=1, seq_length] with position IDs
            that reset to 0 at each document boundary (EOD token).

    Returns:
        Tuple of (packed_seq_params, metrics_dict) or (None, {}) if construction fails.
    """
    pos = position_ids[0]  # shape: [seq_length]
    seq_length = pos.shape[0]

    start_indices = (pos == 0).nonzero(as_tuple=True)[0]
    if start_indices.numel() == 0:
        return None, {}

    if start_indices[0] != 0:
        start_indices = torch.cat([torch.zeros(1, device=pos.device, dtype=start_indices.dtype), start_indices])

    doc_lengths = torch.empty(start_indices.shape[0], device=pos.device, dtype=torch.int32)
    doc_lengths[:-1] = start_indices[1:] - start_indices[:-1]
    doc_lengths[-1] = seq_length - start_indices[-1]

    cu_seqlens = torch.zeros(
        start_indices.shape[0] + 1,
        device=pos.device,
        dtype=torch.int32,
    )
    torch.cumsum(doc_lengths, dim=0, out=cu_seqlens[1:])

    assert cu_seqlens[-1].item() == seq_length, (
        f"cu_seqlens[-1]={cu_seqlens[-1].item()} != seq_length={seq_length}"
    )
    assert (doc_lengths > 0).all(), (
        f"Found zero-length documents: doc_lengths={doc_lengths.tolist()}"
    )

    max_seqlen = doc_lengths.max().item()
    num_docs = doc_lengths.shape[0]

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )

    metrics = {
        "num_docs": num_docs,
        "avg_doc_len": doc_lengths.float().mean().item(),
        "min_doc_len": doc_lengths.min().item(),
        "max_doc_len": max_seqlen,
        "seq_length": seq_length,
    }

    return packed_seq_params, metrics


class MegatronDocMaskTrainer(MegatronPretrainTrainer):
    """Pre-training trainer with cross-document attention masking.

    Extends MegatronPretrainTrainer to build packed_seq_params (cu_seqlens)
    from document boundaries detected via position ID resets at EOD tokens.
    Flash Attention's varlen kernel uses cu_seqlens to apply per-document
    causal masking, preventing tokens in different documents from attending
    to each other.

    When reset_position_ids is False, this behaves identically to
    MegatronPretrainTrainer (packed_seq_params stays None).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._packed_seq_params = None
        self._docmask_metrics = {}
        self._docmask_step = 0

    def get_batch(self, data_iterator):
        """Generate a batch and build packed_seq_params for document masking."""

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            self._packed_seq_params = None
            self._docmask_metrics = {}
            return None, None, None, None, None

        batch = get_batch_on_this_tp_rank(data_iterator)

        self._packed_seq_params = None
        self._docmask_metrics = {}
        args = get_args()

        if args.reset_position_ids:
            position_ids = batch.get("position_ids", None)
            if position_ids is not None:
                assert args.context_parallel_size == 1, \
                    "Document masking via cu_seqlens requires context_parallel_size=1"
                assert position_ids.shape[0] == 1, \
                    f"Document masking requires micro_batch_size=1, got batch dim {position_ids.shape[0]}"

                self._packed_seq_params, self._docmask_metrics = (
                    _build_cu_seqlens_from_position_ids(position_ids)
                )

                self._docmask_step += 1
                is_rank0 = torch.distributed.get_rank() == 0
                if is_rank0 and (self._docmask_step <= 10 or self._docmask_step % 100 == 0):
                    m = self._docmask_metrics
                    if m:
                        cu = self._packed_seq_params.cu_seqlens_q
                        print(
                            f"[docmask] step {self._docmask_step}: "
                            f"num_docs={m['num_docs']}, "
                            f"avg_doc_len={m['avg_doc_len']:.0f}, "
                            f"min_doc_len={m['min_doc_len']}, "
                            f"max_doc_len={m['max_doc_len']}, "
                            f"seq_length={m['seq_length']}, "
                            f"cu_seqlens_shape={cu.shape}, "
                            f"cu_seqlens_dtype={cu.dtype}",
                            flush=True,
                        )

        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    def loss_func(self, loss_mask, output_tensor):
        """Loss function with document masking metrics for wandb logging."""
        loss_val, local_num_tokens, reporting_dict = super().loss_func(
            loss_mask, output_tensor
        )

        if self._docmask_metrics:
            device = loss_val.device
            m = self._docmask_metrics
            reporting_dict["docmask/num_documents"] = torch.tensor(
                [float(m["num_docs"])], dtype=torch.float, device=device
            )
            reporting_dict["docmask/avg_doc_length"] = torch.tensor(
                [m["avg_doc_len"]], dtype=torch.float, device=device
            )
            reporting_dict["docmask/min_doc_length"] = torch.tensor(
                [float(m["min_doc_len"])], dtype=torch.float, device=device
            )
            reporting_dict["docmask/max_doc_length"] = torch.tensor(
                [float(m["max_doc_len"])], dtype=torch.float, device=device
            )
            reporting_dict["docmask/active"] = torch.tensor(
                [1.0], dtype=torch.float, device=device
            )
        else:
            device = loss_val.device
            reporting_dict["docmask/active"] = torch.tensor(
                [0.0], dtype=torch.float, device=device
            )

        return loss_val, local_num_tokens, reporting_dict

    def forward_step(self, data_iterator, model: GPTModel):
        """Forward step with document masking via packed_seq_params."""
        get_args()
        timers = get_timers()

        timers("batch-generator", log_level=2).start()
        global stimer
        with stimer(bdata=True):
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
                data_iterator
            )
        timers("batch-generator").stop()

        with stimer:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                packed_seq_params=self._packed_seq_params,
            )

        return output_tensor, partial(self.loss_func, loss_mask)


if __name__ == "__main__":
    primus_cfg = parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = int(os.getenv("MASTER_PORT"))

    if rank == 0:
        print("[pretrain_docmask.py] Using MegatronDocMaskTrainer "
              "(cu_seqlens document masking when reset_position_ids=true)")

    trainer = MegatronDocMaskTrainer(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
    )

    if rank == 0:
        log_init(primus_cfg, trainer.platform)

    trainer.init()
    trainer.run()
