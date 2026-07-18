###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

"""DPO (Direct Preference Optimization) utilities for Megatron models.

Contains:
  - Distributed log-prob extraction
  - DPO loss function (Bradley-Terry preference loss)
  - DPODataset: PyTorch Dataset for preference pairs
  - Offline preprocessing: convert DPO JSONL into Megatron IndexedDataset
"""

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers

# Ensure Megatron-LM is on the path
_script_dir = Path(__file__).resolve().parent
_third_party = _script_dir.parent.parent.parent / "third_party" / "Megatron-LM"
if _third_party.exists() and str(_third_party) not in sys.path:
    sys.path.insert(0, str(_third_party))

from utils.sft import (
    IGNORE_INDEX,
    _build_tokens_and_loss_mask,
    _compute_shard_ranges,
    _iter_jsonl_shard,
    _merge_indexed_shards,
    indexed_files_exist,
)

DEFAULT_PREPROCESS_WORKERS = 64


def _merge_float_indexed_shards(
    final_prefix: str, part_prefixes: list, cleanup: bool = True
) -> None:
    """Merge IndexedDataset shards stored as float32 (for ref_logprobs)."""
    import numpy
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    builder = IndexedDatasetBuilder(final_prefix + ".bin", dtype=numpy.float32)
    for part_prefix in part_prefixes:
        if indexed_files_exist(part_prefix):
            builder.add_index(part_prefix)
    builder.finalize(final_prefix + ".idx")

    if cleanup:
        for part_prefix in part_prefixes:
            for suffix in (".bin", ".idx"):
                path = part_prefix + suffix
                if os.path.exists(path):
                    os.remove(path)


###############################################################################
# Section A: Distributed Log-Prob Extraction
###############################################################################


# One-shot announce flag for the ROCm chunked-gather workaround (list so it's
# mutable from inside the DistributedLogprob.forward staticmethod).
_CHUNKED_GATHER_ANNOUNCED = [False]


@torch.no_grad()
def _compute_distributed_log_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Compute a stable distributed log softmax across tensor parallel workers.

    Two all-reduce operations:
      1. MAX — global max for numerical stability
      2. SUM — sum of exp(logits) for the softmax denominator

    Args:
        vocab_parallel_logits: [B, S, V // TP] — TP-sharded logits.
        group: TP process group for the all-reduce operations.

    Returns:
        Log softmax with the same shape, normalized across the full vocabulary.
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


class DistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    Forward pass (3 all-reduces):
      1. _compute_distributed_log_softmax (MAX + SUM)
      2. Gather target token's log-prob (SUM — only 1 TP rank has it)

    Backward pass:
      grad_logits = (one_hot(target) - softmax) * grad_output
      This is the standard log-softmax Jacobian, required for gradient flow
      from the DPO loss back through the log-probs into the model.
    """

    @staticmethod
    def forward(
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        vocab_parallel_logits = vocab_parallel_logits.to(dtype=torch.float32)

        log_probs = _compute_distributed_log_softmax(vocab_parallel_logits, group=group)
        softmax_output = log_probs.exp()

        # Gather target-token log-probs. Default: single torch.gather over
        # [B, S, V]. Opt-in per-batch gather (DPO_GATHER_CHUNK_BY_BATCH=1)
        # dodges AMD ROCm's int32 vectorized_gather overflow at large B*S*V.
        if os.environ.get("DPO_GATHER_CHUNK_BY_BATCH") == "1":
            if not _CHUNKED_GATHER_ANNOUNCED[0]:
                try:
                    from megatron.training.utils import print_rank_0
                    print_rank_0(
                        "> DistributedLogprob: DPO_GATHER_CHUNK_BY_BATCH=1 "
                        "active — per-batch gather to dodge AMD ROCm "
                        "vectorized_gather int32 overflow "
                    )
                except Exception:
                    pass
                _CHUNKED_GATHER_ANNOUNCED[0] = True

            gathered = [
                torch.gather(
                    log_probs[b:b + 1], -1,
                    masked_target[b:b + 1].unsqueeze(-1),
                ).squeeze(-1)
                for b in range(log_probs.shape[0])
            ]
            log_probs = torch.cat(gathered, dim=0)
        else:
            log_probs = torch.gather(
                log_probs, -1, masked_target.unsqueeze(-1)
            ).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        if not inference_only:
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple:
        grad_output = grad_outputs[0]
        softmax, target_mask, masked_target = ctx.saved_tensors

        if softmax.ndim == 3:
            B, S, V = softmax.shape

            row = (
                torch.arange(B, device=softmax.device)
                .view(-1, 1)
                .expand(-1, S)
                .reshape(-1)
            )
            col = torch.arange(S, device=softmax.device).expand(B, -1).reshape(-1)
            flat_idx = (row * S + col) * V

            flat_chosen = flat_idx.masked_select(
                ~target_mask.reshape(-1)
            ) + masked_target.masked_select(~target_mask)

            grad_input = softmax.neg()
            grad_input = grad_input.mul_(grad_output.unsqueeze(-1))

            grad_output_selected = grad_output.masked_select(~target_mask)
            grad_input.view(-1).scatter_add_(0, flat_chosen, grad_output_selected)
        else:
            V = softmax.size(-1)
            is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(
                masked_target, num_classes=V
            )
            grad_input = is_chosen.float().sub_(softmax)
            grad_input.mul_(grad_output.unsqueeze(-1))

        return grad_input, None, None, None, None, None, None


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
) -> torch.Tensor:
    """Get log probabilities from TP-sharded vocab logits.

    Performs target shift internally via roll(-1). Callers must NOT shift
    targets externally.

    Args:
        vocab_parallel_logits: [B, S, V // TP] — TP-sharded logits from model forward.
        target: [B, S] — unmodified token IDs (shifting done internally).
        vocab_start_index: global vocab start for this TP rank.
        vocab_end_index: global vocab end for this TP rank.
        tp_group: tensor-parallel process group.
        inference_only: if True, don't save tensors for backward.

    Returns:
        [B, S-1] per-token log-probs (shifted for next-token prediction).
    """
    target = target.roll(shifts=-1, dims=-1)

    logprobs: torch.Tensor = DistributedLogprob.apply(
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        tp_group,
        inference_only,
    ).contiguous()

    return logprobs[:, :-1]


###############################################################################
# Section B: DPO Loss
###############################################################################


def masked_mean(tensor, mask, dim=None, global_normalization_factor=None):
    """Compute masked mean, optionally with a global normalization factor for DP."""
    masked = tensor * mask
    if dim is not None:
        return masked.sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)
    if global_normalization_factor is not None:
        return masked.sum() / global_normalization_factor.clamp(min=1)
    return masked.sum() / mask.sum().clamp(min=1)


class DPOLossFn:
    """DPO loss function.

    Implements:
        L(theta) = w_p * L_pref(theta)

    where:
        L_pref = -E[log sigma(beta * (r_chosen - r_rejected))]
        r = sum_t (log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t))
    """

    def __init__(
        self,
        beta: float = 0.05,
        preference_loss_weight: float = 1.0,
        preference_average_log_probs: bool = False,
    ):
        self.beta = float(beta)
        self.preference_loss_weight = float(preference_loss_weight)
        self.preference_average_log_probs = preference_average_log_probs

    @staticmethod
    def split_output_tensor(tensor):
        """Split interleaved chosen/rejected: chosen=[::2], rejected=[1::2]."""
        return tensor[::2], tensor[1::2]

    def _preference_loss(self, rewards, sample_mask, global_valid_seqs, beta=1.0):
        """Bradley-Terry preference loss."""
        rewards_chosen, rewards_rejected = self.split_output_tensor(rewards)
        rewards_delta = rewards_chosen - rewards_rejected

        per_sample_loss = (
            -torch.nn.functional.logsigmoid(beta * rewards_delta) * sample_mask[::2]
        )

        # Return the LOCAL SUM (not pre-divided by global DP pairs): the
        # schedule divides by num_tokens and DDP's 1/D all-reduce yields the
        # global mean, matching standard CE. The masked_mean metrics below are
        # reporting-only, so they keep the global_normalization_factor.
        preference_loss = (per_sample_loss * sample_mask[::2]).sum()
        accuracy = masked_mean(
            (rewards_chosen > rewards_rejected).float(),
            sample_mask[::2],
            global_normalization_factor=global_valid_seqs / 2,
        )
        rewards_chosen_mean = masked_mean(
            rewards_chosen,
            sample_mask[::2],
            global_normalization_factor=global_valid_seqs / 2,
        )
        rewards_rejected_mean = masked_mean(
            rewards_rejected,
            sample_mask[1::2],
            global_normalization_factor=global_valid_seqs / 2,
        )
        return preference_loss, accuracy, rewards_chosen_mean, rewards_rejected_mean

    def _dpo_loss(
        self, next_token_logits, tokens, token_mask, sample_mask,
        ref_logprobs, global_valid_seqs, tp_rank, tp_group,
    ):
        """Core DPO math.

        Uses the Megatron TP code path for log-prob extraction.
        """
        next_token_logits = next_token_logits.to(torch.float32)

        vocab_size_per_partition = next_token_logits.shape[-1]
        token_logprobs = from_parallel_logits_to_logprobs(
            next_token_logits,
            tokens,
            vocab_start_index=tp_rank * vocab_size_per_partition,
            vocab_end_index=(tp_rank + 1) * vocab_size_per_partition,
            tp_group=tp_group,
            inference_only=False,
        )

        ref_logprobs = ref_logprobs[:, :-1]

        diff = (token_logprobs - ref_logprobs) * token_mask

        rewards = diff.sum(-1)
        if self.preference_average_log_probs:
            rewards = rewards / token_mask.sum(-1).clamp(min=1)

        pref_result = self._preference_loss(
            rewards, sample_mask, global_valid_seqs, self.beta
        )
        return pref_result

    def __call__(
        self, next_token_logits, tokens, token_mask, sample_mask,
        ref_logprobs, global_valid_seqs,
        tp_rank, tp_group,
    ):
        """Full DPO loss."""
        (
            preference_loss, accuracy, rewards_chosen_mean, rewards_rejected_mean,
        ) = self._dpo_loss(
            next_token_logits, tokens, token_mask, sample_mask,
            ref_logprobs, global_valid_seqs, tp_rank, tp_group,
        )

        dpo_loss = self.preference_loss_weight * preference_loss
        num_valid_samples = sample_mask.sum() / 2

        # Scale raw rewards by beta for reporting only (implicit-reward units,
        # matching open-instruct/TRL). Loss math is unchanged — beta still
        # appears once inside logsigmoid in _preference_loss.
        return dpo_loss, {
            "loss": dpo_loss.item(),
            "preference_loss": preference_loss.item(),
            "accuracy": accuracy.item(),
            "rewards_chosen_mean": (self.beta * rewards_chosen_mean).item(),
            "rewards_rejected_mean": (self.beta * rewards_rejected_mean).item(),
            "num_valid_samples": num_valid_samples.item(),
        }


###############################################################################
# Section C: Dataset and Preprocessing (DPODataset modeled on SFTDataset)
###############################################################################


def dpo_indexed_prefixes(data_path: str, seq_length: int) -> Tuple[str, str, str]:
    """Return (tokens_prefix, labels_prefix, meta_path) for a DPO JSONL."""
    base, _ = os.path.splitext(data_path)
    tokens_prefix = f"{base}.dpo_tokens.seqlen{seq_length}"
    labels_prefix = f"{base}.dpo_labels.seqlen{seq_length}"
    meta_path = f"{base}.dpo.meta.seqlen{seq_length}.json"
    return tokens_prefix, labels_prefix, meta_path


class DPODataset(torch.utils.data.Dataset):
    """Preference pair dataset for DPO.

    Stores tokens, labels, and ref_logprobs in IndexedDataset files.
    Data is pre-interleaved: even indices = chosen, odd indices = rejected.
    No packing — each sample is one sequence padded to seq_length.
    """

    def __init__(
        self,
        tokens_prefix: str,
        labels_prefix: str,
        ref_logprobs_prefix: str,
        seq_length: int,
        num_samples: int,
        meta_path: Optional[str] = None,
        skip_ref_logprobs: bool = False,
    ):
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        self.seq_length = seq_length
        self.num_samples = num_samples
        self._skip_ref_logprobs = skip_ref_logprobs

        if not indexed_files_exist(tokens_prefix):
            raise FileNotFoundError(f"DPO tokens IndexedDataset not found at {tokens_prefix}.bin/.idx")
        if not indexed_files_exist(labels_prefix):
            raise FileNotFoundError(f"DPO labels IndexedDataset not found at {labels_prefix}.bin/.idx")

        self.tokens_dataset = IndexedDataset(tokens_prefix)
        self.labels_dataset = IndexedDataset(labels_prefix)

        if skip_ref_logprobs:
            self.ref_logprobs_dataset = None
        else:
            if not indexed_files_exist(ref_logprobs_prefix):
                raise FileNotFoundError(f"DPO ref_logprobs IndexedDataset not found at {ref_logprobs_prefix}.bin/.idx. Pre-compute reference log-probs offline first (set compute_ref_logprobs: true in dpo_config).")
            self.ref_logprobs_dataset = IndexedDataset(ref_logprobs_prefix)

        self._num_raw_samples = len(self.tokens_dataset)

        assert len(self.tokens_dataset) == len(self.labels_dataset), \
            f"Tokens and labels sizes don't match: {len(self.tokens_dataset)} vs {len(self.labels_dataset)}"
        if not skip_ref_logprobs:
            assert len(self.tokens_dataset) == len(self.ref_logprobs_dataset), \
                f"Tokens and ref_logprobs sizes don't match: {len(self.tokens_dataset)} vs {len(self.ref_logprobs_dataset)}"
        assert self._num_raw_samples % 2 == 0, \
            f"DPO dataset must have even number of samples (chosen/rejected pairs), got {self._num_raw_samples}"

        self._cached_position_ids = torch.arange(seq_length, dtype=torch.long)

        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print(f"> DPODataset: {self._num_raw_samples} raw samples "
                  f"({self._num_raw_samples // 2} pairs), "
                  f"meta reports {meta.get('num_samples')} samples "
                  f"from {meta.get('num_pairs')} pairs")
        else:
            print(f"> DPODataset: {self._num_raw_samples} raw samples "
                  f"({self._num_raw_samples // 2} pairs) from {tokens_prefix}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.

        Returns dict with keys:
            tokens:       (seq_length,) long
            labels:       (seq_length,) long
            loss_mask:    (seq_length,) float
            position_ids: (seq_length,) long
            ref_logprobs: (seq_length,) float
        """
        idx = idx % self._num_raw_samples

        tokens = torch.tensor(self.tokens_dataset[idx], dtype=torch.long)
        labels = torch.tensor(self.labels_dataset[idx], dtype=torch.long)
        if self._skip_ref_logprobs:
            ref_logprobs = torch.zeros(self.seq_length, dtype=torch.float)
        else:
            ref_logprobs = torch.tensor(
                self.ref_logprobs_dataset[idx], dtype=torch.float
            )

        loss_mask = (labels != IGNORE_INDEX).float()
        position_ids = self._cached_position_ids

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "ref_logprobs": ref_logprobs,
        }


###############################################################################
# DPO Preprocessing: JSONL → IndexedDataset
###############################################################################


def _dpo_preprocess_shard_worker(args: Tuple) -> Tuple[str, str, int, int]:
    """Process one shard of the DPO JSONL file.

    For each preference pair, tokenizes chosen and rejected messages,
    builds loss masks, shifts labels, pads to seq_length, and writes
    them interleaved: [chosen_0, rejected_0, chosen_1, rejected_1, ...].
    """
    (
        data_path,
        seq_length,
        shard_idx,
        shard_range,
        tokens_part_prefix,
        labels_part_prefix,
        tokenizer_model,
        trust_remote_code,
    ) = args

    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id or pad_token_id

    tokens_builder = IndexedDatasetBuilder(tokens_part_prefix + ".bin")
    labels_builder = IndexedDatasetBuilder(labels_part_prefix + ".bin")

    start, end = shard_range
    num_samples = 0
    num_pairs = 0

    def write_sample(tokens: List[int], labels: List[int]):
        nonlocal num_samples
        current_len = len(tokens)
        if current_len < seq_length:
            pad_len = seq_length - current_len
            tokens = tokens + [pad_token_id] * pad_len
            labels = labels + [IGNORE_INDEX] * pad_len
        tokens = tokens[:seq_length]
        labels = labels[:seq_length]
        tokens_builder.add_item(torch.tensor(tokens, dtype=torch.long))
        tokens_builder.end_document()
        labels_builder.add_item(torch.tensor(labels, dtype=torch.long))
        labels_builder.end_document()
        num_samples += 1

    def process_messages(messages: List[Dict]) -> Tuple[List[int], List[int]]:
        """Tokenize messages and build shifted labels with loss mask."""
        tokens, loss_mask = _build_tokens_and_loss_mask(
            tokenizer, messages, seq_length
        )
        labels = tokens[1:] + [eos_token_id]
        last_is_assistant = messages[-1].get("role", "user") == "assistant"
        loss_mask = loss_mask[1:] + [1 if last_is_assistant else 0]
        labels = [
            (l if loss_mask[i] else IGNORE_INDEX) for i, l in enumerate(labels)
        ]
        return tokens, labels

    for sample in _iter_jsonl_shard(data_path, start, end):
        num_pairs += 1

        chosen_messages = sample.get("chosen")
        rejected_messages = sample.get("rejected")
        if chosen_messages is None or rejected_messages is None:
            raise ValueError(f"DPO sample must have 'chosen' and 'rejected' keys: {list(sample.keys())}")

        chosen_tokens, chosen_labels = process_messages(chosen_messages)
        rejected_tokens, rejected_labels = process_messages(rejected_messages)

        write_sample(chosen_tokens, chosen_labels)
        write_sample(rejected_tokens, rejected_labels)

        if num_pairs % 1000 == 0 and num_pairs > 0:
            print(
                f"[shard {shard_idx}] processed {num_pairs} pairs, "
                f"{num_samples} samples"
            )

    tokens_builder.finalize(tokens_part_prefix + ".idx")
    labels_builder.finalize(labels_part_prefix + ".idx")

    return tokens_part_prefix, labels_part_prefix, num_samples, num_pairs


def preprocess_dpo_to_indexed(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    output_prefix: Optional[str] = None,
    num_workers: int = DEFAULT_PREPROCESS_WORKERS,
) -> Tuple[str, str, str]:
    """Preprocess DPO JSONL into IndexedDataset (tokens + labels).

    Each JSONL line must have 'chosen' and 'rejected' keys, each containing
    a list of message dicts suitable for apply_chat_template().

    Output is interleaved: [chosen_0, rejected_0, chosen_1, rejected_1, ...].

    Args:
        data_path: Path to JSONL file with preference pairs.
        tokenizer: HuggingFace tokenizer with chat template.
        seq_length: Maximum sequence length (all samples padded to this).
        output_prefix: Output file prefix. If None, derived from data_path.
        num_workers: Number of parallel preprocessing workers.

    Returns:
        (tokens_prefix, labels_prefix, meta_path)
    """
    if output_prefix is not None:
        tokens_prefix = f"{output_prefix}.dpo_tokens.seqlen{seq_length}"
        labels_prefix = f"{output_prefix}.dpo_labels.seqlen{seq_length}"
        meta_path = f"{output_prefix}.dpo.meta.seqlen{seq_length}.json"
    else:
        tokens_prefix, labels_prefix, meta_path = dpo_indexed_prefixes(
            data_path, seq_length
        )

    if not data_path.endswith(".jsonl"):
        raise ValueError(f"Only .jsonl files are supported, got: {data_path}")

    file_size = os.path.getsize(data_path)
    num_workers = max(1, min(num_workers, 64))

    if file_size == 0:
        raise ValueError(f"Input file is empty: {data_path}")

    tokenizer_model = getattr(tokenizer, "name_or_path", None)
    if tokenizer_model is None:
        raise ValueError("Tokenizer is missing name_or_path; cannot recreate in worker.")
    trust_remote_code = bool(getattr(tokenizer, "trust_remote_code", False))

    shard_ranges = _compute_shard_ranges(file_size, num_workers)

    part_tokens_prefixes = []
    part_labels_prefixes = []
    worker_args = []
    for shard_idx, shard_range in enumerate(shard_ranges):
        tokens_part_prefix = f"{tokens_prefix}.part-{shard_idx:03d}"
        labels_part_prefix = f"{labels_prefix}.part-{shard_idx:03d}"
        part_tokens_prefixes.append(tokens_part_prefix)
        part_labels_prefixes.append(labels_part_prefix)
        worker_args.append((
            data_path,
            seq_length,
            shard_idx,
            shard_range,
            tokens_part_prefix,
            labels_part_prefix,
            tokenizer_model,
            trust_remote_code,
        ))

    print(f"Preprocessing DPO {data_path} with {len(worker_args)} workers...")
    print(f"  seq_length={seq_length}")
    print(f"  output: {tokens_prefix}, {labels_prefix}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_dpo_preprocess_shard_worker, worker_args)

    total_samples = sum(r[2] for r in results)
    total_pairs = sum(r[3] for r in results)

    print(f"Merging {len(results)} shards...")
    _merge_indexed_shards(tokens_prefix, part_tokens_prefixes, cleanup=True)
    _merge_indexed_shards(labels_prefix, part_labels_prefixes, cleanup=True)

    meta = {
        "seq_length": seq_length,
        "num_pairs": total_pairs,
        "num_samples": total_samples,
        "tokenizer_model": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "trust_remote_code": getattr(tokenizer, "trust_remote_code", None),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done! {total_pairs} pairs -> {total_samples} samples (interleaved)")
    print(f"  tokens: {tokens_prefix}.bin/.idx")
    print(f"  labels: {labels_prefix}.bin/.idx")
    print(f"  meta:   {meta_path}")

    return tokens_prefix, labels_prefix, meta_path


###############################################################################
# Automatic preprocessing (rank 0 preprocesses, others poll-wait)
###############################################################################


def maybe_preprocess_dpo_data(dpo_config: Dict) -> None:
    """Preprocess raw DPO JSONL data into indexed datasets if not already done.

    Same pattern as maybe_preprocess_sft_data() in sft.py:
    rank 0 preprocesses, all other ranks poll-wait for the files to appear.
    After preprocessing, injects resolved prefixes into dpo_config.

    Prerequisites:
        * Megatron must already be initialised (get_args() available).
        * dpo_config must contain 'train_data_path' (JSONL path).
        * If explicit 'dpo_tokens_prefix' / 'dpo_labels_prefix' are already
          set, this function is a no-op.

    Side-effects:
        * Writes .bin / .idx / .meta.json files next to the JSONL.
        * Mutates dpo_config in-place to inject resolved prefixes.
    """
    from megatron.training import get_args
    from megatron.training.utils import print_rank_0

    args = get_args()
    rank = int(os.getenv("RANK", "0"))

    if dpo_config.get("dpo_tokens_prefix") and dpo_config.get("dpo_labels_prefix"):
        print_rank_0("> DPO: explicit prefixes provided, skipping auto-preprocessing.")
        return

    data_path = dpo_config.get("train_data_path")
    if not data_path:
        print_rank_0("> DPO: no train_data_path, skipping auto-preprocessing.")
        return

    seq_length = args.seq_length
    num_workers = dpo_config.get("num_workers", DEFAULT_PREPROCESS_WORKERS)

    tokens_prefix, labels_prefix, meta_path = dpo_indexed_prefixes(
        data_path, seq_length
    )

    needs_preprocess = not (
        indexed_files_exist(tokens_prefix)
        and indexed_files_exist(labels_prefix)
        and os.path.exists(meta_path)
    )

    if needs_preprocess:
        if rank == 0:
            print_rank_0(f"> Preprocessing DPO data: {data_path}")

            tokenizer_model = args.tokenizer_model
            trust_remote_code = getattr(args, "trust_remote_code", False)
            if tokenizer_model is None:
                raise ValueError("tokenizer_model must be set in the Megatron config for DPO preprocessing.")
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer_model, trust_remote_code=trust_remote_code,
            )

            preprocess_dpo_to_indexed(
                data_path=data_path,
                tokenizer=tokenizer,
                seq_length=seq_length,
                output_prefix=None,
                num_workers=num_workers,
            )
            print_rank_0(f"> DPO preprocessing complete: {data_path}")
        else:
            print_rank_0(
                f"> Rank {rank}: waiting for rank 0 to finish DPO preprocessing: "
                f"{data_path}"
            )
            while True:
                if (
                    indexed_files_exist(tokens_prefix)
                    and indexed_files_exist(labels_prefix)
                    and os.path.exists(meta_path)
                ):
                    break
                time.sleep(10)
    else:
        print_rank_0(
            f"> DPO preprocessed data found for {data_path}, skipping preprocessing."
        )

    dpo_config["dpo_tokens_prefix"] = tokens_prefix
    dpo_config["dpo_labels_prefix"] = labels_prefix
    dpo_config["dpo_meta_path"] = meta_path

    print_rank_0("> DPO auto-preprocessing done.")
