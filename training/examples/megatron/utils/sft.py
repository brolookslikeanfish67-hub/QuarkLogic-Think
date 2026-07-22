###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

"""Supervised Fine-Tuning (SFT) utilities for Megatron models.

Contains:
  - Offline preprocessing: convert SFT JSONL into Megatron IndexedDataset
  - SFTDataset: PyTorch Dataset that reads preprocessed indexed datasets

Preprocessing CLI usage:
    python examples/megatron/utils/sft.py \
        --input /path/to/conversations.jsonl \
        --tokenizer-model /path/to/hf_tokenizer \
        --seq-length 4096 \
        --output-prefix /path/to/output/sft_data \
        --enable-packing \
        --workers 64
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import transformers

# Ensure Megatron-LM is on the path
_script_dir = Path(__file__).resolve().parent
_third_party = _script_dir.parent.parent.parent / "third_party" / "Megatron-LM"
if _third_party.exists() and str(_third_party) not in sys.path:
    sys.path.insert(0, str(_third_party))

IGNORE_INDEX = -100
DEFAULT_PREPROCESS_WORKERS = 64


###############################################################################
# Utility helpers
###############################################################################

def indexed_files_exist(prefix: str) -> bool:
    return os.path.exists(prefix + ".bin") and os.path.exists(prefix + ".idx")


def indexed_prefixes_for_json(
    data_path: str,
    seq_length: Optional[int] = None,
    enable_packing: bool = False,
    eod_token_id: Optional[int] = None,
    packing_strategy: str = "greedy",
) -> Tuple[str, str, str]:
    """Return (tokens_prefix, labels_prefix, meta_path) for a JSON/JSONL path.

    When ``eod_token_id`` is set (i.e. a dedicated EOD token distinct from EOS
    is used for document boundaries in packed sequences), the path segment
    changes from ``longcontext`` to ``longcontexteod`` so that new preprocessed
    files don't collide with older data that used EOS as the separator.

    When ``packing_strategy`` is ``"ffd"``, the suffix becomes ``.packed_ffd``
    so FFD-packed files don't collide with greedy-packed files.
    """
    base, _ = os.path.splitext(data_path)
    tag = "longcontexteod" if eod_token_id is not None else "longcontext"
    if not enable_packing:
        pack_suffix = ""
    elif packing_strategy in ("ffd", "ffd_disk"):
        pack_suffix = ".packed_ffd"
    else:
        pack_suffix = ".packed"
    if seq_length is None:
        tokens_prefix = f"{base}.{tag}.sft_tokens{pack_suffix}"
        labels_prefix = f"{base}.{tag}.sft_labels{pack_suffix}"
        meta_path = f"{base}.{tag}.sft{pack_suffix}.meta.json"
    else:
        tokens_prefix = f"{base}.{tag}.sft_tokens.seqlen{seq_length}{pack_suffix}"
        labels_prefix = f"{base}.{tag}.sft_labels.seqlen{seq_length}{pack_suffix}"
        meta_path = f"{base}.{tag}.sft.meta.seqlen{seq_length}{pack_suffix}.json"
    return tokens_prefix, labels_prefix, meta_path


def _resolve_indexed_prefixes(
    data_path: str,
    seq_length: int,
    enable_packing: bool = False,
    eod_token_id: Optional[int] = None,
    packing_strategy: str = "greedy",
) -> Tuple[str, str, str]:
    """Find indexed prefixes, trying seq_length-specific paths first, then generic."""
    tokens_prefix, labels_prefix, meta_path = indexed_prefixes_for_json(
        data_path, seq_length=seq_length, enable_packing=enable_packing,
        eod_token_id=eod_token_id, packing_strategy=packing_strategy,
    )
    if indexed_files_exist(tokens_prefix) and indexed_files_exist(labels_prefix):
        return tokens_prefix, labels_prefix, meta_path
    return indexed_prefixes_for_json(
        data_path, seq_length=None, enable_packing=enable_packing,
        eod_token_id=eod_token_id, packing_strategy=packing_strategy,
    )


def _hash_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


def _build_sft_meta(
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    num_conversations: Optional[int] = None,
    num_samples: Optional[int] = None,
    eod_token_id: Optional[int] = None,
) -> Dict:
    return {
        "seq_length": seq_length,
        "num_conversations": num_conversations,
        "num_samples": num_samples,
        "tokenizer_model": getattr(tokenizer, "name_or_path", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "trust_remote_code": getattr(tokenizer, "trust_remote_code", None),
        "chat_template_hash": _hash_text(getattr(tokenizer, "chat_template", None)),
        "eod_token_id": eod_token_id,
    }


def _load_meta(meta_path: str) -> Optional[Dict]:
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _meta_matches(
    meta: Optional[Dict],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    eod_token_id: Optional[int] = None,
) -> bool:
    if meta is None:
        return False
    current = _build_sft_meta(tokenizer, seq_length, eod_token_id=eod_token_id)
    keys_to_compare = [
        "seq_length",
        "tokenizer_model",
        "tokenizer_class",
        "trust_remote_code",
        "chat_template_hash",
        "eod_token_id",
    ]
    return all(meta.get(k) == current.get(k) for k in keys_to_compare)


###############################################################################
# Message normalization & tokenization
###############################################################################

def _normalize_messages(messages: List[Dict]) -> List[Dict]:
    normalized = []
    for msg in messages:
        normalized_msg = dict(msg)
        content = normalized_msg.get("content")
        if content is not None and not isinstance(content, str):
            normalized_msg["content"] = json.dumps(content, ensure_ascii=False)
        normalized.append(normalized_msg)
    return normalized


def _build_tokens_and_loss_mask(
    tokenizer: transformers.PreTrainedTokenizerBase,
    messages: List[Dict],
    seq_length: int,
) -> Tuple[List[int], List[int]]:
    """Tokenize a conversation and build per-token loss mask.

    Returns:
        tokens: list of token IDs
        loss_mask: list of 0/1 ints (1 = assistant, 0 = user/system)
    """
    tokens: List[int] = []
    loss_mask: List[int] = []
    prev_length = 0

    normalized_messages = _normalize_messages(messages)

    for i, msg in enumerate(normalized_messages):
        prefix_messages = normalized_messages[: i + 1]
        tokens_so_far = tokenizer.apply_chat_template(
            prefix_messages,
            add_generation_prompt=False,
            tokenize=True,
            truncation=True,
            max_length=seq_length,
        )

        turn_start = prev_length
        turn_end = len(tokens_so_far)

        role = msg.get("role", "user")
        if role == "assistant":
            turn_loss_mask = [1] * (turn_end - turn_start)
        else:
            turn_loss_mask = [0] * (turn_end - turn_start)

        tokens = tokens_so_far
        loss_mask.extend(turn_loss_mask)
        prev_length = turn_end

    return tokens, loss_mask


###############################################################################
# Shard-level JSONL iteration
###############################################################################

def _iter_jsonl_shard(data_path: str, start: int, end: int) -> Iterable[Dict]:
    with open(data_path, "rb") as f:
        if start > 0:
            f.seek(start - 1)
            f.readline()
        else:
            f.seek(start)

        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line:
                yield json.loads(line.decode("utf-8"))


def _compute_shard_ranges(file_size: int, num_shards: int) -> List[Tuple[int, int]]:
    shard_size = math.ceil(file_size / num_shards)
    ranges = []
    for i in range(num_shards):
        start = i * shard_size
        end = min(file_size, (i + 1) * shard_size)
        if start >= file_size:
            break
        ranges.append((start, end))
    return ranges


###############################################################################
# Per-shard worker
###############################################################################

def _preprocess_shard_worker(args: Tuple) -> Tuple[str, str, int, int]:
    # Support both 10-element and 11-element arg tuples for backward compat.
    if len(args) == 11:
        (
            data_path, seq_length, shard_idx, shard_range,
            tokens_part_prefix, labels_part_prefix,
            tokenizer_model, trust_remote_code,
            enable_packing, explicit_eod_token_id,
            pad_to_seq_length,
        ) = args
    else:
        (
            data_path, seq_length, shard_idx, shard_range,
            tokens_part_prefix, labels_part_prefix,
            tokenizer_model, trust_remote_code,
            enable_packing, explicit_eod_token_id,
        ) = args
        pad_to_seq_length = True

    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id or pad_token_id

    # EOD marks the sample boundary; it MUST differ from the chat template's
    # per-turn EOS, else position/attn resets fire mid-conversation.
    if explicit_eod_token_id is not None:
        eod_token_id = explicit_eod_token_id
    elif hasattr(tokenizer, "eod_id"):
        eod_token_id = tokenizer.eod_id
    elif hasattr(tokenizer, "eod_token_id"):
        eod_token_id = tokenizer.eod_token_id
    else:
        eod_token_id = eos_token_id

    if shard_idx == 0:
        print(f"[shard 0] eos_token_id={eos_token_id}, eod_token_id={eod_token_id}"
              f"  (distinct={eos_token_id != eod_token_id})")

    tokens_builder = IndexedDatasetBuilder(tokens_part_prefix + ".bin")
    labels_builder = IndexedDatasetBuilder(labels_part_prefix + ".bin")

    start, end = shard_range
    num_samples = 0
    num_conversations = 0

    # Packing accumulator
    packed_tokens: List[int] = []
    packed_labels: List[int] = []

    def write_sample(tokens: List[int], labels: List[int]):
        nonlocal num_samples
        if pad_to_seq_length:
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

    def flush_pack():
        nonlocal packed_tokens, packed_labels
        if packed_tokens:
            write_sample(packed_tokens, packed_labels)
            packed_tokens, packed_labels = [], []

    for sample in _iter_jsonl_shard(data_path, start, end):
        num_conversations += 1
        messages = sample.get("messages") or sample.get("conversations")
        if messages is None:
            raise ValueError(f"Sample must have 'messages' or 'conversations' key: {sample}")

        tokens, loss_mask = _build_tokens_and_loss_mask(tokenizer, messages, seq_length)

        # Shift for next-token prediction
        labels = tokens[1:] + [eos_token_id]
        # Include EOS in loss if last message was assistant
        last_is_assistant = messages[-1].get("role", "user") == "assistant"
        loss_mask = loss_mask[1:] + [1 if last_is_assistant else 0]
        labels = [(l if loss_mask[i] else IGNORE_INDEX) for i, l in enumerate(labels)]

        # Append EOD token at end of conversation (matching pretraining behavior).
        # EOS is used within the conversation (from chat template / label shift),
        # EOD marks the end of the sample as a document boundary marker.
        # Loss is masked at EOD, consistent with eod_mask_loss=true in pretraining.
        tokens.append(eod_token_id)
        labels.append(IGNORE_INDEX)

        conv_len = len(tokens)  # includes trailing EOD

        if not enable_packing:
            write_sample(tokens, labels)
        elif conv_len >= seq_length:
            flush_pack()
            write_sample(tokens, labels)
        else:
            # Each conversation already ends with EOD, so no separate
            # separator is needed -- matching pretraining's document packing.
            if len(packed_tokens) + conv_len > seq_length:
                flush_pack()
            packed_tokens.extend(tokens)
            packed_labels.extend(labels)

        if num_conversations % 1000 == 0 and num_conversations > 0:
            print(f"[shard {shard_idx}] processed {num_conversations} conversations, {num_samples} samples")

    flush_pack()

    tokens_builder.finalize(tokens_part_prefix + ".idx")
    labels_builder.finalize(labels_part_prefix + ".idx")

    return tokens_part_prefix, labels_part_prefix, num_samples, num_conversations


###############################################################################
# Per-shard tokenization worker (for FFD packing)
###############################################################################

def _tokenize_shard_worker(args: Tuple) -> List[Tuple[List[int], List[int]]]:
    """Tokenize conversations in a JSONL shard and return (tokens, labels) pairs.

    Unlike ``_preprocess_shard_worker``, this does NOT pack or write to disk.
    Results are collected by the main process for global bin-packing.
    """
    (
        data_path,
        seq_length,
        shard_idx,
        shard_range,
        tokenizer_model,
        trust_remote_code,
        explicit_eod_token_id,
    ) = args

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
    eos_token_id = tokenizer.eos_token_id or (tokenizer.pad_token_id or 0)

    if explicit_eod_token_id is not None:
        eod_token_id = explicit_eod_token_id
    elif hasattr(tokenizer, "eod_id"):
        eod_token_id = tokenizer.eod_id
    elif hasattr(tokenizer, "eod_token_id"):
        eod_token_id = tokenizer.eod_token_id
    else:
        eod_token_id = eos_token_id

    start, end = shard_range
    results: List[Tuple[List[int], List[int]]] = []
    num_conversations = 0

    for sample in _iter_jsonl_shard(data_path, start, end):
        num_conversations += 1
        messages = sample.get("messages") or sample.get("conversations")
        if messages is None:
            raise ValueError(f"Sample must have 'messages' or 'conversations' key: {sample}")

        tokens, loss_mask = _build_tokens_and_loss_mask(tokenizer, messages, seq_length)

        labels = tokens[1:] + [eos_token_id]
        last_is_assistant = messages[-1].get("role", "user") == "assistant"
        loss_mask = loss_mask[1:] + [1 if last_is_assistant else 0]
        labels = [(l if loss_mask[i] else IGNORE_INDEX) for i, l in enumerate(labels)]

        tokens.append(eod_token_id)
        labels.append(IGNORE_INDEX)

        results.append((tokens, labels))

        if num_conversations % 1000 == 0 and num_conversations > 0:
            print(f"[tokenize shard {shard_idx}] {num_conversations} conversations")

    print(f"[tokenize shard {shard_idx}] done: {num_conversations} conversations")
    return results


###############################################################################
# FFD (First Fit Decreasing) bin-packing
###############################################################################

def _ffd_bin_pack(
    lengths: List[int],
    seq_length: int,
) -> List[List[int]]:
    """Pack items into bins of ``seq_length`` using a two-pointer approach.

    Sorts items by length, then pairs the longest with as many short items
    as fit.  O(n log n) for the sort, O(n) for the pairing.

    Args:
        lengths: Per-item lengths (e.g. token counts per conversation).
        seq_length: Bin capacity.

    Returns a list of bins.  Each bin is a list of indices into ``lengths``.
    """
    n = len(lengths)
    sorted_indices = sorted(range(n), key=lambda i: lengths[i])

    left = 0
    right = n - 1
    bins: List[List[int]] = []

    while left <= right:
        idx = sorted_indices[right]
        cur_bin = [idx]
        remaining = seq_length - lengths[idx]
        right -= 1

        while left <= right and lengths[sorted_indices[left]] <= remaining:
            short_idx = sorted_indices[left]
            cur_bin.append(short_idx)
            remaining -= lengths[short_idx]
            left += 1

        bins.append(cur_bin)

    return bins


###############################################################################
# Merge shards
###############################################################################

def _merge_indexed_shards(
    final_prefix: str, part_prefixes: List[str], cleanup: bool = True
) -> None:
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    builder = IndexedDatasetBuilder(final_prefix + ".bin")
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
# FFD preprocessing orchestrator
###############################################################################

def _preprocess_ffd(
    data_path: str,
    seq_length: int,
    num_workers: int,
    tokenizer_model: str,
    trust_remote_code: bool,
    eod_token_id: Optional[int],
    tokens_prefix: str,
    labels_prefix: str,
) -> Tuple[int, int]:
    """Tokenize all conversations, run FFD bin-packing, write IndexedDataset.

    Returns (total_samples, total_conversations).
    """
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    file_size = os.path.getsize(data_path)
    shard_ranges = _compute_shard_ranges(file_size, num_workers)

    worker_args = []
    for shard_idx, shard_range in enumerate(shard_ranges):
        worker_args.append((
            data_path,
            seq_length,
            shard_idx,
            shard_range,
            tokenizer_model,
            trust_remote_code,
            eod_token_id,
        ))

    print(f"FFD packing: tokenizing {data_path} with {len(worker_args)} workers...")
    print(f"  seq_length={seq_length}")
    print(f"  output: {tokens_prefix}, {labels_prefix}")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        shard_results = pool.map(_tokenize_shard_worker, worker_args)

    conversations: List[Tuple[List[int], List[int]]] = []
    for shard in shard_results:
        conversations.extend(shard)
    del shard_results

    total_conversations = len(conversations)
    total_tokens = sum(len(c[0]) for c in conversations)
    print(f"  tokenized {total_conversations:,} conversations "
          f"({total_tokens:,} tokens) in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"  running FFD bin-packing...")
    conv_lengths = [len(c[0]) for c in conversations]
    bins = _ffd_bin_pack(conv_lengths, seq_length)
    print(f"  packed into {len(bins):,} bins in {time.time()-t1:.1f}s")

    theoretical_min = math.ceil(total_tokens / seq_length)
    utilization = total_tokens / (len(bins) * seq_length) * 100
    print(f"  theoretical minimum bins: {theoretical_min:,}")
    print(f"  utilization: {utilization:.1f}%")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    t2 = time.time()
    print(f"  writing {len(bins):,} packed samples...")
    tokens_builder = IndexedDatasetBuilder(tokens_prefix + ".bin")
    labels_builder = IndexedDatasetBuilder(labels_prefix + ".bin")

    for bin_indices in bins:
        packed_tokens: List[int] = []
        packed_labels: List[int] = []
        for conv_idx in bin_indices:
            packed_tokens.extend(conversations[conv_idx][0])
            packed_labels.extend(conversations[conv_idx][1])

        current_len = len(packed_tokens)
        if current_len < seq_length:
            pad_len = seq_length - current_len
            packed_tokens += [pad_token_id] * pad_len
            packed_labels += [IGNORE_INDEX] * pad_len
        packed_tokens = packed_tokens[:seq_length]
        packed_labels = packed_labels[:seq_length]

        tokens_builder.add_item(torch.tensor(packed_tokens, dtype=torch.long))
        tokens_builder.end_document()
        labels_builder.add_item(torch.tensor(packed_labels, dtype=torch.long))
        labels_builder.end_document()

    tokens_builder.finalize(tokens_prefix + ".idx")
    labels_builder.finalize(labels_prefix + ".idx")

    total_samples = len(bins)
    print(f"  wrote {total_samples:,} samples in {time.time()-t2:.1f}s")

    return total_samples, total_conversations


###############################################################################
# FFD disk-backed preprocessing orchestrator (two-pass, low memory)
###############################################################################

def _preprocess_ffd_disk(
    data_path: str,
    seq_length: int,
    num_workers: int,
    tokenizer_model: str,
    trust_remote_code: bool,
    eod_token_id: Optional[int],
    tokens_prefix: str,
    labels_prefix: str,
) -> Tuple[int, int]:
    """Two-pass FFD packing that avoids holding all data in memory.

    Pass 1: Parallel tokenization via ``_preprocess_shard_worker`` with
    ``enable_packing=False``.  Each worker writes unpacked conversations to
    per-shard IndexedDataset files on disk and returns only scalars via IPC.

    Pass 2: The main process opens the shard files via mmap, reads per-
    conversation lengths from the ``.idx`` headers, computes the global FFD
    bin-packing plan, then streams conversations from disk into the final
    packed IndexedDataset.

    Peak memory is O(num_conversations) ints for the length array plus one
    bin's worth of token data -- orders of magnitude less than the in-memory
    ``_preprocess_ffd`` which holds all conversations in Python lists.

    Returns (total_samples, total_conversations).
    """
    from megatron.core.datasets.indexed_dataset import (
        IndexedDataset,
        IndexedDatasetBuilder,
    )

    file_size = os.path.getsize(data_path)
    shard_ranges = _compute_shard_ranges(file_size, num_workers)

    # ── Pass 1: tokenize and dump unpacked parts to disk ─────────────────
    part_tokens_prefixes: List[str] = []
    part_labels_prefixes: List[str] = []
    worker_args = []
    for shard_idx, shard_range in enumerate(shard_ranges):
        tp = f"{tokens_prefix}.unpacked.part-{shard_idx:03d}"
        lp = f"{labels_prefix}.unpacked.part-{shard_idx:03d}"
        part_tokens_prefixes.append(tp)
        part_labels_prefixes.append(lp)
        worker_args.append((
            data_path,
            seq_length,
            shard_idx,
            shard_range,
            tp,
            lp,
            tokenizer_model,
            trust_remote_code,
            False,  # enable_packing=False: one conversation per sample
            eod_token_id,
            False,  # pad_to_seq_length=False: keep natural lengths for FFD
        ))

    print(f"FFD-disk packing: tokenizing {data_path} with {len(worker_args)} workers...")
    print(f"  seq_length={seq_length}")
    print(f"  output: {tokens_prefix}, {labels_prefix}")

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_preprocess_shard_worker, worker_args)

    total_conversations = sum(r[3] for r in results)
    print(f"  Pass 1 done: {total_conversations:,} conversations "
          f"dumped to {len(results)} shard files in {time.time()-t0:.1f}s")

    # ── Pass 2a: read lengths from .idx files via mmap ───────────────────
    print(f"  Pass 2a: reading conversation lengths from .idx files...")
    t1 = time.time()
    shard_datasets_tok: List[IndexedDataset] = []
    shard_datasets_lab: List[IndexedDataset] = []
    all_lengths: List[int] = []
    conv_to_shard: List[Tuple[int, int]] = []  # (shard_idx, local_idx)

    for shard_idx, (tp, lp) in enumerate(
        zip(part_tokens_prefixes, part_labels_prefixes)
    ):
        if not indexed_files_exist(tp):
            continue
        ds_tok = IndexedDataset(tp)
        ds_lab = IndexedDataset(lp)
        shard_datasets_tok.append(ds_tok)
        shard_datasets_lab.append(ds_lab)
        shard_list_idx = len(shard_datasets_tok) - 1
        for local_idx in range(len(ds_tok)):
            all_lengths.append(int(ds_tok.index.sequence_lengths[local_idx]))
            conv_to_shard.append((shard_list_idx, local_idx))

    total_tokens = sum(all_lengths)
    print(f"  {len(all_lengths):,} conversations, "
          f"{total_tokens:,} tokens read from .idx in {time.time()-t1:.1f}s")

    # ── Pass 2b: FFD bin-packing on lengths only ─────────────────────────
    t2 = time.time()
    print(f"  Pass 2b: sorting and bin-packing {len(all_lengths):,} conversations...")
    bins = _ffd_bin_pack(all_lengths, seq_length)
    print(f"  packed into {len(bins):,} bins in {time.time()-t2:.1f}s")

    theoretical_min = math.ceil(total_tokens / seq_length)
    utilization = total_tokens / (len(bins) * seq_length) * 100
    print(f"  theoretical minimum bins: {theoretical_min:,}")
    print(f"  utilization: {utilization:.1f}%")

    # ── Pass 2c: write packed output by mmap-reading from shard files ────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    t3 = time.time()
    print(f"  Pass 2c: writing {len(bins):,} packed samples from mmap-backed shard files...")
    tokens_builder = IndexedDatasetBuilder(tokens_prefix + ".bin")
    labels_builder = IndexedDatasetBuilder(labels_prefix + ".bin")

    for bin_idx, bin_indices in enumerate(bins):
        packed_tokens: List[int] = []
        packed_labels: List[int] = []
        for global_idx in bin_indices:
            shard_list_idx, local_idx = conv_to_shard[global_idx]
            tok = shard_datasets_tok[shard_list_idx][local_idx]
            lab = shard_datasets_lab[shard_list_idx][local_idx]
            packed_tokens.extend(tok.tolist())
            packed_labels.extend(lab.tolist())

        current_len = len(packed_tokens)
        if current_len < seq_length:
            pad_len = seq_length - current_len
            packed_tokens += [pad_token_id] * pad_len
            packed_labels += [IGNORE_INDEX] * pad_len
        packed_tokens = packed_tokens[:seq_length]
        packed_labels = packed_labels[:seq_length]

        tokens_builder.add_item(torch.tensor(packed_tokens, dtype=torch.long))
        tokens_builder.end_document()
        labels_builder.add_item(torch.tensor(packed_labels, dtype=torch.long))
        labels_builder.end_document()

        if (bin_idx + 1) % 10000 == 0:
            print(f"    {bin_idx + 1:,}/{len(bins):,} bins written")

    tokens_builder.finalize(tokens_prefix + ".idx")
    labels_builder.finalize(labels_prefix + ".idx")

    total_samples = len(bins)
    print(f"  wrote {total_samples:,} packed samples in {time.time()-t3:.1f}s")

    # ── Cleanup unpacked part files ──────────────────────────────────────
    del shard_datasets_tok, shard_datasets_lab
    for prefix in part_tokens_prefixes + part_labels_prefixes:
        for suffix in (".bin", ".idx"):
            path = prefix + suffix
            if os.path.exists(path):
                os.remove(path)
    print(f"  cleaned up {len(part_tokens_prefixes)} shard part files")

    return total_samples, total_conversations


###############################################################################
# Main preprocessing entry point
###############################################################################

def preprocess_sft_to_indexed(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    output_prefix: Optional[str] = None,
    enable_packing: bool = False,
    num_workers: int = DEFAULT_PREPROCESS_WORKERS,
    eod_token_id: Optional[int] = None,
    packing_strategy: str = "greedy",
) -> Tuple[str, str, str]:
    """Preprocess JSONL into IndexedDataset (tokens + labels).

    Args:
        data_path: Path to JSONL file containing conversations
        tokenizer: HuggingFace tokenizer with chat template
        seq_length: Maximum sequence length
        output_prefix: Output file prefix. If None, paths are derived from
                        ``data_path`` via ``indexed_prefixes_for_json``,
                        placing outputs next to the input JSONL (consistent
                        with automatic preprocessing).
        enable_packing: If True, pack multiple conversations into single sequences
        num_workers: Number of parallel preprocessing workers
        eod_token_id: Explicit end-of-document token ID inserted between
            packed conversations.  Must be distinct from the EOS token
            that appears inside multi-turn conversations.  If None, the
            worker auto-detects from the tokenizer (backward compatible,
            but may use EOS which is incorrect for multi-turn packing).
        packing_strategy: ``"greedy"`` (default, sequential first-fit),
            ``"ffd"`` (First Fit Decreasing, in-memory -- fast but needs all
            data in RAM), or ``"ffd_disk"`` (two-pass FFD via disk -- same
            packing quality as ``"ffd"`` but bounded memory).
            Only relevant when ``enable_packing`` is True.

    Returns:
        (tokens_prefix, labels_prefix, meta_path)
    """
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

    if output_prefix is not None:
        # Explicit output prefix -- use the legacy naming scheme
        pack_suffix = ".packed" if enable_packing else ""
        tokens_prefix = f"{output_prefix}.longcontext_tokens.seqlen{seq_length}{pack_suffix}"
        labels_prefix = f"{output_prefix}.longcontext_labels.seqlen{seq_length}{pack_suffix}"
        meta_path = f"{output_prefix}.longcontext.meta.seqlen{seq_length}{pack_suffix}.json"
    else:
        # Derive paths from input data_path (same convention as auto-preprocessing)
        tokens_prefix, labels_prefix, meta_path = indexed_prefixes_for_json(
            data_path, seq_length=seq_length, enable_packing=enable_packing,
            eod_token_id=eod_token_id, packing_strategy=packing_strategy,
        )

    if not data_path.endswith(".jsonl"):
        raise ValueError(f"Only .jsonl files are supported, got: {data_path}")

    file_size = os.path.getsize(data_path)
    num_workers = max(1, min(num_workers, 256))

    if file_size == 0:
        raise ValueError(f"Input file is empty: {data_path}")

    tokenizer_model = getattr(tokenizer, "name_or_path", None)
    if tokenizer_model is None:
        raise ValueError("Tokenizer is missing name_or_path; cannot recreate in worker.")
    trust_remote_code = bool(getattr(tokenizer, "trust_remote_code", False))

    use_ffd = enable_packing and packing_strategy in ("ffd", "ffd_disk")

    if use_ffd:
        ffd_func = (
            _preprocess_ffd_disk if packing_strategy == "ffd_disk"
            else _preprocess_ffd
        )
        total_samples, total_conversations = ffd_func(
            data_path=data_path,
            seq_length=seq_length,
            num_workers=num_workers,
            tokenizer_model=tokenizer_model,
            trust_remote_code=trust_remote_code,
            eod_token_id=eod_token_id,
            tokens_prefix=tokens_prefix,
            labels_prefix=labels_prefix,
        )
    else:
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
                enable_packing,
                eod_token_id,
            ))

        print(f"Preprocessing {data_path} with {len(worker_args)} workers...")
        print(f"  seq_length={seq_length}, packing={enable_packing}")
        print(f"  output: {tokens_prefix}, {labels_prefix}")

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_preprocess_shard_worker, worker_args)

        total_samples = sum(r[2] for r in results)
        total_conversations = sum(r[3] for r in results)

        print(f"Merging {len(results)} shards...")
        _merge_indexed_shards(tokens_prefix, part_tokens_prefixes, cleanup=True)
        _merge_indexed_shards(labels_prefix, part_labels_prefixes, cleanup=True)

    meta = _build_sft_meta(
        tokenizer, seq_length, total_conversations, total_samples,
        eod_token_id=eod_token_id,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done! {total_conversations} conversations -> {total_samples} samples")
    print(f"  tokens: {tokens_prefix}.bin/.idx")
    print(f"  labels: {labels_prefix}.bin/.idx")
    print(f"  meta:   {meta_path}")

    return tokens_prefix, labels_prefix, meta_path


###############################################################################
# Automatic preprocessing (consistent with ML2 behaviour)
###############################################################################

def maybe_preprocess_sft_data(sft_config: Dict) -> None:
    """Preprocess raw JSONL data into indexed datasets if not already done.

    This mirrors the behaviour of ML2's ``_maybe_preprocess_sft_data``:
    if preprocessed indexed files do not already exist for the JSONL paths
    given in ``sft_config``, rank 0 runs the preprocessing while all other
    ranks poll-wait for the files to appear.

    After preprocessing (or discovery of existing files), the function
    populates ``sft_config`` with the resolved prefixes and meta path so that
    downstream code (``train_valid_test_datasets_provider``) can consume them
    without the user having to manually specify those paths.

    Prerequisites:
        * Megatron must already be initialised (``get_args()`` available).
        * ``sft_config`` must contain at least ``train_data_path`` (pointing
          to a ``.jsonl`` file).  Optional keys: ``valid_data_path``,
          ``test_data_path``, ``enable_packing``, ``num_workers``.
        * If explicit ``sft_tokens_prefix`` / ``sft_labels_prefix`` are
          already set in ``sft_config``, this function is a no-op for those
          splits (backwards compatible).

    Side-effects:
        * Writes ``.bin`` / ``.idx`` / ``.meta.json`` files next to the JSONL.
        * Mutates ``sft_config`` in-place to inject resolved prefixes and
          meta path.
    """
    from megatron.training import get_args
    from megatron.training.utils import print_rank_0

    args = get_args()
    rank = int(os.getenv("RANK", "0"))

    # If explicit prefixes are already provided, skip auto-preprocessing
    if sft_config.get("sft_tokens_prefix") and sft_config.get("sft_labels_prefix"):
        print_rank_0("> SFT: explicit prefixes provided, skipping auto-preprocessing.")
        return

    # Collect raw JSONL paths from sft_config
    split_keys = [
        ("train_data_path", "sft_tokens_prefix", "sft_labels_prefix", "sft_meta_path"),
        ("valid_data_path", "sft_valid_tokens_prefix", "sft_valid_labels_prefix", None),
        ("test_data_path", "sft_test_tokens_prefix", "sft_test_labels_prefix", None),
    ]

    seq_length = args.seq_length
    enable_packing = sft_config.get("enable_packing", False)
    num_workers = sft_config.get("num_workers", DEFAULT_PREPROCESS_WORKERS)
    eod_token_id = sft_config.get("eod_token_id", None)  # explicit EOD, or None for auto
    packing_strategy = sft_config.get("packing_strategy", "greedy")

    if enable_packing:
        print_rank_0(f"> SFT packing enabled (strategy={packing_strategy}): "
                     "multiple conversations will be packed per sequence")
    if eod_token_id is not None:
        print_rank_0(f"> SFT eod_token_id={eod_token_id} (explicit, distinct from EOS)")

    # Build a HuggingFace tokenizer for preprocessing / meta-matching
    tokenizer_model = args.tokenizer_model
    trust_remote_code = getattr(args, "trust_remote_code", False)
    if tokenizer_model is None:
        raise ValueError("tokenizer_model must be set in the Megatron config for SFT preprocessing.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code,
    )

    for data_key, tok_key, lab_key, meta_key in split_keys:
        data_path = sft_config.get(data_key)
        if not data_path:
            continue

        # Multiple JSONL files: merge into one on rank 0, then treat as a single
        # file. `awk 1` guarantees a newline between concatenated files. The name
        # is derived from the raw (CWD-independent) path strings so all ranks
        # agree on it; only rank 0 reads/writes it, others poll for the .bin/.idx.
        if isinstance(data_path, list):
            digest = hashlib.md5(
                "".join(p.strip() for p in data_path).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:8]
            merged = f"{os.path.splitext(data_path[0].strip())[0]}.merged.{digest}.jsonl"
            if rank == 0 and not os.path.exists(merged):
                print_rank_0(f"> Merging {len(data_path)} SFT JSONL files -> {merged}")
                with open(merged + ".tmp", "w") as out:
                    subprocess.run(
                        ["awk", "1", *[p.strip() for p in data_path]],
                        stdout=out, check=True,
                    )
                os.replace(merged + ".tmp", merged)
            data_path = merged

        # Resolve expected indexed file paths
        tokens_prefix, labels_prefix, meta_path = _resolve_indexed_prefixes(
            data_path, seq_length, enable_packing=enable_packing,
            eod_token_id=eod_token_id, packing_strategy=packing_strategy,
        )
        meta = _load_meta(meta_path)
        needs_preprocess = not (
            indexed_files_exist(tokens_prefix)
            and indexed_files_exist(labels_prefix)
            and _meta_matches(meta, tokenizer, seq_length, eod_token_id=eod_token_id)
        )

        if needs_preprocess:
            if rank == 0:
                print_rank_0(f"> Preprocessing SFT data: {data_path}")
                preprocess_sft_to_indexed(
                    data_path=data_path,
                    tokenizer=tokenizer,
                    seq_length=seq_length,
                    output_prefix=None,  # derive paths from data_path
                    enable_packing=enable_packing,
                    num_workers=num_workers,
                    eod_token_id=eod_token_id,
                    packing_strategy=packing_strategy,
                )
                print_rank_0(f"> Preprocessing complete: {data_path}")
            else:
                # Wait for rank 0 to finish preprocessing by polling for files
                wait_tok, wait_lab, wait_meta = indexed_prefixes_for_json(
                    data_path, seq_length=seq_length, enable_packing=enable_packing,
                    eod_token_id=eod_token_id, packing_strategy=packing_strategy,
                )
                print_rank_0(
                    f"> Rank {rank}: waiting for rank 0 to finish preprocessing: {data_path}"
                )
                while True:
                    m = _load_meta(wait_meta)
                    if (
                        indexed_files_exist(wait_tok)
                        and indexed_files_exist(wait_lab)
                        and _meta_matches(m, tokenizer, seq_length, eod_token_id=eod_token_id)
                    ):
                        break
                    time.sleep(10)

            # Re-resolve after preprocessing (now the files exist)
            tokens_prefix, labels_prefix, meta_path = _resolve_indexed_prefixes(
                data_path, seq_length, enable_packing=enable_packing,
                eod_token_id=eod_token_id, packing_strategy=packing_strategy,
            )
        else:
            print_rank_0(
                f"> SFT preprocessed data found for {data_path}, skipping preprocessing."
            )

        # Inject resolved paths into sft_config so the trainer can use them
        sft_config[tok_key] = tokens_prefix
        sft_config[lab_key] = labels_prefix
        if meta_key:
            sft_config[meta_key] = meta_path

    print_rank_0("> SFT auto-preprocessing done.")


###############################################################################
# SFTDataset
###############################################################################

class SFTDataset(torch.utils.data.Dataset):
    """Dataset for Supervised Fine-Tuning with answer-only loss masking.

    Uses IndexedDataset files (mmap-backed) for worker-safe memory usage.
    Requires offline preprocessing from JSONL into .bin/.idx files
    (see ``preprocess_sft_to_indexed``).
    """

    def __init__(
        self,
        tokens_prefix: str,
        labels_prefix: str,
        seq_length: int,
        num_samples: int,
        meta_path: Optional[str] = None,
        eod_token_id: int = 0,
        reset_position_ids: bool = False,
        reset_attention_mask: bool = False,
        create_attention_mask: bool = True,
    ):
        """Initialize the SFT dataset.

        Args:
            tokens_prefix: Path prefix for tokens IndexedDataset (.bin/.idx)
            labels_prefix: Path prefix for labels IndexedDataset (.bin/.idx)
            seq_length: Maximum sequence length
            num_samples: Number of samples requested (for Megatron compatibility;
                         indices wrap around if num_samples > actual dataset size)
            meta_path: Optional path to meta.json for logging
            eod_token_id: End-of-document token ID used for attention mask and
                position ID resets in packed sequences.  Must be a token that
                appears **only** at document (sample) boundaries in the
                preprocessed data -- NOT the EOS token that the chat template
                inserts at the end of each assistant turn.  Use a dedicated
                placeholder token (e.g. ``<｜place▁holder▁no▁42｜>`` = 128042
                for DeepSeek-V3) that is never produced by normal tokenization.
            reset_position_ids: Reset position IDs at EOD boundaries in packed
                sequences.  Safe to enable when ``eod_token_id`` is distinct
                from the turn-ending EOS.
            reset_attention_mask: Reset attention mask at EOD boundaries in
                packed sequences, preventing cross-document attention.  Safe
                to enable when ``eod_token_id`` is distinct from the
                turn-ending EOS.
            create_attention_mask: Whether to generate the explicit attention mask
                tensor.  Should be True to keep the batch dict consistent with
                what ``get_batch_on_this_tp_rank`` expects on non-data-loading
                TP ranks (which check ``create_attention_mask_in_dataloader``).
        """
        from megatron.core.datasets.indexed_dataset import IndexedDataset

        self.seq_length = seq_length
        self.num_samples = num_samples
        self.eod_token_id = eod_token_id
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.create_attention_mask = create_attention_mask

        # Validate files exist
        if not indexed_files_exist(tokens_prefix):
            raise FileNotFoundError(f"Tokens IndexedDataset not found at {tokens_prefix}.bin/.idx. Run preprocess_sft.py first.")
        if not indexed_files_exist(labels_prefix):
            raise FileNotFoundError(f"Labels IndexedDataset not found at {labels_prefix}.bin/.idx. Run preprocess_sft.py first.")

        self.tokens_dataset = IndexedDataset(tokens_prefix)
        self.labels_dataset = IndexedDataset(labels_prefix)
        self._num_raw_samples = len(self.tokens_dataset)

        assert len(self.tokens_dataset) == len(self.labels_dataset), \
            f"Tokens and labels dataset sizes don't match: {len(self.tokens_dataset)} vs {len(self.labels_dataset)}"

        # Cache monotonic position IDs only when no per-sample resets are needed
        self._use_cached_position_ids = not reset_position_ids
        self._cached_position_ids = torch.arange(self.seq_length, dtype=torch.long)

        # Log meta info if available
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print(f"> SFTDataset: {self._num_raw_samples} raw samples, "
                  f"meta reports {meta.get('num_samples')} samples "
                  f"from {meta.get('num_conversations')} conversations")
        else:
            print(f"> SFTDataset: {self._num_raw_samples} raw samples "
                  f"from {tokens_prefix}")
        print(f">   eod_token_id={eod_token_id}, "
              f"reset_position_ids={reset_position_ids}, "
              f"reset_attention_mask={reset_attention_mask}, "
              f"create_attention_mask={create_attention_mask}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample.

        Returns dict with keys matching GPTDataset output:
            tokens:         (seq_length,) long
            labels:         (seq_length,) long
            loss_mask:      (seq_length,) float
            position_ids:   (seq_length,) long
            attention_mask: (1, seq_length, seq_length) bool  [optional]
        """
        idx = idx % self._num_raw_samples

        tokens = torch.tensor(self.tokens_dataset[idx], dtype=torch.long)
        labels = torch.tensor(self.labels_dataset[idx], dtype=torch.long)
        loss_mask = (labels != IGNORE_INDEX).float()

        # Position IDs and attention mask. Resets fire at every eod_token_id,
        # safe because it appears only at sample boundaries (never mid-turn).
        if self.reset_position_ids or self.reset_attention_mask:
            from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids

            attention_mask, _, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                eod_token=self.eod_token_id,
                reset_position_ids=self.reset_position_ids,
                reset_attention_mask=self.reset_attention_mask,
                eod_mask_loss=False,  # SFT handles loss_mask separately
                create_attention_mask=self.create_attention_mask,
            )
        else:
            position_ids = self._cached_position_ids
            if self.create_attention_mask:
                # Plain causal mask, still emitted so the batch dict stays
                # consistent for get_batch_on_this_tp_rank's TP broadcast.
                attention_mask = torch.tril(
                    torch.ones(
                        (self.seq_length, self.seq_length), dtype=torch.bool
                    )
                ).unsqueeze(0)  # (1, S, S)
                attention_mask = attention_mask < 0.5  # True = masked positions
            else:
                attention_mask = None

        result = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }

        if attention_mask is not None:
            result["attention_mask"] = attention_mask

        return result


###############################################################################
# CLI entry point (for standalone preprocessing)
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SFT JSONL data into Megatron IndexedDataset format"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input JSONL file with conversations"
    )
    parser.add_argument(
        "--tokenizer-model", type=str, required=True,
        help="Path or name of HuggingFace tokenizer"
    )
    parser.add_argument(
        "--seq-length", type=int, required=True,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-prefix", type=str, required=True,
        help="Output file prefix (creates {prefix}_tokens.bin/.idx and {prefix}_labels.bin/.idx)"
    )
    parser.add_argument(
        "--enable-packing", action="store_true",
        help="Pack multiple conversations per sequence"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_PREPROCESS_WORKERS,
        help=f"Number of parallel preprocessing workers (default: {DEFAULT_PREPROCESS_WORKERS})"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code when loading tokenizer"
    )

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_model, trust_remote_code=args.trust_remote_code
    )

    preprocess_sft_to_indexed(
        data_path=args.input,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        output_prefix=args.output_prefix,
        enable_packing=args.enable_packing,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
