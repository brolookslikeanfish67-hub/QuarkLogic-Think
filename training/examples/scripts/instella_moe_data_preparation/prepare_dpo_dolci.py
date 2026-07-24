#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Sample HF dataset allenai/Dolci-Think-DPO-7B and save as JSONL for DPO training.

Each output line is saved as {"chosen": [...], "rejected": [...]}, where each value is a
list of {"role": ..., "content": ...} dicts (see utils/dpo.py).

Example:
    python prepare_dpo_dolci.py \
        --dataset allenai/Dolci-Think-DPO-7B \
        --output /path/to/dolci-think-dpo-7b-10pct.jsonl \
        --cache-dir /path/to/cache \
        --fraction 0.1 --seed 42
"""

import argparse
import json
import os
import sys
import time


def _log(msg, end="\n"):
    sys.stderr.write(msg + end)
    sys.stderr.flush()


SOURCE_PRESETS = {
    # The 3 instruction-following slices of Dolci-Think-DPO-7B (~16k pairs).
    "if-only": [
        "valpy_if_qwq_reasoning_verified_no_reasoning",
        "IF_sft_data_verified_permissive",
        "tulu-3-sft-personas-instruction-following-o3",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Sample HF DPO dataset to JSONL")
    parser.add_argument("--dataset", default="allenai/Dolci-Think-DPO-7B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="Fraction to sample (0-1); 1.0 = full dataset. "
                             "Applied after any source filtering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter-sources", default=None,
                        help="Comma-separated dataset_source values to keep.")
    parser.add_argument("--preset", default=None, choices=sorted(SOURCE_PRESETS.keys()),
                        help="Named source preset (unioned with --filter-sources).")
    parser.add_argument("--align-dp-size", type=int, default=0,
                        help="If > 0, truncate so 2*num_pairs is a multiple of this "
                             "value (pass the DP world size). Prevents the "
                             "compute_ref_logprobs EP all-to-all from stalling when "
                             "the last rank gets a short shard.")
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        _log(f"Set HF_HOME={args.cache_dir}")

    keep_sources = set()
    if args.preset:
        keep_sources.update(SOURCE_PRESETS[args.preset])
        _log(f"Preset --preset={args.preset} -> {len(SOURCE_PRESETS[args.preset])} sources")
    if args.filter_sources:
        cli_sources = [s.strip() for s in args.filter_sources.split(",") if s.strip()]
        keep_sources.update(cli_sources)
        _log(f"CLI --filter-sources adds {len(cli_sources)} sources")
    if keep_sources:
        _log(f"Active source filter ({len(keep_sources)} sources):")
        for s in sorted(keep_sources):
            _log(f"  - {s}")

    from datasets import load_dataset

    _log(f"Loading dataset '{args.dataset}' ...")
    t0 = time.time()
    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds)

    if keep_sources:
        if "dataset_source" not in ds.column_names:
            raise ValueError(
                "--filter-sources/--preset require a 'dataset_source' column, "
                f"but dataset columns are: {ds.column_names}"
            )
        before = total
        ds = ds.filter(
            lambda ex: ex["dataset_source"] in keep_sources,
            num_proc=max(1, (os.cpu_count() or 4) // 2),
        )
        total = len(ds)
        _log(f"Filtered by dataset_source: {before:,} -> {total:,} "
             f"({100.0 * total / max(before, 1):.2f}% kept)")

    if args.fraction < 1.0:
        n = int(total * args.fraction)
        _log(f"Total: {total:,}  |  Sampling {args.fraction*100:.1f}% = {n:,} pairs (seed={args.seed})")
        ds_sampled = ds.shuffle(seed=args.seed).select(range(n))
        del ds
    else:
        n = total
        _log(f"Total: {total:,}  |  Using full dataset (no sampling)")
        ds_sampled = ds

    if args.align_dp_size > 0:
        # DPO emits 2 sequences (chosen + rejected) per pair, so the token
        # dataset length is 2*n; align that to the DP world size.
        N = args.align_dp_size
        seqs = 2 * n
        seqs_aligned = (seqs // N) * N
        n_aligned = seqs_aligned // 2
        if n_aligned < n:
            _log(f"--align-dp-size={N}: truncating {n:,} -> {n_aligned:,} pairs "
                 f"({seqs:,} -> {seqs_aligned:,} seqs)")
            ds_sampled = ds_sampled.select(range(n_aligned))
            n = n_aligned
        else:
            _log(f"--align-dp-size={N}: already aligned ({seqs:,} seqs), no truncation.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    _log(f"Writing {n:,} preference pairs to {args.output} ...")
    with open(args.output, "w") as f:
        for i, ex in enumerate(ds_sampled):
            f.write(json.dumps({"chosen": ex["chosen"], "rejected": ex["rejected"]},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 10000 == 0:
                _log(f"\r  {i+1:,}/{n:,}", end="")
    _log(f"\r  {n:,}/{n:,}")

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.output) / 1e6
    _log(f"Done in {elapsed:.1f}s  ->  {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
