#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Filter an SFT JSONL to records that fit within the training sequence length.

A record is kept when ``apply_chat_template`` (same tokenizer used at training
time) produces at most ``--max-tokens`` tokens. Two streaming passes keep memory
constant regardless of file size:

    Pass 1: tokenize every line in a worker pool, record kept line-indices.
    Pass 2: stream the input again, write only the kept lines (original order).

``--target-count N`` additionally subsamples the kept set down to exactly N
records (seeded), which is how the math subset goes 380k sample -> 300k @ <=32k.

Usage:
    python filter_32k.py --input in.jsonl --output out.jsonl
    python filter_32k.py --input in.jsonl --output out.jsonl --target-count 300000
"""
import argparse
import json
import multiprocessing as mp
import os
import time


def _init_worker(tokenizer_model, trust_remote_code):
    import transformers
    global _tokenizer
    _tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )


def _count_tokens(line):
    data = json.loads(line)
    out = _tokenizer.apply_chat_template(
        data["messages"], add_generation_prompt=False, tokenize=True
    )
    # Newer `transformers` returns BatchEncoding (dict-like), not a raw list.
    if hasattr(out, "input_ids"):
        return len(out["input_ids"])
    return len(out)


def _iter_lines(path):
    with open(path, "r") as f:
        for line in f:
            yield line


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Maximum total tokens per record (default: 32768)")
    parser.add_argument("--tokenizer-model", default="deepseek-ai/DeepSeek-V3",
                        help="HuggingFace tokenizer name or local path")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--chunksize", type=int, default=64,
                        help="Lines per dispatch chunk (default: 64)")
    parser.add_argument("--target-count", type=int, default=None,
                        help="If set, randomly subsample this many records from the "
                             "kept set (all kept records are written if below target)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --target-count subsampling (default: 42)")
    args = parser.parse_args()

    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Max tokens:  {args.max_tokens}")
    print(f"Tokenizer:   {args.tokenizer_model}")
    print(f"Workers:     {args.workers}")
    print()

    t1 = time.time()
    print(f"Pass 1: streaming tokenize with {args.workers} workers ...")
    kept_indices = []
    token_counts_kept = []
    total = 0
    filtered_count = 0

    with mp.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(args.tokenizer_model, args.trust_remote_code),
    ) as pool:
        for i, n_tokens in enumerate(
            pool.imap(_count_tokens, _iter_lines(args.input), chunksize=args.chunksize)
        ):
            total += 1
            if n_tokens <= args.max_tokens:
                kept_indices.append(i)
                token_counts_kept.append(n_tokens)
            else:
                filtered_count += 1

            if (i + 1) % 10_000 == 0:
                rate = (i + 1) / (time.time() - t1)
                print(f"  {i+1:,} processed "
                      f"({filtered_count:,} filtered so far, {rate:.0f} rec/s)", flush=True)

    kept = len(kept_indices)
    print(f"\nPass 1 done in {time.time() - t1:.1f}s")
    print(f"  Total:    {total:,}")
    print(f"  Kept:     {kept:,} ({100 * kept / max(total, 1):.1f}%)")
    print(f"  Filtered: {filtered_count:,} ({100 * filtered_count / max(total, 1):.1f}%)")

    if token_counts_kept:
        import statistics
        sorted_tokens = sorted(token_counts_kept)
        print("\nKept records token stats:")
        print(f"  Mean:   {statistics.mean(sorted_tokens):,.0f}")
        print(f"  Median: {statistics.median(sorted_tokens):,.0f}")
        print(f"  P90:    {sorted_tokens[int(0.9 * len(sorted_tokens))]:,}")
        print(f"  Max:    {sorted_tokens[-1]:,}")

    if args.target_count is not None:
        if kept > args.target_count:
            import random
            rng = random.Random(args.seed)
            print(f"\nSubsampling {args.target_count:,} from {kept:,} kept "
                  f"records (seed={args.seed}) ...")
            kept_indices = sorted(rng.sample(kept_indices, args.target_count))
            kept = len(kept_indices)
        else:
            print(f"\nWARNING: kept count ({kept:,}) is below --target-count "
                  f"({args.target_count:,}); writing all kept records.")

    keep_set = set(kept_indices)

    t2 = time.time()
    print(f"\nPass 2: streaming write {kept:,} records to {args.output} ...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    written = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for i, line in enumerate(fin):
            if i in keep_set:
                fout.write(line if line.endswith("\n") else line + "\n")
                written += 1
                if written % 50_000 == 0:
                    print(f"  {written:,} / {kept:,} written ...", flush=True)
    print(f"Pass 2 done in {time.time() - t2:.1f}s ({written:,} records)")
    print(f"\nDone! Total time: {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()
