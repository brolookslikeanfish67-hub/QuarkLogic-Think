#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reformat a Nemotron SFT subset into shuffled ``{"messages": [...]}`` JSONL.

One script handles the three subsets used by the Instella phase-1 SFT mixture,
selected with ``--subset``:

    math     nvidia/Nemotron-Cascade-2-SFT-Data          (math/math_notool.jsonl)
    science  nvidia/Nemotron-Cascade-2-SFT-Data          (science/science.jsonl)
    cp       nvidia/Nemotron-SFT-Competitive-Programming-v2 (Python shards)

For every subset it: drops ``system`` messages, keeps assistant
``<think>...</think>`` reasoning, and writes ``{"messages": [...]}`` JSONL. The
per-subset preset also applies, where relevant, a ``generator`` filter, math
reasoning-prefix stripping, a ``reasoning_content`` -> ``<think>`` merge, and
reservoir sampling to a target size.

This is step 1 of the pipeline; step 2 (``filter_32k.py``) trims each output to
records that fit in the training sequence length. Final shuffling happens again
at pack time, so this script only shuffles the sampled subsets for good measure.

Usage:
    python prepare_nemotron_sft.py --subset math    --input <math_notool.jsonl> --output <out.jsonl>
    python prepare_nemotron_sft.py --subset science --input <science.jsonl>     --output <out.jsonl>
    python prepare_nemotron_sft.py --subset cp       --input <cp_00.jsonl> <cp_01.jsonl> --output <out.jsonl>
"""
import argparse
import json
import os
import random

# Math prompts carry a boxed-reasoning instruction prefix that we strip so the
# user turn is just the problem statement (both brace variants observed upstream).
MATH_USER_PREFIXES = [
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n",
    "Please reason step by step, and put your final answer within \\boxed{}.\n\n",
]

# Per-subset defaults. `sample_size=None` means keep every matching record.
PRESETS = {
    "math": {
        "generator": "DeepSeek-V3.2-Speciale",
        "strip_math_prefix": True,
        "merge_reasoning": False,
        "sample_size": 380_000,
    },
    "science": {
        "generator": "GPT-OSS-120B",
        "strip_math_prefix": False,
        "merge_reasoning": False,
        "sample_size": 200_000,
    },
    "cp": {
        "generator": None,
        "strip_math_prefix": False,
        "merge_reasoning": True,
        "sample_size": None,
    },
}


def reformat_record(d, *, strip_math_prefix, merge_reasoning):
    """Convert one raw record's messages into the shared SFT schema."""
    messages = []
    for msg in d["messages"]:
        role = msg["role"]
        if role == "system":
            continue

        content = msg["content"]
        if role == "user" and strip_math_prefix:
            for prefix in MATH_USER_PREFIXES:
                if content.startswith(prefix):
                    content = content[len(prefix):]
                    break
        elif role == "assistant" and merge_reasoning:
            # CP stores reasoning separately; math/science already inline <think>.
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"

        messages.append({"role": role, "content": content})
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", required=True, choices=sorted(PRESETS),
                        help="Which Nemotron subset preset to apply")
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more raw JSONL shards to read")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--generator", default=None,
                        help="Override the preset generator filter (empty string disables it)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Override the preset reservoir-sample size (0 keeps all)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    preset = PRESETS[args.subset]
    generator = preset["generator"] if args.generator is None else (args.generator or None)
    if args.sample_size is None:
        sample_size = preset["sample_size"]
    else:
        sample_size = args.sample_size or None

    rng = random.Random(args.seed)

    reservoir = []          # used when sample_size is set
    total_lines = 0
    match_count = 0

    print(f"Subset:      {args.subset}")
    print(f"Inputs:      {args.input}")
    print(f"Output:      {args.output}")
    print(f"Generator:   {generator}")
    print(f"Sample size: {sample_size}")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # When keeping all records we stream straight to disk (constant memory).
    fout = open(args.output, "w") if sample_size is None else None

    for fpath in args.input:
        print(f"Reading {fpath} ...")
        with open(fpath, "r") as f:
            for line in f:
                total_lines += 1

                # Fast substring pre-filter before the (costly) JSON parse.
                if generator is not None and generator not in line:
                    continue

                d = json.loads(line)
                if generator is not None and d.get("generator") != generator:
                    continue

                match_count += 1
                record = reformat_record(
                    d,
                    strip_math_prefix=preset["strip_math_prefix"],
                    merge_reasoning=preset["merge_reasoning"],
                )

                if sample_size is None:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                elif match_count <= sample_size:
                    reservoir.append(record)
                else:
                    j = rng.randint(0, match_count - 1)
                    if j < sample_size:
                        reservoir[j] = record

                if match_count % 200_000 == 0:
                    print(f"  {match_count:,} matches / {total_lines:,} lines ...", flush=True)

    print(f"\nTotal lines:      {total_lines:,}")
    print(f"Matching records: {match_count:,}")

    if sample_size is None:
        fout.close()
        written = match_count
    else:
        if match_count < sample_size:
            print(f"WARNING: only {match_count:,} matches (requested {sample_size:,})")
        rng.shuffle(reservoir)
        print(f"Writing {len(reservoir):,} sampled records to {args.output} ...")
        with open(args.output, "w") as f:
            for record in reservoir:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written = len(reservoir)

    with open(args.output, "r") as f:
        first = json.loads(f.readline())
    print("\nSanity check — first saved record:")
    print(f"  Num messages: {len(first['messages'])}")
    for i, m in enumerate(first["messages"]):
        print(f"  [{i}] role={m['role']}, len={len(m['content'])}")
        print(f"      preview: {m['content'][:150]}")

    print(f"\nDone! Wrote {written:,} records to {args.output}")


if __name__ == "__main__":
    main()
