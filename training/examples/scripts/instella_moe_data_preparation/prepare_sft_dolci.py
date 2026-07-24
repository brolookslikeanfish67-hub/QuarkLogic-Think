#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Convert the HF Dolci-Think-SFT-7B dataset to shuffled JSONL for SFT training ({"messages": [...]})."""
import argparse
import json
import os
import random
import time


def main():
    parser = argparse.ArgumentParser(description="Export an HF SFT dataset to shuffled JSONL")
    parser.add_argument("--dataset", default="allenai/Dolci-Think-SFT-7B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.cache_dir
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset(args.dataset, cache_dir=args.cache_dir)[args.split]
    print(f"Loaded: {len(ds):,} rows  ({time.time()-t0:.1f}s)")

    indices = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(indices)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"Writing to {args.output} ...")
    count = 0
    with open(args.output, "w") as f:
        for i in indices:
            f.write(json.dumps({"messages": ds[i]["messages"]}) + "\n")
            count += 1
            if count % 100000 == 0:
                print(f"  {count:,} written  ({time.time()-t0:.1f}s)")

    print(f"Done: {count:,} rows in {time.time()-t0:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
