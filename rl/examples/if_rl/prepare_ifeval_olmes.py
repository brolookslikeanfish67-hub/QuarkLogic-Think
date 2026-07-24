#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""OLMES-align an IFEval eval file: strip the system prompt into a user-only copy.

OLMES ifeval is user-only; the shipped ifeval_eval.jsonl carries an extra system
prompt that the training data and IFBench lack. Constraint metadata is untouched.
IFBench is already user-only and needs no processing.
"""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="input ifeval_eval.jsonl")
    ap.add_argument("dst", help="output user-only jsonl")
    args = ap.parse_args()

    n_in = n_stripped = 0
    with open(args.src) as f, open(args.dst, "w") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_in += 1
            pr = r.get("prompt")
            if isinstance(pr, list):
                kept = [m for m in pr if m.get("role") != "system"]
                if len(kept) != len(pr):
                    n_stripped += 1
                r["prompt"] = kept
            o.write(json.dumps(r) + "\n")

    print(f"[OLMES-align] IFEval: wrote {n_in} prompts "
          f"({n_stripped} system prompt(s) stripped) -> {args.dst}")


if __name__ == "__main__":
    main()
