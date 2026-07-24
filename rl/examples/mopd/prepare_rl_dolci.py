#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Convert the HF Dolci-Think-RL dataset to a domain-tagged JSONL for MOPD.

Each output line is a miles-rollout-ready record:
    {"prompt": [{"role": "user", "content": ...}],
     "label": <ground_truth>,
     "metadata": {"domain": "if"|"general", "rm_type": ...,
                  "dataset_source": ..., "constraint": ...}}

`metadata.domain` drives two-teacher routing in the MOPD stage: "if" prompts are
distilled from the IF-RL teacher and "general" prompts from the frozen anchor
teacher. Domain is derived from each row's `dataset_source`:
    domain = "if"      if dataset_source contains "IF_multi_constraints"
    domain = "general" otherwise   (math_rlvr / code_rlvr / rlvr_general_mix)

`--if-frac` sets the target fraction of IF-domain prompts in the mix; the two
domains are sampled to hit that ratio (capped by availability).

Pass `--if-only` to export just the IF (`IF_multi_constraints`) prompts with no
general anchor -- the "IF RLVR Mixture" used to train the IF-RL expert in the
first stage. `--if-frac` is ignored in this mode.

Prompt handling: the raw `prompt` field is a flattened string prefixed with
"user: ". Embedded "assistant:"/"user:" substrings occur *inside* the content
(e.g. example dialogues), so we do NOT parse turns -- we strip the single leading
"user: " and emit one user message (correct for single-turn prompts; safe for
the rest).

Examples:
    # MOPD: domain-tagged if/general mix at if-frac 0.5
    python prepare_rl_dolci.py \
        --dataset allenai/Dolci-Think-RL-7B \
        --output /path/to/dolci-think-rl-opd_if0.5.jsonl \
        --cache-dir /path/to/cache \
        --if-frac 0.5 --seed 0

    # IF-RL: IF-only prompts (IF RLVR Mixture)
    python prepare_rl_dolci.py \
        --dataset allenai/Dolci-Think-RL-7B \
        --output /path/to/dolci_think_rl_if.jsonl \
        --cache-dir /path/to/cache \
        --if-only --seed 0
"""

import argparse
import json
import os
import random
import sys
import time

USER_PREFIX = "user: "


def _log(msg, end="\n"):
    sys.stderr.write(msg + end)
    sys.stderr.flush()


def rm_type_for(ds: str) -> str:
    ds = ds or ""
    if "IF_multi_constraints" in ds:
        return "ifbench"
    if "math_rlvr" in ds:
        return "math"
    if "code_rlvr" in ds:
        return "code"
    if "general_mix" in ds:
        return "chat"
    return "general"


def to_record(prompt, ground_truth, constraint, ds):
    content = (prompt[len(USER_PREFIX):]
               if isinstance(prompt, str) and prompt.startswith(USER_PREFIX)
               else prompt)
    domain = "if" if "IF_multi_constraints" in (ds or "") else "general"
    return {
        "prompt": [{"role": "user", "content": content}],
        "label": ground_truth,
        "metadata": {
            "domain": domain,
            "rm_type": rm_type_for(ds),
            "dataset_source": ds,
            "constraint": constraint,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export the HF Dolci-Think-RL dataset to a domain-tagged MOPD JSONL")
    parser.add_argument("--dataset", default="allenai/Dolci-Think-RL-7B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--if-frac", type=float, default=0.5,
                        help="Target fraction of IF-domain prompts in the mix (0-1). "
                             "Ignored when --if-only is set.")
    parser.add_argument("--if-only", action="store_true",
                        help="Export only the IF (IF_multi_constraints) prompts with no "
                             "general anchor (the IF RLVR Mixture for the IF-RL stage).")
    parser.add_argument("--max-total", type=int, default=0,
                        help="Cap on total prompts (0 = use all available at the target ratio).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        _log(f"Set HF_HOME={args.cache_dir}")

    from datasets import load_dataset

    _log(f"Loading dataset '{args.dataset}' (split={args.split}) ...")
    t0 = time.time()
    ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    _log(f"Loaded {len(ds):,} rows  ({time.time()-t0:.1f}s)")

    if_rows, gen_rows = [], []
    for ex in ds:
        rec = to_record(
            ex.get("prompt"),
            ex.get("ground_truth"),
            ex.get("constraint"),
            ex.get("dataset_source"),
        )
        (if_rows if rec["metadata"]["domain"] == "if" else gen_rows).append(rec)

    rng = random.Random(args.seed)
    rng.shuffle(if_rows)
    rng.shuffle(gen_rows)

    f = args.if_frac
    if args.if_only:
        # IF RLVR Mixture: all IF prompts, no general anchor.
        n_if = min(len(if_rows), args.max_total) if args.max_total > 0 else len(if_rows)
        n_gen = 0
    elif args.max_total and args.max_total > 0:
        n_if = min(len(if_rows), round(args.max_total * f))
        n_gen = min(len(gen_rows), args.max_total - n_if)
    else:
        # Use all IF rows, take general to hit the target fraction (capped by supply).
        n_if = len(if_rows)
        n_gen = min(len(gen_rows), round(n_if * (1 - f) / max(f, 1e-9)))
        # If general is the limiter, trim IF to preserve the ratio.
        if n_gen < round(n_if * (1 - f) / max(f, 1e-9)):
            n_if = min(n_if, round(n_gen * f / max(1 - f, 1e-9)))

    recs = if_rows[:n_if] + gen_rows[:n_gen]
    rng.shuffle(recs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    _log(f"Writing {len(recs):,} prompts to {args.output} ...")
    with open(args.output, "w") as out:
        for r in recs:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(recs)
    _log(f"Done in {time.time()-t0:.1f}s  ->  {args.output}")
    _log(f"  IF      : {n_if:,}  (available {len(if_rows):,})")
    _log(f"  general : {n_gen:,}  (available {len(gen_rows):,})")
    _log(f"  IF frac : {n_if / max(total, 1):.3f}")


if __name__ == "__main__":
    main()
