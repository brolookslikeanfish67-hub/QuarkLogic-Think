#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Active sampling filter: drop prompt groups with zero reward variance.

Groups where all N samples receive the same reward produce zero GRPO gradient.
This filter drops them during oversampling so compute is spent on useful groups.

Optionally retires prompts where the model nearly always succeeds (configurable
via PROMPT_RETIREMENT_THRESHOLD env var; disabled by default for partial-credit
rewards where retirement is counterproductive).

Usage:
    --over-sampling-batch-size 128
    --dynamic-sampling-filter-path
        examples.if_rl.dynamic_sampling_filter.check_reward_nonzero_std_and_retirement
"""

import os

import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

RETIREMENT_THRESHOLD = float(os.environ.get("PROMPT_RETIREMENT_THRESHOLD", "1.0"))


def check_reward_nonzero_std_and_retirement(
    args, samples: list[Sample], **kwargs
) -> DynamicFilterOutput:
    rewards = [sample.get_reward_value(args) for sample in samples]
    rewards_t = torch.tensor(rewards, dtype=torch.float)

    if rewards_t.std().item() == 0.0:
        return DynamicFilterOutput(
            keep=False,
            reason=f"zero_std_{round(rewards[0], 1)}",
        )

    if RETIREMENT_THRESHOLD < 1.0:
        pass_rate = (rewards_t > 0.0).float().mean().item()
        if pass_rate >= RETIREMENT_THRESHOLD:
            return DynamicFilterOutput(
                keep=False,
                reason=f"retired_pass_rate_{pass_rate:.2f}",
            )

    return DynamicFilterOutput(keep=True)
