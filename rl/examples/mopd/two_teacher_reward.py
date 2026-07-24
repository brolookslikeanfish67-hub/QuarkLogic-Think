###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Two-teacher on-policy distillation reward functions for Instella.

Design
------
The student (the DPO model being fine-tuned) generates on-policy rollouts.
Each rollout is scored by a *teacher* SGLang server that returns the teacher's
token-level log-probabilities over the student's sampled tokens. The
`on_policy_distillation` advantage estimator then uses
    advantage = teacher_log_prob - student_log_prob
per token.

The twist here is TWO teachers, routed per-prompt by domain:
  * IF-domain prompts      -> IF-RL model teacher      (transfers IF behavior)
  * everything else        -> original DPO model teacher (self-anchor; keeps
                              the student close to its init on non-IF domains,
                              preventing the general-capability regressions an
                              IF-only update would cause)

Routing key
-----------
Each prompt's domain is read from the rollout `Sample`:
  1. sample.metadata["domain"]  (preferred; set via --metadata-key)
  2. sample.label               (fallback)
  3. "general"                  (default)
A prompt is treated as IF-domain iff its domain string (lower-cased) is in
OPD_IF_DOMAINS.

Teacher endpoints + routing are configured via env vars (no core changes):
  IF_TEACHER_URL        e.g. http://127.0.0.1:13141/generate
  GENERAL_TEACHER_URL   e.g. http://127.0.0.1:13142/generate
  OPD_IF_DOMAINS        comma-list, default "if,ifeval,ifbench,instruction_following"
If a URL is unset, the module falls back to args.rm_url for that teacher.
"""

import os

import aiohttp
import asyncio
import logging

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_IF_DOMAINS = {
    d.strip().lower()
    for d in os.environ.get(
        "OPD_IF_DOMAINS", "if,ifeval,ifbench,instruction_following"
    ).split(",")
    if d.strip()
}

# Optional per-actor cap on concurrent teacher scoring requests. The rollout
# can fire hundreds of prefill-only return_logprob requests at once, which
# maximizes DP-rank batch imbalance and is the trigger for the SGLang
# dp-attention lockstep all_gather hang. Limiting in-flight requests sharply
# reduces that trigger. OPD_TEACHER_MAX_CONCURRENCY<=0 (default) = unlimited.
# The semaphore is created lazily so it binds to the running event loop.
_TEACHER_SEM: "asyncio.Semaphore | None" = None
_TEACHER_SEM_LIMIT = -1


def _teacher_semaphore():
    """Return the shared scoring semaphore (or None if unlimited)."""
    global _TEACHER_SEM, _TEACHER_SEM_LIMIT
    if _TEACHER_SEM_LIMIT < 0:
        try:
            _TEACHER_SEM_LIMIT = int(os.environ.get("OPD_TEACHER_MAX_CONCURRENCY", "0"))
        except ValueError:
            _TEACHER_SEM_LIMIT = 0
    if _TEACHER_SEM_LIMIT <= 0:
        return None
    if _TEACHER_SEM is None:
        _TEACHER_SEM = asyncio.Semaphore(_TEACHER_SEM_LIMIT)
    return _TEACHER_SEM


def _sample_domain(sample: Sample) -> str:
    md = getattr(sample, "metadata", None) or {}
    domain = md.get("domain")
    if domain is None:
        domain = getattr(sample, "label", None)
    return str(domain).lower() if domain is not None else "general"


def _candidate_urls(args, sample: Sample) -> list[str]:
    """Ordered teacher endpoints for this sample's domain: the assigned (primary)
    teacher first, then the common backup teachers. reward_func tries them in
    order, so a hung/failed primary fails over to a warm backup instead of
    killing the run. Backups are read from IF_TEACHER_BACKUP_URLS /
    GENERAL_TEACHER_BACKUP_URLS (comma-separated /generate URLs)."""
    is_if = _sample_domain(sample) in _IF_DOMAINS
    if is_if:
        primary = os.environ.get("IF_TEACHER_URL")
        backups = os.environ.get("IF_TEACHER_BACKUP_URLS", "")
    else:
        primary = os.environ.get("GENERAL_TEACHER_URL")
        backups = os.environ.get("GENERAL_TEACHER_BACKUP_URLS", "")
    candidates = [primary or args.rm_url] + [u.strip() for u in backups.split(",") if u.strip()]
    seen, ordered = set(), []
    for u in candidates:
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _is_eval_sample(sample: Sample) -> bool:
    """Eval samples come from the eval-config datasets and are NOT tagged with a
    training ``domain`` (our OPD prompts always carry metadata.domain). They do
    carry an rm_type (math/ifbench/...). Distillation (training) samples always
    have metadata.domain, so absence of "domain" => eval sample."""
    md = sample.metadata if isinstance(sample.metadata, dict) else {}
    return "domain" not in md


async def reward_func(args, sample, **kwargs):
    """Two-purpose reward dispatched per sample (one custom_rm_path serves both
    training and eval, like reward_model.batched_reward does for rm_type):

      * TRAINING (sample has metadata["domain"]) — query the domain-appropriate
        teacher server and return the raw SGLang /generate JSON (stored on
        sample.reward); teacher log-probs are extracted in post_process_rewards.
      * EVAL (no metadata["domain"], has rm_type) — delegate to our IFEvalG-based
        multi-domain verifier (examples.if_rl.reward_model.batched_reward)
        so AIME/IFEval/IFBench eval produce real pass-rate scores. We must NOT use
        miles' built-in rm_hub ifbench scorer: it only knows IFBench instructions
        and KeyErrors on base-IFEval instruction IDs (e.g. punctuation:no_comma).
        batched_reward dispatches rm_type=math->DeepScaler and rm_type=ifbench->
        IFEvalG (which handles BOTH IFEval and IFBench), so ALL eval routes here.
    """
    if _is_eval_sample(sample):
        from examples.if_rl.reward_model import batched_reward

        # Force evaluation=True so the IFEvalG grader scores IFEval/IFBench as
        # STRICT prompt-level accuracy (all constraints must pass => 1.0, else
        # 0.0), matching standard IFEval/IFBench. batched_reward forwards
        # evaluation -> _score_one, where partial_credit = not evaluation.
        # Drop any caller-supplied "evaluation" to avoid a duplicate kwarg.
        kwargs.pop("evaluation", None)
        return await batched_reward(args, sample, evaluation=True, **kwargs)

    urls = _candidate_urls(args, sample)
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,  # score only, do not generate
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    # Try each endpoint in order; if all fail in a round, back off exponentially
    # and retry the whole list. Only raises if every endpoint stays unreachable.
    attempts = int(os.environ.get("OPD_TEACHER_RETRIES", "8"))
    timeout = aiohttp.ClientTimeout(total=float(os.environ.get("OPD_TEACHER_TIMEOUT", "600")))

    async def _score():
        last_err = None
        for i in range(attempts):
            for j, url in enumerate(urls):
                try:
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, json=payload) as resp:
                            resp.raise_for_status()
                            if j > 0:
                                logger.warning(
                                    "teacher scoring succeeded on backup #%d (%s) "
                                    "after primary failure", j, url,
                                )
                            return await resp.json()
                except Exception as e:  # noqa: BLE001 - fail over to next endpoint
                    last_err = e
                    kind = "primary" if j == 0 else f"backup #{j}"
                    logger.warning(
                        "teacher scoring failed on %s (%s) round %d/%d: %s",
                        kind, url, i + 1, attempts, e,
                    )
                    continue
            # Every endpoint failed this round -> back off before retrying list.
            delay = min(2.0 * (2 ** i), 60.0)
            logger.warning(
                "all %d teacher endpoints failed round %d/%d; retrying in %.0fs: %s",
                len(urls), i + 1, attempts, delay, urls,
            )
            await asyncio.sleep(delay)
        raise RuntimeError(
            f"teacher scoring failed after {attempts} rounds over {urls}: {last_err}"
        )

    sem = _teacher_semaphore()
    if sem is None:
        return await _score()
    async with sem:
        return await _score()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs over the response span and attach to samples.

    Each sample's reward dict already came from its routed teacher, so no
    per-domain branching is needed here.
    """
    rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    teacher_log_probs = [
        torch.tensor(
            [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
            dtype=torch.float32,
        )
        for reward in rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    return teacher_log_probs, teacher_log_probs
