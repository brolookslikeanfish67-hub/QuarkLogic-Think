#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Custom rollout logger: extra training/eval diagnostics for Wandb.

Usage:
    --custom-rollout-log-function-path
        examples.if_rl.custom_rollout_logger.log_rollout_data
    --custom-eval-rollout-log-function-path
        examples.if_rl.custom_rollout_logger.log_eval_rollout_data
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

PREFIX = "olmo3"
_REWARD_SCALE = 10.0

_reward_helpers = {}


def _get_reward_helpers():
    if _reward_helpers:
        return _reward_helpers
    try:
        from examples.if_rl.reward_model import (
            _compute_ifeval_reward,
            _enrich_ifbench_metadata,
            _strip_thinking,
            _check_one_constraint,
        )
        _reward_helpers["compute"] = _compute_ifeval_reward
        _reward_helpers["enrich"] = _enrich_ifbench_metadata
        _reward_helpers["strip"] = _strip_thinking
        _reward_helpers["check"] = _check_one_constraint
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not import reward helpers for partial credit: {e}")
    return _reward_helpers


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _compute_pass_at_k(args, groups: dict, max_reward: float = None) -> dict:
    if max_reward is None:
        max_reward = _REWARD_SCALE
    metrics = {}
    for k in [1, 2, 4, 8]:
        estimates = []
        for group_samples in groups.values():
            rewards = [s.get_reward_value(args) for s in group_samples]
            n = len(rewards)
            c = sum(1 for r in rewards if r >= max_reward - 0.01)
            estimates.append(_estimate_pass_at_k(n, c, k))
        if estimates:
            metrics[f"pass@{k}"] = float(np.mean(estimates))
    return metrics


def _compute_group_metrics(args, samples):
    metrics = {
        "removed_sample_ratio": float(np.mean([int(s.remove_sample) for s in samples])),
        "stop_rate": float(np.mean([int(s.status != Sample.Status.TRUNCATED) for s in samples])),
    }

    if args.n_samples_per_prompt <= 1:
        return metrics

    groups = group_by(samples, lambda s: s.group_index)
    if not groups:
        return metrics

    n_groups = len(groups)
    all_solved = 0
    all_zero = 0
    group_pass_rates = []
    solved_lengths = []
    unsolved_lengths = []

    for group_samples in groups.values():
        rewards = [s.get_reward_value(args) for s in group_samples]
        positive = [r > 0 for r in rewards]
        group_pass_rates.append(sum(positive) / len(positive) if positive else 0.0)

        if all(positive):
            all_solved += 1
        if not any(positive):
            all_zero += 1

        for s, pos in zip(group_samples, positive):
            (solved_lengths if pos else unsolved_lengths).append(s.effective_response_length)

    metrics.update({
        "all_solved_ratio": all_solved / n_groups,
        "all_zero_ratio": all_zero / n_groups,
        "mean_pass_rate": float(np.mean(group_pass_rates)),
        "num_groups": n_groups,
    })
    if solved_lengths:
        metrics["solved_response_len"] = float(np.mean(solved_lengths))
    if unsolved_lengths:
        metrics["unsolved_response_len"] = float(np.mean(unsolved_lengths))

    return metrics


def _compute_advantage_metrics(args, samples) -> dict:
    if args.n_samples_per_prompt <= 1:
        return {}
    groups = group_by(samples, lambda s: s.group_index)
    if not groups:
        return {}

    use_std = getattr(args, "grpo_std_normalization", True)
    all_advantages = []
    for group_samples in groups.values():
        rewards = np.array([s.get_reward_value(args) for s in group_samples])
        mean = rewards.mean()
        adv = (rewards - mean) / (rewards.std() + 1e-6) if use_std else (rewards - mean)
        all_advantages.extend(adv.tolist())

    flat = np.array(all_advantages)
    return {
        "advantages_max": float(np.max(flat)),
        "advantages_min": float(np.min(flat)),
        "advantages_std": float(np.std(flat)),
        "advantages_abs_mean": float(np.mean(np.abs(flat))),
    }


def _compute_raw_reward_metrics(args, samples) -> dict:
    rewards = [s.get_reward_value(args) for s in samples]
    if not rewards:
        return {}
    arr = np.array(rewards)
    return {
        "raw_reward_mean": float(np.mean(arr)),
        "raw_reward_std": float(np.std(arr)),
        "raw_reward_max": float(np.max(arr)),
        "raw_reward_min": float(np.min(arr)),
        "raw_reward_nonzero_frac": float(np.mean(arr > 0)),
        "raw_reward_perfect_frac": float(np.mean(arr >= _REWARD_SCALE - 0.01)),
    }


def _compute_reward_domain_metrics(args, samples) -> dict:
    """Per-domain reward + response-length breakdown, keyed off metadata.rm_type."""
    if not samples:
        return {}
    by_domain = {}
    for s in samples:
        meta = s.metadata if isinstance(s.metadata, dict) else {}
        domain = (meta.get("rm_type") or "unknown").strip() or "unknown"
        by_domain.setdefault(domain, []).append(s)

    n_total = len(samples)
    metrics = {}
    for domain, dsamples in by_domain.items():
        arr = np.array([s.get_reward_value(args) for s in dsamples])
        metrics[f"reward/{domain}/count"] = int(arr.size)
        metrics[f"reward/{domain}/frac"] = float(arr.size / n_total)
        metrics[f"reward/{domain}/mean"] = float(np.mean(arr))
        metrics[f"reward/{domain}/nonzero_frac"] = float(np.mean(arr > 0))

        lens = np.array([s.effective_response_length for s in dsamples], dtype=float)
        solved = lens[arr > 0]
        unsolved = lens[arr <= 0]
        metrics[f"length/{domain}/mean"] = float(np.mean(lens))
        metrics[f"length/{domain}/max"] = float(np.max(lens))
        if solved.size:
            metrics[f"length/{domain}/solved_mean"] = float(np.mean(solved))
        if unsolved.size:
            metrics[f"length/{domain}/unsolved_mean"] = float(np.mean(unsolved))
    return metrics


def _compute_judge_metrics(samples) -> dict:
    """Chat-judge utilization metrics from per-sample metadata['_chat_rm'] telemetry."""
    tels = [
        s.metadata["_chat_rm"]
        for s in samples
        if isinstance(s.metadata, dict) and isinstance(s.metadata.get("_chat_rm"), dict)
    ]
    if not tels:
        return {}

    n_samples = len(tels)
    total_calls = sum(int(t.get("judge_calls", 0)) for t in tels)
    total_fail = sum(int(t.get("judge_fail", 0)) for t in tels)
    total_ms = sum(float(t.get("judge_ms", 0.0)) for t in tels)
    disabled = sum(1 for t in tels if t.get("judge_disabled"))

    per_call_ms = []
    for t in tels:
        calls = int(t.get("judge_calls", 0))
        if calls > 0:
            per_call_ms.extend([float(t.get("judge_ms", 0.0)) / calls] * calls)

    metrics = {
        "judge/samples": int(n_samples),
        "judge/calls": int(total_calls),
        "judge/disabled_frac": float(disabled / n_samples),
    }
    if total_calls > 0:
        metrics["judge/fail_rate"] = float(total_fail / total_calls)
        metrics["judge/latency_ms_mean"] = float(total_ms / total_calls)
    if per_call_ms:
        metrics["judge/latency_ms_p95"] = float(np.percentile(np.array(per_call_ms), 95))
    return metrics


def _compute_length_metrics(samples) -> dict:
    lengths = [s.effective_response_length for s in samples]
    if not lengths:
        return {}
    arr = np.array(lengths)
    return {
        "response_len_mean": float(np.mean(arr)),
        "response_len_max": float(np.max(arr)),
        "response_len_min": float(np.min(arr)),
        "response_len_std": float(np.std(arr)),
    }


def _compute_ifeval_loose_partial(response: str, metadata: dict) -> float:
    helpers = _get_reward_helpers()
    check_fn = helpers.get("check")
    strip_fn = helpers.get("strip")
    if check_fn is None:
        return 0.0

    instruction_ids = metadata.get("instruction_id_list", [])
    kwargs_list = metadata.get("kwargs", [])
    prompt_text = metadata.get("prompt_text", "")

    if not instruction_ids:
        return 0.0
    if not kwargs_list or len(kwargs_list) != len(instruction_ids):
        kwargs_list = [{}] * len(instruction_ids)

    # Strip thinking to match the reward's _clean_response.
    resp = strip_fn(response) if strip_fn else response
    if not resp.strip():
        return 0.0

    lines = resp.split("\n")
    variants = [
        resp, resp.replace("*", ""),
        "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
        "\n".join(lines[1:]).strip().replace("*", ""),
        "\n".join(lines[:-1]).strip().replace("*", ""),
        "\n".join(lines[1:-1]).strip().replace("*", ""),
    ]

    passed = sum(
        1 for inst_id, kw in zip(instruction_ids, kwargs_list)
        if any(check_fn(inst_id, kw, v, prompt_text) for v in variants)
    )
    return passed / len(instruction_ids) if instruction_ids else 0.0


def _per_constraint_pass(response: str, metadata: dict):
    """Return (strict_bools, loose_bools), one bool per constraint.

    STRICT: constraint satisfied on the think-stripped response as-is.
    LOOSE: satisfied on any of the 8 response variants.
    """
    helpers = _get_reward_helpers()
    check_fn = helpers.get("check")
    strip_fn = helpers.get("strip")
    if check_fn is None:
        return [], []

    instruction_ids = metadata.get("instruction_id_list", [])
    kwargs_list = metadata.get("kwargs", [])
    prompt_text = metadata.get("prompt_text", "")
    if not instruction_ids:
        return [], []
    if not kwargs_list or len(kwargs_list) != len(instruction_ids):
        kwargs_list = [{}] * len(instruction_ids)

    # Strip thinking to match the reward's _clean_response.
    resp = strip_fn(response) if strip_fn else response
    if not resp.strip():
        n = len(instruction_ids)
        return [False] * n, [False] * n

    lines = resp.split("\n")
    loose_variants = [
        resp.replace("*", ""),
        "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
        "\n".join(lines[1:]).strip().replace("*", ""),
        "\n".join(lines[:-1]).strip().replace("*", ""),
        "\n".join(lines[1:-1]).strip().replace("*", ""),
    ]

    strict_bools, loose_bools = [], []
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        s = bool(check_fn(inst_id, kw, resp, prompt_text))
        strict_bools.append(s)
        loose_bools.append(
            s or any(check_fn(inst_id, kw, v, prompt_text) for v in loose_variants)
        )
    return strict_bools, loose_bools


def _compute_eval_partial_metrics(samples):
    """Standard IFEval metrics: prompt/inst-level strict/loose accuracy."""
    helpers = _get_reward_helpers()
    enrich_fn = helpers.get("enrich")
    check_fn = helpers.get("check")
    if not enrich_fn or check_fn is None:
        return {}

    ifeval_samples = [
        s for s in samples
        if (s.metadata if isinstance(s.metadata, dict) else {}).get("rm_type") == "ifbench"
    ]
    if not ifeval_samples:
        return {}

    p_strict, p_loose, i_strict, i_loose = [], [], [], []
    for s in ifeval_samples:
        meta = s.metadata if isinstance(s.metadata, dict) else {}
        meta = enrich_fn(meta, s.label, s.prompt)
        sb, lb = _per_constraint_pass(s.response or "", meta)
        if not sb:
            continue
        p_strict.append(1.0 if all(sb) else 0.0)
        p_loose.append(1.0 if all(lb) else 0.0)
        i_strict.append(sum(sb) / len(sb))
        i_loose.append(sum(lb) / len(lb))

    metrics = {}
    if p_strict:
        metrics["prompt_level_strict_acc"] = float(np.mean(p_strict))
        metrics["prompt_level_loose_acc"] = float(np.mean(p_loose))
        metrics["inst_level_strict_acc"] = float(np.mean(i_strict))
        metrics["inst_level_loose_acc"] = float(np.mean(i_loose))
        # back-compat aliases
        metrics["strict_partial"] = metrics["inst_level_strict_acc"]
        metrics["loose_partial"] = metrics["inst_level_loose_acc"]
    return metrics


# --- Eval-only disk dump ---

def _eval_dump_dir(args) -> str:
    """Eval dump dir. Override with INSTELLA_EVAL_DUMP_DIR; else derived from --save."""
    base = os.environ.get("INSTELLA_EVAL_DUMP_DIR")
    if base:
        return base
    save = getattr(args, "save", None)
    if save:
        return os.path.join(os.path.dirname(os.path.normpath(save)), "eval_dumps")
    return "eval_dumps"


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return str(x)


def _dump_eval_samples(rollout_id, args, data) -> None:
    """Write eval samples to <dump_dir>/eval_<rollout_id>.jsonl. Failures are swallowed."""
    try:
        dump_dir = _eval_dump_dir(args)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(dump_dir, f"eval_{rollout_id}.jsonl")

        helpers = _get_reward_helpers()
        enrich_fn = helpers.get("enrich")
        step = compute_rollout_step(args, rollout_id)

        n_rows = 0
        with open(out_path, "w") as f:
            for key, info in data.items():
                samples = info.get("samples") if isinstance(info, dict) else None
                if not samples:
                    continue
                for s in samples:
                    meta = s.metadata if isinstance(s.metadata, dict) else {}
                    row = {
                        "rollout_id": rollout_id,
                        "step": step,
                        "dataset": key,
                        "group_index": getattr(s, "group_index", None),
                        "reward": _jsonable(getattr(s, "reward", None)),
                        "status": str(getattr(s, "status", "")),
                        "response_length": getattr(s, "effective_response_length", None),
                        "prompt": _jsonable(getattr(s, "prompt", None)),
                        "label": _jsonable(getattr(s, "label", None)),
                        "response": getattr(s, "response", None),
                    }
                    if meta.get("rm_type") == "ifbench" and enrich_fn is not None:
                        try:
                            emeta = enrich_fn(meta, s.label, s.prompt)
                            sb, lb = _per_constraint_pass(s.response or "", emeta)
                            if sb:
                                row["instruction_id_list"] = emeta.get("instruction_id_list", [])
                                row["constraint_strict"] = [bool(b) for b in sb]
                                row["constraint_loose"] = [bool(b) for b in lb]
                                row["prompt_level_strict"] = bool(all(sb))
                                row["prompt_level_loose"] = bool(all(lb))
                        except Exception as e:  # noqa: BLE001
                            row["constraint_error"] = str(e)
                    f.write(json.dumps(row) + "\n")
                    n_rows += 1
        logger.info(f"{PREFIX}_eval dump: wrote {n_rows} samples -> {out_path}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{PREFIX}_eval dump failed (non-fatal): {e}")


# --- Training-rollout disk dump (opt-in via INSTELLA_TRAIN_DUMP_DIR) ---

def _rollout_logprob_summary(s):
    """Per-token rollout-logprob summary to locate degenerate ("salad") onset."""
    lp = getattr(s, "rollout_log_probs", None)
    if not lp:
        return {}
    arr = np.asarray(lp, dtype=np.float32)
    n = len(arr)
    # First index of a sustained low-logprob run (window of 8 tokens all < -6).
    onset = None
    if n >= 8:
        low = arr < -6.0
        run = np.convolve(low.astype(np.int32), np.ones(8, np.int32), "valid")
        idx = np.flatnonzero(run >= 8)
        if idx.size:
            onset = int(idx[0])
    return {
        "rollout_logprob_mean": float(arr.mean()),
        "rollout_logprob_min": float(arr.min()),
        "rollout_logprob_argmin_frac": float(int(arr.argmin()) / max(n, 1)),
        "rollout_frac_lt_m6": float((arr < -6.0).mean()),
        "rollout_frac_lt_m9": float((arr < -9.0).mean()),
        "salad_onset_idx": onset,
        "salad_onset_frac": (float(onset / n) if onset is not None else None),
    }


def _train_dump_dir(args):
    base = os.environ.get("INSTELLA_TRAIN_DUMP_DIR")
    if base:
        return base
    return None  # opt-in only: no dir => no train dump


def _dump_train_samples(rollout_id, args, samples) -> None:
    """Dump training rollouts to <dir>/train_<rollout_id>.jsonl (opt-in).

    Env: INSTELLA_TRAIN_DUMP_EVERY (default 1), INSTELLA_TRAIN_DUMP_LOGPROBS.
    """
    try:
        dump_dir = _train_dump_dir(args)
        if not dump_dir:
            return
        every = int(os.environ.get("INSTELLA_TRAIN_DUMP_EVERY", "1") or "1")
        if every > 1 and (rollout_id % every) != 0:
            return
        full_lp = os.environ.get("INSTELLA_TRAIN_DUMP_LOGPROBS", "0") == "1"
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(dump_dir, f"train_{rollout_id}.jsonl")
        step = compute_rollout_step(args, rollout_id)
        n_rows = 0
        with open(out_path, "w") as f:
            for s in samples:
                row = {
                    "rollout_id": rollout_id,
                    "step": step,
                    "group_index": getattr(s, "group_index", None),
                    "reward": _jsonable(getattr(s, "reward", None)),
                    "status": str(getattr(s, "status", "")),
                    "response_length": getattr(s, "effective_response_length", None),
                    "raw_response_length": getattr(s, "response_length", None),
                    "prompt": _jsonable(getattr(s, "prompt", None)),
                    "label": _jsonable(getattr(s, "label", None)),
                    "response": getattr(s, "response", None),
                }
                row.update(_rollout_logprob_summary(s))
                if full_lp and getattr(s, "rollout_log_probs", None):
                    row["rollout_log_probs"] = [round(float(x), 3) for x in s.rollout_log_probs]
                f.write(json.dumps(row) + "\n")
                n_rows += 1
        logger.info(f"{PREFIX}_train dump: wrote {n_rows} samples -> {out_path}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{PREFIX}_train dump failed (non-fatal): {e}")


# --- Public API ---

def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Training rollout logger — adds olmo3/* metrics to Wandb."""
    _dump_train_samples(rollout_id, args, samples)
    log_dict = {f"{PREFIX}/{k}": v for k, v in _compute_group_metrics(args, samples).items()}

    if args.n_samples_per_prompt > 1:
        groups = group_by(samples, lambda s: s.group_index)
        for mk, mv in _compute_pass_at_k(args, groups).items():
            log_dict[f"{PREFIX}/{mk}"] = mv

    for sub in [_compute_advantage_metrics, _compute_raw_reward_metrics, _compute_reward_domain_metrics]:
        for mk, mv in sub(args, samples).items():
            log_dict[f"{PREFIX}/{mk}"] = mv

    for mk, mv in _compute_length_metrics(samples).items():
        log_dict[f"{PREFIX}/{mk}"] = mv

    for mk, mv in _compute_judge_metrics(samples).items():
        log_dict[f"{PREFIX}/{mk}"] = mv

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    logger.info(f"{PREFIX} {rollout_id}: {log_dict}")
    tracking_utils.log(args, log_dict, step_key="rollout/step")
    return False


def log_eval_rollout_data(rollout_id, args, data, extra_metrics):
    """Eval rollout logger — adds olmo3/eval_* metrics with partial-credit IFEval."""
    _dump_eval_samples(rollout_id, args, data)

    log_dict = {}
    for key, info in data.items():
        samples = info.get("samples")
        if samples is None:
            continue

        for mk, mv in _compute_group_metrics(args, samples).items():
            log_dict[f"{PREFIX}/eval_{key}/{mk}"] = mv

        n_per = info.get("n_samples_per_eval_prompt", 1)
        if n_per > 1:
            groups = group_by(samples, lambda s: s.group_index)
            for k_val in sorted({1, n_per}):
                estimates = []
                for gs in groups.values():
                    rewards = [s.reward if isinstance(s.reward, (int, float)) else 0.0 for s in gs]
                    n = len(rewards)
                    if n < k_val:
                        continue
                    # Count only perfect-reward samples, matching _compute_pass_at_k.
                    c = sum(1 for r in rewards if r >= _REWARD_SCALE - 0.01)
                    estimates.append(_estimate_pass_at_k(n, c, k_val))
                if estimates:
                    log_dict[f"{PREFIX}/eval_{key}/pass@{k_val}"] = float(np.mean(estimates))

        for mk, mv in _compute_eval_partial_metrics(samples).items():
            log_dict[f"{PREFIX}/eval_{key}/{mk}"] = mv
        for mk, mv in _compute_raw_reward_metrics(args, samples).items():
            log_dict[f"{PREFIX}/eval_{key}/{mk}"] = mv
        for mk, mv in _compute_length_metrics(samples).items():
            log_dict[f"{PREFIX}/eval_{key}/{mk}"] = mv

    if log_dict:
        step = compute_rollout_step(args, rollout_id)
        log_dict["eval/step"] = step
        logger.info(f"{PREFIX}_eval {rollout_id}: {log_dict}")
        tracking_utils.log(args, log_dict, step_key="eval/step")
    return False
