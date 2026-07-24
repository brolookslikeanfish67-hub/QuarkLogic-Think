#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eval rollout with periodic KV cache flushing.

Flushes happen:
  - Before every training rollout
  - Between eval datasets
  - Within a large eval dataset every EVAL_FLUSH_INTERVAL samples

Usage:
    --eval-function-path examples.if_rl.eval_with_flush.generate_rollout
"""

import asyncio
import copy
import logging
import os
import time
from argparse import Namespace
from typing import Any

import httpx
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.sglang_rollout import (
    GenerateState,
    generate_and_rm,
    generate_rollout as _upstream_generate_rollout,
)
from miles.utils.async_utils import run
from miles.utils.data import Dataset
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.http_utils import get
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

FLUSH_EVERY_N_SAMPLES = int(os.environ.get("EVAL_FLUSH_INTERVAL", "500"))
# Heartbeat to wandb every N completed eval samples. Prevents wandb from
# mislabeling the run as "crashed" during long eval-first passes where no
# step metric advances. Set to 0 to disable.
EVAL_HEARTBEAT_EVERY_N = int(os.environ.get("EVAL_HEARTBEAT_EVERY_N", "5"))

EVAL_PROMPT_DATASET: dict[tuple, Dataset] = {}


async def _get_engine_urls(args: Namespace) -> list[str]:
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers"
        )
        return response["urls"]
    else:
        response = await get(
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers"
        )
        return [worker["url"] for worker in response["workers"]]


async def _flush_all_engines(args: Namespace) -> bool:
    try:
        urls = await _get_engine_urls(args)
    except Exception as e:
        logger.warning(f"Could not get engine URLs for flush: {e}")
        return False

    logger.info(f"Flushing KV cache on {len(urls)} engines")
    all_ok = True

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for url in urls:
            flushed = False
            for attempt in range(10):
                try:
                    resp = await client.get(f"{url}/flush_cache")
                    if resp.status_code == 200:
                        flushed = True
                        break
                except Exception as e:
                    logger.warning(f"flush_cache attempt {attempt + 1} for {url}: {e}")
                await asyncio.sleep(1)
            if not flushed:
                logger.error(f"Failed to flush cache for {url} after 10 attempts")
                all_ok = False

    return all_ok


async def _eval_dataset_with_periodic_flush(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    all_samples_raw: list[tuple[int, Any, dict]] = []
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.group_index = _i
            sample.index = sample_index
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            all_samples_raw.append((sample_index, sample, sampling_params))
            sample_index += 1

    total = len(all_samples_raw)
    data: list[Any] = []
    do_print = True
    pbar = tqdm(total=total, desc=f"Eval {dataset_cfg.name}", disable=False)
    _eval_start = time.monotonic()
    _completed = 0

    for chunk_start in range(0, total, FLUSH_EVERY_N_SAMPLES):
        chunk = all_samples_raw[chunk_start : chunk_start + FLUSH_EVERY_N_SAMPLES]

        if chunk_start > 0:
            logger.info(f"Periodic flush at {chunk_start}/{total} samples")
            await _flush_all_engines(args)
            await asyncio.sleep(2)

        tasks = [
            asyncio.create_task(
                generate_and_rm(args, sample, sampling_params=sp, evaluation=True)
            )
            for _idx, sample, sp in chunk
        ]

        for coro in asyncio.as_completed(tasks):
            sample = await coro
            if do_print:
                logger.info(f"eval example: {[str(sample.prompt) + sample.response]} reward={sample.reward}")
                do_print = False
            if isinstance(sample, list):
                data.extend(sample)
            else:
                data.append(sample)
            pbar.update(1)
            _completed += 1
            if EVAL_HEARTBEAT_EVERY_N > 0 and _completed % EVAL_HEARTBEAT_EVERY_N == 0:
                try:
                    import wandb as _wandb
                    if _wandb.run is not None:
                        _wandb.log({
                            f"_eval_progress/{dataset_cfg.name}/completed": _completed,
                            f"_eval_progress/{dataset_cfg.name}/total": total,
                            f"_eval_progress/{dataset_cfg.name}/elapsed_s": time.monotonic() - _eval_start,
                            f"_eval_progress/{dataset_cfg.name}/wall_clock": time.time(),
                        })
                except Exception as e:  # never let a heartbeat failure break eval
                    logger.debug(f"eval heartbeat log skipped: {e}")

    pbar.close()
    data.sort(key=lambda s: s.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [s.reward if not reward_key else s.reward[reward_key] for s in data],
            "truncated": [s.status == Sample.Status.TRUNCATED for s in data],
            "samples": data,
            "n_samples_per_eval_prompt": dataset_cfg.n_samples_per_eval_prompt,
        }
    }


async def _eval_with_flush(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    datasets = getattr(args, "eval_datasets", []) or []
    results: dict[str, dict[str, list[Any]]] = {}

    for i, dataset_cfg in enumerate(datasets):
        logger.info(f"Eval dataset {i + 1}/{len(datasets)}: {dataset_cfg.name}")
        r = await _eval_dataset_with_periodic_flush(args, rollout_id, dataset_cfg)
        results.update(r)

        if i < len(datasets) - 1:
            try:
                await _flush_all_engines(args)
            except Exception as e:
                logger.warning(f"flush_cache between datasets failed: {e}; continuing")

    return RolloutFnEvalOutput(data=results)


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Drop-in replacement for miles generate_rollout with KV cache flushing."""
    if not evaluation:
        try:
            run(_flush_all_engines(args))
            logger.info("Flushed KV cache before training rollout")
        except Exception as e:
            logger.warning(f"Pre-rollout flush failed: {e}; continuing")
        return _upstream_generate_rollout(args, rollout_id, data_source, evaluation=False)

    return run(_eval_with_flush(args, rollout_id))
