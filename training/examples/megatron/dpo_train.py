###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

"""DPO (Direct Preference Optimization) training for Instella.

Extends MegatronPretrainTrainer directly (no packing, no DocMask needed).

The trainer class hierarchy:
    MegatronPretrainTrainer       (base: get_batch, loss_func, forward_step)
      -> MegatronDPOTrainer       (DPO loss, raw-logits forward, ref_logprobs)

DPO mode is triggered when the experiment YAML contains a top-level
``dpo_config`` block.

Key differences from SFT/pretraining:
    1. model() called WITHOUT labels= -> returns raw TP-sharded logits
    2. Log-probs extracted via DistributedLogprob autograd Function
    3. DPO preference loss instead of cross-entropy
    4. Batch contains ref_logprobs (pre-computed)

Requirements:
    - tensor_model_parallel_size: 1  (TP>1 not supported)
    - pipeline_model_parallel_size: 1
    - create_attention_mask_in_dataloader: false

Usage:
    # Two-step DPO workflow:
    EXP=examples/megatron/configs/instella_16B_dpo_think.yaml \\
    bash ./examples/run_instella.sh --task dpo --compute-ref-logprobs

    EXP=examples/megatron/configs/instella_16B_dpo_think.yaml \\
    bash ./examples/run_instella.sh --task dpo
"""

import json
import os
import sys
from functools import partial

# Make `primus` and the vendored Megatron-LM importable before the megatron
# imports below, so this entrypoint works whether launched via run_instella.sh
# or torchrun'd directly. Unlike sft_train.py (which imports pretrain_docmask
# and inherits its bootstrap), dpo_train.py extends MegatronPretrainTrainer
# directly, so it must self-bootstrap. setup_backend_path resolves
# third_party/Megatron-LM (and honors BACKEND_PATH / --backend_path).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from primus.pretrain import setup_backend_path

setup_backend_path("megatron", verbose=False)

import torch
import yaml
from megatron.core import mpu
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers
from megatron.training.utils import get_batch_on_this_cp_rank

from primus.core.launcher.initialize import log_init
from primus.core.launcher.parser import parse_args
from primus.modules.trainer.megatron.pre_trainer import MegatronPretrainTrainer

stimer = StragglerDetector()


# ---------------------------------------------------------------------------
# DPO Trainer (only used when dpo_config is present in YAML)
# ---------------------------------------------------------------------------

def _build_dpo_trainer_class(dpo_config):
    """Build a DPO trainer class that captures dpo_config in a closure.

    The returned class inherits from MegatronPretrainTrainer directly
    (no DocMask — DPO has no sequence packing). Overrides get_batch,
    forward_step, loss computation, dataset provider, and setup.
    """
    from utils.dpo import (
        DPODataset,
        DPOLossFn,
        _merge_float_indexed_shards,
        from_parallel_logits_to_logprobs,
        indexed_files_exist,
        maybe_preprocess_dpo_data,
    )
    _compute_ref_logprobs_mode = dpo_config.get("compute_ref_logprobs", False)

    class MegatronDPOTrainer(MegatronPretrainTrainer):
        """DPO Trainer for preference optimization.

        Overrides:
            get_batch()      -- custom: bypasses get_batch_on_this_tp_rank
                                (which drops ref_logprobs), asserts TP=1
            forward_step()   -- model() WITHOUT labels -> raw logits
            dpo_loss_func()  -- DPO preference loss via DistributedLogprob
            setup()          -- auto-preprocess, compute train_iters
            train_valid_test_datasets_provider() -- return DPODataset
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ref_logprobs = None
            self._tokens = None
            self._loss_mask = None

        def get_batch(self, data_iterator):
            """Get batch with ref_logprobs, bypassing get_batch_on_this_tp_rank.

            Megatron's get_batch_on_this_tp_rank only handles 5 standard keys
            (tokens, labels, loss_mask, attention_mask, position_ids) and drops
            ref_logprobs. Since DPO requires TP=1, we read from the data
            iterator directly — no TP broadcast needed.
            """
            if (not mpu.is_pipeline_first_stage()) and (
                not mpu.is_pipeline_last_stage()
            ):
                self._ref_logprobs = None
                self._tokens = None
                self._loss_mask = None
                return None, None, None, None, None

            assert mpu.get_tensor_model_parallel_world_size() == 1, \
                "DPO trainer requires tensor_model_parallel_size=1 (ref_logprobs are not broadcast across TP ranks)"

            data = next(data_iterator)
            tokens = data["tokens"].cuda(non_blocking=True)
            labels = data["labels"].cuda(non_blocking=True)
            loss_mask = data["loss_mask"].cuda(non_blocking=True)
            attention_mask = None
            position_ids = data["position_ids"].cuda(non_blocking=True)

            self._ref_logprobs = data["ref_logprobs"].cuda(non_blocking=True)
            self._tokens = tokens
            self._loss_mask = loss_mask

            batch = {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            batch = get_batch_on_this_cp_rank(batch)
            return batch.values()

        def forward_step(self, data_iterator, model):
            """Forward WITHOUT labels -> get raw logits, not CE loss."""
            timers = get_timers()

            timers("batch-generator", log_level=2).start()
            global stimer
            with stimer(bdata=True):
                tokens, labels, loss_mask, attention_mask, position_ids = (
                    self.get_batch(data_iterator)
                )
            timers("batch-generator").stop()

            with stimer:
                output_tensor = model(tokens, position_ids, attention_mask)

            return output_tensor, partial(
                self.dpo_loss_func,
                self._tokens,
                self._loss_mask,
                self._ref_logprobs,
            )

        def dpo_loss_func(self, tokens, loss_mask, ref_logprobs, output_tensor):
            """DPO loss using DistributedLogprob for gradient flow.

            Called by Megatron's pipeline schedule on the last PP stage.
            Returns (loss, local_num_tokens, reporting_dict) matching
            Megatron's loss_func contract.
            """
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_group = mpu.get_tensor_model_parallel_group()

            token_mask = loss_mask[:, 1:]

            sample_mask = torch.ones(
                tokens.shape[0], dtype=torch.float, device=tokens.device
            )

            global_valid_seqs = sample_mask.sum().clone()
            torch.distributed.all_reduce(
                global_valid_seqs, group=mpu.get_data_parallel_group()
            )

            loss_fn = DPOLossFn(
                beta=dpo_config.get("beta", 0.05),
                preference_loss_weight=dpo_config.get(
                    "preference_loss_weight", 1.0
                ),
                preference_average_log_probs=dpo_config.get(
                    "preference_average_log_probs", False
                ),
            )
            loss, metrics = loss_fn(
                output_tensor,
                tokens,
                token_mask,
                sample_mask,
                ref_logprobs,
                global_valid_seqs,
                tp_rank,
                tp_group,
            )
            num_pairs = torch.tensor(
                float(sample_mask[::2].sum()),
                dtype=torch.float,
                device=loss.device,
            )
            reporting_loss = loss.clone().detach()
            torch.distributed.all_reduce(
                reporting_loss, group=mpu.get_data_parallel_group()
            )
            global_num_pairs = num_pairs.clone()
            torch.distributed.all_reduce(
                global_num_pairs, group=mpu.get_data_parallel_group()
            )

            local_num_tokens = num_pairs.clone().detach().to(torch.int)

            reporting_dict = {"dpo loss": (reporting_loss, global_num_pairs)}
            metric_names = ["accuracy", "rewards_chosen_mean", "rewards_rejected_mean"]
            metric_stack = torch.tensor(
                [metrics.get(n, 0.0) for n in metric_names],
                dtype=torch.float, device=loss.device,
            )
            torch.distributed.all_reduce(
                metric_stack, group=mpu.get_data_parallel_group()
            )
            acc_t, rc_t, rr_t = metric_stack.unbind()
            reporting_dict["dpo accuracy"] = acc_t
            reporting_dict["dpo rewards_chosen"] = rc_t
            reporting_dict["dpo rewards_rejected"] = rr_t

            return loss, local_num_tokens, reporting_dict

        def setup(self):
            """Auto-preprocess data if needed, compute train_iters."""
            from megatron.training import get_args
            from megatron.training.utils import print_rank_0

            args = get_args()
            self._run_complete = False

            window_size = getattr(args, "window_size", None)
            if window_size is not None and not isinstance(window_size, tuple):
                from megatron.training.arguments import tuple_type
                if isinstance(window_size, list):
                    args.window_size = tuple(int(x) for x in window_size)
                else:
                    args.window_size = tuple_type(window_size)
                print_rank_0(
                    f"> DPO: coerced args.window_size "
                    f"{window_size!r} -> {args.window_size!r} "
                    f"(YAML overrides bypass Megatron's tuple_type "
                    f"argparse callback; see HISTORY §13.14)."
                )

            if _compute_ref_logprobs_mode:
                data_path = dpo_config.get("train_data_path", "")
                base, _ = os.path.splitext(data_path)
                ref_prefix = f"{base}.dpo_ref_logprobs.seqlen{args.seq_length}"
                if indexed_files_exist(ref_prefix):
                    raise RuntimeError(f"DPO compute-ref-logprobs: found a stale artifact at {ref_prefix}.bin/.idx; delete it or point dpo_ref_logprobs_prefix at it to reuse")

            if not _compute_ref_logprobs_mode:
                latest_file = os.path.join(
                    args.save, "latest_checkpointed_iteration.txt"
                )
                if args.auto_continue_train and os.path.exists(latest_file):
                    with open(latest_file, "r") as f:
                        latest_iter = int(f.read().strip())
                    maybe_preprocess_dpo_data(dpo_config)
                    num_epochs = dpo_config.get("num_epochs")
                    meta_path = dpo_config.get("dpo_meta_path")
                    if (
                        num_epochs is not None
                        and meta_path
                        and os.path.exists(meta_path)
                    ):
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                        num_samples = meta.get("num_samples", 0)
                        expected_iters = int(num_epochs * (
                            num_samples // args.global_batch_size
                        ))
                        if latest_iter >= expected_iters:
                            print_rank_0(
                                f"> DPO: Run already complete "
                                f"(checkpoint iter {latest_iter} >= "
                                f"train_iters {expected_iters}). Skipping."
                            )
                            self._run_complete = True
                            return

            assert getattr(args, "dataloader_type", "single") == "single", \
                "DPO requires dataloader_type='single' to preserve chosen/rejected pairing"
            assert args.micro_batch_size % 2 == 0, \
                f"DPO requires an even micro_batch_size (paired chosen/rejected), got {args.micro_batch_size}"
            assert args.micro_batch_size == 2, \
                f"DPO loss reporting is only correct at micro_batch_size=2, got {args.micro_batch_size}"
            assert args.pipeline_model_parallel_size == 1, \
                f"DPO requires pipeline_model_parallel_size=1, got {args.pipeline_model_parallel_size}"

            maybe_preprocess_dpo_data(dpo_config)

            num_epochs = dpo_config.get("num_epochs")
            if num_epochs is not None and args.train_iters is None:
                meta_path = dpo_config.get("dpo_meta_path")
                if meta_path and os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    num_samples = meta.get("num_samples")
                    if num_samples:
                        global_batch_size = args.global_batch_size
                        computed_iters = int(num_epochs * (
                            num_samples // global_batch_size
                        ))
                        args.train_iters = computed_iters
                        if args.lr_decay_iters is None:
                            args.lr_decay_iters = computed_iters
                        assert args.lr_warmup_iters == 0, \
                            "When dpo_config.num_epochs is set, lr_warmup_iters must be 0 in the YAML (warmup is auto-derived as 10% of computed train_iters)"
                        args.lr_warmup_iters = max(1, computed_iters // 10)
                        print_rank_0(
                            f"> DPO: computed train_iters={computed_iters} from "
                            f"{num_epochs} epochs x {num_samples} samples "
                            f"/ {global_batch_size} batch"
                        )
                    else:
                        raise ValueError("dpo_config.num_epochs is set but meta.json has no num_samples; re-run preprocessing")
                elif meta_path:
                    raise FileNotFoundError(f"dpo_config.dpo_meta_path={meta_path} not found; set train_data_path or preprocess manually")

            super().setup()

        def train_valid_test_datasets_provider(
            self, train_val_test_num_samples
        ):
            """Return DPODataset instead of GPTDataset."""
            from megatron.training import get_args
            from megatron.training.utils import print_rank_0

            args = get_args()

            tokens_prefix = dpo_config.get("dpo_tokens_prefix")
            labels_prefix = dpo_config.get("dpo_labels_prefix")
            ref_logprobs_prefix = dpo_config.get(
                "dpo_ref_logprobs_prefix", ""
            )
            meta_path = dpo_config.get("dpo_meta_path")

            if not tokens_prefix or not labels_prefix:
                raise ValueError("dpo_config has no resolved dataset prefixes; set train_data_path or provide dpo_tokens_prefix and dpo_labels_prefix")
            if not _compute_ref_logprobs_mode and not ref_logprobs_prefix:
                raise ValueError("dpo_config must set dpo_ref_logprobs_prefix (or compute_ref_logprobs: true to generate them)")

            print_rank_0("> Building DPO datasets...")
            print_rank_0(f"  tokens:      {tokens_prefix}")
            print_rank_0(f"  labels:      {labels_prefix}")
            if _compute_ref_logprobs_mode:
                print_rank_0(
                    "  ref_logprobs: SKIPPED (compute_ref_logprobs mode)"
                )
            else:
                print_rank_0(f"  ref_logprobs: {ref_logprobs_prefix}")

            train_ds = DPODataset(
                tokens_prefix=tokens_prefix,
                labels_prefix=labels_prefix,
                ref_logprobs_prefix=ref_logprobs_prefix,
                seq_length=args.seq_length,
                num_samples=train_val_test_num_samples[0],
                meta_path=meta_path,
                skip_ref_logprobs=_compute_ref_logprobs_mode,
            )

            print_rank_0("> Finished building DPO datasets")
            return train_ds, None, None

        def run(self, *run_args, **run_kwargs):
            """Override run() to compute ref logprobs when flag is set."""
            if getattr(self, "_run_complete", False):
                from megatron.training.utils import print_rank_0
                print_rank_0("> DPO: Nothing to do — exiting gracefully.")
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
                return

            if not _compute_ref_logprobs_mode:
                return super().run(*run_args, **run_kwargs)

            self._compute_and_save_ref_logprobs()

            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()

        def _compute_and_save_ref_logprobs(self):
            """Forward-only pass: compute ref logprobs and write to IndexedDataset.

            Each DP rank processes a contiguous shard of the dataset
            sequentially (no shuffling), writes a part file, then rank 0
            merges all parts into the final IndexedDataset.
            """
            import numpy
            from megatron.core.datasets.indexed_dataset import (
                IndexedDataset as MegatronIndexedDataset,
                IndexedDatasetBuilder,
            )
            from megatron.training.utils import print_rank_0

            args = get_args()
            seq_length = args.seq_length

            tokens_prefix = dpo_config["dpo_tokens_prefix"]
            # Prefer the explicit YAML prefix so ref logprobs from different
            # checkpoints/seq_lengths don't collide, and compute/train modes
            # read and write the same path.
            explicit_prefix = dpo_config.get("dpo_ref_logprobs_prefix", "")
            if explicit_prefix:
                output_prefix = explicit_prefix
            else:
                data_path = dpo_config.get("train_data_path", "")
                base, _ = os.path.splitext(data_path)
                output_prefix = f"{base}.dpo_ref_logprobs.seqlen{seq_length}"

            if indexed_files_exist(output_prefix):
                print_rank_0(
                    f"> Ref logprobs already exist at "
                    f"{output_prefix}.bin/.idx — skipping."
                )
                return

            dp_rank = mpu.get_data_parallel_rank()
            dp_size = mpu.get_data_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_group = mpu.get_tensor_model_parallel_group()
            global_rank = int(os.getenv("RANK", "0"))

            tokens_ds = MegatronIndexedDataset(tokens_prefix)
            total_samples = len(tokens_ds)

            per_rank = (total_samples + dp_size - 1) // dp_size
            start_idx = dp_rank * per_rank
            end_idx = min(start_idx + per_rank, total_samples)
            my_count = end_idx - start_idx

            print_rank_0(
                f"> Computing ref logprobs: {total_samples} samples, "
                f"dp_size={dp_size}, micro_batch_size={args.micro_batch_size}"
            )

            part_prefix = f"{output_prefix}.part-{dp_rank:03d}"
            builder = IndexedDatasetBuilder(
                part_prefix + ".bin", dtype=numpy.float32
            )

            model = self.model[0]
            model.eval()

            mbs = args.micro_batch_size
            num_processed = 0

            with torch.no_grad():
                for batch_start in range(start_idx, end_idx, mbs):
                    batch_end = min(batch_start + mbs, end_idx)
                    actual_bs = batch_end - batch_start

                    batch_tokens = []
                    for i in range(batch_start, batch_end):
                        t = torch.tensor(
                            tokens_ds[i], dtype=torch.long
                        )
                        batch_tokens.append(t)
                    tokens = torch.stack(batch_tokens).cuda()

                    position_ids = torch.arange(
                        seq_length, dtype=torch.long, device=tokens.device
                    ).unsqueeze(0).expand(actual_bs, -1)

                    logits = model(tokens, position_ids, None)

                    vocab_per_tp = logits.shape[-1]
                    logprobs = from_parallel_logits_to_logprobs(
                        logits,
                        tokens,
                        vocab_start_index=tp_rank * vocab_per_tp,
                        vocab_end_index=(tp_rank + 1) * vocab_per_tp,
                        tp_group=tp_group,
                        inference_only=True,
                    )
                    # [B, S-1] -> [B, S]: pad with zero at end so
                    # ref_logprobs[:, :-1] recovers the original [B, S-1]
                    logprobs = torch.nn.functional.pad(
                        logprobs, (0, 1), value=0.0
                    )

                    for i in range(actual_bs):
                        builder.add_item(logprobs[i].cpu())
                        builder.end_document()

                    num_processed += actual_bs
                    if num_processed % 100 < mbs or batch_end == end_idx:
                        print(
                            f"  [dp_rank={dp_rank}] "
                            f"{num_processed}/{my_count} samples",
                            flush=True,
                        )

            builder.finalize(part_prefix + ".idx")
            print(
                f"  [dp_rank={dp_rank}] wrote {my_count} samples "
                f"to {part_prefix}",
                flush=True,
            )

            torch.distributed.barrier()

            if global_rank == 0:
                part_prefixes = [
                    f"{output_prefix}.part-{r:03d}"
                    for r in range(dp_size)
                ]
                _merge_float_indexed_shards(
                    output_prefix, part_prefixes, cleanup=True
                )
                print(
                    f"> Ref logprobs written: {output_prefix}.bin/.idx "
                    f"({total_samples} samples)"
                )

            torch.distributed.barrier()
            print_rank_0("> compute_ref_logprobs complete.")

    return MegatronDPOTrainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_env_vars(obj):
    """Resolve ${VAR:default} patterns in strings, recursing into dicts/lists.

    yaml.safe_load doesn't expand env vars — Primus's config system only
    resolves them for overrides, not for custom top-level blocks like
    dpo_config. This handles the same ${VAR:default} syntax.
    """
    import re
    _ENV_RE = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

    def _resolve_str(s):
        def _repl(m):
            var, default = m.group(1), m.group(2)
            val = os.environ.get(var)
            if val is not None:
                return val
            if default is not None:
                return default
            return m.group(0)
        return _ENV_RE.sub(_repl, s)

    if isinstance(obj, str):
        return _resolve_str(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


if __name__ == "__main__":
    dpo_config = None
    try:
        exp_idx = sys.argv.index("--exp")
        exp_yaml_path = sys.argv[exp_idx + 1]
        with open(exp_yaml_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        dpo_config = raw_cfg.get("dpo_config", None)
        if dpo_config is not None:
            dpo_config = _resolve_env_vars(dpo_config)
    except (ValueError, IndexError, FileNotFoundError):
        pass

    primus_cfg = parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = int(os.getenv("MASTER_PORT"))

    if dpo_config:
        if rank == 0:
            print("[dpo_train.py] DPO mode detected (dpo_config present in YAML)")
        TrainerClass = _build_dpo_trainer_class(dpo_config)
    else:
        raise ValueError("[dpo_train.py] dpo_config not found in YAML; this script requires a dpo_config block")

    trainer = TrainerClass(
        module_name="pre_trainer",
        primus_config=primus_cfg,
        module_rank=rank,
        module_world_size=world_size,
        module_master_addr=master_addr,
        module_master_port=master_port,
    )

    if rank == 0:
        log_init(primus_cfg, trainer.platform)

    trainer.init()
    trainer.run()
