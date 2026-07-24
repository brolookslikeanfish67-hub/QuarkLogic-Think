###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#################################################################################

"""SFT (Supervised Fine-Tuning) with cross-document attention masking.

Extends MegatronDocMaskTrainer to swap the dataset provider from GPTDataset
to SFTDataset while preserving cu_seqlens document masking, packed_seq_params
for Flash Attention varlen, and all other training behaviour.

The trainer class hierarchy:
    MegatronPretrainTrainer       (base: get_batch, loss_func, forward_step)
      -> MegatronDocMaskTrainer   (cu_seqlens, packed_seq_params, docmask metrics)
        -> MegatronSFTTrainer     (SFT dataset provider + auto train_iters)

SFT mode is triggered when the experiment YAML contains a top-level
``sft_config`` block.  When absent, this script falls back to standard
document-masked pre-training (identical to pretrain_docmask.py).

Requirements (same as pretrain_docmask.py):
    - context_parallel_size: 1
    - reset_position_ids: true
    - micro_batch_size: 1
    - create_attention_mask_in_dataloader: false

Usage:
    EXP=examples/megatron/configs_instella_moe/instella_moe-sft_phase1.yaml \\
    bash ./examples/run_instella.sh --task sft
"""

import json
import os
import sys

import yaml

from primus.core.launcher.initialize import log_init
from primus.core.launcher.parser import parse_args

from pretrain_docmask import MegatronDocMaskTrainer


# ---------------------------------------------------------------------------
# SFT Trainer (only used when sft_config is present in YAML)
# ---------------------------------------------------------------------------

def _build_sft_trainer_class(sft_config):
    """Build an SFT trainer class that overrides the dataset provider.

    This is defined as a factory function so it captures sft_config in a
    closure, keeping the class definition self-contained.

    The returned class inherits from MegatronDocMaskTrainer so that
    get_batch(), forward_step(), and loss_func() -- including cu_seqlens
    document masking and packed_seq_params -- are preserved unchanged.
    Only setup() and train_valid_test_datasets_provider() are overridden.
    """
    from utils.sft import SFTDataset, maybe_preprocess_sft_data

    class MegatronSFTTrainer(MegatronDocMaskTrainer):
        """SFT Trainer with document masking.

        Overrides:
            setup()  -- auto-preprocess JSONL data, compute train_iters
            train_valid_test_datasets_provider()  -- return SFTDataset

        Inherited from MegatronDocMaskTrainer:
            get_batch()     -- builds cu_seqlens from position ID resets
            forward_step()  -- passes packed_seq_params to model
            loss_func()     -- docmask wandb metrics + base loss
        """

        def setup(self):
            """Auto-preprocess data if needed, then inject train_iters."""
            from megatron.training import get_args
            from megatron.training.utils import print_rank_0

            args = get_args()

            maybe_preprocess_sft_data(sft_config)

            num_epochs = sft_config.get("num_epochs")
            if num_epochs is not None and args.train_iters is None:
                meta_path = sft_config.get("sft_meta_path")
                if meta_path and os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    num_samples = meta.get("num_samples")
                    if num_samples:
                        global_batch_size = args.global_batch_size
                        computed_iters = num_epochs * (num_samples // global_batch_size)
                        args.train_iters = computed_iters
                        if args.lr_decay_iters is None:
                            args.lr_decay_iters = computed_iters
                        if args.lr_warmup_iters is None or args.lr_warmup_iters == 0:
                            args.lr_warmup_iters = max(1, computed_iters // 10)
                        print_rank_0(
                            f"> SFT: computed train_iters={computed_iters} from "
                            f"{num_epochs} epochs x {num_samples} samples / {global_batch_size} batch"
                        )
                    else:
                        raise ValueError("sft_config.num_epochs is set but meta.json has no num_samples. Re-run preprocessing to generate an updated meta file.")
                elif meta_path:
                    raise FileNotFoundError(f"sft_config.sft_meta_path={meta_path} not found. Check that train_data_path is set or preprocess manually.")

            super().setup()

        def train_valid_test_datasets_provider(self, train_val_test_num_samples):
            """Build SFT datasets instead of standard GPT pretrain datasets."""
            from megatron.training import get_args, get_tokenizer
            from megatron.training.utils import print_rank_0

            args = get_args()
            tokenizer = get_tokenizer()

            tokens_prefix = sft_config.get("sft_tokens_prefix")
            labels_prefix = sft_config.get("sft_labels_prefix")
            meta_path = sft_config.get("sft_meta_path")

            if not tokens_prefix or not labels_prefix:
                raise ValueError("sft_config has no resolved dataset prefixes. Either set 'train_data_path' for auto-preprocessing or provide explicit 'sft_tokens_prefix' and 'sft_labels_prefix'.")

            enable_packing = sft_config.get("enable_packing", False)
            reset_position_ids = sft_config.get("reset_position_ids", False)
            reset_attention_mask = sft_config.get("reset_attention_mask", False)
            create_attention_mask = getattr(
                args, "create_attention_mask_in_dataloader", True
            )

            eod_token_id = sft_config.get("eod_token_id", None)
            if eod_token_id is None:
                eod_token_id = tokenizer.eod

            print_rank_0(f"> Building SFT datasets...")
            print_rank_0(f"  tokens: {tokens_prefix}")
            print_rank_0(f"  labels: {labels_prefix}")
            print_rank_0(f"  eod_token_id: {eod_token_id}")
            if enable_packing or reset_position_ids or reset_attention_mask:
                print_rank_0(f"  enable_packing: {enable_packing}")
                print_rank_0(f"  reset_position_ids: {reset_position_ids}")
                print_rank_0(f"  reset_attention_mask: {reset_attention_mask}")
                print_rank_0(f"  create_attention_mask: {create_attention_mask}")

            dataset_kwargs = dict(
                seq_length=args.seq_length,
                eod_token_id=eod_token_id,
                reset_position_ids=reset_position_ids,
                reset_attention_mask=reset_attention_mask,
                create_attention_mask=create_attention_mask,
            )

            train_ds = SFTDataset(
                tokens_prefix=tokens_prefix,
                labels_prefix=labels_prefix,
                num_samples=train_val_test_num_samples[0],
                meta_path=meta_path,
                **dataset_kwargs,
            )

            valid_ds = None
            valid_tokens = sft_config.get("sft_valid_tokens_prefix")
            valid_labels = sft_config.get("sft_valid_labels_prefix")
            if valid_tokens and valid_labels:
                valid_ds = SFTDataset(
                    tokens_prefix=valid_tokens,
                    labels_prefix=valid_labels,
                    num_samples=train_val_test_num_samples[1],
                    **dataset_kwargs,
                )

            test_ds = None
            test_tokens = sft_config.get("sft_test_tokens_prefix")
            test_labels = sft_config.get("sft_test_labels_prefix")
            if test_tokens and test_labels:
                test_ds = SFTDataset(
                    tokens_prefix=test_tokens,
                    labels_prefix=test_labels,
                    num_samples=train_val_test_num_samples[2],
                    **dataset_kwargs,
                )

            print_rank_0(f"> Finished building SFT datasets")
            return train_ds, valid_ds, test_ds

    return MegatronSFTTrainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sft_config = None
    try:
        exp_idx = sys.argv.index("--exp")
        exp_yaml_path = sys.argv[exp_idx + 1]
        with open(exp_yaml_path, "r") as f:
            raw_cfg = yaml.safe_load(f)
        sft_config = raw_cfg.get("sft_config", None)
    except (ValueError, IndexError, FileNotFoundError):
        pass

    primus_cfg = parse_args()

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = int(os.getenv("MASTER_PORT"))

    if sft_config:
        if rank == 0:
            print("[sft_train.py] SFT mode detected (sft_config present in YAML)")
        TrainerClass = _build_sft_trainer_class(sft_config)
    else:
        if rank == 0:
            print("[sft_train.py] No sft_config found, falling back to MegatronDocMaskTrainer")
        TrainerClass = MegatronDocMaskTrainer

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
