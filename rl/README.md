# Instella MoE Reinforcement Learning

Reinforcement-learning post-training for Instella MoE, built on top of
[Miles](https://github.com/radixark/miles). This directory contains:

```
rl/
├── miles/                     # submodule, pinned to the miles commit this code targets
├── patches/                   # patches applied on top of the pinned miles
│   └── miles_core_changes.patch          # Instella Think-RL changes to miles core
├── setup/                     # per-node Megatron/SGLang FarSkip setup
│   └── setup_all_instella_rl.sh          # orchestrator (applies all patches + deps)
├── examples/
│   ├── if_rl/                 # instruction-following RL: reward models, rollout hooks, configs, run scripts
│   └── mopd/                  # multi-teacher on-policy distillation (MOPD) recipe
└── README.md
```

The `training/` (Primus-Instella + Megatron-LM) and `inference/`
(FarSkip-Collective SGLang) components live at the repository root and are
referenced by the setup script.

## Setup

From the repository root:

```bash
# 1. Fetch submodules (miles + Megatron-LM).
git submodule update --init --recursive

# 2. Apply the Instella Think-RL changes to the pinned miles checkout.
git -C rl/miles apply ../patches/miles_core_changes.patch

# 3. Make the examples importable as `examples.if_rl` / `examples.mopd` (the run
#    scripts and miles resolve modules relative to the miles root). Symlink both
#    into the miles examples/ tree:
ln -s ../../examples/if_rl rl/miles/examples/if_rl
ln -s ../../examples/mopd  rl/miles/examples/mopd

# 4. On EVERY Ray node, apply the Megatron/SGLang FarSkip patches + deps.
#    Defaults resolve training/, inference/ and rl/miles automatically.
bash rl/setup/setup_all_instella_rl.sh
```

> Why step 3: the run scripts add the miles root to `PYTHONPATH` and load
> functions by dotted path (e.g. `examples.if_rl.reward_model.batched_reward`
> and `examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp`).
> `examples.if_rl`, `examples.mopd`, and the miles-provided
> `examples.train_infer_mismatch_helper` must therefore be reachable under the
> miles `examples/` tree.

## Running: instruction-following RL (`if_rl`)

The self-contained instruction-following launcher used for the teacher run
lives under `examples/if_rl/`:

    # partial, 3-node
    run_rl_instella_if_thinkrl.sh   # instruction following

Checkpoint, data, and output paths default to `/path/to/...` placeholders and
are overridable via environment variables (`HF_CHECKPOINT`, `HF_SGLANG`,
`MEGATRON_CKPT`, `DATA_DIR`, `OUTPUT_DIR`, …) — see the header of the script.

### Prepare the init checkpoint (HF ↔ Megatron)

RL loads the model weights from a Megatron `torch_dist` checkpoint (`MEGATRON_CKPT`,
`--load`) while SGLang rollout loads the HF checkpoint (`HF_SGLANG` / `HF_CHECKPOINT`).
The SFT init is distributed in HF format, so convert it to Megatron `torch_dist` first
(and convert back to HF after training when you want to export/serve the RL result).
Both scripts live in `training/examples/scripts/instella_moe_process_checkpoint/`
(see its [`README.md`](../training/examples/scripts/instella_moe_process_checkpoint/README.md) for env overrides and YaRN RoPE options):

```bash
cd training/examples/scripts/instella_moe_process_checkpoint

# HF -> Megatron torch_dist (GPU, torchrun). Writes <megatron_out>/release/.
bash convert_hf_to_megatron.sh /path/to/hf_checkpoint /path/to/megatron_out

# Megatron torch_dist -> HF (CPU-only), e.g. to export a trained RL checkpoint.
python convert_megatron_to_hf.py \
    --input-dir  /path/to/megatron_out/release \
    --output-dir /path/to/hf_out \
    --origin-hf-dir /path/to/hf_checkpoint
```

## Running: multi-teacher on-policy distillation (`mopd`)

`examples/mopd/` replaces the RL reward/objective with two-teacher OPD: the
student does on-policy rollouts and each response is scored (token-level
logprobs) by the domain-appropriate frozen teacher — an IF-RL teacher for
instruction-following prompts, and the original DPO model as the anchor for
everything else. Training uses `--advantage-estimator on_policy_distillation`
(advantage = `teacher_logprob - student_logprob`) with truncated importance
weighting. Per-prompt routing is read from `sample.metadata["domain"]`.

```
mopd/
├── prepare_rl_dolci.py          # build domain-tagged prompts from Dolci-Think-RL
├── two_teacher_reward.py        # reward_func + post_process_rewards (routes on metadata.domain)
├── serve_teacher.sh             # serve one Instella teacher behind a stable endpoint (e.g. a k8s Deployment)
└── run_rl_instella_mopd.sh         # main 3-node MOPD launcher
```

Steps:

```bash
# 1. Build the domain-tagged mixed prompt set. The IF/general split defaults to
#    --if-frac 0.5; see the build-script args for other options.
python examples/mopd/prepare_rl_dolci.py --if-frac 0.5 ...   # -> opd_mixed_prompts.jsonl

# 2. Serve the two teachers (one serving deployment each, e.g. a k8s Deployment),
#    then point the launcher at their Service addresses via IF_TEACHER_HOST/PORT
#    and GENERAL_TEACHER_HOST/PORT.
MODEL=/path/to/if_rl_teacher_hf bash examples/mopd/serve_teacher.sh

# 3. Launch MOPD (Ray cluster must already be up). Paths default to /path/to/...
#    placeholders overridable via env (HF_CHECKPOINT, HF_SGLANG, MEGATRON_CKPT,
#    IF_TEACHER_MODEL, OPD_DATA, OUTPUT_DIR, MEGATRON_PATH, …).
bash examples/mopd/run_rl_instella_mopd.sh
```

The MOPD reward/eval hooks are loaded by dotted path
(`examples.mopd.two_teacher_reward.reward_func`), and the eval dispatch reuses
the `if_rl` reward model / rollout logger
(`examples.if_rl.eval_with_flush`, `examples.if_rl.custom_rollout_logger`), so
both the `if_rl` and `mopd` symlinks from setup step 3 are required. No
additional miles-core changes beyond `miles_core_changes.patch` are needed.
