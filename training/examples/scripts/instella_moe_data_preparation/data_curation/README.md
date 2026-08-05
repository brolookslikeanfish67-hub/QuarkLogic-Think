# Feedback-driven Data Curation

Iteratively curate SFT data for a student model by turning its own errors into a
data-selection policy, then retrieving matching examples from a large pool.

`curation_pipeline_v0.py` is a single, self-contained walkthrough of the whole
workflow. Steps 1–6 (tagging, embedding, seed/scoring sets, inference + judging,
policy generation, data selection) are runnable: every model is reached through a
configurable OpenAI-compatible endpoint, and Step 4 can launch and tear down a
stock SGLang server for the student on its own. Step 7's evaluation half is also
runnable (it reuses the same self-contained serving path); its training and
checkpoint-conversion half shells out to the cluster's own trainer/converter via
environment-configured paths (set `SKIP_TRAINING=1` to evaluate an existing
checkpoint with no training plumbing). Models are chosen by role via environment
variables (`LLM_LARGE`, `LLM_SMALL`/`FEEDBACK_MODEL`, `STUDENT_MODEL`, `EMBEDDING_MODEL`).

## Workflow

One iteration (see `main`):

1. **Domain tagging** (`tag_domains`) — label every pool example MATH / CODE / OTHER.
2. **Pool embedding** (`embed_pool`) — embed the pool once with `EMBEDDING_MODEL`.
3. **Seed/scoring sets** (`build_seed_scoring_sets`) — build held-out evaluation
   sets: MATH from a JSONL source (`MATH_SEED_JSONL`); CODE and OTHER sampled from
   the domain-tagged pool.
4. **Inference + judging** (`pretrain_inference_and_judge`) — run the current
   student checkpoint on the seed set, judge answers, and produce structured
   error analytics.
5. **Policy generation** (`run_policy_mode` → `call_reflection_lm`) — a large
   reflection LM converts the error analytics into a weighted, domain-tagged
   prompt policy.
6. **Data selection** (`run_select_mode`) — domain-aware retrieval against the
   pool embeddings. The training budget is split across domains by fixed
   fractions. MATH and CODE are policy-driven: each is `(1 - alpha)` greedy
   nearest-neighbor retrieval + `alpha` same-domain random. OTHER has no policy
   and is filled entirely at random. A length filter drops over-long examples.
7. **Train + evaluate** (`train_and_eval`) — fine-tune the student on the curated
   data, convert the checkpoint, and rescore against the scoring sets.
8. **Iterate** with the improved checkpoint.

## Final configuration (Variant C)

- 3-way domains: MATH / CODE / OTHER.
- Domain budget fractions: MATH 0.393, CODE 0.315, OTHER 0.292.
- `alpha = 0.5` — within MATH and CODE, 50% greedy + 50% same-domain random.
  OTHER is always fully random (unaffected by `alpha`).
- Budget: 512K selected examples; 32K-token length filter.
- Error analytics for the policy were generated from the iter-12000 checkpoint.

## Running the pipeline

### Setup

```bash
pip install openai sentence-transformers datasets numpy torch transformers polars
```

The pipeline reaches every model through an OpenAI-compatible endpoint. Roles can
share one endpoint or use separate ones (student local, judge remote, etc.):

```bash
# Shared defaults (used by any role that doesn't set its own endpoint)
export LLM_BASE_URL=http://localhost:30000/v1
export LLM_API_KEY=EMPTY

# Model names per role
export LLM_SMALL=<judge-model>        # classify / judge / feedback
export LLM_LARGE=<reflection-model>   # reflection (policy)
export STUDENT_MODEL=<student-model>  # model being curated for
export EMBEDDING_MODEL=<embedding-model>

# Optional per-role endpoints (fall back to LLM_BASE_URL / LLM_API_KEY)
export STUDENT_BASE_URL=http://localhost:30000/v1   # local student server
export STUDENT_API_KEY=EMPTY
export JUDGE_BASE_URL=https://openrouter.ai/api/v1  # remote judge
export JUDGE_API_KEY=<key>
export REFLECTION_BASE_URL=https://openrouter.ai/api/v1  # remote reflection LM (Step 5)
export REFLECTION_API_KEY=<key>
```

Run the whole loop with `python curation_pipeline_v0.py`, or one step at a time
as below (each step's function is callable directly).

You can serve the student yourself, or let Step 4 launch and tear it down for you
(see below). For a non-student role, serve any OpenAI-compatible model, e.g.:

```bash
cd /home && python -m sglang.launch_server \
    --model-path <served-model> --tp 1 --dtype auto \
    --trust-remote-code --port 30000
```

### Step 1 — Domain tagging

Label every pool example `MATH` / `CODE` / `OTHER` (uses `LLM_SMALL`).

```bash
python - <<'PY'
import curation_pipeline_v0 as c

c.tag_domains("pool.jsonl", "pool.tags.jsonl")
PY
```

### Step 2 — Pool embedding

Embed the pool once with `EMBEDDING_MODEL` into a normalized tensor.

```bash
python - <<'PY'
import curation_pipeline_v0 as c

c.embed_pool("pool.jsonl", "pool.pt")
PY
```

### Step 3 — Seed / scoring sets

Build held-out evaluation sets (MATH from `MATH_SEED_JSONL`, CODE/OTHER from the tagged pool).

```bash
python - <<'PY'
import curation_pipeline_v0 as c

c.build_seed_scoring_sets("pool.jsonl", "pool.tags.jsonl", "eval_sets")
PY
```

### Step 4 — Inference + judging

Run the student on the seed set, judge each answer, and emit error analytics.
This step owns the full lifecycle: if `STUDENT_BASE_URL` already points at a live
server it is reused; otherwise the pipeline **launches a stock SGLang server from
`STUDENT_CHECKPOINT`, waits for `/health`, generates, then tears it down.**

Judging uses `LLM_SMALL` over `JUDGE_BASE_URL` (e.g. a strong judge on OpenRouter),
so the local student and the remote judge stay on separate endpoints.

Serving knobs (env, portable defaults):

```bash
export STUDENT_CHECKPOINT=amd/Instella-MoE-16B-A3B-Think  # or a local checkpoint dir
export STUDENT_SERVE_PORT=30000
export STUDENT_SERVE_TP=8            # tensor-parallel size (1 for small models)
export STUDENT_SERVE_MEM_FRACTION=0.85
export STUDENT_SERVE_TIMEOUT=900     # seconds to wait for /health
# Build-specific launch flags passed verbatim to sglang.launch_server:
export STUDENT_SERVE_EXTRA_ARGS="--ep 8 --disable-shared-experts-fusion --disable-radix-cache"
```

**Serving Instella-MoE (FarSkip MoE) on AMD ROCm.** The public
`amd/Instella-MoE-16B-A3B-Think` uses Gated-MLA + FarSkip-Collective; serve it with
AMD's SGLang build and export the FarSkip runtime flags before launching (Step 4
inherits the environment):

```bash
export USE_SIMPLE_FARSKIP=1 SIMPLE_FARSKIP_NULL_DEBUG=0
export SGLANG_ROCM_FUSED_DECODE_MLA=0 SGLANG_USE_AITER=1 SGLANG_USE_ROCM700A=1
export SGLANG_MOE_PADDING=1 SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
```

Then run the step:

```bash
python - <<'PY'
import curation_pipeline_v0 as c

# checkpoint arg is also used as the served model id; leave STUDENT_BASE_URL unset
# to have this step launch + tear down the server for you.
c.pretrain_inference_and_judge(
    c.STUDENT_CHECKPOINT, "eval_sets/seed_math.jsonl", "analytics.json",
)
PY
```

The seed set is JSONL in chat form — each line `{"messages": [{"role": "user",
"content": ...}, {"role": "assistant", "content": <reference answer>}]}`. Output
`analytics.json` holds the score plus `sampled_errors`, each with structured
feedback (domain, error_type, skills_lacking, what_would_help, summary) that
Step 5 turns into a selection policy.

### Step 5 — Policy generation

Turn the error analytics into a weighted, domain-tagged policy. Uses `LLM_LARGE`
over `REFLECTION_BASE_URL` (e.g. a strong reflection LM on OpenRouter); the
proposed prompt weights are normalized to sum to 1.0. `REFLECTION_MAX_TOKENS`
caps the reflection generation (default 65536) and `REFLECTION_TEMPERATURE`
controls its exploration (default 0.7, matching the reference).

```bash
python - <<'PY'
import curation_pipeline_v0 as c

c.run_policy_mode("analytics.json", "policy.jsonl")
PY
```

### Step 6 — Data selection

Domain-aware retrieval against the pool embeddings; writes `curated.jsonl` (plus
`curated_meta.json`). The `budget` is split across domains by fixed fractions.
MATH/CODE are policy-driven — each prompt greedily retrieves its weighted share of
the nearest pool items (cosine similarity within that domain), and `alpha` is the
fraction filled with same-domain random samples instead (e.g. `alpha=0.5` → 50%
greedy / 50% random). OTHER is always random. Items longer than `max_tokens` are
skipped; set `TOKENIZER_NAME` for exact chat-template token counts (otherwise a
~4-chars/token heuristic is used).

For Instella-MoE the candidate `pool.jsonl` is drawn from
[Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B),
[Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1),
and [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data),
in the shared `{"messages": [...]}` schema. Set `TOKENIZER_NAME=deepseek-ai/DeepSeek-V3`
with `max_tokens=32000` to match the DeepSeek-V3 chat-template length filter used for
the phase-2 SFT data. The resulting 512K `curated.jsonl` is the phase-2 SFT mixture
consumed by `configs_instella_moe/instella_moe-sft_phase2.yaml`.

```bash
python - <<'PY'
import curation_pipeline_v0 as c

c.run_select_mode(
    "policy.jsonl", "pool.pt", "pool.jsonl", "pool.tags.jsonl",
    budget=512000, alpha=0.5,
)
PY
```

### Step 7 — Train + evaluate

Fine-tune the student on `curated.jsonl`, convert the checkpoint, and rescore
against the scoring sets. Training and conversion shell out to the cluster's own
tooling (all paths env-configured, nothing hard-coded); evaluation reuses the same
serve → generate → judge → teardown lifecycle as Step 4.

```bash
# Training + conversion (shelled out to the repo's trainer/converter):
export SFT_BASE_CONFIG=examples/megatron/configs_instella_moe/instella_moe-sft_phase2.yaml
export PRIMUS_DIR=/path/to/Instella-MoE/training   # trainer root; runs SFT_LAUNCH_SCRIPT
export SFT_LAUNCH_SCRIPT="./examples/run_instella.sh --task sft"
export CONVERT_SCRIPT=$PRIMUS_DIR/examples/scripts/instella_moe_process_checkpoint/convert_megatron_to_hf.py
export ORIGIN_HF_DIR=/path/to/hf_checkpoint   # tokenizer/config source for conversion
export HF_OUTPUT_ROOT=/path/to/hf_out         # converted checkpoints land here
# Multi-node passthrough (optional): NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT

python - <<'PY'
import curation_pipeline_v0 as c

c.train_and_eval(
    "curated.jsonl", "/ckpts/iter_12000",
    ["eval_sets/scoring_math.jsonl", "eval_sets/scoring_coding.jsonl",
     "eval_sets/scoring_other.jsonl"],
    resume_iteration=12000, train_steps=3000,
)
PY
```

To evaluate an existing checkpoint without any training/conversion, set
`SKIP_TRAINING=1` (the step then serves `STUDENT_CHECKPOINT`, or reuses a live
`STUDENT_BASE_URL`, and just reports per-set accuracy).
