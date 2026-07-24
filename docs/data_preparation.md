# Training Data Preparation

## Data sources
Below we describe the dataset sources used in each training stage.
The mixtures are defined in the stage configs under `training/examples/megatron/configs_instella_moe`.

### Pre-training
Trained from scratch on 7.1T tokens spanning web, math, code, SFT and other domains.

| Domain | Dataset |
| --- | --- |
| Web | [Nemotron-CC-v2](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) |
| Mathematics | [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1), [MegaMath](https://huggingface.co/datasets/LLM360/MegaMath), [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) |
| Code | [RefineCode](https://huggingface.co/datasets/OpenCoder-LLM/RefineCode-code-corpus-meta), [Nemotron-Pretraining-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v1) |
| SFT | [Nemotron-Pretraining-SFT-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1) |
| Others | [TxT360](https://huggingface.co/datasets/LLM360/TxT360) ( `arxiv`, `dm_maths`, `europarl`, `freelaw`, `hackernews`, `pg19`, `phil_papers`, `pubmed_abstract`, `pubmed_central`, `s2orc_abstract`, `s2orc_fulltext`, `stackexchange`, `ubuntu_irc`, `uspto`, `wikipedia`, `wikipedia_extended` ) |

### Mid-training
The mid-training stage trains with three data-mixture variants and we then merge the checkpoints from the three runs to obtain the final mid-training checkpoint. 

The three variants are all based on the [Dolma 3 Dolmino 100B Mix](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125) dataset. They differ only in a few STEM/reasoning subsets to emphasize different capabilities. `v1` is the original Dolma 3 Dolmino 100B Mix data; `v2` replaces `STEM-Heavy Crawl` with the full subset from the [Dolma 3 Dolmino pool](https://huggingface.co/datasets/allenai/dolma3_dolmino_pool); `v3` additionally pulls in the full pool subsets for `MegaMatt`, `General Reasoning Mix`, `Math Meta-Reasoning`, and `Code Meta-Reasoning`. All other subsets are identical across variants.

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Source</th>
      <th>v1 (Dolmino 100B mix)</th>
      <th>v2</th>
      <th>v3</th>
      <th>Full Dolmino pool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Math (synth)</td>
      <td><code>MegaMatt</code></td>
      <td>1.73B</td>
      <td>1.73B</td>
      <td>3.88B</td>
      <td>3.88B</td>
    </tr>
    <tr>
      <td>Web pages</td>
      <td><code>STEM-Heavy Crawl</code></td>
      <td>4.99B</td>
      <td>5.21B</td>
      <td>5.21B</td>
      <td>5.21B</td>
    </tr>
    <tr>
      <td rowspan="3">Thinking (synth)</td>
      <td><code>General Reasoning Mix</code></td>
      <td>1.87B</td>
      <td>1.87B</td>
      <td>2.48B</td>
      <td>2.48B</td>
    </tr>
    <tr>
      <td><code>Math Meta-Reasoning</code></td>
      <td>381M</td>
      <td>381M</td>
      <td>1.05B</td>
      <td>1.05B</td>
    </tr>
    <tr>
      <td><code>Code Meta-Reasoning</code></td>
      <td>459M</td>
      <td>459M</td>
      <td>1.27B</td>
      <td>1.27B</td>
    </tr>
    <tr>
      <td colspan="2">All other subsets</td>
      <td>unchanged</td>
      <td>unchanged</td>
      <td>unchanged</td>
      <td>&mdash;</td>
    </tr>
    <tr>
      <td colspan="2"><strong>Total mix</strong></td>
      <td><strong>99.95B</strong></td>
      <td><strong>100.17B</strong></td>
      <td><strong>104.41B</strong></td>
      <td>&mdash;</td>
    </tr>
  </tbody>
</table>


### Long-context Extension
In Phase 1, we use a general long-context mixture [Dolma 3 Longmino Mix](https://huggingface.co/datasets/allenai/dolma3_longmino_mix-100B-1125) to teach the model to attend across the full 64K window.

In Phase 2, we focus on recovering the model's performance on math, coding and reasoning. We train the model for 2,400 steps on a curated data mixture with data drawn from the [Dolma 3 Dolmino 100B Mix](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1125), the full [Dolma 3 Dolmino pool](https://huggingface.co/datasets/allenai/dolma3_dolmino_pool), and [Instella-GSM8K-synthetic](https://huggingface.co/datasets/amd/Instella-GSM8K-synthetic). Data samples are concatenated to form 64K-token training samples.

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Dataset </th>
      <th>Source</th>
      <th>Tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Math </td>
      <td><code>dolmino-math</code></td>
      <td>100B mix</td>
      <td>10.7B</td>
    </tr>
    <tr>
      <td><code>cranemath</code></td>
      <td>100B mix</td>
      <td>5.62B</td>
    </tr>
    <tr>
      <td><code>megamatt</code></td>
      <td>Full pool</td>
      <td>3.88B</td>
    </tr>
    <tr>
      <td><code>tinymath-mind</code></td>
      <td>100B mix</td>
      <td>898M</td>
    </tr>
    <tr>
      <td><code>tinymath-pot</code></td>
      <td>100B mix</td>
      <td>241M</td>
    </tr>
    <tr>
      <td><code>instella-gsm8k-synthetic</code></td>
      <td>Instella-GSM8K</td>
      <td>329M</td>
    </tr>
    <tr>
      <td>Code</td>
      <td><code>cranecode</code></td>
      <td>100B mix</td>
      <td>10.0B</td>
    </tr>
    <tr>
      <td rowspan="4">Thinking</td>
      <td><code>general_reasoning_mix</code></td>
      <td>Full pool</td>
      <td>2.48B</td>
    </tr>
    <tr>
      <td><code>omr-rewrite-fullthoughts</code></td>
      <td>100B mix</td>
      <td>850M</td>
    </tr>
    <tr>
      <td><code>math-meta-reasoning</code></td>
      <td>Full pool</td>
      <td>1.05B</td>
    </tr>
    <tr>
      <td><code>code-meta-reasoning</code></td>
      <td>Full pool</td>
      <td>1.27B</td>
    </tr>
    <tr>
      <td colspan="3"><strong>Total mix</strong></td>
      <td><strong>37.32B</strong></td>
    </tr>
  </tbody>
</table>

### SFT
The phase-1 SFT mixture combines a general [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) dataset with three targeted skill slices:

| Domain | Source | Records | What it adds |
|--------|--------|---------|--------------|
| General | [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) | ~2.27M | General instruction-following base |
| Math | [Nemotron-Cascade-2](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) (math) | 300K | Math, reservoir-sampled (`DeepSeek-V3.2-Speciale`, reasoning-prefix stripped) |
| Code | [Nemotron-SFT-Competitive-Programming-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2) (Python) | ~161K | Coding; all records that survive the 32k filter (~52% are too long) |
| Science | [Nemotron-Cascade-2](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) (science) | ~197K | Science (`GPT-OSS-120B`, 200K sampled → 196,874 kept) |

Every source is exported to the same `{"messages": [...]}` JSONL format (system messages dropped, assistant `<think>...</think>` reasoning traces kept) and filtered to ≤ 32,768 tokens under the DeepSeek-V3 chat template.

Phase 2 anneals onto a feedback-driven curated data mixture of 512K examples selected from a candidate SFT data pool to target the phase-1 checkpoint's weaknesses. The candidate pool consists of [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B), [Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) and [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data). See [`data_curation`](../training/examples/scripts/instella_moe_data_preparation/data_curation) for details.

### DPO
We use the contrastive preference data pairs from [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B).

### RL
We use the [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B) dataset for the reinforcement learning stage. In the IF-RL stage, we use the `IF RLVR Mixture` to train the instruction-following expert. For the MOPD stage, we mix the `IF RLVR Mixture` with the general prompts (`--if-frac 0.5`); see [Prepare RL Data](#prepare-rl-data) below.

## Processing scripts
For most stages, the training data is prepared in two steps. First each raw dataset is exported into line-delimited JSONL, then the JSONL is tokenized into the Megatron indexed dataset format read by the training scripts. 
All preparation scripts are available in [`training/examples/scripts/instella_moe_data_preparation`](../training/examples/scripts/instella_moe_data_preparation). For SFT and DPO the trainer consumes JSONL datasets directly and tokenizes on the fly.

### Jsonl to Megatron indexed dataset
Pre-training and mid-training data is used as a Megatron indexed dataset (a `.bin`/`.idx` pair). 
To tokenize a directory of `*.jsonl` files into this format run [`examples/megatron/preprocess_data.py`](../training/examples/megatron/preprocess_data.py). 
Or use the ready-to-use wrapper at [`training/examples/scripts/instella_moe_data_preparation/tokenize_dataset.sh`](../training/examples/scripts/instella_moe_data_preparation/tokenize_dataset.sh).

```bash
# Directory of *.jsonl files (downloaded from HF, e.g. instella_gsm8k_synthetic, 'allenai/dolma3_dolmino_mix-100B-1125' etc.)

python "$PRIMUS_PATH/examples/megatron/preprocess_data.py" \
  --input "$JSONL_DIR/*.jsonl" \
  --output-prefix "$OUTPUT_PREFIX" \
  --tokenizer-type DeepSeekV3Tokenizer \
  --tokenizer-model deepseek-ai/DeepSeek-V3 \
  --append-eod \
  --workers 64 \
  --partitions 8
```

### Prepare Instella-GSM8K-synthetic
[`prepare_instella_gsm8k_synthetic.py`](../training/examples/scripts/instella_moe_data_preparation/prepare_instella_gsm8k_synthetic.py) converts the HF download of `amd/Instella-GSM8K-synthetic` (the `train-*.parquet` files) into `{"text": ...}` JSONL. 
We flatten the user question followed by the assistant answer separated by a newline, and only the `train` split is used, the processing can be run with:

```bash
python prepare_instella_gsm8k_synthetic.py \
  --input-dir /path/to/Instella-GSM8K-synthetic \
  --output-dir /path/to/dataset/data_jsonl
```

We then tokenize the resulting JSONL using the command above.

### Prepare SFT and DPO Data
The Dolci SFT and DPO datasets are exported directly to the JSONL layout expected by the `sft` and `dpo` training tasks.
They are read as-is by the SFT and DPO trainers and do not pass through [`preprocess_data.py`](../training/examples/megatron/preprocess_data.py).

[`prepare_sft_dolci.py`](../training/examples/scripts/instella_moe_data_preparation/prepare_sft_dolci.py) exports [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B) to shuffled `{"messages": [...]}` JSONL.

```bash
python prepare_sft_dolci.py \
  --output /path/to/dataset/dolci-think-sft-7b.jsonl \
  --cache-dir /path/to/cache
```

The phase-1 math/code/science augmentation is produced by a two-step pipeline. [`prepare_nemotron_sft.py`](../training/examples/scripts/instella_moe_data_preparation/prepare_nemotron_sft.py) reformats a raw Nemotron subset (`--subset math|cp|science`) into the shared `{"messages": [...]}` schema — dropping `system` messages, keeping assistant `<think>...</think>` reasoning, and applying the per-subset generator filter, math prefix stripping, `reasoning_content` merge, and reservoir sampling. [`filter_32k.py`](../training/examples/scripts/instella_moe_data_preparation/filter_32k.py) then keeps only records whose `apply_chat_template` output is `≤ 32,768` tokens (with optional `--target-count` subsampling).

[`prepare_sft_phase1.sh`](../training/examples/scripts/instella_moe_data_preparation/prepare_sft_phase1.sh) runs the whole thing end-to-end — download, reformat, filter, and merge — producing the two JSONL files referenced by the phase-1 SFT config:

```bash
# Outputs dolci-full-nemotron-math-300k-cp-161k-max32k.jsonl and
# nemotron-cascade2-science-gpt-oss-120b-200k-max32k.jsonl into OUT_DIR.
OUT_DIR=/path/to/dataset/sft RAW_DIR=/path/to/cache \
  bash prepare_sft_phase1.sh
```

[`prepare_dpo_dolci.py`](../training/examples/scripts/instella_moe_data_preparation/prepare_dpo_dolci.py) exports [Dolci-Think-DPO-7B](https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B) to interleaved `{"chosen": [...], "rejected": [...]}` JSONL. 
We also support sampling to specific sources, for example the instruction-following slices via `--preset if-only`.

```bash
python prepare_dpo_dolci.py \
  --output /path/to/dataset/dolci-think-dpo-7b.jsonl \
  --cache-dir /path/to/cache
```

### Prepare RL Data 
Both prompt sets are exported from Dolci-Think-RL by [`prepare_rl_dolci.py`](../rl/examples/mopd/prepare_rl_dolci.py). Pass `--if-only` to export just the `IF RLVR Mixture` (the `IF_multi_constraints` slice) for the IF-RL stage:

```bash
python rl/examples/mopd/prepare_rl_dolci.py \
  --dataset   allenai/Dolci-Think-RL-7B \
  --output    /path/to/dolci_think_rl_if.jsonl \
  --cache-dir /path/to/cache \
  --if-only --seed 0
```

For the MOPD stage, [`prepare_rl_dolci.py`](../rl/examples/mopd/prepare_rl_dolci.py) builds the domain-tagged mixed prompt set directly from [Dolci-Think-RL-7B](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B). Each record is tagged with `metadata.domain` (`if` when its `dataset_source` is the `IF_multi_constraints` slice, else `general`) so the reward can route it to the IF-RL or the frozen DPO anchor teacher. The IF fraction `--if-frac` is set to `0.5` in our final experiment.

```bash
python rl/examples/mopd/prepare_rl_dolci.py \
  --dataset   allenai/Dolci-Think-RL-7B \
  --output    /path/to/opd_mixed_prompts.jsonl \
  --cache-dir /path/to/cache \
  --if-frac 0.5 --seed 0
```
