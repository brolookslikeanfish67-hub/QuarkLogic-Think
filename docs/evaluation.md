# Evaluation
The models are evaluated using the [OLMES](https://github.com/allenai/olmes) framework. Our evaluation settings are unified across models.

## Base Model Evaluation
The following task configs were used for evaluating the base models:

| Benchmark Name | OLMES Task Name | Context | Primary Metric |
|---|---|---|---|
| **Short-context** | | | |
| BoolQ | `boolq:mc::olmes` | ≤4K | `acc_raw` |
| OpenBookQA | `openbookqa:mc::olmes` | ≤4K | `acc_raw` |
| MMLU | `mmlu:mc::olmes` | ≤4K | `macro` |
| GSM8K | `gsm8k::olmo3:n8:v2` | ≤4K | `pass_at_1` |
| HumanEval+ | `codex_humaneval:3shot::olmo3:n32:v2` | ≤4K | `pass_at_1` |
| MBPP+ | `mbpp:3shot::olmo3:n32:v2` | ≤4K | `pass_at_1` |
| Minerva MATH | `minerva_math::olmes:n4:v2` | ≤4K | `macro` |
| ARC | `arc:mc::xlarge` | ≤4K | `macro` |
| SciQ | `sciq:mc::xlarge` | ≤4K | `acc_per_char` |
| PIQA | `piqa:mc::xlarge` | ≤4K | `acc_raw` |
| HellaSwag | `hellaswag:rc::xlarge` | ≤4K | `acc_per_char` |
| WinoGrande | `winogrande:rc::xlarge` | ≤4K | `acc_raw` |
| **Long-context** | | | |
| HELMET @ 8K | `helmet_all__8192::suite` | 8K | `macro` |
| HELMET @ 16K | `helmet_all__16384::suite` | 16K | `macro` |
| HELMET @ 32K | `helmet_all__32768::suite` | 32K | `macro` |
| HELMET @ 64K | `helmet_all__65536::suite` | 64K | `macro` |
| RULER @ 8K | `ruler_all__8192::suite` | 8K | `macro` |
| RULER @ 16K | `ruler_all__16384::suite` | 16K | `macro` |
| RULER @ 32K | `ruler_all__32768::suite` | 32K | `macro` |
| RULER @ 64K | `ruler_all__65536::suite` | 64K | `macro` |

 

## Post-trained Model Evaluation
For model checkpoints in the post-training stages, we evaluate the model using the following task variants. Each task is run with a per-task context length: reasoning-heavy benchmarks use a 32K context window, while shorter-output benchmarks use 16K. 

| Benchmark Name | OLMES Task Name | Context | Primary Metric |
|---|---|---|---|
| AGIEval (English) | `agi_eval_english::olmo3:adapt` | 16K | `macro` |
| MMLU (CoT) | `mmlu:cot::olmo3:adapt` | 16K | `macro` |
| BBH (CoT) | `bbh:cot::olmo3:adapt` | 16K | `micro` |
| MBPP+ | `mbppplus::olmo3:adapt` | 16K | `pass@1` |
| GPQA | `gpqa::olmo3:adapt` | 16K | `exact_match` |
| HumanEval+ | `codex_humanevalplus::olmo3:adapt` | 16K | `pass@1` |
| Minerva MATH | `minerva_math::olmo3:adapt` | 16K | `macro` |
| AIME 2024 | `aime:2024::olmo3:adapt` | 32K | `pass@1` |
| AIME 2025 | `aime:2025::olmo3:adapt` | 32K | `pass@1` |
| IFEval | `ifeval::olmo3:adapt` | 32K | `prompt_level_loose_acc` |
| LiveCodeBench (v6) | `livecodebench_codegeneration::olmo3:adapt-v6` | 32K | `pass@1` |
| AlpacaEval v3* | `alpaca_eval_v3::olmo3:adapt` | 32K | `length_controlled_winrate` |
