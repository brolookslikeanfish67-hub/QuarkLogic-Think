#!/bin/bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Build the two JSONL files consumed by the Instella phase-1 SFT config:
#
#   1) dolci-full-nemotron-math-300k-cp-161k-max32k.jsonl
#        = Dolci-Think-SFT-7B  +  math 300k (<=32k)  +  CP-Python all (<=32k)
#   2) nemotron-cascade2-science-gpt-oss-120b-200k-max32k.jsonl
#        = science 200k sample (<=32k)
#
# Each raw subset is downloaded from HF, reformatted with prepare_nemotron_sft.py,
# filtered to <=32k tokens with filter_32k.py, then merged with `cat`.
#
# Set the paths via environment variables, e.g.:
#   OUT_DIR=/my/sft RAW_DIR=/my/raw bash prepare_sft_phase1.sh
set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")

# --- paths (must be provided via env) ----------------------------------------
RAW_DIR=${RAW_DIR:?set RAW_DIR to where raw HF downloads should land}
OUT_DIR=${OUT_DIR:?set OUT_DIR to where pipeline outputs should be written}
CACHE_DIR=${CACHE_DIR:-$RAW_DIR}                          # HF_HOME for dolci export
TOKENIZER=${TOKENIZER:-deepseek-ai/DeepSeek-V3}
WORKERS=${WORKERS:-64}

CASCADE=$RAW_DIR/nemotron-cascade2-sft
CP=$RAW_DIR/nemotron-competitive-programming-v2

mkdir -p "$OUT_DIR"

# --- 0. download raw datasets (one-time; resumable) --------------------------
echo "=== Downloading raw datasets from HuggingFace ==="
huggingface-cli download --repo-type dataset nvidia/Nemotron-Cascade-2-SFT-Data \
  math/math_notool.jsonl science/science.jsonl \
  --local-dir "$CASCADE" --resume-download
huggingface-cli download --repo-type dataset nvidia/Nemotron-SFT-Competitive-Programming-v2 \
  data/competitive_programming_python_00.jsonl \
  data/competitive_programming_python_01.jsonl \
  --local-dir "$CP" --resume-download

# --- 1. Dolci base (2.27M) ---------------------------------------------------
echo "=== Exporting Dolci-Think-SFT-7B base ==="
python "$SCRIPT_DIR/prepare_sft_dolci.py" \
  --dataset allenai/Dolci-Think-SFT-7B \
  --output "$OUT_DIR/dolci-think-sft-7b-full.jsonl" \
  --cache-dir "$CACHE_DIR"

# --- 2. math: 380k sample -> 300k @ <=32k ------------------------------------
echo "=== Math ==="
python "$SCRIPT_DIR/prepare_nemotron_sft.py" --subset math \
  --input "$CASCADE/math/math_notool.jsonl" \
  --output "$OUT_DIR/nemotron-math-380k.jsonl"
python "$SCRIPT_DIR/filter_32k.py" \
  --input "$OUT_DIR/nemotron-math-380k.jsonl" \
  --output "$OUT_DIR/nemotron-math-300k-max32k.jsonl" \
  --target-count 300000 --tokenizer-model "$TOKENIZER" --workers "$WORKERS"

# --- 3. CP-Python: all -> all @ <=32k (~161k) --------------------------------
echo "=== Competitive Programming (Python) ==="
python "$SCRIPT_DIR/prepare_nemotron_sft.py" --subset cp \
  --input "$CP/data/competitive_programming_python_00.jsonl" \
          "$CP/data/competitive_programming_python_01.jsonl" \
  --output "$OUT_DIR/nemotron-cp-python-all.jsonl"
python "$SCRIPT_DIR/filter_32k.py" \
  --input "$OUT_DIR/nemotron-cp-python-all.jsonl" \
  --output "$OUT_DIR/nemotron-cp-python-all-max32k.jsonl" \
  --tokenizer-model "$TOKENIZER" --workers "$WORKERS"

# --- 4. science: 200k sample -> @ <=32k --------------------------------------
echo "=== Science ==="
python "$SCRIPT_DIR/prepare_nemotron_sft.py" --subset science \
  --input "$CASCADE/science/science.jsonl" \
  --output "$OUT_DIR/nemotron-science-200k.jsonl"
python "$SCRIPT_DIR/filter_32k.py" \
  --input "$OUT_DIR/nemotron-science-200k.jsonl" \
  --output "$OUT_DIR/nemotron-cascade2-science-gpt-oss-120b-200k-max32k.jsonl" \
  --tokenizer-model "$TOKENIZER" --workers "$WORKERS"

# --- 5. merge file 1 (dolci + math + cp) -------------------------------------
echo "=== Merging dolci + math + CP-Python ==="
cat \
  "$OUT_DIR/dolci-think-sft-7b-full.jsonl" \
  "$OUT_DIR/nemotron-math-300k-max32k.jsonl" \
  "$OUT_DIR/nemotron-cp-python-all-max32k.jsonl" \
  > "$OUT_DIR/dolci-full-nemotron-math-300k-cp-161k-max32k.jsonl"

echo
echo "Done. Phase-1 SFT inputs written to $OUT_DIR:"
echo "  dolci-full-nemotron-math-300k-cp-161k-max32k.jsonl"
echo "  nemotron-cascade2-science-gpt-oss-120b-200k-max32k.jsonl"
