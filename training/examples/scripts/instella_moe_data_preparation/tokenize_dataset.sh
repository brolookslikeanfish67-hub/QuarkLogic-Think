#!/bin/bash
set -e

# Directory of *.jsonl files (downloaded from HF, e.g.instella_gsm8k_synthetic 'allenai/dolma3_dolmino_mix-100B-1125' etc.)
JSONL_DIR=/path/to/dataset/data_jsonl
OUTPUT_PREFIX=/path/to/dataset/data_tokenized/data

PRIMUS_PATH=${PRIMUS_PATH:-$(realpath "$(dirname "$0")/../../..")}

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

python "$PRIMUS_PATH/examples/megatron/preprocess_data.py" \
  --input "$JSONL_DIR/*.jsonl" \
  --output-prefix "$OUTPUT_PREFIX" \
  --tokenizer-type DeepSeekV3Tokenizer \
  --tokenizer-model deepseek-ai/DeepSeek-V3 \
  --append-eod \
  --workers 64 \
  --partitions 8
