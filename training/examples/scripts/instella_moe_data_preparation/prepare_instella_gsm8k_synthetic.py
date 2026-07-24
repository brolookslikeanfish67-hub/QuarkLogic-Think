###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Instella-GSM8K-synthetic parquet to {'text': ...} jsonl."
    )
    parser.add_argument(
        "--input-dir",
        default=os.environ.get("INPUT_DIR"),
        help="Directory with train-*.parquet (HF download of 'amd/Instella-GSM8K-synthetic').",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR"),
        help="Directory to write .jsonl files (default: <input-dir>/../data_jsonl).",
    )
    args = parser.parse_args()
    if not args.input_dir:
        parser.error("--input-dir or INPUT_DIR env var is required")
    return args


def extract_qa_text(messages) -> str:
    """Extract question and answer from a conversation and format as 'Question\nAnswer'."""
    if isinstance(messages, str):
        messages = json.loads(messages)

    question = ""
    answer = ""
    for msg in messages:
        if msg["role"] == "user":
            question = msg["content"].strip()
        elif msg["role"] == "assistant":
            answer = msg["content"].strip()

    return f"{question}\n{answer}"


def process_parquet_file(args):
    """Process a single parquet file: extract Q/A pairs and convert to jsonl format."""
    input_file, output_file, messages_col = args

    try:
        df = pd.read_parquet(input_file)
        original_count = len(df)

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                text = extract_qa_text(row[messages_col])
                json_line = json.dumps({'text': text}, ensure_ascii=False)
                f.write(json_line + '\n')

        return {
            'file': str(input_file.name),
            'rows': original_count,
            'error': None
        }
    except Exception as e:
        return {
            'file': str(input_file.name),
            'rows': 0,
            'error': str(e)
        }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "data_jsonl"

    # Find parquet files for the 'train' split only (exclude 'train_119k' split)
    print(f"Scanning for parquet files in {input_dir}...")
    parquet_files = sorted([
        f for f in input_dir.glob("*.parquet")
        if f.name.startswith("train-")
    ])
    print(f"Found {len(parquet_files)} parquet files (train split only).")

    if not parquet_files:
        print("No parquet files found. Exiting.")
        return

    # Auto-detect the messages column name
    sample_df = pd.read_parquet(parquet_files[0]).head(1)
    print(f"Columns in dataset: {list(sample_df.columns)}")
    messages_col = None
    for candidate in ["messages", "conversations", "conversation", "chat"]:
        if candidate in sample_df.columns:
            messages_col = candidate
            break
    if messages_col is None:
        raise ValueError(
            f"Could not find a messages column. Available columns: {list(sample_df.columns)}"
        )
    print(f"Using messages column: '{messages_col}'")

    # Preview first sample
    first_text = extract_qa_text(sample_df[messages_col].iloc[0])
    print(f"\n--- Sample output ---\n{first_text}\n--- End sample ---\n")

    # Create tasks
    tasks = []
    for pf in parquet_files:
        output_file = output_dir / pf.with_suffix('.jsonl').name
        tasks.append((pf, output_file, messages_col))

    # Process files in parallel
    max_workers = min(32, len(tasks))
    print(f"Processing files with {max_workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_parquet_file, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            result = future.result()
            results.append(result)
            if result['error']:
                print(f"Error processing {result['file']}: {result['error']}")

    # Aggregate statistics
    total_rows = sum(r['rows'] for r in results)
    errors = [r for r in results if r['error']]

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Files processed: {len(results)}")
    if errors:
        print(f"Files with errors: {len(errors)}")
    print("-" * 80)
    print(f"Total rows: {total_rows:>12,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
