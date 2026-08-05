#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Prompt-optimized data curation for student-model fine-tuning.

Workflow (one iteration, see ``main``): a large training pool is domain-tagged
(``tag_domains``) and embedded once (``embed_pool``); held-out AIME/coding
evaluation sets are built (``build_seed_scoring_sets``). The current student
checkpoint is run and judged to produce error analytics
(``pretrain_inference_and_judge``). A reflection LM turns those errors into a
weighted, domain-tagged selection policy (``run_policy_mode``), which drives
domain-aware greedy retrieval of new training data from the pool
(``run_select_mode``). The student is fine-tuned on the curated data and rescored
(``train_and_eval``). Repeat with the improved checkpoint.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DOMAINS = ["MATH", "CODE", "OTHER"]
DOMAIN_BUDGET_FRACTIONS_3WAY = {"MATH": 0.39279, "CODE": 0.31503, "OTHER": 0.29218}
TAG_TO_DOMAIN_3WAY = {
    "MATH": "MATH",
    "COMPETITIVE_CODING": "CODE",
    "GENERAL_CODING": "CODE",
    "OTHER": "OTHER",
}

def _env(name: str, default: str = "") -> str:
    """Return environment variable ``name`` if set and non-empty, else ``default``."""
    val = os.environ.get(name)
    return val if val else default


LLM_LARGE = _env("LLM_LARGE", "LLM_LARGE")
LLM_SMALL = _env("LLM_SMALL", "LLM_SMALL")
STUDENT_MODEL = _env("STUDENT_MODEL", "STUDENT_MODEL")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "EMBEDDING_MODEL")
TOKENIZER_NAME = _env("TOKENIZER_NAME", "")  # for selection length filter; empty -> heuristic
LLM_BASE_URL = _env("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = _env("LLM_API_KEY", "EMPTY")
EVAL_MAX_TOKENS = int(_env("EVAL_MAX_TOKENS", "16000"))

STUDENT_BASE_URL = _env("STUDENT_BASE_URL", LLM_BASE_URL)
STUDENT_API_KEY = _env("STUDENT_API_KEY", LLM_API_KEY)
JUDGE_BASE_URL = _env("JUDGE_BASE_URL", LLM_BASE_URL)
JUDGE_API_KEY = _env("JUDGE_API_KEY", LLM_API_KEY)
REFLECTION_BASE_URL = _env("REFLECTION_BASE_URL", LLM_BASE_URL)
REFLECTION_API_KEY = _env("REFLECTION_API_KEY", LLM_API_KEY)
REFLECTION_MAX_TOKENS = int(_env("REFLECTION_MAX_TOKENS", "65536"))
REFLECTION_TEMPERATURE = float(_env("REFLECTION_TEMPERATURE", "0.7"))

# Student serving (used when no live STUDENT_BASE_URL is provided): the pipeline
# launches a stock OpenAI-compatible SGLang server from STUDENT_CHECKPOINT, then
# tears it down when done. Serving knobs are env-configurable with portable
# defaults; STUDENT_SERVE_EXTRA_ARGS lets you pass build-specific flags (e.g. the
# AMD FarSkip flags required to serve Instella-MoE) without editing this file.
STUDENT_CHECKPOINT = _env("STUDENT_CHECKPOINT", "")
STUDENT_SERVE_PORT = int(_env("STUDENT_SERVE_PORT", "30000"))
STUDENT_SERVE_TP = int(_env("STUDENT_SERVE_TP", "1"))
STUDENT_SERVE_MEM_FRACTION = _env("STUDENT_SERVE_MEM_FRACTION", "0.85")
STUDENT_SERVE_EXTRA_ARGS = _env("STUDENT_SERVE_EXTRA_ARGS", "")
STUDENT_SERVE_TIMEOUT = int(_env("STUDENT_SERVE_TIMEOUT", "900"))

# Step 7 (train + convert): the trainer and checkpoint converter are cluster
# tooling, so the release shells out to the repo's own scripts rather than baking
# in any internal path. All settings are env-configurable. Set SKIP_TRAINING=1 to
# run only the evaluation half against an existing checkpoint (STUDENT_CHECKPOINT)
# or a live STUDENT_BASE_URL — the "eval without training plumbing" path.
SFT_BASE_CONFIG = _env("SFT_BASE_CONFIG", "")           # base SFT YAML to clone
SFT_CONFIG_OUT = _env("SFT_CONFIG_OUT", "sft_curated.yaml")
SFT_SAVE_DIR = _env("SFT_SAVE_DIR", "")                 # trainer save dir (overrides 'save')
PRIMUS_DIR = _env("PRIMUS_DIR", ".")                    # Primus checkout root
SFT_LAUNCH_SCRIPT = _env("SFT_LAUNCH_SCRIPT", "./examples/run_instella.sh --task sft")
CONVERT_SCRIPT = _env("CONVERT_SCRIPT", "convert_megatron_to_hf.py")
CONVERT_MODEL_NAME = _env("CONVERT_MODEL_NAME", "deepseekv3")
ORIGIN_HF_DIR = _env("ORIGIN_HF_DIR", "")               # tokenizer/config source for conversion
HF_OUTPUT_ROOT = _env("HF_OUTPUT_ROOT", "hf_out")       # where converted checkpoints land
SKIP_TRAINING = bool(_env("SKIP_TRAINING", ""))

REFLECTION_MODEL = LLM_LARGE
FEEDBACK_MODEL = LLM_SMALL

CLASSIFY_PROMPT = (
    "Classify the following question into exactly one of four labels:\n"
    "- MATH: pure mathematics — solving equations, proofs, geometry, number theory, "
    "calculus, combinatorics, probability. The answer is a number, expression, or proof. "
    "No code is required.\n"
    "- COMPETITIVE_CODING: the question asks to write a program or algorithm to solve a "
    "problem with input/output format, constraints (like N <= 10^5), time limits, or "
    "test cases. Includes problems from Codeforces, LeetCode, AtCoder, USACO, IOI, ACM-ICPC. "
    "Also includes questions about arrays, graphs, trees, dynamic programming, greedy algorithms "
    "where the goal is to write efficient code.\n"
    "- GENERAL_CODING: standard programming tasks — implement a function, fix a bug, "
    "explain code, write a script, use an API, data processing, web development, etc. "
    "NOT algorithmic contest problems.\n"
    "- OTHER: anything else (science, history, language, general knowledge, etc.)\n\n"
    "Key rule: if the question mentions constraints like N<=10^5, asks for an algorithm, "
    "mentions input/output format, or looks like a contest problem, it is COMPETITIVE_CODING "
    "even if it involves math.\n\n"
    "Return exactly one label, uppercase with underscores, no punctuation, no explanation. "
    "If unsure, return OTHER.\n\nQuestion: "
)

VALID_LABELS = {"MATH", "COMPETITIVE_CODING", "GENERAL_CODING", "OTHER"}

EMBEDDING_INSTRUCTION = (
    "Given a description of desired training data characteristics, "
    "retrieve documents that match those characteristics. "
    "Characteristics include subject area, document type, cognitive level, "
    "reasoning depth, technical correctness, education level, and content quality."
)

JUDGE_SYSTEM = """\
You are a strict answer verifier. You will see a question, a reference (gold) \
solution, and a student's solution. Determine whether the student arrived at \
the correct final answer or output.

The task may be math, coding, science, instruction following, creative writing, \
or anything else. Judge correctness based on the task type:
  - Math/science: equivalent final answers count as correct (0.5 = 1/2, etc.)
  - Code: output must be functionally equivalent (ignore whitespace, variable names)
  - Instruction following: the student must satisfy the stated requirements
  - Open-ended: use the reference as a guide; accept reasonable equivalents

Respond with EXACTLY one JSON object on a single line:
{"correct": true}  or  {"correct": false, "reason": "<brief explanation>"}"""

REFLECTION_SYSTEM_PROMPT = """\
You are optimizing a data selection policy for training a student AI model.

The policy is a list of (prompt, weight) pairs:
  - Each "prompt" describes what kind of training data to select from a large pool
    via embedding similarity search.
  - The "weight" controls what fraction of the training budget goes to that prompt.
  - Weights must sum to <= 1.0. The remainder is filled with randomly selected data.

Based on the error analytics provided, propose a data selection policy. You should:
  - ADD prompts to cover the error patterns observed
  - ADJUST weights to reflect how frequently each pattern appears

Guidelines:
  - Focus on what recurs across errors. A prompt can target a recurring
    domain (e.g. "contract law case analysis"), a recurring skill
    (e.g. "multi-step unit conversions"), or both if the cluster is tight
    enough (e.g. "thermodynamics with unit conversions between energy systems").
  - Each prompt should cover MULTIPLE related errors, not a single problem.
  - Aim for ~25 prompts. Together they should cover all the major error
    clusters, not just the most frequent ones.
  - Look at the error examples to identify PATTERNS, not individual failures.
  - Give higher weights to patterns that appear more frequently in the errors.
  - Each prompt should describe training data in terms of TOPIC (e.g. "number
    theory", "graph algorithms"), SKILLS (e.g. "modular arithmetic, CRT"),
    and COMPLEXITY (e.g. "multi-step reasoning with case analysis").
    Avoid vague or meta-skill prompts like "write correct code",
    "ensure proper formatting", or "verify final answers".

Explain your reasoning, then provide the policy as a JSON array:

```json
[
  {"prompt": "description of desired training data", "weight": 0.XX},
  ...
]
```"""

FEEDBACK_SYSTEM = """\
You are analyzing why a student AI model produced a wrong answer. The task could \
be math, coding, science, instruction following, creative writing, or anything \
else. Given the question, reference solution, and the student's incorrect output, \
provide detailed structured feedback. Be verbose and descriptive in every field — \
write in natural language, not just labels.

Respond with EXACTLY one JSON object:
{
  "domain": "<broad area: math, code, science, instruction_following, creative, general, etc.>",
  "sub_domain": "<specific sub-field within the domain, e.g. 'combinatorics and counting', 'dynamic programming', 'thermodynamics', 'output format compliance'>",
  "concept": "<describe the specific concept or skill being tested in plain English, e.g. 'applying the inclusion-exclusion principle to count overlapping sets', 'handling recursive base cases in tree traversal', 'following multi-part formatting instructions precisely'>",
  "reasoning_complexity": "<describe what level of reasoning this task requires and why, e.g. 'multi-step: requires setting up equations from word problem, solving the system, then verifying constraints', 'single-step: direct formula application', 'multi-step with backtracking: requires trying cases and eliminating invalid ones'>",
  "error_type": "<describe what went wrong in a short phrase, e.g. 'confused area formula with circumference formula', 'off-by-one in loop boundary and wrong conditional ordering', 'ignored explicit bullet point and verb-first requirements'>",
  "skills_lacking": "<describe in detail what knowledge or abilities the student is missing, e.g. 'the student needs a firmer grasp of when to apply πr² vs 2πr, and more generally how to distinguish between area and perimeter formulas across shapes'>",
  "what_would_help": "<describe what kind of training data would help the student get better at this, written as a description suitable for retrieving similar examples from a large pool, e.g. 'geometry problems that require choosing between area, perimeter, and volume formulas for circles, rectangles, and spheres, with explicit worked solutions showing formula selection reasoning'>",
  "summary": "<2-4 sentence detailed explanation of what went wrong, why the student's approach failed, and what the correct approach would have been>"
}"""

FEEDBACK_SYSTEM_CODING = """\
You are analyzing why a student AI model produced a wrong answer on a CODING task. \
All tasks in this set require writing code to solve — including scripts, automation, \
API usage, data processing, file manipulation, web development, and system \
administration tasks. Even if a task looks like "instruction following", the \
solution requires functional code.

Given the question, reference solution, and the student's incorrect output, \
provide detailed structured feedback focused on the coding skills needed. \
Be verbose and descriptive in every field — write in natural language, not labels.

Respond with EXACTLY one JSON object:
{
  "domain": "code",
  "sub_domain": "<specific coding sub-field, e.g. 'string processing', 'file I/O and scripting', 'API integration', 'data parsing and transformation', 'web scraping', 'command-line tool usage', 'database queries', 'text encoding and Unicode handling'>",
  "concept": "<describe the specific coding concept or skill being tested, e.g. 'reading CSV files and computing column statistics with pandas', 'writing a bash script to recursively process files with pandoc', 'implementing UTF-16 endianness detection by analyzing byte order marks'>",
  "reasoning_complexity": "<describe what level of coding reasoning this task requires, e.g. 'multi-step: requires parsing input format, building data structure, applying transformation, formatting output', 'single-step: direct library call with correct parameters'>",
  "error_type": "<describe what went wrong in coding terms, e.g. 'wrong library function used for file traversal', 'failed to handle edge case in string encoding', 'produced pseudocode instead of executable code', 'incorrect regex pattern for parsing'>",
  "skills_lacking": "<describe what coding knowledge or abilities the student is missing, e.g. 'the student needs practice with Python's os.walk for recursive directory traversal and subprocess for calling external tools'>",
  "what_would_help": "<describe what kind of coding training data would help, e.g. 'Python scripts that automate file format conversion using subprocess and os.path, with complete working examples showing error handling and edge cases'>",
  "summary": "<2-4 sentence explanation of what went wrong in coding terms — what the code should have done, what the student produced instead, and what coding approach would have been correct>"
}"""


def _extract_text_for_embedding(messages: list) -> str:
    """Extract the user question and final answer (content after </think>)."""
    user_text, assistant_text = "", ""
    for msg in messages:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":
            user_text = content
        elif role == "assistant":
            idx = content.find("</think>")
            assistant_text = content[idx + len("</think>"):].strip() if idx != -1 else content.strip()
    return f"{user_text}\n\n{assistant_text}" if assistant_text else user_text


def _format_for_embedding(texts: list[str]) -> list[str]:
    """Prepend the retrieval task instruction to each text."""
    return [f"Instruct: {EMBEDDING_INSTRUCTION}\nQuery: {t}" for t in texts]


_CLIENTS: dict = {}


def _client(base_url: str | None = None, api_key: str | None = None):
    base_url = base_url or LLM_BASE_URL
    api_key = api_key or LLM_API_KEY
    key = (base_url, api_key)
    if key not in _CLIENTS:
        from openai import OpenAI

        _CLIENTS[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _CLIENTS[key]


def chat_completion(
    model: str, prompt: str, system: str | None = None, max_tokens: int = 5,
    temperature: float = 0.0, base_url: str | None = None, api_key: str | None = None,
) -> str:
    """Send a chat request to a served model and return its text."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    resp = _client(base_url, api_key).chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message.content or ""


def _loads_lenient(raw: str) -> dict:
    """Parse a JSON object, tolerating invalid backslash escapes from weak models."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw))


def chat_json(
    model: str, system: str, user: str,
    base_url: str | None = None, api_key: str | None = None,
) -> dict:
    """Send a system+user chat request and parse the JSON object from the reply."""
    content = chat_completion(
        model, user, system=system, max_tokens=1024,
        base_url=base_url, api_key=api_key,
    )
    return _loads_lenient(re.search(r"\{.*\}", content, re.DOTALL).group(0))


def classify_domain(model: str, prompt: str) -> str:
    """Return one of VALID_LABELS for a question (OTHER if the reply is unrecognized)."""
    raw = chat_completion(model, prompt).strip().upper().replace(" ", "_").rstrip(".")
    return raw if raw in VALID_LABELS else "OTHER"


def tag_domains(pool_jsonl: str, out_jsonl: str) -> None:
    """Tag every pool example with a fine-grained domain label using FEEDBACK_MODEL.

    Writes {idx, domain} per line, where domain is one of VALID_LABELS
    (MATH / COMPETITIVE_CODING / GENERAL_CODING / OTHER). The collapse to the
    3-way MATH/CODE/OTHER split happens downstream via TAG_TO_DOMAIN_3WAY.
    """
    with open(pool_jsonl) as f_in, open(out_jsonl, "w") as f_out:
        for idx, line in enumerate(f_in):
            messages = json.loads(line).get("messages", [])
            user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
            label = classify_domain(FEEDBACK_MODEL, CLASSIFY_PROMPT + user_text[:2000])
            f_out.write(json.dumps({"idx": idx, "domain": label}) + "\n")


def embed_pool(pool_jsonl: str, out_pt: str) -> None:
    """Embed the full training pool with EMBEDDING_MODEL into an L2-normalized tensor."""
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    texts = [
        _extract_text_for_embedding(json.loads(line).get("messages", []))
        for line in open(pool_jsonl)
    ]
    embeddings = model.encode(
        _format_for_embedding(texts), normalize_embeddings=True, show_progress_bar=True
    )
    torch.save(torch.from_numpy(np.asarray(embeddings, dtype=np.float32)), out_pt)


def embed_prompts_local(prompts: list[str]) -> torch.Tensor:
    """Embed policy prompts with EMBEDDING_MODEL into an L2-normalized tensor."""
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    embs = model.encode(
        _format_for_embedding(prompts), normalize_embeddings=True, show_progress_bar=False
    )
    return torch.from_numpy(np.asarray(embs, dtype=np.float32))


def _write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _get_question(example: dict) -> str:
    return next((m["content"] for m in example.get("messages", []) if m["role"] == "user"), "")


def _get_gold(example: dict) -> str:
    return next(
        (m["content"] for m in reversed(example.get("messages", [])) if m["role"] == "assistant"),
        "",
    )


def _final_answer(text: str, max_chars: int = 3000) -> str:
    """Return the content after </think>, capped to the last max_chars."""
    text = text.strip()
    idx = text.find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):].strip()
    return text if len(text) <= max_chars else "..." + text[-max_chars:]


def load_math_problems(hf_cache: str | None = None) -> list[dict]:
    """Load math problems from a JSONL source (path from MATH_SEED_JSONL).

    Each line is a JSON object with at least ``question`` and ``answer`` fields,
    and optional ``year`` and ``id``. The concrete dataset and its schema are not
    assumed here, so any similarly shaped data (including the bundled
    ``sample_math.jsonl``) can drive the pipeline.
    """
    path = _env("MATH_SEED_JSONL", "sample_math.jsonl")
    problems = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            problems.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "year": row.get("year"),
                    "id": row.get("id"),
                }
            )
    return problems


def build_seed_scoring_sets(
    pool_jsonl: str,
    tags_jsonl: str,
    out_dir: str,
    hf_cache: str | None = None,
    seed: int = 42,
    math_seed_size: int = 600,
    code_total: int = 2000,
    other_scoring_size: int = 500,
) -> None:
    """Build per-domain seed and scoring sets for curation.

    Each set holds questions paired with a reference (gold) answer. A seed set
    drives reflection / policy generation; a scoring set is held out for
    evaluation. Sources by domain:

      - MATH: problems from ``load_math_problems``. Shuffled, then math_seed_size
        seed / remainder scoring. The gold answer is rendered as
        "The answer is <answer>.".
      - CODE: examples sampled from the domain-tagged pool (tags COMPETITIVE_CODING
        or GENERAL_CODING), code_total total, split 2/3 seed and 1/3 scoring.
      - OTHER: other_scoring_size examples sampled from the OTHER-tagged pool
        (scoring only).

    Pool indices placed in any set are recorded in excluded_pool_indices.json so
    they can be excluded from the training pool. The retrieval pools used later in
    selection are separate from these evaluation sets.
    """
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    math = [
        {
            "messages": [
                {"role": "user", "content": p["question"]},
                {"role": "assistant", "content": f"The answer is {p['answer']}."},
            ],
            "meta": {"source": "math", "year": p.get("year"), "id": p.get("id")},
        }
        for p in load_math_problems(hf_cache)
    ]
    rng.shuffle(math)
    _write_jsonl(math[:math_seed_size], f"{out_dir}/seed_math.jsonl")
    _write_jsonl(math[math_seed_size:], f"{out_dir}/scoring_math.jsonl")

    tags = {}
    for line in open(tags_jsonl):
        d = json.loads(line)
        tags[d["idx"]] = d["domain"]

    code_idx = [i for i, dom in tags.items() if dom in {"COMPETITIVE_CODING", "GENERAL_CODING"}]
    rng.shuffle(code_idx)
    code_idx = code_idx[:code_total]
    split = (len(code_idx) * 2) // 3
    seed_code, scoring_code = sorted(code_idx[:split]), sorted(code_idx[split:])

    code_set = set(code_idx)
    other_idx = [i for i, dom in tags.items() if dom == "OTHER" and i not in code_set]
    rng.shuffle(other_idx)
    other_idx = sorted(other_idx[:other_scoring_size])

    pool = load_pool(pool_jsonl)

    def rows(indices: list[int]) -> list[dict]:
        return [
            {**json.loads(pool[i]), "meta": {"source": "pool", "pool_idx": i, "domain": tags.get(i)}}
            for i in indices
        ]

    _write_jsonl(rows(seed_code), f"{out_dir}/seed_coding.jsonl")
    _write_jsonl(rows(scoring_code), f"{out_dir}/scoring_coding.jsonl")
    _write_jsonl(rows(other_idx), f"{out_dir}/scoring_other.jsonl")

    excluded = sorted(set(seed_code) | set(scoring_code) | set(other_idx))
    with open(f"{out_dir}/excluded_pool_indices.json", "w") as f:
        json.dump(excluded, f)


def student_generate(checkpoint: str, question: str) -> str:
    """Generate the student model's answer via the served STUDENT_MODEL.

    ``checkpoint`` identifies the checkpoint in the production pipeline; here the
    student is reached through the same endpoint as STUDENT_MODEL.
    """
    return chat_completion(
        STUDENT_MODEL, question, max_tokens=EVAL_MAX_TOKENS,
        base_url=STUDENT_BASE_URL, api_key=STUDENT_API_KEY,
    )


def judge_answer(question: str, gold: str, prediction: str) -> dict:
    """Score a student answer against the reference using FEEDBACK_MODEL."""
    user_msg = (
        f"## Question\n{question[:2000]}\n\n"
        f"## Reference Answer\n{_final_answer(gold)}\n\n"
        f"## Student Answer\n{_final_answer(prediction)}"
    )
    verdict = chat_json(
        FEEDBACK_MODEL, JUDGE_SYSTEM, user_msg,
        base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY,
    )
    return {"score": 1.0 if verdict.get("correct") else 0.0, "reason": verdict.get("reason")}


def analyze_error(
    question: str, gold: str, prediction: str, feedback_system: str = FEEDBACK_SYSTEM,
) -> dict:
    """Produce structured feedback for a wrong answer using FEEDBACK_MODEL.

    ``feedback_system`` selects the analysis prompt; pass ``FEEDBACK_SYSTEM_CODING``
    for coding-domain problems to elicit code-specific feedback.
    """
    user_msg = (
        f"## Question\n{question[:2000]}\n\n"
        f"## Reference Answer\n{_final_answer(gold)}\n\n"
        f"## Student Answer\n{_final_answer(prediction)}"
    )
    return chat_json(
        FEEDBACK_MODEL, feedback_system, user_msg,
        base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY,
    )


def _feedback_system_for(example: dict) -> str:
    """Pick the feedback prompt for an example based on its (3-way) domain.

    Coding examples (seed rows tagged COMPETITIVE_CODING / GENERAL_CODING) use the
    code-specific prompt; everything else uses the general prompt.
    """
    raw_domain = example.get("meta", {}).get("domain")
    if TAG_TO_DOMAIN_3WAY.get(raw_domain) == "CODE":
        return FEEDBACK_SYSTEM_CODING
    return FEEDBACK_SYSTEM


def _server_healthy(base_url: str) -> bool:
    """Return True if an OpenAI-compatible server answers /health at base_url."""
    import urllib.request

    health = base_url.rstrip("/").removesuffix("/v1") + "/health"
    try:
        with urllib.request.urlopen(health, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def serve_student(checkpoint: str, port: int = STUDENT_SERVE_PORT):
    """Launch a stock SGLang OpenAI-compatible server for ``checkpoint``.

    Uses the portable ``python -m sglang.launch_server`` entrypoint; serving
    knobs come from STUDENT_SERVE_* env vars. Build-specific flags (e.g. the AMD
    FarSkip flags needed for Instella-MoE) are passed via STUDENT_SERVE_EXTRA_ARGS.
    Returns the server process; poll for /health up to STUDENT_SERVE_TIMEOUT.
    """
    import shlex
    import subprocess
    import time

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", checkpoint, "--port", str(port),
        "--tp", str(STUDENT_SERVE_TP),
        "--mem-fraction-static", str(STUDENT_SERVE_MEM_FRACTION),
        "--trust-remote-code",
    ] + shlex.split(STUDENT_SERVE_EXTRA_ARGS)
    print(f"[serve] launching student: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, start_new_session=True)
    base = f"http://localhost:{port}/v1"
    deadline = time.time() + STUDENT_SERVE_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"student server exited early (code {proc.returncode})")
        if _server_healthy(base):
            print(f"[serve] student healthy on {base}")
            return proc
        time.sleep(5)
    stop_student(proc)
    raise RuntimeError(f"student server not healthy within {STUDENT_SERVE_TIMEOUT}s")


def stop_student(proc) -> None:
    """Tear down a server started by serve_student (kills the process group)."""
    import os as _os
    import signal
    import subprocess

    if proc is None:
        return
    try:
        _os.killpg(_os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    print("[serve] student stopped")


def pretrain_inference_and_judge(checkpoint: str, seed_set: str, out_analytics: str) -> None:
    """Own the full student inference step: serve → generate → judge → analytics → tear down.

    If STUDENT_BASE_URL already points at a live server, that endpoint is reused
    and no server is launched. Otherwise a stock SGLang server is started from
    ``checkpoint`` and stopped once analytics are written.
    """
    global STUDENT_BASE_URL
    proc = None
    if not _server_healthy(STUDENT_BASE_URL):
        proc = serve_student(checkpoint or STUDENT_CHECKPOINT)
        STUDENT_BASE_URL = f"http://localhost:{STUDENT_SERVE_PORT}/v1"
    try:
        examples = [json.loads(line) for line in open(seed_set)]
        results = []
        for ex in examples:
            question, gold = _get_question(ex), _get_gold(ex)
            prediction = student_generate(checkpoint, question)
            verdict = judge_answer(question, gold, prediction)
            if verdict["score"] < 1.0:
                verdict["feedback"] = analyze_error(
                    question, gold, prediction, _feedback_system_for(ex)
                )
            results.append(verdict)
    finally:
        stop_student(proc)
    analytics = compute_error_analytics(results, policy=[], selection_meta={})
    with open(out_analytics, "w") as f:
        json.dump(analytics, f, indent=2)


def compute_error_analytics(
    results: list[dict],
    policy: list[dict],
    selection_meta: dict,
    n_sample_errors: int = 100,
) -> dict:
    """Build structured error analytics for the reflection LM."""
    errors = [r for r in results if r["score"] < 1.0]
    correct = [r for r in results if r["score"] >= 1.0]
    total = len(results)

    sampled = errors
    if len(errors) > n_sample_errors:
        sampled = random.sample(errors, n_sample_errors)

    return {
        "total_score": sum(r["score"] for r in results) / max(total, 1),
        "total_errors": len(errors),
        "total_correct": len(correct),
        "total_examples": total,
        "current_policy": policy,
        "selection_meta": selection_meta,
        "sampled_errors": sampled,
    }


def format_analytics_for_llm(analytics: dict, history: list | None = None) -> str:
    """Format analytics into a human-readable string for the reflection LM."""
    lines = []

    lines.append("## Current Performance")
    lines.append(f"Score: {analytics['total_score']:.3f} "
                 f"({analytics['total_correct']}/{analytics['total_examples']} correct)")
    lines.append("")

    lines.append("## Current Data Selection Policy")
    policy = analytics["current_policy"]
    if policy:
        total_weight = sum(p["weight"] for p in policy)
        for p in policy:
            lines.append(f'  - [w={p["weight"]:.2f}] {p["prompt"]}')
        lines.append(f"  Total allocated weight: {total_weight:.2f} "
                     f"(remainder {1.0 - total_weight:.2f} filled randomly)")
    else:
        lines.append("  (empty — all data selected randomly)")
    lines.append("")

    sampled = analytics.get("sampled_errors", [])
    if sampled:
        lines.append(f"## Error Feedback ({len(sampled)} randomly sampled from "
                     f"{analytics['total_errors']} total errors)")
        for i, err in enumerate(sampled):
            fb = err.get("feedback") or {}
            if not fb:
                continue
            lines.append(f"  {i+1}.")
            if fb.get("domain"):
                lines.append(f"     Domain:     {fb['domain']}")
            if fb.get("error_type"):
                lines.append(f"     Error:      {fb['error_type']}")
            if fb.get("skills_lacking"):
                lines.append(f"     Skills:     {fb['skills_lacking']}")
            if fb.get("what_would_help"):
                lines.append(f"     Would help: {fb['what_would_help']}")
            if fb.get("summary"):
                lines.append(f"     Summary:    {fb['summary']}")
            lines.append("")

    return "\n".join(lines)


def call_reflection_lm(analytics_text: str) -> list[dict]:
    """Ask LLM_LARGE to propose a data selection policy: a list of {prompt, weight}."""
    content = chat_completion(
        REFLECTION_MODEL, analytics_text, system=REFLECTION_SYSTEM_PROMPT,
        max_tokens=REFLECTION_MAX_TOKENS, temperature=REFLECTION_TEMPERATURE,
        base_url=REFLECTION_BASE_URL, api_key=REFLECTION_API_KEY,
    )
    match = re.search(r"```json\s*(\[.*?\])\s*```", content, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(re.search(r"\[.*\]", content, re.DOTALL).group(0))


def run_policy_mode(analytics_json: str, out_policy_jsonl: str) -> None:
    """Turn one domain's error analytics into a weighted, domain-tagged prompt policy.

    In practice this runs once per seed domain; each call produces prompts whose
    weights are normalized to sum to 1.0 within that domain.
    """
    with open(analytics_json) as f:
        analytics = json.load(f)
    domain = analytics.get("domain", "OTHER")
    policy = call_reflection_lm(format_analytics_for_llm(analytics))
    total = sum(p["weight"] for p in policy) or 1.0
    with open(out_policy_jsonl, "w") as f:
        for p in policy:
            f.write(json.dumps(
                {"prompt": p["prompt"], "weight": p["weight"] / total, "domain": domain}
            ) + "\n")


def load_pool(pool_jsonl: str) -> list[bytes]:
    """Load the full pool as raw JSONL lines; item parsing is deferred to selection."""
    with open(pool_jsonl, "rb") as f:
        return [line for line in f if line.strip()]


_TOKENIZER = None
_TOKENIZER_TRIED = False


def _get_tokenizer():
    """Lazily load TOKENIZER_NAME (or None to use the heuristic fallback)."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if not _TOKENIZER_TRIED:
        _TOKENIZER_TRIED = True
        if TOKENIZER_NAME:
            from transformers import AutoTokenizer

            _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    return _TOKENIZER


def _token_count(raw_item: bytes) -> int:
    """Return the chat-template token length of a pool item.

    Uses TOKENIZER_NAME's chat template when configured (faithful); otherwise a
    ~4-chars/token heuristic so selection stays runnable without a tokenizer.
    """
    try:
        messages = json.loads(raw_item).get("messages", [])
    except Exception:
        return max(1, len(raw_item) // 4)
    tok = _get_tokenizer()
    if tok is not None:
        try:
            text = tok.apply_chat_template(messages, tokenize=False)
            return len(tok(text, add_special_tokens=False).input_ids)
        except Exception:
            pass
    return max(1, sum(len(m.get("content", "")) for m in messages) // 4)


def run_select_mode(
    policy_jsonl: str,
    pool_pt: str,
    pool_jsonl: str,
    tags_jsonl: str,
    budget: int,
    max_tokens: int = 32_000,
    alpha: float = 0.5,
    out_jsonl: str = "curated.jsonl",
    seed: int = 42,
) -> None:
    """Select training data by domain-aware greedy retrieval against pool embeddings.

    Each policy entry is {prompt, weight, domain}. The budget is split across
    domains by fixed fractions. Behavior is domain-aware:

      - MATH and CODE are policy-driven: each prompt greedily retrieves its weighted
        share of the nearest pool items (cosine similarity) within that domain's
        scope. A fraction alpha of the domain budget is instead filled with
        same-domain random samples, so the domain is (1-alpha) greedy + alpha
        random. The final run (Variant C) uses alpha=0.5, i.e. 50/50.
      - OTHER has no policy and is unaffected by alpha: its entire budget is filled
        with random samples from the OTHER scope.

    Items longer than max_tokens are skipped, and any greedy shortfall is topped up
    with same-domain random samples.
    """
    rng = random.Random(seed)
    policy = [json.loads(line) for line in open(policy_jsonl)]
    pool = load_pool(pool_jsonl)
    normed = torch.nn.functional.normalize(
        torch.load(pool_pt, map_location="cpu").float(), dim=1
    )

    domain_of = {}
    for line in open(tags_jsonl):
        d = json.loads(line)
        domain_of[d["idx"]] = TAG_TO_DOMAIN_3WAY.get(d["domain"], "OTHER")
    scope = {dom: [i for i, dd in domain_of.items() if dd == dom] for dom in DOMAINS}

    budgets = {dom: int(budget * DOMAIN_BUDGET_FRACTIONS_3WAY[dom]) for dom in DOMAINS}
    selected: set[int] = set()
    meta = {"budgets": budgets, "alpha": alpha, "per_prompt": []}

    prompt_embs = torch.nn.functional.normalize(
        embed_prompts_local([p["prompt"] for p in policy]), dim=1
    )

    for entry, q in zip(policy, prompt_embs):
        dom = entry["domain"].upper()
        if dom == "OTHER":
            continue
        idx = torch.tensor(scope.get(dom, []))
        count = int(budgets.get(dom, 0) * entry["weight"] * (1.0 - alpha))
        if len(idx) == 0 or count <= 0:
            continue
        sims = normed[idx] @ q
        order = torch.topk(sims, k=min(count * 2, len(idx))).indices.tolist()
        chosen = 0
        for j in order:
            gi = int(idx[j])
            if gi in selected or _token_count(pool[gi]) > max_tokens:
                continue
            selected.add(gi)
            chosen += 1
            if chosen >= count:
                break
        meta["per_prompt"].append(
            {"prompt": entry["prompt"], "domain": dom, "requested": count, "selected": chosen}
        )

    # Same-domain random fill: tops MATH/CODE up to budget (contributing the alpha
    # random share) and fills the entire OTHER budget, which is always random.
    for dom in DOMAINS:
        have = sum(1 for gi in selected if domain_of.get(gi) == dom)
        need = budgets[dom] - have
        if need > 0:
            candidates = [i for i in scope[dom] if i not in selected]
            rng.shuffle(candidates)
            selected.update(candidates[:need])

    with open(out_jsonl, "w") as f:
        for gi in sorted(selected):
            f.write(json.dumps(json.loads(pool[gi])) + "\n")
    with open(out_jsonl.replace(".jsonl", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def generate_sft_config(curated_jsonl: str, checkpoint_dir: str, resume_iteration: int, train_steps: int) -> str:
    """Clone ``SFT_BASE_CONFIG`` into a continuation run on the curated data.

    Retargets the SFT data to ``curated_jsonl`` and, under the trainer overrides,
    resumes from ``checkpoint_dir`` and extends training to
    ``resume_iteration + train_steps`` total iters (``auto_continue_train`` in the
    base config restores the iteration counter). The base config path and save dir
    are env-driven, so no internal config location is baked into the release.
    Returns the written config path.
    """
    import yaml

    if not SFT_BASE_CONFIG:
        raise RuntimeError("Set SFT_BASE_CONFIG to a base SFT YAML to clone for the curated run.")
    with open(SFT_BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    total_iters = resume_iteration + train_steps
    cfg.setdefault("sft_config", {})["train_data_path"] = [os.path.abspath(curated_jsonl)]

    overrides = cfg["modules"]["pre_trainer"]["overrides"]
    overrides["load"] = checkpoint_dir
    overrides["train_iters"] = total_iters
    overrides["lr_decay_iters"] = int(_env("LR_DECAY_ITERS", str(total_iters)))
    if SFT_SAVE_DIR:
        overrides["save"] = SFT_SAVE_DIR

    with open(SFT_CONFIG_OUT, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return SFT_CONFIG_OUT


def run_sft_training(config_path: str) -> None:
    """Launch fine-tuning by shelling into the Primus trainer (cluster-specific).

    Runs ``bash SFT_LAUNCH_SCRIPT`` from ``PRIMUS_DIR`` with ``EXP`` pointing at the
    config, inheriting NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT from the
    environment (single-node defaults if unset). The launcher and its multi-node
    knobs are env-driven so a changed trainer path does not require editing code.
    """
    import subprocess

    env = dict(os.environ)
    env["EXP"] = os.path.abspath(config_path)
    env.setdefault("NNODES", "1")
    env.setdefault("NODE_RANK", "0")
    cmd = f"cd {PRIMUS_DIR} && mkdir -p output && bash {SFT_LAUNCH_SCRIPT}"
    subprocess.run(["bash", "-c", cmd], env=env, check=True)


def convert_checkpoint(checkpoint_dir: str, iteration: int) -> str:
    """Convert a trained Megatron checkpoint to a servable HF dir via ``CONVERT_SCRIPT``.

    Uses the repo's ``convert_megatron_to_hf.py`` (CPU-only). ``checkpoint_dir`` is
    the Megatron save dir; the converter reads its ``release`` subdir (which holds
    ``common.pt``). Returns the HF output dir, which is directly usable as an SGLang
    ``--model-path``. Skips conversion if the output already exists.
    """
    import subprocess

    out_dir = os.path.join(HF_OUTPUT_ROOT, f"iter_{iteration:07d}_hf")
    if os.path.isdir(out_dir):
        return out_dir
    cmd = [
        sys.executable, CONVERT_SCRIPT,
        "--input-dir", os.path.join(checkpoint_dir, "release"),
        "--output-dir", out_dir,
        "--model-name", CONVERT_MODEL_NAME,
    ]
    if ORIGIN_HF_DIR:
        cmd += ["--origin-hf-dir", ORIGIN_HF_DIR]
    subprocess.run(cmd, check=True)
    return out_dir


def train_and_eval(
    curated_jsonl: str,
    checkpoint_dir: str,
    scoring_sets: list[str],
    resume_iteration: int,
    train_steps: int,
) -> None:
    """Fine-tune the student on curated data, convert the checkpoint, and score it.

    Steps 1-2 (config generation, training, checkpoint conversion) shell out to the
    cluster's own trainer/converter via the env-configured helpers above. Set
    ``SKIP_TRAINING=1`` to evaluate an existing checkpoint (``STUDENT_CHECKPOINT``)
    or a live ``STUDENT_BASE_URL`` with no training plumbing. Steps 3-4 reuse the
    self-contained serve -> generate -> judge -> teardown lifecycle (the same path
    as ``pretrain_inference_and_judge``) and report per-set accuracy.
    """
    global STUDENT_BASE_URL

    served = STUDENT_CHECKPOINT
    if not SKIP_TRAINING:
        config = generate_sft_config(curated_jsonl, checkpoint_dir, resume_iteration, train_steps)
        run_sft_training(config)
        served = convert_checkpoint(checkpoint_dir, resume_iteration + train_steps)

    proc = None
    if not _server_healthy(STUDENT_BASE_URL):
        proc = serve_student(served)
        STUDENT_BASE_URL = f"http://localhost:{STUDENT_SERVE_PORT}/v1"
    try:
        for scoring_path in scoring_sets:
            examples = [json.loads(line) for line in open(scoring_path)]
            scores = [
                judge_answer(_get_question(e), _get_gold(e), student_generate(served, _get_question(e)))["score"]
                for e in examples
            ]
            accuracy = sum(scores) / max(len(scores), 1)
            print(f"{os.path.basename(scoring_path)}: {accuracy:.3f} ({len(scores)} examples)")
    finally:
        stop_student(proc)


def main() -> None:
    """Run one curation iteration end to end."""
    tag_domains("pool.jsonl", "pool.tags.jsonl")
    embed_pool("pool.jsonl", "pool.pt")
    build_seed_scoring_sets("pool.jsonl", "pool.tags.jsonl", "eval_sets")

    pretrain_inference_and_judge("iter_12000", "eval_sets/seed_math.jsonl", "analytics.json")
    run_policy_mode("analytics.json", "policy.jsonl")
    run_select_mode("policy.jsonl", "pool.pt", "pool.jsonl", "pool.tags.jsonl",
                    budget=512_000, alpha=0.5)
    train_and_eval("curated.jsonl", "iter_12000",
                   ["eval_sets/scoring_math.jsonl", "eval_sets/scoring_coding.jsonl",
                    "eval_sets/scoring_other.jsonl"],
                   resume_iteration=12_000, train_steps=3_000)


if __name__ == "__main__":
    main()
