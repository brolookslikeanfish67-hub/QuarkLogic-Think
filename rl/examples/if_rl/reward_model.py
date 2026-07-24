#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Reward model for Instella Think-RL instruction-following (IF) training.

Scores IFEval/IFBench constraints in-process via the vendored IFEvalG library
(Apache 2.0, from AllenAI open-instruct). Rewards are scaled to [0, 10].

Usage:
    --custom-rm-path examples.if_rl.reward_model.batched_reward

Dispatches on ``metadata.rm_type``; the math and chat (LLM-as-a-judge) paths are
optional and inert unless their companion modules are importable (not shipped
with this IF example).
"""

import asyncio
import json
import logging
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer
from miles.rollout.rm_hub.math_utils import grade_answer_verl
from miles.utils.types import Sample

try:
    from . import chat_reward
except ImportError:
    try:
        _self_dir = os.path.dirname(__file__)
        if _self_dir not in sys.path:
            sys.path.insert(0, _self_dir)
        import chat_reward
    except ImportError:
        chat_reward = None

try:
    from . import olmo3_math_verify
except ImportError:
    try:
        _self_dir = os.path.dirname(__file__)
        if _self_dir not in sys.path:
            sys.path.insert(0, _self_dir)
        import olmo3_math_verify
    except ImportError:
        olmo3_math_verify = None

logger = logging.getLogger(__name__)

REWARD_SCALE = 10.0
_CODE_TIMEOUT = int(os.environ.get("CODE_EXEC_TIMEOUT", "10"))
# Optional repetition gate (off by default). When INSTELLA_REPETITION_GATE=1, a
# response whose answer exceeds REPETITION_THRESHOLD repeated 4-grams has its
# TRAINING reward zeroed (eval untouched) and logs a "REPGATE" line.
_REPETITION_THRESHOLD = float(os.environ.get("REPETITION_THRESHOLD", "0.5"))
_REPETITION_GATE = os.environ.get("INSTELLA_REPETITION_GATE", "0") == "1"
# Zero the reward when the per-prompt pass rate is below this (0.0 = any pass counts).
_CODE_PASS_RATE_THRESHOLD = float(os.environ.get("CODE_PASS_RATE_REWARD_THRESHOLD", "0.0"))
# Cap on tests evaluated per sample (0 = no cap); local subprocess path only.
_MAX_CODE_TESTS = int(os.environ.get("MAX_CODE_TESTS", "0"))
# Per-test subprocess parallelism, local path only.
_CODE_TEST_WORKERS = int(os.environ.get("CODE_TEST_WORKERS", "8"))
# Max concurrent samples in batched_reward (bounds subprocess / HTTP fan-out).
_BATCHED_REWARD_CONCURRENCY = int(os.environ.get("BATCHED_REWARD_CONCURRENCY", "128"))

_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*([^\n]+)")
_ASSERT_FUNC_RE = re.compile(r"assert\s+(\w+)\s*\(")
_TEST_FUNC_RE = re.compile(r"^def (test_\w+)\s*\(", re.MULTILINE)
_SPECIAL_TOKEN_RE = re.compile(r"<｜[^>]*｜>")


# --- Response cleaning ---

def _clean_response(response: str) -> str:
    """Strip <think> traces, <answer> tags, and model special tokens."""
    if "</think>" in response:
        response = response.split("</think>", 1)[-1]
    elif _THINK_PREFILL:
        # <think> is prefilled into the prompt; a response with no </think> never
        # closed that block -> treat as an empty/format-failed answer.
        response = ""
    elif response.lstrip().startswith("<think>"):
        response = ""
    response = response.replace("<answer>", "").replace("</answer>", "")
    response = _SPECIAL_TOKEN_RE.sub("", response)
    return response.strip()


_strip_thinking = _clean_response


# Think-format gate. A verifiable reward is only credited when the response has a
# proper reasoning block; thinking is then stripped and only the post-</think>
# answer is graded. Without it, malformed/no-think responses collect vacuous
# partial credit from the IF verifier. Training requires a clean OPEN+CLOSE
# (<think>...</think>); eval requires only the CLOSE. Set INSTELLA_REQUIRE_THINK=0
# to disable (e.g. no-think configs).
_REQUIRE_THINK = os.environ.get("INSTELLA_REQUIRE_THINK", "1") == "1"

# Think-prefill mode: when the chat template prefills <think> into the prompt the
# model never emits a leading <think>, so the gate only checks for a clean CLOSE.
_THINK_PREFILL = os.environ.get("INSTELLA_THINK_PREFILL", "0") == "1"


def _has_clean_think(response: str) -> bool:
    if _THINK_PREFILL:
        return "</think>" in response
    return response.lstrip().startswith("<think>") and "</think>" in response


def _repetition_ratio(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


# --- Math reward ---

def _extract_last_number(text: str) -> str | None:
    matches = re.findall(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)", text)
    return matches[-1] if matches else None


def _extract_last_letter(text: str) -> str | None:
    matches = re.findall(r"(?:^|\s|[:(])([A-E])(?:\s|[.),;:]|$)", text)
    return matches[-1] if matches else None


def _normalize_numeric(s: str) -> str | None:
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "")
    try:
        val = float(s)
        return str(int(val)) if val == int(val) else str(val)
    except (ValueError, OverflowError):
        return None


def _extract_answer_line(text: str) -> str | None:
    matches = _ANSWER_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def lenient_math_reward(response: str, label: str) -> float:
    """Grade math response with lenient fallback heuristics."""
    if not label or not response:
        return 0.0

    label_str = str(label).strip()

    if grade_answer_verl(response, label_str):
        return 1.0

    if "</think>" in response:
        after_think = response.split("</think>")[-1]
        extracted = extract_boxed_answer(after_think)
        if extracted is not None and grade_answer_verl(f"\\boxed{{{extracted}}}", label_str):
            return 1.0

    answer_line = _extract_answer_line(response)
    if answer_line is not None:
        answer_norm = _normalize_numeric(answer_line)
        label_norm = _normalize_numeric(label_str)
        if answer_norm is not None and label_norm is not None and answer_norm == label_norm:
            return 1.0

    if len(label_str) == 1 and label_str.upper() in "ABCDE":
        last_letter = _extract_last_letter(response)
        if last_letter and last_letter.upper() == label_str.upper():
            return 1.0

    # NOTE: the bare "last number anywhere" fallback was removed deliberately —
    # it false-positives on integer-answer math (any incidental number matches).

    return 0.0


def strict_math_reward(response: str, label: str) -> float:
    """Strict math grade: only credits answers in ``\\boxed{}``.

    Drops lenient's answer-line / A-E-letter / last-number fallbacks to avoid
    false positives on integer-answer benchmarks. Pair with boxed-instructed
    prompts so the policy is trained to emit ``\\boxed{}``.
    """
    if not label or not response:
        return 0.0

    label_str = str(label).strip()

    if grade_answer_verl(response, label_str):
        return 1.0

    if "</think>" in response:
        after_think = response.split("</think>")[-1]
        extracted = extract_boxed_answer(after_think)
        if extracted is not None and grade_answer_verl(f"\\boxed{{{extracted}}}", label_str):
            return 1.0

    return 0.0


def olmo3_math_reward(response: str, label: str) -> float:
    """Verify via the olmo3_math_verify module; falls back to strict_math_reward
    if that module failed to import."""
    if olmo3_math_verify is None:
        return strict_math_reward(response, label)
    if not label or not response:
        return 0.0
    return 1.0 if olmo3_math_verify.verify(response, str(label)) else 0.0


# Math grader selection via MATH_REWARD_MODE:
#   olmo3   - verify via olmo3_math_verify module (default)
#   strict  - boxed-only (grade_answer_verl); requires \boxed in the response
#   lenient - boxed + Answer-line + A-E fallbacks
_MATH_REWARD_MODE = os.environ.get("MATH_REWARD_MODE", "olmo3").strip().lower()


def math_reward(response: str, label: str) -> float:
    """Dispatch to the configured math grader (see MATH_REWARD_MODE)."""
    if _MATH_REWARD_MODE == "strict":
        return strict_math_reward(response, label)
    if _MATH_REWARD_MODE == "lenient":
        return lenient_math_reward(response, label)
    return olmo3_math_reward(response, label)


# --- Code reward — remote code-execution API client ---
# Retained for multi-domain reuse (the code-exec service is not shipped here).
# When CODE_EXEC_API_URL points at a service, code paths POST there instead of
# forking subprocesses on the training node (avoids fork-after-HIP-init UB and
# running untrusted code next to training weights). Set to "local"/unset for the
# on-node subprocess path.

_CODE_EXEC_API_URL = os.environ.get("CODE_EXEC_API_URL", "").strip().rstrip("/")
_USE_REMOTE_CODE_API = bool(_CODE_EXEC_API_URL) and _CODE_EXEC_API_URL.lower() != "local"
# Per-call HTTP retry budget for transient failures (keep small).
_REMOTE_RETRIES = int(os.environ.get("CODE_EXEC_API_RETRIES", "2"))
# Hard cap on the per-request HTTP timeout.
_REMOTE_MAX_TIMEOUT = float(os.environ.get("CODE_EXEC_API_MAX_TIMEOUT", "120"))
# TCP connect timeout; << read_timeout so we fail fast when the pod is down.
_REMOTE_CONNECT_TIMEOUT_S = float(os.environ.get("CODE_EXEC_API_CONNECT_TIMEOUT_S", "2"))
# Circuit breaker: trip after N consecutive failures, fast-fail for the cooldown.
_REMOTE_CB_FAIL_THRESHOLD = int(os.environ.get("CODE_EXEC_API_CB_FAIL_THRESHOLD", "5"))
_REMOTE_CB_COOLDOWN_S = float(os.environ.get("CODE_EXEC_API_CB_COOLDOWN_S", "30"))
# Requests connection-pool size (should exceed BATCHED_REWARD_CONCURRENCY).
_REMOTE_POOL_MAXSIZE = int(os.environ.get("CODE_EXEC_API_POOL_MAXSIZE", "256"))


_session: requests.Session | None = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                s = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=_REMOTE_POOL_MAXSIZE,
                    pool_maxsize=_REMOTE_POOL_MAXSIZE,
                    max_retries=0,  # we handle retries explicitly
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                _session = s
    return _session


_cb_lock = threading.Lock()
_cb_failures = 0
_cb_open_until = 0.0  # monotonic timestamp; <= time.monotonic() means CLOSED


def _cb_is_open() -> bool:
    return time.monotonic() < _cb_open_until


def _cb_record_success() -> None:
    global _cb_failures, _cb_open_until
    if _cb_failures == 0 and _cb_open_until == 0.0:
        return
    with _cb_lock:
        _cb_failures = 0
        _cb_open_until = 0.0


def _cb_record_failure() -> None:
    global _cb_failures, _cb_open_until
    with _cb_lock:
        _cb_failures += 1
        if _cb_failures >= _REMOTE_CB_FAIL_THRESHOLD:
            _cb_open_until = time.monotonic() + _REMOTE_CB_COOLDOWN_S
            logger.warning(
                "code-exec API circuit breaker OPEN for %.1fs after %d consecutive failures",
                _REMOTE_CB_COOLDOWN_S, _cb_failures,
            )


def _remote_post(path: str, payload: dict, timeout: float):
    """POST JSON to the code-exec API with bounded retries + circuit breaker.

    Returns the parsed JSON body on success, or None on persistent failure
    (after which the caller should treat all tests as failed: result=0).
    """
    if _cb_is_open():
        logger.debug("code-exec API circuit breaker open; skipping POST %s", path)
        return None

    url = _CODE_EXEC_API_URL + path
    read_timeout = max(1.0, float(timeout) - _REMOTE_CONNECT_TIMEOUT_S)
    session = _get_session()
    last_exc: Exception | None = None

    for attempt in range(_REMOTE_RETRIES + 1):
        try:
            resp = session.post(
                url,
                json=payload,
                timeout=(_REMOTE_CONNECT_TIMEOUT_S, read_timeout),
                headers={"Content-Type": "application/json"},
            )
        except (requests.exceptions.RequestException, socket.timeout, OSError) as e:
            last_exc = e
        else:
            if 200 <= resp.status_code < 300:
                try:
                    data = resp.json()
                except ValueError as e:  # malformed JSON
                    last_exc = e
                else:
                    _cb_record_success()
                    return data
            elif 400 <= resp.status_code < 500:
                # Client error: don't retry, don't trip CB (it's our payload bug).
                logger.warning(
                    "code-exec API %s returned HTTP %d; not retrying. body[:200]=%r",
                    path, resp.status_code, resp.text[:200],
                )
                return None
            else:
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}")

        if attempt < _REMOTE_RETRIES:
            # Exponential backoff with jitter; capped so we don't sleep forever.
            sleep_s = min(2.0, 0.25 * (2 ** attempt)) + random.uniform(0, 0.1)
            time.sleep(sleep_s)

    _cb_record_failure()
    logger.warning(
        "Remote code-exec POST %s failed after %d attempts: %s",
        path, _REMOTE_RETRIES + 1, last_exc,
    )
    return None


def _remote_run_program(program: str, tests, max_execution_time: float, stdio: bool):
    """Run ``program`` against ``tests`` on the remote API.

    Returns a list of 0/1 ints of length ``len(tests)``. On any error or
    network failure we return all-zeros (treat as failed), matching the
    semantics of the local fork path when a subprocess raises.
    """
    if not tests:
        return []
    path = "/test_program_stdio" if stdio else "/test_program"
    # Wall-clock budget: max_execution_time per test plus buffer for spawn/piping.
    request_timeout = min(_REMOTE_MAX_TIMEOUT,
                          max_execution_time * max(1, len(tests)) + 10.0)
    resp = _remote_post(
        path,
        {"program": program, "tests": tests, "max_execution_time": float(max_execution_time)},
        timeout=request_timeout,
    )
    if not isinstance(resp, dict):
        return [0] * len(tests)
    results = resp.get("results")
    if not isinstance(results, list) or len(results) != len(tests):
        return [0] * len(tests)
    return [1 if r == 1 or r is True else 0 for r in results]


# --- Code reward ---

def _extract_python_code(response: str, entry_point: str | None = None) -> str:
    """Extract the model's code: the last fenced ```python block, or the whole
    response if unfenced. ``entry_point`` is accepted but unused."""
    del entry_point
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if not matches:
        return response
    return matches[-1].strip()


def _maybe_cap_tests(test_cases):
    if _MAX_CODE_TESTS > 0 and len(test_cases) > _MAX_CODE_TESTS:
        return test_cases[:_MAX_CODE_TESTS]
    return test_cases


def _run_single_assert(code: str, tc: str) -> bool:
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"{code}\n\n{tc}\n"],
            capture_output=True, text=True, timeout=_CODE_TIMEOUT,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def _execute_code_with_tests(code: str, test_cases: list[str]) -> float:
    if not code or not test_cases:
        return 0.0

    test_cases = _maybe_cap_tests(test_cases)

    # Remote path: one POST runs all tests on the dedicated CPU service.
    if _USE_REMOTE_CODE_API:
        results = _remote_run_program(code, list(test_cases),
                                      max_execution_time=float(_CODE_TIMEOUT),
                                      stdio=False)
        return sum(results) / len(test_cases) if test_cases else 0.0

    # Fast path: run all asserts in one subprocess; if it succeeds, all pass.
    full_code = f"{code}\n\n" + "\n".join(test_cases) + "\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=_CODE_TIMEOUT,
        )
        if result.returncode == 0:
            return 1.0
    except (subprocess.TimeoutExpired, Exception):
        pass

    # Slow path: run each test in its own subprocess, in parallel.
    workers = max(1, min(_CODE_TEST_WORKERS, len(test_cases)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda tc: _run_single_assert(code, tc), test_cases))
    return sum(results) / len(test_cases)


def _lines_equal_with_decimal(a: str, b: str) -> bool:
    """Per-line comparison with decimal-aware fallback (1.0 == 1 == 1.00)."""
    a_lines = [ln.strip() for ln in a.strip().split("\n")]
    b_lines = [ln.strip() for ln in b.strip().split("\n")]
    if len(a_lines) != len(b_lines):
        return False
    for al, bl in zip(a_lines, b_lines):
        if al == bl:
            continue
        try:
            from decimal import Decimal
            a_dec = [Decimal(tok) for tok in al.split()]
            b_dec = [Decimal(tok) for tok in bl.split()]
        except Exception:
            return False
        if a_dec != b_dec:
            return False
    return True


def _normalize_stdio_field(v) -> str:
    """Join list-of-lines with '\n'."""
    if isinstance(v, list):
        return "\n".join(str(x) for x in v)
    return v if isinstance(v, str) else str(v)


def _run_single_stdio(code: str, tc: dict) -> bool:
    if not isinstance(tc, dict):
        return False
    gt_in = _normalize_stdio_field(tc.get("input", ""))
    gt_out = _normalize_stdio_field(tc.get("output", ""))
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            input=gt_in,
            capture_output=True, text=True, timeout=_CODE_TIMEOUT,
        )
        return r.returncode == 0 and _lines_equal_with_decimal(r.stdout, gt_out)
    except (subprocess.TimeoutExpired, Exception):
        return False


def _execute_code_stdio(code: str, test_cases: list[dict]) -> float:
    """Stdio grading: each test case is {"input": str, "output": str}.

    Runs the program as a fresh subprocess, pipes input via stdin, compares
    stdout per-line (decimal-aware). Per-test execution is parallel, bounded by
    CODE_TEST_WORKERS.
    """
    if not code or not test_cases:
        return 0.0

    test_cases = _maybe_cap_tests(test_cases)

    # Remote path: one POST runs all stdio cases on the code-exec service.
    if _USE_REMOTE_CODE_API:
        # Normalize list-of-lines to "\n".join'd strings up front.
        norm = [
            {"input":  _normalize_stdio_field(tc.get("input", "")),
             "output": _normalize_stdio_field(tc.get("output", ""))}
            for tc in test_cases if isinstance(tc, dict)
        ]
        if not norm:
            return 0.0
        results = _remote_run_program(code, norm,
                                      max_execution_time=float(_CODE_TIMEOUT),
                                      stdio=True)
        return sum(results) / len(norm)

    workers = max(1, min(_CODE_TEST_WORKERS, len(test_cases)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda tc: _run_single_stdio(code, tc), test_cases))
    return sum(results) / len(test_cases)


def _extract_test_functions(test_str: str) -> list[str]:
    matches = list(_TEST_FUNC_RE.finditer(test_str))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(test_str)
        blocks.append(test_str[start:end].rstrip())
    return blocks


def _run_pytest_tests(code: str, test_str: str) -> float:
    test_funcs = _extract_test_functions(test_str)

    # Remote path: send each test_func as a self-contained test ending in
    # `<fname>()`, average the per-function 0/1 results.
    if _USE_REMOTE_CODE_API:
        if not test_funcs:
            results = _remote_run_program(code, [test_str + "\n"],
                                          max_execution_time=float(_CODE_TIMEOUT),
                                          stdio=False)
            return float(results[0]) if results else 0.0
        compound_tests = []
        for tf in test_funcs:
            fname = _TEST_FUNC_RE.search(tf).group(1)
            compound_tests.append(f"{tf}\n\n{fname}()\n")
        results = _remote_run_program(code, compound_tests,
                                      max_execution_time=float(_CODE_TIMEOUT),
                                      stdio=False)
        return sum(results) / len(test_funcs) if test_funcs else 0.0

    if not test_funcs:
        try:
            r = subprocess.run(
                [sys.executable, "-c", f"{code}\n\n{test_str}\n"],
                capture_output=True, text=True, timeout=_CODE_TIMEOUT,
            )
            return 1.0 if r.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, Exception):
            return 0.0

    all_code = f"{code}\n\n" + "\n\n".join(test_funcs)
    calls = "\n".join(f"{_TEST_FUNC_RE.search(tf).group(1)}()" for tf in test_funcs)
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"{all_code}\n\n{calls}\n"],
            capture_output=True, text=True, timeout=_CODE_TIMEOUT,
        )
        if r.returncode == 0:
            return 1.0
    except (subprocess.TimeoutExpired, Exception):
        pass

    n_passed = 0
    for tf in test_funcs:
        fname = _TEST_FUNC_RE.search(tf).group(1)
        try:
            r = subprocess.run(
                [sys.executable, "-c", f"{code}\n\n{tf}\n\n{fname}()\n"],
                capture_output=True, text=True, timeout=_CODE_TIMEOUT,
            )
            if r.returncode == 0:
                n_passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass
    return n_passed / len(test_funcs)


def _infer_entry_point(test_cases) -> str:
    if isinstance(test_cases, dict):
        return test_cases.get("entry_point", "")
    if isinstance(test_cases, list):
        for tc in test_cases:
            m = _ASSERT_FUNC_RE.search(str(tc))
            if m:
                return m.group(1)
    return ""


def compute_code_reward(response: str, label: str) -> float:
    response = _strip_thinking(response)
    try:
        test_cases = json.loads(label)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    entry_point = _infer_entry_point(test_cases)
    code = _extract_python_code(response, entry_point=entry_point or None)
    if not code:
        return 0.0

    if isinstance(test_cases, dict):
        test_str = test_cases.get("test", "")
        if not test_str:
            return 0.0
        if _TEST_FUNC_RE.search(test_str):
            return _run_pytest_tests(code, test_str)
        if entry_point and "check(" in test_str:
            compound = f"{test_str}\ncheck({entry_point})\n"
            if _USE_REMOTE_CODE_API:
                results = _remote_run_program(code, [compound],
                                              max_execution_time=float(_CODE_TIMEOUT),
                                              stdio=False)
                return float(results[0]) if results else 0.0
            try:
                r = subprocess.run(
                    [sys.executable, "-c", f"{code}\n\n{compound}"],
                    capture_output=True, text=True, timeout=_CODE_TIMEOUT,
                )
                return 1.0 if r.returncode == 0 else 0.0
            except (subprocess.TimeoutExpired, Exception):
                return 0.0
        return _run_pytest_tests(code, test_str)

    if isinstance(test_cases, list):
        if test_cases and isinstance(test_cases[0], dict) and (
            "input" in test_cases[0] or "output" in test_cases[0]
        ):
            return _execute_code_stdio(code, test_cases)
        return _execute_code_with_tests(code, test_cases)

    return 0.0


# --- IFEval / IFBench reward ---

try:
    from .IFEvalG import instructions_registry
except ImportError:
    try:
        _ifeval_dir = os.path.dirname(__file__)
        if _ifeval_dir not in sys.path:
            sys.path.insert(0, _ifeval_dir)
        from IFEvalG import instructions_registry
    except ImportError:
        instructions_registry = None
        logger.warning("IFEvalG not found; ifbench rewards will return 0.0")

_ifbench_registry = None
# Path to a local IFBench checkout (for its extended instruction registry).
# Set INSTELLA_IFBENCH_DIR; extended constraints just return False if unset.
_IFBENCH_DIR = os.environ.get("INSTELLA_IFBENCH_DIR", "")
if _IFBENCH_DIR:
    try:
        if _IFBENCH_DIR not in sys.path:
            sys.path.insert(0, _IFBENCH_DIR)
        import importlib
        _ifbench_registry = importlib.import_module("instructions_registry")
        if _IFBENCH_DIR in sys.path:
            sys.path.remove(_IFBENCH_DIR)
        logger.info(f"IFBench loaded: {len(_ifbench_registry.INSTRUCTION_DICT)} constraint types")
    except Exception as _e:
        logger.warning(f"IFBench verifier not loaded ({_e}); extended constraints will return False")
else:
    logger.info("INSTELLA_IFBENCH_DIR unset; extended IFBench constraints will return False")


def _get_checker_class(inst_id):
    if instructions_registry and inst_id in instructions_registry.INSTRUCTION_DICT:
        return instructions_registry.INSTRUCTION_DICT[inst_id]
    if _ifbench_registry and inst_id in _ifbench_registry.INSTRUCTION_DICT:
        return _ifbench_registry.INSTRUCTION_DICT[inst_id]
    return None


def _check_one_constraint(inst_id, kwargs, response, prompt_text=None):
    cls = _get_checker_class(inst_id)
    if cls is None:
        return False
    try:
        kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
        checker = cls(inst_id)
        checker.build_description(**kwargs)
        args = checker.get_instruction_args()
        if args and "prompt" in args and prompt_text:
            checker.build_description(prompt=prompt_text)
        return bool(response.strip() and checker.check_following(response))
    except Exception:
        return False


def _compute_ifeval_reward(response: str, metadata: dict, partial_credit: bool = False) -> float:
    if instructions_registry is None:
        return 0.0

    instruction_ids = metadata.get("instruction_id_list", [])
    kwargs_list = metadata.get("kwargs", [])
    prompt_text = metadata.get("prompt_text", "")

    if not instruction_ids:
        return 0.0
    if not kwargs_list or len(kwargs_list) != len(instruction_ids):
        kwargs_list = [{}] * len(instruction_ids)
    if not response.strip():
        return 0.0

    passed = 0
    for inst_id, kw in zip(instruction_ids, kwargs_list):
        if _check_one_constraint(inst_id, kw, response, prompt_text):
            passed += 1
        elif not partial_credit:
            return 0.0

    return (passed / len(instruction_ids)) if partial_credit else 1.0


def _enrich_ifbench_metadata(meta: dict, label, prompt) -> dict:
    """Populate instruction_id_list/kwargs from label if missing in metadata."""
    if meta.get("instruction_id_list"):
        return meta

    meta = dict(meta)
    parsed = label
    if isinstance(parsed, str):
        try:
            import ast
            parsed = ast.literal_eval(parsed)
        except Exception:
            return meta

    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        constraint = parsed[0]
        meta["instruction_id_list"] = constraint.get("instruction_id", [])
        meta["kwargs"] = constraint.get("kwargs", [])

    if "prompt_text" not in meta:
        if isinstance(prompt, list):
            meta["prompt_text"] = prompt[0].get("content", "") if prompt else ""
        elif isinstance(prompt, str):
            meta["prompt_text"] = prompt

    return meta


# --- Per-sample scoring — routes by metadata.rm_type ---

def _score_one(sample: Sample, evaluation: bool = False) -> float:
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (meta.get("rm_type") or "").strip()
    response = sample.response or ""
    label = sample.label

    # Chat / judge path, routed before the think/repetition gates (those are
    # verifiable-domain heuristics; the judge scores answer quality directly).
    if rm_type in (chat_reward.CHAT_RM_TYPES if chat_reward is not None else ()):
        # Per-sample judge telemetry for the rollout logger's metrics.
        tel: dict = {}
        score = chat_reward.score_chat(sample.prompt, response, label, rm_type, telemetry=tel)
        if isinstance(sample.metadata, dict) and tel:
            sample.metadata["_chat_rm"] = tel
        return score * REWARD_SCALE

    # Think-format gate: training requires a clean open+close; eval requires only
    # the close.
    if _REQUIRE_THINK:
        if evaluation:
            if "</think>" not in response:
                return 0.0
        elif not _has_clean_think(response):
            return 0.0

    # Repetition gate (INSTELLA_REPETITION_GATE, default OFF): zero the TRAINING
    # reward on degenerate high-repetition responses; eval left untouched.
    if _REPETITION_GATE:
        answer_text = _strip_thinking(response) if "</think>" in response else response
        rep4 = _repetition_ratio(answer_text)
        fired = rep4 > _REPETITION_THRESHOLD
        logger.info(
            "REPGATE words=%d rep4=%.3f fired=%d eval=%d rm=%s",
            len(response.split()), rep4, int(fired), int(evaluation), rm_type,
        )
        if fired and not evaluation:
            return 0.0

    if rm_type == "ifbench":
        meta = _enrich_ifbench_metadata(meta, label, sample.prompt)
        raw = _compute_ifeval_reward(_strip_thinking(response), meta, partial_credit=not evaluation)
        return raw * REWARD_SCALE

    if rm_type == "code":
        pass_rate = compute_code_reward(response, str(label) if label is not None else "")
        if pass_rate < _CODE_PASS_RATE_THRESHOLD:
            return 0.0
        return pass_rate * REWARD_SCALE

    if rm_type in ("math", "deepscaler"):
        # Grade only the post-</think> answer (fall back to full response if absent).
        answer = _strip_thinking(response) if "</think>" in response else response
        return math_reward(answer, str(label) if label is not None else "") * REWARD_SCALE

    # Unknown rm_type: return 0.0 rather than guessing a verifier.
    logger.warning("Unknown rm_type=%r; returning 0.0 reward (no verifier match).", rm_type)
    return 0.0


# --- Entry point (custom_rm_path) ---

async def batched_reward(args, samples, **kwargs):
    """Multi-domain reward entry point (IF here; math/code/chat when enabled).

    Each sample is scored in a worker thread so the synchronous _score_one
    doesn't stall the event loop; a semaphore caps concurrency at
    BATCHED_REWARD_CONCURRENCY.
    """
    evaluation = kwargs.get("evaluation", False)
    if isinstance(samples, Sample):
        return _score_one(samples, evaluation=evaluation)
    if not samples:
        return []

    sem = asyncio.Semaphore(max(1, _BATCHED_REWARD_CONCURRENCY))

    async def _one(sample):
        async with sem:
            return await asyncio.to_thread(_score_one, sample, evaluation=evaluation)

    return await asyncio.gather(*[_one(s) for s in samples])
