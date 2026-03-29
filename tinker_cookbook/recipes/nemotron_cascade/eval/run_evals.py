"""
Benchmark evaluations for Nemotron-Cascade-2 checkpoints.

Uses Tinker sampling directly with our own grading:
  - GSM8K: Math grading via existing math_grading utilities
  - IFEval: Our 48-type instruction following verifier
  - MMLU-Pro: Multi-task language understanding
  - MATH-500: Hendrycks MATH test set
  - GPQA-Diamond: Graduate-level science QA (multiple choice)
  - AIME 2025: Math competition problems (integer answers 0-999)
  - MBPP: Python code generation with execution-based testing
  - LongBench v2: Long-context comprehension (multiple subtasks)

Compares base model vs SFT vs IF-RL checkpoints.
"""

import argparse
import asyncio
import json
import logging
import math
import os
from datetime import datetime
from typing import cast

import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import Message
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GSM8K Evaluation
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...} handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def _extract_number(text: str) -> str:
    """Extract a number from text, stripping LaTeX formatting."""
    import re
    # Remove LaTeX commands
    cleaned = re.sub(r'\\text\{[^}]*\}', '', text)
    cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "").replace("$", "")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    # Find number
    match = re.search(r'[-]?\d+\.?\d*', cleaned)
    return match.group(0) if match else cleaned.strip()


def _extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from model response."""
    import re

    # Try boxed format first (handles nested braces)
    boxed = _extract_boxed(text)
    if boxed:
        return _extract_number(boxed)

    # Try "#### answer" format
    hash_match = re.search(r'####\s*(.+)', text)
    if hash_match:
        return _extract_number(hash_match.group(1))

    # Try "the answer is X" pattern
    answer_match = re.search(r'(?:answer is|answer:)\s*\$?([0-9,.-]+)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")

    # Last number in the text
    numbers = re.findall(r'[-]?\d+[,\d]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def _check_gsm8k(response: str, expected: str) -> bool:
    extracted = _extract_gsm8k_answer(response)
    try:
        return abs(float(extracted) - float(expected.replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return extracted.strip() == expected.strip()


async def eval_gsm8k(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on GSM8K test set with concurrent sampling."""
    ds = cast(Dataset, load_dataset("openai/gsm8k", "main", split="test"))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    import re
    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(row: dict) -> bool | None:
        async with semaphore:
            question = row["question"]
            answer_match = re.search(r'####\s*(.+)', row["answer"])
            expected = answer_match.group(1).strip().replace(",", "") if answer_match else ""
            messages: list[Message] = [
                {"role": "user", "content": question + " Show your work step by step, then give the final numerical answer."},
            ]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)
                return _check_gsm8k(content, expected)
            except Exception as e:
                logger.warning(f"GSM8K eval failed: {e}")
                return None

    # Launch all concurrently
    logger.info(f"GSM8K: evaluating {len(ds)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in ds]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r is True)
    total = sum(1 for r in results if r is not None)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"GSM8K final: {correct}/{total} = {accuracy:.4f}")
    return {"gsm8k/accuracy": accuracy, "gsm8k/correct": correct, "gsm8k/total": total}


# ---------------------------------------------------------------------------
# IFEval Evaluation
# ---------------------------------------------------------------------------

async def eval_ifeval(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on IFEval using our verifier with concurrent sampling."""
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.if_rl import verify_all_instructions

    # Load IFEval data from our downloaded RL data
    ifeval_path = os.path.expanduser("~/data/nemotron-cascade-2/rl_if_rl.jsonl")
    rows = []
    with open(ifeval_path) as f:
        for line in f:
            rows.append(json.loads(line))

    if limit:
        rows = rows[:limit]

    semaphore = asyncio.Semaphore(concurrency)

    def _parse_kwargs(raw_kwargs: list, instruction_ids: list) -> list[dict]:
        kwargs_list = []
        for kw in raw_kwargs:
            if kw is None:
                kwargs_list.append({})
            elif isinstance(kw, str):
                try:
                    kwargs_list.append(json.loads(kw))
                except json.JSONDecodeError:
                    kwargs_list.append({})
            elif isinstance(kw, dict):
                kwargs_list.append(kw)
            else:
                kwargs_list.append({})
        while len(kwargs_list) < len(instruction_ids):
            kwargs_list.append({})
        return kwargs_list

    async def eval_one(row: dict) -> tuple[float, bool] | None:
        async with semaphore:
            prompt_messages = row["responses_create_params"]["input"]
            instruction_ids = row.get("instruction_id_list", [])
            raw_kwargs = row.get("kwargs", [])
            kwargs_list = _parse_kwargs(raw_kwargs, instruction_ids)

            messages: list[Message] = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in prompt_messages
            ]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)
                fraction, _ = verify_all_instructions(content, instruction_ids, kwargs_list)
                return (fraction, fraction == 1.0)
            except Exception as e:
                logger.warning(f"IFEval eval failed: {e}")
                return None

    # Launch all concurrently
    logger.info(f"IFEval: evaluating {len(rows)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in rows]
    results = await asyncio.gather(*tasks)

    valid = [r for r in results if r is not None]
    total_prompts = len(valid)
    total_score = sum(r[0] for r in valid)
    strict_correct = sum(1 for r in valid if r[1])

    loose_acc = total_score / total_prompts if total_prompts > 0 else 0
    strict_acc = strict_correct / total_prompts if total_prompts > 0 else 0
    logger.info(f"IFEval final: loose={loose_acc:.4f}, strict={strict_acc:.4f} ({total_prompts} prompts)")
    return {
        "ifeval/loose_accuracy": loose_acc,
        "ifeval/strict_accuracy": strict_acc,
        "ifeval/total": total_prompts,
    }


# ---------------------------------------------------------------------------
# MMLU Evaluation
# ---------------------------------------------------------------------------

async def eval_mmlu(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on MMLU (0-shot) with concurrent sampling."""
    import re

    # Use MMLU-Redux for cleaner data
    try:
        ds = cast(Dataset, load_dataset("TIGER-Lab/MMLU-Pro", split="test"))
        dataset_name = "mmlu_pro"
    except Exception:
        ds = cast(Dataset, load_dataset("cais/mmlu", "all", split="test"))
        dataset_name = "mmlu"

    if limit:
        ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))

    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(row: dict) -> bool | None:
        async with semaphore:
            # Format MMLU question
            question = row.get("question", row.get("input", ""))
            choices = row.get("options", row.get("choices", []))
            answer_idx = row.get("answer_index", row.get("answer", None))

            # Build the prompt
            if choices:
                choice_text = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
                prompt = f"{question}\n\n{choice_text}\n\nAnswer with just the letter (A, B, C, D, etc.)."
            else:
                prompt = f"{question}\n\nAnswer with just the letter."

            # Determine expected answer
            if isinstance(answer_idx, int):
                expected = chr(65 + answer_idx)
            elif isinstance(answer_idx, str) and len(answer_idx) == 1:
                expected = answer_idx.upper()
            else:
                expected = str(answer_idx).strip().upper()

            messages: list[Message] = [{"role": "user", "content": prompt}]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)
                # Extract letter answer
                # Look for standalone letter
                letters = re.findall(r'\b([A-J])\b', content[-200:])
                if letters:
                    extracted = letters[-1]
                else:
                    extracted = content.strip()[-1:].upper()
                return extracted == expected
            except Exception as e:
                logger.warning(f"MMLU eval failed: {e}")
                return None

    logger.info(f"MMLU ({dataset_name}): evaluating {len(ds)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in ds]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r is True)
    total = sum(1 for r in results if r is not None)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"MMLU final: {correct}/{total} = {accuracy:.4f}")
    return {"mmlu/accuracy": accuracy, "mmlu/correct": correct, "mmlu/total": total}


# ---------------------------------------------------------------------------
# MATH-500 Evaluation
# ---------------------------------------------------------------------------

async def eval_math500(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on MATH-500 (Hendrycks MATH test set)."""
    from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer

    ds = cast(Dataset, load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(row: dict) -> bool | None:
        async with semaphore:
            problem = row["problem"]
            try:
                expected = extract_boxed(row["solution"])
            except ValueError:
                return None

            messages: list[Message] = [
                {"role": "user", "content": problem + " Put your final answer in \\boxed{}."},
            ]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)
                try:
                    given = extract_boxed(content)
                    return grade_answer(given, expected)
                except ValueError:
                    return False
            except Exception as e:
                logger.warning(f"MATH-500 eval failed: {e}")
                return None

    logger.info(f"MATH-500: evaluating {len(ds)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in ds]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r is True)
    total = sum(1 for r in results if r is not None)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"MATH-500 final: {correct}/{total} = {accuracy:.4f}")
    return {"math500/accuracy": accuracy, "math500/correct": correct, "math500/total": total}


# ---------------------------------------------------------------------------
# GPQA-Diamond Evaluation
# ---------------------------------------------------------------------------

async def eval_gpqa(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on GPQA-Diamond — hard graduate-level science QA (multiple choice A/B/C/D)."""
    import re

    ds = cast(Dataset, load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(row: dict) -> bool | None:
        async with semaphore:
            question = row["Question"]
            # GPQA has columns: Correct Answer, Incorrect Answer 1/2/3
            # and a "Random ID" but the correct answer letter depends on
            # the shuffled order stored in the dataset.
            # The dataset provides pre-shuffled choices and a correct answer letter.
            correct_answer = row.get("Answer", row.get("Correct Answer", ""))

            # Build choices from the available choice columns
            choice_cols = [
                col for col in row.keys()
                if col.startswith("Choice") or col in ("choice_a", "choice_b", "choice_c", "choice_d")
            ]

            if choice_cols:
                # Dataset has explicit choice columns
                choices = [row[c] for c in sorted(choice_cols) if row.get(c)]
            else:
                # Fallback: construct from Correct Answer + Incorrect Answers
                choices = [row.get("Correct Answer", "")]
                for i in range(1, 4):
                    inc = row.get(f"Incorrect Answer {i}", "")
                    if inc:
                        choices.append(inc)

            if not choices:
                return None

            # Determine expected letter
            if correct_answer in ("A", "B", "C", "D"):
                expected = correct_answer
            else:
                # Find which choice matches the correct answer text
                expected = "A"
                for i, c in enumerate(choices):
                    if c.strip() == str(correct_answer).strip():
                        expected = chr(65 + i)
                        break

            choice_text = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
            prompt = (
                f"{question}\n\n{choice_text}\n\n"
                "Think step by step, then give your final answer as a single letter (A, B, C, or D)."
            )
            messages: list[Message] = [{"role": "user", "content": prompt}]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)
                # Extract the answer letter from the response
                # Check for boxed format first
                boxed = _extract_boxed(content)
                if boxed and boxed.strip().upper() in ("A", "B", "C", "D"):
                    extracted = boxed.strip().upper()
                else:
                    # Look for patterns like "The answer is (B)" or just a standalone letter
                    answer_match = re.search(
                        r'(?:answer is|answer:)\s*\(?([A-D])\)?',
                        content, re.IGNORECASE,
                    )
                    if answer_match:
                        extracted = answer_match.group(1).upper()
                    else:
                        # Last standalone A-D letter
                        letters = re.findall(r'\b([A-D])\b', content[-300:])
                        extracted = letters[-1] if letters else ""
                return extracted == expected
            except Exception as e:
                logger.warning(f"GPQA eval failed: {e}")
                return None

    logger.info(f"GPQA-Diamond: evaluating {len(ds)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in ds]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r is True)
    total = sum(1 for r in results if r is not None)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"GPQA-Diamond final: {correct}/{total} = {accuracy:.4f}")
    return {"gpqa_diamond/accuracy": accuracy, "gpqa_diamond/correct": correct, "gpqa_diamond/total": total}


# ---------------------------------------------------------------------------
# AIME 2025 Evaluation
# ---------------------------------------------------------------------------

async def eval_aime2025(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on AIME 2025 — math competition problems with integer answers (0-999)."""
    import re

    # Try several known HuggingFace dataset names for AIME 2025
    ds = None
    for dataset_id in (
        "HuggingFaceH4/aime-2025",
        "yentinglin/aime_2025",
        "Maxwell-Jia/AIME_2025",
        "opencompass/AIME2025",
        "di-zhang-fdu/AIME24-25",
    ):
        try:
            ds = cast(Dataset, load_dataset(dataset_id, split="test"))
            logger.info(f"Loaded AIME 2025 from {dataset_id} ({len(ds)} problems)")
            break
        except Exception:
            try:
                ds = cast(Dataset, load_dataset(dataset_id, split="train"))
                logger.info(f"Loaded AIME 2025 from {dataset_id} ({len(ds)} problems)")
                break
            except Exception:
                continue

    if ds is None:
        logger.warning("Could not load AIME 2025 dataset from HuggingFace. Skipping.")
        return {"aime2025/accuracy": 0.0, "aime2025/correct": 0, "aime2025/total": 0}

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(row: dict) -> bool | None:
        async with semaphore:
            # Field names vary across datasets
            problem = row.get("problem", row.get("question", row.get("Problem", "")))
            expected_raw = row.get("answer", row.get("Answer", row.get("expected_answer", "")))
            if not problem or expected_raw is None:
                return None

            try:
                expected = int(str(expected_raw).strip())
            except (ValueError, TypeError):
                # Try extracting number
                m = re.search(r'\d+', str(expected_raw))
                if m:
                    expected = int(m.group(0))
                else:
                    return None

            prompt = (
                f"{problem}\n\n"
                "This is an AIME problem. The answer is an integer from 000 to 999. "
                "Show your work step by step, then put your final answer in \\boxed{}."
            )
            messages: list[Message] = [{"role": "user", "content": prompt}]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)
                # Extract answer from boxed or last number
                boxed = _extract_boxed(content)
                if boxed:
                    extracted_str = _extract_number(boxed)
                else:
                    extracted_str = _extract_gsm8k_answer(content)

                try:
                    extracted_val = int(float(extracted_str))
                    return extracted_val == expected
                except (ValueError, TypeError):
                    return False
            except Exception as e:
                logger.warning(f"AIME 2025 eval failed: {e}")
                return None

    logger.info(f"AIME 2025: evaluating {len(ds)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in ds]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r is True)
    total = sum(1 for r in results if r is not None)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"AIME 2025 final: {correct}/{total} = {accuracy:.4f}")
    return {"aime2025/accuracy": accuracy, "aime2025/correct": correct, "aime2025/total": total}


# ---------------------------------------------------------------------------
# MBPP Evaluation (code generation)
# ---------------------------------------------------------------------------

async def eval_mbpp(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on MBPP (Mostly Basic Python Programming) via code execution.

    Generates a Python function and runs assertion-based test cases.
    Uses subprocess execution (no Modal dependency for eval).
    """
    import subprocess
    import tempfile

    try:
        ds = cast(Dataset, load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    except Exception:
        ds = cast(Dataset, load_dataset("google-research-datasets/mbpp", "full", split="test"))

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    semaphore = asyncio.Semaphore(concurrency)

    def _run_python_tests(code: str, test_assertions: list[str], timeout: int = 15) -> bool:
        """Run code + assertion tests in a subprocess. Returns True if all pass."""
        test_code = code + "\n\n" + "\n".join(test_assertions)
        try:
            result = subprocess.run(
                ["python3", "-c", test_code],
                capture_output=True,
                timeout=timeout,
                text=True,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False

    async def eval_one(row: dict) -> bool | None:
        async with semaphore:
            task_prompt = row.get("prompt", row.get("text", ""))
            test_list = row.get("test_list", [])
            if not task_prompt or not test_list:
                return None

            # Show 1 test example in the prompt for format guidance
            example_test = test_list[0] if test_list else ""
            prompt = (
                f"{task_prompt}\n\n"
                f"Example test: `{example_test}`\n\n"
                "Write a Python function that satisfies the requirements. "
                "Provide ONLY the function definition in a ```python code block."
            )
            messages: list[Message] = [{"role": "user", "content": prompt}]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)

                # Extract code using the same logic as code_rl_env
                import re
                match = re.search(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
                if not match:
                    matches = re.findall(r'```(?:\w*)\s*\n(.*?)\n```', content, re.DOTALL)
                    code = matches[-1].strip() if matches else content.strip()
                else:
                    code = match.group(1).strip()

                # Run tests in a thread to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                passed = await loop.run_in_executor(None, _run_python_tests, code, test_list)
                return passed
            except Exception as e:
                logger.warning(f"MBPP eval failed: {e}")
                return None

    logger.info(f"MBPP: evaluating {len(ds)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in ds]
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r is True)
    total = sum(1 for r in results if r is not None)
    accuracy = correct / total if total > 0 else 0
    logger.info(f"MBPP final: {correct}/{total} = {accuracy:.4f}")
    return {"mbpp/accuracy": accuracy, "mbpp/correct": correct, "mbpp/total": total}


# ---------------------------------------------------------------------------
# LongBench v2 Evaluation
# ---------------------------------------------------------------------------

async def eval_longbench(
    completer: TinkerMessageCompleter,
    limit: int | None = None,
    concurrency: int = 128,
) -> dict[str, float]:
    """Evaluate on LongBench v2 — long-context comprehension across multiple subtasks.

    Loads THUDM/LongBench-v2 which contains multiple-choice questions with
    long contexts. Falls back to THUDM/LongBench if v2 is unavailable.
    """
    import re

    ds = None
    dataset_version = "v2"
    for dataset_id, split in (
        ("THUDM/LongBench-v2", "test"),
        ("THUDM/LongBench-v2", "train"),
        ("THUDM/LongBench", "test"),
    ):
        try:
            ds = cast(Dataset, load_dataset(dataset_id, split=split))
            dataset_version = "v2" if "v2" in dataset_id else "v1"
            logger.info(f"Loaded LongBench from {dataset_id}/{split} ({len(ds)} examples)")
            break
        except Exception:
            # Try loading with a specific config name
            try:
                ds = cast(Dataset, load_dataset(dataset_id, "default", split=split))
                dataset_version = "v2" if "v2" in dataset_id else "v1"
                logger.info(f"Loaded LongBench from {dataset_id}/default/{split} ({len(ds)} examples)")
                break
            except Exception:
                continue

    if ds is None:
        logger.warning("Could not load LongBench dataset. Skipping.")
        return {"longbench/accuracy": 0.0, "longbench/correct": 0, "longbench/total": 0}

    if limit:
        ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))

    semaphore = asyncio.Semaphore(concurrency)

    async def eval_one(row: dict) -> tuple[bool, str] | None:
        """Returns (correct, subtask) or None on failure."""
        async with semaphore:
            # LongBench v2 format: context, question, choice_A/B/C/D, answer
            # LongBench v1 format: context, input, answers (list)
            context = row.get("context", "")
            subtask = row.get("domain", row.get("dataset", "unknown"))

            if dataset_version == "v2":
                question = row.get("question", row.get("input", ""))
                # v2 has choice_A, choice_B, choice_C, choice_D columns
                choices = []
                for letter in ("A", "B", "C", "D"):
                    choice = row.get(f"choice_{letter}", "")
                    if choice:
                        choices.append(choice)

                expected = str(row.get("answer", "")).strip().upper()

                if choices:
                    choice_text = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
                    user_content = (
                        f"Read the following text carefully, then answer the question.\n\n"
                        f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                        f"Question: {question}\n\n{choice_text}\n\n"
                        "Answer with just the letter (A, B, C, or D)."
                    )
                else:
                    user_content = (
                        f"Read the following text carefully, then answer the question.\n\n"
                        f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                        f"Question: {question}\n\n"
                        "Give a concise answer."
                    )
            else:
                # LongBench v1 format
                question = row.get("input", "")
                expected_answers = row.get("answers", row.get("all_classes", []))
                expected = expected_answers[0] if expected_answers else ""
                user_content = (
                    f"Read the following text carefully, then answer the question.\n\n"
                    f"--- TEXT ---\n{context}\n--- END TEXT ---\n\n"
                    f"Question: {question}\n\n"
                    "Give a concise answer."
                )

            if not question or not context:
                return None

            messages: list[Message] = [{"role": "user", "content": user_content}]
            try:
                response = await completer(messages)
                content = renderers.get_text_content(response)

                if dataset_version == "v2" and expected in ("A", "B", "C", "D"):
                    # Multiple choice grading
                    letters = re.findall(r'\b([A-D])\b', content[-300:])
                    extracted = letters[-1] if letters else ""
                    return (extracted == expected, subtask)
                else:
                    # Open-ended: check if expected answer appears in response
                    expected_lower = str(expected).strip().lower()
                    content_lower = content.strip().lower()
                    correct = expected_lower in content_lower
                    return (correct, subtask)
            except Exception as e:
                logger.warning(f"LongBench eval failed: {e}")
                return None

    logger.info(f"LongBench ({dataset_version}): evaluating {len(ds)} samples with concurrency={concurrency}")
    tasks = [eval_one(row) for row in ds]
    results = await asyncio.gather(*tasks)

    valid = [r for r in results if r is not None]
    correct = sum(1 for r in valid if r[0])
    total = len(valid)
    accuracy = correct / total if total > 0 else 0

    # Per-subtask breakdown
    subtask_results: dict[str, list[bool]] = {}
    for r in valid:
        st = r[1]
        subtask_results.setdefault(st, []).append(r[0])

    metrics: dict[str, float] = {
        "longbench/accuracy": accuracy,
        "longbench/correct": correct,
        "longbench/total": total,
    }
    for st, st_results in sorted(subtask_results.items()):
        st_acc = sum(st_results) / len(st_results) if st_results else 0
        metrics[f"longbench/{st}/accuracy"] = st_acc

    logger.info(f"LongBench final: {correct}/{total} = {accuracy:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "gsm8k": eval_gsm8k,
    "ifeval": eval_ifeval,
    "mmlu": eval_mmlu,
    "math500": eval_math500,
    "gpqa": eval_gpqa,
    "aime2025": eval_aime2025,
    "mbpp": eval_mbpp,
    "longbench": eval_longbench,
}


async def run_eval(
    model_name: str,
    checkpoint_path: str | None,
    benchmarks: list[str],
    limit: int | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    output_dir: str | None = None,
):
    """Run evaluation on a single checkpoint."""
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()
    if checkpoint_path:
        sampling_client = await service_client.create_sampling_client_async(model_path=checkpoint_path)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        sampling_client = await service_client.create_sampling_client_async(base_model=model_name)
        logger.info("Using base model")

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    all_results = {}
    for bench_name in benchmarks:
        if bench_name not in BENCHMARKS:
            logger.warning(f"Unknown benchmark: {bench_name}")
            continue
        logger.info(f"\n--- Running {bench_name} ---")
        results = await BENCHMARKS[bench_name](completer, limit=limit)
        all_results.update(results)

    # Print results
    print("\n" + "=" * 50)
    cp_label = checkpoint_path.split("/")[-1] if checkpoint_path else "base"
    print(f"Results for: {cp_label}")
    print("=" * 50)
    for k, v in sorted(all_results.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump({
                "model": model_name,
                "checkpoint": checkpoint_path,
                "timestamp": datetime.now().isoformat(),
                "results": {k: float(v) if isinstance(v, float) else v for k, v in all_results.items()},
            }, f, indent=2)

    return all_results


async def compare_checkpoints(
    model_name: str,
    checkpoints: dict[str, str | None],
    benchmarks: list[str],
    limit: int | None = None,
    output_dir: str | None = None,
):
    """Run evals on multiple checkpoints and print comparison."""
    all_results = {}
    for name, cp in checkpoints.items():
        logger.info(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
        cp_out = os.path.join(output_dir, name) if output_dir else None
        results = await run_eval(model_name, cp, benchmarks, limit=limit, output_dir=cp_out)
        all_results[name] = results

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    metrics = sorted(set(k for r in all_results.values() for k in r.keys()))
    names = list(all_results.keys())
    header = f"{'Metric':<35}" + "".join(f"{n:<15}" for n in names)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:<35}"
        for n in names:
            v = all_results[n].get(m, "N/A")
            row += f"{v:<15.4f}" if isinstance(v, float) else f"{str(v):<15}"
        print(row)

    if output_dir:
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(all_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-120b:peft:131072")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--benchmarks", default="gsm8k,ifeval", help="Comma-separated list")
    parser.add_argument("--limit", type=int, default=100, help="Max samples per benchmark")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--ifrl-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    benchmarks = args.benchmarks.split(",")

    if args.compare:
        cps = {"base": None}
        if args.sft_checkpoint:
            cps["sft"] = args.sft_checkpoint
        if args.ifrl_checkpoint:
            cps["ifrl"] = args.ifrl_checkpoint
        out = args.output_dir or os.path.expanduser(
            f"~/data/nemotron-cascade-2/evals/compare_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        asyncio.run(compare_checkpoints(args.model, cps, benchmarks, limit=args.limit, output_dir=out))
    else:
        out = args.output_dir or os.path.expanduser(
            f"~/data/nemotron-cascade-2/evals/eval_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        asyncio.run(run_eval(args.model, args.checkpoint, benchmarks, limit=args.limit, output_dir=out))


if __name__ == "__main__":
    main()
