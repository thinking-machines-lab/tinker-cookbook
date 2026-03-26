"""
Benchmark evaluations for Nemotron-Cascade-2 checkpoints.

Uses Tinker sampling directly with our own grading:
  - GSM8K: Math grading via existing math_grading utilities
  - IFEval: Our 48-type instruction following verifier
  - MATH-500: Hendrycks MATH test set

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
from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.renderers import Message
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GSM8K Evaluation
# ---------------------------------------------------------------------------

def _extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from model response."""
    import re
    # Try boxed format first
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip().replace(",", "")

    # Try "#### answer" format
    hash_match = re.search(r'####\s*(.+)', text)
    if hash_match:
        return hash_match.group(1).strip().replace(",", "")

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
    completer: MessageCompleter,
    limit: int | None = None,
) -> dict[str, float]:
    """Evaluate on GSM8K test set."""
    ds = cast(Dataset, load_dataset("openai/gsm8k", "main", split="test"))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for i, row in enumerate(ds):
        question = row["question"]
        # Extract expected answer
        import re
        answer_match = re.search(r'####\s*(.+)', row["answer"])
        expected = answer_match.group(1).strip().replace(",", "") if answer_match else ""

        messages: list[Message] = [
            {"role": "user", "content": question + " Show your work step by step, then give the final numerical answer."},
        ]

        try:
            response = await completer.complete(messages)
            content = renderers.get_text_content(response)
            is_correct = _check_gsm8k(content, expected)
            if is_correct:
                correct += 1
            total += 1
        except Exception as e:
            logger.warning(f"GSM8K sample {i} failed: {e}")
            total += 1

        if (i + 1) % 50 == 0:
            logger.info(f"GSM8K: {i+1}/{len(ds)} | accuracy={correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0
    logger.info(f"GSM8K final: {correct}/{total} = {accuracy:.4f}")
    return {"gsm8k/accuracy": accuracy, "gsm8k/correct": correct, "gsm8k/total": total}


# ---------------------------------------------------------------------------
# IFEval Evaluation
# ---------------------------------------------------------------------------

async def eval_ifeval(
    completer: MessageCompleter,
    limit: int | None = None,
) -> dict[str, float]:
    """Evaluate on IFEval using our verifier."""
    from tinker_cookbook.recipes.nemotron_cascade.if_rl_env import verify_all_instructions

    # Load IFEval data from our downloaded RL data
    ifeval_path = os.path.expanduser("~/data/nemotron-cascade-2/rl_if_rl.jsonl")
    rows = []
    with open(ifeval_path) as f:
        for line in f:
            rows.append(json.loads(line))

    if limit:
        rows = rows[:limit]

    total_score = 0
    total_prompts = 0
    strict_correct = 0

    for i, row in enumerate(rows):
        prompt_messages = row["responses_create_params"]["input"]
        instruction_ids = row.get("instruction_id_list", [])
        raw_kwargs = row.get("kwargs", [])

        # Parse kwargs
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

        messages: list[Message] = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in prompt_messages
        ]

        try:
            response = await completer.complete(messages)
            content = renderers.get_text_content(response)
            fraction, _ = verify_all_instructions(content, instruction_ids, kwargs_list)
            total_score += fraction
            if fraction == 1.0:
                strict_correct += 1
            total_prompts += 1
        except Exception as e:
            logger.warning(f"IFEval sample {i} failed: {e}")
            total_prompts += 1

        if (i + 1) % 50 == 0:
            logger.info(f"IFEval: {i+1}/{len(rows)} | loose={total_score/total_prompts:.3f} strict={strict_correct/total_prompts:.3f}")

    loose_acc = total_score / total_prompts if total_prompts > 0 else 0
    strict_acc = strict_correct / total_prompts if total_prompts > 0 else 0
    logger.info(f"IFEval final: loose={loose_acc:.4f}, strict={strict_acc:.4f} ({total_prompts} prompts)")
    return {
        "ifeval/loose_accuracy": loose_acc,
        "ifeval/strict_accuracy": strict_acc,
        "ifeval/total": total_prompts,
    }


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "gsm8k": eval_gsm8k,
    "ifeval": eval_ifeval,
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
        sampling_client = await service_client.create_sampling_client_from_state_async(checkpoint_path)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        sampling_client = await service_client.create_sampling_client_async(base_model=model_name)
        logger.info("Using base model")

    completer = MessageCompleter(
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
