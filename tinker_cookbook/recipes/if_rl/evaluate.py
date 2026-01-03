"""Unified evaluation for IFEval and IFBench using if-verifiable library.

Usage:
    python -m tinker_cookbook.recipes.if_rl.evaluate --benchmark ifeval
    python -m tinker_cookbook.recipes.if_rl.evaluate --benchmark ifbench
    python -m tinker_cookbook.recipes.if_rl.evaluate --benchmark ifeval --tinker_checkpoint_url "tinker://..."
"""

import asyncio
import json
from typing import Literal

import chz
import tinker
from if_verifiable import EvaluationScores, get_eval_data, run_eval_async

from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

ROLLOUT_CONCURRENCY = 256
Benchmark = Literal["ifeval", "ifbench"]


@chz.chz
class CLIConfig:
    benchmark: Benchmark = chz.field(default="ifeval", doc="Benchmark: 'ifeval' or 'ifbench'")
    base_model: str = chz.field(default="Qwen/Qwen3-4B-Instruct-2507", doc="Base model")
    tinker_checkpoint_url: str | None = chz.field(default=None, doc="Optional tinker:// checkpoint")
    output_file: str | None = chz.field(default=None, doc="Path to save results as JSONL")
    max_tokens: int = chz.field(default=2048, doc="Maximum tokens to generate")
    temperature: float = chz.field(default=0.0, doc="Sampling temperature")
    seed: int = chz.field(default=42, doc="Random seed")


async def run_evaluation(config: CLIConfig) -> dict[str, float]:
    """Run IF evaluation using if-verifiable library."""
    print(f"Loading {config.benchmark.upper()} dataset...")
    samples = list(get_eval_data(config.benchmark))
    print(f"Total samples: {len(samples)}")

    print(f"Loading model: {config.base_model}")
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(config.base_model), tokenizer=tokenizer
    )

    service_client = tinker.ServiceClient()
    if config.tinker_checkpoint_url:
        print(f"Sampling from checkpoint: {config.tinker_checkpoint_url}")
        sampling_client = service_client.create_sampling_client(
            model_path=config.tinker_checkpoint_url, base_model=config.base_model
        )
    else:
        print(f"Sampling from base model: {config.base_model}")
        sampling_client = service_client.create_sampling_client(base_model=config.base_model)

    policy = TinkerTokenCompleter(
        sampling_client, max_tokens=config.max_tokens, temperature=config.temperature
    )
    semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)

    async def get_response(prompt: str) -> str:
        messages: list[renderers.Message] = [{"role": "user", "content": prompt}]
        async with semaphore:
            response = await policy(
                renderer.build_generation_prompt(messages), renderer.get_stop_sequences()
            )
        message, _ = renderer.parse_response(response.tokens)
        return renderers.get_text_content(message)

    print(f"\nEvaluating {len(samples)} items...")
    results = await run_eval_async(config.benchmark, [get_response(s.prompt) for s in samples])

    if config.output_file:
        with open(config.output_file, "w") as f:
            for sample, response, _, scores in results:
                f.write(
                    json.dumps(
                        {
                            "prompt": sample.prompt,
                            "response": response,
                            "partial_strict": scores.partial_strict,
                            "partial_loose": scores.partial_loose,
                            "binary_strict": scores.binary_strict,
                            "binary_loose": scores.binary_loose,
                        }
                    )
                    + "\n"
                )

    n = len(results)
    scores_list: list[EvaluationScores] = [r[3] for r in results]
    prompt_strict = sum(s.binary_strict for s in scores_list) / n
    prompt_loose = sum(s.binary_loose for s in scores_list) / n
    instr_strict = sum(s.partial_strict for s in scores_list) / n
    instr_loose = sum(s.partial_loose for s in scores_list) / n

    print("\n" + "=" * 70)
    print(f"{config.benchmark.upper()} RESULTS")
    print("=" * 70)
    if config.tinker_checkpoint_url:
        print(f"Checkpoint: {config.tinker_checkpoint_url}")
    print(f"Base model: {config.base_model}")
    print(f"Samples: {n}")
    print("-" * 70)
    print(f"\nPrompt-level (strict):      {prompt_strict:.4f}")
    print(f"Prompt-level (loose):       {prompt_loose:.4f}")
    print(f"Instruction-level (strict): {instr_strict:.4f}")
    print(f"Instruction-level (loose):  {instr_loose:.4f}")
    print("=" * 70)
    if config.output_file:
        print(f"\nResults saved to: {config.output_file}")

    return {
        "prompt_strict": prompt_strict,
        "prompt_loose": prompt_loose,
        "instruction_strict": instr_strict,
        "instruction_loose": instr_loose,
    }


if __name__ == "__main__":
    asyncio.run(run_evaluation(chz.entrypoint(CLIConfig)))
