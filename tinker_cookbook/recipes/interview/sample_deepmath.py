"""
Sample from Qwen3-30B-A3B (thinking mode) on DeepMath problems.

Loads 10 problems from DeepMath-103K, samples completions with thinking enabled,
and saves the results to a JSON file.

Usage:
    python -m tinker_cookbook.recipes.interview.sample_deepmath
"""

import asyncio
import json
import logging
from pathlib import Path

import tinker
from datasets import load_dataset
from dotenv import load_dotenv

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
NUM_PROBLEMS = 10
MAX_TOKENS = 16384
TEMPERATURE = 0.6
OUTPUT_PATH = Path("/tmp/tinker-examples/interview/deepmath_samples.json")


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Load dataset
    logger.info("Loading DeepMath-103K dataset...")
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    ds = ds.shuffle(seed=42)
    problems = [ds[i] for i in range(NUM_PROBLEMS)]

    # Set up renderer and tokenizer
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    logger.info(f"Using renderer: {renderer_name}")
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    stop_sequences = renderer.get_stop_sequences()

    # Create sampling client directly from base model
    logger.info(f"Creating sampling client for {MODEL_NAME}...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    # Build all prompts
    sample_params = tinker.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=stop_sequences,
    )
    prompts = []
    for problem in problems:
        question = problem["question"]
        messages: list[renderers.Message] = [
            {
                "role": "user",
                "content": question
                + " Write your answer in \\boxed{} format. Don't think for too long unnecessarily, especially when you have a reasonable degree of confidence.",
            },
        ]
        prompts.append(renderer.build_generation_prompt(messages))

    # Fire all sample requests in parallel
    logger.info(f"Submitting {NUM_PROBLEMS} sample requests in parallel...")
    sample_results = await asyncio.gather(
        *[
            sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=sample_params,
            )
            for prompt in prompts
        ]
    )

    # Process results
    results = []
    for i, (problem, sample_result) in enumerate(zip(problems, sample_results)):
        response_tokens = sample_result.sequences[0].tokens
        parsed_message, parse_success = renderer.parse_response(response_tokens)

        # Extract thinking and visible content
        content = parsed_message["content"]
        thinking = ""
        visible = ""
        if isinstance(content, list):
            for part in content:
                if part["type"] == "thinking":
                    thinking = part["thinking"]
                elif part["type"] == "text":
                    visible += part["text"]
        else:
            visible = content

        result = {
            "index": i,
            "question": problem["question"],
            "ground_truth": problem["final_answer"],
            "thinking": thinking,
            "response": visible,
            "full_raw": tokenizer.decode(response_tokens),
            "num_tokens": len(response_tokens),
            "parse_success": parse_success,
        }
        results.append(result)
        logger.info(
            f"[{i + 1}/{NUM_PROBLEMS}] {len(response_tokens)} tokens, "
            f"thinking: {len(thinking)} chars, response: {len(visible)} chars"
        )

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
