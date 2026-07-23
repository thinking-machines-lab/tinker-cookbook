"""Compare effort-conditioned Inkling responses.

The script samples the same prompt concurrently at every configured effort. It
maps standard scalar values back to their preset names and prints each response,
generated-token count, and parse termination. Its default prompt is AIME 2025
problem 26, for which it also prints the expected answer.

    uv run python -m tinker_cookbook.scripts.inkling.sample_reasoning

Override the defaults with, for example, ``efforts=[0.3,0.9]``. The generation
budget scales with effort; pass ``max_tokens=...`` to pin one budget for every
effort instead.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast

import chz
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.renderers import Message, ParseTermination, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

_AIME_DATASET = "MathArena/aime_2025"
_DEFAULT_PROBLEM_NUMBER = 26

# Default sweep presets and their TMLv0 renderer effort values.
_EFFORT_NAME_TO_FLOAT = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
}
_EFFORT_NAMES_BY_VALUE = {value: name for name, value in _EFFORT_NAME_TO_FLOAT.items()}


def _default_max_tokens(effort: float) -> int:
    """Pick a generation budget that scales with reasoning effort.

    Higher effort generally produces longer reasoning traces, so a budget that
    is comfortable at medium effort can truncate xhigh responses mid-thought.
    """
    if effort < 0.2:
        return 4096
    if effort < 0.9:
        return 8192
    return 16384


@dataclass(frozen=True)
class SampleResult:
    effort: float
    answer: str
    num_generated_tokens: int
    max_tokens: int
    termination: ParseTermination


def _load_aime_problem(problem_number: int) -> tuple[str, int]:
    dataset = cast(Dataset, load_dataset(_AIME_DATASET, split="train"))
    rows = dataset.filter(lambda row: row["problem_idx"] == problem_number)
    if len(rows) != 1:
        raise ValueError(f"AIME 2025 problem number must be between 1 and {len(dataset)}.")

    row = rows[0]
    prompt = (
        f"{row['problem']}\n\n"
        "This is an AIME problem. Show your work, then put the final three-digit "
        "answer in \\boxed{}."
    )
    return prompt, int(row["answer"])


@chz.chz
class Config:
    problem_number: int = chz.field(
        default=_DEFAULT_PROBLEM_NUMBER,
        doc="One-indexed problem number from the public MathArena AIME 2025 dataset.",
    )
    prompt: str | None = chz.field(
        default=None,
        doc="Custom prompt. When set, overrides problem_number and has no expected answer.",
    )
    efforts: list[float] = chz.field(
        default_factory=lambda: list(_EFFORT_NAME_TO_FLOAT.values()),
        doc="TMLv0 reasoning-effort values to compare.",
    )
    model_name: str = chz.field(default="thinkingmachines/Inkling")
    base_url: str | None = None
    max_tokens: int | None = chz.field(
        default=None,
        doc="Generation budget for every effort. When None, scales with effort.",
    )
    temperature: float = 1.0


async def _sample_at_effort(
    sampling_client: tinker.SamplingClient,
    renderer: TmlV0Renderer,
    messages: list[Message],
    effort: float,
    cfg: Config,
) -> SampleResult:
    prompt = renderer.build_generation_prompt(messages, effort=effort)
    max_tokens = cfg.max_tokens if cfg.max_tokens is not None else _default_max_tokens(effort)
    response = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=cfg.temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    sequence = response.sequences[0]
    message, termination = renderer.parse_response(sequence.tokens)
    return SampleResult(
        effort=effort,
        answer=get_text_content(message),
        num_generated_tokens=len(sequence.tokens),
        max_tokens=max_tokens,
        termination=termination,
    )


async def async_main(cfg: Config) -> None:
    if not cfg.efforts:
        raise ValueError("Provide at least one reasoning-effort value.")

    if cfg.prompt is None:
        prompt, expected_answer = _load_aime_problem(cfg.problem_number)
    else:
        prompt, expected_answer = cfg.prompt, None

    renderer = TmlV0Renderer(get_tokenizer(cfg.model_name))
    messages = [Message(role="user", content=prompt)]
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    sampling_client = await service_client.create_sampling_client_async(base_model=cfg.model_name)

    results = await asyncio.gather(
        *(
            _sample_at_effort(sampling_client, renderer, messages, effort, cfg)
            for effort in cfg.efforts
        ),
    )

    print(f"Prompt: {prompt}")
    if expected_answer is not None:
        print(f"Expected answer: {expected_answer:03d}")
    for result in results:
        effort_name = _EFFORT_NAMES_BY_VALUE.get(result.effort, "custom")
        print(f"\nEffort: {effort_name} ({result.effort:g})")
        print(f"Generated tokens: {result.num_generated_tokens} (max_tokens: {result.max_tokens})")
        print(f"Answer: {result.answer}")
        print(f"Termination: {result.termination.value}")
        if result.num_generated_tokens >= result.max_tokens:
            print("Response hit max_tokens; raise it to avoid truncation.")


def cli_main(cfg: Config) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    cli_main(chz.entrypoint(Config))
