import asyncio
from typing import Literal, cast

import chz
import tinker
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.math_with_confidence.env import (
    BrierRewardMode,
    DEFAULT_CONSISTENCY_GRADER_MODEL,
    MathWithConfidenceEnv,
    get_dataset_builder,
    parse_answer_and_confidence,
)

console = Console()


@chz.chz
class InteractiveConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    model_path: str | None = None
    renderer_name: str | None = None

    dataset_name: Literal["math", "polaris", "deepmath", "gsm8k"] = "math"
    split: Literal["train", "test"] = "test"
    start_batch_index: int = 0
    num_examples: int = 5
    seed: int = 0

    alpha: float = 0.5
    consistency_v2_coef: float = 0.0
    brier_reward_mode: BrierRewardMode = "one_minus_squared_error"
    include_fewshot: bool = True
    consistency_grader_model_name: str = DEFAULT_CONSISTENCY_GRADER_MODEL
    consistency_grader_max_tokens: int = 256

    max_tokens: int = 768
    temperature: float = 0.7
    base_url: str | None = None
    pause_between_examples: bool = True


def _metric_style(value: float) -> str:
    return "green" if value > 0 else "red"


def _make_result_table(
    reward: float,
    correct: float,
    confidence: float,
    brier_term: float,
    consistency_v2: float,
    valid_format: bool,
) -> Table:
    table = Table(title="Grading Result")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("format_valid", f"[{'green' if valid_format else 'red'}]{valid_format}[/]")
    table.add_row("correct", f"[{_metric_style(correct)}]{correct:.3f}[/]")
    table.add_row("confidence", f"[cyan]{confidence:.3f}[/]")
    table.add_row("brier_term", f"[{_metric_style(brier_term)}]{brier_term:.4f}[/]")
    table.add_row("consistency_v2", f"[{_metric_style(consistency_v2)}]{consistency_v2:.4f}[/]")
    table.add_row("total_reward", f"[{_metric_style(reward)}]{reward:.4f}[/]")
    return table


async def cli_main(cfg: InteractiveConfig):
    renderer_name = cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name)

    dataset_builder = get_dataset_builder(
        dataset_name=cfg.dataset_name,
        batch_size=1,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        group_size=1,
        alpha=cfg.alpha,
        consistency_v2_coef=cfg.consistency_v2_coef,
        brier_reward_mode=cfg.brier_reward_mode,
        include_fewshot=cfg.include_fewshot,
        base_url=cfg.base_url,
        consistency_grader_model_name=cfg.consistency_grader_model_name,
        consistency_grader_max_tokens=cfg.consistency_grader_max_tokens,
        seed=cfg.seed,
    )
    train_ds, test_ds = await dataset_builder()
    dataset = test_ds if cfg.split == "test" and test_ds is not None else train_ds
    if cfg.split == "test" and test_ds is None:
        console.print("[yellow]Test split not available for this dataset; using train split.[/]")

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    if cfg.model_path:
        sampling_client = service_client.create_sampling_client(
            base_model=cfg.model_name,
            model_path=cfg.model_path,
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    console.print(
        f"[bold]Inspecting {cfg.num_examples} examples[/] from dataset={cfg.dataset_name}, split={cfg.split}"
    )
    for batch_index in range(cfg.start_batch_index, cfg.start_batch_index + cfg.num_examples):
        builders = dataset.get_batch(batch_index)
        if not builders:
            console.print(f"[yellow]No builders returned for batch {batch_index}. Stopping.[/]")
            break
        envs = await builders[0].make_envs()
        env = cast(MathWithConfidenceEnv, envs[0])

        ob, stop = await env.initial_observation()
        prompt_text = env.renderer.tokenizer.decode(ob.to_ints())
        sample = await policy(ob, stop)
        step = await env.step(sample.tokens)

        raw_response = str(step.logs.get("response", ""))
        parsed = parse_answer_and_confidence(raw_response)
        confidence = float(step.metrics.get("confidence", 0.0))
        correct = float(step.metrics.get("correct", 0.0))
        brier_term = float(step.metrics.get("brier_term", 0.0))
        consistency_v2 = float(step.metrics.get("consistency_v2", 0.0))

        console.rule(f"Example batch={batch_index}")
        console.print(Panel(prompt_text, title="Prompt", border_style="blue"))
        console.print(Panel(raw_response, title="Model Response", border_style="magenta"))
        console.print(
            Panel(
                str(step.logs.get("reference_answer", "")),
                title="Reference Answer",
                border_style="green",
            )
        )
        console.print(
            Panel(
                f"answer={parsed.answer!r}\nconfidence={parsed.confidence!r}",
                title="Parsed Output",
                border_style="cyan",
            )
        )
        if parsed.parse_error is not None:
            console.print(
                Panel(parsed.parse_error, title="Format Parse Error", border_style="red"),
            )
        grader_response_v2 = str(step.logs.get("consistency_v2_grader_response", "")).strip()
        if grader_response_v2:
            console.print(
                Panel(
                    grader_response_v2,
                    title="Consistency V2 Grader Response",
                    border_style="yellow",
                )
            )
        console.print(
            _make_result_table(
                step.reward,
                correct,
                confidence,
                brier_term,
                consistency_v2,
                parsed.valid_format,
            )
        )

        if cfg.pause_between_examples and batch_index < cfg.start_batch_index + cfg.num_examples - 1:
            if not Confirm.ask("Continue to next example?", default=True):
                break


if __name__ == "__main__":
    cfg = chz.entrypoint(InteractiveConfig)
    asyncio.run(cli_main(cfg))
