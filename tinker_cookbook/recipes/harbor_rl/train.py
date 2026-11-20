"""CLI entry point for Harbor RL training."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import chz
import tinker
from aiohttp import web
from typing_extensions import override

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.capture import current_scope
from tinker_cookbook.completers import TinkerTokenCompleter, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    HarborDatasetBuilder,
    HarborEnvGroupBuilder,
    HarborTask,
    SandboxFactory,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, Transition
from tinker_cookbook.tokenizer_utils import get_tokenizer

if TYPE_CHECKING:
    from harbor import AsyncHarbor  # pyright: ignore[reportMissingImports]
    from harbor.server.models import RolloutSummary  # pyright: ignore[reportMissingImports]

    from tinker_cookbook.capture.proxy.app import SampleResponseLike

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for Harbor RL training."""

    # Model configuration
    model_name: str = "moonshotai/Kimi-K2.6"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    max_tokens: int = 8192
    temperature: float = 1.0

    # Environment configuration
    max_turns: int = 10
    sandbox_timeout: int = 3600
    command_timeout: int = 120
    grader_timeout: int = 60
    max_trajectory_tokens: int = 32 * 1024
    max_generation_tokens: int | None = None
    context_overflow_reward: float = -0.1

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 8
    learning_rate: float = 1e-5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging / eval / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    # Display name for the Tinker session; defaults to wandb_name, or the
    # generated run name when wandb_name is unset.
    session_name: str | None = None
    eval_every: int = 5
    save_every: int = 5

    # Service configuration
    harbor_server_url: str | None = None
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Setting this enables async training; see AsyncConfig.sampling_steps_ahead.
    sampling_steps_ahead: int | None = None
    # Deprecated; use sampling_steps_ahead.
    max_steps_off_policy: int | None = None

    max_steps: int | None = None


async def cli_main(
    cli_config: CLIConfig,
    tasks: list[HarborTask],
    sandbox_factory: SandboxFactory | None = None,
) -> None:
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"harbor-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or cli_utils.runs_path("harbor_rl", run_name)
    wandb_name = cli_config.wandb_name or run_name
    max_generation_tokens = (
        cli_config.max_generation_tokens
        if cli_config.max_generation_tokens is not None
        else cli_config.max_tokens
    )

    dataset_builder = HarborDatasetBuilder(
        tasks=tasks,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        max_turns=cli_config.max_turns,
        sandbox_timeout=cli_config.sandbox_timeout,
        command_timeout=cli_config.command_timeout,
        grader_timeout=cli_config.grader_timeout,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens,
        context_overflow_reward=cli_config.context_overflow_reward,
        sandbox_factory=sandbox_factory,
    )

    relay = (
        HarborRolloutStrategy(cli_config.harbor_server_url)
        if cli_config.harbor_server_url
        else None
    )
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        recipe_name="recipe_harbor_rl",
        rollout_error_tolerance=relay,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        session_name=cli_config.session_name or wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            sampling_steps_ahead=cli_config.sampling_steps_ahead,
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.sampling_steps_ahead is not None
        or cli_config.max_steps_off_policy is not None
        else None,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    if relay is None:
        await main(config)
    else:
        from tinker_cookbook.capture.store import capture_to_store

        with capture_to_store(run_name, data_dir=Path(log_path) / "capture"):
            async with _relay_sessions():
                await main(config)


class RelaySession:
    """Share one policy sampler and connector across active Harbor jobs."""

    def __init__(self, policy: TinkerTokenCompleter):
        self.session_id = str(uuid4())
        self.ready = asyncio.Event()
        self.finished = asyncio.Event()
        self.jobs = 0
        self.policy = policy
        self.turns: dict[str, list[Transition]] = defaultdict(list)

    async def sample(
        self, prompt: tinker.ModelInput, num_samples: int, sampling_params: tinker.SamplingParams
    ) -> SampleResponseLike:
        rollout_id = current_scope()["rollout_id"]
        if not isinstance(rollout_id, str):
            raise ValueError("Inference request is missing its Harbor rollout ID")
        params = sampling_params.model_copy(
            update={
                "max_tokens": min(
                    sampling_params.max_tokens or self.policy.max_tokens, self.policy.max_tokens
                ),
                "temperature": self.policy.temperature,
            }
        )
        response = await self.policy.sampling_client.sample(prompt, num_samples, params)
        (seq,) = response.sequences
        if not seq.tokens or seq.logprobs is None or len(seq.logprobs) != len(seq.tokens):
            raise ValueError("Incomplete token capture from Tinker")
        action = TokensWithLogprobs(seq.tokens, seq.logprobs, seq.stop_reason, seq.sequence_id)
        self.turns[rollout_id].append(Transition(prompt, action, reward=0, episode_done=False))
        # Pyright cannot match the SDK's cached tokens property to the proxy protocol.
        return cast("SampleResponseLike", response)


@dataclass
class RelaySessions:
    connectors: asyncio.TaskGroup
    active: dict[tuple[tinker.SamplingClient, int, float], RelaySession] = field(
        default_factory=dict
    )


_relays = ContextVar[RelaySessions]("harbor_relay_sessions")


@asynccontextmanager
async def _relay_sessions() -> AsyncIterator[None]:
    async with asyncio.TaskGroup() as connectors:
        token = _relays.set(RelaySessions(connectors))
        try:
            yield
        finally:
            _relays.reset(token)


@dataclass(frozen=True)
class HarborRolloutStrategy(RolloutStrategy):
    harbor_server_url: str

    @asynccontextmanager
    async def _session(
        self, group: HarborEnvGroupBuilder, policy: TinkerTokenCompleter
    ) -> AsyncIterator[RelaySession]:
        # The trainer wraps the same sampling client in a new completer for each group.
        key = (policy.sampling_client, policy.max_tokens, policy.temperature)
        relays = _relays.get()
        session = relays.active.get(key)
        if session is None:
            session = relays.active[key] = RelaySession(policy)
            relays.connectors.create_task(_serve_session(session, group, self.harbor_server_url))
        session.jobs += 1
        try:
            await session.ready.wait()
            yield session
        finally:
            session.jobs -= 1
            if session.jobs == 0:
                del relays.active[key]
                session.finished.set()

    @override
    async def execute(
        self, env_group_builder: EnvGroupBuilder, policy: TokenCompleter
    ) -> RolloutResult:
        from harbor import AsyncHarbor  # pyright: ignore[reportMissingImports]

        group = cast(HarborEnvGroupBuilder, env_group_builder)
        async with (
            self._session(group, cast(TinkerTokenCompleter, policy)) as session,
            AsyncHarbor(base_url=self.harbor_server_url) as client,
        ):
            trajectories = []
            async for rollout in _run_job(client, session.session_id, group):
                reward = (rollout.rewards or {}).get("reward")
                turns = session.turns.pop(str(rollout.id), [])
                if reward is None or not turns:
                    raise ValueError(
                        f"Rollout {rollout.id} missing valid reward or captured samples"
                    )
                turns[-1].reward = float(reward)
                turns[-1].episode_done = True
                trajectories.append(Trajectory(turns, final_ob=tinker.ModelInput.empty()))
                logger.info(
                    "Harbor rollout %s: reward=%s, samples=%d", rollout.id, reward, len(turns)
                )
            if len(trajectories) != group.group_size:
                raise ValueError("Incomplete Harbor rollout group")
            return RolloutResult(trajectories, envs=[], errors=[])


async def _serve_session(
    session: RelaySession, group: HarborEnvGroupBuilder, harbor_server_url: str
) -> None:
    from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app
    from tinker_cookbook.recipes.harbor_rl.connector import Connector

    renderer = get_renderer(
        group.renderer_name or model_info.get_recommended_renderer_name(group.model_name),
        get_tokenizer(group.model_name),
        model_name=group.model_name,
    )
    app = make_app(ProxyDeps(renderer, session, group.model_name))
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        port = runner.addresses[0][1]
        connector = Connector(
            harbor_server_url,
            f"http://127.0.0.1:{port}",
            session_id=session.session_id,
            api_key=os.getenv("HARBOR_API_KEY"),
        )
        async with asyncio.TaskGroup() as tg:
            connection = tg.create_task(connector.run())
            try:
                await asyncio.wait_for(connector.ready.wait(), 30)
                session.ready.set()
                await session.finished.wait()
            finally:
                connection.cancel()
    finally:
        await runner.cleanup()


async def _run_job(
    client: AsyncHarbor, session_id: str, group: HarborEnvGroupBuilder
) -> AsyncIterator[RolloutSummary]:
    job = await client.jobs.submit(
        {
            "tasks": [{"path": str(group.task.task_dir.resolve())}],
            "agents": [
                {
                    "name": "claude-code",
                    "model_name": group.model_name,
                    "skip_model_parsing": True,
                    "kwargs": {"max_turns": group.max_turns},
                }
            ],
            "environment": {"type": "daytona", "delete": True},
            "n_attempts": group.group_size,
            "n_concurrent_trials": group.group_size,
            "retry": {"max_retries": 0},
        },
        relay_inference=True,
        session_id=session_id,
    )
    logger.info("Harbor job %s using relay session %s", job.job_id, session_id)
    try:
        while (await client.jobs.get(job.job_id)).state not in {"finished", "cancelled"}:
            await asyncio.sleep(1)
    except BaseException:
        await client.jobs.cancel(job.job_id)
        raise
    token = ""
    while True:
        page = await client.rollouts.list(job_ids=[job.job_id], page_token=token)
        for rollout in page.items:
            yield rollout
        if not page.next_page_token:
            return
        token = page.next_page_token
