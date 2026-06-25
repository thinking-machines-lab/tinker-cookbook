"""GRPO post-training for the RILL agent app, via the sampling proxy.

The trainer never imports the agent loop. Each step it:

1. saves the current policy weights and points the embedded sampling proxy at them;
2. triggers ``group_size`` rollouts per task by POSTing to the production app's ``/solve``
   endpoint, each pointed at a unique proxy URL (``/v1/<rollout_id>``);
3. pulls the tokens the proxy captured for those rollouts;
4. grades each rollout's final program against the held-out expected output (GRPO centers
   the reward within each task's group);
5. builds training datums from the captured tokens (one per turn, weighted by the
   rollout's advantage) and runs ``forward_backward`` + ``optim_step``.

The agent app must be running separately (it's the production service). Point ``app_url``
at it.

Example::

    # terminal 1: the production app
    python -m tinker_cookbook.recipes.rill_rl.agent_app.server
    # terminal 2: train against it
    python -m tinker_cookbook.recipes.rill_rl.training.train \\
        model_name=Qwen/Qwen3.5-4B app_url=http://127.0.0.1:8000 \\
        group_size=8 groups_per_batch=32 max_turns=3
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import chz
import httpx
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.rill_rl.training.eval import log_rollouts_html
from tinker_cookbook.recipes.rill_rl.training.grading import shaped_reward
from tinker_cookbook.recipes.rill_rl.training.proxy import SamplingProxy, TurnCapture
from tinker_cookbook.recipes.rill_rl.training.tasks import RillTask, build_tasks
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.git_rev import recipe_user_metadata

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None  # Tinker service base url
    model_name: str = "Qwen/Qwen3.5-4B"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    group_size: int = 8
    groups_per_batch: int = 32
    max_turns: int = 3
    max_tokens: int = 2048
    temperature: float = 1.0
    num_batches: int | None = None  # None = one pass over the training tasks
    seed: int = 0

    # Async off-policy RL: let the rollout producer run up to this many batches ahead of
    # the trainer (overlapping sampling with the optimizer step). 0/None = on-policy and
    # sequential. importance_sampling on the captured logprobs corrects for the staleness.
    max_steps_off_policy: int | None = None

    # The production agent app (already running) and the embedded sampling proxy.
    app_url: str = "http://127.0.0.1:8000"
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8100
    rollout_timeout_s: float = 600.0

    log_path: str = "/tmp/tinker-examples/rill_rl"
    save_every: int = 20
    # Write a logtree HTML of every rollout each Nth batch (0 = off). Watch the model
    # learn the OOD syntax: {log_path}/rollouts_step{N}.html.
    log_rollouts_every: int = 1
    ttl_seconds: int | None = 604800


def _start_proxy(proxy: SamplingProxy, host: str, port: int) -> None:
    """Run the proxy under uvicorn in a background thread and wait until it serves."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(proxy.app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if getattr(server, "started", False):
            return
        time.sleep(0.1)
    logger.warning("proxy may not have started within timeout")


async def _solve_via_app(
    http: httpx.AsyncClient,
    app_url: str,
    *,
    prompt: str,
    rollout_id: str,
    proxy_base: str,
    model: str,
    max_turns: int,
) -> tuple[str, list[dict]]:
    """Trigger one rollout through the app's HTTP endpoint; return (final_program, transcript)."""
    resp = await http.post(
        f"{app_url}/solve",
        json={
            "prompt": prompt,
            "model": model,
            "max_turns": max_turns,
            # Route this rollout's sampling through the proxy under a unique id.
            "openai_base_url": f"{proxy_base}/v1/{rollout_id}",
            "openai_api_key": "rl",
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("program", ""), data.get("transcript", [])


def _build_datums(turns: list[TurnCapture], advantage: float) -> list[types.Datum]:
    """One datum per sampled turn, all weighted by the rollout's advantage (rl_loop.py shape)."""
    datums: list[types.Datum] = []
    for cap in turns:
        if len(cap.sampled_tokens) < 1:
            continue
        prompt = types.ModelInput.from_ints(cap.prompt_tokens)
        ob_len = prompt.length - 1
        model_input = prompt.append(types.EncodedTextChunk(tokens=cap.sampled_tokens[:-1]))
        target_tokens = [0] * ob_len + cap.sampled_tokens
        padded_logprobs = [0.0] * ob_len + cap.logprobs
        padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
        assert (
            model_input.length
            == len(target_tokens)
            == len(padded_logprobs)
            == len(padded_advantages)
        )
        datums.append(
            types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                },
            )
        )
    return datums


async def _run_batch_rollouts(
    cfg: Config, tasks: list[RillTask], batch_idx: int, proxy_base: str
) -> dict[str, tuple[str, list[dict]]]:
    """Fire group_size rollouts per task at the app; return {rollout_id: (program, transcript)}."""
    results: dict[str, tuple[str, list[dict]]] = {}
    limits = httpx.Limits(max_connections=cfg.group_size * len(tasks))
    async with httpx.AsyncClient(timeout=cfg.rollout_timeout_s, limits=limits) as http:

        async def one(ti: int, g: int):
            rid = f"{batch_idx}-{ti}-{g}"
            try:
                results[rid] = await _solve_via_app(
                    http,
                    cfg.app_url,
                    prompt=tasks[ti].prompt,
                    rollout_id=rid,
                    proxy_base=proxy_base,
                    model=cfg.model_name,
                    max_turns=cfg.max_turns,
                )
            except Exception as e:
                logger.warning("rollout %s failed: %r", rid, e)
                results[rid] = ("", [])

        await asyncio.gather(
            *(one(ti, g) for ti in range(len(tasks)) for g in range(cfg.group_size))
        )
    return results


def _batch_tasks(config: Config, train_tasks: list[RillTask], batch_idx: int) -> list[RillTask]:
    # Cycle through the task pool so a modest set of distinct tasks supports many batches.
    return [
        train_tasks[(batch_idx * config.groups_per_batch + j) % len(train_tasks)]
        for j in range(config.groups_per_batch)
    ]


def _grade_and_build(
    config: Config,
    batch_tasks: list[RillTask],
    batch_idx: int,
    rollout_results: dict[str, tuple[str, list[dict]]],
    captures: dict[str, list[TurnCapture]],
) -> tuple[list[types.Datum], dict[str, float], list[dict]]:
    """Grade the batch, center advantages within each group, and build training datums."""
    datums: list[types.Datum] = []
    rewards_all: list[float] = []
    correct_all: list[float] = []
    rollout_log: list[dict] = []
    for ti, task in enumerate(batch_tasks):
        rewards_g: list[float] = []
        infos_g: list[dict] = []
        rids: list[str] = []
        for g in range(config.group_size):
            rid = f"{batch_idx}-{ti}-{g}"
            program, transcript = rollout_results.get(rid, ("", []))
            reward, info = shaped_reward(program, task)
            rewards_g.append(reward)
            infos_g.append(info)
            rids.append(rid)
            rollout_log.append(
                {
                    "family": task.family,
                    "name": f"{task.name}#g{g}",
                    "prompt": task.prompt,
                    "expect": "; ".join(f"solve({a})={e}" for a, e in task.tests),
                    "program": program,
                    "correct": bool(info["correct"]),
                    "reward": reward,
                    "transcript": transcript,
                }
            )
        mean_r = sum(rewards_g) / len(rewards_g)
        rewards_all.extend(rewards_g)
        correct_all.extend(float(i["correct"]) for i in infos_g)
        if all(r == mean_r for r in rewards_g):
            continue  # zero advantage everywhere; nothing to learn from this group
        for g, rid in enumerate(rids):
            datums.extend(_build_datums(captures.get(rid, []), rewards_g[g] - mean_r))

    metrics = {
        "reward/mean": sum(rewards_all) / max(len(rewards_all), 1),
        "reward/pass@1": sum(correct_all) / max(len(correct_all), 1),
        "data/num_datums": float(len(datums)),
        "data/num_rollouts": float(len(rewards_all)),
    }
    return datums, metrics, rollout_log


def _maybe_log_rollouts(config: Config, batch_idx: int, rollout_log: list[dict]) -> None:
    if config.log_rollouts_every > 0 and batch_idx % config.log_rollouts_every == 0:
        rollout_log.sort(key=lambda r: (r["family"], r["name"]))
        path = os.path.join(config.log_path, f"rollouts_step{batch_idx:06d}.html")
        log_rollouts_html(path, config.model_name, rollout_log)


def _train_step(training_client, datums: list[types.Datum], adam) -> dict[str, float]:
    """Blocking forward_backward + optim_step (run via asyncio.to_thread in the async loop)."""
    fwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
    opt = training_client.optim_step(adam)
    fwd.result()
    res = opt.result()
    return dict(res.metrics) if res.metrics else {}


def _publish_policy(training_client, proxy: SamplingProxy, temperature: float) -> None:
    """Save current weights and point the proxy at them (blocking)."""
    proxy.set_policy(
        training_client.save_weights_and_get_sampling_client(), temperature=temperature
    )


def _save_state(config: Config, training_client, batch_idx: int) -> None:
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name=f"{batch_idx:06d}",
        log_path=config.log_path,
        kind="state",
        loop_state={"batch": batch_idx},
        ttl_seconds=config.ttl_seconds,
    )


def _setup(config: Config):
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)

    proxy = SamplingProxy(renderer, default_max_tokens=config.max_tokens)
    _start_proxy(proxy, config.proxy_host, config.proxy_port)
    proxy_base = f"http://{config.proxy_host}:{config.proxy_port}"
    logger.info("Sampling proxy serving at %s; app at %s", proxy_base, config.app_url)

    train_tasks, _ = build_tasks(seed=config.seed)
    n_batches = config.num_batches or (len(train_tasks) // config.groups_per_batch)

    service = tinker.ServiceClient(
        base_url=config.base_url, user_metadata=recipe_user_metadata("recipe_rill_rl")
    )
    training_client = service.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )
    adam = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    return proxy, proxy_base, train_tasks, n_batches, training_client, adam


def _log_batch(ml_logger, batch_idx, metrics, datums, lag, t0):
    metrics = {**metrics, "progress/batch": batch_idx, "time/total": time.time() - t0}
    if lag is not None:
        metrics["off_policy/lag_batches"] = float(lag)
    ml_logger.log_metrics(metrics, step=batch_idx)
    logger.info(
        "batch %d: pass@1=%.3f mean_reward=%.3f datums=%d%s (%.1fs)",
        batch_idx,
        metrics["reward/pass@1"],
        metrics["reward/mean"],
        len(datums),
        f" lag={lag}" if lag is not None else "",
        metrics["time/total"],
    )


def _run_sync(config, proxy, proxy_base, train_tasks, n_batches, training_client, adam, ml_logger):
    """On-policy: sample a full batch, train on it, repeat (no overlap)."""
    for batch_idx in range(n_batches):
        t0 = time.time()
        if config.save_every > 0 and batch_idx > 0 and batch_idx % config.save_every == 0:
            _save_state(config, training_client, batch_idx)

        _publish_policy(training_client, proxy, config.temperature)
        proxy.reset_captures()
        batch_tasks = _batch_tasks(config, train_tasks, batch_idx)
        rollout_results = asyncio.run(_run_batch_rollouts(config, batch_tasks, batch_idx, proxy_base))
        captures = proxy.pop_captures()

        datums, metrics, rollout_log = _grade_and_build(
            config, batch_tasks, batch_idx, rollout_results, captures
        )
        _maybe_log_rollouts(config, batch_idx, rollout_log)
        if datums:
            metrics.update(_train_step(training_client, datums, adam))
        else:
            logger.warning("batch %d: no datums (all groups zero-advantage)", batch_idx)
        _log_batch(ml_logger, batch_idx, metrics, datums, None, t0)


async def _run_async(
    config, proxy, proxy_base, train_tasks, n_batches, training_client, adam, ml_logger
):
    """Off-policy: a producer keeps sampling at the current policy while the trainer steps
    in a thread, bounded to `max_steps_off_policy` batches of lookahead. The captured
    sampling logprobs + importance_sampling loss correct for the staleness."""
    max_off = config.max_steps_off_policy or 0
    lookahead = asyncio.Semaphore(max_off + 1)
    queue: asyncio.Queue = asyncio.Queue()
    counters = {"produced": 0, "trained": 0}

    await asyncio.to_thread(_publish_policy, training_client, proxy, config.temperature)

    async def producer():
        for batch_idx in range(n_batches):
            await lookahead.acquire()  # block if too far ahead of the trainer
            batch_tasks = _batch_tasks(config, train_tasks, batch_idx)
            results = await _run_batch_rollouts(config, batch_tasks, batch_idx, proxy_base)
            ids = [
                f"{batch_idx}-{ti}-{g}"
                for ti in range(len(batch_tasks))
                for g in range(config.group_size)
            ]
            captures = proxy.pop_captures(ids=ids)
            lag = counters["produced"] - counters["trained"]
            await queue.put((batch_idx, batch_tasks, results, captures, lag))
            counters["produced"] += 1
        await queue.put(None)

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            batch_idx, batch_tasks, results, captures, lag = item
            t0 = time.time()
            if config.save_every > 0 and batch_idx > 0 and batch_idx % config.save_every == 0:
                await asyncio.to_thread(_save_state, config, training_client, batch_idx)

            datums, metrics, rollout_log = _grade_and_build(
                config, batch_tasks, batch_idx, results, captures
            )
            _maybe_log_rollouts(config, batch_idx, rollout_log)
            if datums:
                metrics.update(await asyncio.to_thread(_train_step, training_client, datums, adam))
            else:
                logger.warning("batch %d: no datums (all groups zero-advantage)", batch_idx)
            # Publish fresher weights for subsequent rollouts.
            await asyncio.to_thread(_publish_policy, training_client, proxy, config.temperature)
            counters["trained"] += 1
            lookahead.release()
            _log_batch(ml_logger, batch_idx, metrics, datums, lag, t0)

    await asyncio.gather(producer(), consumer())


def main(config: Config) -> None:
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )
    proxy, proxy_base, train_tasks, n_batches, training_client, adam = _setup(config)

    mode = "async off-policy" if (config.max_steps_off_policy or 0) > 0 else "on-policy"
    logger.info(
        "Training %d batches of %d tasks (%s, max_steps_off_policy=%s)",
        n_batches,
        config.groups_per_batch,
        mode,
        config.max_steps_off_policy,
    )
    args = (config, proxy, proxy_base, train_tasks, n_batches, training_client, adam, ml_logger)
    if (config.max_steps_off_policy or 0) > 0:
        asyncio.run(_run_async(*args))
    else:
        _run_sync(*args)

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_batches},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("Training complete")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
