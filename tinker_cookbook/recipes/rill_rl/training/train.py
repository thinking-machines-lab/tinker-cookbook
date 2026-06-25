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
from tinker_cookbook.recipes.rill_rl.training.grading import RillTask, shaped_reward
from tinker_cookbook.recipes.rill_rl.training.proxy import SamplingProxy, TurnCapture
from tinker_cookbook.recipes.rill_rl.training.tasks import build_tasks
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


def main(config: Config) -> None:
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

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

    logger.info("Training for %d batches of %d tasks", n_batches, config.groups_per_batch)
    for batch_idx in range(n_batches):
        t0 = time.time()

        if config.save_every > 0 and batch_idx > 0 and batch_idx % config.save_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        # 1. point the proxy at the current policy.
        sampling_client = training_client.save_weights_and_get_sampling_client()
        proxy.set_policy(sampling_client, temperature=config.temperature)
        proxy.reset_captures()

        # 2. trigger rollouts through the app.
        start = batch_idx * config.groups_per_batch
        batch_tasks = train_tasks[start : start + config.groups_per_batch]
        rollout_results = asyncio.run(
            _run_batch_rollouts(config, batch_tasks, batch_idx, proxy_base)
        )

        # 3. collect captured tokens.
        captures = proxy.pop_captures()

        # 4. grade + GRPO advantages within each task's group; 5. build datums.
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
                        "expect": task.expect,
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

        # Log every rollout this batch so you can watch the model learn the syntax.
        if config.log_rollouts_every > 0 and batch_idx % config.log_rollouts_every == 0:
            rollout_log.sort(key=lambda r: (r["family"], r["name"]))
            html_path = os.path.join(config.log_path, f"rollouts_step{batch_idx:06d}.html")
            log_rollouts_html(html_path, config.model_name, rollout_log)

        # 6. optimizer step.
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "reward/mean": sum(rewards_all) / max(len(rewards_all), 1),
            "reward/pass@1": sum(correct_all) / max(len(correct_all), 1),
            "data/num_datums": len(datums),
            "data/num_rollouts": len(rewards_all),
        }
        if datums:
            fwd = training_client.forward_backward(datums, loss_fn="importance_sampling")
            opt = training_client.optim_step(adam)
            fwd.result()
            opt_result = opt.result()
            if opt_result.metrics:
                metrics.update(opt_result.metrics)
        else:
            logger.warning("batch %d: no datums (all groups zero-advantage)", batch_idx)

        metrics["time/total"] = time.time() - t0
        ml_logger.log_metrics(metrics, step=batch_idx)
        logger.info(
            "batch %d: pass@1=%.3f mean_reward=%.3f datums=%d (%.1fs)",
            batch_idx,
            metrics["reward/pass@1"],
            metrics["reward/mean"],
            len(datums),
            metrics["time/total"],
        )

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
