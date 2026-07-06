"""On-policy distillation: transfer a stronger open teacher's skill into a smaller student.

Unlike off-policy SFT on teacher-generated text (which suffers exposure bias when the
teacher writes in its own style), on-policy distillation trains the student on the *student's
own* rollouts and uses the teacher only to score them token by token:

    per-token advantage  a_t = clip( logπ_teacher(y_t | y_<t) - logπ_student(y_t | y_<t) )

Fed through the same ``importance_sampling`` update the GRPO trainer uses, this is a
per-token reverse-KL policy gradient: push the student up on tokens the teacher likes more
than the student currently does, down on tokens it likes less. When the student already
matches the teacher, a_t ~ 0 and there is no update.

Reuses the trainer's proxy (student sampling + per-token capture), the app rollout driver,
and the optim step. The teacher only needs ``SamplingClient.compute_logprobs`` on the
student's ``prompt+sampled`` tokens (a forward pass, no gradient, no generation). Student and
teacher must share a tokenizer (true within a model family, e.g. Qwen3.5-9B and -397B).

    python -m tinker_cookbook.recipes.cog_rl.training.distill \\
        student_model=Qwen/Qwen3.5-9B \\
        teacher_checkpoint=tinker://<397b-t>:train:0/sampler_weights/final \\
        teacher_base_model=Qwen/Qwen3.5-397B-A17B \\
        init_state_path=tinker://<9b-sft>:train:0/weights/9b-sft \\
        project_id=<id> run_label=9b-distill num_batches=80
"""

from __future__ import annotations

import asyncio
import logging
import time

import chz
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.cog_rl.training.grading import shaped_reward
from tinker_cookbook.recipes.cog_rl.training.proxy import SamplingProxy, TurnCapture
from tinker_cookbook.recipes.cog_rl.training.tasks import get_tasks
from tinker_cookbook.recipes.cog_rl.training.train import (
    _batch_tasks,
    _maybe_log_rollouts,
    _run_batch_rollouts,
    _start_proxy,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    # shared with train.Config so the reused helpers can read it
    base_url: str | None = None
    project_id: str | None = None
    run_label: str | None = None
    model_name: str = "Qwen/Qwen3.5-9B"  # the STUDENT (alias: student_model)
    lora_rank: int = 32
    learning_rate: float = 4e-5
    group_size: int = 2
    groups_per_batch: int = 32
    max_turns: int = 3
    max_tokens: int = 2048
    temperature: float = 1.0
    num_batches: int | None = 80
    seed: int = 0
    task_source: str = "corpus"
    init_state_path: str | None = None  # warm-start the student (e.g. an SFT checkpoint)
    app_url: str = "http://127.0.0.1:8000"
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8100
    rollout_timeout_s: float = 600.0
    log_path: str = "/tmp/dylan/cog_distill"
    log_rollouts_every: int = 40
    save_every: int = 20
    ttl_seconds: int | None = None
    # distillation-specific
    teacher_checkpoint: str = ""  # tinker://.../sampler_weights/... of the trained teacher
    teacher_base_model: str = "Qwen/Qwen3.5-397B-A17B"
    adv_clip: float = 4.0  # clip per-token (teacher-student) logprob delta to [-c, c]
    advantage_norm: str = "std"  # "std" = divide per-token deltas by batch std; "none" = raw
    teacher_concurrency: int = 16


async def _teacher_logprobs(
    teacher: tinker.SamplingClient, caps_by_rid: dict[str, list[TurnCapture]], concurrency: int
) -> dict[str, list[list[float]]]:
    """Teacher per-sampled-token logprobs for every captured turn.

    For a turn with ``prompt_tokens`` (len P) and ``sampled_tokens`` (len S), compute_logprobs
    on ``prompt+sampled`` returns per-position logprobs; indices P..P+S-1 are the teacher's
    logprobs of the student's sampled tokens. Returns {rid: [per-turn [len-S lp list]]}.
    """
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, list[list[float]]] = {rid: [] for rid in caps_by_rid}

    async def one(rid: str, idx: int, cap: TurnCapture):
        P, S = len(cap.prompt_tokens), len(cap.sampled_tokens)
        if S < 1:
            return rid, idx, []
        full = types.ModelInput.from_ints(cap.prompt_tokens + cap.sampled_tokens)
        async with sem:
            lps = await teacher.compute_logprobs_async(full)
        seg = lps[P : P + S]
        # None where the teacher couldn't score a token -> fall back to the student's own
        # logprob so that token contributes zero advantage.
        seg = [seg[j] if seg[j] is not None else cap.logprobs[j] for j in range(S)]
        return rid, idx, seg

    tasks = [one(rid, i, c) for rid, caps in caps_by_rid.items() for i, c in enumerate(caps)]
    for rid in caps_by_rid:
        out[rid] = [[] for _ in caps_by_rid[rid]]
    for coro in asyncio.as_completed(tasks):
        rid, idx, seg = await coro
        out[rid][idx] = seg
    return out


def _batch_adv_scale(
    caps_by_rid: dict[str, list[TurnCapture]], tlps: dict[str, list[list[float]]], norm: str
) -> float:
    """A single per-batch scale for the per-token (teacher-student) deltas.

    OPD deltas are intrinsically small (avg ~0.1-0.2, i.e. the fwd-KL), much smaller than
    GRPO's group-centered reward advantages (~0.5), so at a fixed LR OPD under-steps. Dividing
    by the batch std (keeping sign/mean) puts the advantage on a unit scale so the learning
    rate alone controls the step size, decoupled from how close the student already is.
    """
    if norm != "std":
        return 1.0
    deltas: list[float] = []
    for rid, caps in caps_by_rid.items():
        for cap, tlp in zip(caps, tlps.get(rid, [])):
            S = len(cap.sampled_tokens)
            if S < 1 or len(tlp) != S:
                continue
            deltas.extend(tlp[j] - cap.logprobs[j] for j in range(S))
    if len(deltas) < 2:
        return 1.0
    mean = sum(deltas) / len(deltas)
    var = sum((d - mean) ** 2 for d in deltas) / len(deltas)
    std = var**0.5
    return 1.0 / (std + 1e-6)


def _build_distill_datums(
    caps: list[TurnCapture], teacher_lps: list[list[float]], adv_clip: float, scale: float = 1.0
) -> tuple[list[types.Datum], float]:
    """One datum per turn; per-token advantage = clip(scale * (teacher_lp - student_lp))."""
    datums: list[types.Datum] = []
    kl_sum, kl_n = 0.0, 0
    for cap, tlp in zip(caps, teacher_lps):
        S = len(cap.sampled_tokens)
        if S < 1 or len(tlp) != S:
            continue
        prompt = types.ModelInput.from_ints(cap.prompt_tokens)
        ob_len = prompt.length - 1
        model_input = prompt.append(types.EncodedTextChunk(tokens=cap.sampled_tokens[:-1]))
        adv = [max(-adv_clip, min(adv_clip, scale * (tlp[j] - cap.logprobs[j]))) for j in range(S)]
        kl_sum += sum(cap.logprobs[j] - tlp[j] for j in range(S))  # student-teacher = fwd KL est
        kl_n += S
        target_tokens = [0] * ob_len + cap.sampled_tokens
        padded_logprobs = [0.0] * ob_len + cap.logprobs
        padded_adv = [0.0] * ob_len + adv
        assert model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_adv)
        datums.append(
            types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(padded_adv)),
                },
            )
        )
    mean_kl = kl_sum / max(kl_n, 1)
    return datums, mean_kl


def main(config: Config) -> None:
    logging.basicConfig(level=logging.INFO)
    if not config.teacher_checkpoint:
        raise SystemExit("teacher_checkpoint is required")

    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(config.model_name), tokenizer
    )
    proxy = SamplingProxy(renderer, default_max_tokens=config.max_tokens)
    _start_proxy(proxy, config.proxy_host, config.proxy_port)
    proxy_base = f"http://{config.proxy_host}:{config.proxy_port}"

    train_tasks, _ = get_tasks(config.task_source, seed=config.seed)
    n_batches = config.num_batches or (len(train_tasks) // config.groups_per_batch)

    service = tinker.ServiceClient(base_url=config.base_url, project_id=config.project_id)
    run_meta: dict[str, str] = {}
    if config.project_id:
        from tinker_cookbook.recipes.cog_rl.training import checkpoints as ck

        run_meta = {ck.PROJECT_TAG: config.project_id, ck.LABEL_TAG: config.run_label or "distill"}
    if config.init_state_path:
        logger.info("student warm-start from %s", config.init_state_path)
        student = service.create_training_client_from_state(
            config.init_state_path, user_metadata=run_meta or None
        )
    else:
        student = service.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank, user_metadata=run_meta or None
        )
    adam = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    teacher = service.create_sampling_client(
        model_path=config.teacher_checkpoint, base_model=config.teacher_base_model
    )
    logger.info("teacher: %s", config.teacher_checkpoint)

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )
    t0 = time.time()

    async def _loop():
        for batch_idx in range(n_batches):
            proxy.set_policy(
                student.save_weights_and_get_sampling_client(), temperature=config.temperature
            )
            batch_tasks = _batch_tasks(config, train_tasks, batch_idx)
            results = await _run_batch_rollouts(config, batch_tasks, batch_idx, proxy_base)
            caps = proxy.pop_captures()
            tlps = await _teacher_logprobs(teacher, caps, config.teacher_concurrency)
            scale = _batch_adv_scale(caps, tlps, config.advantage_norm)

            datums: list[types.Datum] = []
            mean_kls: list[float] = []
            correct = 0
            rollout_log: list[dict] = []
            for ti, task in enumerate(batch_tasks):
                for g in range(config.group_size):
                    rid = f"{batch_idx}-{ti}-{g}"
                    program, transcript = results.get(rid, ("", []))
                    _, info = shaped_reward(program, task)
                    correct += int(info["correct"])
                    rollout_log.append(
                        {
                            "family": task.family,
                            "name": f"{task.name}#g{g}",
                            "prompt": task.prompt,
                            "expect": "; ".join(f"solve({a})={e}" for a, e in task.tests),
                            "program": program,
                            "correct": bool(info["correct"]),
                            "reward": 0.0,
                            "transcript": transcript,
                        }
                    )
                    d, mk = _build_distill_datums(
                        caps.get(rid, []), tlps.get(rid, []), config.adv_clip, scale
                    )
                    datums.extend(d)
                    if d:
                        mean_kls.append(mk)

            n_roll = config.group_size * len(batch_tasks)
            metrics = {
                "distill/fwd_kl": sum(mean_kls) / max(len(mean_kls), 1),
                "distill/adv_scale": scale,
                "eval/pass@1": correct / max(n_roll, 1),
                "data/num_datums": float(len(datums)),
                "data/num_rollouts": float(n_roll),
            }
            if datums:
                fwd = student.forward_backward(datums, loss_fn="importance_sampling")
                opt = student.optim_step(adam)
                fwd.result()
                opt.result()
            metrics.update({"progress/batch": batch_idx, "time/total": time.time() - t0})
            ml_logger.log_metrics(metrics, step=batch_idx)
            logger.info(
                "batch %d: pass@1=%.3f fwd_kl=%.3f datums=%d",
                batch_idx,
                metrics["eval/pass@1"],
                metrics["distill/fwd_kl"],
                len(datums),
            )
            _maybe_log_rollouts(config, batch_idx, rollout_log)
            if config.save_every and batch_idx > 0 and batch_idx % config.save_every == 0:
                student.save_weights_for_sampler(f"{batch_idx:06d}").result()

        path = student.save_weights_for_sampler(config.run_label or "final").result().path
        # Also save a trainable state so a further stage (e.g. GRPO) can warm-start from it.
        state_path = student.save_state(config.run_label or "final").result().path
        logger.info("distill done; sampler: %s state: %s", path, state_path)
        print(f"DISTILL_DONE {path}", flush=True)
        print(f"DISTILL_STATE {state_path}", flush=True)

    asyncio.run(_loop())
    ml_logger.close()


if __name__ == "__main__":
    chz.nested_entrypoint(main)
