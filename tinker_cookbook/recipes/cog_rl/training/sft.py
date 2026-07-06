"""SFT warm-start on verified gold Cog solutions (Experiment 8: self-distill, then GRPO).

The gold solutions come from ``curate.py`` — programs the *open model itself* produced and the
interpreter verified against the hidden tests (rejection-sampling self-distillation, no external
model). Supervised fine-tuning on them teaches the OOD surface syntax directly, which is what
GRPO otherwise spends many steps merely discovering. After this, run ``train.py`` with
``init_state_path=<this checkpoint>`` to GRPO from the warm start.

    python -m tinker_cookbook.recipes.cog_rl.training.sft \\
        --gold /tmp/dylan/cog_gold_gpt55.jsonl --project <id> --label exp8-sft

Prints the saved training (re-trainable) and sampler (servable) tinker paths.
"""

from __future__ import annotations

import argparse
import json
import random

import tinker
from tinker import types

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.cog_rl.agent_app.prompts import COG_SYSTEM_PROMPT
from tinker_cookbook.recipes.cog_rl.training.checkpoints import LABEL_TAG, PROJECT_TAG
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _datums(gold_path: str, renderer, max_length: int) -> list[types.Datum]:
    datums: list[types.Datum] = []
    with open(gold_path) as f:
        for line in f:
            row = json.loads(line)
            # CoT distillation: if the harvest kept the teacher's full response (reasoning +
            # code), train on it verbatim; otherwise wrap the bare program in a code block.
            if row.get("response"):
                assistant = row["response"].strip()
            else:
                assistant = f"```cog\n{row['program'].strip()}\n```"
            messages = [
                {"role": "system", "content": COG_SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": assistant},
            ]
            # Single assistant message per conversation, so LAST == ALL (and this avoids the
            # sequence-extension warning for renderers without the extension property).
            datums.append(
                conversation_to_datum(
                    messages, renderer, max_length, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
                )
            )
    return datums


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT warm-start on gold Cog solutions.")
    ap.add_argument("--gold", required=True, help="JSONL from curate.py")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--project", default=None)
    ap.add_argument("--label", default="exp8-sft")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(args.model), tokenizer
    )
    datums = _datums(args.gold, renderer, args.max_length)
    print(f"loaded {len(datums)} gold SFT examples", flush=True)

    service = tinker.ServiceClient(project_id=args.project)
    meta = {PROJECT_TAG: args.project or "", LABEL_TAG: args.label, "sft_gold": args.gold}
    tc = service.create_lora_training_client(
        base_model=args.model, rank=args.lora_rank, user_metadata=meta
    )
    adam = types.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    rng = random.Random(args.seed)
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(datums)))
        rng.shuffle(order)
        ep_loss, ep_tok = 0.0, 0.0
        for i in range(0, len(order), args.batch_size):
            batch = [datums[j] for j in order[i : i + args.batch_size]]
            fwd = tc.forward_backward(batch, loss_fn="cross_entropy").result()
            tc.optim_step(adam).result()
            # weighted-mean loss already reduced per-datum; sum logprob*weight for reporting.
            for out, d in zip(fwd.loss_fn_outputs, batch):
                w = d.loss_fn_inputs["weights"].to_numpy()
                lp = out["logprobs"].to_numpy()
                ep_loss += float(-(lp * w).sum())
                ep_tok += float(w.sum())
            step += 1
        print(
            f"epoch {epoch + 1}/{args.epochs}: mean_nll={ep_loss / max(ep_tok, 1):.4f}", flush=True
        )

    training_path = tc.save_state(args.label).result().path
    sampler_path = tc.save_weights_for_sampler(args.label).result().path
    print(f"\nSFT done. label={args.label}")
    print(f"  training: {training_path}")
    print(f"  sampler:  {sampler_path}")


if __name__ == "__main__":
    main()
