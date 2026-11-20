"""Convert Tinker LoRA checkpoints into PEFT adapters that SGLang can serve.

    python -m tinker_cookbook.inference.sglang.prepare \
        --base-model Qwen/Qwen3.8-27B \
        --adapter lora1=tinker://<run-id>/sampler_weights/final \
        --adapter lora2=tinker://<run-id>/sampler_weights/final

Downloads each checkpoint from the Tinker API, converts it to PEFT format under
``--out``, and prints the ``--lora-paths`` value for the launch command.

The conversion reads the base model's parameter *names and shapes* (never its
values) to remap Tinker's internal adapter keys, so the base model is fetched
from the Hugging Face cache. Point ``HF_HUB_CACHE`` at the same cache the SGLang
container mounts and the download is paid once instead of twice; or pass a local
directory as ``--base-model`` to skip the Hub entirely.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from tinker_cookbook import weights
from tinker_cookbook.inference.sglang.common import (
    DEFAULT_LORA_ROOT,
    adapter_dir,
    lora_paths_arg,
    parse_adapter_spec,
    read_adapter_config,
)

# LoRA keys the converter leaves pointing at a parameter that does not exist:
# GLM-5.3 adapters say ``model.lm_head``, which matches no remap, and SGLang
# then reserves an lm_head slot it silently never fills. Workaround; the real
# fix belongs next to the existing remap in
# ``weights/_merge_utils.build_name_remaps``, where it would also reach vLLM.
_KEY_FIXUPS: tuple[tuple[str, str], ...] = (
    ("base_model.model.model.lm_head.", "base_model.model.lm_head."),
)


def _apply_key_fixups(adapter_path: str) -> int:
    """Rewrite known-bad LoRA keys in a converted adapter. Returns the count.

    Only rewrites when the corrected key is absent, so an adapter that already
    carries the right name is left alone.
    """
    from safetensors.torch import load_file, save_file

    weights_file = Path(adapter_path) / "adapter_model.safetensors"
    tensors = load_file(str(weights_file))
    renamed = 0
    for bad, good in _KEY_FIXUPS:
        if any(k.startswith(good) for k in tensors):
            continue
        for key in [k for k in tensors if k.startswith(bad)]:
            tensors[good + key[len(bad) :]] = tensors.pop(key)
            renamed += 1
    if renamed:
        save_file(tensors, str(weights_file))
    return renamed


_EXPERT_KEY = re.compile(r"^(?P<pre>.*\.experts)\.(?P<idx>\d+)\.(?P<rest>.+\.lora_[AB])\.weight$")


def _restore_shared_outer_experts(adapter_path: str) -> int:
    """Undo the copying of a shared-outer expert factor, one copy per expert.

    ``build_lora_adapter`` expands Tinker's 3D expert LoRA into one 2D tensor
    per expert; for a shared-outer adapter (Inkling: ``expert_dim=1``) that
    turns one tensor into N identical copies, which SGLang refuses as a mixed
    format. Stack them back and collapse the expert dimension to 1 wherever
    every slice is identical — lossless, they were exact copies. Returns the
    number of expert groups rebuilt.
    """
    import torch
    from safetensors.torch import load_file, save_file

    weights_file = Path(adapter_path) / "adapter_model.safetensors"
    tensors = load_file(str(weights_file))

    groups: dict[str, dict[int, str]] = {}
    for key in tensors:
        if (m := _EXPERT_KEY.match(key)) is None:
            continue
        target = f"{m.group('pre')}.{m.group('rest')}.weight"
        groups.setdefault(target, {})[int(m.group("idx"))] = key
    if not groups:
        return 0

    for target, members in groups.items():
        stacked = torch.stack([tensors[members[i]] for i in sorted(members)])
        if all(torch.equal(stacked[0], stacked[i]) for i in range(1, stacked.shape[0])):
            stacked = stacked[:1]
        for key in members.values():
            del tensors[key]
        tensors[target] = stacked.contiguous()

    save_file(tensors, str(weights_file))
    return len(groups)


def _rewrite_target_modules(adapter_path: str) -> list[str]:
    """Re-derive target_modules from the weight keys after the rewrites above."""
    from safetensors.torch import load_file

    tensors = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
    config_file = Path(adapter_path) / "adapter_config.json"
    config = json.loads(config_file.read_text())
    modules = sorted(
        {k.rsplit(".lora_", 1)[0].rsplit(".", 1)[-1] for k in tensors if ".lora_" in k}
    )
    config["target_modules"] = modules
    config_file.write_text(json.dumps(config, indent=2))
    return modules


# Module names as Tinker writes them, mapped to the names SGLang resolves.
# Only the declared target name changes -- never the weights: SGLang's loader
# reconciles weight keys itself (concatenating per-slice attention, renaming
# w1/w3/w2). Rewriting attention weights to the fused name instead would hit
# the merged-adapter path, which cannot hold four distinct lora_A factors.
_SERVING_TARGET_NAMES: dict[str, str] = {
    "in_proj_q": "in_proj_qkvz",
    "in_proj_k": "in_proj_qkvz",
    "in_proj_v": "in_proj_qkvz",
    "in_proj_z": "in_proj_qkvz",
    "wq_du": "qkvr",
    "wk_dv": "qkvr",
    "wv_dv": "qkvr",
    "wr_du": "qkvr",
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}


def _declare_serving_targets(adapter_path: str) -> int:
    """Put the names SGLang resolves into target_modules, leaving weights alone."""
    config_file = Path(adapter_path) / "adapter_config.json"
    config = json.loads(config_file.read_text())
    modules: list[str] = list(config.get("target_modules") or [])
    mapped = sorted({_SERVING_TARGET_NAMES.get(m, m) for m in modules})
    if mapped == sorted(modules):
        return 0
    config["target_modules"] = mapped
    config_file.write_text(json.dumps(config, indent=2))
    return len({m for m in modules if m in _SERVING_TARGET_NAMES})


def prepare_adapter(
    *, base_model: str, tinker_path: str, name: str, out: str, overwrite: bool = False
) -> str:
    """Download one Tinker checkpoint and convert it to a PEFT adapter.

    Returns the local directory holding the converted adapter.
    """
    output_path = adapter_dir(out, base_model, name)
    if overwrite:
        shutil.rmtree(output_path, ignore_errors=True)
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = weights.download(tinker_path=tinker_path, output_dir=tmp)
        weights.build_lora_adapter(
            base_model=base_model, adapter_path=downloaded, output_path=output_path
        )
    if renamed := _apply_key_fixups(output_path):
        print(f"[prepare] {name}: repaired {renamed} lm_head key(s) (see _KEY_FIXUPS)")
    changed = _restore_shared_outer_experts(output_path)
    if changed:
        modules = _rewrite_target_modules(output_path)
        print(f"[prepare] {name}: rebuilt expert tensors -> target_modules={','.join(modules)}")
    # Last: _rewrite_target_modules re-derives names from the weight keys, which
    # are deliberately left as Tinker wrote them, so aliasing must follow it.
    if aliased := _declare_serving_targets(output_path):
        print(f"[prepare] {name}: declared {aliased} module(s) under the name SGLang resolves")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base-model", required=True, help="HF model name or local directory")
    parser.add_argument(
        "--adapter",
        action="append",
        required=True,
        metavar="NAME=tinker://...",
        help="Adapter to convert; repeat once per adapter",
    )
    parser.add_argument("--out", default="./adapters", help="Host directory for the adapter tree")
    parser.add_argument(
        "--lora-root",
        default=DEFAULT_LORA_ROOT,
        help="Where --out is mounted inside the container (used for the printed --lora-paths)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace adapters that exist")
    args = parser.parse_args()

    adapters = [parse_adapter_spec(spec) for spec in args.adapter]
    summaries: list[tuple[str, int, list[str]]] = []
    for name, tinker_path in adapters:
        print(f"[prepare] {name}: {tinker_path}")
        path = prepare_adapter(
            base_model=args.base_model,
            tinker_path=tinker_path,
            name=name,
            out=args.out,
            overwrite=args.overwrite,
        )
        rank, target_modules = read_adapter_config(path)
        summaries.append((name, rank, target_modules))
        print(f"[prepare] {name}: r={rank} target_modules={','.join(target_modules)}")

    module_sets = {tuple(modules) for _, _, modules in summaries}
    if len(module_sets) > 1:
        print(
            "\n[prepare] note: adapters do not target the same modules. SGLang applies the "
            "union, so an adapter may be served with modules it was not trained on."
        )

    names = [name for name, _, _ in summaries]
    print(f"\nAdapters written to {Path(args.out).resolve()}")
    print(f"\nMount:\n  -v {Path(args.out).resolve()}:{args.lora_root}:ro")
    print(
        f"\nLaunch with:\n  --lora-paths {lora_paths_arg(names, args.base_model, args.lora_root)}"
    )


if __name__ == "__main__":
    main()
