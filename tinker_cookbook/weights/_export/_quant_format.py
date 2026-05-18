"""Quantization format abstractions for shard-based export.

Defines two protocols that plug into :func:`run_shard_merge`:

- :class:`ShardHooks` — for models with pre-quantized weights on disk that
  need dequant → merge → requant as an atomic per-key operation (e.g. INT4
  packed format for Kimi K2/K2.6).

- :class:`QuantizationFormat` — for post-merge quantization where the output
  model needs selected weights quantized after LoRA merge (e.g. FP8 blockwise
  for DeepSeek V3).

Concrete implementations live in format-specific modules:
``_quantized.py`` (FP8), ``_shard_packed_int4.py`` (INT4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from tinker_cookbook.weights._merge import MergeOp


@runtime_checkable
class ShardHooks(Protocol):
    """Hooks for pre-quantized weights: dequant → merge → requant.

    Used when the base model checkpoint stores weights in a compressed
    on-disk format (e.g. INT4 packed) that differs from the key names
    the merge planner expects. The hooks bridge this gap by:

    1. Creating virtual key names for merge planning
       (:meth:`augment_for_planning`).
    2. Intercepting shard processing to dequantize, apply merge ops,
       and re-quantize in place (:meth:`try_apply`).
    """

    def augment_for_planning(
        self,
        model_state_keys: set[str],
        model_shapes: dict[str, tuple[int, ...]],
    ) -> tuple[set[str], dict[str, tuple[int, ...]]]:
        """Add virtual keys so the merge planner can target packed weights.

        Returns:
            Tuple of ``(augmented_keys, augmented_shapes)``.
        """
        ...

    def try_apply(
        self,
        key: str,
        tensors: dict[str, torch.Tensor],
        merge_ops: dict[str, list[MergeOp]],
    ) -> int:
        """Try to handle a key with dequant → merge → requant.

        Called for every tensor key in a shard. If ``key`` is a packed
        weight with pending merge ops, applies them atomically and
        modifies ``tensors`` in-place.

        Returns:
            Number of merge ops applied (0 if this key wasn't handled).
        """
        ...


@runtime_checkable
class QuantizationFormat(Protocol):
    """Post-merge quantization format.

    Used when the output model needs quantization applied after LoRA merge
    (e.g. FP8 for DeepSeek V3). Plugs into the shard engine as a series
    of per-tensor transforms applied around the merge step.
    """

    def filter_model_keys(self, keys: set[str]) -> set[str]:
        """Filter model state keys before merge planning.

        Exclude keys that shouldn't be merge targets (e.g. native
        scale tensors, placeholder layers).
        """
        ...

    def should_skip_output_key(self, key: str) -> bool:
        """Whether to exclude this key from the output.

        Use to skip keys that exist in the input checkpoint but
        shouldn't appear in the output (e.g. native ``weight_scale_inv``
        tensors that are replaced by new ``weight_scale`` tensors).
        """
        ...

    def pre_merge_transform(
        self,
        key: str,
        tensor: torch.Tensor,
        shard_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Transform a tensor BEFORE merge ops are applied.

        Use to dequantize native quantized weights to a merge-safe dtype.
        ``shard_tensors`` provides access to sibling tensors in the same
        shard (e.g. for looking up scale tensors).

        Return the (possibly transformed) tensor.
        """
        ...

    def post_merge_transform(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Transform a tensor AFTER merge ops are applied.

        Returns a dict of ``key → tensor`` pairs for the output. For
        quantization this typically produces the original key (quantized)
        plus a scale key. For pass-through, returns ``{key: tensor}``.
        """
        ...

    def finalize_config(
        self,
        config_dict: dict,
        weight_map: dict[str, str],
    ) -> dict:
        """Patch config.json with format-specific metadata.

        Called after all shards are processed. Return the modified config.
        """
        ...
