"""INT4 group-quantized weight packing and unpacking.

Supports the compressed-tensors ``pack-quantized`` format used by models like
Kimi K2.5.  In this format, eight 4-bit signed integers (range [−8, 7]) are
packed into a single ``int32`` from LSB to MSB.

Key conventions (matching compressed-tensors / vLLM):

- **Packing**: value[0] occupies bits [0:4], value[1] bits [4:8], …, value[7]
  bits [28:32].
- **Scale**: one ``bfloat16`` scale per group of ``group_size`` consecutive
  elements along the last dimension.  Dequant formula:
  ``float_val = int4_val * scale``.
- **Shape**: the ``weight_shape`` tensor stores the original ``[out_dim, in_dim]``
  dimensions as an ``int32`` vector.
"""

from __future__ import annotations

import torch


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 values from an I32 packed tensor.

    Args:
        packed: Shape ``(rows, packed_cols)`` with dtype ``int32``.
            Each element holds 8 signed 4-bit values.

    Returns:
        ``int8`` tensor of shape ``(rows, packed_cols * 8)`` with values in
        [−8, 7].
    """
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    # (rows, packed_cols, 1) >> (8,) → (rows, packed_cols, 8)
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    # Unsigned [0, 15] → signed [−8, 7]
    nibbles = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    rows, packed_cols, _ = nibbles.shape
    return nibbles.reshape(rows, packed_cols * 8).to(torch.int8)


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values (range [−8, 7]) into I32 packed format.

    Args:
        values: Shape ``(rows, cols)`` with cols divisible by 8.
            Values must be in [−8, 7].

    Returns:
        ``int32`` tensor of shape ``(rows, cols // 8)``.
    """
    rows, cols = values.shape
    if cols % 8 != 0:
        raise ValueError(f"cols ({cols}) must be divisible by 8")
    # Mask to unsigned nibble [0, 15]
    unsigned = values.to(torch.int32) & 0xF
    unsigned = unsigned.reshape(rows, -1, 8)
    shifts = torch.arange(0, 32, 4, device=values.device, dtype=torch.int32)
    packed = (unsigned << shifts).sum(dim=-1)
    return packed.to(torch.int32)


def dequantize_int4_group(
    packed: torch.Tensor,
    scale: torch.Tensor,
    original_shape: tuple[int, int],
    group_size: int = 32,
) -> torch.Tensor:
    """Dequantize INT4 group-quantized packed weights to bfloat16.

    Args:
        packed: I32 packed tensor, shape ``(out_dim, in_dim // 8)``.
        scale: BF16 per-group scale, shape ``(out_dim, in_dim // group_size)``.
        original_shape: ``(out_dim, in_dim)`` — the original weight dimensions.
        group_size: Number of elements per quantization group (default 32).

    Returns:
        BF16 tensor of shape ``original_shape``.
    """
    out_dim, in_dim = original_shape
    unpacked = unpack_int4(packed)  # (out_dim, packed_cols * 8)
    # Trim to original in_dim (handles padding if any)
    unpacked = unpacked[:, :in_dim]
    # Group-wise dequant
    grouped = unpacked.reshape(out_dim, -1, group_size).float()
    dequantized = grouped * scale.unsqueeze(-1).float()
    return dequantized.reshape(out_dim, in_dim).to(torch.bfloat16)


def quantize_int4_group(
    tensor: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D float tensor to INT4 with symmetric group quantization.

    Args:
        tensor: 2D float/bfloat16 tensor of shape ``(out_dim, in_dim)``.
            ``in_dim`` must be divisible by ``group_size``.
        group_size: Number of elements per quantization group (default 32).

    Returns:
        Tuple of ``(packed, scale)`` where:
        - ``packed``: I32 tensor of shape ``(out_dim, in_dim // 8)``
        - ``scale``: BF16 tensor of shape ``(out_dim, in_dim // group_size)``
    """
    out_dim, in_dim = tensor.shape
    if in_dim % group_size != 0:
        raise ValueError(f"in_dim ({in_dim}) must be divisible by group_size ({group_size})")
    grouped = tensor.float().reshape(out_dim, -1, group_size)  # (out, n_groups, gs)
    # Symmetric scale: max(|vals|) / 7
    group_max = grouped.abs().amax(dim=-1)  # (out, n_groups)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-12)
    # Quantize
    quantized = (grouped / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    quantized = quantized.reshape(out_dim, in_dim)
    packed = pack_int4(quantized)
    return packed, scale.to(torch.bfloat16)
