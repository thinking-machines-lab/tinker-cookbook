"""MXFP4 (OCP Microscaling FP4) quantization and dequantization.

Supports the MX block-quantized format used by models like GPT-OSS.
In this format, weights are stored as:

- **blocks**: ``uint8`` tensor where each byte holds 2 FP4 E2M1 values
  (low nibble = first value, high nibble = second value).
- **scales**: ``uint8`` tensor of E8M0 exponents (one per block of 32 values).

The OCP MX block size is 32 elements. Each block of 32 FP4 values is packed
into 16 ``uint8`` bytes, and shares a single E8M0 scale factor.

Key conventions:

- **FP4 E2M1**: 4-bit float with 1 sign bit, 2 exponent bits, 1 mantissa bit.
  Representable values: ``{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}``.
- **E8M0 scale**: 8-bit exponent-only format. ``scale = 2^(uint8 - 127)``.
- **Dequant**: ``float_val = fp4_lookup[nibble] * 2^(scale_e8m0 - 127)``
"""

from __future__ import annotations

import torch

# OCP MX block size: 32 elements per scale factor.
BLOCK_SIZE = 32

# FP4 E2M1 lookup table: 4-bit index -> float value.
# Indices 0-7 are positive, 8-15 are the sign-flipped negatives.
_FP4_E2M1_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

# Max representable magnitude in FP4 E2M1.
_FP4_MAX = 6.0


def unpack_mxfp4(blocks: torch.Tensor) -> torch.Tensor:
    """Unpack MXFP4 values from uint8 packed tensor.

    Each ``uint8`` byte holds two FP4 values: low nibble (bits 0-3) is the
    first value, high nibble (bits 4-7) is the second.

    Args:
        blocks: ``uint8`` tensor of shape ``(..., packed_per_block)`` where
            ``packed_per_block = BLOCK_SIZE // 2 = 16``.

    Returns:
        ``float32`` tensor of shape ``(..., BLOCK_SIZE)`` with dequantized
        FP4 values (before scale application).
    """
    lut = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=blocks.device)
    low = (blocks & 0x0F).to(torch.int64)
    high = ((blocks >> 4) & 0x0F).to(torch.int64)
    # Interleave: [low0, high0, low1, high1, ...]
    low_vals = lut[low]
    high_vals = lut[high]
    interleaved = torch.stack([low_vals, high_vals], dim=-1)
    return interleaved.reshape(*blocks.shape[:-1], blocks.shape[-1] * 2)


def pack_mxfp4(values: torch.Tensor) -> torch.Tensor:
    """Pack float values into MXFP4 uint8 format.

    Snaps each value to the nearest FP4 E2M1 representable value, then
    packs pairs into ``uint8`` bytes (low nibble + high nibble).

    Args:
        values: Float tensor of shape ``(..., BLOCK_SIZE)`` where the last
            dimension is divisible by 2. Values should already be scaled
            to the FP4 representable range.

    Returns:
        ``uint8`` tensor of shape ``(..., BLOCK_SIZE // 2)``.
    """
    lut = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=values.device)
    # Find nearest FP4 value for each input: minimize |value - lut[i]|
    # Shape: (..., block_size, 16) -> argmin over last dim
    diffs = (values.unsqueeze(-1) - lut.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)
    # Pack pairs: low nibble = even indices, high nibble = odd indices
    pairs = indices.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    return packed


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize MXFP4 block-quantized weights to bfloat16.

    Args:
        blocks: ``uint8`` packed tensor, shape ``(*batch, out_dim, n_blocks, 16)``.
        scales: ``uint8`` E8M0 scale tensor, shape ``(*batch, out_dim, n_blocks)``.
        original_shape: Target output shape ``(*batch, out_dim, in_dim)`` where
            ``in_dim = n_blocks * BLOCK_SIZE``.

    Returns:
        ``bfloat16`` tensor of shape ``original_shape``.
    """
    # Unpack FP4 values: (..., out_dim, n_blocks, 16) -> (..., out_dim, n_blocks, 32)
    unpacked = unpack_mxfp4(blocks)

    # Decode E8M0 scales: uint8 -> float via 2^(val - 127)
    scale_float = torch.pow(2.0, scales.float() - 127.0)

    # Apply scales: broadcast (n_blocks,) over (n_blocks, 32)
    dequantized = unpacked * scale_float.unsqueeze(-1)

    # Reshape to original dimensions
    return dequantized.reshape(original_shape).to(torch.bfloat16)


def quantize_mxfp4(
    tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to MXFP4 block-quantized format.

    Args:
        tensor: Float/bfloat16 tensor of shape ``(*batch, out_dim, in_dim)``.
            ``in_dim`` must be divisible by ``block_size``.
        block_size: Number of elements per quantization block (default 32).

    Returns:
        Tuple of ``(blocks, scales)`` where:
        - ``blocks``: ``uint8`` tensor of shape ``(*batch, out_dim, n_blocks, 16)``
        - ``scales``: ``uint8`` tensor of shape ``(*batch, out_dim, n_blocks)``
    """
    in_dim = tensor.shape[-1]
    if in_dim % block_size != 0:
        raise ValueError(f"in_dim ({in_dim}) must be divisible by block_size ({block_size})")

    # Reshape into blocks: (..., out_dim, n_blocks, block_size)
    n_blocks = in_dim // block_size
    grouped = tensor.float().reshape(*tensor.shape[:-1], n_blocks, block_size)

    # Compute E8M0 scale per block: exponent = ceil(log2(max(|vals|) / FP4_MAX))
    block_max = grouped.abs().amax(dim=-1)  # (..., out_dim, n_blocks)
    # Avoid log2(0) — clamp to tiny positive
    block_max = block_max.clamp(min=1e-12)
    scale_exp = torch.ceil(torch.log2(block_max / _FP4_MAX)).clamp(min=-127, max=127)
    scales_uint8 = (scale_exp + 127).to(torch.uint8)

    # Quantize: normalize by scale, then snap to nearest FP4
    scale_float = torch.pow(2.0, scale_exp).unsqueeze(-1)  # (..., n_blocks, 1)
    normalized = grouped / scale_float
    # Clamp to FP4 range before packing
    normalized = normalized.clamp(-_FP4_MAX, _FP4_MAX)

    # Pack to uint8
    blocks = pack_mxfp4(normalized)

    return blocks, scales_uint8
