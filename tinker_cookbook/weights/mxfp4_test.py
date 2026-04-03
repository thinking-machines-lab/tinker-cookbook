"""Unit tests for MXFP4 quantization math."""

import pytest
import torch

from tinker_cookbook.weights._mxfp4 import (
    dequantize_mxfp4,
    pack_mxfp4,
    quantize_mxfp4,
    unpack_mxfp4,
)


class TestUnpackPack:
    def test_roundtrip_all_fp4_values(self):
        """All 16 FP4 E2M1 values survive pack/unpack."""
        # Create a block with all 16 possible nibble values (0-15)
        # packed as pairs: (0,1), (2,3), ..., (14,15)
        packed = torch.zeros(8, dtype=torch.uint8)
        for i in range(8):
            low = i * 2
            high = i * 2 + 1
            packed[i] = low | (high << 4)

        unpacked = unpack_mxfp4(packed.unsqueeze(0))  # (1, 16)
        repacked = pack_mxfp4(unpacked)  # (1, 8)
        reunpacked = unpack_mxfp4(repacked)

        assert torch.equal(unpacked, reunpacked)

    def test_unpack_known_values(self):
        """Verify unpack produces correct FP4 float values."""
        # Pack: low=2 (FP4=1.0), high=5 (FP4=3.0)
        packed = torch.tensor([2 | (5 << 4)], dtype=torch.uint8)
        unpacked = unpack_mxfp4(packed.unsqueeze(0))
        assert unpacked[0, 0].item() == pytest.approx(1.0)
        assert unpacked[0, 1].item() == pytest.approx(3.0)

    def test_unpack_negative_values(self):
        """Verify negative FP4 values."""
        # Pack: low=10 (FP4=-1.0), high=13 (FP4=-3.0)
        packed = torch.tensor([10 | (13 << 4)], dtype=torch.uint8)
        unpacked = unpack_mxfp4(packed.unsqueeze(0))
        assert unpacked[0, 0].item() == pytest.approx(-1.0)
        assert unpacked[0, 1].item() == pytest.approx(-3.0)

    def test_pack_snaps_to_nearest(self):
        """Values between FP4 representable points snap to nearest."""
        # 0.7 is between 0.5 and 1.0, should snap to 0.5 (closer)
        # 1.3 is between 1.0 and 1.5, should snap to 1.5 (closer)
        values = torch.tensor([[0.7, 1.3]], dtype=torch.float32)
        packed = pack_mxfp4(values)
        unpacked = unpack_mxfp4(packed)
        assert unpacked[0, 0].item() == pytest.approx(0.5)
        assert unpacked[0, 1].item() == pytest.approx(1.5)


class TestE8M0Scales:
    def test_scale_encoding(self):
        """E8M0 scale: uint8=127 -> 2^0 = 1.0."""
        blocks = torch.zeros(1, 1, 1, 16, dtype=torch.uint8)
        scales = torch.tensor([[[127]]], dtype=torch.uint8)
        result = dequantize_mxfp4(blocks, scales, (1, 1, 32))
        # All zeros * scale 1.0 = all zeros
        assert (result == 0).all()

    def test_scale_power_of_two(self):
        """E8M0 scale: uint8=130 -> 2^3 = 8.0."""
        # Put FP4 value 1.0 (nibble=2) in low position
        blocks = torch.tensor([[[[2 | (0 << 4)] + [0] * 15]]], dtype=torch.uint8)
        scales = torch.tensor([[[130]]], dtype=torch.uint8)  # 2^(130-127) = 8.0
        result = dequantize_mxfp4(blocks, scales, (1, 1, 32))
        # FP4(1.0) * 8.0 = 8.0
        assert result[0, 0, 0].item() == pytest.approx(8.0, abs=0.1)


class TestQuantizeDequantize:
    def test_roundtrip_small_tensor(self):
        """Quantize -> dequantize produces values close to original."""
        torch.manual_seed(42)
        tensor = torch.randn(2, 4, 64, dtype=torch.bfloat16) * 0.1
        blocks, scales = quantize_mxfp4(tensor)
        recovered = dequantize_mxfp4(blocks, scales, tensor.shape)
        # FP4 has very low precision, but relative error should be bounded
        rel_error = (recovered.float() - tensor.float()).abs() / (tensor.float().abs() + 1e-6)
        assert rel_error.mean() < 1.0, f"Mean relative error too high: {rel_error.mean()}"

    def test_roundtrip_preserves_shape(self):
        """Output shape matches input shape."""
        tensor = torch.randn(4, 128, 64, dtype=torch.bfloat16) * 0.01
        blocks, scales = quantize_mxfp4(tensor)
        recovered = dequantize_mxfp4(blocks, scales, tensor.shape)
        assert recovered.shape == tensor.shape

    def test_blocks_dtype_uint8(self):
        tensor = torch.randn(2, 4, 32, dtype=torch.bfloat16)
        blocks, scales = quantize_mxfp4(tensor)
        assert blocks.dtype == torch.uint8
        assert scales.dtype == torch.uint8

    def test_blocks_shape(self):
        """Blocks pack 2 values per byte: last dim = block_size // 2."""
        tensor = torch.randn(2, 4, 64, dtype=torch.bfloat16)
        blocks, scales = quantize_mxfp4(tensor)
        # 64 / 32 = 2 blocks, each packed to 16 bytes
        assert blocks.shape == (2, 4, 2, 16)
        assert scales.shape == (2, 4, 2)

    def test_in_dim_not_divisible_raises(self):
        tensor = torch.randn(2, 4, 33, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="divisible by block_size"):
            quantize_mxfp4(tensor)

    def test_zeros_roundtrip(self):
        tensor = torch.zeros(2, 4, 32, dtype=torch.bfloat16)
        blocks, scales = quantize_mxfp4(tensor)
        recovered = dequantize_mxfp4(blocks, scales, tensor.shape)
        assert (recovered == 0).all()

    def test_exact_fp4_values_roundtrip(self):
        """Values that are exactly representable in FP4 should round-trip exactly."""
        # FP4 representable: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
        exact_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0]
        # Pad to block_size=32
        padded = exact_vals + [0.0] * (32 - len(exact_vals))
        tensor = torch.tensor([[padded]], dtype=torch.bfloat16)
        blocks, scales = quantize_mxfp4(tensor)
        recovered = dequantize_mxfp4(blocks, scales, tensor.shape)
        for i, val in enumerate(exact_vals):
            assert recovered[0, 0, i].item() == pytest.approx(val, abs=0.01), (
                f"Value {val} at index {i} became {recovered[0, 0, i].item()}"
            )
