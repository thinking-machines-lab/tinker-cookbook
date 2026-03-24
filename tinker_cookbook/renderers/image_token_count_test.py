"""Tests for image token count calculation in image_to_chunk.

Regression test for https://github.com/thinking-machines-lab/tinker-cookbook/issues/181:
The slow (non-fast) Qwen2VLImageProcessor in transformers <5.0 had a bug where it ignored the
model's min_pixels/max_pixels config and used hardcoded defaults (min_pixels=3136 instead of
65536). This caused image_to_chunk to compute wrong expected_tokens, leading to a 400 error
from the Tinker server ("Expected N tokens, got M from image").

The root cause was a flattened if/else in the processor's __init__ that always overwrote the
size config with defaults. This was fixed in transformers 5.x by nesting the conditional.

These tests verify that:
1. The loaded image processor uses the model config's min_pixels/max_pixels (not hardcoded defaults)
2. image_to_chunk produces correct expected_tokens for various image dimensions
"""

import pytest
import tinker.types
import transformers
from PIL import Image

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers.base import image_to_chunk

_transformers_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
_requires_transformers_5 = pytest.mark.skipif(
    _transformers_version < (5, 0),
    reason="Qwen2VLImageProcessor has a known bug in transformers <5.0 that ignores model config "
    "min_pixels/max_pixels (https://github.com/huggingface/transformers/issues/42910)",
)

# The Qwen3-VL preprocessor_config.json specifies these values.
# The buggy slow processor on transformers <5.0 would use 3136 / 1003520 instead.
QWEN3_VL_EXPECTED_MIN_PIXELS = 65536
QWEN3_VL_EXPECTED_MAX_PIXELS = 16777216

QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@_requires_transformers_5
def test_qwen3_vl_image_processor_uses_config_pixels() -> None:
    """The image processor must use min/max pixels from the model config, not hardcoded defaults.

    On transformers <5.0, the slow processor always used min_pixels=3136 (hardcoded)
    instead of min_pixels=65536 (from config), causing wrong token counts.
    """
    processor = get_image_processor(QWEN3_VL_MODEL)

    assert processor.min_pixels == QWEN3_VL_EXPECTED_MIN_PIXELS, (
        f"Image processor min_pixels={processor.min_pixels}, expected {QWEN3_VL_EXPECTED_MIN_PIXELS}. "
        f"This may indicate the slow (non-fast) image processor is loaded with buggy defaults. "
        f"See https://github.com/huggingface/transformers/issues/42910"
    )
    assert processor.max_pixels == QWEN3_VL_EXPECTED_MAX_PIXELS, (
        f"Image processor max_pixels={processor.max_pixels}, expected {QWEN3_VL_EXPECTED_MAX_PIXELS}."
    )


@_requires_transformers_5
def test_qwen3_vl_image_to_chunk_token_counts() -> None:
    """Verify image_to_chunk computes correct expected_tokens for various image sizes.

    The expected token counts are derived from the correct formula:
        smart_resize(h, w, factor=32, min_pixels=65536) -> (rh, rw)
        patches = (rh // patch_size) * (rw // patch_size)
        tokens = patches // merge_size**2

    With the buggy processor (min_pixels=3136), small images get far fewer tokens
    because they aren't upscaled to meet the min_pixels threshold.
    """
    processor = get_image_processor(QWEN3_VL_MODEL)

    # (width, height, correct_tokens, buggy_tokens_with_min_pixels_3136)
    # The buggy column documents what the old processor would compute, to show the discrepancy.
    test_cases: list[tuple[int, int, int]] = [
        (224, 224, 64),  # buggy: 49 (no upscale with min_pixels=3136)
        (150, 224, 70),  # buggy: 35 — exactly the "Expected 35, got 70" from issue #181
        (224, 150, 70),  # buggy: 35
        (400, 300, 108),  # buggy: 108 (large enough that min_pixels doesn't matter)
    ]

    for width, height, expected_tokens in test_cases:
        image = Image.new("RGB", (width, height))
        chunk = image_to_chunk(image, processor)

        assert isinstance(chunk, tinker.types.ImageChunk)
        assert chunk.expected_tokens == expected_tokens, (
            f"image_to_chunk({width}x{height}) returned expected_tokens={chunk.expected_tokens}, "
            f"want {expected_tokens}. If the value is lower, the image processor may be using "
            f"wrong min_pixels (hardcoded defaults instead of model config)."
        )
