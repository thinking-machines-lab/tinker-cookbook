"""Tests for image token count calculation in image_to_chunk.

Regression test for https://github.com/thinking-machines-lab/tinker-cookbook/issues/181:
The slow (non-fast) Qwen2VLImageProcessor in transformers <5.0 had a bug where it ignored the
model's min_pixels/max_pixels config and used hardcoded defaults (min_pixels=3136 instead of
65536). This caused image_to_chunk to compute wrong expected_tokens, leading to a 400 error
from the Tinker server ("Expected N tokens, got M from image").

These tests verify that:
1. The loaded image processor uses the model config's min_pixels/max_pixels (not hardcoded defaults)
2. image_to_chunk produces correct expected_tokens for various image dimensions
"""

from typing import Any

import tinker.types
from PIL import Image

from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers.base import image_to_chunk

# The Qwen3-VL preprocessor_config.json specifies these values (via size.shortest_edge / size.longest_edge).
QWEN3_VL_EXPECTED_MIN_PIXELS = 65536
QWEN3_VL_EXPECTED_MAX_PIXELS = 16777216

QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def test_qwen3_vl_image_processor_uses_config_pixels() -> None:
    """The image processor must use min/max pixels from the model config, not hardcoded defaults."""
    processor: Any = get_image_processor(QWEN3_VL_MODEL)

    # In transformers >=5.5, min/max pixels are stored as size.shortest_edge / size.longest_edge.
    # In earlier 5.x versions they were direct attributes (min_pixels / max_pixels).
    if hasattr(processor, "min_pixels"):
        actual_min = processor.min_pixels
        actual_max = processor.max_pixels
    else:
        actual_min = processor.size["shortest_edge"]
        actual_max = processor.size["longest_edge"]

    assert actual_min == QWEN3_VL_EXPECTED_MIN_PIXELS, (
        f"Image processor min_pixels={actual_min}, expected {QWEN3_VL_EXPECTED_MIN_PIXELS}. "
        f"See https://github.com/huggingface/transformers/issues/42910"
    )
    assert actual_max == QWEN3_VL_EXPECTED_MAX_PIXELS, (
        f"Image processor max_pixels={actual_max}, expected {QWEN3_VL_EXPECTED_MAX_PIXELS}."
    )


def test_qwen3_vl_image_to_chunk_token_counts() -> None:
    """Verify image_to_chunk computes correct expected_tokens for various image sizes.

    The expected token counts are derived from the correct formula:
        smart_resize(h, w, factor=32, min_pixels=65536) -> (rh, rw)
        patches = (rh // patch_size) * (rw // patch_size)
        tokens = patches // merge_size**2

    With the buggy processor (min_pixels=3136), small images get far fewer tokens
    because they aren't upscaled to meet the min_pixels threshold.
    """
    processor: Any = get_image_processor(QWEN3_VL_MODEL)

    # (width, height, correct_tokens)
    test_cases: list[tuple[int, int, int]] = [
        (224, 224, 64),
        (150, 224, 70),  # Exactly the "Expected 35, got 70" from issue #181
        (224, 150, 70),
        (400, 300, 108),
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
