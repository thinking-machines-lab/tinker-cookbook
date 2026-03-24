"""
Utilities for working with image processors. Create new types to avoid needing to import AutoImageProcessor and BaseImageProcessor.


Avoid importing AutoImageProcessor and BaseImageProcessor until runtime, because they're slow imports.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

from PIL import Image

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.image_processing_utils import BaseImageProcessor

    ImageProcessor: TypeAlias = BaseImageProcessor
else:
    # make it importable from other files as a type in runtime
    ImageProcessor: TypeAlias = Any


@cache
def get_image_processor(model_name: str) -> ImageProcessor:
    model_name = model_name.split(":")[0]

    import transformers
    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    kwargs: dict[str, Any] = {}
    if os.environ.get("HF_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"):
        kwargs["trust_remote_code"] = True

    if model_name == "moonshotai/Kimi-K2.5":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "3367c8d1c68584429fab7faf845a32d5195b6ac1"

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True, **kwargs)

    if not getattr(processor, "is_fast", False) and tuple(
        int(x) for x in transformers.__version__.split(".")[:2]
    ) < (5, 0):
        logging.getLogger(__name__).warning(
            "Loaded a slow (non-fast) image processor on transformers <5.0. "
            "The slow Qwen2VLImageProcessor has a known bug that ignores the model's "
            "min_pixels/max_pixels config, producing wrong image token counts. "
            "This causes 'Expected N tokens, got M from image' errors from the Tinker server. "
            "Upgrade to transformers >=5.0 or install torchvision to enable the fast processor. "
            "See https://github.com/huggingface/transformers/issues/42910"
        )

    return processor


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    """
    Resize an image so that its longest side is at most max_size pixels.

    Preserves aspect ratio and uses LANCZOS resampling for quality.
    Returns the original image if it's already smaller than max_size.
    """

    width, height = image.size
    if max(width, height) <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
