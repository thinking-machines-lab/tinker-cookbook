"""
Utilities for working with image processors. Create new types to avoid needing to import AutoImageProcessor and BaseImageProcessor.


Avoid importing AutoImageProcessor and BaseImageProcessor until runtime, because they're slow imports.
"""

from __future__ import annotations

import os
from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

from PIL import Image

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.image_processing_utils import BaseImageProcessor

    ImageProcessor: TypeAlias = BaseImageProcessor | None
else:
    # make it importable from other files as a type in runtime
    ImageProcessor: TypeAlias = Any


@cache
def get_image_processor(model_name: str) -> ImageProcessor:
    model_name = model_name.split(":", 1)[0]

    # tml-renderers models tokenize images inside tml_renderers.v0
    # (BlackPaddedImageTokenizer); they have no Hugging Face image processor.
    # The tml_v0 cookbook renderer ignores
    # the image_processor argument, so return None instead of failing in AutoImageProcessor.
    if model_name == "thinkingmachines/Inkling":
        return None

    from transformers.models.auto.image_processing_auto import AutoImageProcessor

    kwargs: dict[str, Any] = {}
    if os.environ.get("HF_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"):
        kwargs["trust_remote_code"] = True

    if model_name == "moonshotai/Kimi-K2.5":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "3367c8d1c68584429fab7faf845a32d5195b6ac1"
    elif model_name == "moonshotai/Kimi-K2.6":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "b5aabbfb20227ed42becbf5541dbffd213942c58"

    processor = AutoImageProcessor.from_pretrained(model_name, **kwargs)

    return processor


def image_to_data_uri(image: Any) -> str:
    """Materialize a local image into a base64 JPEG ``data:`` URI.

    Accepts a ``PIL.Image.Image``, a ``data:`` URI (returned unchanged), or a local file
    path / ``file://`` URI. Remote URLs (``http(s)``, ``gs``, ``s3``, ...) are refused:
    callers should materialize remote assets into local inputs (files or in-memory base64 data) first.
    """
    import base64
    import io
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    def open_local_image(path: Path) -> Image.Image:
        with Image.open(path.expanduser()) as opened:
            return opened.copy()

    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        if image.startswith("data:"):
            return image
        parsed = urlparse(image)
        if parsed.scheme == "":
            pil_image = open_local_image(Path(image))
        elif parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
            pil_image = open_local_image(Path(unquote(parsed.path)))
        else:
            raise ValueError(
                f"does not fetch remote image URLs (scheme {parsed.scheme!r}); materialize "
                f"{image!r} into a PIL image, a local path, or a base64 data: URI first"
            )
    else:
        raise TypeError(f"image must be a PIL.Image.Image or str; got {type(image)!r}")

    if pil_image.mode in ("RGBA", "LA", "P"):
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
