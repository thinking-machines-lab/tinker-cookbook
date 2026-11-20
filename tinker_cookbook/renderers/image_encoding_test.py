"""JPEG byte compatibility and resource cleanup, without model downloads or API calls."""

import base64
import io
import random
from concurrent.futures import ThreadPoolExecutor
from typing import BinaryIO, cast

import pytest
from PIL import Image

from tinker_cookbook.renderers import base


class PatchProcessor:
    merge_size = 2

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict[str, object]
    ) -> int:
        assert images_kwargs == {}
        return ((height + 15) // 16) * ((width + 15) // 16) * self.merge_size**2


def make_image(mode: str, size: tuple[int, int]) -> Image.Image:
    pixels = random.Random(0).randbytes(size[0] * size[1] * 3)
    return Image.frombytes("RGB", size, pixels).convert(mode)


def reference_jpeg(image: Image.Image) -> bytes:
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        return buffer.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA", "P", "L", "CMYK", "YCbCr"])
@pytest.mark.parametrize("size", [(17, 33), (64, 48), (127, 95)])
def test_image_chunk_matches_buffer_encoding(mode: str, size: tuple[int, int]) -> None:
    image = make_image(mode, size)
    original_pixels = image.tobytes()
    processor = PatchProcessor()

    chunk = base.image_to_chunk(image, cast(base.ImageProcessorProtocol, processor))

    assert chunk.data == reference_jpeg(image)
    assert chunk.format == "jpeg"
    assert chunk.expected_tokens == ((size[1] + 15) // 16) * ((size[0] + 15) // 16)
    assert image.mode == mode
    assert image.size == size
    assert image.tobytes() == original_pixels


def test_concurrent_image_chunks_match_buffer_encoding() -> None:
    images = [make_image(mode, (127, 95)) for mode in ["RGB", "RGBA", "L", "CMYK"] * 4]
    expected = [reference_jpeg(image) for image in images]
    processor = cast(base.ImageProcessorProtocol, PatchProcessor())

    with ThreadPoolExecutor(max_workers=8) as pool:
        chunks = list(pool.map(lambda image: base.image_to_chunk(image, processor), images))

    assert [chunk.data for chunk in chunks] == expected
    assert all(chunk.format == "jpeg" and chunk.expected_tokens == 48 for chunk in chunks)


def test_data_uri_matches_pil_input() -> None:
    image = make_image("RGBA", (64, 48))
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    processor = cast(base.ImageProcessorProtocol, PatchProcessor())

    assert base.image_to_chunk(uri, processor) == base.image_to_chunk(image, processor)


def test_resize_config_token_count() -> None:
    class ResizeProcessor:
        def get_resize_config(self, image_data: dict[str, object]) -> dict[str, int]:
            assert image_data["type"] == "image"
            image = image_data["image"]
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"
            assert image.size == (64, 48)
            return {"num_tokens": 37}

    image = make_image("RGBA", (64, 48))
    chunk = base.image_to_chunk(image, cast(base.ImageProcessorProtocol, ResizeProcessor()))
    assert chunk.data == reference_jpeg(image)
    assert chunk.expected_tokens == 37


def test_nemotron_token_count() -> None:
    class NemotronProcessor:
        _downsample_factor = 2
        max_model_len = 100
        max_num_patches = 1000
        min_num_patches = 16

        def _compute_target_patches(self, image: Image.Image, budget: int) -> tuple[int, int]:
            assert image.size == (64, 48)
            assert budget == 384
            return 12, 8

    image = make_image("RGB", (64, 48))
    chunk = base.image_to_chunk(image, cast(base.ImageProcessorProtocol, NemotronProcessor()))
    assert chunk.data == reference_jpeg(image)
    assert chunk.expected_tokens == 24


@pytest.mark.parametrize("encoding_fails", [False, True])
def test_temporary_file_closed(monkeypatch: pytest.MonkeyPatch, encoding_fails: bool) -> None:
    opened: list[BinaryIO] = []
    temporary_file = base.tempfile.TemporaryFile

    def create_file() -> BinaryIO:
        result = temporary_file()
        opened.append(result)
        return result

    def fail_save(
        self: Image.Image, fp: BinaryIO, format: str | None = None, **params: object
    ) -> None:
        fp.write(b"partial JPEG")
        raise OSError("encoding failed")

    monkeypatch.setattr(base.tempfile, "TemporaryFile", create_file)
    image = make_image("RGB", (64, 48))
    processor = cast(base.ImageProcessorProtocol, PatchProcessor())

    if encoding_fails:
        monkeypatch.setattr(Image.Image, "save", fail_save)
        with pytest.raises(OSError, match="encoding failed"):
            base.image_to_chunk(image, processor)
    else:
        base.image_to_chunk(image, processor)

    assert len(opened) == 1
    assert opened[0].closed
