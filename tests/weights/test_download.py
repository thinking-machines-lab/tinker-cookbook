"""Integration test for weights.download().

Requires TINKER_API_KEY to be set. Skipped otherwise.
"""

import os
import tempfile
from pathlib import Path

import pytest

from tinker_cookbook.weights import download


@pytest.mark.integration
class TestDownloadIntegration:
    """Download a real adapter from Tinker and verify the extracted files."""

    def _get_test_tinker_path(self) -> str:
        """Return a known tinker checkpoint path for testing.

        Uses the smoke test checkpoint if available via env var,
        otherwise skips.
        """
        path = os.environ.get("TINKER_TEST_CHECKPOINT_PATH")
        if not path:
            pytest.skip(
                "Set TINKER_TEST_CHECKPOINT_PATH to a valid tinker:// path to run this test"
            )
        return path

    def test_download_and_extract(self):
        tinker_path = self._get_test_tinker_path()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = str(Path(tmpdir) / "adapter")

            result = download(tinker_path=tinker_path, output_dir=output_dir)

            assert result == output_dir
            out = Path(output_dir)
            assert out.is_dir(), f"Output directory not created: {output_dir}"

            # Verify at least one file was extracted
            files = list(out.rglob("*"))
            assert len(files) > 0, "No files extracted from archive"

            # If it's a LoRA adapter, check for expected files
            if (out / "adapter_model.safetensors").exists():
                assert (out / "adapter_config.json").exists(), (
                    "adapter_model.safetensors found but adapter_config.json missing"
                )
