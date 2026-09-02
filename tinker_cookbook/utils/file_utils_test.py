import json
from pathlib import Path

from tinker_cookbook.utils.file_utils import read_jsonl


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"step": 0, "loss": 1.0}),
                "",
                "   ",
                json.dumps({"step": 1, "loss": 0.5}),
                "",
            ]
        )
    )

    assert read_jsonl(str(path)) == [
        {"step": 0, "loss": 1.0},
        {"step": 1, "loss": 0.5},
    ]
