"""Focused test for the prompt_distillation token DB integration.

Runs ``create_data_async`` with stub clients and a recording writer to assert
the ``record_sample`` call shape (typed attrs incl. row identity, free-form
kwargs), without any network or real generation.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import tinker

from tinker_cookbook.recipes.prompt_distillation import create_data


@dataclass
class _FakeSequence:
    tokens: list[int]
    stop_reason: str = "stop"


@dataclass
class _FakeResult:
    sequences: list[_FakeSequence]


class _FakeSamplingClient:
    async def sample_async(self, prompt, sampling_params, num_samples) -> _FakeResult:
        return _FakeResult(sequences=[_FakeSequence(tokens=[10, 11])])


class _FakeTokenizer:
    def decode(self, tokens: list[int]) -> str:
        return "Final Answer: en"


class _FakeRenderer:
    def build_generation_prompt(self, messages) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints([1, 2, 3])

    def get_stop_sequences(self) -> list[str]:
        return []


class _RecordingWriter:
    """Stub writer capturing record_sample call kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_sample(self, model_input, sequence, **kwargs) -> None:
        self.calls.append({"model_input": model_input, "sequence": sequence, **kwargs})


def test_record_sample_call_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # create_data_async reads the example data via a repo-root-relative path.
    repo_root = Path(create_data.__file__).parents[3]
    monkeypatch.chdir(repo_root)

    config = create_data.Config(output_file=str(tmp_path / "out.jsonl"))
    writer = _RecordingWriter()
    asyncio.run(
        create_data.create_data_async(
            config, _FakeSamplingClient(), _FakeTokenizer(), _FakeRenderer(), writer
        )
    )

    with open("tinker_cookbook/example_data/multilingual.txt") as f:
        n_sentences = len(f.readlines())
    assert len(writer.calls) == n_sentences

    call = writer.calls[0]
    # Categorical dimensions go through the typed attrs channel, with the
    # sentence index as promotable row identity.
    attrs = call["attrs"]
    assert attrs["teacher_model"] == create_data.TEACHER_MODEL
    assert attrs["source_dataset"] == "multilingual.txt"
    assert attrs["row_id"] == f"multilingual/{call['group_idx']}"
    # The raw sentence stays a free-form kwarg (extra JSON column).
    assert isinstance(call["sentence"], str) and call["sentence"]
