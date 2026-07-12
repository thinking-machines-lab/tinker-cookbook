"""Tests for completer sample capture (Hook 3): the sample sink in
``completers.py``, ``capture_samples``, and ``TokenDbWriter.record_sample``."""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import tinker

from tinker_cookbook import completers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.tokendb.capture import (
    CaptureContext,
    capture_samples,
    sample_to_row,
    set_capture_context,
)
from tinker_cookbook.tokendb.capture_test import FakeTokenizer, ListWriter
from tinker_cookbook.tokendb.writer import TokenDbWriter
from tinker_cookbook.tokendb.writer_test import read_all_segments


@dataclass
class FakeSequence:
    tokens: list[int]
    logprobs: list[float] | None = None
    stop_reason: str = "stop"


@dataclass
class FakeSampleResult:
    sequences: list[FakeSequence]


@dataclass
class FakeSamplingClient:
    """Stub sampling client returning canned sequences; records its calls."""

    sequences: list[FakeSequence] = field(
        default_factory=lambda: [FakeSequence(tokens=[10, 11], logprobs=[-0.5, -0.25])]
    )
    calls: list[dict] = field(default_factory=list)

    async def sample_async(self, prompt, num_samples, sampling_params) -> FakeSampleResult:
        self.calls.append(
            {"prompt": prompt, "num_samples": num_samples, "sampling_params": sampling_params}
        )
        return FakeSampleResult(sequences=list(self.sequences))


@pytest.fixture(autouse=True)
def _reset_sample_sink():
    """Each test starts and ends with no registered sink."""
    completers.set_sample_sink(None)
    yield
    completers.set_sample_sink(None)


def make_completer(client: FakeSamplingClient | None = None) -> TinkerTokenCompleter:
    return TinkerTokenCompleter(
        sampling_client=client or FakeSamplingClient(),  # type: ignore[arg-type]
        max_tokens=32,
    )


def run_completer(completer: TinkerTokenCompleter, prompt_tokens: list[int] | None = None):
    model_input = tinker.ModelInput.from_ints(prompt_tokens or [1, 2, 3])
    return asyncio.run(completer(model_input, stop=[99]))


class TestSampleSink:
    def test_no_sink_no_behavior_change(self):
        assert completers.get_sample_sink() is None
        result = run_completer(make_completer())
        assert result.tokens == [10, 11]
        assert result.maybe_logprobs == [-0.5, -0.25]
        assert result.stop_reason == "stop"

    def test_sink_fires_with_prompt_sequences_and_metadata(self):
        seen: list[tuple] = []
        completers.set_sample_sink(lambda mi, seqs, meta: seen.append((mi, seqs, meta)))
        run_completer(make_completer(), prompt_tokens=[1, 2, 3])
        assert len(seen) == 1
        model_input, sequences, metadata = seen[0]
        assert model_input.to_ints() == [1, 2, 3]
        assert [s.tokens for s in sequences] == [[10, 11]]
        assert metadata["completer"] == "TinkerTokenCompleter"
        assert metadata["max_tokens"] == 32
        assert metadata["temperature"] == 1.0

    def test_sink_exception_does_not_break_sampling(self):
        def bad_sink(mi, seqs, meta):
            raise RuntimeError("boom")

        completers.set_sample_sink(bad_sink)
        result = run_completer(make_completer())
        assert result.tokens == [10, 11]


class TestCaptureSamples:
    def test_rows_have_sample_source_and_metadata(self):
        writer = ListWriter()
        with capture_samples(writer, dataset="multilingual"):
            run_completer(make_completer(), prompt_tokens=[1, 2, 3])
            run_completer(make_completer(), prompt_tokens=[4, 5])
        assert len(writer.rows) == 2
        for row in writer.rows:
            assert row.source == "sample"
            assert row.ob_is_delta is False
            assert row.split == "sample"
            assert row.iteration == -1
            assert row.extra["dataset"] == "multilingual"
            assert row.extra["completer"] == "TinkerTokenCompleter"
            assert row.ac_tokens == [10, 11]
            assert row.ac_logprobs == [-0.5, -0.25]
            assert row.stop_reason == "stop"
        assert [row.ob_tokens for row in writer.rows] == [[1, 2, 3], [4, 5]]
        # group_idx counts sample calls; traj_idx indexes within one call.
        assert [(row.group_idx, row.traj_idx) for row in writer.rows] == [(0, 0), (1, 0)]

    def test_identity_from_capture_context(self):
        writer = ListWriter()
        ctx = CaptureContext(split="train", iteration=7, sampling_client_step=3, tags=("t1",))
        with capture_samples(writer), set_capture_context(ctx):
            run_completer(make_completer())
        (row,) = writer.rows
        assert row.split == "train"
        assert row.iteration == 7
        assert row.sampling_client_step == 3
        assert row.tags == ["t1"]

    def test_sink_restored_on_exit(self):
        sentinel = lambda mi, seqs, meta: None  # noqa: E731
        completers.set_sample_sink(sentinel)
        with capture_samples(ListWriter()):
            assert completers.get_sample_sink() is not sentinel
        assert completers.get_sample_sink() is sentinel

    def test_writer_failure_does_not_break_sampling(self):
        class ExplodingWriter(ListWriter):
            def append_rows(self, rows) -> None:
                raise RuntimeError("disk full")

        with capture_samples(ExplodingWriter()):
            result = run_completer(make_completer())
        assert result.tokens == [10, 11]

    def test_parquet_roundtrip(self, tmp_path: Path):
        with TokenDbWriter(tmp_path, flush_interval_s=3600) as writer:
            with capture_samples(writer, purpose="test"):
                run_completer(make_completer(), prompt_tokens=[1, 2, 3])
        table = read_all_segments(tmp_path)
        assert table.num_rows == 1
        record = table.to_pylist()[0]
        assert record["source"] == "sample"
        assert record["ob_tokens"] == [1, 2, 3]
        assert record["ac_tokens"] == [10, 11]
        assert json.loads(record["extra"])["purpose"] == "test"


class TestSampleToRow:
    def test_tokens_with_logprobs_input(self):
        seq = completers.TokensWithLogprobs(tokens=[7, 8], maybe_logprobs=None)
        row = sample_to_row(tinker.ModelInput.from_ints([1]), seq)
        assert row.ac_tokens == [7, 8]
        assert row.ac_logprobs is None  # maybe_logprobs=None must not raise

    def test_text_decoded_with_tokenizer(self):
        seq = FakeSequence(tokens=[7, 8], logprobs=[-1.0, -2.0])
        row = sample_to_row(tinker.ModelInput.from_ints([1, 2]), seq, tokenizer=FakeTokenizer())
        assert row.ob_text == "t1 t2"
        assert row.ac_text == "t7 t8"


class TestRecordSample:
    def test_parquet_roundtrip(self, tmp_path: Path):
        seq = FakeSequence(tokens=[10, 11], logprobs=[-0.5, -0.25], stop_reason="length")
        with TokenDbWriter(tmp_path, flush_interval_s=3600) as writer:
            row = writer.record_sample(
                tinker.ModelInput.from_ints([1, 2, 3]),
                seq,
                group_idx=4,
                tokenizer=FakeTokenizer(),
                prompt_id="p4",
            )
            assert row.source == "sample"
        table = read_all_segments(tmp_path)
        assert table.num_rows == 1
        record = table.to_pylist()[0]
        assert record["source"] == "sample"
        assert record["split"] == "sample"
        assert record["iteration"] == -1
        assert record["group_idx"] == 4
        assert record["ob_tokens"] == [1, 2, 3]
        assert record["ob_is_delta"] is False
        assert record["ac_tokens"] == [10, 11]
        assert record["ac_logprobs"] == pytest.approx([-0.5, -0.25])
        assert record["stop_reason"] == "length"
        assert record["ac_text"] == "t10 t11"
        assert json.loads(record["extra"]) == {"prompt_id": "p4"}
        assert record["run_id"] == writer.run_id
