from pathlib import Path

from tinker_cookbook.utils.misc_utils import expand_path_or_uri, is_uri, iteration_dir


def test_expand_path_or_uri_preserves_cloud_uri():
    uri = "gs://bucket/path/to/run"
    assert is_uri(uri)
    assert expand_path_or_uri(uri) == uri


def test_expand_path_or_uri_preserves_s3_uri():
    uri = "s3://bucket/path/to/run"
    assert is_uri(uri)
    assert expand_path_or_uri(uri) == uri


def test_is_uri_includes_file_uri():
    assert is_uri("file:///tmp/run")


def test_expand_path_or_uri_preserves_file_uri():
    uri = "file:///tmp/run"
    assert expand_path_or_uri(uri) == uri


def test_expand_path_or_uri_expands_local_path():
    assert expand_path_or_uri("~/run").startswith(str(Path("~").expanduser()))


def test_iteration_dir_for_local_path():
    assert iteration_dir("/tmp/run", 3) == Path("/tmp/run/iteration_000003")


def test_iteration_dir_skips_uri_paths():
    assert iteration_dir("gs://bucket/path/to/run", 3) is None
