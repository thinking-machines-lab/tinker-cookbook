"""Tests for Storage protocol and LocalStorage."""

import asyncio
import pickle
from pathlib import Path

import pytest

from tinker_cookbook.stores.storage import (
    AsyncStorage,
    LocalStorage,
    Storage,
    storage_join,
)


class TestLocalStorage:
    def test_write_and_read(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        s.write("test.txt", b"hello")
        assert s.read("test.txt") == b"hello"

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        s.write("a/b/c.txt", b"deep")
        assert s.read("a/b/c.txt") == b"deep"

    def test_read_missing_raises(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        with pytest.raises(FileNotFoundError):
            s.read("nonexistent")

    def test_append(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        s.append("log.txt", b"line1\n")
        s.append("log.txt", b"line2\n")
        assert s.read("log.txt") == b"line1\nline2\n"

    def test_exists(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        assert not s.exists("nope")
        s.write("yes.txt", b"")
        assert s.exists("yes.txt")

    def test_stat(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        assert s.stat("nope") is None
        s.write("f.txt", b"12345")
        stat = s.stat("f.txt")
        assert stat is not None
        assert stat.size == 5
        assert stat.mtime > 0

    def test_read_range(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        s.write("data.bin", b"0123456789")
        assert s.read_range("data.bin", 3) == b"3456789"
        assert s.read_range("data.bin", 3, 4) == b"3456"

    def test_list_dir(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        s.write("dir/a.txt", b"")
        s.write("dir/b.txt", b"")
        s.write("dir/sub/c.txt", b"")
        items = s.list_dir("dir")
        assert "a.txt" in items
        assert "b.txt" in items
        assert "sub" in items

    def test_remove(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        s.write("f.txt", b"data")
        assert s.exists("f.txt")
        s.remove("f.txt")
        assert not s.exists("f.txt")
        s.remove("nonexistent")  # no error

    def test_remove_dir(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        s.write("d/f.txt", b"data")
        # Non-empty dir: silently ignored
        s.remove_dir("d")
        assert s.exists("d/f.txt")
        # Remove the file, then the dir
        s.remove("d/f.txt")
        s.remove_dir("d")
        assert not (tmp_path / "d").exists()
        # Missing dir: no error
        s.remove_dir("nonexistent")

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        with pytest.raises(ValueError, match="escapes"):
            s.read("../../etc/passwd")

    def test_context_manager(self, tmp_path: Path) -> None:
        with LocalStorage(tmp_path) as s:
            s.write("f.txt", b"ok")
            assert s.read("f.txt") == b"ok"

    def test_implements_protocol(self, tmp_path: Path) -> None:
        assert isinstance(LocalStorage(tmp_path), Storage)

    def test_pickle_serializable(self, tmp_path: Path) -> None:
        s = LocalStorage(tmp_path)
        restored = pickle.loads(pickle.dumps(s))
        assert restored.root == s.root


class TestAsyncStorage:
    def test_implements_async_protocol(self, tmp_path: Path) -> None:
        assert isinstance(LocalStorage(tmp_path), AsyncStorage)

    def test_aread_awrite(self, tmp_path: Path) -> None:
        async def _test():
            s = LocalStorage(tmp_path)
            await s.awrite("async.txt", b"async data")
            data = await s.aread("async.txt")
            assert data == b"async data"
            assert await s.aexists("async.txt")
            stat = await s.astat("async.txt")
            assert stat is not None and stat.size == 10

        asyncio.run(_test())

    def test_aappend(self, tmp_path: Path) -> None:
        async def _test():
            s = LocalStorage(tmp_path)
            await s.aappend("log.txt", b"line1\n")
            await s.aappend("log.txt", b"line2\n")
            data = await s.aread("log.txt")
            assert data == b"line1\nline2\n"

        asyncio.run(_test())

    def test_aread_range(self, tmp_path: Path) -> None:
        async def _test():
            s = LocalStorage(tmp_path)
            await s.awrite("data.bin", b"0123456789")
            assert await s.aread_range("data.bin", 3) == b"3456789"
            assert await s.aread_range("data.bin", 3, 4) == b"3456"

        asyncio.run(_test())

    def test_alist_dir(self, tmp_path: Path) -> None:
        async def _test():
            s = LocalStorage(tmp_path)
            await s.awrite("dir/a.txt", b"")
            await s.awrite("dir/b.txt", b"")
            items = await s.alist_dir("dir")
            assert "a.txt" in items
            assert "b.txt" in items

        asyncio.run(_test())

    def test_aremove(self, tmp_path: Path) -> None:
        async def _test():
            s = LocalStorage(tmp_path)
            await s.awrite("f.txt", b"data")
            assert await s.aexists("f.txt")
            await s.aremove("f.txt")
            assert not await s.aexists("f.txt")
            await s.aremove("nonexistent")  # no error

        asyncio.run(_test())

    def test_async_context_manager(self, tmp_path: Path) -> None:
        async def _test():
            async with LocalStorage(tmp_path) as s:
                await s.awrite("f.txt", b"ok")
                assert await s.aread("f.txt") == b"ok"

        asyncio.run(_test())


class TestStorageJoin:
    def test_normal(self) -> None:
        assert storage_join("a", "b", "c") == "a/b/c"

    def test_empty_prefix(self) -> None:
        assert storage_join("", "metrics.jsonl") == "metrics.jsonl"

    def test_all_empty(self) -> None:
        assert storage_join("", "", "") == ""

    def test_single(self) -> None:
        assert storage_join("file.txt") == "file.txt"

    def test_trailing_slash(self) -> None:
        assert storage_join("a/", "b") == "a/b"

    def test_leading_slash_inner(self) -> None:
        assert storage_join("a", "/b") == "a/b"

    def test_double_slash(self) -> None:
        assert storage_join("a/", "/b/") == "a/b"

    def test_dot_segment(self) -> None:
        assert storage_join("a", "./b") == "a/b"

    def test_double_dot(self) -> None:
        assert storage_join("a/b", "../c") == "a/c"
