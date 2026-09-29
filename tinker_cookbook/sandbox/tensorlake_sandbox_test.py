"""Offline tests for TensorlakeSandboxPool. TensorlakeSandbox.create is mocked."""

import asyncio
from typing import Self
from unittest import mock

import pytest

pytest.importorskip("tensorlake")

from tinker_cookbook.exceptions import SandboxError
from tinker_cookbook.sandbox import tensorlake_sandbox
from tinker_cookbook.sandbox.sandbox_interface import SandboxResult

_OK = SandboxResult(stdout="", stderr="", exit_code=0)


class _FakeSandbox:
    def __init__(self, live: set["_FakeSandbox"]) -> None:
        self._live = live
        live.add(self)

    async def cleanup(self) -> None:
        self._live.discard(self)

    async def write_file(self, *args: object, **kwargs: object) -> SandboxResult:
        return _OK

    async def run_command(self, *args: object, **kwargs: object) -> SandboxResult:
        return _OK

    async def checkpoint(self, include_memory: bool = True) -> str:
        return "snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup_command", "expected_created"),
    [
        (None, 2),  # max_concurrency creations are in flight
        ("pip install numpy", 1),  # only the template creation is in flight
    ],
)
async def test_terminate_stops_queued_and_in_flight_work(
    setup_command: str | None, expected_created: int
) -> None:
    live: set[_FakeSandbox] = set()
    created: list[_FakeSandbox] = []
    pending = 0
    release = asyncio.Event()

    async def fake_create(**kwargs: object) -> _FakeSandbox:
        nonlocal pending
        pending += 1
        await release.wait()
        sandbox = _FakeSandbox(live)
        created.append(sandbox)
        return sandbox

    with mock.patch.object(tensorlake_sandbox.TensorlakeSandbox, "create", fake_create):
        pool = tensorlake_sandbox.TensorlakeSandboxPool(
            max_concurrency=2, setup_command=setup_command
        )
        requests = [
            asyncio.create_task(pool.run_in_workdir({"a.py": "x"}, ["python", "a.py"]))
            for _ in range(6)
        ]
        while pending < expected_created:
            await asyncio.sleep(0)

        # terminate() waits for creations in progress, so let them finish.
        terminate = asyncio.create_task(pool.terminate())
        await asyncio.sleep(0)
        release.set()
        await terminate

        assert not live
        results = await asyncio.gather(*requests, return_exceptions=True)

    assert len(created) == expected_created
    assert not live
    assert all(isinstance(r, SandboxError) for r in results)


@pytest.mark.asyncio
async def test_run_after_terminate_raises() -> None:
    pool = tensorlake_sandbox.TensorlakeSandboxPool(max_concurrency=1)
    await pool.terminate()
    with pytest.raises(SandboxError, match="terminated"):
        await pool.run_in_workdir({}, ["true"])


@pytest.mark.asyncio
async def test_terminate_deletes_setup_snapshot() -> None:
    deleted: list[str] = []

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def delete_snapshot(self, snapshot_id: str) -> None:
            deleted.append(snapshot_id)

    live: set[_FakeSandbox] = set()

    async def fake_create(**kwargs: object) -> _FakeSandbox:
        return _FakeSandbox(live)

    with (
        mock.patch.object(tensorlake_sandbox.TensorlakeSandbox, "create", fake_create),
        mock.patch.object(tensorlake_sandbox, "AsyncSandboxClient", _FakeClient),
    ):
        pool = tensorlake_sandbox.TensorlakeSandboxPool(setup_command="pip install numpy")
        await pool.run_in_workdir({"a.py": "x"}, ["python", "a.py"])
        await pool.terminate()

    assert deleted == ["snapshot"]
    assert not live


class _FakeClient:
    deleted: list[str] = []

    def __init__(self, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.deleted.append(snapshot_id)


@pytest.mark.asyncio
async def test_cancelled_caller_does_not_leak_sandbox() -> None:
    live: set[_FakeSandbox] = set()
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_create(**kwargs: object) -> _FakeSandbox:
        # Like the SDK, allocate the remote sandbox before startup finishes.
        sandbox = _FakeSandbox(live)
        started.set()
        await release.wait()
        return sandbox

    with mock.patch.object(tensorlake_sandbox.TensorlakeSandbox, "create", fake_create):
        pool = tensorlake_sandbox.TensorlakeSandboxPool(max_concurrency=1)
        request = asyncio.create_task(pool.run_in_workdir({}, ["true"]))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

        # The creation continues after the cancel, and the pool reclaims it.
        release.set()
        await pool.terminate()

    assert not live


@pytest.mark.asyncio
async def test_terminate_waits_for_setup_checkpoint() -> None:
    live: set[_FakeSandbox] = set()
    checkpointing = asyncio.Event()
    release = asyncio.Event()

    class _SlowCheckpointSandbox(_FakeSandbox):
        async def checkpoint(self, include_memory: bool = True) -> str:
            checkpointing.set()
            await release.wait()
            return "snapshot"

    async def fake_create(**kwargs: object) -> _FakeSandbox:
        return _SlowCheckpointSandbox(live)

    _FakeClient.deleted = []
    with (
        mock.patch.object(tensorlake_sandbox.TensorlakeSandbox, "create", fake_create),
        mock.patch.object(tensorlake_sandbox, "AsyncSandboxClient", _FakeClient),
    ):
        pool = tensorlake_sandbox.TensorlakeSandboxPool(setup_command="pip install numpy")
        request = asyncio.create_task(pool.run_in_workdir({}, ["true"]))
        await checkpointing.wait()

        # Give terminate() time to finish if it does not wait for the checkpoint.
        terminate = asyncio.create_task(pool.terminate())
        for _ in range(20):
            await asyncio.sleep(0)
        assert not terminate.done()
        release.set()
        await terminate
        with pytest.raises(SandboxError):
            await request

    assert _FakeClient.deleted == ["snapshot"]
    assert not live


def test_registry_image_name_matches_harbor() -> None:
    # Harbor derives the same name, so its published images resolve here.
    assert (
        tensorlake_sandbox.registry_image_name("alexgshaw/chess-best-move:20251031")
        == "alexgshaw-chess-best-move-20251031-0a9b0f29"
    )
    assert tensorlake_sandbox.registry_image_name(
        "  org/app:1.0 "
    ) == tensorlake_sandbox.registry_image_name("org/app:1.0")
    with pytest.raises(ValueError):
        tensorlake_sandbox.registry_image_name("   ")


def test_resolve_registry_image_imports_on_miss() -> None:
    import tensorlake.image.sandbox_builder as builder

    name = tensorlake_sandbox.registry_image_name("org/app:1.0")
    with (
        mock.patch.object(builder, "find_sandbox_image_by_name", return_value=None) as find,
        mock.patch.object(builder, "import_sandbox_image") as import_image,
    ):
        assert tensorlake_sandbox.resolve_registry_image("org/app:1.0") == name
        find.assert_called_once_with(name)
        import_image.assert_called_once_with("org/app:1.0", registered_name=name)

    with (
        mock.patch.object(builder, "find_sandbox_image_by_name", return_value=None),
        mock.patch.object(builder, "import_sandbox_image") as import_image,
        pytest.raises(LookupError),
    ):
        tensorlake_sandbox.resolve_registry_image("org/app:1.0", import_if_missing=False)
    import_image.assert_not_called()


def test_resolve_registry_image_skips_import_on_hit() -> None:
    import tensorlake.image.sandbox_builder as builder

    with (
        mock.patch.object(builder, "find_sandbox_image_by_name", return_value={"name": "x"}),
        mock.patch.object(builder, "import_sandbox_image") as import_image,
    ):
        tensorlake_sandbox.resolve_registry_image("org/app:1.0")
    import_image.assert_not_called()


def test_structured_lifecycle_error_is_terminated() -> None:
    from tensorlake.sandbox import RemoteAPIError

    err = RemoteAPIError(409, '{"sandbox_id": "abc", "status": "terminated", "reason": "Timeout"}')
    assert str(err) == "API error (status 409): Sandbox abc terminated (Timeout)"
    assert tensorlake_sandbox._is_sandbox_terminated(err)
    assert not tensorlake_sandbox._is_sandbox_terminated(RemoteAPIError(409, "conflict"))
