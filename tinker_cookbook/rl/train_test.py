"""Unit tests for rl.train helpers that do not require the Tinker API."""

from pathlib import Path

from tinker_cookbook.rl.train import _get_logtree_scope
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore
from tinker_cookbook.utils import logtree


def test_get_logtree_scope_writes_json_and_html(tmp_path: Path) -> None:
    """The scope routes both the logtree JSON and HTML through the store."""
    store = TrainingRunStore(LocalStorage(tmp_path))
    with _get_logtree_scope(
        num_groups_to_log=2,
        f_name="train",
        scope_name="RL Iteration 0",
        iteration=0,
        store=store,
    ):
        logtree.log_text("hello-from-rollout")

    # JSON logtree round-trips through the store.
    data = store.read_logtree(0, base_name="train")
    assert data is not None
    assert data["title"] == "RL Iteration 0"

    # HTML logtree is a full document containing the logged content.
    html = store.storage.read("iteration_000000/train_logtree.html").decode("utf-8")
    assert html.startswith("<!doctype html>")
    assert "hello-from-rollout" in html


def test_get_logtree_scope_uses_base_name(tmp_path: Path) -> None:
    """f_name controls the artifact base name (e.g. per-evaluator logtrees)."""
    store = TrainingRunStore(LocalStorage(tmp_path))
    with _get_logtree_scope(
        num_groups_to_log=1,
        f_name="eval_gsm8k",
        scope_name="Eval gsm8k",
        iteration=3,
        store=store,
    ):
        logtree.log_text("eval-rollout")

    assert store.read_logtree(3, base_name="eval_gsm8k") is not None
    assert store.storage.exists("iteration_000003/eval_gsm8k_logtree.html")


def test_get_logtree_scope_skips_when_num_groups_zero(tmp_path: Path) -> None:
    """num_groups_to_log <= 0 disables logtree capture and writes nothing."""
    store = TrainingRunStore(LocalStorage(tmp_path))
    with _get_logtree_scope(
        num_groups_to_log=0,
        f_name="train",
        scope_name="RL Iteration 0",
        iteration=0,
        store=store,
    ):
        logtree.log_text("should-not-be-written")

    assert store.read_logtree(0, base_name="train") is None
    assert not store.storage.exists("iteration_000000/train_logtree.html")


def test_get_logtree_scope_no_store_is_noop() -> None:
    """With no store the scope is a safe no-op (logging gracefully disabled)."""
    with _get_logtree_scope(
        num_groups_to_log=2,
        f_name="train",
        scope_name="RL Iteration 0",
        iteration=0,
        store=None,
    ):
        logtree.log_text("no-store")  # no active trace -> graceful no-op
