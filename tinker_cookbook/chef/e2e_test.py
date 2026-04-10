"""End-to-end UI tests for Tinker Chef using Playwright.

These tests launch the full FastAPI server with fixture data and use
Playwright to verify the frontend renders correctly.

Run with: pytest tinker_cookbook/chef/e2e_test.py --headed  (to watch)
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
Page = playwright.Page
expect = playwright.expect

# Port for the test server (avoid conflicts with dev server)
TEST_PORT = 8199
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a comprehensive fixture directory for e2e tests."""
    root = tmp_path_factory.mktemp("e2e_data")

    # Create a training run with realistic data
    run_dir = root / "math_rl_run"
    run_dir.mkdir()

    # Config
    config = {
        "model_name": "Llama-3.1-8B",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "n_batches": 50,
        "lora_rank": 16,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Metrics (50 steps)
    metrics_lines = []
    for step in range(50):
        metrics_lines.append(json.dumps({
            "step": step,
            "train_mean_nll": 2.5 - step * 0.03,
            "env/all/reward/total": min(1.0, step * 0.02),
            "env/all/turns_per_episode": 3 + step * 0.05,
            "optim/kl_sample_train_v1": 0.01 + step * 0.001,
            "time/forward_backward": 1.0 + (step % 5) * 0.1,
            "time/policy_sample": 0.5 + (step % 3) * 0.05,
        }))
    (run_dir / "metrics.jsonl").write_text("\n".join(metrics_lines) + "\n")

    # Checkpoints
    for ckpt_step in [10, 20, 30, 40]:
        ckpt = {
            "state_path": f"tinker:///ckpt/{ckpt_step:06d}",
            "name": f"{ckpt_step:06d}",
            "kind": "both",
            "timestamp": 1700000000.0 + ckpt_step * 60,
            "loop_state": {"epoch": 0, "batch": ckpt_step},
        }
        with open(run_dir / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps(ckpt) + "\n")

    # Timing spans with concurrency
    timing_lines = []
    for step in range(5):
        timing_lines.append(json.dumps({
            "step": step,
            "spans": [
                {"name": "forward_backward", "duration": 1.5, "wall_start": 0.0, "wall_end": 1.5},
                {"name": "policy_sample", "duration": 0.8, "wall_start": 0.0, "wall_end": 0.8},
                {"name": "env_step", "duration": 0.6, "wall_start": 0.2, "wall_end": 0.8},
                {"name": "optim_step", "duration": 0.5, "wall_start": 1.5, "wall_end": 2.0},
            ],
        }))
    (run_dir / "timing_spans.jsonl").write_text("\n".join(timing_lines) + "\n")

    # Iteration directories with rollouts and logtree
    for iteration in [0, 10, 20]:
        iter_dir = run_dir / f"iteration_{iteration:06d}"
        iter_dir.mkdir()

        rollouts = []
        for group_idx in range(2):
            for traj_idx in range(3):
                reward = 0.0 if iteration == 0 else (group_idx + traj_idx) * 0.2 + iteration * 0.01
                rollouts.append({
                    "schema_version": 1, "split": "train", "iteration": iteration,
                    "group_idx": group_idx, "traj_idx": traj_idx,
                    "tags": ["math", "gsm8k"] if group_idx == 0 else ["code", "mbpp"],
                    "sampling_client_step": iteration,
                    "total_reward": reward,
                    "final_reward": reward * 0.5,
                    "trajectory_metrics": {},
                    "steps": [
                        {"step_idx": 0, "ob_len": 128, "ac_len": 45, "reward": 0.0,
                         "episode_done": False, "metrics": {"step_time": 0.5}, "logs": {}},
                        {"step_idx": 1, "ob_len": 200, "ac_len": 60, "reward": reward,
                         "episode_done": True, "metrics": {"step_time": 0.3}, "logs": {}},
                    ],
                    "final_ob_len": 350,
                })
        (iter_dir / "train_rollout_summaries.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rollouts) + "\n"
        )

        logtree = {
            "title": f"RL Iteration {iteration}",
            "started_at": "2024-04-04T12:34:56",
            "root": {
                "tag": "div", "attrs": {"class": "lt-root"},
                "children": [],
                "data": {"type": "conversation", "messages": [
                    {"role": "user", "content": f"Solve: What is {iteration}*2?"},
                    {"role": "assistant", "content": f"The answer is {iteration*2}."},
                ]},
            },
        }
        (iter_dir / "train_logtree.json").write_text(json.dumps(logtree))

    return root


@pytest.fixture(scope="session")
def server(fixture_dir: Path):
    """Start the Tinker Chef server as a subprocess."""
    # Ensure the worktree root is on PYTHONPATH so the subprocess can find tinker_cookbook
    worktree_root = Path(__file__).resolve().parent.parent.parent
    env = {**os.environ, "PYTHONPATH": str(worktree_root)}

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "tinker_cookbook.chef.cli",
            "serve", str(fixture_dir),
            "--port", str(TEST_PORT),
            "--log-level", "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for server to start
    for _ in range(30):
        try:
            import urllib.request
            urllib.request.urlopen(f"{BASE_URL}/api/runs", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise RuntimeError(
            f"Server failed to start.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    yield proc

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def browser_context_args():
    """Configure Playwright browser context."""
    return {"base_url": BASE_URL}


# ── Page tests ────────────────────────────────────────────────────────


class TestDashboard:
    def test_shows_dashboard(self, page: Page, server) -> None:
        page.goto("/")
        expect(page.get_by_role("heading", name="Dashboard")).to_be_visible()
        expect(page.locator("text=math_rl_run").first).to_be_visible()

    def test_shows_model_name(self, page: Page, server) -> None:
        page.goto("/")
        expect(page.locator("text=Llama-3.1-8B").first).to_be_visible()

    def test_navigate_to_run(self, page: Page, server) -> None:
        page.goto("/")
        page.locator("text=math_rl_run").first.click()
        page.wait_for_url("**/runs/math_rl_run", timeout=10000)
        expect(page.locator("text=Metrics")).to_be_visible()


class TestRunDetailPage:
    def test_shows_tabs(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run")
        expect(page.get_by_role("button", name="Metrics")).to_be_visible()
        expect(page.get_by_role("button", name="Rollouts")).to_be_visible()
        expect(page.get_by_role("button", name="Config")).to_be_visible()

    def test_stat_cards_visible(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run")
        # Stat cards answer "is training working?" at a glance
        expect(page.locator("text=Reward").first).to_be_visible(timeout=10000)

    def test_metrics_default_tab(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run")
        # Metrics is the default tab — charts should render
        page.wait_for_selector(".recharts-wrapper", timeout=10000)
        expect(page.locator(".recharts-wrapper").first).to_be_visible()

    def test_rollouts_tab(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run")
        page.locator("button", has_text="Rollouts").click()
        page.wait_for_timeout(1000)  # Wait for tab to render
        expect(page.locator("text=Iteration").first).to_be_visible(timeout=10000)

    def test_timing_tab(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run")
        page.locator("button", has_text="Timing").click()
        page.wait_for_timeout(1000)
        expect(page.locator("text=Wall Time per Step").first).to_be_visible(timeout=10000)

    def test_config_tab(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run")
        page.locator("button", has_text="Config").click()
        expect(page.locator("text=learning_rate")).to_be_visible(timeout=10000)

    def test_rollout_navigation(self, page: Page, server) -> None:
        # Navigate directly to rollout detail to avoid tab timing issues
        page.goto("/runs/math_rl_run/iterations/10/rollouts/0/0")
        expect(page.locator("text=Total Reward")).to_be_visible(timeout=10000)
        # Verify prev/next navigation exists
        expect(page.locator("text=Prev")).to_be_visible()
        expect(page.locator("text=Next")).to_be_visible()


class TestRolloutDetailPage:
    def test_shows_rollout_info(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run/iterations/10/rollouts/0/0")
        # Should show rollout metadata
        expect(page.locator("text=Rollout")).to_be_visible(timeout=10000)
        expect(page.locator("text=math").first).to_be_visible()
        # Should show total reward
        expect(page.locator("text=Total Reward")).to_be_visible()

    def test_shows_reward(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run/iterations/10/rollouts/0/0")
        # Should show reward information
        expect(page.locator("text=Total Reward")).to_be_visible()

    def test_breadcrumb_navigation(self, page: Page, server) -> None:
        page.goto("/runs/math_rl_run/iterations/10/rollouts/0/0")
        # Should have breadcrumb back to run
        page.locator("a", has_text="math_rl_run").click()
        page.wait_for_url("**/runs/math_rl_run")
