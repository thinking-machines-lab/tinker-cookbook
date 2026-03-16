# Harbor RL

RL training on Harbor formatted tasks (e.g., Terminal Bench 2.0) with sandboxed code execution. An agent gets a bash tool inside a sandboxed container, attempts a task, and receives reward based on test results.

## HarborTask
Harbor offers a standardized format for SWE/Terminal-Bench style task.
Adhering to this allows seperation between task creation layer and evaluation/training harness layer.
We can download the harbor datasets through `uvx harbor datasets download terminal-bench@2.0`.
By default, the task will land in `~/.cache/harbor/tasks/` with the structure
```
~/.cache/harbor/tasks/
  └── <shortuuid(task_id)>/       # deterministic hash for deduplication
      └── <task_name>/            # human-readable task directory
          ├── environment/
          │   └── Dockerfile
          ├── tests/
          │   └── test.sh
          ├── instruction.md
          ├── task.toml
          └── solution/
```
To use harbor tasks for training or evaluation, we designed the following interface

```python
@dataclass(frozen=True)
class HarborTask:
    task_name: str
    instruction: str
    task_dir: Path      # must contain environment/Dockerfile and tests/test.sh
    config: dict[str, Any] = field(default_factory=dict)
```

You can load your downloaded tasks (e.g., 89 Terminal-Bench tasks) via `load_harbor_tasks()` in `launch_terminal_bench.py`:

```python
from tinker_cookbook.recipes.harbor_rl.launch_terminal_bench import load_harbor_tasks

tasks = load_harbor_tasks()  # reads from ~/.cache/harbor/tasks/ by default
print(f"Loaded {len(tasks)} tasks")
print(tasks[0].task_name, tasks[0].task_dir)
```
The training environment is implemented against this interface.
You can customize your own task as long as they conforms to the interface above.

## Sandbox Protocol and custom backends

### The Protocol

`tinker_cookbook.sandbox.sandbox_interface` defines `SandboxInterface`:

```python
@runtime_checkable
class SandboxInterface(Protocol):
    async def run_command(self, command: str, workdir: str | None = None, timeout: int = 60, max_output_bytes: int | None = None) -> SandboxResult: ...
    async def read_file(self, path: str, max_bytes: int | None = None, timeout: int = 60) -> SandboxResult: ...
    async def write_file(self, path: str, content: str | bytes, executable: bool = False, timeout: int = 60) -> SandboxResult: ...
    async def send_heartbeat(self) -> None: ...
    async def cleanup(self) -> None: ...
```

`ModalSandbox` implements this interface.

### SandboxFactory and injection

`harbor_env.py` defines a factory type and default:

```python
SandboxFactory = Callable[[modal.Image, int], Awaitable[SandboxInterface]]

async def default_sandbox_factory(image: modal.Image, timeout: int) -> SandboxInterface:
    return await ModalSandbox.create(image=image, timeout=timeout)
```

`cli_main()` accepts an optional `sandbox_factory` parameter. When `None`, it falls back to `default_sandbox_factory` (Modal). The factory flows through: `cli_main` -> `HarborDatasetBuilder` -> `HarborEnvGroupBuilder.make_envs()`.

## Running

First, download the Terminal-Bench tasks:

```bash
uvx harbor datasets download terminal-bench@2.0
```

Then launch training:

```bash
uv run python tinker_cookbook/recipes/harbor_rl/scripts/train_terminal_bench.py
```

## Evaluation

Evaluate a Tinker endpoint on Harbor datasets without training.
****
Download datasets:
```bash
uvx harbor datasets download terminal-bench@2.0 -o ~/.cache/harbor/tasks/terminal-bench-2.0
uvx harbor datasets download swebench-verified@1.0 -o ~/.cache/harbor/tasks/swebench-verified-1.0
```

Run evaluation:
```bash
uv run python tinker_cookbook/recipes/harbor_rl/scripts/eval_terminal_bench.py
```

Key parameters in `EvalConfig`: `max_turns` (10), `max_tokens` (2048), `temperature` (0.0).

`run_eval()` also accepts `sandbox_factory` for custom sandbox backends and `output_path` to control where results are written (default: `tinker_cookbook/recipes/harbor_rl/scripts/results/<timestamp>/`).
