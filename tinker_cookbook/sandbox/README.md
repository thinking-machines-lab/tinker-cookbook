# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently three available backends: SandboxFusion for local execution, and Modal and Tensorlake for cloud execution.

## Backends

### SandboxFusion (local Docker)

[Sandbox Fusion](https://bytedance.github.io/SandboxFusion/) is a Docker-based code execution sandbox. Start a local sandbox in Docker with:

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

For RL workloads, you may want higher concurrency. See [`recipes/code_rl/sandbox_config/local.yaml`](../recipes/code_rl/sandbox_config/local.yaml) for an example configuration that can be mounted with `-v`, and see [`recipes/code_rl/README.md`](../recipes/code_rl/README.md) for instructions on using it.

If you prefer not to use Docker, see the [Sandbox Fusion repository](https://github.com/bytedance/SandboxFusion?tab=readme-ov-file#installation) for manual setup.

Example usage:

```python
from tinker_cookbook.sandbox import SandboxFusionClient

client = SandboxFusionClient()
success, response = await client.run(
    code="print('hello')",
    files={"data.txt": "some content"},
    timeout=30,
)
await client.close()
```

Environment variables:

- `SANDBOX_URL`: Endpoint URL (default: `http://localhost:8080/run_code`)
- `SANDBOX_MAX_CONCURRENCY`: Max concurrent requests (default: 4)

### Modal (cloud)

[Modal Sandboxes](https://modal.com/products/sandboxes) provide cloud-based isolated execution environments. Requires authentication with: `modal token new`

Example usage:

```python
from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox, ModalSandboxPool

# Single sandbox (conforms to SandboxInterface)
sandbox = await ModalSandbox.create()
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python /workspace/code.py", workdir="/workspace")
print(result.stdout)
await sandbox.cleanup()

# Pool for concurrent execution (recommended for RL workloads)
pool = ModalSandboxPool(pool_size=32)
result = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)
print(result.stdout)
```

Environment variables:

- `MODAL_POOL_SIZE`: Number of sandboxes in the pool (default: 32)

### Tensorlake (cloud)

[Tensorlake Sandboxes](https://docs.tensorlake.ai/sandboxes/introduction) are cloud microVMs that start in under a second and support snapshot/restore. Requires an API key: set `TENSORLAKE_API_KEY`.

This backend adds three things the other backends do not have:

- **Snapshot and restore.** Prepare one sandbox (install packages, start a service, load data), call `checkpoint()`, and start every rollout in a group from that snapshot. Each rollout gets an identical start state without repeating the setup.
- **Long-running services.** `start_process()` runs a server in the background and `open_port()` returns a URL for it. This is what agent environments need when the model talks to an app instead of running one-shot scripts.
- **Prebuilt images.** `resolve_registry_image()` finds an image in your project or in Tensorlake's public images. The Harbor recipe uses this to start Terminal-Bench tasks without building their Dockerfiles.

It also lets you restrict outbound network access per sandbox.

Install with: `uv pip install 'tinker-cookbook[tensorlake]'`

Example usage:

```python
from tinker_cookbook.sandbox.tensorlake_sandbox import (
    TensorlakeSandbox,
    TensorlakeSandboxPool,
    build_image_from_dockerfile,
    resolve_registry_image,
)

# Single sandbox (conforms to SandboxInterface)
sandbox = await TensorlakeSandbox.create()
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python3 /workspace/code.py", workdir="/workspace")
print(result.stdout)
await sandbox.cleanup()

# Pool for one-shot runs (same API as ModalSandboxPool). Each call gets a fresh
# sandbox; setup_command runs once and is snapshotted.
pool = TensorlakeSandboxPool(
    max_concurrency=32, setup_command="pip install --break-system-packages numpy"
)
result = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python3", "code.py"],
)
```

Agent environments often run a service inside the sandbox and need a copy of the start state for each rollout in a group:

```python
# Use an image from a container registry. Finds it in your project or in
# Tensorlake's public images, and imports it on a miss (blocks while importing).
image = await asyncio.to_thread(resolve_registry_image, "alexgshaw/chess-best-move:20251031")

# Or build an image from a Dockerfile (cached by content; blocks while building)
image = await asyncio.to_thread(build_image_from_dockerfile, "env/Dockerfile", context_dir=".")

sandbox = await TensorlakeSandbox.create(image=image, timeout=3600)
await sandbox.start_process("uvicorn app:app --port 8080", workdir="/app")
base_url = await sandbox.open_port(8080)  # http://127.0.0.1:<port>, via an authenticated tunnel

# Snapshot the prepared state, then start one sandbox per rollout from it
snapshot_id = await sandbox.checkpoint()
group = await asyncio.gather(*(TensorlakeSandbox.create(snapshot_id=snapshot_id) for _ in range(8)))
```

Commands run as root by default, like Modal. Pass `user=None` to use the image's `USER`. Use `allow_internet_access=False` or `allow_out=[...]` to limit outbound network access.

Environment variables:

- `TENSORLAKE_API_KEY`: API key (required)
- `TENSORLAKE_API_URL`: API endpoint (default: `https://api.tensorlake.ai`)
- `TENSORLAKE_MAX_CONCURRENCY`: Max concurrent sandboxes in `TensorlakeSandboxPool` (default: 32)
