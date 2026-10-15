# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently three available backends: SandboxFusion for local execution, and Modal or Sail for cloud execution.

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

### Sail (cloud)

[Sailboxes](https://docs.sailresearch.com/sailbox-sdk) are Sail's cloud sandboxes. Requires an API key: `export SAIL_API_KEY=...`

Example usage:

```python
import sail

from tinker_cookbook.sandbox.sail_sandbox import SailSandbox, SailSandboxPool

image = sail.Image.from_registry("python:3.12-slim")

# Single sandbox (conforms to SandboxInterface)
sandbox = await SailSandbox.create(image=image)
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python /workspace/code.py", workdir="/workspace")
print(result.stdout)
await sandbox.cleanup()

# Pool for concurrent execution (recommended for RL workloads)
pool = SailSandboxPool(pool_size=32, image=image)
result = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)
print(result.stdout)
```

`SailSandbox.create` defaults to the `s` size (1 vCPU), which has the fastest cold starts. Pass `size="m"` or `size="l"` for more CPUs, and `memory_limit_gib` or `disk_limit_gib` to raise a ceiling; `sail.Sailbox.size_for(...)` picks the smallest size that covers a CPU, memory, and disk requirement. Pass `egress_policy`, such as `sail.EgressPolicy.no_network()`, to restrict the Sailbox's outbound network access. Commands run as root, as they do in a Modal sandbox, whatever `USER` the image sets. A Sailbox has no lifetime cap unless you pass `max_lifetime_seconds`; Sail puts an idle Sailbox to sleep, unbilled, and wakes it on the next command. `SailSandboxPool` terminates its Sailboxes when you await `terminate()`, and any still alive when the process exits. If Sail rejects sandbox creation with an error that a retry cannot fix (for example a bad `SAIL_API_KEY` or an image that fails to build), the pool stops creating Sailboxes, hands out the warm ones it has, and then `SailSandboxPool.run_in_workdir` raises `SandboxError` instead of waiting.

Environment variables:

- `SAIL_POOL_SIZE`: Number of sandboxes in the pool (default: 32)
- `SAIL_CREATION_RATE_LIMIT`: Max sandboxes created per second (default: 4)
