# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently two available backends: SandboxFusion for local execution and Modal for cloud execution.

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

# Single sandbox
sandbox = ModalSandbox()
exit_code, stdout, stderr = await sandbox.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)

# Pool for concurrent execution (recommended for RL workloads)
pool = ModalSandboxPool(pool_size=32)
exit_code, stdout, stderr = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)
```

Environment variables:

- `MODAL_POOL_SIZE`: Number of sandboxes in the pool (default: 32)
