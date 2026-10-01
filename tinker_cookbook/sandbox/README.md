# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently three available backends: SandboxFusion for local execution, and Modal or Daytona for cloud execution.

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

### Daytona (cloud)

[Daytona Sandboxes](https://www.daytona.io) provide cloud-based isolated execution environments. Requires `DAYTONA_API_KEY` (or `DAYTONA_JWT_TOKEN` + `DAYTONA_ORGANIZATION_ID`).

Install the extra:

```bash
uv pip install 'tinker-cookbook[daytona] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

Example usage (stateful, implements `SandboxInterface`):

```python
from tinker_cookbook.sandbox.daytona_sandbox import DaytonaSandbox

sandbox = await DaytonaSandbox.create()
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python /workspace/code.py", workdir="/workspace")
print(result.stdout)
await sandbox.cleanup()
```

Example usage (stateless grading, drop-in for `code_rl`):

```python
from tinker_cookbook.sandbox.daytona_sandbox import run_code_in_daytona

success, response = await run_code_in_daytona(
    code="print(2 + 2)",
    files={"data.txt": "some content"},
    timeout=30,
)
```

Harbor-style per-task Dockerfiles (drop-in for `harbor_rl`):

```python
from tinker_cookbook.sandbox.daytona_sandbox import daytona_sandbox_factory

# Pass to cli_main(sandbox_factory=daytona_sandbox_factory) or the
# HarborDatasetBuilder / HarborEnvGroupBuilder constructors.
```

Optional: set `DAYTONA_SNAPSHOT` to a pre-created snapshot name to skip image builds. Not required — image builds are cached automatically across sandboxes.
