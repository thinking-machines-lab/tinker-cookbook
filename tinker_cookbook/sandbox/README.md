# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently three available backends: SandboxFusion for local execution, Modal for cloud execution, and Fystash for warm Firecracker rooms.

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

### Fystash (cloud Firecracker rooms)

[Fystash](https://fystash.ai) provides warm Firecracker rooms as a `SandboxInterface` backend (`SandboxBackend.FYSTASH`). No optional extra is required (stdlib HTTP).

```python
from tinker_cookbook.sandbox import FystashSandbox, FystashSandboxPool

sandbox = await FystashSandbox.create(timeout=600)
await sandbox.write_file("/tmp/hi.py", "print('hello')")
result = await sandbox.run_command("python /tmp/hi.py")
print(result.stdout)
await sandbox.cleanup()

# Optional capacity-backed pool (harbor_rl / code_rl)
pool = FystashSandboxPool(pool_size=8)
await pool.start()
result = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)
await pool.terminate()
```

Harbor RL selection:

```bash
export FYSTASH_API_KEY=key-…          # https://fystash.ai/signup
# sandbox_backend=fystash on train/eval, or:
export TINKER_SANDBOX_BACKEND=fystash
```

Environment variables:

- `FYSTASH_API_KEY`: Required
- `FYSTASH_API`: API base (default `https://api.fystash.ai`)
- `FYSTASH_TEMPLATE_ID`: Template id (default `default`; ignored when `FYSTASH_DOCKER_IMAGE` is set)
- `FYSTASH_DOCKER_IMAGE`: Optional DinD pull only — does **not** build Harbor task Dockerfiles
- `FYSTASH_POOL_SIZE`: When `>0`, harbor_rl factory acquires from `FystashSandboxPool` (default cold create when unset/`0`)
- `FYSTASH_POOL_CREATE_RATE`: Concurrent replenish (default `4`)
- `FYSTASH_CAPACITY_TTL_S`: Capacity reservation TTL (default `3600`)

**Honesty:** Modal's harbor factory builds `environment/Dockerfile`. Fystash does not — template or prebuilt image pull only. See [`recipes/harbor_rl/README.md`](../recipes/harbor_rl/README.md) for the full Fystash subsection.
