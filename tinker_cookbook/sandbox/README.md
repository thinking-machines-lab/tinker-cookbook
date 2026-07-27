# Sandboxing

This directory contains code execution backends for sandboxed evaluation (e.g., grading code in RL environments).

There are currently three available backends: SandboxFusion for local execution, and Modal and Hyperbrowser for cloud execution.

Recipes and evals that need a cloud sandbox default to Modal. Set `TINKER_SANDBOX_BACKEND=hyperbrowser` (or pass `--sandbox_backend=hyperbrowser` / `BenchmarkConfig(sandbox_backend=...)`) to switch without any code changes.

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

### Hyperbrowser (cloud)

[Hyperbrowser Sandboxes](https://hyperbrowser.ai/docs/sandboxes/introduction) provide cloud-based isolated execution environments. Install with `pip install 'tinker-cookbook[hyperbrowser]'` and authenticate by exporting `HYPERBROWSER_API_KEY`.

Example usage:

```python
from tinker_cookbook.sandbox.hyperbrowser_sandbox import (
    HyperbrowserImage,
    HyperbrowserSandbox,
    HyperbrowserSandboxPool,
)

# Single sandbox (conforms to SandboxInterface)
sandbox = await HyperbrowserSandbox.create()
await sandbox.write_file("/workspace/code.py", "print('hello')")
result = await sandbox.run_command("python /workspace/code.py", workdir="/workspace")
print(result.stdout)
await sandbox.cleanup()

# Pool for concurrent execution (recommended for RL workloads)
pool = HyperbrowserSandboxPool(pool_size=32)
result = await pool.run_in_workdir(
    files={"code.py": "print('hello')"},
    command=["python", "code.py"],
)
print(result.stdout)
```

Environment variables:

- `HYPERBROWSER_API_KEY`: API key (required)
- `HYPERBROWSER_POOL_SIZE`: Number of sandboxes in the pool (default: 32)
- `HYPERBROWSER_CREATION_RATE_LIMIT`: Max sandboxes created per second (default: 4)
- `HYPERBROWSER_REGION`: Sandbox region, e.g. `us-east` (default: account default)
- `HYPERBROWSER_SANDBOX_IMAGE`: Default base image (default: `python`)
- `HYPERBROWSER_RUN_AS`: User to run commands as (default: `root`)
- `HYPERBROWSER_IMAGE_CACHE_DIR`: Where built-image markers are recorded (default: `~/.cache/tinker-cookbook/hyperbrowser`)

#### Images

`HyperbrowserImage` mirrors the `modal.Image` surface:

```python
HyperbrowserImage.base("python").apt_install("git").pip_install("numpy")
HyperbrowserImage.from_registry("python:3.12-slim")
HyperbrowserImage.from_dockerfile("environment/Dockerfile", "environment/")
HyperbrowserImage.from_hyperbrowser("an-image-you-already-uploaded")
```

Base images are `python`, `node`, `node-chromium`, `claude-code`, `codex`, and `openclaw` — all Ubuntu 24.04.

One difference from Modal is worth knowing. Modal builds Dockerfiles remotely; Hyperbrowser has no server-side builder, so `from_registry` and `from_dockerfile` run `docker buildx build` locally and upload the exported root filesystem. To keep that off the hot path, images are named after a hash of their contents, and a spec whose image is already uploaded to your account is launched directly — so a given Dockerfile is built once ever, not once per rollout. Layers added to a `base(...)` image skip Docker entirely by running as sandbox startup commands, which is why the code-execution benchmarks and `code_rl` grading need no local Docker.

To prebuild an image on a Docker-capable machine and use it from one without Docker:

```python
image = HyperbrowserImage.from_dockerfile("environment/Dockerfile", "environment/")
resolved = await image.resolve(await get_client())  # builds and uploads
print(resolved.image_name)  # pass to HyperbrowserImage.from_hyperbrowser(...) elsewhere
```

Sandbox lifetime is fixed at creation (Hyperbrowser has no extend API), so size `timeout` for the whole episode. `send_heartbeat` detects a dead sandbox but cannot postpone its expiry.
