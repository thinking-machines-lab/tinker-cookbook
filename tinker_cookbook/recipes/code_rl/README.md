# Replicating DeepCoder with Tinker

Competitive programming problems are a common testbed for RL with LLMs. The recent [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) blog post introduces a dataset and training pipeline for this purpose. This recipe demonstrates a similar setup using `Qwen3-4B-Instruct-2507`.

## Running This Demo

### Sandboxing

Sandboxing is essential for safely executing generated code during training and evaluation. We use [Sandbox Fusion](https://bytedance.github.io/SandboxFusion/) to sandbox code execution. You can start a local sandbox in Docker with:

```bash
docker run -it -p 8080:8080 \
    -v ${TINKER_COOKBOOK_ROOT}/tinker_cookbook/recipes/code_rl/sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml \
    volcengine/sandbox-fusion:server-20250609
```

Here, `${TINKER_COOKBOOK_ROOT}` is the absolute path to your local `tinker-cookbook` repository. The training script reads the sandbox endpoint from the `SANDBOX_URL` environment variable. By default it uses `http://localhost:8080/run_code`. Example:

```bash
export SANDBOX_URL=http://localhost:8080/run_code
```

If you prefer not to use Docker, you can set up the sandbox manually by following the instructions in the [Sandbox Fusion repository](https://github.com/bytedance/SandboxFusion?tab=readme-ov-file#installation).

### Example command

Train a `Qwen3-4B-Instruct-2507` model with:

```bash
python -m tinker_cookbook.recipes.code_rl.train \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    group_size=8 groups_per_batch=128 \
    learning_rate=4e-5 \
    lora_rank=32 \
    max_tokens=24576
```

After 100 steps of training, you can expect the following performance on **LiveCodeBench v6 (2025.02–2025.05)**:

| Model | Pass@1 | Pass@8 |
|-------|--------|--------|
| Qwen3-4B-Instruct-2507 (before training) | 33.8% | 44.3% |
| Qwen3-4B-Instruct-2507 (after 100 steps) | 42.7% | 55.0% |

[1] Luo, M., Tan, S., Huang, R., Patel, A., Ariyak, A., Wu, Q., Shi, X., Xin, R., Cai, C., Weber, M., Zhang, C., Li, L. E., Popa, R. A., & Stoica, I. (2025). DeepCoder: A fully open-source 14B coder at O3-mini level.
