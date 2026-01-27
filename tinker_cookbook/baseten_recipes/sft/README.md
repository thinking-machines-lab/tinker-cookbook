# SFT Recipe - Supervised Fine-Tuning with Tinker

Complete pipeline for supervised fine-tuning (SFT) of language models using LoRA, with optional model merging and HuggingFace Hub upload.

**Configuration-driven**: Uses YAML config files and `.env` for API keys.

## Overview

This recipe provides a production-ready SFT pipeline that:

1. **Trains** any model from the Tinker model lineup using LoRA
2. **Reuses** robust components from `tinker_cookbook/supervised/`
3. **Merges** LoRA adapters into base model weights
4. **Uploads** merged model to HuggingFace Hub with auto-generated model card

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies with uv
uv pip install pyyaml python-dotenv

# Or with regular pip
pip install pyyaml python-dotenv

# Create .env file with API keys (copy from root .env.example)
cat > .env << EOF
TINKER_API_KEY=your_tinker_api_key_here
HF_TOKEN=your_huggingface_token_here
EOF
```

### 2. Create Configuration

```bash
cd tinker_cookbook/baseten_recipes/sft

# Option 1: Use a pre-made config from configs/ folder
# Example: configs/phare.yaml for phare-silver-messages dataset
# Edit the config as needed

# Option 2: Copy and edit the example config
cp configs/example.yaml configs/my_config.yaml
# Edit configs/my_config.yaml with your training settings
```

### 3. Run Training

```bash
# With uv (recommended)
uv run python -m tinker_cookbook.baseten_recipes.sft.train configs/phare.yaml

# Or without uv
python -m tinker_cookbook.baseten_recipes.sft.train configs/phare.yaml
```

## Configuration File

The `config.yaml` file controls all training parameters. See `config.example.yaml` for a full example:

### Model Configuration

```yaml
model:
  name: "meta-llama/Llama-3.1-8B"  # HuggingFace model name
  lora_rank: 32                     # LoRA rank (higher = more capacity)
  renderer_name: null               # Auto-detect if null
```

### Dataset Configuration

```yaml
dataset:
  name: "no_robots"                 # "no_robots", "tulu3", or path to .jsonl
  max_length: 16384                 # Maximum sequence length
  batch_size: 256                   # Training batch size
  train_on_what: null               # Training target (see options below)
  num_examples: null                # Limit examples (null = use all)
```

**train_on_what options:**
- `null` (default): Uses dataset-specific default
- `"ALL_ASSISTANT_MESSAGES"`: Train on all assistant responses
- `"LAST_ASSISTANT_MESSAGE"`: Train on only final assistant response
- `"ALL_TEXT"`: Train on entire conversation

### Training Parameters

```yaml
training:
  learning_rate: null               # Auto-compute if null (recommended)
  lr_schedule: "linear"             # "linear", "cosine", or "constant"
  num_epochs: 1                     # Number of training epochs
```

### Checkpointing

```yaml
checkpointing:
  save_every: 20                    # Save checkpoint every N steps
  eval_every: 20                    # Run evaluation every N steps
  ttl_seconds: 604800               # Checkpoint TTL (7 days)
```

### Post-Training Pipeline

```yaml
post_training:
  enabled: false                    # Set to true to merge and upload
  hf_repo_id: null                  # HuggingFace repo (e.g., "username/model-name")
  hf_private: true                  # Make HuggingFace repo private
  download_dir: "/tmp/tinker-merged-models"
```

### Logging

```yaml
logging:
  log_path: null                    # Custom log path (auto-generated if null)
  wandb_project: null               # Weights & Biases project name
  wandb_name: null                  # W&B run name (auto-generated if null)
```

### Infrastructure

```yaml
infrastructure:
  base_url: null                    # Tinker API base URL
  load_checkpoint_path: null        # Resume from checkpoint
  behavior_if_log_dir_exists: "ask" # "ask", "overwrite", or "fail"
```

## Supported Models

All models from the [Tinker model lineup](https://tinker-docs.thinkingmachines.ai/model-lineup):

- **Llama**: `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-70B`
- **Qwen**: `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B` (MoE), `Qwen/Qwen3-VL-30B-A3B-Instruct` (vision)
- **GPT-OSS**: `openai/gpt-oss-20b`

## Supported Datasets

### Built-in Datasets

- **`no_robots`**: HuggingFaceH4/no_robots - 10k high-quality conversations
- **`tulu3`**: allenai/tulu-3-sft-mixture - Large-scale SFT mixture

### Any HuggingFace Dataset with 'messages' Column

Any HuggingFace dataset that has a `messages` column will work automatically. Just specify the full dataset name:

- `parsed/phare-silver-messages`
- `your-org/your-dataset`
- etc.

The dataset must have a `messages` column with the standard format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Custom JSONL Format

Provide a `.jsonl` file where each line is:

```json
{
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

Vision models also support image inputs:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "https://example.com/image.jpg"},
        {"type": "text", "text": "What's in this image?"}
      ]
    },
    {"role": "assistant", "content": "The image shows..."}
  ]
}
```

## Usage Examples

### Example 1: Basic Training on NoRobots

```yaml
# config.yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  lora_rank: 32

dataset:
  name: "no_robots"
  batch_size: 256
  num_examples: null  # Use full dataset

training:
  learning_rate: null  # Auto-compute
  num_epochs: 1
```

```bash
uv run python -m tinker_cookbook.baseten_recipes.sft.train config.yaml
```

### Example 2: Quick Experiment (Limited Examples)

```yaml
# config.yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  lora_rank: 32

dataset:
  name: "tulu3"
  batch_size: 128
  num_examples: 1000  # Only use 1000 examples

training:
  num_epochs: 1
```

### Example 3: High-Rank LoRA Training

```yaml
# config.yaml
model:
  name: "Qwen/Qwen3-8B"
  lora_rank: 128  # Higher rank for more capacity

dataset:
  name: "tulu3"
  batch_size: 128

training:
  learning_rate: 3e-4
  lr_schedule: "cosine"
  num_epochs: 2
```

### Example 4: Training with HuggingFace Upload

```yaml
# config.yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  lora_rank: 32

dataset:
  name: "no_robots"
  batch_size: 256

training:
  num_epochs: 1

post_training:
  enabled: true
  hf_repo_id: "myusername/llama-norobots-sft"
  hf_private: false  # Public repo

logging:
  wandb_project: "my-sft-experiments"
```

```bash
# Make sure HF_TOKEN is in .env
uv run python -m tinker_cookbook.baseten_recipes.sft.train config.yaml
```

### Example 5: Custom Dataset

```yaml
# config.yaml
model:
  name: "Qwen/Qwen3-8B"
  lora_rank: 64

dataset:
  name: "./my_conversations.jsonl"  # Path to custom data
  batch_size: 128
  max_length: 8192
  train_on_what: "ALL_ASSISTANT_MESSAGES"

training:
  num_epochs: 3
```

### Example 6: Vision Model Fine-Tuning

```yaml
# config.yaml
model:
  name: "Qwen/Qwen3-VL-30B-A3B-Instruct"
  lora_rank: 32

dataset:
  name: "./my_vision_data.jsonl"
  batch_size: 64
  max_length: 8192

training:
  num_epochs: 1
```

## Post-Training Pipeline

The optional post-training pipeline automates:

1. **Download checkpoint** from Tinker using the SDK
2. **Merge LoRA adapter** with base model
3. **Generate model card** with training metadata
4. **Upload to HuggingFace Hub**

### Requirements

- `HF_TOKEN` in `.env` file
- Sufficient disk space (see table below)
- GPU recommended for faster merge (CPU fallback available)

**Disk Space Requirements:**
| Model Size | Space Needed |
|------------|--------------|
| 8B         | ~35GB        |
| 30B        | ~125GB       |
| 70B        | ~280GB       |

### Option 1: Upload During Training

Enable in your config to run after training:

```yaml
post_training:
  enabled: true
  hf_repo_id: "myusername/my-model"
  hf_private: true
  download_dir: "/tmp/tinker-merged-models"
```

The pipeline will run automatically after training completes.

### Option 2: Upload Later (Without Retraining)

Upload a checkpoint you already trained:

```bash
# From checkpoint path
uv run python -m tinker_cookbook.baseten_recipes.sft.upload_checkpoint \
  --checkpoint-path tinker://xxx/sampler_weights/final \
  --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --hf-repo-id username/model-name \
  --hf-private

# Or from log directory
uv run python -m tinker_cookbook.baseten_recipes.sft.upload_checkpoint \
  --log-dir /tmp/tinker-examples/sft/my-run-2026-01-27 \
  --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --hf-repo-id username/model-name
```

**Additional options:**
- `--hf-public`: Make repo public (default is private)
- `--download-dir /path`: Custom download directory
- `--lora-rank 32`: LoRA rank for model card
- `--dataset-name phare`: Dataset name for model card

## Troubleshooting

### Config File Not Found

```
FileNotFoundError: Config file not found: config.yaml
```

**Solution**: Copy `config.example.yaml` to `config.yaml` and customize it.

### API Key Not Set

```
TINKER_API_KEY not set in environment
```

**Solution**: Create a `.env` file with your API keys:

```bash
cat > .env << EOF
TINKER_API_KEY=your_key_here
HF_TOKEN=your_token_here
EOF
```

### Out of Memory During Training

**Solution**: Reduce `batch_size` or `max_length` in config:

```yaml
dataset:
  batch_size: 64      # Reduced from 256
  max_length: 8192    # Reduced from 16384
```

### Out of Memory During Merge

The merge step loads the full model. For large models (70B+), ensure:
- Sufficient RAM (2x model size)
- GPU with enough VRAM

### HuggingFace Upload Failed

Check:
- `HF_TOKEN` is in `.env` file
- Token has write permissions
- `hf_repo_id` format is `username/model-name`
- You have space quota on HuggingFace

### Renderer Mismatch

If you see tokenization errors, explicitly set the renderer:

```yaml
model:
  name: "Qwen/Qwen3-8B"
  renderer_name: "qwen3"  # Explicitly set
```

## File Structure

```
tinker_cookbook/baseten_recipes/sft/
├── README.md                      # This file
├── configs/                       # Config files
│   ├── example.yaml               # Example configuration template
│   └── phare.yaml                 # Config for phare-silver-messages dataset
├── train.py                       # Main training script
├── upload_checkpoint.py           # Upload trained checkpoint without retraining
├── sft_datasets.py                # Dataset builders
├── dataset_utils.py               # Dataset utilities (limiting, etc.)
├── post_training_pipeline.py      # Download/merge/upload automation
└── __init__.py                    # Package initialization
```

## Advanced Usage

### Resuming from Checkpoint

```yaml
infrastructure:
  load_checkpoint_path: "tinker://xxx/state/checkpoint-100"
```

### Custom Training Targets

```yaml
dataset:
  train_on_what: "LAST_ASSISTANT_MESSAGE"  # Only train on final response
```

Options:
- `ALL_ASSISTANT_MESSAGES`: Train on all assistant responses
- `LAST_ASSISTANT_MESSAGE`: Train on only final assistant response
- `ALL_TEXT`: Train on entire conversation

### Frequent Checkpointing

```yaml
checkpointing:
  save_every: 10
  eval_every: 10
  ttl_seconds: 1209600  # 14 days
```

### Custom Log Directory

```yaml
logging:
  log_path: "/my/custom/log/path"
  wandb_project: "my-project"
  wandb_name: "experiment-1"
```

## Memory Requirements

Approximate VRAM for training (batch_size=256):

| Model Size | LoRA Rank | VRAM (Training) | RAM (Merge) |
|------------|-----------|-----------------|-------------|
| 8B         | 32        | ~20GB          | ~32GB       |
| 30B MoE    | 32        | ~30GB          | ~60GB       |
| 70B        | 32        | ~60GB          | ~140GB      |

For post-training merge, ensure 2x model size in RAM/VRAM.

## Using with uv

This recipe works seamlessly with [uv](https://github.com/astral-sh/uv):

```bash
# Install dependencies
uv pip install -e .

# Run training
uv run python -m tinker_cookbook.baseten_recipes.sft.train config.yaml
```

## Related Documentation

- [Tinker Training & Sampling Guide](https://tinker-docs.thinkingmachines.ai/training-sampling)
- [Supervised Learning Docs](https://tinker-docs.thinkingmachines.ai/supervised-learning)
- [Model Lineup](https://tinker-docs.thinkingmachines.ai/model-lineup)
- [Renderers](https://tinker-docs.thinkingmachines.ai/rendering)
- [Download Weights](https://tinker-docs.thinkingmachines.ai/download-weights)

## License

This recipe is part of tinker-recipes and follows the same license as the parent repository.
