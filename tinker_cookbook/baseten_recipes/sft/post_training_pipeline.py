"""
Post-training pipeline for SFT recipe.

Handles:
1. Downloading checkpoint from Tinker
2. Merging LoRA adapter with base model
3. Generating model card
4. Uploading to HuggingFace Hub
"""

import logging
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from typing import Any

import tinker

logger = logging.getLogger(__name__)


def download_checkpoint(tinker_path: str, output_dir: str) -> str:
    """Download a checkpoint from Tinker and extract it.

    Args:
        tinker_path: Tinker checkpoint path (e.g., tinker://xxx/sampler_weights/final)
        output_dir: Directory to extract the checkpoint to

    Returns:
        Path to the extracted adapter directory
    """
    logger.info(f"Downloading checkpoint from {tinker_path}")

    # Create service and REST client
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    # Get signed download URL
    future = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    checkpoint_archive_url_response = future.result()

    logger.info(f"Download URL expires at: {checkpoint_archive_url_response.expires}")

    # Download the tar archive
    os.makedirs(output_dir, exist_ok=True)
    tar_path = os.path.join(output_dir, "checkpoint.tar")

    logger.info(f"Downloading archive to {tar_path}")
    urllib.request.urlretrieve(checkpoint_archive_url_response.url, tar_path)

    # Extract the archive
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)

    logger.info(f"Extracting archive to {adapter_dir}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(adapter_dir)

    # Verify expected files exist
    expected_files = ["adapter_model.safetensors", "adapter_config.json"]
    for file in expected_files:
        file_path = os.path.join(adapter_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Expected file not found in checkpoint: {file}")

    logger.info(f"Checkpoint downloaded and extracted to {adapter_dir}")
    return adapter_dir


def merge_adapter(base_model: str, adapter_path: str, output_path: str) -> str:
    """Merge LoRA adapter weights with base model.

    Args:
        base_model: HuggingFace model name or path
        adapter_path: Path to adapter directory with adapter_model.safetensors
        output_path: Path to save merged model

    Returns:
        Path to merged model
    """
    logger.info(f"Merging adapter with base model: {base_model}")
    logger.info(f"Adapter path: {adapter_path}")
    logger.info(f"Output path: {output_path}")

    # Call the merge script as subprocess
    merge_script = "tinker_cookbook/scripts/merge_tinker_adapter_to_hf_model.py"

    cmd = [
        "python",
        merge_script,
        "--hf-model",
        base_model,
        "--tinker-adapter-path",
        adapter_path,
        "--output-path",
        output_path,
    ]

    logger.info(f"Running merge command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Merge failed with stderr:\n{result.stderr}")
        raise RuntimeError(f"Adapter merge failed: {result.stderr}")

    logger.info(f"Merge completed successfully")
    logger.info(f"Merge output:\n{result.stdout}")

    return output_path


def generate_model_card(
    base_model: str,
    config: Any,
    metrics: dict[str, Any],
    dataset_name: str,
    repo_id: str,
) -> str:
    """Generate HuggingFace model card from template.

    Args:
        base_model: Base model name
        config: Training configuration
        metrics: Training metrics
        dataset_name: Name of the dataset used
        repo_id: HuggingFace repo ID

    Returns:
        Model card markdown string
    """
    logger.info("Generating model card")

    # Extract training info
    lora_rank = config.lora_rank
    learning_rate = config.learning_rate
    batch_size = getattr(config.dataset_builder.common_config, "batch_size", "N/A")
    num_epochs = config.num_epochs

    # Build metrics table
    metrics_table = "| Metric | Value |\n|--------|-------|\n"
    for key, value in metrics.items():
        metrics_table += f"| {key} | {value} |\n"

    if not metrics:
        metrics_table += "| No metrics available | - |\n"

    # Extract model name for citation
    model_name = repo_id.split("/")[-1]
    author = repo_id.split("/")[0] if "/" in repo_id else "Unknown"
    citation_key = model_name.replace("-", "_").lower()

    # Determine license based on base model
    license_map = {
        "llama": "llama3.1",
        "qwen": "apache-2.0",
        "gpt-oss": "mit",
    }
    license_name = "apache-2.0"  # default
    for key, lic in license_map.items():
        if key in base_model.lower():
            license_name = lic
            break

    model_card = f"""---
license: {license_name}
base_model: {base_model}
tags:
  - tinker
  - lora
  - fine-tuned
  - sft
datasets:
  - {dataset_name}
---

# {model_name}

Fine-tuned using [Tinker](https://thinkingmachines.ai/tinker) from `{base_model}`.

## Training Details

- **Base Model:** {base_model}
- **Dataset:** {dataset_name}
- **LoRA Rank:** {lora_rank}
- **Learning Rate:** {learning_rate}
- **Batch Size:** {batch_size}
- **Training Epochs:** {num_epochs}
- **LR Schedule:** {config.lr_schedule}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Generate text
messages = [
    {{"role": "user", "content": "Hello, how are you?"}}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Training Metrics

{metrics_table}

## Citation

```bibtex
@misc{{{citation_key},
  author = {{{author}}},
  title = {{{model_name}}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

---

*This model was trained using the [Tinker SFT recipe](https://github.com/baseten/tinker-recipes/tree/main/tinker_cookbook/baseten_recipes/sft)*
"""

    return model_card


def upload_to_hub(
    model_path: str, repo_id: str, model_card: str, private: bool, token: str | None = None
) -> str:
    """Upload merged model to HuggingFace Hub.

    Args:
        model_path: Path to merged model directory
        repo_id: HuggingFace repo ID (username/model-name)
        model_card: Model card markdown content
        private: Whether to make the repo private
        token: HuggingFace token (or None to use HF_TOKEN env var)

    Returns:
        HuggingFace repo URL
    """
    logger.info(f"Uploading model to HuggingFace Hub: {repo_id}")

    # Validate repo_id format
    if "/" not in repo_id:
        raise ValueError(f"Invalid repo_id format: {repo_id}. Expected: username/model-name")

    # Check for HF token
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "HuggingFace token not found. Set HF_TOKEN environment variable or pass token parameter."
            )

    # Warn if uploading public repo
    if not private:
        logger.warning(f"⚠️  Uploading to PUBLIC repository: {repo_id}")
        logger.warning("This model will be visible to everyone on HuggingFace Hub")

    # Save model card
    model_card_path = os.path.join(model_path, "README.md")
    with open(model_card_path, "w") as f:
        f.write(model_card)
    logger.info(f"Model card saved to {model_card_path}")

    # Upload using transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library required for upload. Install with: pip install transformers"
        )

    logger.info("Loading model for upload...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info(f"Pushing model to {repo_id}...")
    model.push_to_hub(repo_id, private=private, token=token)

    logger.info(f"Pushing tokenizer to {repo_id}...")
    tokenizer.push_to_hub(repo_id, private=private, token=token)

    hf_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"✓ Model uploaded successfully to {hf_url}")

    return hf_url


async def run_post_training_pipeline(
    checkpoint_path: str,
    base_model: str,
    repo_id: str,
    private: bool,
    download_dir: str,
    training_config: Any,
    metrics: dict[str, Any],
) -> dict[str, str]:
    """Run the complete post-training pipeline.

    Args:
        checkpoint_path: Tinker checkpoint path
        base_model: Base model name
        repo_id: HuggingFace repo ID
        private: Whether to make the repo private
        download_dir: Directory for downloads and merging
        training_config: Training configuration object
        metrics: Training metrics

    Returns:
        Dictionary with paths and URLs
    """
    result = {}

    try:
        # Step 1: Download checkpoint
        logger.info("\n[1/4] Downloading checkpoint...")
        adapter_dir = download_checkpoint(checkpoint_path, download_dir)
        result["adapter_dir"] = adapter_dir

        # Step 2: Merge adapter
        logger.info("\n[2/4] Merging adapter with base model...")
        merged_model_dir = os.path.join(download_dir, "merged_model")
        merge_adapter(base_model, adapter_dir, merged_model_dir)
        result["merged_model_dir"] = merged_model_dir

        # Step 3: Generate model card
        logger.info("\n[3/4] Generating model card...")
        dataset_name = getattr(training_config.dataset_builder, "dataset_name", "custom")
        model_card = generate_model_card(base_model, training_config, metrics, dataset_name, repo_id)
        result["model_card"] = model_card

        # Step 4: Upload to HuggingFace Hub
        logger.info("\n[4/4] Uploading to HuggingFace Hub...")
        hf_url = upload_to_hub(merged_model_dir, repo_id, model_card, private)
        result["hf_url"] = hf_url

        logger.info("\n✓ Post-training pipeline completed successfully!")
        return result

    except Exception as e:
        logger.error(f"Post-training pipeline failed: {e}")
        raise
