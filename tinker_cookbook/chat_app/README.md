# Tinker Chat CLI

This README provides instructions for chatting with models trained using **Tinker**.

---

## Getting Started

You can easily chat with any sampler checkpoint saved using **Tinker** by running the following command:

```bash
python -m tinker_cookbook.chat_app.tinker_chat_cli \
    model_path=tinker://<unique_id>/sampler_weights/final \
    base_model=Qwen/Qwen3.5-9B-Base
```

### Arguments

* **model_path**: Path to the trained Tinker sampler checkpoint. Example: `tinker://<unique_id>/sampler_weights/final`. Note that the Tinker chat CLI will not work with training weights which look like `tinker://<unique_id>/weights/final`. Make sure the checkpoint contains `sampler_weights`.
* **base_model**: Hugging Face base model to use for inference. Example: `Qwen/Qwen3.5-9B-Base`

---

## Customization

You can modify the behavior of the chat by providing additional arguments:

* **max_tokens** *(int, default=512)*
  Maximum number of tokens to generate in the response.

* **temperature** *(float, default=0.7)*
  Controls the randomness of the output. Higher values = more random responses.

* **top_p** *(float, default=0.9)*
  Controls nucleus sampling. The model considers only the top tokens with cumulative probability `p`.

Example:

```bash
python -m tinker_cookbook.chat_app.tinker_chat_cli \
    model_path=tinker://<unique_id>/sampler_weights/final \
    base_model=Qwen/Qwen3.5-9B-Base \
    max_tokens=256 \
    temperature=0.8 \
    top_p=0.95
```
