# Direct Preference Optimization

Please check our [doc](https://tinker-docs.thinkingmachines.ai/cookbook/preferences/dpo-guide/) for background on DPO.

Here is an example command:
```
python -m tinker_cookbook.recipes.preference.dpo.train \
    log_path=/tmp/dpo-hhh-experiment \
    model_name=Qwen/Qwen3.5-9B-Base \
    dataset=hhh \
    renderer_name=role_colon \
    learning_rate=1e-5 \
    dpo_beta=0.1
```

After 50 training steps, you should expect final training metrics like:

| Metric | Value |
| --- | ---: |
| accuracy | 0.551181 |
| chosen_reward | 0.005246 |
| dpo_loss | 0.690139 |
| learning_rate | 2e-7 |
| margin | 0.006933 |
| num_pairs | 254 |
| num_tokens | 106677 |
| rejected_reward | -0.001687 |
