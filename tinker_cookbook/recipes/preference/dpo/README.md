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

After 50 steps, you should expect training metrics like:
```
                   Step 50
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric                         ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ accuracy                       │ 0.568627  │
│ batch_time                     │ 27.953704 │
│ chosen_reward                  │ 0.053621  │
│ dpo_loss                       │ 0.683825  │
│ learning_rate                  │ 0.000009  │
│ margin                         │ 0.002147  │
│ num_pairs                      │ 255       │
│ num_tokens                     │ 112638    │
│ progress                       │ 0.081210  │
│ rejected_reward                │ 0.032152  │
│ test/nll                       │ 1.871778  │
└────────────────────────────────┴───────────┘
```
