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

After 50 training steps, the final logged step should look like:
```
                 Step 49
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Metric                ┃ Value          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ accuracy              │ 0.551181       │
│ chosen_reward         │ 0.005246       │
│ clock_cycle:unique    │ 8433789.000000 │
│ dpo_loss              │ 0.690139       │
│ epoch                 │ 0              │
│ learning_rate         │ 0.000000       │
│ loss:sum              │ -0.573373      │
│ margin                │ 0.006933       │
│ num_pairs             │ 254            │
│ num_tokens            │ 106677         │
│ progress              │ 0.980000       │
│ rejected_reward       │ -0.001687      │
│ time/get_batch        │ 0.422145       │
│ time/get_ref_logprobs │ 8.827843       │
│ time/step             │ 11.413823      │
│ time/total            │ 20.665923      │
└───────────────────────┴────────────────┘
```

This CLI does not emit `test/nll` unless an evaluator is configured.
