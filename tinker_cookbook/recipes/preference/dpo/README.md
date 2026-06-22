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
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric                ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ accuracy              │ 0.515748      │
│ batch_time            │ 13.960676     │
│ chosen_reward         │ 0.008626      │
│ clock_cycle:unique    │ 738784.000000 │
│ dpo_loss              │ 0.690734      │
│ epoch                 │ 0             │
│ learning_rate         │ 0.000000      │
│ loss:sum              │ -0.573885     │
│ margin                │ 0.005681      │
│ num_pairs             │ 254           │
│ num_tokens            │ 106677        │
│ progress              │ 0.980000      │
│ rejected_reward       │ 0.002946      │
│ test/nll              │ 1.798210      │
│ time/evals            │ 6.125040      │
│ time/get_batch        │ 0.437678      │
│ time/get_ref_logprobs │ 2.125185      │
│ time/run_evaluator    │ 6.123103      │
│ time/step             │ 5.270600      │
│ time/total            │ 13.960676     │
└───────────────────────┴───────────────┘
```
