# Skill Benchmark Results

Generated: 2026-03-29

---

## 1. Summary Table

| Skill | Eval | Config | Pass | Total | Rate |
|-------|------|--------|------|-------|------|
| tinker-core | 1 | with_skill | 7 | 7 | 100% |
| tinker-core | 1 | without_skill | 3 | 7 | 43% |
| tinker-core | 2 | with_skill | 7 | 7 | 100% |
| tinker-core | 2 | without_skill | 4 | 7 | 57% |
| tinker-core | 3 | with_skill | 7 | 7 | 100% |
| tinker-core | 3 | without_skill | 5 | 7 | 71% |
| tinker-core | 4 | with_skill | 5 | 5 | 100% |
| tinker-core | 4 | without_skill | 5 | 5 | 100% |
| tinker-sft | 1 | with_skill | 9 | 9 | 100% |
| tinker-sft | 1 | without_skill | 9 | 9 | 100% |
| tinker-sft | 2 | with_skill | 7 | 8 | 88% |
| tinker-sft | 2 | without_skill | 6 | 8 | 75% |
| tinker-sft | 3 | with_skill | 5 | 6 | 83% |
| tinker-sft | 3 | without_skill | 4 | 6 | 67% |
| tinker-rl | 1 | with_skill | 8 | 9 | 89% |
| tinker-rl | 1 | without_skill | 6 | 9 | 67% |
| tinker-rl | 2 | with_skill | 7 | 7 | 100% |
| tinker-rl | 2 | without_skill | 7 | 7 | 100% |
| tinker-rl | 3 | with_skill | 6 | 7 | 86% |
| tinker-rl | 3 | without_skill | 4 | 7 | 57% |
| tinker-rl | 4 | with_skill | 6 | 6 | 100% |
| tinker-rl | 4 | without_skill | 5 | 6 | 83% |
| tinker-preferences | 1 | with_skill | 7 | 8 | 88% |
| tinker-preferences | 1 | without_skill | 7 | 8 | 88% |
| tinker-preferences | 2 | with_skill | 8 | 8 | 100% |
| tinker-preferences | 2 | without_skill | 8 | 8 | 100% |
| tinker-preferences | 3 | with_skill | 4 | 5 | 80% |
| tinker-preferences | 3 | without_skill | 4 | 5 | 80% |
| tinker-ops | 1 | with_skill | 7 | 7 | 100% |
| tinker-ops | 1 | without_skill | 7 | 7 | 100% |
| tinker-ops | 2 | with_skill | 7 | 7 | 100% |
| tinker-ops | 2 | without_skill | 6 | 7 | 86% |
| tinker-ops | 3 | with_skill | 6 | 6 | 100% |
| tinker-ops | 3 | without_skill | 6 | 6 | 100% |
| tinker-ops | 4 | with_skill | 5 | 5 | 100% |
| tinker-ops | 4 | without_skill | 4 | 5 | 80% |
| tinker-dev | 1 | with_skill | 8 | 8 | 100% |
| tinker-dev | 1 | without_skill | 3 | 8 | 38% |
| tinker-dev | 2 | with_skill | 6 | 7 | 86% |
| tinker-dev | 2 | without_skill | 5 | 7 | 71% |

---

## 2. Aggregate by Skill

| Skill | With Pass | With Total | With Rate | Without Pass | Without Total | Without Rate | Delta |
|-------|-----------|-----------|-----------|-------------|--------------|-------------|-------|
| tinker-core | 26 | 26 | **100%** | 17 | 26 | 65% | **+35%** |
| tinker-sft | 21 | 23 | **91%** | 19 | 23 | 83% | **+8%** |
| tinker-rl | 27 | 29 | **93%** | 22 | 29 | 76% | **+17%** |
| tinker-preferences | 19 | 21 | **90%** | 19 | 21 | 90% | **0%** |
| tinker-ops | 25 | 25 | **100%** | 23 | 25 | 92% | **+8%** |
| tinker-dev | 14 | 15 | **93%** | 8 | 15 | 53% | **+40%** |

---

## 3. Grand Total

| Config | Pass | Total | Rate |
|--------|------|-------|------|
| **With Skill** | **132** | **139** | **95.0%** |
| **Without Skill** | **108** | **139** | **77.7%** |
| **Delta** | **+24** | | **+17.3%** |

---

## 4. Most Discriminating Expectations

These expectations consistently PASS with skill and FAIL without skill:

| Expectation | Skill | Eval |
|-------------|-------|------|
| Warns about HF_TOKEN being needed for gated Llama models | tinker-core | 1 |
| Mentions pip install -e . for cookbook installation (not just SDK) | tinker-core | 1 |
| Suggests running a minimal example like sl_basic or rl_basic | tinker-core | 1 |
| Includes pip install tinker command | tinker-core | 1 |
| Uses model_info.get_recommended_renderer_name() (not hardcoded) | tinker-core | 2 |
| Recommends MoE models over dense for cost-effectiveness | tinker-core | 2 |
| Distinguishes between SFT LR and RL LR ranges | tinker-core | 2 |
| Warns that new SamplingClient must be created after saving weights | tinker-core | 3 |
| Mentions that forward() is for eval only, not training | tinker-core | 3 |
| Uses LR in 1e-5 to 4e-5 range for RL | tinker-rl | 1 |
| Warns about sampler desync after weight saves | tinker-rl | 1 |
| Suggests kl_penalty_coef=0.0 for tool-use exploration | tinker-rl | 3 |
| Mentions AsyncConfig for expensive multi-turn environments | tinker-rl | 3 |
| Creates new SamplingClient after weight saves | tinker-rl | 4 |
| Mentions shrinking workloads for debugging | tinker-ops | 2 |
| Uses infrequent_evaluator_builders for expensive evals | tinker-ops | 4 |
| Includes smoke test template in tests/recipes/ | tinker-dev | 1 |
| Uses @pytest.mark.integration and run_recipe() helper | tinker-dev | 1 |
| Uses cli_utils.check_log_dir() before training | tinker-dev | 1 |
| Mentions behavior_if_log_dir_exists=delete for CI tests | tinker-dev | 1 |
| Mentions dimension naming conventions (_P, _G, _T, _D) | tinker-dev | 1 |

---

## 5. Per-Eval Detail

### tinker-core / eval-1
**Prompt:** "I just got my Tinker API key. Walk me through setting everything up and running my first training example."

#### with_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Includes pip install tinker command | PASS | `pip install tinker` shown in Step 2 |
| Includes TINKER_API_KEY environment variable export | PASS | `export TINKER_API_KEY=<your-key>` in Step 1 |
| Shows verification code using ServiceClient and create_lora_training_client | PASS | Step 3 shows both `tinker.ServiceClient()` and `svc.create_lora_training_client(...)` |
| Suggests running a minimal example like sl_basic or rl_basic | PASS | Step 4 shows `python -m tinker_cookbook.recipes.sl_basic` and `rl_basic` |
| Does not reference docs/ files that only exist in the repo clone | PASS | No docs/ references |
| Warns about HF_TOKEN being needed for gated Llama models | PASS | Explicit `export HF_TOKEN=<your-huggingface-token>` with Llama note |
| Mentions pip install -e . for cookbook installation | PASS | `cd tinker-cookbook && pip install -e .` shown |

#### without_skill (3/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Includes pip install tinker command | FAIL | Uses `uv pip install tinker-cookbook` -- installs cookbook, not the SDK directly via `pip install tinker` |
| Includes TINKER_API_KEY environment variable export | PASS | `export TINKER_API_KEY="your-key-here"` shown |
| Shows verification code using ServiceClient and create_lora_training_client | PASS | ServiceClient shown in verification; create_lora_training_client shown in SFT example |
| Suggests running a minimal example like sl_basic or rl_basic | FAIL | Not mentioned; provides inline SFT script instead |
| Does not reference docs/ files that only exist in the repo clone | PASS | No docs/ references |
| Warns about HF_TOKEN being needed for gated Llama models | FAIL | Not mentioned |
| Mentions pip install -e . for cookbook installation | FAIL | Uses `uv pip install tinker-cookbook` from PyPI, not editable local install |

---

### tinker-core / eval-2
**Prompt:** "I want to fine-tune a model for math reasoning. Which model should I use and what learning rate?"

#### with_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Recommends a specific model name from the available lineup | PASS | Recommends Llama-3.1-8B-Instruct, mentions Qwen3-8B and Qwen3-30B-A3B |
| Explains why the model fits math reasoning | PASS | "instruction model...RL can focus on correctness"; "hybrid models...chain-of-thought" |
| Provides LR recommendation using get_lr() or numeric range | PASS | States 4e-5 for RL; mentions get_lr() for Qwen |
| Distinguishes between SFT LR and RL LR | PASS | "RL range of 1e-5 to 4e-5" vs "formula is calibrated for SFT" |
| Uses model_info.get_recommended_renderer_name() | PASS | `model_info.get_recommended_renderer_name(model_name)` in code |
| Recommends MoE models for cost-effectiveness | PASS | "Qwen/Qwen3-30B-A3B (3B active) at a fraction of the cost" |
| Mentions LoRA rank recommendation (default 32) | PASS | "LoRA rank: 32" in batch setup |

#### without_skill (4/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Recommends a specific model name | PASS | Lists Qwen3-8B, Qwen3-32B, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct |
| Explains why model fits math reasoning | PASS | "thinking capabilities, which tend to help on math reasoning" |
| Provides LR using get_lr() or numeric range | PASS | Shows `get_lr(model_name, is_lora=True)` with concrete numbers |
| Distinguishes between SFT LR and RL LR | FAIL | Only discusses LR from get_lr (SFT-calibrated); does not clearly distinguish RL range |
| Uses model_info.get_recommended_renderer_name() | FAIL | Not mentioned; no renderer discussion |
| Recommends MoE models for cost-effectiveness | FAIL | Not mentioned |
| Mentions LoRA rank recommendation | PASS | "at rank-32 LoRA" referenced |

---

### tinker-core / eval-3
**Prompt:** "How do I use the Tinker SDK to do a forward/backward pass and optimizer step? Show me the async pattern."

#### with_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Shows ServiceClient creation | PASS | `svc = ServiceClient()` |
| Shows forward_backward_async and optim_step_async | PASS | Both shown in async loop |
| Demonstrates submitting both back-to-back before awaiting | PASS | Explicitly shown: "Submit both -- don't await yet" |
| Uses correct API types like AdamParams | PASS | `AdamParams(learning_rate=2e-4)` |
| Does not add manual retry wrappers | PASS | No retry logic |
| Warns about sampler desync | PASS | "always create a new SamplingClient" section |
| Mentions forward() for eval only | PASS | "Use forward (not forward_backward) for eval" section |

#### without_skill (5/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Shows ServiceClient creation | PASS | `service_client = tinker.ServiceClient()` |
| Shows forward_backward_async and optim_step_async | PASS | Both shown |
| Demonstrates back-to-back submission | PASS | Pattern shown with clock cycle explanation |
| Uses correct API types | PASS | `types.AdamParams(learning_rate=1e-4)` |
| Does not add retry wrappers | PASS | No retry logic |
| Warns about sampler desync | FAIL | Not mentioned |
| Mentions forward() for eval only | FAIL | Not mentioned |

---

### tinker-core / eval-4
**Prompt:** "Custom training loop with forward/backward, optim step, then evaluates on 50 test problems."

#### with_skill (5/5)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses forward_backward_async and optim_step_async back-to-back | PASS | Both submitted before awaiting |
| Evaluates concurrently with asyncio.gather | PASS | `asyncio.gather(*tasks)` |
| Creates new SamplingClient after saving weights | PASS | `sc = tc.save_weights_and_get_sampling_client()` |
| Shows preparing next batch while GPU is busy | PASS | Comment about preparing next batch during GPU work |
| Does not wrap SDK calls in retry logic | PASS | No retry wrappers |

#### without_skill (5/5)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses forward_backward_async and optim_step_async back-to-back | PASS | Shown in training loop |
| Evaluates concurrently | PASS | `asyncio.gather(*futures)` for evaluation |
| Creates new SamplingClient after saving weights | PASS | `save_weights_and_get_sampling_client_async()` |
| Shows preparing next batch while GPU is busy | PASS | Pipeline batch submission pattern |
| Does not wrap SDK calls in retry logic | PASS | No retry wrappers |

---

### tinker-sft / eval-1
**Prompt:** "I have a JSONL file with chat conversations. Help me write a script to fine-tune Llama-3.1-8B on it."

#### with_skill (9/9)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses FromConversationFileBuilder | PASS | Imported and used |
| Uses ChatDatasetBuilderCommonConfig with correct fields | PASS | All fields present |
| Uses model_info.get_recommended_renderer_name() | PASS | Used |
| Includes TrainOnWhat.ALL_ASSISTANT_MESSAGES or explains options | PASS | Used and TrainOnWhat.ALL_TOKENS mentioned |
| Uses cli_utils.check_log_dir() | PASS | Called before training |
| Script is complete and runnable | PASS | Has `if __name__ == "__main__"` |
| Uses hyperparam_utils.get_lr() or explains LoRA LR ~10x | PASS | Mentions both |
| Explains batch_size is in tokens | PASS | "batch_size=128 is in tokens, not examples" |
| Mentions expected JSONL format | PASS | Shows `{"messages": [...]}` format |

#### without_skill (9/9)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses FromConversationFileBuilder | PASS | Imported and used |
| Uses ChatDatasetBuilderCommonConfig with correct fields | PASS | All fields present |
| Uses model_info.get_recommended_renderer_name() | PASS | Used |
| Includes TrainOnWhat.ALL_ASSISTANT_MESSAGES or explains options | PASS | Used |
| Uses cli_utils.check_log_dir() | PASS | Called before training |
| Script is complete and runnable | PASS | Has `if __name__ == "__main__"` |
| Uses get_lr() or explains LoRA LR | PASS | `get_lr(model_name)` used; "10x higher LR" explained |
| Explains batch_size is in tokens | PASS | "128 tokens per batch" |
| Mentions expected JSONL format | PASS | Shows format with examples |

---

### tinker-sft / eval-2
**Prompt:** "I want to distill knowledge from Qwen3-8B into Qwen3-8B-Base."

#### with_skill (7/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses TeacherConfig with teacher model | PASS | `TeacherConfig(base_model=cli_config.teacher_model)` |
| Uses PromptOnlyDatasetBuilder | PASS | Used with dataset_name |
| Uses DistillationDatasetConfig | PASS | Used to bind dataset to teacher |
| Uses train_on_policy.Config | PASS | `train_on_policy.Config(...)` and `train_on_policy.main(config)` |
| Mentions kl_penalty_coef as key signal | PASS | "The only supervision signal is the KL divergence" |
| Sets student to Qwen3-8B-Base | PASS | `model_name: str = "Qwen/Qwen3-8B-Base"` |
| Explains on-policy vs off-policy distillation | FAIL | Only describes on-policy; no off-policy comparison |
| Mentions multi-teacher as option | PASS | References `on_policy_multi_teacher.py` |

#### without_skill (6/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses TeacherConfig | PASS | `TeacherConfig(base_model="Qwen/Qwen3-8B")` |
| Uses PromptOnlyDatasetBuilder | PASS | Used |
| Uses DistillationDatasetConfig | PASS | Used |
| Uses train_on_policy.Config | PASS | Used |
| Mentions kl_penalty_coef | PASS | `kl_penalty_coef=1.0` with explanation |
| Sets student to Qwen3-8B-Base | PASS | `model_name="Qwen/Qwen3-8B-Base"` |
| Explains on-policy vs off-policy | FAIL | No off-policy discussion |
| Mentions multi-teacher | FAIL | Not mentioned |

---

### tinker-sft / eval-3
**Prompt:** "What renderers are available for Qwen models and how do I handle vision inputs?"

#### with_skill (5/6)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Lists qwen3, qwen3_disable_thinking, qwen3_vl, qwen3_5 variants | PASS | Full table with 7 renderers |
| Shows image content part format with type='image' | PASS | `{"type": "image", "image_url": "https://..."}` |
| Mentions VL renderer for vision | PASS | Names specific VL renderers |
| Uses model_info.get_recommended_renderer_name() | PASS | `model_info.get_recommended_renderer_name(model_name)` shown |
| Explains thinking-enabled vs disabled renderers | PASS | Table distinguishes thinking on/off |
| Mentions register_renderer API | FAIL | Not mentioned |

#### without_skill (4/6)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Lists qwen3, qwen3_disable_thinking, qwen3_vl, qwen3_5 variants | PASS | Full table and descriptions |
| Shows image content part format | PASS | Shows typed content parts with ImagePart |
| Mentions VL renderer for vision | PASS | Detailed VL renderer section |
| Uses model_info.get_recommended_renderer_name() | FAIL | Not mentioned; hardcodes renderer names directly |
| Explains thinking-enabled vs disabled | PASS | Detailed descriptions for each |
| Mentions register_renderer API | FAIL | Not mentioned |

---

### tinker-rl / eval-1
**Prompt:** "I want to train a model with GRPO on GSM8K math problems. Give me a complete script."

#### with_skill (8/9)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses Gsm8kDatasetBuilder | PASS | Imported from tinker_cookbook.recipes.math_rl.math_env |
| Uses train.Config from tinker_cookbook.rl.train | PASS | Used |
| Sets group_size (4-16) and batch_size | PASS | group_size=16, batch_size=128 |
| Uses LR in 1e-5 to 4e-5 range | PASS | learning_rate=4e-5 |
| Uses model_info.get_recommended_renderer_name() | PASS | Used |
| Script is complete and runnable | PASS | Has `if __name__ == "__main__"` |
| Explains advantages centered within group | PASS | "GRPO centers advantages within each group of 16" |
| Mentions available loss functions (importance_sampling, ppo, cispo, dro) | FAIL | Only mentions importance_sampling implicitly via GRPO |
| Warns about sampler desync | PASS | "a new SamplingClient must be created" |

#### without_skill (6/9)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses Gsm8kDatasetBuilder | PASS | Imported and used |
| Uses train.Config | PASS | Used |
| Sets group_size and batch_size | PASS | group_size=8, batch_size=128 |
| Uses LR in 1e-5 to 4e-5 range | FAIL | Uses get_lr() which returns ~4e-4, well above the RL range |
| Uses model_info.get_recommended_renderer_name() | PASS | Used |
| Script is complete and runnable | PASS | Has `if __name__ == "__main__"` |
| Explains advantages centered within group | PASS | "centers advantages across those 8 responses" |
| Mentions available loss functions | FAIL | Only mentions importance_sampling |
| Warns about sampler desync | FAIL | Not mentioned |

---

### tinker-rl / eval-2
**Prompt:** "Custom trivia RL environment."

#### with_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Subclasses ProblemEnv | PASS | `class TriviaEnv(ProblemEnv)` |
| Implements get_question, check_answer, check_format, get_reference_answer | PASS | All four implemented |
| Shows ProblemGroupBuilder | PASS | Used with env_thunk and num_envs |
| Creates RLDatasetBuilder | PASS | `@chz.chz class TriviaDatasetBuilder(RLDatasetBuilder)` |
| Mentions Env objects are single-use | PASS | "envs are single-use" |
| Explains reward formula | PASS | `format_coef * (check_format - 1) + check_answer` |
| Uses @chz.chz on RLDatasetBuilder | PASS | Decorator shown |

#### without_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Subclasses ProblemEnv | PASS | `class TriviaEnv(ProblemEnv)` |
| Implements all four methods | PASS | All four shown |
| Shows ProblemGroupBuilder | PASS | Used with partial() |
| Creates RLDatasetBuilder | PASS | `@chz.chz class TriviaDatasetBuilder(RLDatasetBuilder)` |
| Env objects single-use | PASS | "Env instances are single-use" |
| Explains reward formula | PASS | format_coef and reward computation described |
| @chz.chz on RLDatasetBuilder | PASS | Decorator shown |

---

### tinker-rl / eval-3
**Prompt:** "Train a model to use a search tool in multi-turn conversation."

#### with_skill (6/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses MessageEnv or equivalent | PASS | Uses build_agent_tool_env wrapping EnvFromMessageEnv |
| Implements initial_observation and step or equivalent | PASS | Handled via build_agent_tool_env |
| Wraps with EnvFromMessageEnv or equivalent | PASS | build_agent_tool_env returns wrapped Env |
| Sets max_trajectory_tokens | PASS | max_trajectory_tokens=8192 |
| Mentions AsyncConfig for expensive environments | PASS | `AsyncConfig(max_steps_off_policy=4, groups_per_batch=8)` |
| Suggests kl_penalty_coef=0.0 for tool-use exploration | PASS | `kl_penalty_coef=0.0` with "allow tool-use exploration" |
| Mentions cleanup() on EnvGroupBuilder | FAIL | Not mentioned |

#### without_skill (4/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses MessageEnv or equivalent | PASS | Uses build_agent_tool_env |
| Implements initial_observation and step | PASS | Handled via build_agent_tool_env |
| Wraps with EnvFromMessageEnv | PASS | build_agent_tool_env handles this |
| Sets max_trajectory_tokens | PASS | max_trajectory_tokens=32*1024 |
| Mentions AsyncConfig | FAIL | Not mentioned |
| Suggests kl_penalty_coef=0.0 | FAIL | Not mentioned |
| Mentions cleanup() | FAIL | Not mentioned |

---

### tinker-rl / eval-4
**Prompt:** "RL training loop with reward model scoring -- maximize throughput."

#### with_skill (6/6)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| forward_backward_async + optim_step_async back-to-back | PASS | Shown with clear explanation |
| Overlaps rollouts with training | PASS | "collect next batch rollouts -- GPU is working concurrently" |
| Evaluates/scores concurrently | PASS | `asyncio.gather(*[score_rollout(...)])` |
| Creates new SamplingClient after weight saves | PASS | Dedicated section on sampler desync |
| Mentions AsyncConfig for slow environments | PASS | Full AsyncConfig section |
| Does not write sequential .result() chains | PASS | Shows wrong vs correct pattern |

#### without_skill (5/6)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| forward_backward_async + optim_step_async back-to-back | PASS | Shown |
| Overlaps rollouts with training | PASS | Pipeline pattern shown |
| Evaluates concurrently | PASS | "Submit all sampling requests concurrently" |
| Creates new SamplingClient after weight saves | FAIL | Not mentioned |
| Mentions AsyncConfig | PASS | AsyncConfig section present |
| Does not write sequential .result() chains | PASS | Correct pattern shown |

---

### tinker-preferences / eval-1
**Prompt:** "Run DPO on the HHH preference dataset."

#### with_skill (7/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses DPODatasetBuilderFromComparisons | PASS | Imported and used |
| Uses HHHComparisonBuilder | PASS | `HHHComparisonBuilder()` |
| Sets dpo_beta around 0.1 | PASS | `dpo_beta=0.1` |
| Uses LR around 1e-5 | PASS | `learning_rate=1e-5` |
| Uses train_dpo.Config | PASS | Used |
| Complete and runnable script | PASS | Has `if __name__ == "__main__"` |
| Mentions DPO works best from SFT checkpoint | PASS | Explicitly noted |
| Lists alternative comparison builders | FAIL | No mention of HelpSteer3 or UltraFeedback |

#### without_skill (7/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses DPODatasetBuilderFromComparisons | PASS | Used |
| Uses HHHComparisonBuilder | PASS | Used with test_size |
| Sets dpo_beta around 0.1 | PASS | `dpo_beta=0.1` |
| Uses LR around 1e-5 | PASS | `learning_rate=1e-5` |
| Uses train_dpo.Config | PASS | Used |
| Complete and runnable | PASS | Has `if __name__ == "__main__"` |
| DPO works best from SFT checkpoint | PASS | Mentions load_checkpoint_path |
| Lists alternative comparison builders | FAIL | Not mentioned |

---

### tinker-preferences / eval-2
**Prompt:** "Full RLHF pipeline -- SFT, reward model, then RL."

#### with_skill (8/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Describes all three stages | PASS | SFT, RM, RL clearly separated |
| Checkpoint chaining with get_last_checkpoint() | PASS | `checkpoint_utils.get_last_checkpoint(...)` |
| Uses PreferenceModelBuilderFromChatRenderer | PASS | Used for RM in RL stage |
| Uses PairwisePreferenceRLDatasetBuilder | PASS | Used in RL stage |
| RL LR much lower than SFT | PASS | SFT ~2e-4, RL 1e-5 |
| Mentions RM quality validation | PASS | "RM quality matters more than RL tricks" |
| Explains purpose of each stage | PASS | Clear explanations |
| Uses ChatDatasetBuilderFromComparisons for RM | PASS | Used |

#### without_skill (8/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Describes all three stages | PASS | All three covered |
| Checkpoint chaining with get_last_checkpoint() | PASS | Used |
| Uses PreferenceModelBuilderFromChatRenderer | PASS | Used |
| Uses PairwisePreferenceRLDatasetBuilder | PASS | Used |
| RL LR much lower than SFT | PASS | get_lr for SFT, 1e-5 for RL |
| Mentions RM quality validation | PASS | Monitoring guidance provided |
| Explains purpose of each stage | PASS | Clear explanations |
| Uses ChatDatasetBuilderFromComparisons for RM | PASS | Used |

---

### tinker-preferences / eval-3
**Prompt:** "Custom preference data pairs for DPO."

#### with_skill (4/5)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Shows custom ComparisonBuilder | PASS | `class MyComparisonBuilder(ComparisonBuilder)` |
| Returns (chosen, rejected) tuples | PASS | Returns `list[tuple[list[Message], list[Message]]]` |
| Standard Message format with role/content | PASS | Uses `{"role": ..., "content": ...}` |
| Wraps with DPODatasetBuilderFromComparisons | PASS | Used |
| Mentions preference data quality > quantity | FAIL | Not mentioned |

#### without_skill (4/5)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Shows custom ComparisonBuilder | PASS | `class MyPairwiseDataBuilder(ComparisonDatasetBuilder)` |
| Returns (chosen, rejected) tuples | PASS | Returns LabeledComparison with chosen/rejected |
| Standard Message format | PASS | Uses `{"role": ..., "content": ...}` |
| Wraps with DPODatasetBuilderFromComparisons | PASS | Used |
| Mentions data quality > quantity | FAIL | Not mentioned |

---

### tinker-ops / eval-1
**Prompt:** "Download weights, merge LoRA, publish to HuggingFace."

#### with_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses weights.download() with tinker:// path | PASS | `weights.download(tinker_path="tinker://...")` |
| Uses weights.build_hf_model() | PASS | Shown with all parameters |
| Uses weights.publish_to_hf_hub() with repo_id | PASS | Used |
| Mentions save_weights_for_sampler (not state) | PASS | Explicit warning about the distinction |
| Mentions HF_TOKEN | PASS | "Publishing reads your HuggingFace token from the HF_TOKEN environment variable" |
| Mentions build_lora_adapter as alternative | PASS | Full section on vLLM/SGLang serving |
| Recommends dtype='bfloat16' | PASS | `dtype="bfloat16"` with "recommended for most models" |

#### without_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Uses weights.download() | PASS | Used with tinker:// path |
| Uses weights.build_hf_model() | PASS | Full merge section |
| Uses weights.publish_to_hf_hub() with repo_id | PASS | Used with ModelCardConfig |
| Mentions save_weights_for_sampler | PASS | Path shows sampler_weights |
| Mentions HF_TOKEN | PASS | "Authentication uses the HF_TOKEN environment variable" |
| Mentions build_lora_adapter | PASS | "Option B -- Lightweight PEFT adapter" section |
| Recommends dtype='bfloat16' | PASS | Listed as default |

---

### tinker-ops / eval-2
**Prompt:** "RL training seems stuck. How do I analyze the logs?"

#### with_skill (7/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Loading metrics.jsonl with pandas | PASS | `pd.read_json("path/to/metrics.jsonl", lines=True)` |
| Mentions env/all/reward/total | PASS | Explicitly referenced |
| KL divergence should stay below 0.01 | PASS | "kl_sample_train_v1 spiking above 0.01" |
| Rollout summaries in iteration_NNNNNN/ | PASS | `iteration_000010/train_rollout_summaries.jsonl` |
| Practical fixes | PASS | "Lower your learning rate", "Fix prompt format or reward parser" |
| Mentions HTML reports | PASS | train.html referenced multiple times |
| Mentions shrinking workloads for debugging | PASS | "Reduce batch_size, group_size, or max_tokens" in symptom table |

#### without_skill (6/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Loading metrics.jsonl | PASS | `pd.read_json(...)` shown |
| env/all/reward/total | PASS | Referenced |
| KL < 0.01 | PASS | "Training is considered stable when KL stays below 0.01" |
| Rollout summaries | PASS | `train_rollout_summaries.jsonl` referenced |
| Practical fixes | PASS | "Reduce learning rate", "adjust reward function" |
| HTML reports | PASS | train.html referenced |
| Shrinking workloads for debugging | FAIL | Not mentioned |

---

### tinker-ops / eval-3
**Prompt:** "Inline evaluation during training to track accuracy."

#### with_skill (6/6)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Evaluator taking SamplingClient returning metrics dict | PASS | `SamplingClientEvaluator` with `async def __call__` |
| Adding via evaluator_builders | PASS | `evaluator_builders=[nll_eval_builder]` |
| eval_every parameter | PASS | `eval_every=8` |
| infrequent_evaluator_builders | PASS | Full section with `infrequent_eval_every=50` |
| Inspect AI integration | PASS | `InspectEvaluatorBuilder` section |
| RL test set pattern | PASS | Shows RL evaluator with `eval_accuracy` function |

#### without_skill (6/6)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Evaluator function | PASS | `SamplingClientEvaluator` shown |
| evaluator_builders | PASS | Shown in both SL and RL |
| eval_every | PASS | `eval_every=10` |
| infrequent_evaluator_builders | PASS | `infrequent_eval_every=100` shown |
| Inspect AI | PASS | `InspectEvaluatorBuilder` section |
| RL test set pattern | PASS | `RLTestSetEvaluator` shown |

---

### tinker-ops / eval-4
**Prompt:** "Evaluate 200 test problems during training without slowing things down."

#### with_skill (5/5)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Concurrent evaluation with asyncio.gather | PASS | `asyncio.gather(*[evaluate_one(p) for p in test_problems])` |
| infrequent_evaluator_builders for expensive evals | PASS | "use infrequent_evaluator_builders for heavy evals" |
| Uses SamplingClient.sample() | PASS | `sampling_client.sample_async(...)` |
| Keep evaluators fast | PASS | "Tune eval_every to match your tolerance" |
| Does not evaluate sequentially | PASS | Explicitly contrasts sequential vs concurrent |

#### without_skill (4/5)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Concurrent evaluation | PASS | `asyncio.gather(*[...])` |
| infrequent_evaluator_builders | FAIL | Not mentioned; uses only `evaluator_builders` |
| Uses SamplingClient.sample() | PASS | `sampling_client.sample_async(...)` |
| Keep evaluators fast | PASS | Discusses latency characteristics |
| Does not evaluate sequentially | PASS | Shows slow vs fast pattern comparison |

---

### tinker-dev / eval-1
**Prompt:** "Add a new RL recipe for a coding task with sandbox execution."

#### with_skill (8/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Creates files under tinker_cookbook/recipes/<name>/ | PASS | `tinker_cookbook/recipes/coding_rl/` with proper structure |
| Uses @chz.chz CLIConfig with defaults | PASS | `@chz.chz class CLIConfig` |
| Uses model_info.get_recommended_renderer_name() | PASS | Referenced and used |
| Uses cli_utils.check_log_dir() | PASS | `cli_utils.check_log_dir(log_path, ...)` in Step 5 |
| Includes smoke test template in tests/recipes/ | PASS | Full `test_recipe_coding_rl.py` |
| Uses @pytest.mark.integration and run_recipe() | PASS | Both shown |
| Mentions behavior_if_log_dir_exists=delete for CI | PASS | In test args |
| Mentions dimension naming conventions | PASS | "_P (problems), _G (groups), _T (tokens), _D (datums)" |

#### without_skill (3/8)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Creates files under recipes/<name>/ | PASS | `tinker_cookbook/recipes/my_code_rl/` |
| Uses @chz.chz CLIConfig | PASS | `@chz.chz class CLIConfig` |
| Uses model_info.get_recommended_renderer_name() | PASS | Mentioned in key pitfalls |
| Uses cli_utils.check_log_dir() | FAIL | Not shown in training entry point |
| Includes smoke test template | FAIL | No test file included |
| Uses @pytest.mark.integration and run_recipe() | FAIL | No test shown |
| Mentions behavior_if_log_dir_exists=delete for CI | FAIL | Not mentioned |
| Mentions dimension naming conventions | FAIL | Not mentioned |

---

### tinker-dev / eval-2
**Prompt:** "How do I run the tests? What's the difference between unit and integration tests?"

#### with_skill (6/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Explains unit tests (*_test.py, no API key) | PASS | "colocated with source...files named *_test.py...No API key needed" |
| Explains integration tests (test_recipe_*.py, TINKER_API_KEY) | PASS | "live in tests/recipes/...require TINKER_API_KEY" |
| Shows uv run pytest tinker_cookbook/ | PASS | `uv run pytest tinker_cookbook/` |
| Shows uv run pytest tests/ | PASS | `uv run pytest tests/ -v -x -s` |
| Mentions @pytest.mark.integration | PASS | "marked @pytest.mark.integration" |
| Mentions conftest.py auto-skips | PASS | "tests/conftest.py will automatically skip" |
| Mentions pre-commit checks (ruff, pyright) | FAIL | Not mentioned |

#### without_skill (5/7)
| Expectation | Result | Evidence |
|-------------|--------|----------|
| Explains unit tests | PASS | "colocated with source code...*_test.py naming" |
| Explains integration tests | PASS | "tests/recipes/...end-to-end tests" |
| Shows uv run pytest tinker_cookbook/ | PASS | Shows `pytest tinker_cookbook/` (without uv run, but functionally equivalent) |
| Shows uv run pytest tests/ | PASS | Shows `pytest tests/` |
| Mentions @pytest.mark.integration | FAIL | Not mentioned; discusses downstream_compat marker instead |
| Mentions conftest.py auto-skips | PASS | "smoke tests...are automatically skipped" |
| Mentions pre-commit checks | FAIL | Not mentioned |

---

## 6. Key Findings

### Where skills help most

**tinker-dev (+40%):** The largest improvement. The skill provides critical knowledge about project conventions -- test file templates, `check_log_dir()`, `behavior_if_log_dir_exists=delete`, `@pytest.mark.integration`, `run_recipe()`, and dimension naming. These are project-internal conventions that cannot be inferred from general knowledge.

**tinker-core (+35%):** The skill excels at setup guidance (pip install commands, HF_TOKEN warnings, editable installs), suggesting specific built-in examples (sl_basic, rl_basic), and warning about pitfalls (sampler desync, forward-only for eval). Without the skill, responses are still functional but miss project-specific entry points and operational warnings.

**tinker-rl (+17%):** The skill provides correct hyperparameter ranges (RL LR distinct from SFT LR), warns about sampler desync after weight saves, and knows about AsyncConfig for expensive environments. The without-skill responses sometimes use SFT-level learning rates for RL (dangerous in practice) and miss operational patterns.

### Where skills help least

**tinker-preferences (0%):** Both with-skill and without-skill responses scored identically. The preference training APIs (DPO, RLHF) are well-structured enough that general code knowledge plus the CLAUDE.md context is sufficient. Both configurations missed the same expectations (alternative comparison builders, data quality advice).

**tinker-sft (+8%):** Modest improvement. The SFT workflow is the most straightforward, so without-skill responses are already strong. The skill mainly adds value on multi-teacher distillation awareness and renderer auto-selection.

### Patterns

1. **Project conventions are the biggest differentiator.** Expectations about test patterns, CLI conventions, log directory handling, and naming conventions almost exclusively pass with the skill and fail without.

2. **Operational warnings are skill-dependent.** Sampler desync, forward-only for eval, HF_TOKEN requirements, and LR range distinctions are reliably provided by the skill but often missing without it.

3. **API structure knowledge is NOT a differentiator.** Both configurations correctly use ServiceClient, TrainingClient, Datum types, and builder patterns. The codebase is well-structured enough that the model can infer correct usage from context.

4. **Both configurations fail on the same niche expectations.** Neither reliably mentions register_renderer, multi-teacher distillation details, alternative comparison builders, or pre-commit hooks. These could indicate expectations that are too specific or skill content that needs improvement.

5. **The without-skill model sometimes makes dangerous errors.** Using SFT learning rates for RL (tinker-rl eval-1) would produce a non-functional training run. The skill prevents this class of error by providing calibrated domain knowledge.

### Expectations both configurations consistently fail

| Expectation | Skill | Eval |
|-------------|-------|------|
| Mentions register_renderer API | tinker-sft | 3 |
| Explains on-policy vs off-policy distillation | tinker-sft | 2 |
| Lists alternative comparison builders (HelpSteer3, UltraFeedback) | tinker-preferences | 1 |
| Mentions preference data quality > quantity | tinker-preferences | 3 |
| Mentions available loss functions (ppo, cispo, dro) | tinker-rl | 1 |
| Mentions pre-commit checks (ruff, pyright) | tinker-dev | 2 |
