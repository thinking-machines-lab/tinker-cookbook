# Forecasting with RL

Fine-tune a model (Qwen3.8-27B in this example) with Tinker on binary markets from the
[Prophet Arena subset](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200)
using a chronological split and Brier reward.

## Run

```bash
export TINKER_API_KEY=...

# Optionally download, verify, and inspect the data split
python -m tinker_cookbook.recipes.forecasting.data

# Train the forecasting model
python -m tinker_cookbook.recipes.forecasting.train

# Or train with GLM-5.3 instead
python -m tinker_cookbook.recipes.forecasting.train \
    model_name=zai-org/GLM-5.3:peft:262144 \
    renderer_name=glm5_3_high_reasoning
```

## Prompt

Task: Given a binary prediction-market question and information available at a historical snapshot, forecast the probability that the market will resolve YES. The model does not use tools or live search in this recipe.

Here is the exact prompt template we use:

```text
Forecast whether this market will resolve YES using information available through {snapshot time}.

Event:
{event title}

Market:
{market}

Reference material:
{reference material available at the snapshot}

Resolution criteria:
{resolution criteria}

Market close time: {close time}

Output only the probability of YES as a number between 0 and 1.
```

An example from the pinned dataset is below. Note that this event bundles the
markets `PSG`, `Real Madrid`, and `Tie`, so it becomes three questions. Below is
the `Real Madrid` one, which resolved NO.

```text
Forecast whether this market will resolve YES using information available through 2025-07-08T12:01:09.330143+00:00.

Event:
PSG vs Real Madrid

Market:
Real Madrid

Reference material:
1. 2024–25 Paris Saint-Germain FC season
This article details PSG's 2024–25 season, highlighting their achievements including winning Ligue 1, the Coupe de France, and the UEFA Champions League, marking their first continental treble. It emphasizes the team's strong performance under manager Luis Enrique and the significant contributions of key players like Ousmane Dembélé, who was the season's top scorer with 33 goals. This information is relevant as it showcases PSG's recent form and success leading up to their match against Real Madrid.

2. 2025 FIFA Club World Cup Group H
This article outlines the structure and participants of Group H in the 2025 FIFA Club World Cup, which includes Real Madrid. It provides insights into Real Madrid's recent international competition schedule and performance. Understanding their participation and results in this tournament can offer context for their preparedness and form ahead of the match against PSG.

3. Ousmane Dembélé
This article provides an overview of Ousmane Dembélé's career, focusing on his impactful tenure at PSG since joining in August 2023. It highlights his contributions, including scoring 33 goals in the 2024–25 season and playing a pivotal role in PSG's treble-winning campaign. Dembélé's form and performance are crucial factors to consider when predicting the outcome of the upcoming match against Real Madrid.

...three more sources...

Resolution criteria:
The market resolves YES if this condition occurs: Real Madrid.

Market close time: 2025-07-09T21:06:41.016512+00:00

Output only the probability of YES as a number between 0 and 1.
```

Prompts use the event, named market, reference material, and timestamp from its
earliest snapshot, together with the market close time. Market prices and
outcome fields are omitted.

## Data

One CSV of 1,200 snapshots covering 869 events.

An event bundles one or more markets (median of 2 markets but occasionally over
100), and each market becomes its own binary question, with resolution criteria
derived from the market name. The 1,200 rows expand to 5,232 questions.

Each row is one snapshot, and the loader keeps the earliest snapshot for each
market. Note that the question never changes between snapshots, only the
attached news and the market price, so the earliest snapshot is the
longest-horizon and least informed version of the question.

Events that closed before October 20, 2025 form the training set. Events that
were first observed on that cutoff date or later form the validation set, with
every market of an event kept on the same side. Events that started before the
cutoff and had not closed by it are excluded. That splits 869 events into 613
training and 207 validation, dropping 49 at the boundary. This produces 3,587
and 1,180 questions, which the default caps trim to 1,024 and 256.

The recipe downloads a fixed published version and verifies the CSV contents,
so later dataset updates do not change a rerun.

### Temporal split methodology

This cookbook specifically uses a cutoff of October 20, 2025 UTC, but feel free to adjust based on the model you are using.

- Knowledge cutoff: Try to find the knowledge cutoff for the model you are post-training. Ideally its pretraining cutoff predates the forecasting questions. Qwen3.8 has been reported to self-identify an [early-2025 cutoff](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/discussions/8), although this is not publisher-verified. Adjust the model or split date to reduce outcome contamination.
- Training split: Events whose markets all closed before the cutoff.
- Validation split: Events first observed on or after the cutoff.

We further refine our data via:

- Boundary events: Events open across the cutoff are excluded.
- Deduplication: Each event-market pair becomes one question using its earliest snapshot, and all markets from the same event remain in the same split.

## Configuration

**Model**

Either:

- Qwen3.8-27B (the default), with the Tinker [`qwen3_8_low_reasoning`](../../renderers/qwen3_8.py) renderer
- GLM-5.3, with the Tinker [`glm5_3_high_reasoning`](../../renderers/glm5_3.py) renderer

Both use LoRA rank 32, a `1e-4` learning rate, and 32 training forecasts per
question. When switching to another model, use that model's recommended
renderer and rerun the relevant hyperparameter ablations.

**Training loop**

- 16 questions per step
- 128 training steps (two passes over the 1,024 training questions)

One epoch is 64 steps. Most of the improvement comes with the first epoch, so
setting `max_steps=64` is a reasonable way to shorten the training time.

**Validation**

- every 16 steps and after the final update
- 8 forecasts per question

## Reward

For probability `p` and outcome `y`:

```text
Brier reward = 1 - (p - y)^2
```

This is a strictly proper scoring rule; malformed answers receive `0`. See
[Outcome-based Reinforcement Learning to Predict the Future](https://arxiv.org/abs/2507.16806)
for the outcome-based forecasting setup.

## Result

With Qwen3.8-27B, we produced the following for reference:

| Step | Validation Brier reward | Validation accuracy | Valid format |
|---:|---:|---:|---:|
| 0 | 0.7998 | 72.92% | 99.80% |
| 16 | 0.8166 | 74.32% | 100.00% |
| 32 | 0.8207 | 73.88% | 100.00% |
| 48 | 0.8222 | 73.97% | 100.00% |
| 64 | 0.8262 | 74.54% | 100.00% |
| 80 | 0.8202 | 73.61% | 100.00% |
| 96 | 0.8071 | 72.22% | 99.90% |
| 112 | 0.8288 | 74.78% | 100.00% |
| 128 | 0.8294 | 75.34% | 100.00% |

With GLM-5.3:

| Step | Validation Brier reward | Validation accuracy | Valid format |
|---:|---:|---:|---:|
| 0 | 0.7952 | 72.53% | 98.63% |
| 16 | 0.8254 | 74.78% | 99.95% |
| 32 | 0.8224 | 74.41% | 99.95% |
| 48 | 0.8058 | 72.36% | 100.00% |
| 64 | 0.8413 | 77.32% | 100.00% |
| 80 | 0.8373 | 76.32% | 100.00% |
| 96 | 0.8309 | 75.32% | 100.00% |
| 112 | 0.8387 | 77.05% | 100.00% |
| 128 | 0.8475 | 78.30% | 100.00% |

This shows consistent improvement on temporally **held-out** validation questions and suggests that the learned forecasting behavior generalizes beyond the training events.

For scale, always answering `0.5` scores `0.7500` on this validation split and
always answering the training base rate scores `0.7709`.

## License

Prophet Arena data is distributed under the MIT license; see the
[dataset card](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200)
for terms.
