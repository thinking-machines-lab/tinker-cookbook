# Forecasting with RL

Fine-tune Qwen3.8-27B with Tinker on binary markets from the
[Prophet Arena subset](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200)
using a chronological split and Brier reward.

## Run

```bash
export TINKER_API_KEY=...

python -m tinker_cookbook.recipes.forecasting.data
python -m tinker_cookbook.recipes.forecasting.train
```

## Prompt

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

## Configuration

The default run uses the Qwen3.8 low-reasoning renderer, LoRA rank 32, 16
questions per step, 32 forecasts per question, and a learning rate of `8e-5`. It
trains for 100 steps. Validation runs every 20 steps and after the final update,
with eight forecasts per question.

## Reward

For probability `p` and outcome `y`:

```text
Brier reward = 1 - (p - y)^2
```

This is a strictly proper scoring rule; malformed answers receive `0`. See
[Outcome-based Reinforcement Learning to Predict the Future](https://arxiv.org/abs/2507.16806)
for the outcome-based forecasting setup.

## Result

A default run produced:

| Step | Validation Brier reward | Validation accuracy | Valid format |
|---:|---:|---:|---:|
| 0 | 0.8013 | 73.10% | 99.90% |
| 20 | 0.8076 | 73.75% | 100.00% |
| 40 | 0.8132 | 73.36% | 100.00% |
| 60 | 0.8198 | 74.49% | 99.95% |
| 80 | 0.8223 | 74.24% | 100.00% |
| 100 | 0.8252 | 74.15% | 100.00% |

For scale, always answering `0.5` scores `0.7500` on this validation split and
always answering the training base rate scores `0.7709`.

## License

Prophet Arena data is distributed under the MIT license; see the
[dataset card](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-1200)
for terms.
