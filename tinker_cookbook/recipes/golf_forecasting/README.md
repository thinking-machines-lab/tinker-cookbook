# Golf Forecasting

This recipe trains an LLM to read a golf leaderboard snapshot and output a calibrated probability distribution over likely winners.

The design follows the same pattern as the other Tinker Cookbook recipes:

- `build_dataset.py` fetches and normalizes public data into versioned JSONL artifacts.
- `data.py` defines the forecasting schema and RL dataset builder.
- `env.py` implements a single-turn forecasting environment with dense proper-scoring rewards.
- `train.py` launches RL training with `tinker_cookbook.rl.train`.
- `eval.py` runs fixed offline evaluation on a held-out manifest.
- `program.md` is a Karpathy-style instruction file for a coding agent running the recipe in a loop.

## Data Format

Each JSONL example stores one leaderboard snapshot with its resolved winner:

```json
{
  "example_id": "masters-2025-r4-h10",
  "tournament_id": "masters-2025",
  "tournament_name": "The Masters",
  "course_name": "Augusta National",
  "round_number": 4,
  "event_day": "Sunday",
  "snapshot_timestamp": "2025-04-13T20:05:00Z",
  "players": [
    {
      "name": "Scottie Scheffler",
      "position": "1",
      "score_to_par": -11,
      "strokes_behind": 0,
      "holes_completed": 10,
      "current_hole": 11,
      "holes_remaining": 8,
      "prior_win_prob": 0.34
    }
  ],
  "target_winner": "Scottie Scheffler",
  "other_field_prior": 0.18,
  "system_context": {
    "weather_summary": "Light breeze, scoring expected to stay low"
  },
  "source_urls": ["https://example.com/leaderboard"]
}
```

The model must return strict JSON:

```json
{
  "winner_probs": {
    "Scottie Scheffler": 0.41,
    "Rory McIlroy": 0.23,
    "other": 0.36
  }
}
```

## No Bundled Data

This recipe intentionally does not ship with a hard-coded golf dataset. The expectation is that your coding agent will discover public sources, fetch raw leaderboard history and priors, and build the dataset itself.

The only bundled artifact is a source-manifest template:

- `tinker_cookbook/example_data/golf_forecasting/source_manifest.template.json`

## Building A Dataset

`build_dataset.py` reads a JSON manifest that points to public data sources and optional prior files. It caches raw responses under `raw/`, normalizes them, and writes:

- `train.jsonl`
- `val.jsonl`
- `heldout.jsonl`
- `dataset_manifest.json`

Typical flow:

```bash
# 1. Copy and edit the source-manifest template after discovering real public sources
cp tinker_cookbook/example_data/golf_forecasting/source_manifest.template.json /tmp/golf_sources.json

# 2. Build the normalized dataset
python -m tinker_cookbook.recipes.golf_forecasting.build_dataset \
  source_manifest_path=/tmp/golf_sources.json \
  output_dir=tinker_cookbook/example_data/golf_forecasting \
  fetch_online=true
```

Important rules:

- Keep the held-out split frozen once `dataset_manifest.json` exists.
- Cache raw sources so experiments are reproducible.
- Record source URLs and timestamps for every fetch.

## Training

Example command:

```bash
python -m tinker_cookbook.recipes.golf_forecasting.train \
  model_name="meta-llama/Llama-3.2-1B-Instruct" \
  dataset_manifest_path=tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json \
  group_size=8 \
  groups_per_batch=32 \
  learning_rate=4e-5 \
  max_tokens=256
```

The RL environment rewards forecasts using a normalized multiclass Brier score. Validation rollouts are run automatically through the standard `RLTestSetEvaluator` path in `tinker_cookbook.rl.train`.

## Offline Evaluation

Run the frozen held-out benchmark with:

```bash
python -m tinker_cookbook.recipes.golf_forecasting.eval \
  model_name="meta-llama/Llama-3.2-1B-Instruct" \
  dataset_manifest_path=tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json
```

This writes:

- `metrics.json`
- `predictions.jsonl`

under `tinker_cookbook/recipes/golf_forecasting/results/<timestamp>/`.

Primary metric:

- `eval/log_loss`

Secondary metrics:

- `eval/brier`
- `eval/top1_accuracy`
- `eval/top3_recall`
- `eval/format_valid_rate`

## Autoresearch

Use `program.md` as the top-level instruction file for Cursor or Claude Code. The agent is allowed to:

- edit the golf forecasting recipe files
- discover public data sources
- gather public web data and priors
- rebuild the training dataset
- choose different models, output formats, and training strategies
- maintain a frozen anchor eval plus a flexible research eval

The agent must preserve one rule:

- once it creates the anchor eval manifest, that anchor benchmark stays frozen even if the research eval evolves

## Overnight tmux Launch

If you want to kick off autoresearch in an unattended way, there are a few supported paths below.

Common prerequisites:

- make sure `claude` already works interactively on the machine
- make sure `TINKER_API_KEY` is set
- make sure Claude Code is configured to avoid blocking on approval prompts
- for phone tracking with Claude Remote Control, make sure your Claude Code version and plan support Remote Control and that you are already logged in

The recipe includes a checked-in launcher:

- `tinker_cookbook/recipes/golf_forecasting/run_autoresearch_tmux.sh`

<details>
<summary>Kick off from a remote host with <code>tmux</code></summary>

Use this if you want to start the job on a remote machine, disconnect your SSH session, and let it continue running.

Make the launcher executable if needed:

```bash
chmod +x tinker_cookbook/recipes/golf_forecasting/run_autoresearch_tmux.sh
```

Then launch it:

```bash
export TINKER_API_KEY=your_key_here
tinker_cookbook/recipes/golf_forecasting/run_autoresearch_tmux.sh
```

Useful follow-ups:

```bash
tmux attach -t golf-autoresearch
tail -f ~/golf-autoresearch-logs/golf-autoresearch-*.log
tmux kill-session -t golf-autoresearch
```

Notes:

- if `claude` shows a first-run login or approval prompt, the session may block
- doing one manual dry run first is a good idea
- the script derives the repo root from its own location by default, but you can still override `REPO`
- you can also override `SESSION_NAME`, `LOG_DIR`, and `STARTUP_WAIT_SECONDS`

</details>

<details>
<summary>Kick off locally and track it from your phone with Claude Remote Control</summary>

Claude Code supports an official Remote Control mode that lets you monitor and interact with the same local session from the Claude mobile app or from [claude.ai/code](https://claude.ai/code).

The launcher uses `claude --remote-control`, which starts a normal interactive Claude session and also exposes that same session to your phone.

To launch the autoresearch session with Remote Control enabled:

```bash
export TINKER_API_KEY=your_key_here
REMOTE_CONTROL=1 tinker_cookbook/recipes/golf_forecasting/run_autoresearch_tmux.sh
```

Then:

1. Attach once to the tmux session:

```bash
tmux attach -t golf-autoresearch
```

2. Claude should show the Remote Control URL in the terminal for that interactive session.
3. Scan the QR code with the Claude mobile app, or open the URL, or go to [claude.ai/code](https://claude.ai/code) and select the session by name.
4. Detach from tmux with `Ctrl-b` then `d`.

From that point, the Claude process keeps running on the remote host, and you can monitor or message the same session from your phone.

Notes:

- for Remote Control sessions, the `claude` process must keep running on the host machine
- if the host loses network for too long or the process exits, the phone session disconnects

</details>

<details>
<summary>Kick off directly from Claude Code on the web</summary>

You can also start this autoresearch job directly from Claude Code on the web at [claude.ai/code](https://claude.ai/code).

- Claude Code on the web runs in Anthropic-managed cloud infrastructure
- Remote Control runs on your own machine and exposes that local session remotely

For the web flow, use the official Claude Code on the web setup:

1. Sign in at [claude.ai/code](https://claude.ai/code).
2. Connect GitHub and grant Claude access to the `tinker-cookbook` repository.
3. Create or select a cloud environment.
4. Make sure the cloud environment has the secrets and setup needed for this recipe, especially:
   - `TINKER_API_KEY`
   - any other credentials you want the agent to use for public-data access or model tooling
5. Select the repository and branch you want Claude to work on.
6. Choose an appropriate permission mode such as auto-accept edits if you want a more autonomous run.
7. Paste a kickoff prompt like this:

```text
Read `tinker_cookbook/recipes/golf_forecasting/program.md` and follow it exactly.

Start by discovering public golf data sources, building the first dataset, freezing a clean anchor eval as soon as you have a minimally viable benchmark, defining an initial research eval, establishing a baseline on both, and then entering the endless autoresearch loop.

You have broad freedom to change models, prompts, output formats, training methods, dataset structure, and evaluation design, but once the anchor eval is created it must never change.
```

If you prefer to bootstrap the web setup from the CLI first, the official docs also mention:

- `/web-setup` inside Claude Code to help sync GitHub and create a default cloud environment
- `claude --remote "..."` to start a cloud task from the terminal

Notes:

- the web flow requires GitHub integration because Claude needs to clone and push branches in the cloud environment
- unlike Remote Control, the web flow does not use your local filesystem or local `tmux` session
- if your autoresearch depends on local-only services or files, prefer the `tmux` + Remote Control path instead

</details>

