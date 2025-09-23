# Replicating Search-R1 with Tinker

[Search-R1](https://arxiv.org/pdf/2503.09516) is a recent paper that showcases tool-use RL for multi-hop QA on wikipedia.
It provides a clean setup for testing tool-use RL and also releases their training eval data as resources.

In this demo, we replicated their experiments using Tinker.

### Replication Results

We conducted experiments on a `Qwen/Qwen2.5-7B-Instruct` model and compared with the results reported in the original paper.
The results can be seen here,

| | Natural Questions | Trivia QA | HotpotQA | 2WikiMultihopQA |
|---|---|---|---|---|
| original paper | 42.9 | 62.3 | 38.6 | 34.6 |
| tinker  | **51.6** | **67.3** | **49.7** | **42.8** |

The key differences between our experiment and the original paper include:
1. We used the default importance-weighting REINFORCE loss implemented in Tinker
2. We used the default synchronous rollout logic in tinker cookbook
3. We used gemini embedding and chroma db, motivated by their simplicity to setup for a public demo. In exploratory experiments, the gemini embedding does not improve RL performance over the E5 embedding model used in the original paper.


## Running this demo

### Installation and Setup
This demo is built with chroma db and gemini api. You can install the additional dependencies by
```
uv pip install -e .[vector-search]
```

By default, we use google vertex ai for the embedding service, and you need to set `$GOOGLE_GENAI_USE_VERTEXAI`, `$GCP_VERTEXAI_PROJECT_NUMBER`,  `$GCP_VERTEXAI_REGION`. Or, tweak `./embedding.py` to authenticate differently.

Currently, the tool use RL run relies on a separate chrome vector search service. You can set it up with the following step:
1. You can download a pre-compute wiki18 index: https://huggingface.co/datasets/tianyi-thinks/2018-wiki-index/blob/main/chroma_db.tar.xz
2. Launch chroma service on your localhost. Example command looks like `uv run chroma run --host localhost --path <decompressed_path>/chroma_db --port 8000`

If you launch the chroma service locally, you generally need 160+ GB RAM to load the vector index in memory for good performance.

### Example command

This default command trains a `Qwen3-4B-Instruct-2507` with reasonable hyperparameters.
```
uv run tinker_cookbook/recipes/tool_use/search/rl_search.py
```

With the default hyperparameters, you can expect performance like: [TODO: apologies for the delay but this will be filled in soon].

A successful run generally learns multi-turn search within 10 - 25 steps, which can be monitored by checking if `env/all/turns_per_episode` has increased over 2 turns.

To speed up training, you may consider turning on `--stream_minibatch`. In principle, this system improvement should have minimal effect on training.

### Implementation guidance

1. The tool call rendering / parsing logic is in `tinker_cookbook/renderers.py`. Currently, tool calling is only demonstrated on the Qwen series renderer. Currently, the system prompt necessary for enabling tool calling is included in `./search_env.py`. Changing the tool calling parsing format also requires updating the system prompt accordingly.
2. Extend `./embedding.py` to swap out Gemini embedding.
3. Extend `./tools.py` to swap out chroma vector search service.
