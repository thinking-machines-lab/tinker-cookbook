# Replicating Search-R1 with Tinker

[Search-R1](https://arxiv.org/pdf/2503.09516) is a recent paper that showcases tool-use RL for multi-hop QA on Wikipedia.
It provides a clean setup for testing tool-use RL and also released their training and evaluation data.
In this demo, we demonstrate similar experiments using `Qwen3-4B-Instruct-2507`, and we include our replication results using `Qwen/Qwen2.5-7B-Instruct` at the end.

## Running This Demo

### Installation and Setup

This demo is built with Chroma DB and the Gemini API. You can install the additional dependencies by

```bash
uv pip install -e .[vector-search]
```

By default, we use google vertex ai for the embedding service, and you need to set `$GOOGLE_GENAI_USE_VERTEXAI`, `$GCP_VERTEXAI_PROJECT_NUMBER`, `$GCP_VERTEXAI_REGION`. Or, tweak `./embedding.py` to authenticate differently.

Currently, the tool use RL run relies on a separate Chroma vector search service. You can set it up with the following step:

1. You can download a pre-computed wiki18 index: https://huggingface.co/datasets/tianyi-thinks/2018-wiki-index/blob/main/chroma_db.tar.xz
2. Launch the Chroma service on localhost. Example command: `chroma run --host localhost --path <decompressed_path>/chroma_db --port 8000`

If you launch the chroma service locally, you generally need 160+ GB RAM to load the vector index in memory for good performance.

### Lightweight alternative: local BM25 (no Chroma, runs on a laptop)

If you don't have a 160 GB RAM machine handy, use the local backend instead. It swaps Chroma + Gemini embeddings for an in-memory BM25 index over a small Wikipedia subset (~3k passages, <10 MB). The training loop is identical, so you can iterate on the recipe without any external infrastructure.

```bash
python -m tinker_cookbook.recipes.search_tool.train backend=local
```

The first run downloads the [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia) corpus from HuggingFace. Subsequent runs reuse the cached copy. No API keys are required.

**Trade-offs vs the full setup:**

| | Full (Chroma + Gemini) | Lightweight (local BM25) |
|---|---|---|
| RAM | 160+ GB | <1 GB |
| Retrieval | Dense embedding + vector search | Lexical (BM25) |
| External services | Chroma server, Gemini API | None |
| Corpus | Full Wikipedia (~30M passages) | ~3k curated passages |
| Expected accuracy | Paper-level (see table below) | Lower — suitable for iteration, not benchmarking |

The lightweight variant is best for learning the recipe, debugging, and iterating on the training loop. Use the full Chroma setup when you want to replicate the reported benchmark numbers.

### Example command

This default command trains a `Qwen3-4B-Instruct-2507` with reasonable hyperparameters using the full Chroma + Gemini backend.

```bash
python -m tinker_cookbook.recipes.search_tool.train
```

(The default `backend=chroma` is implicit. Pass `backend=local` to use the lightweight variant described above.)

With the default hyperparameters, you can expect performance like:
| | Natural Questions | Trivia QA | HotpotQA | 2WikiMultihopQA |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 51.8 | 70.2 | 52.0 | 47.7 |

A successful run generally learns multi-turn search within 10-25 steps, which can be monitored by checking if `env/all/turns_per_episode` has increased over 2 turns.

**Note:** The `max_trajectory_tokens` parameter (default: 32,768) limits the total context length for multi-turn interactions. If your searches require longer contexts, you can adjust it with `max_trajectory_tokens=<value>`.

To speed up training, you may consider turning on `--stream_minibatch`. In principle, this system improvement should have minimal effect on training.

### Extensions: How to Include Other Tools?

1. The tool call rendering / parsing logic is in [tinker_cookbook/renderers/](../../renderers/). Tool calling is supported on multiple renderers (Qwen, GPT-OSS, DeepSeek, Kimi). Currently, the system prompt necessary for enabling tool calling is included in `./search_env.py` (`SEARCH_TASK_INSTRUCTIONS`) and is written specifically for Qwen. Changing the tool calling parsing format requires updating the system prompt accordingly.
2. Extend `./embedding.py` to replace the Gemini embedding.
3. Extend `./tools.py` to add new tools using the `@tool` decorator - see `ChromaTool.search()` as an example.

### Replication Results

We conducted experiments on a `Qwen/Qwen2.5-7B-Instruct` model and compared with the results reported in the original paper.
Note this model is not available on Tinker and we chose it specifically to compare with the original paper.
The results can be seen here,

|                | Natural Questions | Trivia QA | HotpotQA | 2WikiMultihopQA |
| -------------- | ----------------- | --------- | -------- | --------------- |
| original paper | 42.9              | 62.3      | 38.6     | 34.6            |
| tinker         | **51.6**          | **67.3**  | **49.7** | **42.8**        |

The key differences between our experiment and the original paper include:

1. We used the default importance-weighting REINFORCE loss implemented in Tinker
2. We used the default synchronous rollout logic in the Tinker Cookbook
3. We used Gemini embedding and Chroma DB, motivated by their ease of setup for a public demo. In exploratory experiments, the Gemini embedding does not improve RL performance over the E5 embedding model used in the original paper.

[1] Jin, B., Zeng, H., Yue, Z., Yoon, J., Arık, S. O., Wang, D., Zamani, H., & Han, J. (2025). Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516.
