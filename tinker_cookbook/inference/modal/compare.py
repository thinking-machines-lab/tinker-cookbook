"""COMPARE: sample the same prompts from Tinker and the Modal endpoint.

Reports how often the two disagree on the greedy output, so you can gauge the
behavior difference between the Tinker sampling client and the Modal deployment.

    FINETUNE=my-ft MODEL=Qwen/Qwen3-8B modal run -m tinker_cookbook.inference.modal.compare \\
        --tinker-path tinker://<run-id>/sampler_weights/<name> --url https://<...>.modal.direct

Pass --prompts path/to/prompts.json (a JSON list of strings) to use your own.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .common import app

DEFAULT_PROMPTS = [
    "What is Modal?",
    "What is 17 times 24?",
    "Name the capital of France.",
    "Write a haiku about GPUs.",
    "Explain the singular value decomposition in one sentence.",
]
MAX_TOKENS = 256


def _modal_output(url: str, model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"]


@app.local_entrypoint()
async def compare(tinker_path: str, url: str, prompts: str = "") -> None:
    import tinker

    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    model, finetune = os.environ["MODEL"], os.environ["FINETUNE"]
    questions = json.load(open(prompts)) if prompts else DEFAULT_PROMPTS

    tok = get_tokenizer(model)
    rend = renderers.get_renderer(model_info.get_recommended_renderer_name(model), tok)
    sc = tinker.ServiceClient().create_sampling_client(model_path=tinker_path)

    disagree = 0
    for q in questions:
        mi = rend.build_generation_prompt([{"role": "user", "content": q}])
        sampled = await sc.sample_async(
            mi, num_samples=1, sampling_params=tinker.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        )
        tinker_out = rend.parse_response(sampled.sequences[0].tokens)[0]["content"]
        modal_out = _modal_output(url, finetune, q)
        same = tinker_out.strip() == modal_out.strip()
        disagree += not same
        print(f"[{'same' if same else 'DIFF'}] {q[:50]}")

    n = len(questions)
    print(f"\ndisagree on {disagree}/{n} = {disagree / n:.0%} of prompts (greedy)")
