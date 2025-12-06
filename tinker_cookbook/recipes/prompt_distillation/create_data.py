import asyncio
import json
import os
import re
from typing import Any

import chz
import tinker
from tqdm.asyncio import tqdm_asyncio

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

LANGUAGE_CLASSIFICATION_PROMPT = """You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax (keywords, operators, braces) and focus ONLY on human language in comments and string literals.
   - Preserve letters and diacritics; do NOT strip accents.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script → hi.
   - Greek script → el.
   - Cyrillic script → ru.
   - Han characters (中文) → zh. (Treat Traditional as zh too.)
   - Arabic script → ar vs ur:
       • If Urdu-only letters appear (e.g., ے, ڑ, ں, ھ, ٹ, ڈ, کھ, گ, چ with Urdu forms), or clear Urdu words, choose ur.
       • Otherwise choose ar.
   (If multiple scripts appear, pick the script that contributes the majority of alphabetic characters. If tied, go to step 5.)

3) Latin-script heuristics (use when text is mainly Latin letters):
   - vi: presence of Vietnamese-specific letters/diacritics (ă â ê ô ơ ư đ, plus dense diacritics across many words).
   - tr: presence of Turkish-specific letters (ı İ ğ Ğ ş Ş ç Ç ö Ö ü Ü) and common function words (ve, bir, için, değil, ama, çok).
   - de: presence of umlauts (ä ö ü) or ß and common function words (und, der, die, das, nicht, ist).
   - es: presence of ñ, ¿, ¡ and common words (y, de, la, el, es, no, por, para, con, gracias, hola).
   - fr: frequent French diacritics (é è ê à ç ô â î û ù) and common words (et, le, la, les, des, une, est, avec, pour, merci, bonjour).
   - en: default among Latin languages if strong evidence for others is absent, but ONLY if English function words are present (the, and, is, are, to, of, in, for, on, with). If evidence is insufficient for any Latin language, prefer 'ot' over guessing.

4) Named entities & loanwords:
   - Do NOT decide based on a single proper noun, brand, or place name.
   - Require at least two function words or repeated language-specific signals (diacritics/letters) before assigning a Latin-language label.

5) Mixed-language text:
   - Determine the dominant language by counting indicative tokens (language-specific letters/diacritics/function words) AFTER preprocessing.
   - If two or more languages are equally dominant or the text is a deliberate multi-language mix, return 'ot'.

6) Very short or noisy inputs:
   - If the text is ≤2 meaningful words or too short to be confident, return 'ot' unless there is a very strong language-specific signal (e.g., “bonjour” → fr, “hola” → es).

7) Transliteration/romanization:
   - If Hindi/Urdu/Arabic/Chinese/Russian/Greek is written purely in Latin letters (romanized) without clear, repeated language-specific cue words, return 'ot'. (Only classify as hi/ur/ar/zh/ru/el when native scripts or highly distinctive romanized patterns are clearly present.)

8) Code-heavy inputs:
   - If the text is mostly code with minimal or no natural-language comments/strings, return 'ot'.
   - If comments/strings clearly indicate a language per rules above, use that label.

9) Ambiguity & confidence:
   - When in doubt, choose 'ot' rather than guessing.

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx ∈ {{ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot}} and nothing else.

Text to classify:
{text}
"""


@chz.chz
class Config:
    output_file: str


def setup_clients():
    # disable tokenizer parallelism warnings
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Creating service client")
    service_client = tinker.ServiceClient()
    print("Creating sampling client")
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B")
    tokenizer = get_tokenizer("Qwen/Qwen3-30B-A3B")
    renderer = renderers.get_renderer("qwen3", tokenizer)

    return sampling_client, tokenizer, renderer


async def create_data_async(cfg: Config, sampling_client: Any, tokenizer: Any, renderer: Any):
    # read sentences from multilingual.txt file
    with open("tinker_cookbook/example_data/multilingual.txt", "r") as f:
        sentences = f.readlines()
    sentences = [sentence.strip() for sentence in sentences]

    print(f"Loaded {len(sentences)} sentences")

    async def sample_from_model(
        sentence: str,
    ) -> tuple[str, str | None]:
        prompt = LANGUAGE_CLASSIFICATION_PROMPT.format(text=sentence)
        tokenized_prompt = tinker.ModelInput.from_ints(tokenizer.encode(prompt))
        params = tinker.SamplingParams(
            max_tokens=1000, temperature=0.15, stop=renderer.get_stop_sequences()
        )
        result = await sampling_client.sample_async(
            prompt=tokenized_prompt, sampling_params=params, num_samples=1
        )
        response = tokenizer.decode(result.sequences[0].tokens)
        # parse the final answer from the response using regex for example: Final Answer: xx where xx is two character label for each language and nothing else. xx is one of the following: en, fr, es, hi, ja, ko, ru, ot.
        # the final answer is the xx part
        search_response = re.search(r"Final Answer: (\w+)", response)
        final_answer = search_response.group(1) if search_response else None
        return (sentence, final_answer)

    answers: list[str | None] = []
    questions: list[str] = []
    for coro in tqdm_asyncio.as_completed(
        [sample_from_model(s) for s in sentences], total=len(sentences)
    ):
        question, answer = await coro
        answers.append(answer)
        questions.append(question)

    # save the input and final answer to a file
    with open(cfg.output_file, "w") as f:
        for question, answer in zip(questions, answers):
            if answer is None:
                continue
            messages = {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ],
            }
            f.write(json.dumps(messages) + "\n")

    return


def main(cfg: Config):
    # check if the output file exists
    if os.path.exists(cfg.output_file):
        print(f"Output file {cfg.output_file} already exists")
        return
    elif not os.path.exists(os.path.dirname(cfg.output_file)):
        # check if the output directory exists
        print(f"Output directory {os.path.dirname(cfg.output_file)} does not exist")
        print(f"Creating directory {os.path.dirname(cfg.output_file)}")
        os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)

    # Setup clients synchronously
    sampling_client, tokenizer, renderer = setup_clients()

    print("Sampling data")
    # Run async data creation
    asyncio.run(create_data_async(cfg, sampling_client, tokenizer, renderer))
    print(f"Saved data to {cfg.output_file}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
