import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial 14: Prompt Distillation

    **Prompt distillation** (also called context distillation) transfers knowledge embedded in a system prompt into the model's weights. The idea:

    1. **Teacher**: Generate labels using a detailed system prompt with classification rules
    2. **Student**: Train on those labels but *without* the system prompt

    After training, the student produces the same outputs as the teacher, but without the inference-time cost of processing the long prompt.

    ```
    Teacher (inference):  [System: 70 lines of rules] + [Text to classify] -> "Final Answer: en"
    Student (training):   [Text to classify] -> "Final Answer: en"   (learns the mapping)
    Student (inference):  [Text to classify] -> "Final Answer: en"   (no system prompt needed)
    ```

    **Our task:** We use a language classification prompt -- a 70-line system prompt with script-based rules, Latin-script heuristics, and edge-case handling. The teacher classifies multilingual text into language codes (en, fr, es, zh, ar, etc.). The student learns to do the same classification with just the raw text as input.
    """)
    return


@app.cell
def _():
    import re

    import tinker
    from tinker import types

    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return get_tokenizer, re, renderers, tinker, types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 -- The classification prompt

    This is the teacher's system prompt -- a detailed set of rules for classifying text into one of 13 language codes. It handles script detection, Latin-script heuristics, mixed-language text, transliteration, and edge cases. At inference time, this prompt consumes hundreds of tokens per request.
    """)
    return


@app.cell
def _():
    CLASSIFICATION_PROMPT = """You are a precise language classifier.

Goal: Classify the language of the provided text into exactly one of these labels:
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French),
hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese),
zh (Chinese - Simplified), ot (Other/Unknown).

Instructions:
1) Preprocess carefully (without changing the intended meaning):
   - Trim whitespace.
   - Ignore URLs, emails, file paths, hashtags, user handles, and emojis.
   - Ignore numbers, math expressions, and standalone punctuation.
   - If there is code, IGNORE code syntax and focus ONLY on human language in comments and string literals.
   - If after ignoring the above there are no alphabetic letters left, output 'ot'.

2) Script-based rules (highest priority):
   - Devanagari script -> hi.
   - Greek script -> el.
   - Cyrillic script -> ru.
   - Han characters -> zh.
   - Arabic script -> ar vs ur (check for Urdu-specific letters).

3) Latin-script heuristics:
   - vi: Vietnamese-specific diacritics.
   - tr: Turkish-specific letters and function words.
   - de: umlauts or eszett and German function words.
   - es: ñ, inverted punctuation, Spanish function words.
   - fr: French diacritics and function words.
   - en: default among Latin languages if strong evidence for others is absent.

4) When in doubt, choose 'ot' rather than guessing.

Output format:
- Respond with EXACTLY one line: "Final Answer: xx"
- Where xx is one of {ar, de, el, en, es, fr, hi, ru, tr, ur, vi, zh, ot} and nothing else.

Text to classify:
{text}"""

    print(f"Classification prompt: {len(CLASSIFICATION_PROMPT)} characters")
    print(f"(This is ~{len(CLASSIFICATION_PROMPT.split())} words the teacher sees per request)")
    return (CLASSIFICATION_PROMPT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- Sample teacher labels

    We pick a diverse set of multilingual sentences and ask the teacher to classify each one. The teacher sees the full system prompt; the student will only see the raw text.
    """)
    return


@app.cell
def _():
    # Diverse multilingual examples -- one per language
    SENTENCES = [
        ("And he said, Mama, I'm home.", "en"),
        ("Et il a dit, maman, je suis à la maison.", "fr"),
        ("Y él dijo: Mamá, estoy en casa.", "es"),
        ("und er hat gesagt, Mama ich bin daheim.", "de"),
        ("Ve Anne, evdeyim dedi.", "tr"),
        ("Và anh ấy nói, Mẹ, con đã về nhà.", "vi"),
        ("他说，妈妈，我回来了。", "zh"),
        ("और उसने कहा, माँ, मैं घर आया हूं।", "hi"),
        ("И он сказал: Мама, я дома.", "ru"),
        ("Και είπε, Μαμά, έφτασα στο σπίτι.", "el"),
        ("وقال، ماما، لقد عدت للمنزل.", "ar"),
        ("اور اس نے کہا امّی، میں گھر آگیا ہوں۔", "ur"),
    ]

    print(f"Test sentences: {len(SENTENCES)} across {len(set(l for _, l in SENTENCES))} languages")
    for _text, _label in SENTENCES[:4]:
        print(f"  [{_label}] {_text[:50]}")
    print(f"  ... and {len(SENTENCES) - 4} more")
    return (SENTENCES,)


@app.cell
async def _(CLASSIFICATION_PROMPT, SENTENCES, get_tokenizer, re, renderers, tinker, types):
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer("qwen3", tokenizer)

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(base_model=MODEL_NAME)

    # Generate teacher labels with the full classification prompt
    teacher_labels = []
    for _text, _expected in SENTENCES:
        _teacher_messages = [
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": _text},
        ]
        _prompt = renderer.build_generation_prompt(_teacher_messages)
        _result = await sampling_client.sample_async(
            prompt=_prompt,
            sampling_params=types.SamplingParams(max_tokens=50, temperature=0.0),
            num_samples=1,
        )
        _response = tokenizer.decode(_result.sequences[0].tokens)
        _match = re.search(r"Final Answer:\s*(\w+)", _response)
        _label = _match.group(1) if _match else "??"
        teacher_labels.append(_label)
        _status = "OK" if _label == _expected else "WRONG"
        print(f"  [{_expected}] -> [{_label}] {_status:5s}  {_text[:45]}")

    _correct = sum(1 for (_t, _e), _l in zip(SENTENCES, teacher_labels) if _l == _e)
    print(f"\nTeacher accuracy: {_correct}/{len(SENTENCES)}")

    return (
        MODEL_NAME,
        renderer,
        service_client,
        teacher_labels,
        tokenizer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- Build student training data

    The student training data pairs each sentence with the teacher's label, but **without** the system prompt. The student sees only: `[user: text] -> [assistant: Final Answer: xx]`.
    """)
    return


@app.cell
def _(SENTENCES, renderer, teacher_labels):
    from tinker_cookbook.supervised.data import conversation_to_datum

    student_data = []
    for (_text, _expected), _label in zip(SENTENCES, teacher_labels):
        _student_messages = [
            {"role": "user", "content": _text},
            {"role": "assistant", "content": f"Final Answer: {_label}"},
        ]
        _datum = conversation_to_datum(_student_messages, renderer, max_length=512)
        student_data.append(_datum)

    print(f"Built {len(student_data)} training examples")
    for i, _datum in enumerate(student_data):
        _w = _datum.loss_fn_inputs["weights"]
        _w_list = _w.tolist() if hasattr(_w, "tolist") else list(_w)
        _n_train = sum(1 for w in _w_list if w > 0)
        print(f"  Example {i:2d}: {_datum.model_input.length:4d} total tokens, {_n_train:3d} trained tokens")
    return (student_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 -- Train the student

    Standard SFT on the teacher's labels. The student learns to map raw text to language codes without seeing the classification rules.
    """)
    return


@app.cell
async def _(MODEL_NAME, service_client, student_data, tinker):
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME,
        rank=32,
    )

    adam_params = tinker.AdamParams(learning_rate=2e-4)
    for _step in range(10):
        _fwd_bwd_future = await training_client.forward_backward_async(student_data, loss_fn="cross_entropy")
        _optim_future = await training_client.optim_step_async(adam_params)
        _fwd_bwd_result = await _fwd_bwd_future.result_async()
        _loss = _fwd_bwd_result.metrics["loss:sum"]
        await _optim_future.result_async()
        print(f"Step {_step:2d}: loss = {_loss:.4f}")

    return (training_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 -- Evaluate: student vs base model

    The key test: can the student classify languages **without** the 70-line system prompt? We compare the trained student against the base model (which has never seen the classification task).
    """)
    return


@app.cell
async def _(SENTENCES, re, renderer, tinker, tokenizer, training_client, types):
    student_client = await training_client.save_weights_and_get_sampling_client_async()

    _student_correct = 0
    print(f"{'Text':<45s}  {'Expected':>8s}  {'Student':>8s}  {'Match':>5s}")
    print("-" * 75)

    for _text, _expected in SENTENCES:
        _student_messages = [{"role": "user", "content": _text}]
        _prompt = renderer.build_generation_prompt(_student_messages)
        _result = await student_client.sample_async(
            prompt=_prompt,
            sampling_params=types.SamplingParams(max_tokens=20, temperature=0.0),
            num_samples=1,
        )
        _response = tokenizer.decode(_result.sequences[0].tokens)
        _match = re.search(r"Final Answer:\s*(\w+)", _response)
        _label = _match.group(1) if _match else "??"
        _ok = _label == _expected
        _student_correct += int(_ok)
        print(f"  {_text[:43]:<43s}  {_expected:>8s}  {_label:>8s}  {'OK' if _ok else 'MISS':>5s}")

    print(f"\nStudent accuracy (no system prompt): {_student_correct}/{len(SENTENCES)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When to use prompt distillation

    **Good use cases:**
    - Classification tasks with detailed rule-based prompts (like this language classifier)
    - Baking safety guidelines into the model (no need to send them every call)
    - Enforcing output format (JSON, specific answer patterns) without format instructions
    - Reducing inference cost by removing long system prompts (our prompt was ~200 words)

    **Limitations:**
    - Works best when the system prompt encodes *rules* rather than *world knowledge*
    - Complex prompts may need many diverse examples to distill well
    - The student can only learn behaviors the teacher demonstrates

    **Scaling up:**
    - The production recipe (`tinker_cookbook.recipes.prompt_distillation`) uses 2100 multilingual sentences across 4 epochs
    - For better results: more examples, more training steps, and diverse inputs
    """)
    return


if __name__ == "__main__":
    app.run()
