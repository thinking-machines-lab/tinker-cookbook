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

    Teacher generates with a rich system prompt; student learns without it.

    **Context distillation** (also called prompt distillation) transfers knowledge embedded in a system prompt into the model's weights. The workflow:

    1. **Teacher**: Generate high-quality completions using a detailed system prompt
    2. **Student**: Train on those completions but *without* the system prompt

    After training, the student behaves as if it had the system prompt, but without the inference-time cost of processing extra tokens.

    ```
    Teacher (inference):  [System: Be concise, use bullets...] + [User question] -> [Good answer]
    Student (training):   [User question] -> [Good answer]   (learns the behavior)
    Student (inference):  [User question] -> [Good answer]   (no system prompt needed)
    ```
    """)
    return


@app.cell
def _():
    import asyncio

    import tinker
    from tinker import types

    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    return asyncio, get_tokenizer, renderers, tinker, types


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1 -- Define teacher and student prompts

    The teacher gets a rich system prompt that encodes the desired behavior. The student gets the same user messages but *without* the system prompt.
    """)
    return


@app.cell
def _():
    TEACHER_SYSTEM_PROMPT = """You are a precise, helpful assistant. Follow these rules:
1. Always structure your response with a one-sentence summary first
2. Use numbered steps for any process or procedure
3. Include a concrete example for every concept
4. Keep total response under 100 words
5. End with a single actionable takeaway"""

    # Example questions for distillation
    QUESTIONS = [
        "How do I make a good cup of coffee?",
        "What is the Pythagorean theorem?",
        "How do I write a for loop in Python?",
        "What causes rainbows?",
        "How do I negotiate a salary raise?",
    ]

    print("Teacher system prompt:")
    print(TEACHER_SYSTEM_PROMPT)
    print(f"\nNumber of questions: {len(QUESTIONS)}")
    return QUESTIONS, TEACHER_SYSTEM_PROMPT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2 -- Generate teacher completions

    We sample completions from the model *with* the system prompt. These become the training targets.
    """)
    return


@app.cell
def _(QUESTIONS, TEACHER_SYSTEM_PROMPT, get_tokenizer, renderers, tinker, types):
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer("qwen3", tokenizer)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    # Generate teacher completions with the rich system prompt
    teacher_completions = []
    for question in QUESTIONS:
        teacher_messages = [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = renderer.build_generation_prompt(teacher_messages)
        result = sampling_client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(max_tokens=200, temperature=0.7),
            num_samples=1,
        ).result()

        completion_text = tokenizer.decode(result.sequences[0].tokens)
        teacher_completions.append(completion_text)
        print(f"Q: {question}")
        print(f"A: {completion_text[:150]}...")
        print()

    return (
        MODEL_NAME,
        renderer,
        sampling_client,
        service_client,
        teacher_completions,
        tokenizer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3 -- Build student training data

    The student training data pairs each question with the teacher's completion, but **without** the system prompt. The model learns to produce teacher-quality outputs from the bare user message.
    """)
    return


@app.cell
def _(QUESTIONS, renderer, teacher_completions, tinker):
    # Build supervised training data: student sees only user message + teacher completion
    student_data = []
    for question, completion in zip(QUESTIONS, teacher_completions):
        # Student conversation: NO system prompt
        student_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": completion},
        ]

        model_input, weights = renderer.build_supervised_example(student_messages)
        datum = tinker.Datum(
            model_input=model_input,
            loss_fn_inputs={"weights": tinker.TensorData.from_list(weights.tolist())},
        )
        student_data.append(datum)

    print(f"Built {len(student_data)} training examples")
    for i, datum in enumerate(student_data):
        n_train_tokens = sum(1 for w in datum.loss_fn_inputs["weights"].data if w > 0)
        print(f"  Example {i}: {datum.model_input.length} total tokens, {n_train_tokens} trained tokens")
    return (student_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4 -- Train the student

    Standard SFT on the teacher's completions. The student learns to reproduce the teacher's behavior without seeing the system prompt.
    """)
    return


@app.cell
def _(MODEL_NAME, service_client, student_data, tinker):
    # Create a training client
    training_client = service_client.create_lora_training_client(
        base_model=MODEL_NAME,
        rank=32,
    )

    # Train for a few steps (small dataset, just for demonstration)
    adam_params = tinker.AdamParams(learning_rate=1e-4)
    for step in range(3):
        fwd_bwd = training_client.forward_backward(student_data, loss_fn="cross_entropy")
        optim = training_client.optim_step(adam_params)
        loss_outputs = fwd_bwd.result().loss_fn_outputs
        nll = sum(o["logprobs"].data[0] for o in loss_outputs) / len(loss_outputs)
        print(f"Step {step}: mean NLL = {nll:.4f}")

    return (training_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 5 -- Compare student vs teacher

    After training, the student should produce structured, concise responses even without the system prompt.
    """)
    return


@app.cell
def _(QUESTIONS, renderer, tinker, tokenizer, training_client, types):
    # Get a sampling client from the trained student
    student_client = training_client.save_weights_and_get_sampling_client()

    # Compare: student (no system prompt) vs teacher behavior
    print("=" * 60)
    for question in QUESTIONS[:2]:
        # Student: no system prompt
        student_messages = [{"role": "user", "content": question}]
        student_prompt = renderer.build_generation_prompt(student_messages)
        student_result = student_client.sample(
            prompt=student_prompt,
            sampling_params=types.SamplingParams(max_tokens=200, temperature=0.7),
            num_samples=1,
        ).result()
        student_text = tokenizer.decode(student_result.sequences[0].tokens)

        print(f"Q: {question}")
        print(f"Student (no system prompt): {student_text[:200]}")
        print("-" * 60)
    return (student_client,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When to use prompt distillation

    **Good use cases:**
    - Baking safety guidelines into the model (no need to send them every call)
    - Enforcing output format (JSON, markdown, bullet points) without format instructions
    - Reducing inference cost by removing long system prompts
    - Creating specialized models from a general-purpose base

    **Limitations:**
    - The student can only learn behaviors the teacher demonstrates
    - Complex system prompts may need many examples to distill well
    - Works best when the system prompt encodes *style* rather than *knowledge*

    **Tips:**
    - Generate multiple teacher completions per question and pick the best
    - Use diverse questions to avoid overfitting to specific topics
    - Evaluate on held-out questions to check generalization
    """)
    return


if __name__ == "__main__":
    app.run()
