import asyncio

from tinker_cookbook.recipes.math_rl.math_env import MathDatasetBuilder


def test_math_dataset_builder():
    builder = MathDatasetBuilder(
        batch_size=1,
        model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        renderer_name="qwen3_instruct",
        group_size=1,
    )
    train_dataset, test_dataset = asyncio.run(builder())

    # Basic dataset statistics
    assert len(train_dataset) == 12_000
    assert len(test_dataset) == 500
    assert len(train_dataset.get_batch(0)) == 1
    assert len(test_dataset.get_batch(0)) == 1

    # Check for contamination of train and test sets
    test_questions = set()
    for i in range(len(test_dataset)):
        batch = test_dataset.get_batch(index=i)
        test_questions.add(batch[0].env_thunk().get_question())  # pyright: ignore
    for i in range(len(train_dataset)):
        batch = train_dataset.get_batch(index=i)
        assert batch[0].env_thunk().get_question() not in test_questions  # pyright: ignore


if __name__ == "__main__":
    test_math_dataset_builder()
