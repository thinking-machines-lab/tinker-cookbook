from tinker_cookbook.recipes.rubric.data import RubricBasedDatapoint, Rubric
import random
import os


def generate_one(rng: random.Random) -> RubricBasedDatapoint:
    x, y = rng.randint(0, 1000), rng.randint(0, 1000)
    return RubricBasedDatapoint(
        convo=[
            {"role": "user", "content": "What is 4 + 5?"},
            {"role": "assistant", "content": "9"},
            {"role": "user", "content": f"What is {x} + {y}?"},
        ],
        rubric_items=[Rubric(rubric_str=f"Does the chatbot correctly get the answer {x + y}?")],
    )


def generate_dataset(
    num_train: int, num_test: int, seed: int, write_dir: str = "tinker_cookbook/example_data/"
) -> tuple[str, str]:
    random.seed(seed)
    rng = random.Random(seed)
    total_datapoints = num_train + num_test
    datapoints = [generate_one(rng) for _ in range(total_datapoints)]

    train_datapoints = datapoints[:num_train]
    train_jsonl_path = os.path.join(write_dir, "example_rubric_train.jsonl")
    with open(train_jsonl_path, "w") as f:
        for datapoint in train_datapoints:
            f.write(datapoint.to_json() + "\n")
    print(f"Generated {len(train_datapoints)} train datapoints in {train_jsonl_path}")

    test_datapoints = datapoints[num_train:]
    test_jsonl_path = os.path.join(write_dir, "example_rubric_test.jsonl")
    with open(test_jsonl_path, "w") as f:
        for datapoint in test_datapoints:
            f.write(datapoint.to_json() + "\n")
    print(f"Generated {len(test_datapoints)} test datapoints in {test_jsonl_path}")

    return train_jsonl_path, test_jsonl_path


if __name__ == "__main__":
    train_jsonl_path, test_jsonl_path = generate_dataset(num_train=10000, num_test=1000, seed=42)
    print(f"Generated train dataset in {train_jsonl_path}")
    print(f"Generated test dataset in {test_jsonl_path}")
