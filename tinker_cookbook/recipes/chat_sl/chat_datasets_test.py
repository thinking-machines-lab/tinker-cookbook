import datasets

from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


def test_lean_workbook_tactics_builder_filters_and_formats(monkeypatch):
    rows = [
        {
            "formal_statement": f"theorem proved_{i} : True := by",
            "state_before": "|- True",
            "tactic": f"tactic_{i}",
            "status": "proved",
        }
        for i in range(514)
    ]
    rows.extend(
        [
            {
                "formal_statement": "theorem empty : True := by",
                "state_before": "|- True",
                "tactic": "",
                "status": "proved",
            },
            {
                "formal_statement": "theorem failed : True := by",
                "state_before": "|- True",
                "tactic": "sorry",
                "status": "failed",
            },
        ]
    )
    hf_dataset = datasets.DatasetDict({"train": datasets.Dataset.from_list(rows)})

    def fake_load_dataset(name: str) -> datasets.DatasetDict:
        assert name == "internlm/Lean-Workbook"
        return hf_dataset

    monkeypatch.setattr(chat_datasets.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        chat_datasets.LeanWorkbookTacticsBuilder,
        "renderer",
        property(lambda self: object()),
    )

    examples: list[list[dict[str, str]]] = []

    def fake_conversation_to_datum(
        messages: list[dict[str, str]],
        renderer: object,
        max_length: int | None,
        train_on_what: TrainOnWhat,
    ) -> object:
        examples.append(messages)
        assert max_length == 1024
        assert train_on_what == TrainOnWhat.LAST_ASSISTANT_MESSAGE
        return messages

    monkeypatch.setattr(chat_datasets, "conversation_to_datum", fake_conversation_to_datum)

    builder = chat_datasets.LeanWorkbookTacticsBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer="Qwen/Qwen3-8B",
            renderer_name="qwen3",
            max_length=1024,
            batch_size=1,
            train_on_what=None,
        ),
    )
    train_dataset, test_dataset = builder()

    assert len(train_dataset) == 2
    assert len(test_dataset) == 512
    train_dataset.get_batch(0)
    test_dataset.get_batch(0)

    assert len(examples) == 2
    assert all(example[0]["role"] == "user" for example in examples)
    assert all(example[1]["role"] == "assistant" for example in examples)
    assert all(example[1]["content"].startswith("tactic_") for example in examples)
    assert all("[THEOREM]" in example[0]["content"] for example in examples)
    assert all("[GOAL]" in example[0]["content"] for example in examples)
    assert all(example[0]["content"].endswith("[PROOFSTEP]") for example in examples)
