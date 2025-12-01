from abc import ABC, abstractmethod
from typing import Any, Callable, TypedDict


class GEPADataInstance(TypedDict):
    input: str
    answer: str
    metadata: dict[str, Any]


class GEPATask(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def seed_prompt(self) -> str: ...

    @property
    def prompt_component_name(self) -> str:
        return "system_prompt"

    @abstractmethod
    def load_data(
        self, seed: int = 0
    ) -> tuple[list[GEPADataInstance], list[GEPADataInstance], list[GEPADataInstance]]: ...

    def score(self, response: str, answer: str, metadata: dict[str, Any] | None = None) -> float:
        response_lower = response.lower().strip()
        answer_lower = answer.lower().strip()
        return 1.0 if answer_lower in response_lower else 0.0


TASK_REGISTRY: dict[str, type[GEPATask]] = {}


def register_task(name: str) -> Callable[[type[GEPATask]], type[GEPATask]]:
    def decorator(cls: type[GEPATask]) -> type[GEPATask]:
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


def get_task(name: str, **kwargs: Any) -> GEPATask:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown GEPA task: {name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name](**kwargs)


def list_tasks() -> list[str]:
    return list(TASK_REGISTRY.keys())


@register_task("aime")
class AIMETask(GEPATask):
    @property
    def name(self) -> str:
        return "aime"

    @property
    def seed_prompt(self) -> str:
        return (
            "You are a helpful assistant. You are given a question and you need to answer it. "
            "The answer should be given at the end of your response in exactly the format '### <final answer>'"
        )

    def load_data(
        self, seed: int = 0
    ) -> tuple[list[GEPADataInstance], list[GEPADataInstance], list[GEPADataInstance]]:
        import gepa.examples.aime

        raw_train, raw_val, raw_test = gepa.examples.aime.init_dataset()

        def convert(items: list[dict]) -> list[GEPADataInstance]:
            return [
                GEPADataInstance(
                    input=item["input"],
                    answer=str(item["answer"]),
                    metadata=item.get("additional_context", {}),
                )
                for item in items
            ]

        return convert(raw_train), convert(raw_val), convert(raw_test)

    def score(self, response: str, answer: str, metadata: dict[str, Any] | None = None) -> float:
        import re

        answer_match = re.search(r"###\s*(\d+)", answer)
        expected = answer_match.group(1) if answer_match else answer.strip()

        response_match = re.search(r"###\s*(\d+)", response)
        if response_match:
            return 1.0 if response_match.group(1) == expected else 0.0

        return 1.0 if expected in response else 0.0


@register_task("gsm8k")
class GSM8KTask(GEPATask):
    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def seed_prompt(self) -> str:
        return (
            "You are a helpful math tutor. Solve the problem step by step. "
            "Provide your final numerical answer at the end in the format: #### <answer>"
        )

    def load_data(
        self, seed: int = 0
    ) -> tuple[list[GEPADataInstance], list[GEPADataInstance], list[GEPADataInstance]]:
        import random

        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main")
        train_data = list(ds["train"])
        test_data = list(ds["test"])

        random.Random(seed).shuffle(train_data)

        def extract_answer(answer_text: str) -> str:
            import re

            match = re.search(r"####\s*(.+)", answer_text)
            if match:
                return match.group(1).strip().replace(",", "")
            return ""

        def convert(items: list[dict]) -> list[GEPADataInstance]:
            result = []
            for item in items:
                answer = extract_answer(item["answer"])
                if answer:
                    result.append(
                        GEPADataInstance(
                            input=item["question"],
                            answer=answer,
                            metadata={"solution": item["answer"]},
                        )
                    )
            return result

        train_converted = convert(train_data)
        split_idx = len(train_converted) // 2
        trainset = train_converted[:split_idx]
        valset = train_converted[split_idx:]
        testset = convert(test_data)

        return trainset, valset, testset

    def score(self, response: str, answer: str, metadata: dict[str, Any] | None = None) -> float:
        import re

        match = re.search(r"####\s*([^\n]+)", response)
        if match:
            extracted = match.group(1).strip().replace(",", "")
            return 1.0 if extracted == answer else 0.0

        return 1.0 if answer in response else 0.0


@register_task("hotpotqa")
class HotpotQATask(GEPATask):
    """Multi-hop question answering using HotpotQA distractor setting."""

    @property
    def name(self) -> str:
        return "hotpotqa"

    @property
    def seed_prompt(self) -> str:
        return (
            "You are a helpful assistant that answers questions by reasoning over multiple documents. "
            "Read the provided context carefully and answer the question. Think step by step, "
            "connecting information from different paragraphs. Provide your final answer at the end "
            "in the format: #### <answer>"
        )

    def load_data(
        self, seed: int = 0
    ) -> tuple[list[GEPADataInstance], list[GEPADataInstance], list[GEPADataInstance]]:
        import random

        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
        train_data = list(ds["train"])
        val_data = list(ds["validation"])

        random.Random(seed).shuffle(train_data)
        random.Random(seed).shuffle(val_data)

        def format_context(context: dict) -> str:
            titles = context["title"]
            sentences_list = context["sentences"]
            parts = []
            for title, sentences in zip(titles, sentences_list):
                text = "".join(sentences)
                parts.append(f"[{title}]\n{text}")
            return "\n\n".join(parts)

        def convert(items: list[dict]) -> list[GEPADataInstance]:
            return [
                GEPADataInstance(
                    input=f"Context:\n{format_context(item['context'])}\n\nQuestion: {item['question']}",
                    answer=item["answer"],
                    metadata={
                        "question_type": item.get("type", ""),
                        "level": item.get("level", ""),
                        "supporting_facts": item.get("supporting_facts", {}),
                    },
                )
                for item in items
            ]

        train_converted = convert(train_data[:2000])
        test_converted = convert(val_data[:500])

        split_idx = len(train_converted) // 2
        return train_converted[:split_idx], train_converted[split_idx:], test_converted

    def score(self, response: str, answer: str, metadata: dict[str, Any] | None = None) -> float:
        import re
        import string

        def normalize(text: str) -> str:
            text = text.lower()
            text = re.sub(r"\b(a|an|the)\b", " ", text)
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = " ".join(text.split())
            return text

        match = re.search(r"####\s*(.+?)(?:\n|$)", response)
        if match:
            extracted = match.group(1).strip()
        else:
            lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
            extracted = lines[-1] if lines else ""

        norm_extracted = normalize(extracted)
        norm_answer = normalize(answer)

        if norm_answer == norm_extracted:
            return 1.0

        if norm_answer in norm_extracted:
            return 0.8

        pred_tokens = set(norm_extracted.split())
        gold_tokens = set(norm_answer.split())

        if not pred_tokens or not gold_tokens:
            return 0.0

        common = pred_tokens & gold_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1 * 0.5
