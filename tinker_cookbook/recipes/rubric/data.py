import json
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

import chz

from tinker_cookbook.renderers import (
    Message,
    Role,
)

Conversation: TypeAlias = list[Message]


@dataclass
class Rubric:
    """
    A rubric should specify 1) what counts as a good response, 2) how the grader language model should output the score, and 3) how to extract the score from the grader's response.
    """

    rubric_str: str
    extraction_regex: str = r"<score>(.*)</score>"
    grader_output_format_instruction: str = (
        "Please output your score between 0 and 1 wrapped in <score> ... </score>"
    )

    def _convert_role(self, role: Role) -> str:
        return "Human" if role in ("user", "system") else "Chatbot"

    def _flatten_convo(self, convo: Conversation) -> str:
        """
        Convert the whole conversation (user's turns + assistant's turns) into a single string. E.g.
        \n\nHuman: ....
        \n\nChatbot: ...
        \n\nHuman: ...
        \n\nChatbot: ...
        """
        return "\n\n".join(
            [f"{self._convert_role(message['role'])}: {message['content']}" for message in convo]
        )

    def get_grader_prompt(self, convo: Conversation) -> Conversation:
        """
        Create a prompt for the grader to grade the conversation based on the rubric.
        The prompt separates the context (prior turns) from the completion (last assistant message)
        so the grader focuses on grading the most recent response.
        """
        # Separate context from the completion to grade
        context = convo[:-1]
        completion = convo[-1]

        lines = [
            "I will show you a conversation context, a chatbot completion to grade, and a rubric.",
            "Please grade the chatbot's completion based on the rubric.",
            "",
            "<context>",
            self._flatten_convo(context) if context else "(No prior context)",
            "</context>",
            "",
            "<completion_to_grade>",
            f"Chatbot: {completion['content']}",
            "</completion_to_grade>",
            "",
            "<rubric>",
            self.rubric_str,
            "</rubric>",
            "",
            f"Please grade the chatbot's completion based on the rubric. {self.grader_output_format_instruction}",
        ]
        return [
            {
                "role": "user",
                "content": "\n".join(lines),
            }
        ]

    def extract_score(self, response: str) -> float:
        match = re.search(self.extraction_regex, response, re.DOTALL)
        if match is not None:
            try:
                return float(match.group(1))
            except ValueError:
                print(f"Warning: Failed to extract score from grader response: {response}")
                return 0.0
        else:
            print(f"Warning: Failed to extract score from grader response: {response}")
            return 0.0

    def to_dict(self) -> dict[str, str]:
        return {
            "rubric_str": self.rubric_str,
            "extraction_regex": self.extraction_regex,
            "grader_output_format_instruction": self.grader_output_format_instruction,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(d: dict[str, str]) -> "Rubric":
        return Rubric(
            rubric_str=d["rubric_str"],
            extraction_regex=d["extraction_regex"],
            grader_output_format_instruction=d["grader_output_format_instruction"],
        )

    @staticmethod
    def from_json(json_str: str) -> "Rubric":
        return Rubric.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class RubricBasedDatapoint:
    """
    A rubric-based datapoint contains a conversation and a rubric.
    In this task, the policy model sees the conversation, create a response, and then the grader language model grades the response based on the rubric.
    """

    convo: Conversation
    rubric_items: Sequence[Rubric]

    def to_json(self) -> str:
        return json.dumps(
            {
                "convo": self.convo,
                "rubric_items": [rubric.to_dict() for rubric in self.rubric_items],
            }
        )

    @staticmethod
    def from_json(json_str: str) -> "RubricBasedDatapoint":
        d = json.loads(json_str)
        return RubricBasedDatapoint(
            convo=d["convo"],
            rubric_items=[Rubric.from_dict(rubric) for rubric in d["rubric_items"]],
        )


@chz.chz
class RubricDatapointListBuilder:
    def __call__(self) -> Sequence[RubricBasedDatapoint]:
        """Load and return a sequence of rubric-based datapoints."""
        raise NotImplementedError("Subclass must implement this method")


@chz.chz
class RubricDatapointListBuilderFromJsonl(RubricDatapointListBuilder):
    jsonl_path: str

    def __call__(self) -> Sequence[RubricBasedDatapoint]:
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(
                f"Data file not found: {self.jsonl_path}\n"
                f"Please generate the example data first by running:\n"
                f"  python -m tinker_cookbook.recipes.rubric.generate_data"
            )
        datapoints = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                datapoints.append(RubricBasedDatapoint.from_json(line))
        return datapoints


@chz.chz
class PrometheusDatapointListBuilder(RubricDatapointListBuilder):
    data_path: str = "prometheus-eval/Feedback-Collection"

    def __call__(self) -> Sequence[RubricBasedDatapoint]:
        from datasets import load_dataset

        train_dataset = load_dataset(self.data_path)["train"]
        return [self.build_rubric_datapoint(item) for item in train_dataset]  # type: ignore

    def build_rubric_datapoint(self, item: dict[str, Any]) -> RubricBasedDatapoint:
        convo: Conversation = [
            {"role": "user", "content": item["orig_instruction"]},
        ]

        rubric_lines = [
            f"Your job is to evaluate the following: {item['orig_criteria']}. Your response should be a score between 1 to 5.",
            "Here is the calibration for each score:",
        ]
        for i in range(1, 6):
            rubric_lines.append(f"<score>{i}.0</score>: {item[f'orig_score{i}_description']}")
        rubric_lines.append(
            f"Here is a reference response that achieved a score of 5: {item['orig_reference_answer']}"
        )
        rubric_text = "\n".join(rubric_lines)

        rubric = Rubric(
            rubric_str=rubric_text,
            extraction_regex=r"<score>(.*)</score>",
            grader_output_format_instruction="Please output your score between 1 and 5 wrapped in <score> ... </score>",
        )

        return RubricBasedDatapoint(
            convo=convo,
            rubric_items=[rubric],
        )
