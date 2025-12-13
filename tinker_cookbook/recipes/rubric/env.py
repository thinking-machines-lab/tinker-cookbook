from tinker_cookbook.rl.types import (
    Action,
    Env,
    StepResult,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)
from tinker_cookbook.renderers import Message, Renderer, Role
from typing import TypeAlias
from tinker_cookbook.completers import MessageCompleter, StopCondition, TinkerMessageCompleter
from tinker.types import ModelInput
from dataclasses import dataclass
from typing import Sequence
import re
import json
import chz
import tinker
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
import asyncio
from tinker_cookbook import model_info


Conversation: TypeAlias = list[Message]

# ANSI color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


@dataclass
class Rubric:
    """
    A rubric should specify 1) what counts as a good response, 2) how the grader language model should output the score, and 3) how to extract the score from the grader's response.
    """
    rubric_str: str
    extraction_regex: str = r"<score>(.*)</score>"
    grader_output_format_instruction: str = "Please output your score between 0 and 1 wrapped in <score> ... </score>"

    def __convert_role(self, role: Role) -> str:
        return "Human" if role in ("user", "system") else "Chatbot"

    def _flatten_convo(self, convo: Conversation) -> str:
        """
        Convert the whole conversation (user's turns + assistant's turns) into a single string. E.g.
        \n\nHuman: ....
        \n\nChatbot: ...
        \n\nHuman: ...
        \n\nChatbot: ...
        """
        return "\n\n".join([f"{self.__convert_role(message['role'])}: {message['content']}" for message in convo])

    def get_grader_prompt(self, convo: Conversation) -> Conversation:
        """
        Create a prompt for the grader to grade the conversation based on the rubric. The prompt should contain 1) the conversation to be graded, and 2) the rubric. 
        """

        prompt = "I will show you 1) a conversation between a human and a chatbot, and 2) a rubric for grading the conversation. Please grade the conversation based on the rubric."

        prompt += f"Here is the conversation: <conversation>\n\n{self._flatten_convo(convo)} \n\n</conversation>\n\nHere is the rubric: <rubric>\n{self.rubric_str}\n</rubric>\n"
        prompt += f"Please grade the conversation based on the rubric. {self.grader_output_format_instruction}"
        return [
            {
                "role": "user",
                "content": prompt,
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
        return json.dumps({
            "convo": self.convo,
            "rubric_items": [rubric.to_dict() for rubric in self.rubric_items],
        })

    @staticmethod
    def from_json(json_str: str) -> "RubricBasedDatapoint":
        d = json.loads(json_str)
        return RubricBasedDatapoint(
            convo=d["convo"],
            rubric_items=[Rubric.from_dict(rubric) for rubric in d["rubric_items"]],
        )


class RubricGradedEnv(Env):

    def __init__(
        self,
        renderer: Renderer,
        datapoint: RubricBasedDatapoint,
        grader_llm: MessageCompleter,
        debug: bool = False,
    ):
        """
        Initialize the RubricGradedEnv. In this environment, the policy model sees the conversation, create a response, and then the grader language model grades the response based on the rubric.
        """
        self.renderer = renderer
        self.datapoint = datapoint
        self.grader_llm = grader_llm
        self.debug = debug

    @property
    def rubric_items(self) -> Sequence[Rubric]:
        return self.datapoint.rubric_items
    
    @property
    def convo(self) -> Conversation:
        return self.datapoint.convo

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()
    
    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self.renderer.build_generation_prompt(self.convo), self.stop_condition

    async def _grade_with_rubric(self, convo: Conversation, rubric: Rubric) -> float:
        # this is the conversation for the grader
        # effectively it's just one user turn
        grader_prompt = rubric.get_grader_prompt(convo)

        # obtain the response from the grader and convert it to a score
        grader_response = await self.grader_llm(grader_prompt)
        grader_response_content = grader_response["content"]
        assert isinstance(grader_response_content, str), "Grader response content must be a string"
        score = rubric.extract_score(grader_response_content)
        if self.debug:
            print(f"{YELLOW}{'='*80}")
            print(f"DEBUG: First Turn of Grader Prompt")
            print(f"{'='*80}{RESET}")
            print(f"{YELLOW}{grader_prompt[0]['content']}{RESET}\n")

            print(f"{MAGENTA}{'='*80}")
            print(f"DEBUG: Score")
            print(f"{'='*80}{RESET}")
            print(f"{MAGENTA}Grader Response: {grader_response_content}{RESET}\n")
            print(f"{MAGENTA}Extracted Score: {score}{RESET}\n")
        return score
    
    async def step(self, action: Action) -> StepResult:
        # obtain the policy action message
        (policy_action_message, _parse_success) = self.renderer.parse_response(action)

        if self.debug:

            print(f"\n{BLUE}{'='*80}")
            print(f"DEBUG: Original Conversation (self.convo)")
            print(f"{'='*80}{RESET}")
            print(f"{BLUE}{json.dumps(self.convo, indent=2)}{RESET}\n")

            print(f"{GREEN}{'='*80}")
            print(f"DEBUG: Policy Action Message")
            print(f"{'='*80}{RESET}")
            print(f"{GREEN}{json.dumps(policy_action_message, indent=2)}{RESET}\n")
                # this shows the full back-and-forth conversation to the grader
        convo = self.convo + [policy_action_message]

        scores = await asyncio.gather(*[self._grade_with_rubric(convo, rubric_item) for rubric_item in self.rubric_items])
        avg_score = sum(scores) / len(scores)

        return StepResult(
            reward=avg_score,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt(convo),
            next_stop_condition=self.stop_condition,
        )


@dataclass(frozen=True)
class RubricGradedEnvGroupBuilder(EnvGroupBuilder):
    renderer: Renderer
    datapoint: RubricBasedDatapoint
    grader_llm: MessageCompleter
    group_size: int

    async def make_envs(self) -> Sequence[RubricGradedEnv]:
        return [
            RubricGradedEnv(
                renderer=self.renderer,
                datapoint=self.datapoint,
                grader_llm=self.grader_llm,
            ) for _ in range(self.group_size)
        ]


@dataclass(frozen=True)
class RubricGradedDataset(RLDataset):
    renderer: Renderer
    batch_size: int
    group_size: int
    datapoints: Sequence[RubricBasedDatapoint]
    grader_llm: MessageCompleter

    def get_batch(self, index: int) -> Sequence[RubricGradedEnvGroupBuilder]:
        batch = [
            RubricGradedEnvGroupBuilder(
                renderer=self.renderer,
                datapoint=self.datapoints[index * self.batch_size + i],
                grader_llm=self.grader_llm,
                group_size=self.group_size,
            )
            for i in range(self.batch_size)
        ]
        return batch

    def __len__(self) -> int:
        return len(self.datapoints) // self.batch_size


@chz.chz
class RubricGradedDatasetBuilder(RLDatasetBuilder):
    renderer_name: str
    model_name_for_tokenizer: str
    batch_size: int
    train_group_size: int
    test_group_size: int = 1

    train_jsonl_path: str
    test_jsonl_path: str | None = None

    base_url: str | None = None
    grader_llm_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    def _get_datapoints_from_jsonl(self, jsonl_path: str | None) -> Sequence[RubricBasedDatapoint] | None:
        if jsonl_path is None:
            return None
        datapoints = []
        with open(jsonl_path, "r") as f:
            for line in f:
                datapoint = RubricBasedDatapoint.from_json(line)
                datapoints.append(datapoint)
        return datapoints
    
    def _get_grader_llm(self) -> MessageCompleter:
        tokenizer = get_tokenizer(self.grader_llm_name)
        renderer_name = model_info.get_recommended_renderer_name(self.grader_llm_name)
        renderer = get_renderer(name=renderer_name, tokenizer=tokenizer)
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(base_model=self.grader_llm_name)
        return TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=2048
        )
    
    async def __call__(self) -> tuple[RubricGradedDataset, RubricGradedDataset | None]:
        train_datapoints = self._get_datapoints_from_jsonl(self.train_jsonl_path)
        test_datapoints = self._get_datapoints_from_jsonl(self.test_jsonl_path)

        renderer = get_renderer(name=self.renderer_name, tokenizer=get_tokenizer(self.model_name_for_tokenizer))

        assert train_datapoints is not None, "Train datapoints are required"
        train_dataset = RubricGradedDataset(
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            datapoints=train_datapoints,
            grader_llm=self._get_grader_llm(),
        )
        if test_datapoints is None:
            return train_dataset, None
        else:
            test_dataset = RubricGradedDataset(
                renderer=renderer,
                batch_size=len(test_datapoints),
                group_size=self.test_group_size,
                datapoints=test_datapoints,
                grader_llm=self._get_grader_llm(),
            )
            return train_dataset, test_dataset