import re
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.recipes.math_rl import math_env
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.renderers import Message
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter


BrierRewardMode = Literal["negative_squared_error", "one_minus_squared_error", "squared_error"]
DEFAULT_CONSISTENCY_GRADER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

_ANSWER_CONFIDENCE_RE = re.compile(
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*<confidence>\s*(?P<confidence>.*?)\s*</confidence>\s*$",
    re.DOTALL,
)
_CONSISTENCY_SCORE_RE = re.compile(r"<score>\s*(.*?)\s*</score>", re.DOTALL)

CONSISTENCY_V2_GRADER_PROMPT = """Estimate the assistant's communicated probability that its solution is correct.

You are given only the assistant's response body (without the final <answer> and <confidence> tags).
Infer how confident the assistant *sounds* from its language (certainty, hedging, caveats, uncertainty).

Output a single number in [0, 1]:
- 1.0 means the assistant communicates near-complete certainty.
- 0.0 means the assistant communicates almost no trust in correctness.

Output only: <score>0.0_to_1.0</score>

<response_body>
{response_body}
</response_body>
"""


@dataclass(frozen=True)
class ParsedAnswerConfidence:
    answer: str | None
    confidence: float | None
    valid_format: bool
    parse_error: str | None = None


def parse_answer_and_confidence(text: str) -> ParsedAnswerConfidence:
    match = _ANSWER_CONFIDENCE_RE.search(text)
    if match is None:
        return ParsedAnswerConfidence(
            answer=None,
            confidence=None,
            valid_format=False,
            parse_error="Could not find <answer>...</answer><confidence>...</confidence> at end.",
        )

    answer = match.group("answer").strip()
    confidence_str = match.group("confidence").strip()
    if answer == "":
        return ParsedAnswerConfidence(
            answer=None,
            confidence=None,
            valid_format=False,
            parse_error="Answer tag was empty.",
        )

    try:
        confidence = float(confidence_str)
    except ValueError:
        return ParsedAnswerConfidence(
            answer=answer,
            confidence=None,
            valid_format=False,
            parse_error=f"Confidence is not a float: {confidence_str!r}",
        )

    if not 0.0 <= confidence <= 1.0:
        return ParsedAnswerConfidence(
            answer=answer,
            confidence=None,
            valid_format=False,
            parse_error=f"Confidence must be in [0, 1], got {confidence}.",
        )
    return ParsedAnswerConfidence(answer=answer, confidence=confidence, valid_format=True)


def compute_brier_term(y: float, p: float, mode: BrierRewardMode) -> float:
    squared_error = (p - y) ** 2
    if mode == "negative_squared_error":
        return -squared_error
    if mode == "one_minus_squared_error":
        return 1.0 - squared_error
    if mode == "squared_error":
        return squared_error
    raise ValueError(f"Unsupported brier mode: {mode}")


def build_consistency_v2_grader_prompt(response_body: str) -> list[Message]:
    return [
        {
            "role": "user",
            "content": CONSISTENCY_V2_GRADER_PROMPT.format(response_body=response_body),
        }
    ]


def extract_consistency_score(grader_response: str) -> float:
    match = _CONSISTENCY_SCORE_RE.search(grader_response)
    if match is not None:
        score_str = match.group(1)
    else:
        score_str = grader_response.strip()
    try:
        value = float(score_str)
    except ValueError:
        logger.warning("Consistency grader returned invalid score: %s", grader_response[:100])
        return 0.0
    return min(1.0, max(0.0, value))


def strip_answer_confidence_suffix(response: str) -> str:
    match = _ANSWER_CONFIDENCE_RE.search(response)
    if match is None:
        return response
    return response[: match.start()].rstrip()


class MathWithConfidenceEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        alpha: float = 1.0,
        consistency_v2_coef: float = 0.5,
        brier_reward_mode: BrierRewardMode = "one_minus_squared_error",
        consistency_grader: MessageCompleter | None = None,
    ):
        # format_coef is set to 0.0 because reward is handled explicitly in step()
        super().__init__(renderer=renderer, convo_prefix=convo_prefix, format_coef=0.0)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout
        self.alpha = alpha
        self.consistency_v2_coef = consistency_v2_coef
        self.brier_reward_mode: BrierRewardMode = cast(BrierRewardMode, brier_reward_mode)
        self.consistency_grader = consistency_grader

    @classmethod
    def question_suffix(cls) -> str:
        return (
            " Solve the problem. At the very end of your response, output exactly:"
            " <answer>...</answer><confidence>...</confidence>"
            " where confidence is a decimal number in [0, 1]."
        )

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathWithConfidenceEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": (
                    "Let's spell it out: s t r a w b e r r y, so there are three r's. "
                    "<answer>3</answer><confidence>0.9</confidence>"
                ),
            },
        ]

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        parsed = parse_answer_and_confidence(sample_str)
        return parsed.valid_format

    def check_answer(self, sample_str: str) -> bool:
        parsed = parse_answer_and_confidence(sample_str)
        if not parsed.valid_format or parsed.answer is None:
            return False
        return math_env.safe_grade(parsed.answer, self.answer, self.grader, self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer

    async def step(self, action: list[int]) -> StepResult:
        convo = self.convo_prefix + [{"role": "user", "content": self.get_question()}]
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))

        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
            logtree.log_text(f"Parse success: {parse_success}")

        parsed = parse_answer_and_confidence(content)
        correct_format = float(parse_success) and float(parsed.valid_format)
        is_correct = (
            float(
                parsed.answer is not None
                and math_env.safe_grade(parsed.answer, self.answer, self.grader, self.timeout)
            )
            if parsed.valid_format
            else 0.0
        )
        confidence = parsed.confidence if parsed.confidence is not None else 0.0
        brier_term = compute_brier_term(
            y=is_correct,
            p=confidence,
            mode=cast(BrierRewardMode, self.brier_reward_mode),
        )
        consistency_v2_score = 0.0
        consistency_v2_inferred_confidence = 0.0
        consistency_v2_diff = 0.0
        if (
            self.consistency_grader is not None
            and parsed.valid_format
            and parsed.confidence is not None
        ):
            response_body = strip_answer_confidence_suffix(content)
            grader_prompt = build_consistency_v2_grader_prompt(response_body=response_body)
            grader_msg = await self.consistency_grader(grader_prompt)
            grader_text = renderers.get_text_content(grader_msg)
            consistency_v2_inferred_confidence = extract_consistency_score(grader_text)
            consistency_v2_diff = consistency_v2_inferred_confidence - parsed.confidence
            consistency_v2_score = 1.0 - consistency_v2_diff**2
            logtree.details(
                grader_text,
                summary="Consistency grader response",
                pre=True,
            )

        total_reward = (
            is_correct
            + self.alpha * brier_term
            + self.consistency_v2_coef * consistency_v2_score
        )

        logger.debug(
            "MathWithConfidence: format=%s correct=%s conf=%.3f brier_term=%.4f reward=%.4f",
            bool(correct_format),
            bool(is_correct),
            confidence,
            brier_term,
            total_reward,
        )
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "format_valid": bool(correct_format),
                    "correct": bool(is_correct),
                    "confidence": f"{confidence:.3f}",
                    "brier_reward_mode": self.brier_reward_mode,
                    "brier_term": f"{brier_term:.4f}",
                    "alpha": self.alpha,
                    "consistency_v2": f"{consistency_v2_score:.4f}",
                    "consistency_v2_coef": self.consistency_v2_coef,
                    "consistency_v2_inferred_confidence": f"{consistency_v2_inferred_confidence:.3f}",
                    "consistency_v2_diff": f"{consistency_v2_diff:.3f}",
                    "reward": f"{total_reward:.4f}",
                },
                caption="Reward components",
            )
            if parsed.parse_error is not None:
                logtree.log_text(f"Parse error: {parsed.parse_error}")

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": is_correct,
                "confidence": confidence,
                "brier_term": brier_term,
                "alpha": self.alpha,
                "consistency_v2": consistency_v2_score,
                "consistency_v2_coef": self.consistency_v2_coef,
                "consistency_v2_inferred_confidence": consistency_v2_inferred_confidence,
                "consistency_v2_diff": consistency_v2_diff,
            },
            logs={
                "format_valid": int(parsed.valid_format),
                "parsed_answer": parsed.answer or "",
                "parse_error": parsed.parse_error or "",
                "brier_reward_mode": self.brier_reward_mode,
            },
        )


class MathWithConfidenceDataset(math_env.MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        alpha: float = 1.0,
        consistency_v2_coef: float = 0.5,
        brier_reward_mode: BrierRewardMode = "one_minus_squared_error",
        consistency_grader: MessageCompleter | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split=split,
            seed=seed,
        )
        self.alpha = alpha
        self.consistency_v2_coef = consistency_v2_coef
        self.brier_reward_mode: BrierRewardMode = cast(BrierRewardMode, brier_reward_mode)
        self.consistency_grader = consistency_grader

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            answer = math_env.extract_boxed(x["solution"])
        except ValueError:
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathWithConfidenceEnv,
                x["problem"],
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=self.brier_reward_mode,
                consistency_grader=self.consistency_grader,
            ),
            num_envs=group_size,
        )


class PolarisWithConfidenceDataset(math_env.PolarisDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        alpha: float = 1.0,
        consistency_v2_coef: float = 0.5,
        brier_reward_mode: BrierRewardMode = "one_minus_squared_error",
        consistency_grader: MessageCompleter | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            seed=seed,
        )
        self.alpha = alpha
        self.consistency_v2_coef = consistency_v2_coef
        self.brier_reward_mode: BrierRewardMode = cast(BrierRewardMode, brier_reward_mode)
        self.consistency_grader = consistency_grader

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathWithConfidenceEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=self.brier_reward_mode,
                consistency_grader=self.consistency_grader,
            ),
            num_envs=group_size,
            dataset_name="polaris",
        )


class DeepMathWithConfidenceDataset(math_env.DeepMathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
        alpha: float = 1.0,
        consistency_v2_coef: float = 0.5,
        brier_reward_mode: BrierRewardMode = "one_minus_squared_error",
        consistency_grader: MessageCompleter | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            seed=seed,
        )
        self.alpha = alpha
        self.consistency_v2_coef = consistency_v2_coef
        self.brier_reward_mode: BrierRewardMode = cast(BrierRewardMode, brier_reward_mode)
        self.consistency_grader = consistency_grader

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        problem = x.get("question", "")
        answer = x.get("final_answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathWithConfidenceEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=self.brier_reward_mode,
                consistency_grader=self.consistency_grader,
            ),
            num_envs=group_size,
            dataset_name="deepmath",
        )


class Gsm8kWithConfidenceDataset(math_env.Gsm8kDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        alpha: float = 1.0,
        consistency_v2_coef: float = 0.5,
        brier_reward_mode: BrierRewardMode = "one_minus_squared_error",
        consistency_grader: MessageCompleter | None = None,
    ):
        super().__init__(
            batch_size=batch_size,
            group_size=group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split=split,
            seed=seed,
        )
        self.alpha = alpha
        self.consistency_v2_coef = consistency_v2_coef
        self.brier_reward_mode: BrierRewardMode = cast(BrierRewardMode, brier_reward_mode)
        self.consistency_grader = consistency_grader

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answer = math_env.extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathWithConfidenceEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=self.brier_reward_mode,
                consistency_grader=self.consistency_grader,
            ),
            num_envs=group_size,
        )


@chz.chz
class MathWithConfidenceDatasetBuilder(RLDatasetBuilder):
    dataset_name: Literal["math", "polaris", "deepmath", "gsm8k"] = "math"
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    alpha: float = 1.0
    consistency_v2_coef: float = 0.5
    brier_reward_mode: BrierRewardMode = "one_minus_squared_error"
    include_fewshot: bool = True
    base_url: str | None = None
    consistency_grader_model_name: str = DEFAULT_CONSISTENCY_GRADER_MODEL
    consistency_grader_max_tokens: int = 256
    seed: int = 0

    def _get_consistency_grader(self) -> MessageCompleter | None:
        grader_tokenizer = get_tokenizer(self.consistency_grader_model_name)
        grader_renderer_name = model_info.get_recommended_renderer_name(
            self.consistency_grader_model_name
        )
        grader_renderer = renderers.get_renderer(grader_renderer_name, tokenizer=grader_tokenizer)
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(
            base_model=self.consistency_grader_model_name
        )
        return TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=grader_renderer,
            max_tokens=self.consistency_grader_max_tokens,
            temperature=0.0,
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        consistency_grader = self._get_consistency_grader()
        brier_mode = cast(BrierRewardMode, self.brier_reward_mode)
        convo_prefix: list[renderers.Message] | None = None
        if self.include_fewshot:
            convo_prefix = MathWithConfidenceEnv.standard_fewshot_prefix()

        if self.dataset_name == "math":
            train_ds = MathWithConfidenceDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="train",
                seed=self.seed,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=brier_mode,
                consistency_grader=consistency_grader,
            )
            test_ds = MathWithConfidenceDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="test",
                seed=self.seed,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=brier_mode,
                consistency_grader=consistency_grader,
            )
            return train_ds, test_ds
        if self.dataset_name == "polaris":
            return (
                PolarisWithConfidenceDataset(
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    seed=self.seed,
                    alpha=self.alpha,
                    consistency_v2_coef=self.consistency_v2_coef,
                    brier_reward_mode=brier_mode,
                    consistency_grader=consistency_grader,
                ),
                None,
            )
        if self.dataset_name == "deepmath":
            return (
                DeepMathWithConfidenceDataset(
                    batch_size=self.batch_size,
                    group_size=self.group_size,
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    seed=self.seed,
                    alpha=self.alpha,
                    consistency_v2_coef=self.consistency_v2_coef,
                    brier_reward_mode=brier_mode,
                    consistency_grader=consistency_grader,
                ),
                None,
            )
        if self.dataset_name == "gsm8k":
            train_ds = Gsm8kWithConfidenceDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="train",
                seed=self.seed,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=brier_mode,
                consistency_grader=consistency_grader,
            )
            test_ds = Gsm8kWithConfidenceDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                convo_prefix=convo_prefix,
                split="test",
                seed=self.seed,
                alpha=self.alpha,
                consistency_v2_coef=self.consistency_v2_coef,
                brier_reward_mode=brier_mode,
                consistency_grader=consistency_grader,
            )
            return train_ds, test_ds
        raise ValueError(f"Unknown dataset_name: {self.dataset_name}")


def get_dataset_builder(
    dataset_name: Literal["math", "polaris", "deepmath", "gsm8k"],
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    alpha: float,
    consistency_v2_coef: float,
    brier_reward_mode: BrierRewardMode,
    include_fewshot: bool,
    base_url: str | None,
    consistency_grader_model_name: str,
    consistency_grader_max_tokens: int,
    seed: int,
) -> RLDatasetBuilder:
    return MathWithConfidenceDatasetBuilder(
        dataset_name=dataset_name,
        batch_size=batch_size,
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        group_size=group_size,
        alpha=alpha,
        consistency_v2_coef=consistency_v2_coef,
        brier_reward_mode=brier_reward_mode,
        include_fewshot=include_fewshot,
        base_url=base_url,
        consistency_grader_model_name=consistency_grader_model_name,
        consistency_grader_max_tokens=consistency_grader_max_tokens,
        seed=seed,
    )
