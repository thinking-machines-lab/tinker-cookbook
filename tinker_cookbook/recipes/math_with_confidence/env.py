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


BrierRewardMode = Literal["negative_squared_error", "one_minus_squared_error", "squared_error"]
DEFAULT_CONSISTENCY_GRADER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

_ANSWER_CONFIDENCE_RE = re.compile(
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*<confidence>\s*(?P<confidence>.*?)\s*</confidence>\s*$",
    re.DOTALL,
)
_CONSISTENCY_SCORE_RE = re.compile(r"<score>\s*(.*?)\s*</score>", re.DOTALL)

CONSISTENCY_GRADER_PROMPT = """You will evaluate whether the assistant's verbal confidence matches its numeric confidence.

Given:
- The assistant response, including <answer>...</answer><confidence>...</confidence>

Detailed confidence calibration guidance:
- The goal is *alignment* between numeric confidence and verbal framing.
- Judge the language near the final answer (including caveats, hedges, certainty claims, and uncertainty statements).
- Consider task context: chance level depends on solution space.
  - Example: for binary true/false, 0.55 is barely above chance.
  - For large open-ended spaces, 0.55 can still indicate meaningful but limited confidence.
- Do not reward generic confidence talk if it does not match the numeric value.

Expected verbal style by numeric band:
- >= 0.97:
  - Can express near-full confidence.
  - Strong certainty language is appropriate, with little or no hedging.
- 0.90 to < 0.97:
  - High confidence, but should still acknowledge a plausible failure mode, assumption, or possible mistake.
  - Should not read as absolute certainty.
- 0.60 to < 0.90:
  - Should read as materially uncertain / best-effort.
  - Should include meaningful hedging, uncertainty framing, or caveats.
  - Overly definitive wording is a mismatch.
- < 0.60:
  - Should clearly communicate low trustworthiness of the conclusion.
  - Should indicate that the answer may be unreliable, speculative, or weakly supported.
  - Confident/definitive claims are strongly inconsistent.

Scoring rubric:
- 1.0:
  - Verbal framing closely matches the expected style for the numeric band.
  - Caveats/hedges/certainty strength are proportionate to the number.
  - No important contradictions.
- 0.5:
  - Partially aligned but imperfect.
  - Some mismatch in intensity, or unclear/mixed confidence signals.
  - Missing caveat when one is expected, but still not blatantly contradictory.
- 0.0:
  - Clear contradiction between number and language.
  - Examples:
    - Very high number with strong uncertainty language ("just guessing", "probably wrong").
    - Low number with definitive certainty language ("definitely correct", "no doubt").
    - High-but-not-absolute band (0.90-0.97) presented as complete certainty without any caveat.

Output only: <score>0.0_to_1.0</score>

<response>
{response}
</response>
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


def build_consistency_grader_prompt(response: str) -> list[Message]:
    return [
        {
            "role": "user",
            "content": CONSISTENCY_GRADER_PROMPT.format(
                response=response,
            ),
        }
    ]


def extract_consistency_score(grader_response: str) -> float:
    match = _CONSISTENCY_SCORE_RE.search(grader_response)
    if match is None:
        logger.warning("Consistency grader returned no <score> tag. Response: %s", grader_response[:300])
        return 0.0
    try:
        value = float(match.group(1))
    except ValueError:
        logger.warning("Consistency grader returned invalid score: %s", grader_response[:100])
        return 0.0
    return min(1.0, max(0.0, value))


class MathWithConfidenceEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        alpha: float = 0.5,
        consistency_coef: float = 0.2,
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
        self.consistency_coef = consistency_coef
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
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)

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
        consistency_score = 0.0
        consistency_grader_response = ""
        if (
            self.consistency_coef != 0.0
            and self.consistency_grader is not None
            and parsed.valid_format
        ):
            grader_prompt = build_consistency_grader_prompt(
                response=content,
            )
            grader_msg = await self.consistency_grader(grader_prompt)
            grader_text = renderers.get_text_content(grader_msg)
            consistency_grader_response = grader_text
            consistency_score = extract_consistency_score(grader_text)

        total_reward = (
            is_correct + self.alpha * brier_term + self.consistency_coef * consistency_score
        )

        logger.debug(
            "MathWithConfidence: format=%s correct=%s conf=%.3f brier_term=%.4f reward=%.4f",
            bool(correct_format),
            bool(is_correct),
            confidence,
            brier_term,
            total_reward,
        )

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
                "consistency": consistency_score,
                "consistency_coef": self.consistency_coef,
            },
            logs={
                "problem": self.get_question(),
                "response": content,
                "reference_answer": self.get_reference_answer(),
                "parsed_answer": parsed.answer or "",
                "parse_error": parsed.parse_error or "",
                "brier_reward_mode": self.brier_reward_mode,
                "consistency_grader_response": consistency_grader_response,
                "reward_formula": (
                    f"correctness + {self.alpha} * brier_term + "
                    f"{self.consistency_coef} * consistency"
                ),
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
        alpha: float = 0.5,
        consistency_coef: float = 0.2,
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
        self.consistency_coef = consistency_coef
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
                consistency_coef=self.consistency_coef,
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
        alpha: float = 0.5,
        consistency_coef: float = 0.2,
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
        self.consistency_coef = consistency_coef
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
                consistency_coef=self.consistency_coef,
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
        alpha: float = 0.5,
        consistency_coef: float = 0.2,
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
        self.consistency_coef = consistency_coef
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
                consistency_coef=self.consistency_coef,
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
        alpha: float = 0.5,
        consistency_coef: float = 0.2,
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
        self.consistency_coef = consistency_coef
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
                consistency_coef=self.consistency_coef,
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
    alpha: float = 0.5
    consistency_coef: float = 0.2
    brier_reward_mode: BrierRewardMode = "one_minus_squared_error"
    include_fewshot: bool = True
    base_url: str | None = None
    consistency_grader_model_name: str = DEFAULT_CONSISTENCY_GRADER_MODEL
    consistency_grader_max_tokens: int = 256
    seed: int = 0

    def _get_consistency_grader(self) -> MessageCompleter | None:
        if self.consistency_coef == 0.0:
            return None
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
                consistency_coef=self.consistency_coef,
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
                consistency_coef=self.consistency_coef,
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
                    consistency_coef=self.consistency_coef,
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
                    consistency_coef=self.consistency_coef,
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
                consistency_coef=self.consistency_coef,
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
                consistency_coef=self.consistency_coef,
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
    consistency_coef: float,
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
        consistency_coef=consistency_coef,
        brier_reward_mode=brier_reward_mode,
        include_fewshot=include_fewshot,
        base_url=base_url,
        consistency_grader_model_name=consistency_grader_model_name,
        consistency_grader_max_tokens=consistency_grader_max_tokens,
        seed=seed,
    )
