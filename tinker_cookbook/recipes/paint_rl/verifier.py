"""Multimodal rewards (LLM-as-a-Verifier, LLM-as-a-Judge) and the held-out evaluator.

The policy writes code; the *reward* comes from a multimodal model that looks
at the rendered paintings. Two interchangeable training rewards share the
same model, thinking effort, token budget, client and concurrency, so a run
with one is directly comparable to a run with the other:

* :class:`PairwiseVerifier` — LLM-as-a-Verifier, built on
  ``llm_verifier.compare`` from the `llm-verifier` package
  (https://github.com/llm-as-a-verifier/llm-as-a-verifier). Every comparison
  shows the verifier two paintings and ONE criterion and asks for a letter
  grade A-T per painting; the grade is read as the *expectation over the
  verifier's token distribution* at each ``<score_X>`` tag, not as the sampled
  letter, so a 5% chance of "A" still moves the reward. Scores are averaged
  over criteria and ``n_evaluations`` repeats, odd repeats swapping the two
  paintings so positional bias cancels. A rollout group is scored with a full
  **round-robin tournament**: every unordered pair of compiled paintings is
  compared, so a painting's reward is its mean grade against every other
  painting in its group.

* :class:`AbsoluteJudge` — the LLM-as-a-Judge baseline: each painting is
  scored on its own on the verifier's three criteria, one call each, as a
  written letter A-J (a 1-10 scale, A best) read from the judge's reply and
  mapped to [0, 1]; its reward is the mean of the three. Paintings with the same scores get the same
  reward (and so the same advantage); how often the top of a group is tied is
  reported as ``scorer/top_tie``.

  Both rewards use ``thinkingmachines/Inkling-Small`` by default, reached
  through Tinker's OpenAI-compatible endpoint. That endpoint returns token
  logprobs but does not support assistant prefill, so the verifier's client
  is set up to read the letter distribution from the model's own sampled
  score tags (the ``llm_verifier`` path for hosted APIs).

* :class:`HeldoutEvaluator` — the held-out yardstick: DeepSeek-V4.1-Flash
  (``deepseek-flash`` on DeepSeek's OpenAI-compatible API, since Tinker takes no
  images for it) scores each held-out painting on its own on each
  of the verifier's three criteria, one written A-J letter per criterion,
  with the judge's prompts, normalized to [0, 1]. It
  never trains anything; it gives every checkpoint of every
  run, verifier or judge, a score on one fixed scale.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import chz
from PIL import Image

try:
    import llm_verifier
except ImportError as e:  # pragma: no cover - environment specific
    raise ImportError("paint_rl needs the llm-verifier package: pip install llm-verifier") from e

if TYPE_CHECKING:
    import tinker

    from tinker_cookbook.renderers import Renderer

logger = logging.getLogger(__name__)

RewardMode = Literal["verifier", "judge"]

# The training reward model and its settings, shared by the verifier and the
# judge so the two arms differ only in how they score.
DEFAULT_REWARD_MODEL = "thinkingmachines/Inkling-Small"
DEFAULT_REWARD_EFFORT = 0.2  # the "low" preset
DEFAULT_REWARD_MAX_TOKENS = 8192
# Concurrent calls to the reward model. A verifier step is 8 groups x 10 pairs
# x 3 criteria x 4 repeats = 960 calls, so the whole step is in flight at once.
DEFAULT_REWARD_CONCURRENCY = 1200

# DeepSeek-V4.1-Flash on DeepSeek's own API. Tinker serves the model but refuses
# image input for it, so the held-out evaluator is the one piece not on Tinker.
DEFAULT_EVALUATOR_MODEL = "deepseek-flash"
DEFAULT_EVALUATOR_API_BASE_URL = "https://api.deepseek.com"
DEFAULT_EVALUATOR_API_KEY_ENV = "DEEPSEEK_API_KEY"

# The judge's and the held-out evaluator's scale: A-J, a standard 1-10 scale
# (A = 10 = 1.0, J = 1 = 0.0).
LETTER_GRADES = 10

TINKER_OPENAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Tinker's named reasoning-effort presets (the OpenAI-compatible endpoint
# takes the name, the CLI takes the float).
EFFORT_PRESETS: dict[float, str] = {
    0.0: "none",
    0.1: "minimal",
    0.2: "low",
    0.7: "medium",
    0.9: "high",
    0.99: "xhigh",
}


# ---------------------------------------------------------------------------
# Criteria (narrow ones, decidable from the image alone)
# ---------------------------------------------------------------------------

GROUND_TRUTH_NOTE = (
    "Judge ONLY what is visible in the rendered painting(s). The code that "
    "produced them is not shown and does not matter. A blank, near-uniform, "
    "or noise-filled canvas is a complete failure regardless of intent. Do "
    "not assume effort or hidden detail: if you cannot see it, it does not count."
)

DEFAULT_CRITERIA: dict[str, str] = {
    "Subject and Prompt Fidelity": (
        "How accurately does the painting depict the requested subject, "
        "scene, and colors? Evaluate whether the main subject is clearly "
        "recognizable, its defining features are present, and any requested "
        "objects, relationships, or color attributes are represented. "
        "Score HIGH for a clear, specific, and faithful depiction. "
        "Score LOW for ambiguous shapes, missing defining features, "
        "incorrect subjects, or omitted scene elements. "
        "Ignore artistic technique and overall visual appeal."
    ),
    "Watercolor Technique": (
        "How skillfully does the painting use watercolor techniques? "
        "Evaluate the quality of translucent washes, pigment variation, "
        "color blending, layered brushstrokes, and the balance between "
        "soft and defined edges. Consider whether the brushwork feels "
        "intentional and whether the requested watercolor style, "
        "such as loose washes, wet-on-wet blending, or ink outlines, "
        "is convincingly executed. "
        "Score HIGH for expressive, controlled, and natural-looking "
        "watercolor effects. Score LOW for flat digital shapes, "
        "mechanical outlines, muddy washes, or random scribbles. "
        "Do not reward complexity or texture for their own sake. "
        "Ignore subject accuracy and overall composition."
    ),
    "Composition and Aesthetic Appeal": (
        "How visually appealing and artistically cohesive is the painting? "
        "Evaluate visual balance, focal point clarity, spatial arrangement, "
        "color harmony, contrast, and the intentional use of negative space. "
        "Consider whether the painting feels complete, expressive, and "
        "visually engaging within its intended style. "
        "Score HIGH for harmonious colors, effective visual hierarchy, "
        "balanced proportions, and a compelling overall impression. "
        "Score LOW for awkward framing, distracting clutter, poor color "
        "relationships, or visually incoherent arrangements. "
        "Minimalist compositions and intentional white space should "
        "not be penalized. Ignore subject accuracy and "
        "watercolor-specific technique."
    ),
}


# What the verifier is told the two "trajectories" are. llm_verifier attaches
# the images in order after the task, so slot A is the first image.
def pairwise_note(granularity: int = 20) -> str:
    last = chr(ord("A") + granularity - 1)
    return (
        "The candidates are watercolor paintings made by code for the painting "
        f"request in the task, attached as images. A letter grade of A means "
        f"outstanding on the criterion and {last} means a complete failure, e.g. "
        "a blank canvas or pure noise. " + GROUND_TRUTH_NOTE
    )


SLOT_A = "Painting A: the FIRST attached image."
SLOT_B = "Painting B: the SECOND attached image."


def resolve_criteria(names: tuple[str, ...]) -> list[tuple[str, str]]:
    """``(name, description)`` for ``names`` (keys of DEFAULT_CRITERIA); empty = all
    three."""
    selected = list(names) or list(DEFAULT_CRITERIA)
    missing = [n for n in selected if n not in DEFAULT_CRITERIA]
    if missing:
        raise ValueError(f"unknown criteria {missing}; known: {list(DEFAULT_CRITERIA)}")
    return [(n, DEFAULT_CRITERIA[n]) for n in selected]


@chz.chz
class VerifierConfig:
    """LLM-as-a-Verifier, called through Tinker's OpenAI-compatible endpoint."""

    model_name: str = DEFAULT_REWARD_MODEL
    # Thinking effort; one of EFFORT_PRESETS.
    effort: float = DEFAULT_REWARD_EFFORT
    # Output budget, shared by the reasoning trace and the verdict.
    max_tokens: int = DEFAULT_REWARD_MAX_TOKENS
    # Repeated verifications K per criterion. Odd repeats swap the prompt
    # slots, so an even K judges every pair equally often in each order.
    n_evaluations: int = 4
    # Letter grades on the scale, A (best) to the granularity-th letter; the
    # reward is the expected grade under the verifier's token distribution.
    # llm_verifier's own scale is 20 (A-T).
    granularity: int = 20
    # True: the expected grade under the token distribution at each score tag.
    # False: the letter the verifier wrote, as an absolute judge reads it.
    use_logprobs: bool = True
    # Criteria names (keys of DEFAULT_CRITERIA); empty = all of them.
    criteria: tuple[str, ...] = ()
    # Longest image side sent to the model.
    image_size: int = 448
    max_concurrency: int = DEFAULT_REWARD_CONCURRENCY
    base_url: str | None = None


@chz.chz
class JudgeConfig:
    """LLM-as-a-Judge baseline: same model, effort, client and criteria as the verifier."""

    model_name: str = DEFAULT_REWARD_MODEL
    effort: float = DEFAULT_REWARD_EFFORT
    max_tokens: int = DEFAULT_REWARD_MAX_TOKENS
    # Criteria names (keys of DEFAULT_CRITERIA); empty = the verifier's three.
    # Each is its own call per painting, and a painting's reward is the mean.
    criteria: tuple[str, ...] = ()
    # One written letter per criterion, A (best) to the Nth letter (worst).
    letter_grades: int = LETTER_GRADES
    # Independent judgments per painting and criterion, averaged.
    n_evaluations: int = 1
    image_size: int = 448
    max_concurrency: int = DEFAULT_REWARD_CONCURRENCY
    base_url: str | None = None


RewardConfig = VerifierConfig | JudgeConfig


@chz.chz
class EvaluatorConfig:
    """The held-out evaluator, on an OpenAI-compatible API (``api_base_url``, the
    default) or Tinker's native sampler (``api_base_url=None``); never enters the
    reward."""

    model_name: str = DEFAULT_EVALUATOR_MODEL
    # Tinker only. None: model_info's recommended renderer for model_name.
    renderer_name: str | None = None
    max_tokens: int = 8192
    temperature: float = 1.0
    n_evaluations: int = 1
    # The verifier's three criteria, each its own call per painting, as
    # the judge scores them.
    criteria: tuple[str, ...] = ()
    # Each criterion graded with one written letter from A (outstanding) to
    # the Nth letter (complete failure), as the judge grades it.
    letter_grades: int = LETTER_GRADES
    image_size: int = 448
    # One checkpoint is 50 prompts x 5 paintings x 3 criteria = 750 calls;
    # 2000 keeps every one of them in flight.
    max_concurrency: int = 2000
    base_url: str | None = None
    # The OpenAI-compatible chat API that serves model_name, with the key read
    # from api_key_env (comma-separated names: later keys are fallbacks once one
    # is refused). None calls model_name on Tinker's native sampler instead
    # (e.g. evaluator.model_name=moonshotai/Kimi-K2.6 evaluator.api_base_url=None).
    api_base_url: str | None = DEFAULT_EVALUATOR_API_BASE_URL
    api_key_env: str = DEFAULT_EVALUATOR_API_KEY_ENV
    # Passed through as the API's reasoning_effort; None keeps its default.
    reasoning_effort: str | None = "low"


@dataclass
class Verdict:
    """One scoring call, kept for logging/artifacts."""

    criterion: str
    rep: int
    slot_a: int
    """Group index of the painting shown in slot A (judge/evaluator: the only painting)."""
    slot_b: int | None
    score_a: float
    score_b: float | None
    text: str
    truncated: bool


@dataclass
class GroupScores:
    """Per-painting rewards in [0, 1] plus the verdict log for a group."""

    scores: list[float]
    verdicts: list[Verdict] = field(default_factory=list)
    n_calls: int = 0
    n_truncated: int = 0


def top_is_tied(scores: list[float]) -> bool:
    """True when more than one painting shares the group's highest score."""
    if len(scores) < 2:
        return False
    top = max(scores)
    return sum(abs(s - top) < 1e-9 for s in scores) > 1


def _fit(image: Image.Image, size: int) -> Image.Image:
    im = image.convert("RGB")
    im.thumbnail((size, size))
    return im


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _mean_per_painting(verdicts: list[Verdict], n: int, ok: list[bool]) -> list[float]:
    """Mean score per painting over its usable verdicts; 0.5 when it has none."""
    total = [0.0] * n
    count = [0] * n
    for v, usable in zip(verdicts, ok, strict=True):
        if not usable:
            continue
        total[v.slot_a] += v.score_a
        count[v.slot_a] += 1
        if v.slot_b is not None and v.score_b is not None:
            total[v.slot_b] += v.score_b
            count[v.slot_b] += 1
    return [total[i] / count[i] if count[i] else 0.5 for i in range(n)]


# ---------------------------------------------------------------------------
# Tinker's OpenAI-compatible endpoint, shared by the verifier and the judge
# ---------------------------------------------------------------------------


def effort_preset(effort: float) -> str:
    """The named preset for ``effort``, as Tinker's OpenAI endpoint expects it."""
    for value, name in EFFORT_PRESETS.items():
        if abs(effort - value) < 1e-9:
            return name
    raise ValueError(f"effort must be one of {sorted(EFFORT_PRESETS)}, got {effort}")


# Extra retries for transient endpoint errors (502s, timeouts), on top of the
# openai client's own quick ones: an outage of a minute or two then costs
# latency instead of turning a whole step's comparisons into ties.
RETRY_DELAYS_S: tuple[float, ...] = (10.0, 30.0, 60.0)

T = TypeVar("T")


def _is_transient(error: Exception) -> bool:
    import openai

    if isinstance(error, openai.APIConnectionError):  # includes timeouts
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


def _with_retries(call: Callable[[], T], what: str) -> T | None:
    """``call()``, retrying transient errors; None when it still fails."""
    for attempt, delay in enumerate((*RETRY_DELAYS_S, None)):
        try:
            return call()
        except Exception as e:
            if delay is not None and _is_transient(e):
                logger.info("%s call failed (%s), retry %d in %.0f s", what, e, attempt + 1, delay)
                time.sleep(delay)
                continue
            logger.warning("%s call failed (%s: %s)", what, type(e).__name__, e)
            return None
    raise AssertionError("unreachable")


def make_tinker_openai_client(base_url: str | None, max_concurrency: int) -> Any:
    """An ``openai`` client on Tinker's OpenAI-compatible endpoint."""
    from openai import DefaultHttpxClient, OpenAI

    key = os.environ.get("TINKER_API_KEY")
    if not key:
        raise llm_verifier.MissingAPIKeyError("set TINKER_API_KEY to use the Tinker reward model")
    import httpx

    return OpenAI(
        base_url=base_url or TINKER_OPENAI_BASE_URL,
        api_key=key,
        max_retries=5,
        # A call takes 5-60 s when the endpoint is idle but queues for minutes
        # under load. Don't time those out: a re-sent call joins the back of the
        # same queue, which grows it until whole steps hit the grading cap. This
        # only catches a dead connection, with one retry still inside the cap.
        timeout=1800.0,
        # The default pool (1000 connections) would queue part of a step.
        http_client=DefaultHttpxClient(
            limits=httpx.Limits(
                max_connections=max_concurrency, max_keepalive_connections=max_concurrency
            )
        ),
    )


class _TinkerOpenAIScorer:
    """A Tinker OpenAI client plus a thread pool sized to the wanted concurrency."""

    def __init__(self, config: RewardConfig):
        self.config = config
        self._client = make_tinker_openai_client(config.base_url, config.max_concurrency)
        # The openai client (and llm_verifier.compare) are synchronous; run
        # calls on a dedicated pool sized to the concurrency this loop wants.
        self._pool = ThreadPoolExecutor(
            max_workers=config.max_concurrency, thread_name_prefix=f"paint-{type(self).__name__}"
        )

    async def _run(self, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)


# ---------------------------------------------------------------------------
# LLM-as-a-Verifier: pairwise, fine-grained, round-robin tournament
# ---------------------------------------------------------------------------


def round_robin_pairs(n: int, rng: random.Random) -> list[tuple[int, int]]:
    """Every unordered pair of ``n`` candidates as a directed (slot A, slot B).

    Which member of a pair starts in slot A is randomized, and the pairs are
    shuffled, so a group's comparisons are not systematically ordered by index;
    odd repeats then re-run each pair with the slots swapped, which is what
    actually cancels the positional bias.
    """
    pairs = [(a, b) if rng.random() < 0.5 else (b, a) for a, b in combinations(range(n), 2)]
    rng.shuffle(pairs)
    return pairs


_SCALE_IN_USE: int | None = None
_LOGPROBS_IN_USE: bool | None = None


def use_letter_logprobs(enabled: bool) -> None:
    """Read verifier grades from token logprobs (llm_verifier's default) or not.

    With ``enabled=False``, llm_verifier's score reader is handed no token
    logprobs, so it falls back to the letter written in the reply. Like the
    scale, this is process-wide and set once.
    """
    global _LOGPROBS_IN_USE
    if _LOGPROBS_IN_USE is not None:
        if enabled != _LOGPROBS_IN_USE:
            raise ValueError("llm_verifier's logprob setting is already fixed in this process")
        return
    _LOGPROBS_IN_USE = enabled
    if enabled:
        return
    from llm_verifier import fine_grained_reward as fgr

    extract_score = fgr.extract_score

    def extract_score_from_text(text: str, tokens: Any, position_logprobs: Any, tag: str) -> float:
        del tokens, position_logprobs  # only the written letter counts
        return extract_score(text, None, None, tag)

    fgr.extract_score = extract_score_from_text


def use_letter_scale(granularity: int) -> None:
    """Point llm_verifier at an A..<granularity-th letter> grade scale.

    llm_verifier reads its scale (letters, their values, the scale text) and
    builds its prompt from module-level state on every call, so a process can
    switch it once, before scoring. One scale per process: a verifier with a
    different granularity in the same process is refused.
    """
    global _SCALE_IN_USE
    if _SCALE_IN_USE is not None:
        if granularity != _SCALE_IN_USE:
            raise ValueError(
                f"llm_verifier already uses a {_SCALE_IN_USE}-letter scale in this process"
            )
        return
    _SCALE_IN_USE = granularity
    if granularity == 20:
        return  # llm_verifier's own scale
    if not 2 <= granularity <= 26:
        raise ValueError(f"granularity must be 2-26 letters, got {granularity}")
    from llm_verifier import fine_grained_reward as fgr

    last = chr(ord("A") + granularity - 1)
    fgr.GRANULARITY = granularity
    fgr.SCALE = {
        "scale_description": (
            f"Rate each painting on a {granularity}-point scale using letters A "
            f"through {last}: A = outstanding (best), {last} = a complete "
            "failure (worst), letters in between evenly spaced."
        ),
        "score_format": f"LETTER_A_TO_{last}",
        "valid_tokens": {
            **{chr(65 + i): float(granularity - i) for i in range(granularity)},
            **{chr(97 + i): float(granularity - i) for i in range(granularity)},
        },
    }
    build_prompt = fgr.build_prompt

    def build_prompt_on_scale(*args: Any, **kwargs: Any) -> str:
        return build_prompt(*args, **kwargs).replace("letter A-T", f"letter A-{last}")

    fgr.build_prompt = build_prompt_on_scale


class PairwiseVerifier(_TinkerOpenAIScorer):
    """LLM-as-a-Verifier reward for a group of paintings (round-robin tournament)."""

    config: VerifierConfig

    def __init__(self, config: VerifierConfig):
        super().__init__(config)
        # llm_verifier's hosted-API path reads the letter distribution at the
        # score tags the model samples itself (Tinker has no assistant
        # prefill), and takes its reasoning effort and budget from the env.
        os.environ["DEEPSEEK_EFFORT"] = effort_preset(config.effort)
        os.environ["DEEPSEEK_MAX_TOKENS"] = str(config.max_tokens)
        self._client._llm_verifier_model = config.model_name
        self._client._llm_verifier_deepseek = True
        use_letter_scale(config.granularity)
        use_letter_logprobs(config.use_logprobs)

    def _compare_sync(
        self, problem: str, image_a: bytes, image_b: bytes, criterion: tuple[str, str]
    ) -> tuple[float, float, bool]:
        name, description = criterion
        result = _with_retries(
            lambda: llm_verifier.compare(
                problem,
                SLOT_A,
                SLOT_B,
                criteria={name: description},
                images=[image_a, image_b],
                ground_truth_note=pairwise_note(self.config.granularity),
                model=self.config.model_name,
                client=self._client,
            ),
            "verifier",
        )
        if result is None:
            # A call that still fails is a tie, as llm_verifier's on_error="tie".
            return 0.5, 0.5, True
        ra, rb = result
        return ra, rb, False

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        """Round-robin rewards for ``images`` (all of them must be valid paintings).

        Every unordered pair of paintings is compared, for every criterion and
        repeat; odd repeats swap the two slots, so with an even
        ``n_evaluations`` each pair is judged equally often in each order and
        the verifier's positional bias cancels pair by pair. A painting's
        reward is the mean of its fine-grained scores over the ``n - 1``
        opponents it met.
        """
        n = len(images)
        if n == 0:
            return GroupScores(scores=[])
        if n == 1:
            return GroupScores(scores=[0.5])
        criteria = resolve_criteria(self.config.criteria)
        pngs = [_png_bytes(_fit(im, self.config.image_size)) for im in images]
        jobs = [
            (a, b, crit, rep, rep % 2 == 1)
            for a, b in round_robin_pairs(n, random.Random(seed))
            for crit in criteria
            for rep in range(self.config.n_evaluations)
        ]

        async def run(job: tuple[int, int, tuple[str, str], int, bool]) -> Verdict:
            a, b, crit, rep, swap = job
            first, second = (b, a) if swap else (a, b)
            r_first, r_second, failed = await self._run(
                self._compare_sync, problem, pngs[first], pngs[second], crit
            )
            return Verdict(
                criterion=crit[0],
                rep=rep,
                slot_a=first,
                slot_b=second,
                score_a=r_first,
                score_b=r_second,
                text="",
                truncated=failed,
            )

        verdicts = list(await asyncio.gather(*(run(job) for job in jobs)))
        # Failed calls count as the 0.5 tie they were scored as.
        return GroupScores(
            scores=_mean_per_painting(verdicts, n, [True] * len(verdicts)),
            verdicts=verdicts,
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


# ---------------------------------------------------------------------------
# LLM-as-a-Judge: one painting at a time, each criterion graded A-J
# ---------------------------------------------------------------------------

_ABSOLUTE_PREAMBLE = (
    "You are an expert reviewer of watercolor paintings. You will be given a "
    "painting request and one painting made for it, and you score it on ONE "
    "criterion.\n\n"
    f"{GROUND_TRUTH_NOTE}\n"
)


def absolute_prompt(
    problem: str,
    criterion: tuple[str, str],
    letter_grades: int = LETTER_GRADES,
) -> str:
    """The single-painting prompt, shared by the judge and the held-out evaluator:
    one letter from A (outstanding) to the ``letter_grades``-th (complete failure)."""
    name, description = criterion
    last = chr(ord("A") + letter_grades - 1)
    ask = (
        f'Grade the painting on this criterion ("{name}") with one letter from A '
        f"(outstanding) to {last} (complete failure), the letters in between "
        "evenly spaced.\n\n"
        "Reason it through first, then END your reply with exactly this line and "
        f"nothing after it, replacing the placeholder with a single letter A-{last}:\n"
        f"<score> LETTER_A_TO_{last} </score>\n"
    )
    return (
        _ABSOLUTE_PREAMBLE
        + f"\n**Evaluation Guideline — {name}:**\n{description}\n\n"
        + ask
        + f"\n**Task (the painting request):**\n{problem}\n"
        "\nThe painting is attached below.\n\nBegin your analysis now."
    )


# A single letter right after <score>, optionally bolded; the letter must
# stand alone (so "<score> LETTER_A_TO_J" placeholders are not read as "L").
_LETTER_TAG = re.compile(r"<score>\s*\**\s*([A-Za-z])(?![A-Za-z_])")


def extract_letter_score(text: str, letter_grades: int = LETTER_GRADES) -> float | None:
    """Last ``<score> X </score>`` letter in the reply in [0, 1]: A is 1, the
    ``letter_grades``-th letter is 0; None when missing or off the scale."""
    matches = _LETTER_TAG.findall(text or "")
    if not matches:
        return None
    index = ord(matches[-1].upper()) - ord("A")
    if not 0 <= index < letter_grades:
        return None
    return 1.0 - index / (letter_grades - 1)


class AbsoluteJudge(_TinkerOpenAIScorer):
    """LLM-as-a-Judge reward: each painting scored independently, one letter per criterion."""

    config: JudgeConfig

    def _ask_sync(self, prompt: str, png: bytes) -> tuple[str, bool]:
        import base64

        b64 = base64.b64encode(png).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
        response = _with_retries(
            lambda: self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.config.max_tokens,
                temperature=1.0,
                # The same thinking settings llm_verifier sends for the verifier.
                extra_body={
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": effort_preset(self.config.effort),
                },
            ),
            "judge",
        )
        if response is None:
            return "", True
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason == "length"

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        """Absolute rewards for ``images``: the mean of each painting's
        normalized grade per criterion. Equal grades give equal rewards."""
        del seed  # independent scores; nothing to randomize
        n = len(images)
        if n == 0:
            return GroupScores(scores=[])
        criteria = resolve_criteria(self.config.criteria)
        g = self.config.letter_grades
        prompts = {name: absolute_prompt(problem, (name, desc), g) for name, desc in criteria}
        pngs = [_png_bytes(_fit(im, self.config.image_size)) for im in images]
        jobs = [
            (i, name, rep)
            for i in range(n)
            for name, _ in criteria
            for rep in range(self.config.n_evaluations)
        ]

        async def run(job: tuple[int, str, int]) -> tuple[Verdict, bool]:
            i, name, rep = job
            text, truncated = await self._run(self._ask_sync, prompts[name], pngs[i])
            score = extract_letter_score(text, g)
            verdict = Verdict(
                criterion=name,
                rep=rep,
                slot_a=i,
                slot_b=None,
                score_a=0.5 if score is None else score,
                score_b=None,
                text=text,
                truncated=truncated or score is None,
            )
            return verdict, score is not None

        pairs = await asyncio.gather(*(run(job) for job in jobs))
        verdicts = [v for v, _ in pairs]
        # A failed or unparseable verdict is left out of the mean; a painting
        # with none at all falls back to 0.5 and is counted in failed_frac.
        return GroupScores(
            scores=_mean_per_painting(verdicts, n, [ok for _, ok in pairs]),
            verdicts=verdicts,
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


# ---------------------------------------------------------------------------
# Held-out evaluator: one call per painting and criterion, graded A-J, on an
# OpenAI-compatible API (default) or Tinker's native sampler
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    text: str
    truncated: bool


class _TinkerVisionClient:
    """A vision model on Tinker's native sampler, prompted through its cookbook renderer."""

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._sampler: tinker.SamplingClient | None = None
        self._renderer: Renderer | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> tuple[tinker.SamplingClient, Renderer]:
        async with self._lock:
            if self._sampler is not None and self._renderer is not None:
                return self._sampler, self._renderer
            import tinker

            from tinker_cookbook import model_info
            from tinker_cookbook.image_processing_utils import get_image_processor
            from tinker_cookbook.renderers import get_renderer
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            name = self.config.model_name
            renderer_name = self.config.renderer_name or model_info.get_recommended_renderer_name(
                name
            )
            self._renderer = get_renderer(
                renderer_name, get_tokenizer(name), image_processor=get_image_processor(name)
            )
            service = tinker.ServiceClient(base_url=self.config.base_url)
            self._sampler = await service.create_sampling_client_async(base_model=name)
            return self._sampler, self._renderer

    async def ask(self, prompt: str, image: Image.Image, num_samples: int) -> list[Reply]:
        """``num_samples`` replies to one prompt, from a single sampling request."""
        sampler, renderer = await self._ensure_started()
        import tinker

        from tinker_cookbook.renderers import Message, get_text_content

        message = Message(
            role="user",
            content=[{"type": "text", "text": prompt}, {"type": "image", "image": image}],
        )
        model_input = renderer.build_generation_prompt([message])
        params = tinker.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            stop=renderer.get_stop_sequences(),
        )
        try:
            async with self._semaphore:
                result = await sampler.sample_async(
                    model_input, num_samples=num_samples, sampling_params=params
                )
        except Exception as e:  # still failing after the SDK's own retries
            logger.warning("held-out evaluator call failed (%s: %s)", type(e).__name__, e)
            return [Reply(text="", truncated=True)] * num_samples
        replies = []
        for sequence in result.sequences:
            reply, _ = renderer.parse_response(sequence.tokens)
            text = get_text_content(reply)
            replies.append(Reply(text=text, truncated=sequence.stop_reason == "length" or not text))
        return replies


class _OpenAIVisionClient:
    """A vision model behind an OpenAI-compatible chat completions API.

    Plain httpx rather than the openai SDK, whose per-request overhead caps a
    single process at a few calls per second.
    """

    def __init__(self, config: EvaluatorConfig):
        import httpx

        self.config = config
        names = [name for name in config.api_key_env.split(",") if name]
        self._keys = [os.environ[name] for name in names if os.environ.get(name)]
        if not self._keys:
            raise llm_verifier.MissingAPIKeyError(
                f"set {' or '.join(names)} to use the held-out evaluator "
                f"({config.model_name} at {config.api_base_url})"
            )
        self._key = 0
        self._client = httpx.AsyncClient(
            base_url=(config.api_base_url or "").rstrip("/"),
            timeout=900.0,
            limits=httpx.Limits(max_connections=config.max_concurrency),
        )
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _one(self, prompt: str, image_url: str) -> Reply:
        import httpx

        body: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.reasoning_effort is not None:
            body["reasoning_effort"] = self.config.reasoning_effort
        error = ""
        async with self._semaphore:
            for attempt in range(8):
                key = self._key
                try:
                    r = await self._client.post(
                        "/chat/completions",
                        json=body,
                        headers={"Authorization": f"Bearer {self._keys[key]}"},
                    )
                except httpx.HTTPError as e:
                    error = f"{type(e).__name__}: {e}"
                else:
                    choices = (r.json() or {}).get("choices") if r.status_code == 200 else None
                    if choices:
                        choice = choices[0]
                        text = (choice.get("message") or {}).get("content") or ""
                        return Reply(
                            text=text, truncated=choice.get("finish_reason") == "length" or not text
                        )
                    # A 200 without choices is retried like a server error.
                    error = f"HTTP {r.status_code}: {r.text[:200]}"
                    # Out of balance or a bad key: move on to the next key, if any.
                    if r.status_code in (401, 402) and key == self._key:
                        if key + 1 >= len(self._keys):
                            break
                        self._key = key + 1
                        logger.warning("evaluator key %d failed (%s); switching keys", key, error)
                        continue
                    if r.status_code < 500 and r.status_code not in (200, 429):
                        break
                await asyncio.sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))
        logger.warning("held-out evaluator call failed (%s)", error)
        return Reply(text="", truncated=True)

    async def ask(self, prompt: str, image: Image.Image, num_samples: int) -> list[Reply]:
        """``num_samples`` replies to one prompt, one request each."""
        import base64

        image_url = "data:image/png;base64," + base64.b64encode(_png_bytes(image)).decode()
        return list(
            await asyncio.gather(*(self._one(prompt, image_url) for _ in range(num_samples)))
        )


class HeldoutEvaluator:
    """Independent absolute scores per painting and criterion, no pairing.

    Each criterion is its own request, so one criterion's reasoning never
    colors another's. All of a group's requests are submitted at once (and
    every group's, since groups are evaluated concurrently), up to
    ``max_concurrency`` in flight; ``n_evaluations`` repeats come from
    ``num_samples`` on the same request.
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._client: _TinkerVisionClient | _OpenAIVisionClient = (
            _OpenAIVisionClient(config) if config.api_base_url else _TinkerVisionClient(config)
        )

    async def score_group(
        self, problem: str, images: list[Image.Image], seed: int = 0
    ) -> GroupScores:
        del seed  # independent scores; nothing to randomize
        n = len(images)
        if n == 0:
            return GroupScores(scores=[])
        criteria = resolve_criteria(self.config.criteria)
        fitted = [_fit(im, self.config.image_size) for im in images]
        jobs = [(i, crit) for i in range(n) for crit in criteria]
        per_job = await asyncio.gather(
            *(
                self._client.ask(
                    absolute_prompt(problem, crit, self.config.letter_grades),
                    fitted[i],
                    self.config.n_evaluations,
                )
                for i, crit in jobs
            )
        )
        verdicts: list[Verdict] = []
        usable: list[bool] = []
        for (i, (name, _)), replies in zip(jobs, per_job, strict=True):
            for rep, reply in enumerate(replies):
                score = extract_letter_score(reply.text, self.config.letter_grades)
                verdicts.append(
                    Verdict(
                        criterion=name,
                        rep=rep,
                        slot_a=i,
                        slot_b=None,
                        score_a=0.5 if score is None else score,
                        score_b=None,
                        text=reply.text,
                        truncated=reply.truncated or score is None,
                    )
                )
                usable.append(score is not None)
        # Failed or unparseable verdicts are left out of the mean; a painting
        # with none at all falls back to a 0.5 tie.
        return GroupScores(
            scores=_mean_per_painting(verdicts, n, usable),
            verdicts=verdicts,
            n_calls=len(verdicts),
            n_truncated=sum(v.truncated for v in verdicts),
        )


Scorer = PairwiseVerifier | AbsoluteJudge
_SCORERS: dict[RewardConfig, Scorer] = {}
_EVALUATORS: dict[EvaluatorConfig, HeldoutEvaluator] = {}


def get_shared_scorer(config: RewardConfig) -> Scorer:
    """One reward scorer per config per process, so the client and its pool are reused."""
    if config not in _SCORERS:
        _SCORERS[config] = (
            PairwiseVerifier(config)
            if isinstance(config, VerifierConfig)
            else AbsoluteJudge(config)
        )
    return _SCORERS[config]


def get_shared_evaluator(config: EvaluatorConfig) -> HeldoutEvaluator:
    """One evaluator per config per process."""
    if config not in _EVALUATORS:
        _EVALUATORS[config] = HeldoutEvaluator(config)
    return _EVALUATORS[config]
