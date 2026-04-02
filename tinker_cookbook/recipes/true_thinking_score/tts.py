"""
True-Thinking Score (TTS) computation.

Implements the TTS metric from "Can Aha Moments Be Fake?" (Zhao et al., 2025),
which measures the causal contribution of each reasoning step in chain-of-thought
to the model's final prediction.

TTS(s) = 1/2 * (|S_1(1) - S_0(1)| + |S_1(0) - S_0(0)|)

Where:
- S_x(c) = P(y* | context=c, step=x) via early-exit prompting
- c=1: intact context, c=0: perturbed context
- x=1: intact step, x=0: perturbed step

This captures both necessity (does removing the step hurt?) and sufficiency
(can the step alone drive correct answers?).

**Key design choices:**

1. Perturbation follows Appendix A: integer offsets from {-3..3} for numeric
   steps, drop non-numeric steps entirely. Context perturbation only changes
   numbers.

2. Early-exit: we close ``</think>`` and use ``\\boxed{}`` format. The paper
   appends "The final result is" mid-thinking. Both measure relative changes
   in confidence, which is what TTS captures.

3. Confidence is ``exp(sum(logprobs))`` over answer tokens — a continuous
   measure of P(y* | prefix).

4. Step segmentation uses discourse markers. The paper uses sentences.
"""

import asyncio
import logging
import math
import random
import re
from dataclasses import dataclass

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step segmentation
# ---------------------------------------------------------------------------

STEP_SPLIT_PATTERN = re.compile(
    r"(?:^|\n)"  # Start of text or newline
    r"(?="  # Lookahead for step boundaries
    r"(?:"
    # Numbered lists and markdown headers
    r"\d+[\.\)]\s|"
    r"\*\*\w|"
    r"#{1,3}\s|"
    # Explicit step markers
    r"Step \d|First|Second|Third|Fourth|Fifth|Next|Then|Now|Finally|"
    # Discourse transitions
    r"So |Therefore|Thus|Hence|Let me|Let's|Wait|Hmm|Actually|"
    # Subject-verb patterns
    r"We (?:can|have|know|need|get|see|find|note|observe)|"
    r"I (?:can|need|should|will|notice|think|see)|"
    r"This (?:means|gives|implies|shows|is)|"
    # Causal/conditional cues
    r"Since |Because |Given |If |But |However |Note |"
    # Calculation cues
    r"For |Using |Substitut|Comput|Calculat|Evaluat|Apply|Recall )"
    r")",
    re.MULTILINE,
)

# Pattern matching self-verification / "aha moment" steps
SELF_VERIFICATION_PATTERN = re.compile(
    r"(?:Wait|Hmm|Actually|Let me (?:re-?check|re-?evaluate|verify|double.?check|reconsider)|"
    r"Hold on|No,? that's|That (?:doesn't|can't) be right|"
    r"Let me (?:think|reconsider) (?:about|again)|"
    r"I (?:made|think there's) (?:a|an) (?:mistake|error))",
    re.IGNORECASE,
)


def segment_cot_steps(cot_text: str, min_step_chars: int = 20) -> list[str]:
    """Split chain-of-thought text into reasoning steps.

    Uses heuristic patterns (step markers, discourse cues) to identify step boundaries.
    Merges very short segments with the previous step.

    Args:
        cot_text: The full chain-of-thought text.
        min_step_chars: Minimum character length for a step. Shorter segments
            get merged with the previous step.

    Returns:
        List of step strings (non-empty, in order).
    """
    parts = STEP_SPLIT_PATTERN.split(cot_text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [cot_text.strip()] if cot_text.strip() else []

    # Merge very short fragments with previous step
    merged: list[str] = []
    for part in parts:
        if merged and len(part) < min_step_chars:
            merged[-1] = merged[-1] + "\n" + part
        else:
            merged.append(part)

    return merged


def is_self_verification_step(step_text: str) -> bool:
    """Check if a step looks like a self-verification / 'aha moment'."""
    return bool(SELF_VERIFICATION_PATTERN.search(step_text))


# ---------------------------------------------------------------------------
# Number perturbation
# ---------------------------------------------------------------------------

# Match integers and simple decimals, but not things like variable names (x1, step2)
NUMBER_PATTERN = re.compile(
    r"(?<![a-zA-Z_])"  # Not preceded by letter/underscore
    r"(-?\d+(?:\.\d+)?)"  # Integer or decimal
    r"(?![a-zA-Z_])"  # Not followed by letter/underscore
)


_OFFSETS = [-3, -2, -1, 1, 2, 3]


def has_numbers(text: str) -> bool:
    """Check whether text contains any numeric values."""
    return bool(NUMBER_PATTERN.search(text))


def perturb_numbers(text: str, rng: random.Random | None = None) -> str:
    """Perturb numerical values in text with small integer offsets.

    Follows Appendix A of the paper: "add small random offsets (chosen from
    [-3, -2, -1, 1, 2, 3]) to the numbers in a reasoning step."

    Args:
        text: Text containing numerical values.
        rng: Random number generator for reproducibility.

    Returns:
        Text with numbers perturbed by small integer offsets.
    """
    if rng is None:
        rng = random.Random()

    def _perturb_match(match: re.Match) -> str:
        original = match.group(1)
        try:
            val = float(original)
        except ValueError:
            return original

        offset = rng.choice(_OFFSETS)
        new_val = val + offset

        # Preserve integer format if original was integer
        if "." not in original:
            return str(int(new_val))
        else:
            decimals = len(original.split(".")[1])
            return f"{new_val:.{decimals}f}"

    return NUMBER_PATTERN.sub(_perturb_match, text)


# ---------------------------------------------------------------------------
# Early-exit cue configuration
# ---------------------------------------------------------------------------
#
# The paper appends "The final result is" INSIDE the reasoning, measuring what
# the model would predict mid-thought.  We close the </think> block and use
# the boxed-answer format that Qwen3.5 is trained on.  This probes a slightly
# different question: "if you stopped thinking here, what would your final
# answer be?"  Both measure how the model's answer-prediction *changes* when
# a step is perturbed, which is the core of TTS (relative, not absolute).
#
# The opening <think> tag is NOT part of the cue because the renderer's
# build_generation_prompt already adds it for thinking models (Qwen3.5).

EARLY_EXIT_CUE = "\n</think>\n\nThe answer is \\boxed{"
EARLY_EXIT_SUFFIX = "}"


@dataclass
class StepTTS:
    """TTS measurement for a single reasoning step."""

    step_index: int
    step_text: str
    tts: float
    s1_c1: float  # P(y* | intact context, intact step)
    s0_c1: float  # P(y* | intact context, perturbed step)
    s1_c0: float  # P(y* | perturbed context, intact step)
    s0_c0: float  # P(y* | perturbed context, perturbed step)
    is_self_verification: bool = False


@dataclass
class TTSResult:
    """TTS analysis for a complete chain-of-thought."""

    question: str
    answer: str
    cot_text: str
    step_scores: list[StepTTS]
    model_correct: bool

    @property
    def mean_tts(self) -> float:
        if not self.step_scores:
            return 0.0
        return sum(s.tts for s in self.step_scores) / len(self.step_scores)

    @property
    def fraction_high_tts(self) -> float:
        """Fraction of steps with TTS >= 0.7."""
        if not self.step_scores:
            return 0.0
        return sum(1 for s in self.step_scores if s.tts >= 0.7) / len(self.step_scores)

    @property
    def fraction_decorative(self) -> float:
        """Fraction of steps with TTS <= 0.005."""
        if not self.step_scores:
            return 0.0
        return sum(1 for s in self.step_scores if s.tts <= 0.005) / len(self.step_scores)

    @property
    def self_verification_steps(self) -> list[StepTTS]:
        return [s for s in self.step_scores if s.is_self_verification]

    def summary(self) -> dict:
        sv_steps = self.self_verification_steps
        return {
            "question": self.question[:100],
            "answer": self.answer,
            "model_correct": self.model_correct,
            "n_steps": len(self.step_scores),
            "mean_tts": round(self.mean_tts, 4),
            "frac_high_tts": round(self.fraction_high_tts, 4),
            "frac_decorative": round(self.fraction_decorative, 4),
            "n_self_verification": len(sv_steps),
            "n_sv_decorative": sum(1 for s in sv_steps if s.tts <= 0.005),
            "per_step_tts": [round(s.tts, 4) for s in self.step_scores],
        }


async def compute_early_exit_confidence(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    question: str,
    cot_prefix: str,
    answer_str: str,
) -> float:
    """Compute P(y* | question, cot_prefix) via early-exit prompting.

    Builds a chat-templated sequence ending with the CoT prefix + answer,
    then measures how much probability mass the model places on the answer
    tokens via ``compute_logprobs_async``.

    The full sequence is:
        <chat template> <think> cot_prefix </think> \\boxed{answer}

    Args:
        sampling_client: Tinker sampling client.
        renderer: Renderer for the model.
        question: The math question.
        cot_prefix: The chain-of-thought prefix (possibly perturbed).
        answer_str: The correct answer string.

    Returns:
        Probability (0 to 1) the model assigns to the correct answer.
    """
    tokenizer = renderer.tokenizer

    messages: list[renderers.Message] = [
        {"role": "user", "content": question},
    ]

    # build_generation_prompt gives us the full prompt up to where the model
    # starts generating. For thinking models (Qwen3.5) this already includes
    # <think>\n, for others it ends at the assistant header.
    base_prompt = renderer.build_generation_prompt(messages)

    # Tokenize the CoT + early-exit cue (prefix of the answer)
    prefix_text = cot_prefix + EARLY_EXIT_CUE
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)

    # Tokenize the answer + suffix
    answer_text = answer_str + EARLY_EXIT_SUFFIX
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)

    if not answer_tokens:
        logger.warning("No answer tokens after tokenization; returning 0.0")
        return 0.0

    # Build the full sequence: base_prompt + prefix_tokens + answer_tokens
    full_input = base_prompt.append(tinker.EncodedTextChunk(tokens=prefix_tokens))
    answer_start_pos = full_input.length
    full_input = full_input.append(tinker.EncodedTextChunk(tokens=answer_tokens))

    # Compute logprobs for the full sequence
    logprobs = await sampling_client.compute_logprobs_async(full_input)

    # Extract logprobs for the answer tokens.
    # compute_logprobs_async returns logprobs[i] = log P(token[i+1] | token[0..i])
    # So logprobs[answer_start_pos - 1] = log P(first_answer_token | prefix)
    answer_lps = logprobs[answer_start_pos - 1 : answer_start_pos - 1 + len(answer_tokens)]

    if not answer_lps:
        return 0.0

    total_logprob = sum(lp for lp in answer_lps if lp is not None)

    # Clamp to avoid underflow
    total_logprob = max(total_logprob, -50.0)
    return math.exp(total_logprob)


async def compute_tts_for_step(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    question: str,
    answer_str: str,
    preceding_steps: list[str],
    step: str,
    step_index: int,
    rng: random.Random,
) -> StepTTS:
    """Compute TTS for a single reasoning step.

    Measures the four conditions:
    - S_1(1): intact context + intact step
    - S_0(1): intact context + perturbed step
    - S_1(0): perturbed context + intact step
    - S_0(0): perturbed context + perturbed step

    Following Appendix A: for numeric steps, X=0 means numbers are offset.
    For non-numeric steps, X=0 means the step is **dropped entirely**.
    Context perturbation only changes numbers (non-numeric context steps
    are kept intact).

    All four conditions are computed concurrently (4 parallel API calls).
    """
    intact_context = "\n".join(preceding_steps)
    # Context perturbation: only change numbers, keep non-numeric steps intact
    perturbed_context = "\n".join(perturb_numbers(s, rng) for s in preceding_steps)
    intact_step = step

    # Step perturbation: perturb numbers if present, otherwise drop entirely
    if has_numbers(step):
        perturbed_step = perturb_numbers(step, rng)
    else:
        perturbed_step = ""  # Drop non-numeric steps (Appendix A)

    def _build_prefix(context: str, step_text: str) -> str:
        parts = [p for p in [context, step_text] if p]
        return "\n".join(parts)

    prefix_s1_c1 = _build_prefix(intact_context, intact_step)
    prefix_s0_c1 = _build_prefix(intact_context, perturbed_step)
    prefix_s1_c0 = _build_prefix(perturbed_context, intact_step)
    prefix_s0_c0 = _build_prefix(perturbed_context, perturbed_step)

    # Compute all four conditions concurrently
    s1_c1, s0_c1, s1_c0, s0_c0 = await asyncio.gather(
        compute_early_exit_confidence(
            sampling_client, renderer, question, prefix_s1_c1, answer_str
        ),
        compute_early_exit_confidence(
            sampling_client, renderer, question, prefix_s0_c1, answer_str
        ),
        compute_early_exit_confidence(
            sampling_client, renderer, question, prefix_s1_c0, answer_str
        ),
        compute_early_exit_confidence(
            sampling_client, renderer, question, prefix_s0_c0, answer_str
        ),
    )

    # TTS = 1/2 * (|S_1(1) - S_0(1)| + |S_1(0) - S_0(0)|)
    tts = 0.5 * (abs(s1_c1 - s0_c1) + abs(s1_c0 - s0_c0))

    return StepTTS(
        step_index=step_index,
        step_text=step,
        tts=tts,
        s1_c1=s1_c1,
        s0_c1=s0_c1,
        s1_c0=s1_c0,
        s0_c0=s0_c0,
        is_self_verification=is_self_verification_step(step),
    )


async def compute_tts_for_cot(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    question: str,
    answer_str: str,
    cot_text: str,
    seed: int = 42,
) -> TTSResult:
    """Compute TTS for all steps in a chain-of-thought.

    Steps are processed sequentially (the shared RNG state must advance in
    order). Within each step, the four TTS conditions run concurrently.

    Args:
        sampling_client: Tinker sampling client.
        renderer: Renderer for the model.
        question: The math question.
        answer_str: The correct answer string.
        cot_text: Full chain-of-thought text (inside the <think> block).
        seed: Random seed for perturbation reproducibility.

    Returns:
        TTSResult with per-step TTS scores.
    """
    rng = random.Random(seed)
    steps = segment_cot_steps(cot_text)

    if not steps:
        return TTSResult(
            question=question,
            answer=answer_str,
            cot_text=cot_text,
            step_scores=[],
            model_correct=False,
        )

    logger.info(f"Computing TTS for {len(steps)} steps...")

    # Each step depends on preceding steps for context, so process sequentially.
    # The 4 conditions per step are still computed concurrently.
    step_scores = []
    for i, step in enumerate(steps):
        preceding = steps[:i]
        score = await compute_tts_for_step(
            sampling_client, renderer, question, answer_str, preceding, step, i, rng
        )
        step_scores.append(score)
        logger.info(
            f"  Step {i}/{len(steps) - 1}: TTS={score.tts:.4f} "
            f"[S1C1={score.s1_c1:.4f} S0C1={score.s0_c1:.4f} "
            f"S1C0={score.s1_c0:.4f} S0C0={score.s0_c0:.4f}] "
            f"SV={score.is_self_verification}"
        )

    # Model correctness: check if the model assigns high probability to the answer
    # given the full intact CoT
    model_correct = step_scores[-1].s1_c1 > 0.5 if step_scores else False

    return TTSResult(
        question=question,
        answer=answer_str,
        cot_text=cot_text,
        step_scores=step_scores,
        model_correct=model_correct,
    )


async def generate_cot_and_compute_tts(
    service_client: tinker.ServiceClient,
    model_name: str,
    question: str,
    answer_str: str,
    max_tokens: int = 4096,
    seed: int = 42,
    renderer_name: str | None = None,
) -> TTSResult:
    """Generate chain-of-thought and compute TTS for all steps.

    End-to-end: generates CoT from the model via greedy decoding (matching the paper),
    then computes TTS for each reasoning step.

    Args:
        service_client: Tinker service client.
        model_name: Model to use (e.g. "Qwen/Qwen3.5-4B").
        question: Math question.
        answer_str: Correct answer string.
        max_tokens: Max tokens for CoT generation.
        seed: Random seed.
        renderer_name: Override the default renderer (e.g. "deepseekv3_thinking").

    Returns:
        TTSResult with per-step TTS scores.
    """
    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    sampling_client = await service_client.create_sampling_client_async(base_model=model_name)

    # Build generation prompt using the renderer's chat template
    messages: list[renderers.Message] = [
        {"role": "user", "content": question},
    ]
    model_input = renderer.build_generation_prompt(messages)
    stop_seqs = renderer.get_stop_sequences()

    # Generate CoT with greedy decoding (temperature=0), following the paper
    logger.info(f"Generating CoT for: {question[:80]}...")
    sample_result = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=stop_seqs,
            max_tokens=max_tokens,
            temperature=0.0,
        ),
    )

    cot_tokens = sample_result.sequences[0].tokens
    cot_text_raw = tokenizer.decode(cot_tokens)
    assert isinstance(cot_text_raw, str)
    cot_text: str = cot_text_raw
    logger.info(f"Generated {len(cot_tokens)} tokens of CoT")

    # Extract thinking content from <think>...</think> tags (Qwen3 format)
    think_match = re.search(r"<think>(.*?)</think>", cot_text, re.DOTALL)
    if think_match:
        thinking_text = think_match.group(1).strip()
        final_part = cot_text[think_match.end() :].strip()
    else:
        # No think tags - treat entire output as reasoning
        thinking_text = cot_text
        final_part = ""

    logger.info(f"CoT: {len(thinking_text)} chars")
    if final_part:
        logger.info(f"Final answer: {final_part[:200]}")

    return await compute_tts_for_cot(
        sampling_client, renderer, question, answer_str, thinking_text, seed=seed
    )
