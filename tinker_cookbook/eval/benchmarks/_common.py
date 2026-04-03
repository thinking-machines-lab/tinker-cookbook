"""Shared utilities for benchmark environments."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import TYPE_CHECKING

from datasets import Dataset

from tinker_cookbook.renderers.base import Message

if TYPE_CHECKING:
    from tinker_cookbook.sandbox.sandbox_interface import SandboxInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message building — respects config.system_prompt
# ---------------------------------------------------------------------------


def build_messages(
    user_content: str,
    system_prompt: str | None = None,
) -> list[Message]:
    """Build a message list, optionally prepending a system prompt.

    Args:
        user_content: The user message content.
        system_prompt: If set, prepended as a system message.

    Returns:
        List of message dicts ready for ``renderer.build_generation_prompt``.
    """
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Response decoding — strips thinking tokens before grading
# ---------------------------------------------------------------------------


def decode_response(action, renderer) -> str:
    """Decode model action tokens into grading text, stripping thinking traces.

    Uses ``renderer.parse_response()`` followed by ``get_text_content()`` to
    extract only the non-thinking text. This correctly handles model-specific
    thinking formats (``<think>`` for Qwen, ``<|begin_of_thought|>`` for
    DeepSeek, etc.) without benchmark-specific logic.

    Note:
        For benchmarks that grade tool calls (BFCL, tau2_bench), use
        ``renderer.tokenizer.decode(action)`` directly instead — this
        function strips tool call content.

    Args:
        action: Raw token sequence (list of ints) from the model.
        renderer: The :class:`~tinker_cookbook.renderers.base.Renderer` used
            for this model. Must match the model family.

    Returns:
        The non-thinking text content of the response, suitable for
        answer extraction and grading.

    Example::

        response = decode_response(action, renderer)
        correct = check_gsm8k(response, expected_answer)
    """
    from tinker_cookbook.renderers import get_text_content

    message, _ = renderer.parse_response(action)
    return get_text_content(message)


# ---------------------------------------------------------------------------
# Common helpers used across many benchmarks
# ---------------------------------------------------------------------------


def make_example_id(prefix: str, seed_text: str) -> str:
    """Create a stable, deterministic example ID from a prefix and seed text.

    Args:
        prefix: Benchmark name prefix (e.g. ``"gsm8k"``, ``"mmlu_pro"``).
        seed_text: Text to hash (e.g. the question text).

    Returns:
        A string like ``"gsm8k_a1b2c3d4e5f6"``.
    """
    return f"{prefix}_{hashlib.md5(seed_text.encode()).hexdigest()[:12]}"


def format_mcq_choices(choices: list[str]) -> str:
    """Format a list of choices as lettered options.

    Args:
        choices: List of choice strings.

    Returns:
        Formatted string like ``"(A) choice1\\n(B) choice2\\n..."``.
    """
    return "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))


def limit_dataset(
    ds: Dataset, max_examples: int | None, shuffle_seed: int | None = None
) -> Dataset:
    """Optionally shuffle and limit a dataset to ``max_examples``.

    Args:
        ds: The dataset to limit.
        max_examples: Maximum number of examples, or ``None`` for all.
        shuffle_seed: If set, shuffle before limiting (for representative sampling).

    Returns:
        The (possibly shuffled and truncated) dataset.
    """
    if max_examples is None:
        return ds
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    return ds.select(range(min(max_examples, len(ds))))


# ---------------------------------------------------------------------------
# Dataset loading with consistent HF_TRUST_REMOTE_CODE + gated dataset handling
# ---------------------------------------------------------------------------


def load_benchmark_dataset(
    path: str,
    name: str | None = None,
    split: str = "test",
    trust_remote_code: bool | None = None,
) -> Dataset:
    """Load a HuggingFace dataset with consistent trust_remote_code and error handling.

    Respects the ``HF_TRUST_REMOTE_CODE`` environment variable (``1``, ``true``,
    ``yes``).  Provides clear error messages for gated datasets that require
    authentication.

    Args:
        path: HuggingFace dataset path (e.g. ``"openai/gsm8k"``).
        name: Dataset configuration name (e.g. ``"main"``).
        split: Dataset split (default ``"test"``).
        trust_remote_code: If ``None``, resolved from ``HF_TRUST_REMOTE_CODE``
            env var. If ``True``/``False``, used directly.

    Returns:
        The loaded Dataset.

    Raises:
        PermissionError: If the dataset is gated and requires authentication.
        RuntimeError: If the dataset cannot be loaded for other reasons.
    """
    from datasets import load_dataset

    resolved_trust = _resolve_trust_remote_code(trust_remote_code)

    kwargs: dict = {"split": split}
    if name is not None:
        kwargs["name"] = name
    if resolved_trust:
        kwargs["trust_remote_code"] = True

    try:
        ds = load_dataset(path, **kwargs)
    except Exception as e:
        err_str = str(e).lower()
        # Gated dataset — needs HF auth + terms acceptance
        if (
            "gated" in err_str
            or "401" in err_str
            or "unauthorized" in err_str
            or "access" in err_str
        ):
            raise PermissionError(
                f"Dataset '{path}' is gated and requires authentication.\n"
                f"1. Create a HuggingFace token at https://huggingface.co/settings/tokens\n"
                f"2. Accept the dataset terms at https://huggingface.co/datasets/{path}\n"
                f"3. Set the token: export HF_TOKEN=hf_...\n"
                f"   Or run: huggingface-cli login\n"
                f"Original error: {e}"
            ) from e
        # Trust remote code needed
        if "trust_remote_code" in err_str or "custom code" in err_str:
            raise RuntimeError(
                f"Dataset '{path}' requires trust_remote_code=True.\n"
                f"Set the environment variable: export HF_TRUST_REMOTE_CODE=1\n"
                f"Original error: {e}"
            ) from e
        raise
    return ds  # type: ignore[return-value]


def _resolve_trust_remote_code(trust_remote_code: bool | None) -> bool:
    """Resolve trust_remote_code from parameter or HF_TRUST_REMOTE_CODE env var."""
    if trust_remote_code is not None:
        return trust_remote_code
    env_val = os.environ.get("HF_TRUST_REMOTE_CODE", "").lower()
    return env_val in ("1", "true", "yes")


def extract_command(text: str) -> str | None:
    """Extract a bash command from a model response.

    Looks for a ``bash``/``sh``/``shell`` fenced code block first, then any
    fenced block whose content does not look like Python, JSON, or XML.

    Returns:
        The extracted command string, or ``None`` if no command is found.
    """
    match = re.search(r"```(?:bash|sh|shell)\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        cmd = match.group(1).strip()
        if cmd and not cmd.startswith(("{", "[", "<", "def ", "class ", "import ")):
            return cmd
    return None


def is_task_complete(text: str) -> bool:
    """Check if the model signals task completion."""
    lower = text.lower()
    return "task_complete" in lower or "task complete" in lower


# ---------------------------------------------------------------------------
# Sandbox factory (shared by terminal_bench, swe_bench, mbpp, livecodebench)
# ---------------------------------------------------------------------------


def get_sandbox_factory(config, *, packages: list[str] | None = None) -> object:
    """Get a sandbox factory from config, falling back to Modal.

    If ``config.sandbox_factory`` is set, returns it directly — the user
    provides their own :class:`~tinker_cookbook.sandbox.SandboxInterface`
    implementation.  Otherwise, falls back to Modal with a clear error
    if Modal is not installed.

    Args:
        config: A :class:`BenchmarkConfig` (imported at call time to avoid
            circular imports).
        packages: Optional apt packages to install in the sandbox image.
            Only applies when using the default Modal backend. For example,
            ``["git", "python3-pip"]`` for SWE-bench.

    Returns:
        An async callable that creates a new sandbox on each call.
    """
    if config.sandbox_factory is not None:
        return config.sandbox_factory

    try:
        import modal

        from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

        image = modal.Image.debian_slim()
        if packages:
            image = image.apt_install(*packages)

        async def _modal_factory():
            return await ModalSandbox.create(image=image, timeout=1800)

        return _modal_factory
    except ImportError:

        async def _missing_factory():
            raise RuntimeError(
                "Code execution benchmarks require a sandbox backend. "
                "Either:\n"
                "  1. Install Modal: pip install 'tinker-cookbook[modal]'\n"
                "  2. Provide a custom sandbox_factory in BenchmarkConfig"
            )

        return _missing_factory


# ---------------------------------------------------------------------------
# Sandbox env mixin (shared by mbpp, livecodebench, terminal_bench, swe_bench)
# ---------------------------------------------------------------------------


class SandboxMixin:
    """Mixin providing sandbox lifecycle management for benchmark envs.

    Provides default ``sandbox = None`` and ``_cleaned_up = False`` so
    subclasses don't need to initialize them manually (though they may
    override in ``__init__`` if needed).

    The runner discovers ``cleanup`` via ``hasattr(env, 'cleanup')``.
    """

    sandbox: SandboxInterface | None = None
    _cleaned_up: bool = False

    async def cleanup(self) -> None:
        """Clean up sandbox resources. Safe to call multiple times."""
        if self._cleaned_up or self.sandbox is None:
            return
        self._cleaned_up = True
        try:
            await self.sandbox.cleanup()
        except Exception:
            logger.debug("Sandbox cleanup failed", exc_info=True)


# ---------------------------------------------------------------------------
# Math answer extraction (shared by gsm8k, aime, math benchmarks)
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    r"""Extract content from ``\boxed{...}`` handling nested braces.

    Returns:
        The content inside the outermost ``\boxed{}``, or ``None`` if not found.
    """
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def extract_number(text: str) -> str:
    """Extract a number from text, stripping LaTeX formatting.

    Removes ``\\text{}``, other LaTeX commands, braces, dollar signs, and
    commas, then returns the first number found (or the stripped text).
    """
    cleaned = re.sub(r"\\text\{[^}]*\}", "", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = cleaned.replace("{", "").replace("}", "").replace("$", "")
    cleaned = cleaned.replace(",", "").replace(" ", "")
    match = re.search(r"[-]?\d+\.?\d*", cleaned)
    return match.group(0) if match else cleaned.strip()


def extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from a model response.

    Tries ``\\boxed{}``, ``#### <answer>``, ``answer is/answer:`` patterns,
    then falls back to the last number in the text.

    Returns:
        Extracted numeric string, or ``""`` if no number is found.
    """
    boxed = extract_boxed(text)
    if boxed:
        return extract_number(boxed)

    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return extract_number(hash_match.group(1))

    answer_match = re.search(r"(?:answer is|answer:)\s*\$?([0-9,.-]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")

    # Bold pattern: **123** or **$123.00** (common in chat model responses)
    bold_numbers = re.findall(r"\*\*\$?([-]?\d+[,\d]*\.?\d*)", text)
    if bold_numbers:
        return bold_numbers[-1].replace(",", "").rstrip(".")

    # Last number in entire text (fallback)
    numbers = re.findall(r"[-]?\d+[,\d]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def check_gsm8k(response: str, expected: str) -> bool:
    """Check if the extracted numeric answer matches the expected answer.

    Args:
        response: Full model response text.
        expected: Expected numeric answer string.

    Returns:
        True if the extracted answer matches (within 1e-5 for floats,
        exact match for non-numeric strings).
    """
    extracted = extract_gsm8k_answer(response)
    # Strip commas from expected (GSM8K answers may contain "2,125" style formatting)
    expected_clean = expected.replace(",", "")
    try:
        return abs(float(extracted) - float(expected_clean)) < 1e-5
    except (ValueError, TypeError):
        return extracted.strip() == expected_clean.strip()


# ---------------------------------------------------------------------------
# MCQ answer extraction (shared by mmlu_pro, gpqa, mmlu_redux)
# ---------------------------------------------------------------------------


def extract_mcq_answer(text: str, valid_letters: str = "ABCD") -> str:
    """Extract a multiple-choice letter from a model response.

    Tries ``\\boxed{X}``, ``answer is (X)`` pattern, then the last standalone
    letter in the final 300 characters.

    Args:
        text: Full model response text.
        valid_letters: Allowed answer letters (default ``"ABCD"``).

    Returns:
        Uppercase letter from *valid_letters*, or ``""`` if none found.
    """
    pattern = f"[{valid_letters}]"

    # Try \boxed{X}
    boxed = extract_boxed(text)
    if boxed is not None:
        boxed_upper = boxed.strip().upper()
        if re.fullmatch(pattern, boxed_upper):
            return boxed_upper

    answer_match = re.search(
        rf"(?:answer is|answer:)\s*\(?([{valid_letters}])\)?",
        text,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).upper()

    letters = re.findall(rf"\b({pattern})\b", text[-300:])
    if letters:
        return letters[-1]
    return ""


# ---------------------------------------------------------------------------
# Python code extraction (shared by mbpp, livecodebench)
# ---------------------------------------------------------------------------


def extract_python_code(text: str) -> str:
    """Extract Python code from a model response.

    Tries a ``python`` fenced code block first, then any fenced block,
    then falls back to the full text.
    """
    match = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    matches = re.findall(r"```(?:\w*)\s*\n(.*?)\n```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Kwargs parsing (shared by ifeval, ifbench)
# ---------------------------------------------------------------------------


def parse_kwargs(raw_kwargs: list, instruction_ids: list | None = None) -> list[dict]:
    """Parse a list of raw kwargs (str/dict/None) into a list of dicts.

    Args:
        raw_kwargs: List of kwargs entries (None, JSON strings, or dicts).
        instruction_ids: If provided, pads the result to match its length.

    Returns:
        List of dicts, one per entry, with unparseable entries replaced by ``{}``.
    """
    kwargs_list: list[dict] = []
    for kw in raw_kwargs:
        if kw is None:
            kwargs_list.append({})
        elif isinstance(kw, str):
            try:
                kwargs_list.append(json.loads(kw))
            except json.JSONDecodeError:
                kwargs_list.append({})
        elif isinstance(kw, dict):
            kwargs_list.append(kw)
        else:
            kwargs_list.append({})
    if instruction_ids is not None:
        while len(kwargs_list) < len(instruction_ids):
            kwargs_list.append({})
    return kwargs_list
