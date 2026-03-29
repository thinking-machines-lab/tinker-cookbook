"""Unit tests for the Nemotron-Cascade-2 recipe.

Tests import availability, config parsing, and utility functions.
No Tinker API or HuggingFace downloads required.
"""

import importlib

import pytest


# ---------------------------------------------------------------------------
# 1. RL env imports — each module importable and has its DatasetBuilder
# ---------------------------------------------------------------------------

ENV_MODULES = {
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.if_rl": "IFRLDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa": "MCQARLDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.structured_output": "StructuredOutputRLDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.workbench": "WorkbenchRLDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl": "CodeRLDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.longctx": "LongContextRLDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.rlhf": "RLHFDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.swe_agentless": "SWERLDatasetBuilder",
    "tinker_cookbook.recipes.nemotron_cascade.rl.envs.swe_agentic": "SWEAgenticDatasetBuilder",
}


@pytest.mark.parametrize("module_path,class_name", ENV_MODULES.items(), ids=ENV_MODULES.values())
def test_rl_env_import(module_path: str, class_name: str):
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert cls is not None, f"{class_name} not found in {module_path}"


# ---------------------------------------------------------------------------
# 2. SFT dataset builder imports
# ---------------------------------------------------------------------------

def test_sft_builder_imports():
    from tinker_cookbook.recipes.nemotron_cascade.sft.datasets import (
        NemotronCascadeSFTBuilder,
        NemotronCascadeSFTFromFileBuilder,
    )
    assert NemotronCascadeSFTBuilder is not None
    assert NemotronCascadeSFTFromFileBuilder is not None


# ---------------------------------------------------------------------------
# 3. train_rl CLI config parsing
# ---------------------------------------------------------------------------

ALL_ENV_NAMES = [
    "if_rl", "mcqa", "structured_output", "workbench",
    "code_rl", "longctx_rl", "rlhf", "swe_rl", "swe_agentic",
]


@pytest.mark.parametrize("env_name", ALL_ENV_NAMES)
def test_cli_config_parses(env_name: str):
    from tinker_cookbook.recipes.nemotron_cascade.rl.train import CLIConfig
    config = CLIConfig(env=env_name)
    assert config.env == env_name
    assert config.group_size > 0
    assert config.groups_per_batch > 0


# ---------------------------------------------------------------------------
# 4. utils — strip_think_blocks
# ---------------------------------------------------------------------------

def test_strip_think_blocks_basic():
    from tinker_cookbook.recipes.nemotron_cascade.utils import strip_think_blocks
    assert strip_think_blocks("hello") == "hello"


def test_strip_think_blocks_removes_think():
    from tinker_cookbook.recipes.nemotron_cascade.utils import strip_think_blocks
    text = "before <think>reasoning here</think> after"
    result = strip_think_blocks(text)
    assert "reasoning here" not in result
    assert "before" in result
    assert "after" in result


def test_strip_think_blocks_unclosed():
    from tinker_cookbook.recipes.nemotron_cascade.utils import strip_think_blocks
    text = "answer <think>this was truncated and never closed"
    result = strip_think_blocks(text)
    assert "truncated" not in result
    assert "answer" in result


def test_strip_think_blocks_empty():
    from tinker_cookbook.recipes.nemotron_cascade.utils import strip_think_blocks
    assert strip_think_blocks("<think>only thinking</think>") == ""


# ---------------------------------------------------------------------------
# 5. Answer extraction — mcqa_rl_env.extract_answer
# ---------------------------------------------------------------------------

def test_extract_answer_boxed():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    assert extract_answer("The answer is \\boxed{B}") == "B"


def test_extract_answer_xml():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    assert extract_answer("<final_answer>C</final_answer>") == "C"


def test_extract_answer_double_parens():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    assert extract_answer("I think ((A))") == "A"


def test_extract_answer_the_answer_is():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    result = extract_answer("After analysis, the answer is D.")
    assert result == "D"


def test_extract_answer_bold():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    assert extract_answer("Therefore **B**") == "B"


def test_extract_answer_standalone_letter():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    assert extract_answer("Some reasoning\nA") == "A"


def test_extract_answer_with_think_block():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    text = "<think>Let me reason about this...</think>The answer is \\boxed{C}"
    assert extract_answer(text) == "C"


def test_extract_answer_option_selected():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import extract_answer
    assert extract_answer("Option Selected: B") == "B"


# ---------------------------------------------------------------------------
# 6. Code extraction — code_rl_env.extract_code
# ---------------------------------------------------------------------------

def test_extract_code_python_fenced():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import extract_code
    response = "Here is my solution:\n```python\ndef solve():\n    return 42\n```"
    code, lang = extract_code(response)
    assert code is not None
    assert "def solve" in code
    assert lang == "python"


def test_extract_code_cpp_fenced():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import extract_code
    response = "Solution:\n```cpp\n#include <iostream>\nint main() { return 0; }\n```"
    code, lang = extract_code(response)
    assert code is not None
    assert "#include" in code
    assert lang == "cpp"


def test_extract_code_unfenced_python():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import extract_code
    response = "import sys\ndef solve(n):\n    return n * 2\n\nprint(solve(5))\n\nThis is a great solution."
    code, lang = extract_code(response)
    assert code is not None
    assert lang == "python"


def test_extract_code_no_code():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import extract_code
    response = "I'm not sure how to solve this problem."
    code, lang = extract_code(response)
    assert code is None
    assert lang == "unknown"


def test_extract_code_strips_think():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import extract_code
    response = "<think>```python\ndraft_code()\n```</think>\n```python\nfinal_code()\n```"
    code, lang = extract_code(response)
    assert code is not None
    assert "final_code" in code
    assert "draft_code" not in code


def test_extract_code_last_match():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import extract_code
    response = "First attempt:\n```python\nfirst()\n```\nBetter version:\n```python\nsecond()\n```"
    code, lang = extract_code(response)
    assert code is not None
    assert "second" in code


def test_extract_code_generic_fenced_with_include():
    from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import extract_code
    response = "```\n#include <stdio.h>\nint main() {}\n```"
    code, lang = extract_code(response)
    assert code is not None
    assert lang == "cpp"
