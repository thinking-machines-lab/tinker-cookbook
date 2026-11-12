"""Regression tests for numeric answer extraction and integer grading."""

import asyncio

import pytest

from tinker_cookbook.eval.benchmarks._common import check_gsm8k, extract_gsm8k_answer
from tinker_cookbook.eval.benchmarks.aime import AIMEMessageEnv


@pytest.mark.parametrize("answer", ["693.75", "-36.5", "1,234.50", "42"])
@pytest.mark.parametrize("marker", ["The answer is ", "Answer: $"])
def test_answer_statement_sentence_period(answer: str, marker: str):
    expected = answer.replace(",", "")
    response = f"{marker}{answer}."
    assert extract_gsm8k_answer(response) == expected
    assert check_gsm8k(response, expected)
    assert not check_gsm8k(response, str(float(expected) + 1))


@pytest.mark.parametrize(
    "response",
    [
        "Total: _**123**_. We used 2 steps.",
        "Total: __**123**__. We used 2 steps.",
        "Total: result**123**units. We used 2 steps.",
        "Total: (**123**). We used 2 steps.",
        "Total: ***123***. We used 2 steps.",
        "Total: **123**. We used 2 steps.",
    ],
)
def test_bold_numeric_formatting_keeps_existing_priority(response: str):
    assert extract_gsm8k_answer(response) == "123"
    assert check_gsm8k(response, "123")
    assert not check_gsm8k(response, "2")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The total is **123**. We used 2 steps.", "123"),
        ("The total is **$1,234.50**. We used 2 steps.", "1234.50"),
        ("The balance is **-36.5**. We used 2 steps.", "-36.5"),
        ("First **12**, finally **42**. We used 2 steps.", "42"),
    ],
)
def test_complete_bold_numeric_answer(response: str, expected: str):
    assert extract_gsm8k_answer(response) == expected
    assert check_gsm8k(response, expected)
    assert not check_gsm8k(response, "2")


@pytest.mark.parametrize(
    ("response", "expected", "reward"),
    [
        ("The answer is 693.75", 693, 0.0),
        ("The answer is 693.75.", 693, 0.0),
        (r"\boxed{693.75}", 693, 0.0),
        ("The answer is -0.75", 0, 0.0),
        ("The answer is 693.00000000000000001", 693, 0.0),
        ("The answer is 692.99999999999999999", 693, 0.0),
        ("The answer is 693", 693, 1.0),
        ("The answer is 693.", 693, 1.0),
        ("The answer is 693.0", 693, 1.0),
        (r"\boxed{693}", 693, 1.0),
        (r"\boxed{007}", 7, 1.0),
        (r"\boxed{000}", 0, 1.0),
        (r"\boxed{999}", 999, 1.0),
        (r"\boxed{694}", 693, 0.0),
        (r"\boxed{-1}", -1, 0.0),
        (r"\boxed{1000}", 1000, 0.0),
        ("No numeric answer.", 693, 0.0),
        ("The answer is 6.9.3", 693, 0.0),
    ],
)
def test_aime_requires_integral_extracted_token(response: str, expected: int, reward: float):
    env = AIMEMessageEnv("A synthetic arithmetic problem", expected)
    result = asyncio.run(env.step({"role": "assistant", "content": response}))
    assert result.reward == reward
    assert result.metrics == {"correct": reward}
    assert result.episode_done
    assert result.next_messages == []


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Using a**2 + b**2 = c**2 with legs 3 and 4, the hypotenuse is 5.", "5"),
        ("She has 2**3 = 8 apples left, so she has 8 apples.", "8"),
        ("The area is (1+2)**2 = 9 square meters.", "9"),
        ("The side is x[0]**2 = 16 meters long.", "16"),
        ("The area is 3**2 = **9** square meters, from 2 sides.", "9"),
        ("That makes 2**3**2 = 512 apples.", "512"),
        ("Since x**2**(n+1) = 64, the answer is 6.", "6"),
        ("With {n}**2 = 25, n is 5.", "5"),
    ],
)
def test_exponent_is_not_a_bold_answer(response: str, expected: str):
    assert check_gsm8k(response, expected)
    assert not check_gsm8k(response, "2")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("She earns **18 dollars** in total, over 3 days.", "18"),
        ("She earns **$18 per day**, over 3 days.", "18"),
        ("(**18 dollars**) over 3 days.", "18"),
        ("She earns _**18 dollars**_ in total, over 3 days.", "18"),
        ("She earned a total of**18**, over 3 days.", "18"),
        ("She earned a total of**$18**, over 3 days.", "18"),
        ("答案是**18元**，共3天。", "18"),
    ],
)
def test_bold_answer_with_trailing_words(response: str, expected: str):
    assert extract_gsm8k_answer(response) == expected
