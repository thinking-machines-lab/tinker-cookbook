"""Tests for LLM judge reward functions (pure function tests only)."""

from tinker_cookbook.rewards.llm_judge import Rubric


class TestRubric:
    def test_extract_score_valid(self):
        rubric = Rubric(rubric_str="test")
        assert rubric.extract_score("blah <score>0.75</score> blah") == 0.75

    def test_extract_score_missing(self):
        rubric = Rubric(rubric_str="test")
        assert rubric.extract_score("no score here") == 0.0

    def test_extract_score_invalid_number(self):
        rubric = Rubric(rubric_str="test")
        assert rubric.extract_score("<score>abc</score>") == 0.0

    def test_extract_score_custom_regex(self):
        rubric = Rubric(
            rubric_str="test",
            extraction_regex=r"\[(\d+\.?\d*)\]",
        )
        assert rubric.extract_score("Score: [0.9]") == 0.9

    def test_get_grader_prompt(self):
        rubric = Rubric(rubric_str="Is the answer correct?")
        convo = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        prompt = rubric.get_grader_prompt(convo)
        assert len(prompt) == 1
        assert prompt[0]["role"] == "user"
        content = prompt[0]["content"]
        assert "What is 2+2?" in content
        assert "Chatbot: 4" in content
        assert "Is the answer correct?" in content

    def test_get_grader_prompt_no_context(self):
        rubric = Rubric(rubric_str="Rate this.")
        convo = [{"role": "assistant", "content": "Hello!"}]
        prompt = rubric.get_grader_prompt(convo)
        content = prompt[0]["content"]
        assert "(No prior context)" in content
