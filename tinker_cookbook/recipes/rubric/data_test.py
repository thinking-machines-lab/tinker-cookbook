from tinker_cookbook.recipes.rubric.data import Rubric


def test_extract_score_accepts_singular_score_tag():
    rubric = Rubric(rubric_str="Correct?")

    assert rubric.extract_score("<score>1</score>") == 1.0


def test_extract_score_accepts_plural_scores_tag():
    rubric = Rubric(rubric_str="Correct?", extraction_regex=r"<score>(.*)</score>")

    assert rubric.extract_score("<scores>0</scores>") == 0.0
