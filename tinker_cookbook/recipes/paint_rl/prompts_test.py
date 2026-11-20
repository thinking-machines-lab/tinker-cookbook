from collections import Counter

from tinker_cookbook.recipes.paint_rl.prompts import (
    COLORS,
    N_TEST_PROMPTS,
    N_TRAIN_PROMPTS,
    SKILL_PROMPTS,
    STYLES,
    SUBJECTS,
    TIERS,
    build_prompt_pool,
    build_prompt_splits,
    system_prompt,
)

# The held-out set every arm (verifier and judge) is evaluated on (seed 0,
# 50 tier-1 prompts spread over every tier-1 subject). Pinned on purpose: an edit to
# the pool or the splitter that moves these prompts makes runs incomparable,
# so it has to be a deliberate change here as well.
PINNED_HELDOUT_SEED0 = [
    ("s22c01t1", "Paint a coral peach in minimal watercolor with lots of white paper."),
    ("s24c12t0", "Paint a blue violet bunch of grapes in wet-on-wet watercolor wash."),
    ("s37c09t2", "Paint an olive mailbox in watercolor with soft ink outlines."),
    ("s41c05t3", "Paint an indigo barn in layered watercolor glazes."),
    ("s00c11t2", "Paint a charcoal gray sunflower in watercolor with soft ink outlines."),
    ("s39c06t0", "Paint a violet windmill in wet-on-wet watercolor wash."),
    ("s23c03t3", "Paint an emerald pomegranate in layered watercolor glazes."),
    ("s02c07t1", "Paint a sepia tulip in minimal watercolor with lots of white paper."),
    ("s03c08t2", "Paint a magenta daisy in watercolor with soft ink outlines."),
    ("s05c04t1", "Paint a teal lily in minimal watercolor with lots of white paper."),
    ("s40c10t3", "Paint a sky blue cottage in layered watercolor glazes."),
    ("s04c00t0", "Paint a crimson iris in wet-on-wet watercolor wash."),
    ("s06c02t0", "Paint a golden hydrangea in wet-on-wet watercolor wash."),
    ("s17c06t3", "Paint a violet field of flowers in layered watercolor glazes."),
    (
        "s38c11t1",
        "Paint a charcoal gray lighthouse in minimal watercolor with lots of white paper.",
    ),
    ("s21c09t2", "Paint an olive pear in watercolor with soft ink outlines."),
    ("s01c00t3", "Paint a crimson rose in layered watercolor glazes."),
    ("s29c07t2", "Paint a sepia watering can in watercolor with soft ink outlines."),
    ("s28c12t0", "Paint a blue violet hot air balloon in wet-on-wet watercolor wash."),
    ("s22c04t1", "Paint a teal peach in minimal watercolor with lots of white paper."),
    ("s24c10t3", "Paint a sky blue bunch of grapes in layered watercolor glazes."),
    ("s37c08t2", "Paint a magenta mailbox in watercolor with soft ink outlines."),
    ("s41c01t0", "Paint a coral barn in wet-on-wet watercolor wash."),
    ("s00c02t1", "Paint a golden sunflower in minimal watercolor with lots of white paper."),
    ("s39c05t3", "Paint an indigo windmill in layered watercolor glazes."),
    ("s23c03t2", "Paint an emerald pomegranate in watercolor with soft ink outlines."),
    ("s02c03t1", "Paint an emerald tulip in minimal watercolor with lots of white paper."),
    ("s03c00t0", "Paint a crimson daisy in wet-on-wet watercolor wash."),
    ("s05c10t2", "Paint a sky blue lily in watercolor with soft ink outlines."),
    ("s40c06t1", "Paint a violet cottage in minimal watercolor with lots of white paper."),
    ("s04c04t0", "Paint a teal iris in wet-on-wet watercolor wash."),
    ("s06c05t3", "Paint an indigo hydrangea in layered watercolor glazes."),
    ("s17c07t0", "Paint a sepia field of flowers in wet-on-wet watercolor wash."),
    ("s38c12t3", "Paint a blue violet lighthouse in layered watercolor glazes."),
    ("s21c11t2", "Paint a charcoal gray pear in watercolor with soft ink outlines."),
    ("s01c01t1", "Paint a coral rose in minimal watercolor with lots of white paper."),
    ("s29c02t0", "Paint a golden watering can in wet-on-wet watercolor wash."),
    ("s28c09t2", "Paint an olive hot air balloon in watercolor with soft ink outlines."),
    ("s22c08t1", "Paint a magenta peach in minimal watercolor with lots of white paper."),
    ("s24c07t3", "Paint a sepia bunch of grapes in layered watercolor glazes."),
    ("s37c03t1", "Paint an emerald mailbox in minimal watercolor with lots of white paper."),
    ("s41c10t0", "Paint a sky blue barn in wet-on-wet watercolor wash."),
    ("s00c01t2", "Paint a coral sunflower in watercolor with soft ink outlines."),
    ("s39c09t3", "Paint an olive windmill in layered watercolor glazes."),
    ("s23c11t2", "Paint a charcoal gray pomegranate in watercolor with soft ink outlines."),
    ("s02c02t0", "Paint a golden tulip in wet-on-wet watercolor wash."),
    ("s03c05t1", "Paint an indigo daisy in minimal watercolor with lots of white paper."),
    ("s05c00t3", "Paint a crimson lily in layered watercolor glazes."),
    ("s40c08t3", "Paint a magenta cottage in layered watercolor glazes."),
    ("s04c06t2", "Paint a violet iris in watercolor with soft ink outlines."),
]


def test_pool_shape():
    pool = build_prompt_pool()
    # only tier 1 is in the pool, and no skill prompt is tier 1
    assert TIERS == (1,) and all(p.tier == 1 for p in pool)
    assert not any(p.family == "skill" for p in pool)
    tier1 = [s for s in SUBJECTS if s.tier in TIERS]
    assert len(pool) == len(tier1) * len(COLORS) * len(STYLES) == 988
    assert len({p.id for p in pool}) == len(pool)
    assert len({s.phrase for s in SUBJECTS}) == len(SUBJECTS)
    assert Counter(s.family for s in SUBJECTS) == {
        "flower": 21, "object": 17, "scene": 13, "animal": 2,
    }  # fmt: skip
    assert Counter(s.tier for s in SUBJECTS) == {1: 19, 2: 24, 3: 10}


def test_animal_family():
    assert {s.name for s in SUBJECTS if s.family == "animal"} == {
        "flock of birds", "fish",
    }  # fmt: skip


def test_subject_variants():
    pool = build_prompt_pool()
    assert sum(s.name == "field of flowers" for s in SUBJECTS) == 4
    # only the tier-1 "field of flowers" is in the pool
    assert len([p for p in pool if p.subject == "field of flowers"]) == len(COLORS) * len(STYLES)
    assert not any(p.subject.startswith("field of flowers ") for p in pool)


def test_articles():
    pool = build_prompt_pool()
    for p in pool:
        if p.family == "skill":  # worded in full, no color slot
            continue
        article = p.text.split()[1]
        assert article == ("an" if p.color[0] in "aeiou" else "a"), p.text


def _spread(counts: Counter[str]) -> int:
    return max(counts.values()) - min(counts.values())


def test_default_split_is_pinned():
    _, test = build_prompt_splits(seed=0)
    assert [(p.id, p.text) for p in test] == PINNED_HELDOUT_SEED0
    assert all(p.kind == "test" for p in test)


def test_split_invariants():
    train, test = build_prompt_splits(seed=0)
    train2, test2 = build_prompt_splits(seed=0)
    assert [p.id for p in train] == [p.id for p in train2]
    assert [p.id for p in test] == [p.id for p in test2]
    assert not {p.id for p in train} & {p.id for p in test}
    assert len(train) == N_TRAIN_PROMPTS and len(test) == N_TEST_PROMPTS
    # no held-out subjects: every tier-1 subject is trained on and held out
    everything = {s.phrase for s in SUBJECTS if s.tier in TIERS}
    assert {p.subject for p in train} == everything
    assert {p.subject for p in test} == everything
    assert all(p.tier in TIERS for p in train + test)
    # balanced over subjects, colors and styles
    for prompts in (train, test):
        base = prompts
        assert _spread(Counter(p.subject for p in base)) <= 1
        colors = Counter(p.color for p in base)
        styles = Counter(p.style for p in prompts)
        assert set(colors) == set(COLORS) and _spread(colors) <= 1
        assert set(styles) == set(STYLES) and _spread(styles) <= 2
    assert all(p.kind == "train" for p in train)


def test_skill_prompts_stay_defined():
    # out of the pool (none is tier 1) but kept, with their ids, for later
    assert all(k.tier not in TIERS for k in SKILL_PROMPTS)
    assert {k.category for k in SKILL_PROMPTS} == {
        "object counting", "spatial relationships", "shape and geometry", "depth and perspective",
    }  # fmt: skip


def test_other_seeds_and_sizes_work():
    train, test = build_prompt_splits(n_train=40, n_test=8, seed=7)
    assert len(train) == 40 and len(test) == 8
    assert not {p.id for p in train} & {p.id for p in test}


def test_system_prompt_mentions_canvas_size():
    text = system_prompt(512)
    assert "createCanvas(512, 512, WEBGL)" in text
    assert "-256" in text and "{size}" not in text


def test_system_prompt_does_not_ask_for_noloop():
    # The page stops the draw loop itself; a noLoop() rule invites brush.noLoop().
    assert "noLoop" not in system_prompt(512)
