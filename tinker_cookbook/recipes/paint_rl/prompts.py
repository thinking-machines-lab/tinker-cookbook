"""Policy system prompt and the painting-prompt dataset.

The system prompt follows the lesson from the "paint with code" project:
a short, strict allowlist of brush calls constrains the model better than a
long API reference (which mostly produced hallucinated APIs).

The prompt pool is ``subject x color x style``. It is built so the verifier
has something to resolve:

* **Subjects are what watercolorists commonly paint**: flowers, still-life
  objects, scenes and a few animals, in three difficulty tiers: tier 1 a single
  subject, tier 2 a subject in a setting, tier 3 two named elements in a
  spatial relation. A vague flower is the easiest blob to pass off, so the
  subject-fidelity criterion asks for a clear, specific depiction.
* **Colors and styles are visually separable** at 448 px (no golden/ochre,
  no rose-pink/coral/crimson, no loose/splashy/wet-on-wet near-synonyms), so
  the fidelity and technique criteria score signal rather than noise. The
  one deliberate near pair is blue violet next to violet and indigo.
* **A few skill prompts.** Eight fully worded requests test object counting,
  spatial relationships, shape and perspective (``SKILL_PROMPTS``). They name
  their own colors, so each is crossed with the four styles only, and they
  make up about a tenth of training; the rest is the simple subject base.
* **Held-out prompts, not held-out subjects.** The held-out set is a sample
  of the whole pool spread over every subject, color, style and tier; its
  exact prompts never appear in training, but every subject does. Training
  prompts are balanced over subjects so no subject is memorized more than
  another.

On this branch the pool is restricted to ``TIERS`` (tier 1 only), so both
training and held-out prompts are single subjects; the higher-tier subjects
and the skill prompts (all tier 2 or 3) stay defined, with their ids, but
are left out of the pool.

The split is deterministic in the seed and pinned by ``prompts_test.py``: a
change to the pool that moves the held-out set fails the test on purpose.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, TypeVar

SYSTEM_PROMPT_TEMPLATE = """You are a generative artist who paints watercolors with code, using p5.js and the p5.brush library.

Write ONE complete JavaScript sketch that paints the requested subject. Reply with a single ```javascript code block and nothing else.

Hard rules:
- Define `function setup()` only. No draw() loop, no async code, no images, fonts, fetch, DOM, or external files.
- setup() must start with exactly: createCanvas({size}, {size}, WEBGL); angleMode(DEGREES); background(<paper color>); brush.load();
- WEBGL coordinates: (0, 0) is the CENTER of the canvas. x and y run from -{half} to {half}; y grows downwards.
- Use ONLY these brush functions (any other brush.* call throws and the painting is lost):
    brush.pick(name)                 name in "2B","HB","2H","cpencil","pen","rotring","spray","marker","marker2","charcoal"
    brush.stroke(css_color)  brush.noStroke()  brush.strokeWeight(multiplier)
    brush.fill(css_color, opacity_0_to_255)  brush.noFill()
    brush.bleed(strength_0_to_0.5, "out" or "in")  brush.fillTexture(texture_0_to_1, border_0_to_1)
    brush.line(x1, y1, x2, y2)  brush.flowLine(x, y, length, angle_degrees)  brush.circle(x, y, radius)  brush.rect(x, y, w, h)
    brush.polygon([[x, y], ...])  brush.spline([[x, y], ...], curvature_0_to_1)
    brush.beginShape(curvature_0_to_1); brush.vertex(x, y); ...; brush.endShape(CLOSE)
    brush.hatch(spacing, angle_degrees)  brush.noHatch()
    brush.field(name)                name in "curved","truncated","zigzag","seabed","waves"   brush.noField()
    brush.push()  brush.pop()  brush.seed(number)
- List the vertices of every filled shape CLOCKWISE, otherwise the watercolor bleed inverts.
- Plain p5 helpers are fine for maths and color: random, noise, sin, cos, map, lerp, constrain, color, lerpColor, PI.

Painting advice: build the image from several translucent washes (fill opacity 40-120 with bleed 0.1-0.4), let colors overlap and pool, vary edge softness, keep pencil or marker outlines sparse and broken, avoid piling up many brush.line and brush.flowLine calls (use them only where a line is truly needed, and paint forms with filled shapes instead), and fill most of the canvas with a clear, recognizable subject.
"""


def system_prompt(canvas_size: int) -> str:
    """The policy's system prompt for a ``canvas_size`` x ``canvas_size`` canvas."""
    return SYSTEM_PROMPT_TEMPLATE.replace("{size}", str(canvas_size)).replace(
        "{half}", str(canvas_size // 2)
    )


# ---------------------------------------------------------------------------
# Subjects
# ---------------------------------------------------------------------------

Family = Literal["flower", "object", "scene", "animal", "skill"]
PromptKind = Literal["train", "test"]


@dataclass(frozen=True)
class Subject:
    """``name`` is the noun the color word attaches to; ``context`` is the rest."""

    name: str
    context: str
    tier: int
    family: Family

    @property
    def phrase(self) -> str:
        return f"{self.name} {self.context}".strip()


# What watercolorists paint most: flowers, still-life objects (including
# fruit) and scenes, each family spanning all three tiers, plus two animals.
SUBJECTS: tuple[Subject, ...] = (
    # flowers
    Subject("sunflower", "", 1, "flower"),
    Subject("rose", "", 1, "flower"),
    Subject("tulip", "", 1, "flower"),
    Subject("daisy", "", 1, "flower"),
    Subject("iris", "", 1, "flower"),
    Subject("lily", "", 1, "flower"),
    Subject("hydrangea", "", 1, "flower"),
    Subject("orchid", "in a terracotta pot", 2, "flower"),
    Subject("foxgloves", "along a cottage wall", 2, "flower"),
    Subject("lavender field", "stretching toward a distant farmhouse", 3, "flower"),
    Subject("lotus blossom", "floating on a still pond", 2, "flower"),
    Subject("cherry blossom branch", "against a pale sky", 2, "flower"),
    Subject("poppy field", "under a summer sky", 2, "flower"),
    Subject("magnolia branch", "in a glass vase", 2, "flower"),
    Subject("bouquet of peonies", "in a ceramic jug", 2, "flower"),
    Subject("wisteria vine", "trailing over a garden gate", 3, "flower"),
    Subject("clump of daffodils", "growing beside a stone birdbath", 3, "flower"),
    Subject("field of flowers", "", 1, "flower"),
    Subject("field of flowers", "on a rolling hillside", 2, "flower"),
    Subject("field of flowers", "at sunset", 2, "flower"),
    Subject("field of flowers", "with a path leading to a distant cottage", 3, "flower"),
    # still-life objects
    Subject("pear", "", 1, "object"),
    Subject("peach", "", 1, "object"),
    Subject("pomegranate", "", 1, "object"),
    Subject("bunch of grapes", "", 1, "object"),
    Subject("sliced watermelon", "on a white plate", 2, "object"),
    Subject("bowl of cherries", "on a lace cloth", 2, "object"),
    Subject("bunch of bananas", "beside a striped teacup", 3, "object"),
    Subject("hot air balloon", "", 1, "object"),
    Subject("watering can", "", 1, "object"),
    Subject("cup of coffee", "on a wooden table", 2, "object"),
    Subject("lantern", "hanging in a stone doorway", 2, "object"),
    Subject("bicycle", "leaning against a brick wall", 2, "object"),
    Subject("stack of old books", "on a windowsill", 2, "object"),
    Subject("basket of apples", "on a checked tablecloth", 2, "object"),
    Subject("bowl of lemons", "beside a glass bottle", 3, "object"),
    Subject("straw hat", "hanging on a chair beside an open window", 3, "object"),
    Subject("mailbox", "", 1, "object"),
    # scenes
    Subject("lighthouse", "", 1, "scene"),
    Subject("windmill", "", 1, "scene"),
    Subject("cottage", "", 1, "scene"),
    Subject("barn", "", 1, "scene"),
    Subject("mountain lake", "at dawn", 2, "scene"),
    Subject("old oak tree", "alone in a meadow", 2, "scene"),
    Subject("sailboat", "on a calm sea", 2, "scene"),
    Subject("forest", "in autumn", 2, "scene"),
    Subject("lighthouse", "with a sailboat passing in front of it", 3, "scene"),
    Subject("village street", "leading to a church under a crescent moon", 3, "scene"),
    Subject("stone bridge", "arching over a quiet river", 3, "scene"),
    Subject("classical building", "with columns and arched windows", 2, "scene"),
    Subject("forest path", "in dappled sunlight", 2, "scene"),
    # animals
    Subject("flock of birds", "in flight across the sky", 2, "animal"),
    Subject("fish", "swimming in a clear stream", 2, "animal"),
)

# Hues that stay distinct in a 448 px watercolor ("blue violet" is the one
# near neighbor: it sits between "violet" and "indigo").
COLORS: tuple[str, ...] = (
    "crimson",
    "coral",
    "golden",
    "emerald",
    "teal",
    "indigo",
    "violet",
    "sepia",
    "magenta",
    "olive",
    "sky blue",
    "charcoal gray",
    "blue violet",
)

# Four styles, each with a visual signature a reviewer can check.
STYLES: tuple[str, ...] = (
    "wet-on-wet watercolor wash",
    "minimal watercolor with lots of white paper",
    "watercolor with soft ink outlines",
    "layered watercolor glazes",
)


@dataclass(frozen=True)
class PaintPrompt:
    """One painting request. ``id`` names the artifact folder."""

    id: str
    text: str
    subject: str
    tier: int
    family: str
    color: str
    style: str
    kind: PromptKind = "train"

    def with_kind(self, kind: PromptKind) -> PaintPrompt:
        return PaintPrompt(**{**asdict(self), "kind": kind})


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


@dataclass(frozen=True)
class SkillPrompt:
    """A request that tests one compositional skill, worded in full.

    These carry their own colors, so they are not crossed with COLORS, only
    with STYLES: each is 4 prompts in the pool, which keeps them a small share
    of training next to the subject x color x style base.
    """

    request: str
    """The request without a style or final period."""
    category: str
    tier: int


SKILL_PROMPTS: tuple[SkillPrompt, ...] = (
    SkillPrompt("Paint exactly five red tulips in a vase", "object counting", 2),
    SkillPrompt("Paint exactly three white sailboats on a calm lake", "object counting", 2),
    SkillPrompt("Paint a blue vase to the left of a yellow teacup", "spatial relationships", 3),
    SkillPrompt(
        "Paint a red apple on top of a stack of two green books", "spatial relationships", 3
    ),
    SkillPrompt(
        "Paint a round ceramic bowl with a narrow base and wide opening", "shape and geometry", 2
    ),
    SkillPrompt("Paint a diamond-shaped kite with a long ribbon tail", "shape and geometry", 2),
    SkillPrompt(
        "Paint a winding road that becomes narrower toward the horizon", "depth and perspective", 2
    ),
    SkillPrompt(
        "Paint a row of trees along a canal, shrinking into the distance",
        "depth and perspective",
        2,
    ),
)


# Difficulty tiers kept in the pool, for training and held-out alike.
TIERS: tuple[int, ...] = (1,)


def build_prompt_pool() -> list[PaintPrompt]:
    """Every (subject, color, style) combination, in a fixed order, then
    every skill prompt in each style; only those in ``TIERS``. Ids index the
    full SUBJECTS / SKILL_PROMPTS lists, so they do not move with ``TIERS``."""
    prompts: list[PaintPrompt] = []
    for si, subject in enumerate(SUBJECTS):
        if subject.tier not in TIERS:
            continue
        for ci, color in enumerate(COLORS):
            for ti, style in enumerate(STYLES):
                text = f"Paint {_article(color)} {color} {subject.phrase} in {style}."
                prompts.append(
                    PaintPrompt(
                        id=f"s{si:02d}c{ci:02d}t{ti}",
                        text=text,
                        subject=subject.phrase,
                        tier=subject.tier,
                        family=subject.family,
                        color=color,
                        style=style,
                    )
                )
    for ki, skill in enumerate(SKILL_PROMPTS):
        if skill.tier not in TIERS:
            continue
        for ti, style in enumerate(STYLES):
            prompts.append(
                PaintPrompt(
                    id=f"k{ki:02d}t{ti}",
                    # The comma keeps "in a vase in layered glazes" unambiguous.
                    text=f"{skill.request}, in {style}.",
                    subject=skill.request,
                    tier=skill.tier,
                    family="skill",
                    color="",  # the colors are in the request itself
                    style=style,
                )
            )
    return prompts


T = TypeVar("T")


def _stratified_pick(
    items: Iterable[T],
    n: int,
    key: Callable[[T], object],
    rng: random.Random,
    balance: Sequence[Callable[[T], object]] = (),
) -> list[T]:
    """``n`` items spread as evenly as possible over the values of ``key``.

    Each stratum is shuffled, the strata are shuffled, and items are taken
    round-robin, so counts per stratum differ by at most one. Within a
    stratum the item whose ``balance`` attributes (e.g. color, style) have
    been used least so far is taken, so a small draw still covers every
    color and style rather than landing three sepias and no ink outlines.
    """
    strata: dict[object, list[T]] = defaultdict(list)
    for item in items:
        strata[key(item)].append(item)
    order = list(strata)
    rng.shuffle(order)
    for k in order:
        rng.shuffle(strata[k])
    used: list[dict[object, int]] = [defaultdict(int) for _ in balance]
    picked: list[T] = []
    while len(picked) < n and any(strata[k] for k in order):
        for k in order:
            if not strata[k] or len(picked) >= n:
                continue
            # Least-used balance attributes first; the shuffle breaks ties.
            best = min(
                range(len(strata[k])),
                key=lambda i: tuple(u[f(strata[k][i])] for u, f in zip(used, balance, strict=True)),
            )
            item = strata[k].pop(best)
            for u, f in zip(used, balance, strict=True):
                u[f(item)] += 1
            picked.append(item)
    if len(picked) < n:
        raise ValueError(f"asked for {n} prompts but only {len(picked)} are available")
    return picked


# Default split sizes; train.py's CLI uses these too.
N_TRAIN_PROMPTS = 256
# Held-out prompts, a sample of the whole pool.
N_TEST_PROMPTS = 50


def build_prompt_splits(
    n_train: int = N_TRAIN_PROMPTS,
    n_test: int = N_TEST_PROMPTS,
    seed: int = 0,
) -> tuple[list[PaintPrompt], list[PaintPrompt]]:
    """(train, test) prompt lists.

    * test: spread over subjects (then colors, styles and tiers), so every
      subject is evaluated; no subject is held out of training.
    * train: balanced over subjects, colors and styles, never one of the
      exact test prompts.
    """
    rng = random.Random(seed)
    pool = build_prompt_pool()
    by_look = (lambda p: p.color, lambda p: p.style)
    test = [
        p.with_kind("test")
        for p in _stratified_pick(
            pool, n_test, lambda p: p.subject, rng, balance=(*by_look, lambda p: p.tier)
        )
    ]
    held = {p.id for p in test}
    train = _stratified_pick(
        [p for p in pool if p.id not in held], n_train, lambda p: p.subject, rng, balance=by_look
    )
    return train, test


def split_summary(prompts: Sequence[PaintPrompt]) -> dict[str, int]:
    """Counts by kind and tier, for logging."""
    counts: dict[str, int] = defaultdict(int)
    for p in prompts:
        counts[p.kind] += 1
        counts[f"{p.kind}/tier{p.tier}"] += 1
    return dict(sorted(counts.items()))
