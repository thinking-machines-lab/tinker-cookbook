"""Network-free unit tests for the EkaCare medical-ASR env helpers.

The data path (``load_ekacare_split``) needs the HF dataset + the optional
``tml_renderers`` build, so it is exercised by running the recipe rather than
here; these cover the pure logic: medical-entity parsing and the entity CER
metric.
"""

from __future__ import annotations

from tinker_cookbook.recipes.audio.medical_asr.env import (
    _entity_cer,
    _entity_scores,
    _norm_medical,
    _parse_entities,
)


def test_parse_entities_extracts_surface_strings() -> None:
    assert _parse_entities('[["BestoChem Formulations India limited", "drugs", [[0, 36]]]]') == [
        "BestoChem Formulations India limited"
    ]
    assert _parse_entities(
        '[["adequate rest","advices",[[11,24]]],["fever","clinical_findings",[[0,5]]]]'
    ) == ["adequate rest", "fever"]


def test_parse_entities_is_defensive() -> None:
    assert _parse_entities(None) == []
    assert _parse_entities("") == []
    assert _parse_entities("not json{") == []
    assert _parse_entities("[]") == []
    assert _parse_entities('[[123, "x", []], ["ok", "drugs", []]]') == ["ok"]


def test_norm_medical_collapses_units_and_possessives() -> None:
    assert _norm_medical("Telmisartan 40 mg Tablet") == _norm_medical("telmisartan 40mg tablet")
    assert _norm_medical("Behcet's disease") == _norm_medical("behcet disease")


def test_entity_cer_is_graded() -> None:
    # Exact (after normalization) -> 0.
    assert (
        _entity_cer(_norm_medical("40mg tablet"), _norm_medical("take 40 mg tablet daily")) == 0.0
    )
    # One-letter drug-name slip -> small, not 1.0.
    small = _entity_cer(_norm_medical("udinol"), _norm_medical("odinol 300mg tablet"))
    assert 0.0 < small <= 0.25
    # Completely absent term -> ~1.0.
    assert _entity_cer(_norm_medical("azithromycin"), _norm_medical("patient rested well")) > 0.7


def test_entity_scores_aggregates() -> None:
    ents = ["Telmisartan 40 mg Tablet", "hypertension"]
    # hyp gets the drug (spacing differs) but drops "hypertension".
    weighted, length, hits, count = _entity_scores(ents, "telmisartan 40mg tablet was prescribed")
    assert count == 2
    assert hits == 1  # drug within tolerance, condition missed
    assert length > 0
    corpus_cer = weighted / length
    assert 0.0 < corpus_cer < 1.0  # blend of a ~0 hit and a ~1 miss
    # No entities -> contributes nothing.
    assert _entity_scores([], "anything") == (0.0, 0, 0, 0)
