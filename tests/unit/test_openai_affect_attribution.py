"""Typed emotional claims preserve reference scope and exact report provenance."""

import copy
import json
from datetime import datetime, timezone

import pytest

from sensemem import (
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    ModelInput,
    ObservationContext,
)
from sensemem.models._openai_output import _formation_results
from sensemem.models._openai_requests import _formation_aliases, _formation_history_payload
from sensemem.models._openai_schemas import response_schema
from sensemem.records import MemoryRecord
from sensemem.semantics import AffectLink, AffectObjectRef, AffectWitness, RecordSemantics
from sensemem.types import RecordKind

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)
QUOTE = "Ana explicitly says, 'I feel angry because the launch was canceled.'"


def _proposals() -> list[dict[str, object]]:
    witness = {"source_id": "observation_0", "reporter": "Ana", "quote": QUOTE}
    return [
        {
            "local_id": "feeling",
            "kind": "affect",
            "subject": "Ana",
            "value": "angry",
            "content": QUOTE,
            "confidence": 0.9,
            "cue_modality": "text",
            "valence": -0.6,
            "arousal": 0.7,
            "affect_witnesses": [witness],
            "evidence_observation_ids": ["observation_0"],
        },
        {
            "local_id": "launch",
            "kind": "event",
            "content": "The launch was canceled.",
            "confidence": 0.9,
            "evidence_observation_ids": ["observation_0"],
        },
        {
            "kind": "relation",
            "subject": "Ana",
            "predicate": "affect_elicitor",
            "value": "launch",
            "content": QUOTE,
            "confidence": 0.8,
            "affect_witnesses": [witness],
            "evidence_observation_ids": ["observation_0"],
            "affect_link": {
                "affect_id": "proposal:feeling",
                "role": "elicitor",
                "object_ref": {"memory_id": "proposal:launch", "identity_id": None},
                "polarity": "affirmed",
            },
        },
    ]


def _parse(
    proposals: list[dict[str, object]],
    text: str = QUOTE,
    *,
    source_basis: EvidenceBasis = EvidenceBasis.USER_STATEMENT,
) -> tuple[FormationProposal, ...]:
    source = FormationInput(
        "source",
        ModelInput(text=text),
        ObservationContext(basis=source_basis),
    )
    return _formation_results(
        json.dumps({"items": [{"observation_id": "observation_0", "proposals": proposals}]}),
        (source,),
        source_aliases=_formation_aliases((source,)),
    )[0]


def test_local_attribution_preserves_separate_inferences_and_exact_report() -> None:
    affect, event, relation = _parse(_proposals())
    assert affect.basis is relation.basis is EvidenceBasis.MODEL_INFERENCE
    assert event.basis is EvidenceBasis.MODEL_INFERENCE
    assert relation.affect_link == AffectLink(
        affect_id="proposal:feeling",
        role="elicitor",
        object_ref=AffectObjectRef(memory_id="proposal:launch"),
    )
    assert relation.affect_witnesses == (
        AffectWitness(source_id="source", reporter="Ana", quote=QUOTE),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"predicate": "caused_by"},
        {"affect_link": None},
        {"affect_witnesses": [{"source_id": "source", "reporter": "Ana", "quote": QUOTE}]},
        {"affect_witnesses": [{"source_id": "observation_0", "reporter": "Bob", "quote": QUOTE}]},
        {
            "affect_witnesses": [
                {"source_id": "observation_0", "reporter": "Ana", "quote": "Ana invented this."}
            ]
        },
        {"affect_witnesses": 1},
        {
            "affect_link": {
                "affect_id": "guessed-id",
                "role": "elicitor",
                "object_ref": {"memory_id": "proposal:launch", "identity_id": None},
                "polarity": "affirmed",
            }
        },
        {
            "affect_link": {
                "affect_id": "proposal:missing",
                "role": "elicitor",
                "object_ref": {"memory_id": "proposal:launch", "identity_id": None},
                "polarity": "affirmed",
            }
        },
    ],
)
def test_bad_attribution_costs_only_that_relation(change: dict[str, object]) -> None:
    proposals = _proposals()
    proposals[-1].update(change)
    assert [proposal.kind for proposal in _parse(proposals)] == [
        RecordKind.AFFECT,
        RecordKind.EVENT,
    ]


def test_third_party_quote_is_preserved_without_self_statement_authority() -> None:
    text = "Lea reports, 'Ana feels angry because the launch was canceled.'"
    proposals = _proposals()
    for proposal in (proposals[0], proposals[-1]):
        proposal["content"] = text
        proposal["affect_witnesses"] = [
            {"source_id": "observation_0", "reporter": "Lea", "quote": text}
        ]
    affect, _event, relation = _parse(proposals, text)
    assert affect.basis is relation.basis is EvidenceBasis.MODEL_INFERENCE
    assert relation.affect_witnesses[0].reporter == "Lea"


@pytest.mark.parametrize("source_basis", tuple(EvidenceBasis))
def test_a_quoted_subject_cannot_be_misrepresented_as_an_authoritative_reporter(
    source_basis: EvidenceBasis,
) -> None:
    text = "Ana says Bob feels angry because the launch was canceled."
    proposals = _proposals()
    for proposal in (proposals[0], proposals[-1]):
        proposal.update(subject="Bob", content=text)
        proposal["affect_witnesses"] = [
            {"source_id": "observation_0", "reporter": "Bob", "quote": text}
        ]
    affect, _event, relation = _parse(proposals, text, source_basis=source_basis)
    assert affect.basis is relation.basis is EvidenceBasis.MODEL_INFERENCE
    assert relation.affect_witnesses[0] == AffectWitness(
        source_id="source", reporter="Bob", quote=text
    )


def test_duplicate_local_names_do_not_make_a_valid_reference() -> None:
    proposals = _proposals()
    proposals[1]["local_id"] = "feeling"
    proposals.append(
        {
            "kind": "event",
            "content": "An independently valid event.",
            "confidence": 0.9,
            "evidence_observation_ids": ["observation_0"],
        }
    )
    (survivor,) = _parse(proposals)
    assert survivor.content == "An independently valid event."


def test_link_cannot_borrow_another_observations_private_history() -> None:
    private = MemoryRecord(id="private", content="A private event.", created_at=NOW)
    inputs = (
        FormationInput("source", ModelInput(text=QUOTE), ObservationContext()),
        FormationInput(
            "other", ModelInput(text="Other source."), ObservationContext(), history=(private,)
        ),
    )
    proposals = _proposals()
    proposals[0].update(affect_witnesses=[])
    proposals[-1].update(affect_witnesses=[])
    proposals[-1]["affect_link"] = {
        "affect_id": "proposal:feeling",
        "role": "elicitor",
        "object_ref": {"memory_id": "history_0", "identity_id": None},
        "polarity": "affirmed",
    }
    first, second = _formation_results(
        json.dumps(
            {
                "items": [
                    {"observation_id": "observation_0", "proposals": proposals},
                    {"observation_id": "observation_1", "proposals": []},
                ]
            }
        ),
        inputs,
        source_aliases=_formation_aliases(inputs),
    )
    assert [proposal.kind for proposal in first] == [RecordKind.AFFECT, RecordKind.EVENT]
    assert second == ()


def test_historical_link_aliases_resolve_and_private_history_cannot_be_named() -> None:
    history = tuple(
        MemoryRecord(id=identifier, content="stored claim", created_at=NOW)
        for identifier in ("feeling-id", "launch-id")
    )
    source = FormationInput("source", ModelInput(text=QUOTE), ObservationContext(), history=history)
    aliases = _formation_aliases((source,))
    relation = copy.deepcopy(_proposals()[-1])
    relation.update(affect_witnesses=[])
    relation["evidence_observation_ids"] = ["observation_0", "history_0", "history_1"]
    relation["affect_link"] = {
        "affect_id": "history_0",
        "role": "elicitor",
        "object_ref": {"memory_id": "history_1", "identity_id": None},
        "polarity": "denied",
    }
    parsed = _formation_results(
        json.dumps({"items": [{"observation_id": "observation_0", "proposals": [relation]}]}),
        (source,),
        source_aliases=aliases,
    )[0]
    assert parsed[0].affect_link == AffectLink(
        affect_id="feeling-id",
        role="elicitor",
        object_ref=AffectObjectRef(memory_id="launch-id"),
        polarity="denied",
    )


def test_history_payload_uses_aliases_for_links_and_keeps_reporter() -> None:
    context = RecordSemantics(
        kind=RecordKind.RELATION,
        basis=EvidenceBasis.MODEL_INFERENCE,
        confidence=0.7,
        valid_from=None,
        valid_until=None,
        recorded_at=NOW,
        subject="Ana",
        predicate="affect_elicitor",
        value="launch",
        affect_link=AffectLink(
            affect_id="feeling", role="elicitor", object_ref=AffectObjectRef(memory_id="launch")
        ),
        affect_witnesses=(
            AffectWitness(source_id="source", reporter="Lea", quote="Lea reports it."),
        ),
    )
    row = MemoryRecord(id="relation", content="Lea's report", created_at=NOW, context=context)
    payload = _formation_history_payload(
        row, "history_0", {"feeling": "history_1", "launch": "history_2", "source": "history_3"}
    )
    payload_context = payload["context"]
    assert isinstance(payload_context, dict)
    assert payload_context["affect_link"]["affect_id"] == "history_1"
    assert payload_context["affect_witnesses"] == [
        {"source_id": "history_3", "reporter": "Lea", "quote": "Lea reports it."}
    ]


def test_strict_schema_adds_attribution_only_to_formation_affect_and_relation() -> None:
    schema = response_schema("formation")
    properties = schema["properties"]
    assert isinstance(properties, dict)
    variants = properties["items"]["items"]["properties"]["proposals"]["items"]["anyOf"]
    by_kind = {variant["properties"]["kind"]["enum"][0]: variant for variant in variants}
    assert "affect_link" in by_kind["relation"]["required"]
    assert "affect_witnesses" in by_kind["affect"]["required"]
    assert "local_id" in by_kind["event"]["required"]
    assert "affect_link" not in by_kind["event"]["properties"]
