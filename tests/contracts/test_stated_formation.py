"""Source provenance never authorizes an automatically extracted subclaim."""

import json

import pytest

from sensemem import EvidenceBasis, FormationInput, ModelInput, ObservationContext
from sensemem.models._openai_output import _formation_results
from sensemem.models._openai_requests import _formation_aliases
from sensemem.models._openai_schemas import response_schema
from sensemem.prompts import generation as prompts


def test_formation_prompt_keeps_changed_preferences_in_the_existing_property() -> None:
    prompt = prompts._FORMATION_SYSTEM_PROMPT
    assert "reuse its supported historical kind and predicate" in prompt
    assert all(key in prompt for key in ("preferred_drink", "home_city", "current_location"))
    assert "current single-valued preference\nis a STATE" in prompt
    assert "Do not add current, only, new, or changed to the predicate" in prompt
    assert "preference_status=changed" in prompt


@pytest.mark.parametrize("kind", ("state", "trait"))
@pytest.mark.parametrize("source_basis", tuple(EvidenceBasis))
@pytest.mark.parametrize(
    "text",
    (
        "Rava has been patient for years.",
        "Lea reports: Rava has been patient for years.",
        "Rava is not patient.",
    ),
)
def test_original_words_do_not_authorize_the_models_interpretation(
    kind: str, source_basis: EvidenceBasis, text: str
) -> None:
    source = FormationInput("source", ModelInput(text=text), ObservationContext(basis=source_basis))
    proposal = {
        "kind": kind,
        "content": text,
        "subject": "Rava",
        "predicate": "disposition",
        "value": "patient",
        "confidence": 0.9,
        "evidence_observation_ids": ["observation_0"],
    }
    (proposals,) = _formation_results(
        json.dumps({"items": [{"observation_id": "observation_0", "proposals": [proposal]}]}),
        (source,),
        source_aliases=_formation_aliases((source,)),
    )
    assert len(proposals) == 1
    assert proposals[0].basis is EvidenceBasis.MODEL_INFERENCE
    assert source.context.basis is source_basis


@pytest.mark.parametrize(
    "extra", ({"statement_quote": "Rava is patient."}, {"basis": "user_statement"})
)
def test_model_authority_fields_are_rejected(extra: dict[str, object]) -> None:
    source = FormationInput(
        "source",
        ModelInput(text="Rava is patient."),
        ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
    )
    proposal = {
        "kind": "trait",
        "content": source.content.text,
        "subject": "Rava",
        "predicate": "disposition",
        "value": "patient",
        "confidence": 0.9,
        "evidence_observation_ids": ["observation_0"],
        **extra,
    }
    assert _formation_results(
        json.dumps({"items": [{"observation_id": "observation_0", "proposals": [proposal]}]}),
        (source,),
        source_aliases=_formation_aliases((source,)),
    ) == ((),)


def test_formation_schema_does_not_offer_authority_fields() -> None:
    schema = json.dumps(response_schema("formation"))
    assert "statement_quote" not in schema
    assert '"basis"' not in schema
