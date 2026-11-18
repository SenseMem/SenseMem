"""Provider output admission stays separate from request construction and model calls."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from sensemem import FormationInput, ModelInput, ObservationContext
from sensemem.exceptions import ModelError, ModelOutputTruncatedError
from sensemem.models._openai_output import (
    _consolidation_results,
    _embedding_vectors,
    _evidence_operation_response,
    _formation_results,
    _InvalidStructuredResponse,
    _json_completion_text,
    _parse_media_requests,
    _visual_description,
)
from sensemem.records import MemoryRecord

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("finish", "refusal", "reason", "retryable_output"),
    [
        ("stop", "private refusal", "model_refused", False),
        ("content_filter", None, "model_refused", False),
        ("length", None, "output_truncated", False),
        ("tool_calls", None, "response_invalid", True),
    ],
)
def test_json_completion_admission_preserves_failure_class_and_safe_errors(
    finish: str, refusal: str | None, reason: str, retryable_output: bool
) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=finish,
                message=SimpleNamespace(content="private response", refusal=refusal),
            )
        ]
    )
    with pytest.raises(ModelError) as raised:
        _json_completion_text(response, subject="formation", stage="form")
    assert raised.value.reason == reason
    assert raised.value.stage == "form"
    assert isinstance(raised.value, _InvalidStructuredResponse) is retryable_output
    assert isinstance(raised.value, ModelOutputTruncatedError) is (finish == "length")
    assert "private" not in str(raised.value)


@pytest.mark.parametrize(
    ("finish", "allowed", "count", "expected"),
    [
        ("stop", False, 2, "protocol_finish"),
        ("tool_calls", False, 2, "protocol_type"),
        ("tool_calls", True, 2, "protocol_count"),
        ("tool_calls", True, 1, "protocol_type"),
    ],
)
def test_native_protocol_admission_preserves_first_rejection_when_fields_are_invalid(
    finish: str, allowed: bool, count: int, expected: str
) -> None:
    call = SimpleNamespace(type="wrong", function=SimpleNamespace(name="wrong", arguments={}))
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=finish,
                message=SimpleNamespace(refusal=None, tool_calls=[call] * count),
            )
        ]
    )
    answer, arguments, rejection = _evidence_operation_response(
        response, tools_allowed=allowed, hits=()
    )
    assert answer is None and arguments is None
    assert rejection == expected


def test_embeddings_restore_provider_order_and_reject_repeated_indices() -> None:
    first = SimpleNamespace(index=0, embedding=[1.0, 2.0])
    second = SimpleNamespace(index=1, embedding=[3.0, 4.0])
    assert _embedding_vectors(SimpleNamespace(data=[second, first]), 2, 2) == (
        (1.0, 2.0),
        (3.0, 4.0),
    )
    with pytest.raises(ModelError, match="embedding response was invalid"):
        _embedding_vectors(SimpleNamespace(data=[first, first]), 2, 2)


def test_formation_uses_only_the_targets_supplied_history_aliases() -> None:
    first_history = MemoryRecord(id="first-history", content="Ada is the visitor.", created_at=NOW)
    other_history = MemoryRecord(id="other-history", content="Ben is the visitor.", created_at=NOW)
    inputs = (
        FormationInput(
            "first", ModelInput(text="She arrived."), ObservationContext(), history=(first_history,)
        ),
        FormationInput(
            "second", ModelInput(text="He left."), ObservationContext(), history=(other_history,)
        ),
    )
    proposals = [
        {
            "kind": "event",
            "content": "Ada arrived.",
            "confidence": 0.9,
            "evidence_observation_ids": ["observation_0", history],
        }
        for history in ("history_0", "history_1")
    ]
    content = json.dumps(
        {
            "items": [
                {"observation_id": "observation_0", "proposals": proposals},
                {"observation_id": "observation_1", "proposals": []},
            ]
        }
    )
    first, second = _formation_results(
        content,
        inputs,
        source_aliases=({"first-history": "history_0"}, {"other-history": "history_1"}),
    )
    assert len(first) == 1 and second == ()
    assert first[0].evidence_ids == ("first", "first-history")


def test_invalid_consolidation_key_does_not_echo_provider_content() -> None:
    with pytest.raises(_InvalidStructuredResponse) as raised:
        _consolidation_results(
            '{"operations":[{"intent":"reinforce","private response":true}]}', ()
        )
    assert raised.value.stage == "consolidate"
    assert "private response" not in str(raised.value)


def test_visual_output_separates_standing_facts_and_only_stated_speaker_names() -> None:
    parsed = _visual_description(
        "Shown: a person by a toolbox.\n"
        "Fact: speaker_1 is called Ada.\n"
        "Fact: speaker_1 prefers tea.\n"
        "Fact: the visitor is called Ben."
    )
    assert parsed.text == "Shown: a person by a toolbox."
    assert len(parsed.facts) == 3
    assert [(name.speaker_label, name.name) for name in parsed.speaker_names] == [
        ("speaker_1", "Ada")
    ]


@pytest.mark.parametrize(
    "requests",
    [
        [{"media_index": 0, "reason": "detail_missing"}] * 2,
        [{"media_index": 1, "reason": "detail_missing"}],
        [{"media_index": 0, "reason": "answer_the_question"}],
    ],
)
def test_media_diagnosis_rejects_duplicates_unlisted_indices_and_invented_reasons(
    requests: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        _parse_media_requests(json.dumps({"requests": requests}), available=1, limit=2)
