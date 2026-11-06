"""Schema diagnostics retain static shapes without rejected provider input or context."""

import json
from dataclasses import FrozenInstanceError, asdict, fields
from typing import cast

import pytest
from pydantic import ValidationError as PydanticValidationError
from test_evidence_operations import QUESTION, _hit, _proposal, _text

from sensemem.models._evidence_operations import (
    EvidenceOperation,
    InvalidEvidenceOperation,
    execute_evidence_operation,
    make_evidence_catalogue,
)

PRIVATE = "PRIVATE_SYNTHETIC_PROVIDER_INPUT"


def _payload() -> dict[str, object]:
    return _proposal(operation="count", anchors=(_text("E1", "No relevant entries."),)).model_dump(
        mode="json"
    )


def _failure(payload: dict[str, object] | str) -> InvalidEvidenceOperation:
    with pytest.raises(InvalidEvidenceOperation) as caught:
        execute_evidence_operation(
            json.dumps(payload) if isinstance(payload, dict) else payload,
            make_evidence_catalogue((_hit("source", "No relevant entries."),), visible_media={}),
            question=QUESTION,
        )
    error = caught.value
    assert error.category == "schema"
    assert error.args == ("native evidence operation arguments were invalid",)
    assert error.__context__ is None and error.__cause__ is None
    assert PRIVATE not in json.dumps([asdict(issue) for issue in error.schema_issues])
    return error


def test_missing_required_nullable_field_reports_only_its_static_name() -> None:
    payload = _payload()
    del payload["measure"]
    issues = _failure(payload).schema_issues
    assert [(issue.error_type, issue.location) for issue in issues] == [("missing", ("measure",))]
    for field in fields(issues[0]):
        with pytest.raises(FrozenInstanceError):
            setattr(issues[0], field.name, PRIVATE)


def test_wrong_scalar_type_has_sanitized_array_item_location() -> None:
    payload = _payload()
    payload["contributions"] = [
        {"member_id": "one", "basis": "whole", "value_ids": ["value"], "parts_complete": PRIVATE}
    ]
    issues = _failure(payload).schema_issues
    assert [(issue.error_type, issue.location) for issue in issues] == [
        ("bool_type", ("contributions", "item", "parts_complete"))
    ]


def test_wrong_collection_type_does_not_retain_its_value() -> None:
    payload = _payload()
    payload["members"] = PRIVATE
    issues = _failure(payload).schema_issues
    assert [(issue.error_type, issue.location) for issue in issues] == [
        ("tuple_type", ("members",))
    ]


@pytest.mark.parametrize("key", [PRIVATE, "members"])
def test_extra_user_controlled_field_is_mapped_to_other(key: str) -> None:
    payload = _payload()
    cast(dict[str, object], payload["scope"])[key] = PRIVATE
    issues = _failure(payload).schema_issues
    assert [(issue.error_type, issue.location) for issue in issues] == [
        ("extra_forbidden", ("scope", "other"))
    ]


def test_malformed_json_reports_root_without_input_or_decoder_message() -> None:
    issues = _failure('{"private":"' + PRIVATE).schema_issues
    assert [(issue.error_type, issue.location) for issue in issues] == [("json_invalid", ("root",))]


def test_anchor_model_validator_context_is_discarded() -> None:
    payload = _payload()
    inventory = cast(dict[str, object], payload["inventory"])
    inventory["anchors"] = [
        {"evidence_label": "E1", "quote": None, "attachment": None, "observation": PRIVATE}
    ]
    issues = _failure(payload).schema_issues
    assert [(issue.error_type, issue.location) for issue in issues] == [
        ("value_error", ("inventory", "anchors", "item"))
    ]


def test_schema_issue_count_is_bounded() -> None:
    payload = _payload()
    payload.update({f"{PRIVATE}_{index}": PRIVATE for index in range(25)})
    issues = _failure(payload).schema_issues
    assert len(issues) == 16
    assert all(
        issue.error_type == "extra_forbidden" and issue.location == ("other",) for issue in issues
    )


def test_unlisted_pydantic_code_and_unknown_location_are_mapped_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid(_arguments: object) -> object:
        raise PydanticValidationError.from_exception_data(
            PRIVATE, [{"type": "int_type", "loc": (PRIVATE, 58), "input": PRIVATE}]
        )

    monkeypatch.setattr(EvidenceOperation, "model_validate_json", invalid)
    issues = _failure(_payload()).schema_issues
    assert [(issue.error_type, issue.location) for issue in issues] == [
        ("other", ("other", "item"))
    ]


def test_non_schema_failures_do_not_fabricate_schema_issues() -> None:
    payload = _payload()
    scope = cast(dict[str, object], payload["scope"])
    scope["basis"] = "runtime_closed"
    with pytest.raises(InvalidEvidenceOperation) as caught:
        execute_evidence_operation(
            json.dumps(payload),
            make_evidence_catalogue((_hit("source", "No relevant entries."),), visible_media={}),
            question=QUESTION,
        )
    assert caught.value.category == "invalid_evidence"
    assert caught.value.schema_issues == ()
