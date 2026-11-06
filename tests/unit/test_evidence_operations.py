"""Pure scoped execution checks; these fixtures do not establish real-model quality."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Literal, cast

import pytest

from sensemem.models._evidence_operations import (
    EVIDENCE_OPERATION_PARAMETERS,
    EVIDENCE_OPERATION_TOOL_NAME,
    EvidenceAnchor,
    EvidenceOperation,
    EvidenceOperationOutcome,
    IdentityKey,
    IdentityRelation,
    InvalidEvidenceOperation,
    MemberContribution,
    MemberInventory,
    OperationMember,
    OperationValue,
    Predicate,
    PredicateVerdict,
    ScopeAssessment,
    execute_evidence_operation,
    make_evidence_catalogue,
)
from sensemem.types import AbstentionReason, SearchHit

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
QUESTION = (
    "Within the supplied records, give the total paid, weight, count or matching bird lying down."
)


def _hit(identifier: str, content: str, source: str | None = None) -> SearchHit:
    return SearchHit(
        id=identifier,
        content=content,
        created_at=NOW,
        score=0.8,
        metadata={} if source is None else {"source_id": source},
    )


def _text(label: str, quote: str) -> EvidenceAnchor:
    return EvidenceAnchor(evidence_label=label, quote=quote, attachment=None, observation=None)


def _native(label: str, attachment: str, observation: str) -> EvidenceAnchor:
    return EvidenceAnchor(
        evidence_label=label, quote=None, attachment=attachment, observation=observation
    )


def _member(
    identifier: str,
    anchor: EvidenceAnchor,
    *,
    conditions: tuple[PredicateVerdict, ...] = (),
    identity: IdentityKey | None = None,
) -> OperationMember:
    return OperationMember(id=identifier, anchor=anchor, conditions=conditions, identity=identity)


def _fact(
    identifier: str,
    member: str,
    anchor: EvidenceAnchor,
    value: str,
    *,
    label: str = "TOTAL",
    role: str = "paid",
    unit: str | None = "EUR",
    extent: Literal["whole", "part", "quantity", "per_unit"] = "whole",
    part_key: str | None = None,
) -> OperationValue:
    return OperationValue(
        id=identifier,
        member_id=member,
        role=role,
        label=label,
        value=value,
        unit=unit,
        extent=extent,
        part_key=part_key,
        anchor=anchor,
    )


def _contribution(
    member: str,
    *facts: str,
    basis: Literal["whole", "partition", "quantity_per_unit"] = "whole",
    complete: bool = True,
) -> MemberContribution:
    return MemberContribution(
        member_id=member, basis=basis, value_ids=tuple(facts), parts_complete=complete
    )


def _proposal(
    *,
    operation: Literal["select_sources", "count", "scalar", "sum", "difference"] = "scalar",
    unit: Literal["source", "entity"] = "entity",
    quantifier: Literal["exact", "confirmed_subset", "one"] = "exact",
    measure: str = "paid",
    labels: tuple[str, ...] = ("E1",),
    members: tuple[OperationMember, ...] = (),
    values: tuple[OperationValue, ...] = (),
    contributions: tuple[MemberContribution, ...] = (),
    predicates: tuple[Predicate, ...] = (),
    identities: tuple[IdentityRelation, ...] = (),
    anchors: tuple[EvidenceAnchor, ...] = (),
    inventory_status: Literal["complete", "partial", "unknown"] = "complete",
    scope_status: Literal["adequate", "uncertain", "incomplete"] = "adequate",
) -> EvidenceOperation:
    numeric = operation in {"scalar", "sum", "difference"}
    return EvidenceOperation(
        operation=operation,
        member_unit=unit,
        quantifier=quantifier,
        measure=measure if numeric else None,
        format="source_ids"
        if operation == "select_sources"
        else "with_unit"
        if numeric
        else "number",
        scope=ScopeAssessment(
            kind="open_corpus",
            basis="model_assessed",
            status=scope_status,
            question_quote="supplied records",
            evidence_labels=labels,
            reason="The relevant supplied sources support this requested claim.",
        ),
        predicates=predicates,
        inventory=MemberInventory(status=inventory_status, anchors=anchors),
        members=members,
        identities=identities,
        values=values,
        contributions=contributions,
    )


def _run(
    proposal: EvidenceOperation,
    hits: Sequence[SearchHit],
    *,
    visible_media: Mapping[str, frozenset[str]] | None = None,
) -> EvidenceOperationOutcome:
    return execute_evidence_operation(
        proposal.model_dump_json(),
        make_evidence_catalogue(hits, visible_media={} if visible_media is None else visible_media),
        question=QUESTION,
    )


def test_schema_has_native_name_and_strict_required_objects() -> None:
    assert EVIDENCE_OPERATION_TOOL_NAME == "submit_evidence_operation"
    assert EVIDENCE_OPERATION_PARAMETERS["additionalProperties"] is False
    assert "scope" in cast(list[str], EVIDENCE_OPERATION_PARAMETERS["required"])
    with pytest.raises(InvalidEvidenceOperation, match="arguments were invalid"):
        execute_evidence_operation(
            '{"operation":"scalar","unexpected":"private provider body"}',
            make_evidence_catalogue((), visible_media={}),
            question=QUESTION,
        )


def test_catalogue_preserves_alias_order_and_shared_native_attachment() -> None:
    first = _hit("a", "Source A", "photo-a")
    second = _hit("b", "Source B", "photo-b")
    catalogue = make_evidence_catalogue(
        (first, second, first),
        visible_media={"E1": frozenset({"attachment 1"}), "E2": frozenset({"attachment 1"})},
    )
    assert catalogue.hits == (first, second)
    proposal = _proposal(
        operation="select_sources",
        unit="source",
        labels=("E1", "E2"),
        members=(
            _member("a", _native("E1", "attachment 1", "A bird")),
            _member("b", _native("E2", "attachment 1", "A bird")),
        ),
    )
    result = execute_evidence_operation(proposal.model_dump_json(), catalogue, question=QUESTION)
    assert result.answer == "photo-a, photo-b"


def test_scalar_total_ignores_unrelated_sources_and_line_inventory() -> None:
    anchor = _text("E1", "Subtotal 50 EUR. TOTAL 58.40 EUR.")
    proposal = _proposal(
        members=(_member("invoice", anchor),),
        values=(_fact("total", "invoice", anchor, "58.40"),),
        contributions=(_contribution("invoice", "total"),),
        inventory_status="unknown",
    )
    result = _run(
        proposal, (_hit("invoice", anchor.quote or ""), _hit("unrelated", "Unknown date"))
    )
    assert (result.status, result.answer, result.scope_assurance) == (
        "executed",
        "58.4 EUR",
        "model_assessed",
    )
    assert result.reason is None
    assert len(result.hits) == 1


@pytest.mark.parametrize(
    "anchor",
    [
        _text("E2", "TOTAL 12 EUR"),
        _text("E1", "TOTAL 2 EUR"),
        _native("E1", "attachment 2", "TOTAL 12 EUR"),
    ],
)
def test_foreign_quote_source_or_unsupplied_native_binding_is_invalid(
    anchor: EvidenceAnchor,
) -> None:
    proposal = _proposal(
        members=(_member("invoice", anchor),),
        values=(_fact("total", "invoice", anchor, "12"),),
        contributions=(_contribution("invoice", "total"),),
    )
    with pytest.raises(InvalidEvidenceOperation):
        _run(
            proposal,
            (_hit("invoice", "TOTAL 12 EUR"),),
            visible_media={"E1": frozenset({"attachment 1"})},
        )


def test_runtime_closed_cannot_be_minted_by_model() -> None:
    anchor = _text("E1", "TOTAL 12 EUR")
    proposal = _proposal(members=(_member("one", anchor),))
    proposal = proposal.model_copy(
        update={"scope": proposal.scope.model_copy(update={"basis": "runtime_closed"})}
    )
    with pytest.raises(InvalidEvidenceOperation, match="runtime scope receipt"):
        _run(proposal, (_hit("one", "TOTAL 12 EUR"),))


def test_false_condition_excludes_member_even_when_another_condition_is_unknown() -> None:
    predicates = (
        Predicate(id="bird", question_quote="bird"),
        Predicate(id="pose", question_quote="lying down"),
    )
    bird = _text("E1", "A bird lying down")
    dog = _text("E2", "A dog; pose unreadable")
    members = (
        _member(
            "bird",
            bird,
            conditions=(
                PredicateVerdict(predicate_id="bird", verdict="true", anchors=(bird,)),
                PredicateVerdict(predicate_id="pose", verdict="true", anchors=(bird,)),
            ),
        ),
        _member(
            "dog",
            dog,
            conditions=(
                PredicateVerdict(predicate_id="bird", verdict="false", anchors=(dog,)),
                PredicateVerdict(predicate_id="pose", verdict="unknown", anchors=()),
            ),
        ),
    )
    result = _run(
        _proposal(
            operation="select_sources",
            unit="source",
            labels=("E1", "E2"),
            members=members,
            predicates=predicates,
        ),
        (_hit("a", bird.quote or "", "photo-a"), _hit("b", dog.quote or "", "photo-b")),
    )
    assert result.status == "executed"
    assert result.answer == "photo-a"


def test_relevant_unknown_requires_evidence_but_confirmed_subset_can_execute() -> None:
    anchor = _text("E1", "A bird; pose unreadable")
    predicate = Predicate(id="pose", question_quote="lying down")
    proposal = _proposal(
        operation="count",
        unit="source",
        predicates=(predicate,),
        members=(
            _member(
                "one",
                anchor,
                conditions=(PredicateVerdict(predicate_id="pose", verdict="unknown", anchors=()),),
            ),
        ),
    )
    result = _run(proposal, (_hit("a", anchor.quote or "", "photo-a"),))
    assert result.status == "needs_evidence"
    assert result.answer is None
    assert result.missing == ("membership_unknown",)
    assert result.reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    subset = proposal.model_copy(update={"quantifier": "confirmed_subset"})
    result = _run(subset, (_hit("a", anchor.quote or "", "photo-a"),))
    assert (result.status, result.answer) == ("executed", "0")


def test_resolved_empty_count_is_zero_not_unknown() -> None:
    anchor = _text("E1", "No matching entries.")
    result = _run(
        _proposal(operation="count", anchors=(anchor,)), (_hit("one", anchor.quote or ""),)
    )
    assert (result.status, result.answer) == ("executed", "0")


def test_source_projection_uses_host_metadata_and_not_asset_or_internal_id() -> None:
    first = _hit("internal-a", "A bird", "photo-a")
    second = _hit("internal-b", "Another bird", "photo-a")
    proposal = _proposal(
        operation="select_sources",
        unit="source",
        labels=("E1", "E2"),
        members=(
            _member("a", _text("E1", "A bird")),
            _member("b", _text("E2", "Another bird")),
        ),
    )
    assert _run(proposal, (first, second)).answer == "photo-a"
    with pytest.raises(InvalidEvidenceOperation, match="application source metadata"):
        _run(proposal, (_hit("hash-not-public", "A bird"), second))
    custom = _hit("a", "A bird")
    custom = SearchHit(
        id=custom.id,
        content=custom.content,
        score=custom.score,
        created_at=NOW,
        metadata={"document_key": "receipt-a"},
    )
    catalogue = make_evidence_catalogue(
        (custom,), visible_media={}, source_metadata_key="document_key"
    )
    single = _proposal(
        operation="select_sources", unit="source", members=(_member("a", _text("E1", "A bird")),)
    )
    assert (
        execute_evidence_operation(single.model_dump_json(), catalogue, question=QUESTION).answer
        == "receipt-a"
    )


def test_native_numeric_transcription_is_separate_from_stored_caption() -> None:
    anchor = _native("E1", "attachment 1", "TOTAL 14 EUR")
    proposal = _proposal(
        members=(_member("a", anchor),),
        values=(_fact("total", "a", anchor, "14"),),
        contributions=(_contribution("a", "total"),),
    )
    result = _run(
        proposal,
        (_hit("one", "Stored caption says TOTAL 19 EUR"),),
        visible_media={"E1": frozenset({"attachment 1"})},
    )
    assert result.answer == "14 EUR"
    # This proves binding and calculation, not that the visual transcription was correct.


def _two_values(*, zero: bool = False) -> tuple[EvidenceOperation, tuple[SearchHit, ...]]:
    value = "0" if zero else "15"
    anchors = (_text("E1", f"TOTAL {value} EUR"), _text("E2", f"TOTAL {value} EUR"))
    proposal = _proposal(
        operation="sum",
        labels=("E1", "E2"),
        members=tuple(
            _member(identifier, anchor)
            for identifier, anchor in zip(("a", "b"), anchors, strict=True)
        ),
        values=tuple(
            _fact(identifier, identifier, anchor, value)
            for identifier, anchor in zip(("a", "b"), anchors, strict=True)
        ),
        contributions=(_contribution("a", "a"), _contribution("b", "b")),
    )
    return proposal, (_hit("a", anchors[0].quote or ""), _hit("b", anchors[1].quote or ""))


def test_unknown_transaction_identity_blocks_sum_but_not_source_projection() -> None:
    proposal, hits = _two_values()
    result = _run(proposal, hits)
    assert result.status == "needs_evidence"
    assert result.answer is None
    assert result.missing == ("identity_unknown",)
    projection = proposal.model_copy(
        update={
            "operation": "select_sources",
            "format": "source_ids",
            "measure": None,
            "values": (),
            "contributions": (),
        }
    )
    assert (
        _run(projection, (_hit("a", "TOTAL 15 EUR", "a"), _hit("b", "TOTAL 15 EUR", "b"))).answer
        == "a, b"
    )


def test_identity_unknown_does_not_block_an_invariant_zero_sum() -> None:
    proposal, hits = _two_values(zero=True)
    result = _run(proposal, hits)
    assert (result.status, result.answer) == ("executed", "0 EUR")


def test_distinct_source_bound_identity_keys_allow_sum_and_equal_keys_deduplicate() -> None:
    anchors = (
        _text("E1", "Vale A TOTAL 25 EUR"),
        _text("E2", "Vale A duplicate TOTAL 25 EUR"),
        _text("E3", "Vale B TOTAL 40 EUR"),
    )
    members = tuple(
        _member(
            identifier, anchor, identity=IdentityKey(namespace="Vale", value=key, anchor=anchor)
        )
        for identifier, key, anchor in zip(
            ("a", "duplicate", "b"), ("A", "A", "B"), anchors, strict=True
        )
    )
    proposal = _proposal(
        operation="sum",
        labels=("E1", "E2", "E3"),
        members=members,
        values=(_fact("a", "a", anchors[0], "25"), _fact("b", "b", anchors[2], "40")),
        contributions=(_contribution("a", "a"), _contribution("b", "b")),
    )
    result = _run(
        proposal, tuple(_hit(str(i), anchor.quote or "") for i, anchor in enumerate(anchors))
    )
    assert (result.status, result.answer) == ("executed", "65 EUR")


def test_explicit_same_and_different_conflict_is_invalid() -> None:
    proposal, hits = _two_values()
    anchors = tuple(member.anchor for member in proposal.members)
    relations = tuple(
        IdentityRelation(left="a", right="b", relation=relation, anchors=anchors)
        for relation in ("same", "different")
    )
    with pytest.raises(InvalidEvidenceOperation, match="contradicts"):
        _run(proposal.model_copy(update={"identities": relations}), hits)


def test_partition_requires_disjoint_supported_parts_and_complete_measure() -> None:
    anchor = _text("E1", "Deposit 10 EUR; Settlement 15 EUR; TOTAL 25 EUR")
    member = _member("a", anchor)
    values = (
        _fact(
            "deposit",
            "a",
            _text("E1", "Deposit 10 EUR"),
            "10",
            label="Deposit",
            extent="part",
            part_key="Deposit",
        ),
        _fact(
            "settlement",
            "a",
            _text("E1", "Settlement 15 EUR"),
            "15",
            label="Settlement",
            extent="part",
            part_key="Settlement",
        ),
        _fact("whole", "a", _text("E1", "TOTAL 25 EUR"), "25"),
    )
    proposal = _proposal(
        members=(member,),
        values=values,
        contributions=(_contribution("a", "deposit", "settlement", basis="partition"),),
    )
    assert _run(proposal, (_hit("a", anchor.quote or ""),)).answer == "25 EUR"
    incomplete = proposal.model_copy(
        update={
            "contributions": (
                _contribution("a", "deposit", "settlement", basis="partition", complete=False),
            )
        }
    )
    assert _run(incomplete, (_hit("a", anchor.quote or ""),)).status == "needs_evidence"
    invalid = proposal.model_copy(
        update={
            "contributions": (
                _contribution("a", "whole", "deposit", "settlement", basis="partition"),
            )
        }
    )
    with pytest.raises(InvalidEvidenceOperation, match="partition"):
        _run(invalid, (_hit("a", anchor.quote or ""),))


def test_explicit_integer_quantity_times_per_unit_binds_one_member() -> None:
    anchor = _text("E1", "Pack count 3; Unit weight 200 g; TOTAL 600 g")
    facts = (
        _fact(
            "quantity",
            "a",
            _text("E1", "Pack count 3"),
            "3",
            label="Pack count",
            role="quantity",
            unit=None,
            extent="quantity",
        ),
        _fact(
            "unit",
            "a",
            _text("E1", "Unit weight 200 g"),
            "200",
            label="Unit weight",
            role="weight",
            unit="g",
            extent="per_unit",
        ),
        _fact("whole", "a", _text("E1", "TOTAL 600 g"), "600", role="weight", unit="g"),
    )
    proposal = _proposal(
        measure="weight",
        members=(_member("a", anchor),),
        values=facts,
        contributions=(_contribution("a", "quantity", "unit", basis="quantity_per_unit"),),
    )
    assert _run(proposal, (_hit("a", anchor.quote or ""),)).answer == "600 g"
    invalid = proposal.model_copy(
        update={
            "contributions": (
                _contribution("a", "quantity", "unit", "whole", basis="quantity_per_unit"),
            )
        }
    )
    with pytest.raises(InvalidEvidenceOperation, match="one quantity"):
        _run(invalid, (_hit("a", anchor.quote or ""),))


def test_conflicting_duplicate_measure_values_need_evidence_not_invalid_fallback() -> None:
    anchor = _text("E1", "TOTAL 10 EUR; corrected? TOTAL 12 EUR")
    proposal = _proposal(
        members=(_member("a", anchor),),
        values=(
            _fact("first", "a", _text("E1", "TOTAL 10 EUR"), "10"),
            _fact("other", "a", _text("E1", "TOTAL 12 EUR"), "12"),
        ),
        contributions=(_contribution("a", "first"),),
    )
    result = _run(proposal, (_hit("a", anchor.quote or ""),))
    assert result.status == "needs_evidence"
    assert result.answer is None
    assert result.missing == ("conflicting_values",)


def test_model_scope_uncertainty_needs_evidence_and_missing_final_value_is_normal() -> None:
    anchor = _text("E1", "A relevant invoice with unreadable amount")
    proposal = _proposal(members=(_member("a", anchor),), scope_status="uncertain")
    result = _run(proposal, (_hit("a", anchor.quote or ""),))
    assert result.status == "needs_evidence"
    assert set(result.missing) == {"scope_insufficient", "value_missing"}


def test_invalid_errors_never_echo_model_arguments() -> None:
    with pytest.raises(InvalidEvidenceOperation) as caught:
        execute_evidence_operation(
            json.dumps({"secret": "private-memory-provider-body"}),
            make_evidence_catalogue((), visible_media={}),
            question=QUESTION,
        )
    assert "private-memory-provider-body" not in str(caught.value)


@pytest.mark.parametrize("inventory", ["complete", "partial", "unknown"])
def test_one_request_without_positive_match_does_not_turn_unknown_into_empty(
    inventory: Literal["complete", "partial", "unknown"],
) -> None:
    anchor = _text("E1", "A bird; pose unreadable")
    proposal = _proposal(
        operation="select_sources",
        unit="source",
        quantifier="one",
        inventory_status=inventory,
        predicates=(Predicate(id="pose", question_quote="lying down"),),
        members=(
            _member(
                "a",
                anchor,
                conditions=(PredicateVerdict(predicate_id="pose", verdict="unknown", anchors=()),),
            ),
        ),
    )
    result = _run(proposal, (_hit("a", anchor.quote or "", "photo-a"),))
    assert result.status == "needs_evidence"
    assert result.answer is None
    assert "membership_unknown" in result.missing


def test_one_without_candidates_and_partial_inventory_needs_evidence() -> None:
    anchor = _text("E1", "The relevant image is incomplete")
    proposal = _proposal(
        operation="select_sources", quantifier="one", inventory_status="partial", anchors=(anchor,)
    )
    assert _run(proposal, (_hit("a", anchor.quote or ""),)).status == "needs_evidence"
    subset = proposal.model_copy(update={"quantifier": "confirmed_subset"})
    assert _run(subset, (_hit("a", anchor.quote or ""),)).answer == "No confirmed matching sources."


def test_cross_namespace_keys_can_be_same_entity_but_are_not_automatically_distinct() -> None:
    anchors = (
        _text("E1", "Hotel-booking B17 TOTAL 25 EUR"),
        _text("E2", "Bank-payment P8 TOTAL 25 EUR"),
    )
    members = (
        _member(
            "booking",
            anchors[0],
            identity=IdentityKey(namespace="Hotel-booking", value="B17", anchor=anchors[0]),
        ),
        _member(
            "payment",
            anchors[1],
            identity=IdentityKey(namespace="Bank-payment", value="P8", anchor=anchors[1]),
        ),
    )
    proposal = _proposal(
        operation="count",
        labels=("E1", "E2"),
        members=members,
    )
    hits = tuple(_hit(str(i), anchor.quote or "") for i, anchor in enumerate(anchors))
    result = _run(proposal, hits)
    assert result.status == "needs_evidence"
    assert result.missing == ("identity_unknown",)
    same = IdentityRelation(left="booking", right="payment", relation="same", anchors=anchors)
    result = _run(proposal.model_copy(update={"identities": (same,)}), hits)
    assert (result.status, result.answer) == ("executed", "1")


def test_same_namespace_different_keys_conflict_with_explicit_same() -> None:
    anchors = (_text("E1", "Vale A TOTAL 10 EUR"), _text("E2", "Vale B TOTAL 10 EUR"))
    members = tuple(
        _member(
            identifier,
            anchor,
            identity=IdentityKey(namespace="Vale", value=identifier, anchor=anchor),
        )
        for identifier, anchor in zip(("A", "B"), anchors, strict=True)
    )
    proposal = _proposal(
        operation="count",
        labels=("E1", "E2"),
        members=members,
        identities=(IdentityRelation(left="A", right="B", relation="same", anchors=anchors),),
    )
    with pytest.raises(InvalidEvidenceOperation, match="identity key conflicts"):
        _run(proposal, tuple(_hit(str(i), anchor.quote or "") for i, anchor in enumerate(anchors)))


def test_difference_uses_explicit_contribution_order_not_member_order() -> None:
    anchors = (_text("E1", "TOTAL 10 EUR"), _text("E2", "TOTAL 3 EUR"))
    proposal = _proposal(
        operation="difference",
        labels=("E1", "E2"),
        members=(_member("a", anchors[0]), _member("b", anchors[1])),
        values=(_fact("a", "a", anchors[0], "10"), _fact("b", "b", anchors[1], "3")),
        contributions=(_contribution("b", "b"), _contribution("a", "a")),
    )
    result = _run(
        proposal, tuple(_hit(str(i), anchor.quote or "") for i, anchor in enumerate(anchors))
    )
    assert (result.status, result.answer) == ("executed", "-7 EUR")


@pytest.mark.parametrize("value", ["-2", "2.5"])
def test_quantity_must_be_explicit_nonnegative_integer(value: str) -> None:
    anchor = _text("E1", f"Pack count {value}; Unit weight 200 g")
    proposal = _proposal(
        measure="weight",
        members=(_member("a", anchor),),
        values=(
            _fact(
                "quantity",
                "a",
                _text("E1", f"Pack count {value}"),
                value,
                label="Pack count",
                role="quantity",
                unit=None,
                extent="quantity",
            ),
            _fact(
                "unit",
                "a",
                _text("E1", "Unit weight 200 g"),
                "200",
                label="Unit weight",
                role="weight",
                unit="g",
                extent="per_unit",
            ),
        ),
        contributions=(_contribution("a", "quantity", "unit", basis="quantity_per_unit"),),
    )
    with pytest.raises(InvalidEvidenceOperation, match="integer"):
        _run(proposal, (_hit("a", anchor.quote or ""),))


def test_product_cannot_take_quantity_from_a_different_member() -> None:
    anchors = (_text("E1", "Pack count 3"), _text("E2", "Unit weight 200 g"))
    proposal = _proposal(
        measure="weight",
        labels=("E1", "E2"),
        members=(_member("a", anchors[0]), _member("b", anchors[1])),
        values=(
            _fact(
                "quantity",
                "a",
                anchors[0],
                "3",
                label="Pack count",
                role="quantity",
                unit=None,
                extent="quantity",
            ),
            _fact(
                "unit",
                "b",
                anchors[1],
                "200",
                label="Unit weight",
                role="weight",
                unit="g",
                extent="per_unit",
            ),
        ),
        contributions=(_contribution("a", "quantity", "unit", basis="quantity_per_unit"),),
    )
    with pytest.raises(InvalidEvidenceOperation, match="different members"):
        _run(proposal, tuple(_hit(str(i), anchor.quote or "") for i, anchor in enumerate(anchors)))


def test_partition_cannot_reuse_one_numeric_source_occurrence_with_different_keys() -> None:
    anchor = _text("E1", "Deposit 10 EUR")
    proposal = _proposal(
        members=(_member("a", anchor),),
        values=(
            _fact("one", "a", anchor, "10", label="Deposit", extent="part", part_key="Deposit"),
            _fact("two", "a", anchor, "10", label="Deposit", extent="part", part_key="EUR"),
        ),
        contributions=(_contribution("a", "one", "two", basis="partition"),),
    )
    with pytest.raises(InvalidEvidenceOperation, match="numeric occurrence"):
        _run(proposal, (_hit("a", anchor.quote or ""),))


def test_partial_numeric_quote_cannot_remove_source_sign() -> None:
    anchor = _text("E1", "3 EUR")
    proposal = _proposal(
        members=(_member("a", _text("E1", "TOTAL -3 EUR")),),
        values=(_fact("a", "a", anchor, "3", label="EUR"),),
        contributions=(_contribution("a", "a"),),
    )
    with pytest.raises(InvalidEvidenceOperation, match="part of its source number"):
        _run(proposal, (_hit("a", "TOTAL -3 EUR"),))


@pytest.mark.parametrize(
    ("content", "value", "unit"),
    [("TOTAL -3", "3", "-"), ("TOTAL 3 EUR_suffix", "3", "EUR")],
)
def test_a_sign_or_partial_unit_cannot_be_reclassified_as_literal_unit(
    content: str, value: str, unit: str
) -> None:
    anchor = _text("E1", content)
    proposal = _proposal(
        members=(_member("a", anchor),),
        values=(_fact("a", "a", anchor, value, unit=unit),),
        contributions=(_contribution("a", "a"),),
    )
    with pytest.raises(InvalidEvidenceOperation):
        _run(proposal, (_hit("a", content),))


def test_one_total_literal_cannot_be_consumed_by_two_distinct_entities() -> None:
    anchor = _text("E1", "Item A and Item B; TOTAL 10 EUR")
    proposal = _proposal(
        operation="sum",
        members=(_member("a", anchor), _member("b", anchor)),
        identities=(
            IdentityRelation(left="a", right="b", relation="different", anchors=(anchor,)),
        ),
        values=(_fact("a", "a", anchor, "10"), _fact("b", "b", anchor, "10")),
        contributions=(_contribution("a", "a"), _contribution("b", "b")),
    )
    with pytest.raises(InvalidEvidenceOperation, match="across different members"):
        _run(proposal, (_hit("a", anchor.quote or ""),))
