"""Context proposals cannot edit evidence, hide competing values, or overspend a budget."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from sensemem import (
    AssetRef,
    ContentPart,
    ContextBudget,
    ContextSelection,
    ContextSelectionRequest,
    EvidenceBasis,
    MemoryCategory,
    Modality,
    RecordKind,
    RecordSemantics,
    SearchHit,
    ValidationError,
)
from sensemem.context import EvidenceClosure, compile_context

REFERENCE = datetime(2026, 11, 16, tzinfo=timezone.utc)


class ProposedSelection:
    def __init__(self, proposed: object) -> None:
        self.proposed = proposed
        self.requests: list[ContextSelectionRequest] = []

    def select(self, request: ContextSelectionRequest) -> ContextSelection:
        self.requests.append(request)
        return ContextSelection(cast(tuple[str, ...], self.proposed))

    def close(self) -> None:
        pass


def _hit(memory_id: str, *, context: RecordSemantics | None = None) -> SearchHit:
    return SearchHit(
        id=memory_id,
        content=f"evidence {memory_id}",
        score=0.8,
        created_at=REFERENCE,
        context=context,
    )


@pytest.mark.parametrize(
    ("proposed", "reason"),
    [
        (("invented",), "unknown candidate anchor ID"),
        (("source", "source"), "duplicate candidate anchor IDs"),
        (["source"], "tuple of candidate anchor IDs"),
        ((1,), "tuple of candidate anchor IDs"),
    ],
)
def test_invalid_selector_output_is_refused(proposed: object, reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        compile_context(
            "goal",
            (_hit("source"),),
            budget=ContextBudget(),
            reference_at=REFERENCE,
            context_selector=ProposedSelection(proposed),
        )


def test_custom_order_replaces_default_section_diversity() -> None:
    candidates = (
        *(_hit(f"fact-{index}") for index in range(7)),
        replace(_hit("episode"), memory_type=MemoryCategory.EPISODIC),
        replace(_hit("procedure"), memory_type=MemoryCategory.PROCEDURAL),
    )
    default = compile_context(
        "goal", candidates, budget=ContextBudget(max_items=6), reference_at=REFERENCE
    )
    custom = compile_context(
        "goal",
        candidates,
        budget=ContextBudget(max_items=6),
        reference_at=REFERENCE,
        context_selector=ProposedSelection(tuple(hit.id for hit in candidates)),
    )
    assert default.episodes and default.procedures
    assert not custom.episodes and not custom.procedures
    assert {hit.id for hit in custom.hits} == {f"fact-{index}" for index in range(6)}


def test_kernel_admits_only_complete_shared_closures_within_the_item_budget() -> None:
    source = _hit("source")
    first, second = (_hit(memory_id) for memory_id in ("first", "second"))
    closures = (
        EvidenceClosure(first, (first, source)),
        EvidenceClosure(second, (second, source)),
    )
    for max_items, expected in (
        (1, set()),
        (2, {"second", "source"}),
        (3, {"first", "second", "source"}),
    ):
        selector = ProposedSelection(("second", "first"))
        bundle = compile_context(
            "goal",
            (first, second),
            budget=ContextBudget(max_items=max_items),
            reference_at=REFERENCE,
            closures=closures,
            context_selector=selector,
        )
        assert {hit.id for hit in bundle.hits} == expected
        assert bundle.chars <= bundle.budget.max_chars
        assert selector.requests[-1].closed_evidence
        assert selector.requests[-1].candidates[0].required_ids == ("first", "source")


def test_selecting_only_one_competing_value_still_admits_and_reports_its_peer() -> None:
    first, second = (
        _hit(
            memory_id,
            context=RecordSemantics(
                kind=RecordKind.STATE,
                basis=EvidenceBasis.USER_STATEMENT,
                confidence=0.9,
                valid_from=None,
                valid_until=None,
                recorded_at=REFERENCE,
                lineage_id="location",
                subject="user",
                predicate="location",
                value=value,
            ),
        )
        for memory_id, value in (("first", "Paris"), ("second", "Berlin"))
    )
    selector = ProposedSelection(("first",))
    bundle = compile_context(
        "Where is the user?",
        (first, second),
        budget=ContextBudget(max_items=2),
        reference_at=REFERENCE,
        closures=(EvidenceClosure(first, (first,)), EvidenceClosure(second, (second,))),
        context_selector=selector,
    )
    assert {hit.id for hit in bundle.hits} == {"first", "second"}
    assert len(bundle.conflicts) == 1
    assert set(selector.requests[-1].candidates[0].required_ids) == {"first", "second"}
    assert bundle.omitted == 0


def test_selector_cannot_reintroduce_candidates_removed_by_budget_filters() -> None:
    fresh = _hit("fresh")
    stale = replace(_hit("stale"), created_at=REFERENCE - timedelta(days=30))
    selector = ProposedSelection(("stale",))
    with pytest.raises(ValidationError, match="unknown candidate anchor ID"):
        compile_context(
            "goal",
            (fresh, stale),
            budget=ContextBudget(freshness=timedelta(days=1)),
            reference_at=REFERENCE,
            context_selector=selector,
        )
    assert tuple(candidate.hit.id for candidate in selector.requests[-1].candidates) == ("fresh",)


def test_selector_cannot_deliver_an_oversized_assertion_by_itself() -> None:
    source = replace(_hit("source"), content="source " * 1000)
    assertion = _hit("assertion")
    bundle = compile_context(
        "goal",
        (assertion,),
        budget=ContextBudget(max_chars=1000),
        reference_at=REFERENCE,
        closures=(EvidenceClosure(assertion, (assertion, source)),),
        context_selector=ProposedSelection(("assertion",)),
    )
    assert not bundle.hits
    assert bundle.chars <= 1000


def test_default_selector_prices_a_ranked_text_projection_before_a_lower_full_record() -> None:
    source = replace(
        _hit("media"),
        score=0.9,
        modality=Modality.VIDEO,
        assets=(
            AssetRef(
                "clip",
                modality=Modality.VIDEO,
                media_type="video/mp4",
                size_bytes=0,
                sha256="0" * 64,
                path=Path("clip.mp4"),
            ),
        ),
        parts=(
            ContentPart("asset", "clip"),
            ContentPart("model_interpretation", "A model reports a valve inspection."),
        ),
    )
    hits = (source, _hit("lower-ranked-text"))
    bundle = compile_context(
        "valve inspection",
        hits,
        budget=ContextBudget(max_chars=1600, max_items=1),
        reference_at=REFERENCE,
    )
    assert bundle.hits == ()
    assert tuple(projection.source_memory_id for projection in bundle.projections) == (source.id,)
    assert bundle.chars <= 1600

    selector = ProposedSelection(())
    excluded = compile_context(
        "valve inspection",
        hits,
        budget=ContextBudget(max_chars=1600, max_items=1),
        reference_at=REFERENCE,
        context_selector=selector,
    )
    assert excluded.hits == () and excluded.projections == ()
    request = selector.requests[0]
    assert request.projection_section_chars > 0
    assert request.candidates[0].projection_chars is not None
    assert request.candidates[0].projection_chars < request.candidates[0].chars
