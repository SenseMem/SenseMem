"""Correction withdraws complete downstream assessments and rollback restores them."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemoryPlugins,
    MemoryRecord,
    MemorySettings,
    MemoryTrigger,
    QueryFailure,
    RecordKind,
    RetrievalScope,
    ValidationError,
)


def _open(path: Path, independent: bool) -> Memory:
    return Memory.from_plugins(
        path,
        plugins=MemoryPlugins(embedder=TinyEmbedder()),
        settings=MemorySettings(independent_evidence=independent, minimum_relevance=0),
    )


def _derive(
    memory: Memory,
    sources: tuple[str, ...],
    value: str,
    *,
    kind: RecordKind = RecordKind.RELATION,
    forget: bool = False,
) -> str:
    operation = memory.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=sources,
            target_ids=sources if forget else (),
            proposal=FormationProposal(
                kind=kind,
                content=f"Rava is {value}.",
                subject="Rava",
                predicate="disposition" if kind is RecordKind.TRAIT else "behavior",
                value=value,
                confidence=0.6,
            ),
        )
    )
    return next(iter((*operation.created_ids, *operation.changed_ids)))


@pytest.mark.parametrize("independent", [False, True])
def test_correction_hides_a_dependent_trait_and_its_downstream_claim(
    tmp_path: Path, independent: bool
) -> None:
    with _open(tmp_path, independent) as memory:
        first = memory.add("First calm wait.")
        second = memory.add("Independent calm wait.")
        source = _derive(memory, (first.id,), "calm")
        trait = _derive(memory, (source,), "patient", kind=RecordKind.TRAIT)
        _derive(memory, (second.id,), "patient", kind=RecordKind.TRAIT)
        downstream = _derive(memory, (trait,), "willing to wait")
        known_before_correction = datetime.now(timezone.utc)
        counterevidence = memory.add("The first wait was misinterpreted; Rava was impatient.")

        correction = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CORRECT,
                target_ids=(source,),
                evidence_ids=(counterevidence.id,),
            )
        )

        corrected = memory.get(source).context
        remaining = memory.get(trait).context
        unavailable = memory.get(downstream).context
        assert corrected is not None and corrected.retired_at is not None
        assert remaining is not None and not remaining.visible
        assert remaining.confidence == pytest.approx(0.6)
        assert remaining.evidence_ids == (second.id,)
        assert unavailable is not None and not unavailable.visible
        assert unavailable.confidence == 0
        assert unavailable.evidence_ids == ()
        assert {trait, downstream}.isdisjoint(
            hit.id for hit in memory.search("patient", limit=20).hits
        )
        historical = memory.search(
            "patient", limit=20, scope=RetrievalScope(known_at=known_before_correction)
        )
        assert {trait, downstream} <= {hit.id for hit in historical.hits}

        assert memory.rollback(correction.operation_id)
        restored_source = memory.get(source).context
        restored_trait = memory.get(trait).context
        restored_downstream = memory.get(downstream).context
        assert restored_source is not None and restored_source.retired_at is None
        assert restored_trait is not None and restored_trait.visible
        assert restored_trait.confidence == pytest.approx(0.84)
        assert set(restored_trait.evidence_ids) == {source, second.id}
        assert restored_downstream is not None and restored_downstream.visible
        assert restored_downstream.confidence == pytest.approx(0.6)
        assert restored_downstream.evidence_ids == (trait,)


@pytest.mark.parametrize("independent", [False, True])
def test_correction_preserves_an_alternative_and_never_shortens_a_joint_clause(
    tmp_path: Path, independent: bool
) -> None:
    with _open(tmp_path, independent) as memory:
        first = memory.add("First event.")
        second = memory.add("Second event.")
        other = memory.add("Independent event.")
        source = _derive(memory, (first.id,), "calm")
        joint = _derive(memory, (source, second.id), "quiet")
        alternative = _derive(memory, (source,), "patient")
        _derive(memory, (other.id,), "patient")
        downstream = _derive(memory, (alternative,), "willing to wait")

        correction = memory.apply(
            MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(source,))
        )
        withdrawn_joint = memory.get(joint).context
        surviving = memory.get(alternative).context
        dependent = memory.get(downstream).context
        assert withdrawn_joint is not None and not withdrawn_joint.visible
        assert withdrawn_joint.confidence == 0
        assert withdrawn_joint.evidence_ids == ()
        assert surviving is not None and surviving.visible
        assert surviving.evidence_ids == (other.id,)
        assert surviving.confidence == pytest.approx(0.6)
        assert dependent is not None and dependent.visible

        assert memory.rollback(correction.operation_id)
        restored_joint = memory.get(joint).context
        assert restored_joint is not None and restored_joint.visible
        assert set(restored_joint.evidence_ids) == {source, second.id}


@pytest.mark.parametrize("independent", [False, True])
def test_historical_state_segments_and_consolidation_retained_sources_keep_support(
    tmp_path: Path, independent: bool
) -> None:
    clock = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with _open(tmp_path, independent) as memory:
        first = memory.add("Rava was in the kitchen.", occurred_at=clock)
        second = memory.add("Rava waited calmly elsewhere.")
        state = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(first.id,),
                target_ids=(first.id,),
                proposal=FormationProposal(
                    kind=RecordKind.STATE,
                    content="Rava is in the kitchen.",
                    subject="Rava",
                    predicate="current_location",
                    value="kitchen",
                    confidence=0.6,
                    valid_from=clock,
                ),
            )
        ).created_ids[0]
        trait = _derive(memory, (state,), "patient", kind=RecordKind.TRAIT)
        _derive(memory, (second.id,), "patient", kind=RecordKind.TRAIT)
        later = memory.add("Rava entered the hall.", occurred_at=clock + timedelta(days=1))
        memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(later.id,),
                proposal=FormationProposal(
                    kind=RecordKind.STATE,
                    content="Rava is in the hall.",
                    subject="Rava",
                    predicate="current_location",
                    value="hall",
                    confidence=0.6,
                    valid_from=clock + timedelta(days=1),
                ),
            )
        )

        historical_state = memory.get(state).context
        supported_trait = memory.get(trait).context
        assert memory.get(first.id).forgotten_at is not None
        assert historical_state is not None and historical_state.retired_at is None
        assert historical_state.valid_until == clock + timedelta(days=1)
        assert supported_trait is not None and supported_trait.visible
        assert supported_trait.confidence == pytest.approx(0.84)


@pytest.mark.parametrize("independent", [False, True])
def test_host_can_correct_a_consolidation_retained_intermediate(
    tmp_path: Path, independent: bool
) -> None:
    with _open(tmp_path, independent) as memory:
        source = memory.add("The first wait was calm.")
        intermediate = _derive(memory, (source.id,), "calm", forget=True)
        final = _derive(memory, (intermediate,), "patient", forget=True)
        assert memory.get(intermediate).forgotten_at is not None

        correction = memory.apply(
            MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(intermediate,))
        )
        corrected = memory.get(intermediate).context
        dependent = memory.get(final).context
        assert corrected is not None and corrected.retired_at is not None
        assert dependent is not None and not dependent.visible
        assert final not in {hit.id for hit in memory.search("patient", limit=20).hits}

        assert memory.rollback(correction.operation_id)
        restored = memory.get(intermediate).context
        restored_dependent = memory.get(final).context
        assert restored is not None and restored.retired_at is None
        assert memory.get(intermediate).forgotten_at is not None
        assert restored_dependent is not None and restored_dependent.visible


@pytest.mark.parametrize("retained", [False, True])
def test_host_correction_refuses_forgotten_evidence_without_retained_permission(
    tmp_path: Path, retained: bool
) -> None:
    with _open(tmp_path, True) as memory:
        source = memory.add("The first wait was calm.")
        intermediate = _derive(memory, (source.id,), "calm")
        if retained:
            _derive(memory, (intermediate,), "patient", forget=True)
        assert memory.forget((intermediate,)) is not None

        with pytest.raises(ValidationError) as failure:
            memory.apply(MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(intermediate,)))
        assert failure.value.reason == "stale"
        unchanged = memory.get(intermediate).context
        assert unchanged is not None and unchanged.retired_at is None


def test_backend_cannot_correct_a_consolidation_retained_intermediate(tmp_path: Path) -> None:
    class Corrector:
        consolidation_model = "correction-support-fixture"
        consolidation_recipe = "correction-support-fixture:v1"

        operation: MemoryOperation | None = None

        def consolidate(
            self,
            evidence: Sequence[MemoryRecord],
            *,
            trigger: MemoryTrigger,
            query_failure: QueryFailure | None = None,
        ) -> tuple[MemoryOperation, ...]:
            return () if self.operation is None else (self.operation,)

        def close(self) -> None:
            pass

    corrector = Corrector()
    with Memory.from_plugins(
        tmp_path,
        plugins=MemoryPlugins(embedder=TinyEmbedder(), consolidator=corrector),
        settings=MemorySettings(independent_evidence=True, minimum_relevance=0),
    ) as memory:
        source = memory.add("The first wait was calm.")
        intermediate = _derive(memory, (source.id,), "calm", forget=True)
        _derive(memory, (intermediate,), "patient", forget=True)
        counterevidence = memory.add("The earlier assessment was wrong.")
        corrector.operation = MemoryOperation(
            intent=MemoryIntent.CORRECT,
            target_ids=(intermediate,),
            evidence_ids=(counterevidence.id,),
        )

        report = memory.consolidate(evidence_ids=(intermediate, counterevidence.id))
        assert report.operations == ()
        assert [reason for _operation, reason in report.rejected] == ["stale"]
        unchanged = memory.get(intermediate).context
        assert unchanged is not None and unchanged.retired_at is None
