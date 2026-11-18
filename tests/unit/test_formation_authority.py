"""Model derivation cannot grant host authority on either automatic write path."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder

from sensemem import (
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryNotFoundError,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    MemoryTrigger,
    ObservationContext,
    QueryFailure,
    RecordKind,
)


class ClaimBackend:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = consolidation_model = "authority-fixture"
    formation_space = consolidation_recipe = "authority-fixture:v1"

    def __init__(self, kind: RecordKind, basis: EvidenceBasis) -> None:
        self.kind = kind
        self.basis = basis
        self.value = "patient"

    def proposal(self) -> FormationProposal:
        return FormationProposal(
            kind=self.kind,
            basis=self.basis,
            content=f"Rava is {self.value}.",
            subject="Rava",
            predicate="disposition",
            value=self.value,
            confidence=0.9,
        )

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple((self.proposal(),) for _source in inputs)

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
        query_failure: QueryFailure | None = None,
    ) -> tuple[MemoryOperation, ...]:
        return (
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=tuple(source.id for source in evidence),
                proposal=self.proposal(),
            ),
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("kind", (RecordKind.STATE, RecordKind.TRAIT))
@pytest.mark.parametrize("basis", (EvidenceBasis.USER_STATEMENT, EvidenceBasis.RESPONSE_FEEDBACK))
def test_custom_backend_cannot_grant_authority_to_automatic_subclaims(
    tmp_path: Path, kind: RecordKind, basis: EvidenceBasis
) -> None:
    backend = ClaimBackend(kind, basis)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=backend,
        consolidator=backend,
    ) as memory:
        source = memory.add(
            "Lea reports that Rava is not patient.",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )
        report = memory.consolidate(evidence_ids=(source.id,))
        assert report.operations == ()
        assert len(report.rejected) == 1
        assert [row.id for row in memory.list().items] == [source.id]
        assert source.context is not None
        assert source.context.basis is EvidenceBasis.USER_STATEMENT


@pytest.mark.parametrize("kind", (RecordKind.STATE, RecordKind.TRAIT))
def test_host_confirms_typed_claim_without_delegating_authority_to_later_inference(
    tmp_path: Path, kind: RecordKind
) -> None:
    backend = ClaimBackend(kind, EvidenceBasis.MODEL_INFERENCE)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=backend,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add(
            "Rava has been patient for years.",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )
        (inferred,) = [
            row for row in memory.list().items if row.context and row.context.kind is kind
        ]
        assert inferred.context is not None
        assert inferred.context.basis is EvidenceBasis.MODEL_INFERENCE
        assert inferred.context.visible is (kind is RecordKind.STATE)

        confirmation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=kind,
                    basis=EvidenceBasis.USER_STATEMENT,
                    content="Rava is patient.",
                    subject="Rava",
                    predicate="disposition",
                    value="patient",
                    confidence=0.9,
                ),
            )
        )
        (confirmed_id,) = confirmation.created_ids
        confirmed = memory.get(confirmed_id)
        assert confirmed.context is not None
        assert confirmed.context.basis is EvidenceBasis.USER_STATEMENT
        assert confirmed.context.visible

        backend.value = "impatient"
        memory.add("An observer reports that Rava is impatient.")
        standing = memory.get(confirmed_id)
        assert standing.context is not None
        assert standing.context.visible and standing.context.retired_at is None
        assert standing.context.valid_until is None

        assert memory.rollback(confirmation.operation_id)
        with pytest.raises(MemoryNotFoundError):
            memory.get(confirmed_id)
        assert confirmed_id not in {hit.id for hit in memory.search("Rava").hits}
