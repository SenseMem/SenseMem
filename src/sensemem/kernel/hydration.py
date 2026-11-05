"""Hydrating stored rows into the public record and hit values."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from time import perf_counter

from sensemem.context import EvidenceClosure, _rejection
from sensemem.control import load_operation
from sensemem.exceptions import StorageError
from sensemem.infrastructure.local.store import StoredAsset, StoredMemory, StoredOperation
from sensemem.kernel.content import PreparedContent, decode_metadata
from sensemem.kernel.runtime import Storage, translate_storage_errors
from sensemem.models.base import ModelInput
from sensemem.types import (
    AssetRef,
    ContextBudget,
    ContextUnknown,
    ContextUnknownKind,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    Modality,
    RetrievalScope,
    SearchHit,
)


def operation_record(logged: StoredOperation) -> MemoryOperationRecord:
    try:
        trigger = MemoryTrigger(logged.trigger)
    except ValueError:
        raise StorageError("a logged memory operation has an unknown trigger") from None
    return MemoryOperationRecord(
        operation_id=logged.operation_id,
        operation=load_operation(logged.operation_json),
        trigger=trigger,
        applied_at=logged.applied_at,
        model_id=logged.model_id,
        recipe=logged.recipe,
        created_ids=logged.created_ids,
        changed_ids=logged.changed_ids,
        forgotten_ids=logged.forgotten_ids,
        superseded=logged.superseded,
        rolled_back_at=logged.rolled_back_at,
        outcome=None if logged.outcome is None else _memory_outcome(logged.outcome),
        outcome_note=logged.outcome_note,
    )


def _memory_outcome(value: str) -> MemoryOutcome:
    try:
        return MemoryOutcome(value)
    except ValueError:
        raise StorageError("a logged memory operation has an unknown outcome") from None


class Hydrator:
    """Hydrate public records, assets, and complete scoped evidence closures."""

    def __init__(
        self,
        *,
        storage: Storage,
    ) -> None:
        self._assets = storage.assets
        self._store = storage.store

    def memory_record(self, memory: StoredMemory) -> MemoryRecord:
        return MemoryRecord(
            id=memory.memory_id,
            content=memory.content,
            created_at=memory.created_at,
            occurred_at=memory.occurred_at,
            occurred_end=memory.occurred_end,
            metadata=decode_metadata(memory.metadata_json),
            assets=tuple(self.asset_ref(asset) for asset in memory.assets),
            modality=Modality(memory.modality),
            memory_type=MemoryType(memory.memory_type),
            context=memory.context,
            forgotten_at=memory.forgotten_at,
            place_id=memory.place_id,
        )

    def search_hit(self, memory: StoredMemory, relevance: float) -> SearchHit:
        return SearchHit(
            id=memory.memory_id,
            content=memory.content,
            score=max(0.0, min(1.0, relevance)),
            created_at=memory.created_at,
            occurred_at=memory.occurred_at,
            occurred_end=memory.occurred_end,
            metadata=decode_metadata(memory.metadata_json),
            assets=tuple(self.asset_ref(asset) for asset in memory.assets),
            modality=Modality(memory.modality),
            memory_type=MemoryType(memory.memory_type),
            context=memory.context,
            forgotten_at=memory.forgotten_at,
            place_id=memory.place_id,
        )

    def asset_ref(self, asset: StoredAsset) -> AssetRef:
        with translate_storage_errors("resolve local media"):
            path = self._assets.resolve(asset)
        return AssetRef(
            id=asset.asset_id,
            modality=Modality(asset.modality),
            media_type=asset.mime_type,
            size_bytes=asset.size_bytes,
            sha256=asset.sha256,
            name=asset.name,
            path=path,
        )

    def model_input(self, prepared: PreparedContent) -> ModelInput:
        return ModelInput(
            text=prepared.text,
            assets=tuple(self.asset_ref(asset) for asset in prepared.assets),
        )

    def evidence_closures(  # noqa: C901 - bounded scoped provenance traversal
        self,
        anchors: Sequence[SearchHit],
        *,
        budget: ContextBudget,
        reference_at: datetime,
        scope: RetrievalScope | None,
        started_at: float | None = None,
    ) -> tuple[tuple[EvidenceClosure, ...], tuple[ContextUnknown, ...]]:
        """Hydrate complete finite provenance for compilation and answer delivery."""
        started_at = perf_counter() if started_at is None else started_at
        by_id = {anchor.id: anchor for anchor in anchors}
        frontier = tuple(
            source_id
            for anchor in anchors
            if anchor.context is not None
            for source_id in anchor.context.evidence_ids
        )
        expanded: set[str] = set()
        bounded: set[str] = set()
        deadline = False
        max_nodes = max(1, len(anchors) * budget.max_items)
        while frontier:
            if (
                budget.max_latency_ms is not None
                and (perf_counter() - started_at) * 1_000 > budget.max_latency_ms
            ):
                deadline = True
                break
            current = tuple(dict.fromkeys(frontier))
            wanted = tuple(source_id for source_id in current if source_id not in by_id)
            selected: tuple[str, ...] = ()
            if wanted:
                remaining = max_nodes - len(by_id)
                if remaining <= 0:
                    bounded.update(wanted)
                    break
                selected, rejected = wanted[:remaining], wanted[remaining:]
                bounded.update(rejected)
                with translate_storage_errors("hydrate evidence sources"):
                    hydrated = self._store.records.read_memories(
                        selected,
                        valid_at=None if scope is None else scope.valid_at,
                        known_at=None if scope is None else scope.known_at,
                        near=None if scope is None else scope.near,
                        radius_m=None if scope is None else scope.radius_m,
                        place_id=None if scope is None else scope.place_id,
                        identity_id=None if scope is None else scope.identity_id,
                        active_only=True,
                    )
                for memory in hydrated:
                    by_id[memory.memory_id] = self.search_hit(memory, 0.0)
            frontier = tuple(
                child
                for source_id in current
                if source_id not in expanded
                and (source := by_id.get(source_id)) is not None
                and source.context is not None
                for child in source.context.evidence_ids
            )
            expanded.update(current)
        closures: list[EvidenceClosure] = []
        failed: dict[str, str] = {}
        max_depth = min(budget.max_items, 256)

        def traverse(  # noqa: C901 - tri-colour validation retains all hard eligibility checks
            anchor: SearchHit,
        ) -> tuple[tuple[str, ...], str | None]:
            states: dict[str, int] = {}
            ordered: list[str] = []

            def visit(memory_id: str, depth: int) -> str | None:
                if depth > max_depth:
                    return "bounded"
                if memory_id in bounded:
                    return "bounded"
                hit = by_id.get(memory_id)
                if hit is None:
                    return (
                        "unavailable before the compilation deadline"
                        if deadline
                        else "unavailable under the requested scope"
                    )
                if _rejection(hit, budget, reference_at) is not None:
                    return "excluded by the requested context bounds"
                state = states.get(memory_id, 0)
                if state == 1:
                    return "cyclic"
                if state == 2:
                    return None
                states[memory_id] = 1
                context = hit.context
                if context is not None:
                    for source_id in context.evidence_ids:
                        if (reason := visit(source_id, depth + 1)) is not None:
                            return reason
                states[memory_id] = 2
                ordered.append(memory_id)
                return None

            reason = visit(anchor.id, 0)
            return tuple(ordered), reason

        for anchor in anchors:
            ordered, reason = traverse(anchor)
            if reason is not None:
                failed[anchor.id] = reason
                continue
            # `visit` is postorder; the anchor remains query-ranked and leads its own closure.
            closures.append(
                EvidenceClosure(
                    anchor,
                    (
                        anchor,
                        *(by_id[memory_id] for memory_id in ordered if memory_id != anchor.id),
                    ),
                )
            )
        reasons: dict[str, int] = {}
        for reason in failed.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        unknowns = tuple(
            ContextUnknown(
                kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                detail=f"{count} ranked assertions required evidence {reason}",
            )
            for reason, count in sorted(reasons.items())
        )
        return tuple(closures), unknowns
