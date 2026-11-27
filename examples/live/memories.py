"""Present the native memory record, source graph, and reversible operation log."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from typing import Literal, Self
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from sensemem import (
    AsyncMemory,
    EvidenceBasis,
    MemoryIntent,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryRecord,
    ObservationContext,
    PendingCapture,
    RecordKind,
    SearchHit,
)

from .dialogue import DialogueAssistant, DialogueUser


class AssetCard(BaseModel):
    id: str
    modality: str | None
    media_type: str | None
    size_bytes: int | None
    sha256: str | None
    name: str | None
    url: str


class PartCard(BaseModel):
    kind: str
    value: str


class ProcessingError(BaseModel):
    code: Literal["processing_failed"] = "processing_failed"
    attempts: int
    awaiting: str


class EventSummary(BaseModel):
    """Read the native metadata boundary; prose markers are never a presentation schema."""

    model_config = ConfigDict(extra="ignore")
    summary: str = Field(strict=True, min_length=1, max_length=16_384)
    summary_coverage: Literal["unverified"]
    event_id: str = Field(strict=True, min_length=1)
    part_index: int = Field(strict=True, ge=0)
    part_count: int = Field(strict=True, ge=1)
    occurred_at: AwareDatetime
    occurred_end: AwareDatetime

    @model_validator(mode="after")
    def valid_event(self) -> Self:
        if not self.summary.strip() or self.part_index >= self.part_count:
            raise ValueError("event summary or part position is invalid")
        if self.occurred_end <= self.occurred_at:
            raise ValueError("event summary interval is invalid")
        return self


class DialogueCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user: DialogueUser | None = None
    assistant: DialogueAssistant
    completion: Literal["generated", "text_only", "completed", "interrupted", "stopped"]


class MemoryCard(BaseModel):
    id: str
    content: str
    created_at: datetime
    occurred_at: datetime | None
    occurred_end: datetime | None
    pending: bool
    awaiting: str | None
    error: ProcessingError | None
    assets: list[AssetCard]
    parts: list[PartCard]
    dialogue: DialogueCard | None
    modality: str
    memory_type: str
    kind: str
    basis: str
    confidence: float | None
    visible: bool
    retired_at: datetime | None
    forgotten_at: datetime | None
    source_id: str | None
    lineage_id: str | None
    supersedes_id: str | None
    evidence_ids: tuple[str, ...]
    subject: str | None
    predicate: str | None
    value: str | None
    model_id: str | None
    recipe: str | None
    can_correct: bool
    can_confirm: bool
    needs_corroboration: bool
    event_summary: str | None
    event_summary_scope: Literal["local", "event"] | None
    event_summary_verified: Literal[False] = False
    event_summary_shared: bool
    event_summary_event_id: str | None
    event_summary_part_index: int | None
    event_summary_part_count: int | None
    event_summary_occurred_at: datetime | None
    event_summary_occurred_end: datetime | None


class OperationCard(BaseModel):
    id: int
    intent: str
    applied_at: datetime
    rolled_back_at: datetime | None
    evidence_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    created_ids: tuple[str, ...]
    changed_ids: tuple[str, ...]
    forgotten_ids: tuple[str, ...]
    rationale: str | None
    model_id: str | None
    recipe: str | None
    superseded: tuple[tuple[str, int], ...]
    trigger: str
    can_rollback: bool


class MemoryDetail(BaseModel):
    item: MemoryCard
    evidence: list[MemoryCard]
    versions: list[MemoryCard]
    operations: list[OperationCard]


class CorrectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(strict=True, min_length=1, max_length=4000)

    @field_validator("content")
    @classmethod
    def meaningful_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a correction must contain text")
        return value.strip()


class MemoryChange(BaseModel):
    item: MemoryCard
    operation: OperationCard


class ForgetResult(BaseModel):
    operation: OperationCard | None


class RollbackResult(BaseModel):
    restored: bool


class OperationList(BaseModel):
    items: list[OperationCard]


def operation_card(record: MemoryOperationRecord) -> OperationCard:
    operation = record.operation
    return OperationCard(
        id=record.operation_id,
        intent=operation.intent.value,
        applied_at=record.applied_at,
        rolled_back_at=record.rolled_back_at,
        evidence_ids=operation.evidence_ids,
        target_ids=operation.target_ids,
        created_ids=record.created_ids,
        changed_ids=record.changed_ids,
        forgotten_ids=record.forgotten_ids,
        rationale=operation.rationale,
        model_id=record.model_id,
        recipe=record.recipe,
        superseded=record.superseded,
        trigger=record.trigger.value,
        can_rollback=record.rolled_back_at is None
        and operation.identity is None
        and operation.intent
        in {
            MemoryIntent.CONSOLIDATE,
            MemoryIntent.CORRECT,
            MemoryIntent.REINFORCE,
            MemoryIntent.FORGET,
        },
    )


async def _cards(
    memory: AsyncMemory, records: Sequence[MemoryRecord | SearchHit]
) -> list[MemoryCard]:
    waiting: dict[str, PendingCapture] = {}
    for offset in range(0, len(records), 100):
        pending = await memory.pending_captures(
            limit=100, memory_ids=tuple(record.id for record in records[offset : offset + 100])
        )
        waiting.update((item.memory_id, item) for item in pending)
    cards: list[MemoryCard] = []
    for record in records:
        context = record.context
        item = waiting.get(record.id)
        derived = context is not None and context.kind is not RecordKind.OBSERVATION
        summary = None
        if not derived and (metadata := record.metadata.get("event_segmentation")) is not None:
            # Application metadata is unknown input; keep the original record available.
            with suppress(PydanticValidationError):
                summary = EventSummary.model_validate(metadata)
        dialogue = None
        if not derived and (metadata := record.metadata.get("dialogue")) is not None:
            with suppress(PydanticValidationError):
                dialogue = DialogueCard.model_validate(metadata)
        editable = (
            derived
            and context is not None
            and context.retired_at is None
            and record.forgotten_at is None
            and not (
                context.identity_id is not None
                and (context.kind is RecordKind.ENTITY or context.predicate == "consent")
            )
        )
        cards.append(
            MemoryCard(
                id=record.id,
                content=record.content,
                created_at=record.created_at,
                occurred_at=record.occurred_at,
                occurred_end=record.occurred_end,
                pending=item is not None and item.awaiting == "enrichment",
                awaiting=None if item is None else item.awaiting,
                error=None
                if item is None or item.last_error is None
                else ProcessingError(attempts=item.attempts, awaiting=item.awaiting),
                assets=[
                    AssetCard(
                        id=asset.id,
                        modality=None if asset.modality is None else asset.modality.value,
                        media_type=asset.media_type,
                        size_bytes=asset.size_bytes,
                        sha256=asset.sha256,
                        name=asset.name,
                        url=f"/api/memories/{quote(record.id, safe='')}/assets/"
                        f"{quote(asset.id, safe='')}",
                    )
                    for asset in record.assets
                ],
                parts=[PartCard(kind=part.kind, value=part.value) for part in record.parts],
                dialogue=dialogue,
                modality=record.modality.value,
                memory_type=record.memory_type.value,
                kind=RecordKind.OBSERVATION.value if context is None else context.kind.value,
                basis=EvidenceBasis.OBSERVATION.value if context is None else context.basis.value,
                confidence=None if context is None else context.confidence,
                visible=context is None or context.visible,
                retired_at=None if context is None else context.retired_at,
                forgotten_at=record.forgotten_at,
                source_id=None if context is None else context.source_id,
                lineage_id=None if context is None else context.lineage_id,
                supersedes_id=None if context is None else context.supersedes_id,
                evidence_ids=() if context is None else context.evidence_ids,
                subject=None if context is None else context.subject,
                predicate=None if context is None else context.predicate,
                value=None if context is None else context.value,
                model_id=None if context is None else context.model_id,
                recipe=None if context is None else context.recipe,
                can_correct=editable,
                can_confirm=editable,
                needs_corroboration=context is not None
                and context.kind is RecordKind.TRAIT
                and context.basis is EvidenceBasis.MODEL_INFERENCE
                and not context.visible
                and context.retired_at is None
                and record.forgotten_at is None,
                event_summary=None if summary is None else summary.summary,
                event_summary_scope=None
                if summary is None
                else "event"
                if summary.part_count > 1
                else "local",
                event_summary_shared=summary is not None and summary.part_count > 1,
                event_summary_event_id=None if summary is None else summary.event_id,
                event_summary_part_index=None if summary is None else summary.part_index,
                event_summary_part_count=None if summary is None else summary.part_count,
                event_summary_occurred_at=None if summary is None else summary.occurred_at,
                event_summary_occurred_end=None if summary is None else summary.occurred_end,
            )
        )
    return cards


async def memory_cards(
    memory: AsyncMemory, records: Sequence[MemoryRecord | SearchHit]
) -> list[dict[str, object]]:
    """Keep list and recall cards faithful to the same native state as the detail view."""
    return [card.model_dump(mode="json") for card in await _cards(memory, records)]


async def memory_detail(memory: AsyncMemory, memory_id: str) -> MemoryDetail:
    selected = await memory.get(memory_id)
    lineage = None if selected.context is None else selected.context.lineage_id
    versions = [selected]
    if lineage is not None:
        versions = []
        cursor: str | None = None
        while True:
            page = await memory.list(limit=100, cursor=cursor)
            versions.extend(
                record
                for record in page.items
                if record.context is not None and record.context.lineage_id == lineage
            )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
    versions.sort(key=lambda record: (record.created_at, record.id))
    exported = await memory.export(memory_ids=tuple(record.id for record in versions))
    operations = {row.operation_id: row for row in exported.operations}
    # A correction's counterevidence belongs to its operation, not the withdrawn claim.
    # Walk both native links so an older interpretation still opens the original testimony.
    seen = {record.id for record in versions}
    frontier = {
        source
        for record in versions
        if record.context is not None
        for source in record.context.evidence_ids
    } | {source for row in exported.operations for source in row.operation.evidence_ids}
    evidence: list[MemoryRecord] = []
    while frontier - seen:
        ids = tuple(sorted(frontier - seen))
        seen.update(ids)
        exported_sources = await memory.export(memory_ids=ids)
        sources = exported_sources.records
        operations.update((row.operation_id, row) for row in exported_sources.operations)
        evidence.extend(sources)
        frontier = {
            source
            for record in sources
            if record.context is not None
            for source in record.context.evidence_ids
        } | {source for row in exported_sources.operations for source in row.operation.evidence_ids}
    evidence.sort(key=lambda record: (record.created_at, record.id))
    return MemoryDetail(
        item=(await _cards(memory, (selected,)))[0],
        evidence=await _cards(memory, evidence),
        versions=await _cards(memory, versions),
        operations=[operation_card(operations[key]) for key in sorted(operations)],
    )


def memory_router(
    open_memory: Callable[[], Awaitable[AsyncMemory]], on_change: Callable[[], object]
) -> APIRouter:
    """Mount local demo controls; every durable effect is a public SDK operation."""
    router = APIRouter()

    @router.get("/api/memories/{memory_id}")
    async def detail(memory_id: str) -> MemoryDetail:
        return await memory_detail(await open_memory(), memory_id)

    @router.get("/api/operations")
    async def operations() -> OperationList:
        return OperationList(
            items=[operation_card(row) for row in await (await open_memory()).operations()]
        )

    async def change(memory_id: str, content: str | None) -> MemoryChange:
        memory = await open_memory()
        target = await memory.get(memory_id)
        card = (await _cards(memory, (target,)))[0]
        if not card.can_correct:
            raise HTTPException(409, "This memory is not a current, editable interpretation.")
        confirmation = content is None
        source = await memory.capture(
            f"User confirmed this memory is still accurate\n{target.content}"
            if content is None
            else content,
            occurred_at=datetime.now(timezone.utc),
            context=ObservationContext(
                basis=EvidenceBasis.RESPONSE_FEEDBACK
                if confirmation
                else EvidenceBasis.USER_STATEMENT,
                source_id=f"demo-{'confirmation' if confirmation else 'correction'}-{uuid4().hex}",
            ),
        )
        try:
            operation = await memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE if confirmation else MemoryIntent.CORRECT,
                    target_ids=(memory_id,),
                    evidence_ids=(source.id,),
                    rationale="User confirmed this memory in the demo."
                    if confirmation
                    else "User submitted a correction in the demo. Original experience retained.",
                )
            )
        finally:
            # A captured statement remains real evidence even if the chosen target moved.
            on_change()
        return MemoryChange(
            item=(await _cards(memory, (source,)))[0], operation=operation_card(operation)
        )

    @router.post("/api/memories/{memory_id}/corrections")
    async def correct(memory_id: str, supplied: CorrectionInput) -> MemoryChange:
        return await change(memory_id, supplied.content)

    @router.post("/api/memories/{memory_id}/confirmations")
    async def confirm(memory_id: str) -> MemoryChange:
        return await change(memory_id, None)

    @router.post("/api/memories/{memory_id}/forget")
    async def forget(memory_id: str) -> ForgetResult:
        operation = await (await open_memory()).forget((memory_id,))
        on_change()
        return ForgetResult(operation=None if operation is None else operation_card(operation))

    @router.post("/api/operations/{operation_id}/rollback")
    async def rollback(operation_id: int) -> RollbackResult:
        restored = await (await open_memory()).rollback(operation_id)
        on_change()
        return RollbackResult(restored=restored)

    return router
