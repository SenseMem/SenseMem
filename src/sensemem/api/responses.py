"""Shared transport response schemas and safe projections, independent of REST and MCP."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
)

from sensemem.capabilities import MemoryCapabilities
from sensemem.content import ContentPart
from sensemem.context_bundle import ContextBundle
from sensemem.context_values import (
    ContextProof,
    ContextProofScoring,
    ContextSupplement,
    ContextUnknownKind,
)
from sensemem.control import dump_operation
from sensemem.control_values import (
    ExportBundle,
    MemoryIntent,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryTrigger,
)
from sensemem.identity import (
    ConsentState,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    PerceptionCoverage,
    SpeakerSegment,
)
from sensemem.records import RetrievalTrace
from sensemem.semantics import RecordSemantics
from sensemem.types import AbstentionReason, MemoryCategory, Modality


class _ResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class AssetResponse(_ResponseModel):
    id: str
    modality: Modality
    media_type: str
    size_bytes: int
    sha256: str
    name: str | None = None


class MemoryResponse(_ResponseModel):
    id: str
    content: str
    modality: Modality
    memory_type: MemoryCategory
    assets: tuple[AssetResponse, ...] = ()
    parts: tuple[ContentPart, ...] = ()
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None = None
    occurred_end: AwareDatetime | None = None
    metadata: dict[str, JsonValue]
    context: RecordSemantics | None = None
    place_id: str | None = None
    forgotten_at: AwareDatetime | None = None


class SearchHitResponse(MemoryResponse):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class AffectCueResponse(SearchHitResponse):
    """One affect entry of a compiled bundle, with the evidence hop the cue hangs on.

    The observations the cue cites are its own context.evidence_ids. event_ids are the
    events formed from those same observations: co-occurrence inside one capture, never an
    attributed cause.
    """

    event_ids: tuple[str, ...] = ()


class MemoryBatchResponse(_ResponseModel):
    memories: tuple[MemoryResponse, ...]


class SearchResponse(_ResponseModel):
    hits: tuple[SearchHitResponse, ...]
    # Null unless the request asked for it, so the default response keeps the shape clients read.
    trace: RetrievalTrace | None = None


class AnswerResponse(_ResponseModel):
    answer: str
    hits: tuple[SearchHitResponse, ...]
    abstained: bool
    abstention_reason: AbstentionReason | None


class DeleteResponse(_ResponseModel):
    deleted: bool


class ReinforceResponse(_ResponseModel):
    reinforced: int


class SettleResponse(_ResponseModel):
    settled: int


class PendingCaptureResponse(_ResponseModel):
    memory_id: str
    enqueued_at: AwareDatetime
    attempts: int
    last_error: str | None = None
    awaiting: Literal["enrichment", "formation"]
    cursor: str


class PendingCapturesResponse(_ResponseModel):
    items: tuple[PendingCaptureResponse, ...]
    next_cursor: str | None = None


class PerceptionResponse(_ResponseModel):
    speech: tuple[SpeakerSegment, ...]
    faces: tuple[FaceObservation, ...]
    coverage: tuple[PerceptionCoverage, ...]


class IdentityResponse(_ResponseModel):
    identity: IdentityProfile | None


class RegisterResponse(_ResponseModel):
    registered: bool


class UnlinkResponse(_ResponseModel):
    restored_identity_id: str | None


class ForgetResponse(_ResponseModel):
    erasure: IdentityErasure


class ConsentResponse(_ResponseModel):
    consent: ConsentState | None


class IdentityClaimResponse(_ResponseModel):
    identity_id: str
    name: str
    relationship: str | None


class ConsentClaimResponse(_ResponseModel):
    identity_id: str
    state: ConsentState
    note: str | None


class IdentityChangeResponse(_ResponseModel):
    identity_id: str
    moved_ids: tuple[str, ...]


class MemoryOperationResponse(_ResponseModel):
    """One control-plane log row, flattened exactly as sensemem operations prints it.

    The operation's own fields are lifted beside the log row's because a reader wants one
    record of what was proposed and what it did, not two nested objects to join.
    """

    operation_id: int
    intent: MemoryIntent
    trigger: MemoryTrigger
    evidence_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    claim: IdentityClaimResponse | None
    consent: ConsentClaimResponse | None
    identity: IdentityChangeResponse | None
    # The canonical logged payload, exactly as sensemem operations prints it: a
    # consolidation's subject is what it proposed, so a row without this reported an intent and
    # no statement, and the replay path had to read the proposal from the SDK instead.
    proposal: dict[str, object] | None
    rationale: str | None
    model_id: str | None
    recipe: str | None
    created_ids: tuple[str, ...]
    changed_ids: tuple[str, ...]
    forgotten_ids: tuple[str, ...]
    superseded: tuple[tuple[str, int], ...]
    applied_at: datetime
    rolled_back_at: datetime | None
    outcome: MemoryOutcome | None
    outcome_note: str | None


class RecordConsentResponse(_ResponseModel):
    operation: MemoryOperationResponse | None


class ExportResponse(_ResponseModel):
    exported_at: datetime
    identity_id: str | None
    identities: tuple[IdentityProfile, ...]
    records: tuple[MemoryResponse, ...]
    operations: tuple[MemoryOperationResponse, ...]


class RetentionResponse(_ResponseModel):
    dry_run: bool
    media_memory_ids: tuple[str, ...]
    forgotten_memory_ids: tuple[str, ...]
    cascade_memory_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    capture_memory_ids: tuple[str, ...]
    deleted: int


def _operation_response(record: MemoryOperationRecord) -> MemoryOperationResponse:
    operation = record.operation
    # Read back from the canonical logged payload rather than re-derived from the value type, so
    # the row REST serves and the row the CLI prints are one document -- and apply accepts it.
    decoded: object = json.loads(dump_operation(operation))
    assert isinstance(decoded, dict)
    proposal_value = decoded.get("proposal")
    assert proposal_value is None or isinstance(proposal_value, dict)
    return MemoryOperationResponse(
        operation_id=record.operation_id,
        intent=operation.intent,
        trigger=record.trigger,
        evidence_ids=operation.evidence_ids,
        target_ids=operation.target_ids,
        claim=(
            None
            if operation.claim is None
            else IdentityClaimResponse.model_validate(operation.claim)
        ),
        consent=(
            None
            if operation.consent is None
            else ConsentClaimResponse.model_validate(operation.consent)
        ),
        identity=(
            None
            if operation.identity is None
            else IdentityChangeResponse.model_validate(operation.identity)
        ),
        proposal=proposal_value,
        rationale=operation.rationale,
        model_id=record.model_id,
        recipe=record.recipe,
        created_ids=record.created_ids,
        changed_ids=record.changed_ids,
        forgotten_ids=record.forgotten_ids,
        superseded=record.superseded,
        applied_at=record.applied_at,
        rolled_back_at=record.rolled_back_at,
        outcome=record.outcome,
        outcome_note=record.outcome_note,
    )


def export_response(bundle: ExportBundle) -> ExportResponse:
    return ExportResponse(
        exported_at=bundle.exported_at,
        identity_id=bundle.identity_id,
        identities=bundle.identities,
        records=tuple(MemoryResponse.model_validate(record) for record in bundle.records),
        operations=tuple(_operation_response(record) for record in bundle.operations),
    )


class ContextBudgetResponse(_ResponseModel):
    max_chars: int
    max_items: int
    max_media_items: int | None
    memory_types: tuple[MemoryCategory, ...] | None
    min_confidence: float
    freshness_seconds: float | None
    max_latency_ms: int | None
    selected_proofs: bool


class ContextConflictResponse(_ResponseModel):
    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]


class ProvisionalActorResponse(_ResponseModel):
    identity_id: str
    memory_ids: tuple[str, ...]


class NamedActorResponse(_ResponseModel):
    identity_id: str
    name: str
    memory_ids: tuple[str, ...]
    naming_assertion_id: str | None


class ContextUnknownResponse(_ResponseModel):
    kind: ContextUnknownKind
    detail: str


class TextSpanPieceResponse(_ResponseModel):
    role: Literal["context", "body"]
    start_codepoint: int
    end_codepoint: int
    source_text: str
    sha256: str


class TextSpanSelectorResponse(_ResponseModel):
    parent_content_sha256: str
    embedding_input_sha256: str
    recipe_version: str
    pieces: tuple[TextSpanPieceResponse, ...]


class ContextExcerptResponse(_ResponseModel):
    source_memory_id: str
    matched_index_id: str
    content: str
    selector: TextSpanSelectorResponse
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None
    occurred_end: AwareDatetime | None
    memory_type: MemoryCategory
    context: RecordSemantics | None
    place_id: str | None


class ContextProjectionResponse(_ResponseModel):
    source_memory_id: str
    parts: tuple[ContentPart, ...]
    omitted_assets: tuple[AssetResponse, ...]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None
    occurred_end: AwareDatetime | None
    memory_type: MemoryCategory
    context: RecordSemantics | None
    place_id: str | None
    coverage: Literal["text_only"]


class ContextBundleResponse(_ResponseModel):
    goal: str
    reference_at: AwareDatetime
    budget: ContextBudgetResponse
    # Beside the ranked entity hits: a person a naming assertion names, reached through
    # other evidence, and a recognized person no visible assertion names -- reported rather
    # than omitted either way.
    actors: tuple[SearchHitResponse | NamedActorResponse | ProvisionalActorResponse, ...]
    relationships: tuple[SearchHitResponse, ...]
    scene: tuple[SearchHitResponse, ...]
    episodes: tuple[SearchHitResponse, ...]
    facts: tuple[SearchHitResponse, ...]
    procedures: tuple[SearchHitResponse, ...]
    affect: tuple[AffectCueResponse, ...]
    traits: tuple[SearchHitResponse, ...]
    conflicts: tuple[ContextConflictResponse, ...]
    unknowns: tuple[ContextUnknownResponse, ...]
    occurred_from: AwareDatetime | None
    occurred_until: AwareDatetime | None
    frames: tuple[str, ...]
    places: tuple[str, ...]
    omitted: int
    chars: int
    elapsed_ms: int
    deadline_exceeded: bool
    excerpts: tuple[ContextExcerptResponse, ...] = ()
    projections: tuple[ContextProjectionResponse, ...] = ()
    proofs: tuple[ContextProof, ...] = ()
    proof_scoring: ContextProofScoring = "independent_pair"
    supplements: tuple[ContextSupplement, ...] = ()
    omitted_supplements: int = 0
    # The deterministic text of ContextBundle.render(), so a caller need not re-derive it.
    rendered: str


class PageResponse(_ResponseModel):
    items: tuple[MemoryResponse, ...]
    next_cursor: str | None = None


class EmbeddingRouteResponse(_ResponseModel):
    modalities: tuple[Modality, ...]
    max_assets_per_modality: int | None


class CapabilitiesResponse(_ResponseModel):
    """What the composition behind this process can actually do."""

    embedding: tuple[Modality, ...]
    embedding_model: str
    embedding_space: str
    embedding_dimension: int
    embedding_routes: tuple[EmbeddingRouteResponse, ...]
    generation: tuple[Modality, ...]
    generation_routes: tuple[tuple[Modality, ...], ...]
    transcription: tuple[Modality, ...]
    vision: tuple[Modality, ...]
    face: tuple[Modality, ...]
    formation: tuple[Modality, ...]
    reranking: tuple[Modality, ...]
    reranking_model: str | None
    generation_model: str | None
    transcription_space: str | None
    vision_model: str | None
    face_model: str | None
    formation_model: str | None
    consolidation_model: str | None
    speaker_recognition: bool
    # Derived from the backends above, not declared: which optional operations this composition
    # can serve *and this app routes to*. MemoryCapabilities.document() is the one place that
    # derivation happens; _ROUTED_OPERATIONS is what /healthz may promise of it.
    operations: tuple[str, ...]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    capabilities: CapabilitiesResponse


def bundle_response(bundle: ContextBundle) -> ContextBundleResponse:
    """Publish ContextBundle.document(), the same projection MCP and the CLI publish."""
    return ContextBundleResponse.model_validate(bundle.document())


def _capabilities_response(
    capabilities: MemoryCapabilities, *, served: frozenset[str]
) -> CapabilitiesResponse:
    """Serialize the one capability document MCP and the CLI publish too, minus what has no route."""
    return CapabilitiesResponse.model_validate(capabilities.document(served=served))
