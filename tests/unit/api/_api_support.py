"""Shared values and result fixtures for REST and MCP adapter tests."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from sensemem.content import AssetRef, ContentInput
from sensemem.context_bundle import ContextBundle
from sensemem.context_values import (
    AffectCue,
    ContextBudget,
    ContextConflict,
    ContextSupplement,
    ContextUnknown,
    ContextUnknownKind,
    NamedActor,
    ProvisionalActor,
)
from sensemem.identity import (
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    PerceptionKind,
    PerceptionResult,
    SpeakerSegment,
)
from sensemem.records import (
    AnswerResult,
    MemoryRecord,
    RetrievalCandidateTrace,
    RetrievalRejection,
    RetrievalTrace,
    SearchHit,
    SearchResult,
)
from sensemem.semantics import RecordSemantics
from sensemem.spatial import RetrievalScope
from sensemem.types import AnswerPolicy, EvidenceBasis, MemoryCategory, Modality, RecordKind

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
OCCURRED_FROM = datetime(2026, 8, 27, tzinfo=timezone.utc)
OCCURRED_UNTIL = datetime(2026, 8, 28, tzinfo=timezone.utc)
ENVELOPE_FIELDS = {
    "code",
    "reason",
    "retryable",
    "stage",
    "subject",
    "message",
    "trace_id",
    "issues",
}
ASSET = AssetRef(
    id="asset_image",
    modality=Modality.IMAGE,
    media_type="image/png",
    size_bytes=3,
    sha256="a" * 64,
    name="frame.png",
    path=Path("/private/sensemem/assets/frame.png"),
)
PROVISIONAL = ProvisionalActor(identity_id="identity_2", memory_ids=("memory_2",))
AFFECT_CONTEXT = RecordSemantics(
    kind=RecordKind.AFFECT,
    basis=EvidenceBasis.MODEL_INFERENCE,
    confidence=0.72,
    valid_from=None,
    valid_until=None,
    recorded_at=NOW,
    evidence_ids=("memory_2",),
    cue_modality=Modality.AUDIO,
    valence=-0.2,
    arousal=0.8,
)
NAMED = NamedActor(
    identity_id="identity_3",
    name="Alice",
    memory_ids=("memory_2",),
    naming_assertion_id="memory_3",
)
CONFLICT = ContextConflict(
    lineage_id="lineage_1",
    subject="ana",
    predicate="location",
    values=("berlin", "paris"),
    memory_ids=("memory_1", "memory_2"),
)
UNKNOWN = ContextUnknown(
    kind=ContextUnknownKind.BUDGET_EXCLUDED,
    detail="3 candidates did not fit 24 items and 16000 chars",
)
EVIDENCE_UNAVAILABLE = ContextUnknown(
    kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
    detail="a required supporting memory was unavailable",
)
SEGMENT = SpeakerSegment(
    asset_id="b" * 64,
    start_ms=0,
    end_ms=1500,
    text="Where is the toolbox?",
    speaker_id="speaker_1",
    speaker_name="Ann",
    identity_score=0.82,
)
FACE = FaceObservation(
    asset_id="c" * 64,
    bounding_box=(0.1, 0.2, 0.3, 0.4),
    identity_id="identity_1",
    identity_name="Ann",
    identity_score=0.77,
    observed_at_ms=250,
)
ERASURE = IdentityErasure(
    identity_id="identity_1",
    alias_ids=("identity_4",),
    face_exemplars=3,
    voice_exemplars=2,
    face_observations=7,
    speech_segments=11,
)
PROFILE = IdentityProfile(
    identity_id="identity_1",
    name="Ann",
    relationship="daughter",
    confirmed=True,
    evidence_ids=("memory_1",),
)


class MemoryStub:
    """Record SDK calls shared by the REST and MCP transport fixtures."""

    def __init__(self, failure: Exception | None = None, *, restored: str = "identity_2") -> None:
        self.calls: list[tuple[object, ...]] = []
        self.close_count = 0
        self.failure = failure
        self.profile: IdentityProfile | None = PROFILE
        self.restored: str | None = restored

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
        explain: bool = False,
    ) -> SearchResult:
        self._fail()
        self.calls.append(
            (
                "search",
                query,
                limit,
                memory_type,
                reference_at,
                occurred_from,
                occurred_until,
            )
        )
        return SearchResult(hits=() if explain else (_hit(),), trace=_trace() if explain else None)

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
        supplements: Sequence[ContextSupplement] = (),
    ) -> ContextBundle:
        self._fail()
        call = ("compile", goal, budget, reference_at, scope, allow_partial_sources)
        self.calls.append((*call, supplements) if supplements else call)
        return replace(
            _bundle(budget or ContextBudget(), reference_at or NOW), supplements=tuple(supplements)
        )

    def reinforce(self, memory_ids: Sequence[str]) -> int:
        self._fail()
        self.calls.append(("reinforce", tuple(memory_ids)))
        return len(tuple(dict.fromkeys(memory_ids)))

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        answer_policy: AnswerPolicy = "strict",
    ) -> AnswerResult:
        self._fail()
        self.calls.append(("ask", question, limit, memory_type, reference_at, answer_policy))
        return AnswerResult(answer="The toolbox is blue.", hits=(_hit(),))

    def perceive(
        self, memory_id: str, *, kinds: frozenset[PerceptionKind] | None = None
    ) -> PerceptionResult:
        self._fail()
        self.calls.append(("perceive", memory_id, kinds))
        return PerceptionResult(speech=(SEGMENT,), faces=(FACE,))

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        self._fail()
        self.calls.append(("register_identity", identity_id, name, relationship))

    def identity(self, identity_id: str) -> IdentityProfile | None:
        self._fail()
        self.calls.append(("identity", identity_id))
        return self.profile

    def unlink_identity(self, alias_id: str) -> str | None:
        self._fail()
        self.calls.append(("unlink_identity", alias_id))
        return self.restored

    def forget_identity(self, identity_id: str) -> IdentityErasure:
        self._fail()
        self.calls.append(("forget_identity", identity_id))
        return ERASURE

    def close(self) -> None:
        self.close_count += 1

    def _fail(self) -> None:
        if self.failure is not None:
            raise self.failure


def _record(
    memory_id: str = "memory_1",
    content: str = "The toolbox is blue.",
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    assets: tuple[AssetRef, ...] = (),
    modality: Modality = Modality.TEXT,
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content=content,
        modality=modality,
        memory_type=memory_type,
        assets=assets,
        created_at=NOW,
        occurred_at=occurred_at,
        occurred_end=occurred_end,
        metadata=metadata or {},
    )


def _bundle(budget: ContextBudget, reference_at: datetime) -> ContextBundle:
    """Two populated sections, empty ones, one conflict, and one explicit unknown."""
    return ContextBundle(
        goal="What should I bring?",
        reference_at=reference_at,
        budget=budget,
        actors=(PROVISIONAL,),
        relationships=(),
        scene=(),
        episodes=(_media_hit(),),
        facts=(_hit(),),
        procedures=(),
        affect=(_affect_cue(),),
        traits=(),
        conflicts=(CONFLICT,),
        unknowns=(UNKNOWN, EVIDENCE_UNAVAILABLE),
        occurred_from=OCCURRED_FROM,
        occurred_until=OCCURRED_UNTIL,
        frames=("home/map",),
        places=("kitchen",),
        omitted=3,
        chars=42,
        elapsed_ms=7,
        deadline_exceeded=False,
    )


def _affect_cue() -> AffectCue:
    """One affect entry, whose evidence hop must reach the wire beside the hit fields."""
    return AffectCue(
        id="memory_3",
        content="The user sounded tense.",
        score=0.6,
        memory_type=MemoryCategory.EPISODIC,
        created_at=NOW,
        context=AFFECT_CONTEXT,
        event_ids=("memory_9",),
    )


def _media_hit() -> SearchHit:
    """A hit carrying a stored asset, whose local path must never be serialized."""
    return SearchHit(
        id="memory_2",
        content="The toolbox is blue.",
        score=0.7,
        modality=Modality.IMAGE,
        memory_type=MemoryCategory.EPISODIC,
        assets=(ASSET,),
        created_at=NOW,
    )


def _trace() -> RetrievalTrace:
    return RetrievalTrace(
        candidates=(
            RetrievalCandidateTrace(
                memory_id="memory_1",
                index_ids=("index_1",),
                dense_relevance=0.42,
                gate_relevance=0.31,
                final_score=0.4,
                rejected_by=RetrievalRejection.MINIMUM_RELEVANCE,
            ),
        ),
        candidate_limit=50,
        exhaustive=True,
        ambiguous=True,
    )


def _hit() -> SearchHit:
    return SearchHit(
        id="memory_1",
        content="The toolbox is blue.",
        score=0.9,
        modality=Modality.TEXT,
        created_at=NOW,
    )
