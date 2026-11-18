"""Minimal model values shared by local and remote backends."""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, cast, runtime_checkable

import sensemem.perception as perception
from sensemem._value_validation import _require_aware, _require_interval
from sensemem.capabilities import EmbeddingRoute
from sensemem.content import AssetRef, ContentPart
from sensemem.control_values import MemoryOperation, MemoryTrigger, QueryFailure
from sensemem.events import EventSegment
from sensemem.exceptions import ValidationError
from sensemem.perception import VisualDescription
from sensemem.records import AnswerResult, MemoryRecord, SearchHit, VisualRetrievalMatch
from sensemem.semantics import FormationProposal
from sensemem.spatial import ObservationContext
from sensemem.types import AnswerPolicy, Modality


class EmbedTask(str, Enum):
    """Retrieval semantics for asymmetric embedding models."""

    QUERY = "retrieval.query"
    DOCUMENT = "retrieval.document"


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Aggregated text and ordered, resolved media assets for one model call."""

    text: str = ""
    assets: tuple[AssetRef, ...] = ()
    image_retrieval_matches: tuple[VisualRetrievalMatch, ...] = ()
    source_metadata: Mapping[str, dict[str, object]] = field(default_factory=dict, compare=False)
    evidence_sources: tuple[SearchHit, ...] = field(default=(), compare=False)
    parts: tuple[ContentPart, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("model input text must be text")
        assets = tuple(self.assets)
        if not self.text.strip() and not assets:
            raise ValidationError("model input must contain text or media")
        if any(not isinstance(asset, AssetRef) or not asset.is_resolved for asset in assets):
            raise ValidationError("model input assets must be resolved")
        matches = tuple(self.image_retrieval_matches)
        if any(not isinstance(match, VisualRetrievalMatch) for match in matches):
            raise ValidationError(
                "image retrieval matches must contain VisualRetrievalMatch values"
            )
        images = {asset.id for asset in assets if asset.modality is Modality.IMAGE}
        if any(match.query_asset_id not in images for match in matches):
            raise ValidationError("image retrieval matches must refer to a query image")
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "assets", assets)
        object.__setattr__(self, "image_retrieval_matches", matches)
        if not isinstance(self.source_metadata, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in self.source_metadata.items()
        ):
            raise ValidationError("source metadata must map asset IDs to metadata objects")
        sources = tuple(self.evidence_sources)
        if any(not isinstance(hit, SearchHit) for hit in sources):
            raise ValidationError("evidence sources must contain SearchHit values")
        object.__setattr__(self, "evidence_sources", sources)
        if any(not isinstance(part, ContentPart) for part in self.parts):
            raise ValidationError("model input parts must contain ContentPart values")
        if not self.parts:
            object.__setattr__(
                self,
                "parts",
                (
                    *((ContentPart("text", self.text),) if self.text else ()),
                    *(ContentPart("asset", asset.id) for asset in assets),
                ),
            )

    @property
    def modality(self) -> Modality:
        """Return text, one media kind, or omni for two or more media kinds."""
        kinds = {cast(Modality, asset.modality) for asset in self.assets}
        if not kinds:
            return Modality.TEXT
        if len(kinds) == 1:
            return next(iter(kinds))
        return Modality.OMNI

    @property
    def modalities(self) -> frozenset[Modality]:
        """Return every atomic modality present in this input."""
        values = {cast(Modality, asset.modality) for asset in self.assets}
        if self.text:
            values.add(Modality.TEXT)
        return frozenset(values)

    @property
    def assets_per_modality(self) -> int:
        """Maximum number of assets of any one media kind in this input."""
        return max(Counter(asset.modality for asset in self.assets).values(), default=0)

    @property
    def ordered_content(self) -> tuple[str | AssetRef, ...]:
        """Interleave retained source parts, then append any additional model context."""
        assets = {asset.id: asset for asset in self.assets}
        result: list[str | AssetRef] = []
        offset = 0
        for part in self.parts:
            if part.kind == "asset":
                if part.value in assets:
                    result.append(assets[part.value])
                continue
            position = self.text.find(part.value, offset)
            if position < 0:
                # A routed transcript may replace the original textual representation.
                return (*((self.text,) if self.text else ()), *self.assets)
            if prefix := self.text[offset:position].strip():
                result.append(prefix)
            result.append(part.value)
            offset = position + len(part.value)
        if suffix := self.text[offset:].strip():
            result.append(suffix)
        present = {value.id for value in result if isinstance(value, AssetRef)}
        result.extend(asset for asset in self.assets if asset.id not in present)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class EventObservation:
    """One ordered source chunk, retaining its capture interval and native evidence."""

    content: ModelInput
    occurred_at: datetime
    occurred_end: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.content, ModelInput):
            raise ValidationError("event observation content must be a ModelInput")
        if not isinstance(self.occurred_at, datetime) or not isinstance(
            self.occurred_end, datetime
        ):
            raise ValidationError("event observation requires occurred_at and occurred_end")
        _require_interval(self.occurred_at, self.occurred_end)


@runtime_checkable
class EventSegmentationBackend(Protocol):
    """Assess complete prefixes and summarize an unfinished suffix without finalizing it.

    A final ``EventSegment(complete=False)`` covers the entire remaining window. The runtime
    can carry this untrusted interpretation into the next bounded assessment while preserving
    every original media chunk outside the model's context window.
    """

    @property
    def event_model(self) -> str: ...

    @property
    def event_recipe(self) -> str: ...

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]: ...


@dataclass(frozen=True, slots=True)
class FormationInput:
    """One formation target and optional read-only, text-only historical evidence.

    History is background, not additional targets. A proposal must still cite its target;
    a historical record may be cited only while its current version remains active.
    ``history_truncated`` reports an incomplete window, never absence of counterevidence.
    Resolve relative dates against ``occurred_at``, falling back to ``created_at``; these
    source timestamps are separate from the assertion validity in ``context``.
    """

    memory_id: str
    content: ModelInput
    context: ObservationContext
    history: tuple[MemoryRecord, ...] = ()
    history_truncated: bool = False
    occurred_at: datetime | None = None
    occurred_end: datetime | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValidationError("formation input memory_id must not be blank")
        if not isinstance(self.content, ModelInput):
            raise ValidationError("formation input content must be a ModelInput")
        if not isinstance(self.context, ObservationContext):
            raise ValidationError("formation input context must be an ObservationContext")
        _require_interval(self.occurred_at, self.occurred_end)
        _require_aware(self.created_at, "formation input created_at")
        try:
            history = tuple(self.history)
        except TypeError:
            raise ValidationError("formation history must contain MemoryRecord values") from None
        if any(not isinstance(record, MemoryRecord) for record in history):
            raise ValidationError("formation history must contain MemoryRecord values")
        identifiers = [record.id for record in history]
        if self.memory_id in identifiers or len(set(identifiers)) != len(identifiers):
            raise ValidationError("formation history IDs must be unique and exclude the target")
        if any(record.assets or not record.content.strip() for record in history):
            raise ValidationError("formation history must contain text without media assets")
        if not isinstance(self.history_truncated, bool):
            raise ValidationError("formation history_truncated must be a boolean")
        object.__setattr__(self, "history", history)


def _modalities(
    values: Iterable[Modality],
    name: str,
) -> frozenset[Modality]:
    try:
        normalized = frozenset(
            value if isinstance(value, Modality) else Modality(value) for value in values
        )
    except (TypeError, ValueError):
        raise ValidationError(f"{name} capabilities contain an invalid modality") from None
    if Modality.OMNI in normalized:
        raise ValidationError(f"{name} capabilities must contain only atomic modalities")
    return normalized


@runtime_checkable
class EmbeddingBackend(Protocol):
    """One thread-safe embedding model with a stable vector-space recipe."""

    @property
    def embedding_capabilities(self) -> frozenset[Modality]: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def embedding_space(self) -> str: ...

    @property
    def embedding_dimension(self) -> int: ...

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class GenerationMediaBackend(Protocol):
    """Optional provider transport admission before source enrichment spends local work."""

    def grounding_assets(
        self, question: ModelInput, hits: Sequence[SearchHit]
    ) -> tuple[AssetRef, ...]: ...


@runtime_checkable
class GenerationRoutingBackend(Protocol):
    """Request-local selection and its native combination coverage."""

    @property
    def generation_routes(self) -> tuple[frozenset[Modality], ...]:
        """Native modality combinations covered by individual executable routes."""
        ...

    def select(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        convertible: frozenset[Modality] = frozenset(),
    ) -> GenerationBackend: ...


@runtime_checkable
class MediaSelectionBackend(Protocol):
    """Infer requested media indices from a kernel-built, untrusted evidence payload."""

    def select_media(
        self, payload: Mapping[str, object], *, count: int, max_items: int
    ) -> tuple[int, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class SourceMetadataBackend(Protocol):
    """Enrich integrity-checked source assets without rewriting their stored records."""

    def enrich(self, assets: Sequence[AssetRef]) -> Mapping[str, dict[str, object]]: ...

    def close(self) -> None: ...


@runtime_checkable
class EmbeddingInputSupport(Protocol):
    """Optional exact native combinations, including repeated-media constraints."""

    @property
    def embedding_routes(self) -> tuple[EmbeddingRoute, ...]: ...


def embedding_routes(
    backend: EmbeddingBackend, modalities: frozenset[Modality]
) -> tuple[EmbeddingRoute, ...]:
    """Read a backend's native input declaration; unrestricted backends accept every subset."""
    return (
        backend.embedding_routes
        if isinstance(backend, EmbeddingInputSupport)
        else EmbeddingRoute.combinations(modalities)
    )


@runtime_checkable
class RerankingBackend(Protocol):
    """A thread-safe multimodal pair scorer; positions retain caller-owned source IDs."""

    @property
    def reranking_capabilities(self) -> frozenset[Modality]: ...

    @property
    def reranking_model(self) -> str: ...

    def rerank(self, query: ModelInput, documents: Sequence[ModelInput]) -> tuple[float, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class FormationBackend(Protocol):
    """One thread-safe model that proposes MODEL_INFERENCE semantics without writing storage."""

    @property
    def formation_capabilities(self) -> frozenset[Modality]: ...

    @property
    def formation_model(self) -> str: ...

    @property
    def formation_space(self) -> str: ...

    def form(
        self,
        inputs: Sequence[FormationInput],
    ) -> tuple[tuple[FormationProposal, ...], ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class ConsolidationBackend(Protocol):
    """One thread-safe model that proposes memory operations over a bounded evidence set."""

    @property
    def consolidation_model(self) -> str: ...

    @property
    def consolidation_recipe(self) -> str: ...

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
        query_failure: QueryFailure | None = None,
    ) -> tuple[MemoryOperation, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class SpeechBackend(Protocol):
    """One thread-safe speech analyzer with a stable recognition recipe."""

    @property
    def transcription_capabilities(self) -> frozenset[Modality]: ...

    @property
    def transcription_model(self) -> str: ...

    @property
    def transcription_space(self) -> str: ...

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[perception.SpeechAnalysis, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class FaceBackend(Protocol):
    """One thread-safe local face analyzer with a stable embedding recipe."""

    @property
    def face_capabilities(self) -> frozenset[Modality]: ...

    @property
    def face_model(self) -> str: ...

    @property
    def face_space(self) -> str: ...

    @property
    def face_analysis_space(self) -> str: ...

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[perception.FaceAnalysis, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class GenerationBackend(Protocol):
    """One thread-safe grounded-answer adapter over a provider SDK.

    `exhaustive` says the hits are every record a recall program's predicate matched, in time
    order, rather than a ranking -- the one thing a grounded prompt cannot infer from the hits
    themselves and gets wrong by describing their order as rank.
    """

    @property
    def generation_capabilities(self) -> frozenset[Modality]: ...

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult: ...

    def close(self) -> None: ...


@runtime_checkable
class StreamingGenerationBackend(Protocol):
    """Optional capability that yields provider deltas and returns their grounded result.

    The terminal `AnswerResult` carries the exact grounding and abstention status. Its answer
    matches the deltas under `strict`; `best_effort` may remove a confidence marker from it.
    """

    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]: ...


@runtime_checkable
class RecallPlanningBackend(Protocol):
    """Optional generation capability that plans how to retrieve for one question.

    The return value is the model's own JSON text, which the kernel validates: a backend that
    cannot plan, or one whose plan is unusable, costs the caller a fallback to plain similarity
    search and nothing else. `corpus_digest` is one line describing what is in the store, so a
    plan cannot ask for a time span or a modality that does not exist. `attempted` is a bounded
    read-only investigation context, including initial evidence on the first round and new
    observations on later rounds. Its source excerpts are untrusted data. A plan may name
    unresolved facts and retain only the R aliases actually visible in that context.
    """

    def plan_recall(
        self,
        question: str,
        *,
        reference_at: datetime,
        corpus_digest: str,
        attempted: str = "",
    ) -> str | None: ...


@runtime_checkable
class TranscriptionBackend(Protocol):
    """One thread-safe plain-transcription adapter over a provider SDK."""

    @property
    def transcription_capabilities(self) -> frozenset[Modality]: ...

    @property
    def transcription_model(self) -> str: ...

    @property
    def transcription_space(self) -> str: ...

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]: ...

    def close(self) -> None: ...


@runtime_checkable
class VisionAnalysisBackend(Protocol):
    """Optional structured visual facts and explicit source-local name proposals."""

    def describe_analysis(self, inputs: Sequence[ModelInput]) -> tuple[VisualDescription, ...]: ...


@runtime_checkable
class VisionDescriptionBackend(Protocol):
    """One thread-safe visual-description adapter for text embedding fallback.

    ``vision_space`` identifies the whole caption recipe -- the model *and* the prompt or protocol
    version -- so a store can cache one caption per asset without a later recipe silently serving
    an earlier one. It is not the model name: an adapter that edits its prompt must return a
    different space, because the caption lands inside a searchable document where a stale one is
    invisible.
    """

    @property
    def vision_capabilities(self) -> frozenset[Modality]: ...

    @property
    def vision_model(self) -> str: ...

    @property
    def vision_space(self) -> str: ...

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        """Return one caption per input, in order -- never more, never fewer, never empty.

        `inputs[i].text`, when set, is derived transcript context this write already knows about
        the visual -- the clip's own words under the `speaker_N` labels the index prints, not
        anything the caller wrote. It is context to read, not content to echo back: repeating it
        verbatim in the caption duplicates text the document already carries under its own
        section.

        Captions are plain untrusted text. Structured facts and naming proposals require
        the optional ``VisionAnalysisBackend`` and its typed ``VisualDescription`` result;
        prose prefixes never authorize identity writes.
        """
        ...

    def close(self) -> None: ...
