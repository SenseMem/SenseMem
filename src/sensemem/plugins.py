"""Explicit capability composition for one local memory instance."""

from __future__ import annotations

from dataclasses import dataclass

from sensemem._validation import nonempty_text
from sensemem.candidate_sources import CandidateSource
from sensemem.context_selection import ContextSelector
from sensemem.exceptions import ValidationError
from sensemem.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    RerankingBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from sensemem.processing import IngestionStage
from sensemem.query_policy import QueryPolicy
from sensemem.ranking_policy import RankingPolicy
from sensemem.tools import ToolRegistry


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryPlugins:
    """Capability objects, validated at assembly and owned and closed by ``Memory``."""

    embedder: EmbeddingBackend
    answerer: GenerationBackend | None = None
    transcriber: SpeechBackend | TranscriptionBackend | None = None
    vision_describer: VisionDescriptionBackend | None = None
    face_analyzer: FaceBackend | None = None
    former: FormationBackend | None = None
    consolidator: ConsolidationBackend | None = None
    reranker: RerankingBackend | None = None
    query_policy: QueryPolicy | None = None
    candidate_source: CandidateSource | None = None
    ranking_policy: RankingPolicy | None = None
    context_selector: ContextSelector | None = None
    ingestion_stages: tuple[IngestionStage, ...] = ()
    tools: ToolRegistry | None = None


def validate_computation_plugins(
    ranking_policy: RankingPolicy,
    context_selector: ContextSelector,
    ingestion_stages: tuple[IngestionStage, ...],
    query_policy: QueryPolicy,
    candidate_source: CandidateSource,
) -> None:
    """Validate computation roles and ordered stage identities before opening local storage."""
    if not isinstance(ranking_policy, RankingPolicy):
        raise ValidationError("plugins.ranking_policy must implement RankingPolicy")
    if not isinstance(context_selector, ContextSelector):
        raise ValidationError("plugins.context_selector must implement ContextSelector")
    if not isinstance(query_policy, QueryPolicy):
        raise ValidationError("plugins.query_policy must implement QueryPolicy")
    if not isinstance(candidate_source, CandidateSource):
        raise ValidationError("plugins.candidate_source must implement CandidateSource")
    if not isinstance(candidate_source.requires_embeddings, bool):
        raise ValidationError("candidate source requires_embeddings must be a boolean")
    stage_names: set[str] = set()
    for stage in ingestion_stages:
        if not isinstance(stage, IngestionStage):
            raise ValidationError("plugins.ingestion_stages must implement IngestionStage")
        name = nonempty_text(stage.name, "ingestion stage name")
        nonempty_text(stage.space, "ingestion stage space")
        if name in stage_names:
            raise ValidationError("ingestion stage names must be unique")
        stage_names.add(name)
