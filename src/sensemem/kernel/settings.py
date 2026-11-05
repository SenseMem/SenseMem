"""Validated local policy for one open memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sensemem._validation import unit_interval
from sensemem.exceptions import ValidationError
from sensemem.infrastructure.local.store import RECALL_MAX_ROWS
from sensemem.kernel.metadata import index_recipe_for, validated_index_quantization
from sensemem.kernel.validation import (
    positive_int,
    positive_seconds,
    strict_bool,
    validated_decay_half_life,
    validated_retrieval_mode,
)
from sensemem.plugins import MemoryConfig
from sensemem.types import IndexQuantization, RetentionPolicy, RetrievalMode


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated local policy for one open memory, in the units the planes consume."""

    index_quantization: IndexQuantization
    index_recipe: str
    retrieval_mode: RetrievalMode
    speaker_similarity: float
    speaker_margin: float
    face_similarity: float
    face_margin: float
    identity_link_min_assets: int
    index_speech: bool
    independent_evidence: bool
    formation_history_max_rows: int
    formation_history_budget: int
    minimum_relevance: float
    ambiguity_margin: float
    evidence_budget: int | None
    recall_planning: bool
    recall_query_expansion: bool
    recall_max_queries: int
    query_vision: bool
    recall_set_budget: int
    recall_set_max_rows: int
    recall_rounds: int
    evidence_expansion: bool
    evidence_expansion_max_rows: int
    decay_half_life: timedelta | None
    reinforce_on_answer: bool
    memory_budget_records: int | None
    query_failure_window: timedelta
    query_failure_history: int
    retention: RetentionPolicy


def _expansion_max_rows(value: object) -> int:
    """Bound the expansion read the way the store bounds every other recall read.

    The value goes straight to `related_memories(max_rows=...)`, which refuses anything above
    `RECALL_MAX_ROWS`. Unbounded here, a caller who set it high built a memory that opened fine
    and then raised out of the store on every `ask`; policy is validated once, at wiring.
    """
    rows = positive_int(value, "evidence_expansion_max_rows")
    if rows > RECALL_MAX_ROWS:
        raise ValidationError(f"evidence_expansion_max_rows must be at most {RECALL_MAX_ROWS}")
    return rows


def resolve_settings(config: MemoryConfig | None) -> Settings:
    """Validate every `MemoryConfig` field once, at wiring, and fail before storage opens."""
    if config is None:
        config = MemoryConfig()
    elif not isinstance(config, MemoryConfig):
        raise ValidationError("config must be a MemoryConfig value")
    quantization = validated_index_quantization(config.index_quantization)
    settings = Settings(
        index_quantization=quantization,
        index_recipe=index_recipe_for(quantization),
        retrieval_mode=validated_retrieval_mode(config.retrieval_mode),
        speaker_similarity=unit_interval(config.speaker_similarity, "speaker_similarity"),
        speaker_margin=unit_interval(config.speaker_margin, "speaker_margin"),
        face_similarity=unit_interval(config.face_similarity, "face_similarity"),
        face_margin=unit_interval(config.face_margin, "face_margin"),
        identity_link_min_assets=positive_int(
            config.identity_link_min_assets, "identity_link_min_assets"
        ),
        index_speech=strict_bool(config.index_speech, "index_speech"),
        independent_evidence=strict_bool(config.independent_evidence, "independent_evidence"),
        formation_history_max_rows=_history_max_rows(config.formation_history_max_rows),
        formation_history_budget=positive_int(
            config.formation_history_budget_chars, "formation_history_budget_chars"
        ),
        minimum_relevance=unit_interval(config.minimum_relevance, "minimum_relevance"),
        ambiguity_margin=unit_interval(config.ambiguity_margin, "ambiguity_margin"),
        evidence_budget=None
        if config.evidence_budget_chars is None
        else positive_int(config.evidence_budget_chars, "evidence_budget_chars"),
        recall_planning=strict_bool(config.recall_planning, "recall_planning"),
        recall_query_expansion=strict_bool(config.recall_query_expansion, "recall_query_expansion"),
        query_vision=strict_bool(config.query_vision, "query_vision"),
        recall_set_budget=positive_int(config.recall_set_budget_chars, "recall_set_budget_chars"),
        recall_set_max_rows=positive_int(config.recall_set_max_rows, "recall_set_max_rows"),
        recall_rounds=positive_int(config.recall_rounds, "recall_rounds"),
        recall_max_queries=positive_int(config.recall_max_queries, "recall_max_queries"),
        evidence_expansion=strict_bool(config.evidence_expansion, "evidence_expansion"),
        evidence_expansion_max_rows=_expansion_max_rows(config.evidence_expansion_max_rows),
        decay_half_life=validated_decay_half_life(config.decay_half_life_days),
        reinforce_on_answer=strict_bool(config.reinforce_on_answer, "reinforce_on_answer"),
        memory_budget_records=(
            None
            if config.memory_budget_records is None
            else positive_int(config.memory_budget_records, "memory_budget_records")
        ),
        query_failure_window=positive_seconds(
            config.query_failure_window_seconds, "query_failure_window_seconds"
        ),
        query_failure_history=positive_int(config.query_failure_history, "query_failure_history"),
        retention=config.retention,
    )
    if not isinstance(settings.retention, RetentionPolicy):
        raise ValidationError("retention must be a RetentionPolicy value")
    if settings.recall_max_queries > 32:
        raise ValidationError("recall_max_queries must be at most 32")
    return settings


def _history_max_rows(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 128:
        raise ValidationError("formation_history_max_rows must be between 0 and 128")
    return value
