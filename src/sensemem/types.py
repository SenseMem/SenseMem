"""Shared enums and literal contracts for the public SDK."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Literal, TypeAlias


class Modality(str, Enum):
    """A native SenseMem input modality; omni means multiple media kinds."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    OMNI = "omni"


class MemoryCategory(str, Enum):
    """The cognitive role a memory serves for an agent."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class EvidenceBasis(str, Enum):
    """How a source or derived memory became known."""

    OBSERVATION = "observation"
    USER_STATEMENT = "user_statement"
    MODEL_INFERENCE = "model_inference"
    RESPONSE_FEEDBACK = "response_feedback"


class RecordKind(str, Enum):
    """The typed semantic role maintained for a memory record."""

    OBSERVATION = "observation"
    ENTITY = "entity"
    EVENT = "event"
    STATE = "state"
    RELATION = "relation"
    AFFECT = "affect"
    TRAIT = "trait"
    RESPONSE_POLICY = "response_policy"


CATEGORY_BY_KIND: Mapping[RecordKind, MemoryCategory] = MappingProxyType(
    {
        RecordKind.EVENT: MemoryCategory.EPISODIC,
        RecordKind.AFFECT: MemoryCategory.EPISODIC,
        RecordKind.RESPONSE_POLICY: MemoryCategory.PROCEDURAL,
    }
)


class SpatialAnchor(str, Enum):
    """Whether a pose locates the observer or the observed subject."""

    OBSERVER = "observer"
    SUBJECT = "subject"


class AbstentionReason(str, Enum):
    """Why an answer backend declined to answer from retrieved evidence."""

    NO_EVIDENCE = "no_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


AnswerPolicy: TypeAlias = Literal["strict", "best_effort"]


class IndexQuantization(str, Enum):
    """Explicit compression applied only to the rebuildable vector index."""

    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    RABITQ = "rabitq"


class RetrievalMode(str, Enum):
    """Candidate route policy for one local memory instance."""

    HYBRID = "hybrid"
    DENSE = "dense"
    LEXICAL = "lexical"
