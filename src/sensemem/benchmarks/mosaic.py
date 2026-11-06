"""Pinned MOSAIC persona corpora and end-to-end scoring identities."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MOSAIC_DATA_REVISION = "c4567178d7430d87484db1ab207f5c8aaa26199a"
MOSAIC_SOURCE_REVISION = "82dc5ccf8063f41b28145f0c0e49747549aff457"
MOSAIC_ADAPTER_VERSION = "mosaic_native_official_v2"
MOSAIC_SCORER_VERSION = "mosaic_judge_82dc5ccf8063_v2"
MOSAIC_DATA_SHA256 = "c17cbab45499222e0590fdb0d557548366ed3d9deafe2ecd06ecd475625b51b7"
MOSAIC_QIDS_SHA256 = "8a42ac2fe6ad94490d083bd83e0a817f9ae205f461f38d8b791c6edafc6636a4"


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class MosaicMedia(_ReleaseModel):
    modality: Literal["image", "audio", "video"]
    media_path: str = Field(min_length=1)


class MosaicMessage(_ReleaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    multimodal_content: MosaicMedia | None = None
    fact_key: str = ""


class MosaicSession(_ReleaseModel):
    session_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    messages: tuple[MosaicMessage, ...]


class MosaicEvidence(_ReleaseModel):
    fact_key: str = Field(min_length=1)
    required_for_answer: bool = True
    available_at_query: bool = True


class MosaicQuestion(_ReleaseModel):
    question_id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    answerability: Literal["answerable", "unanswerable"]
    question: str = Field(min_length=1)
    answer: str
    answer_aliases: tuple[str, ...] = ()
    question_date: str = Field(min_length=1)
    haystack_session_ids: tuple[str, ...] = ()
    evidence_plan: tuple[MosaicEvidence, ...] = ()


class MosaicPersona(_ReleaseModel):
    persona_id: str
    sessions: tuple[MosaicSession, ...]
    questions: tuple[MosaicQuestion, ...]


class _RawQuestion(MosaicQuestion):
    haystack_sessions: tuple[MosaicSession, ...] = Field(min_length=1)


class _Persona(_ReleaseModel):
    uuid: str = Field(min_length=1)


class _Release(_ReleaseModel):
    persona: _Persona
    items: tuple[_RawQuestion, ...] = Field(min_length=1)


class _QuestionId(_ReleaseModel):
    question_id: str = Field(min_length=1)


class _QuestionIndex(_ReleaseModel):
    items: tuple[_QuestionId, ...] = Field(min_length=1)


def _selected_files(
    dataset: Path,
    limit: int | float | None,
    offset: int,
) -> tuple[tuple[Path, ...], set[str] | None]:
    paths = (
        tuple(sorted(dataset.glob("stage2_with_media_*.json"))) if dataset.is_dir() else (dataset,)
    )
    if limit in (None, -1) and offset == 0:
        return paths, None
    # Pydantic skips the large haystack field while reading this small question index.
    indexed = tuple((path, _QuestionIndex.model_validate_json(path.read_bytes())) for path in paths)
    questions = tuple(question for _, index in indexed for question in index.items)
    count = (
        len(questions)
        if limit is None or limit == -1
        else (max(1, math.ceil(len(questions) * limit)) if limit < 1 else int(limit))
    )
    ids = {question.question_id for question in questions[offset : offset + count]}
    return tuple(
        path for path, index in indexed if any(q.question_id in ids for q in index.items)
    ), ids


def load_mosaic(
    dataset: Path,
    *,
    limit: int | float | None = None,
    offset: int = 0,
) -> tuple[MosaicPersona, ...]:
    """Normalize each file once, retaining only its official deduplicated history.

    Raw questions repeat whole sessions. Releasing each raw persona before reading
    the next keeps batch loading from retaining all repeated haystacks at once.
    """
    paths, selected_ids = _selected_files(dataset, limit, offset)
    personas = []
    ids: set[str] = set()
    for path in paths:
        release = _Release.model_validate_json(path.read_bytes())
        sessions: dict[str, MosaicSession] = {}
        questions = []
        for raw in release.items:
            if raw.question_id in ids:
                raise ValueError("MOSAIC contains duplicate question IDs")
            ids.add(raw.question_id)
            if raw.answerability == "answerable" and not raw.answer.strip():
                raise ValueError(f"MOSAIC missing reference for {raw.question_id}")
            for session in raw.haystack_sessions:
                sessions.setdefault(session.session_id, session)
            if selected_ids is None or raw.question_id in selected_ids:
                questions.append(
                    MosaicQuestion(
                        question_id=raw.question_id,
                        question_type=raw.question_type,
                        answerability=raw.answerability,
                        question=raw.question,
                        answer=raw.answer,
                        answer_aliases=raw.answer_aliases,
                        question_date=raw.question_date,
                        haystack_session_ids=tuple(
                            session.session_id for session in raw.haystack_sessions
                        ),
                        evidence_plan=raw.evidence_plan,
                    )
                )
        personas.append(
            MosaicPersona(
                persona_id=release.persona.uuid,
                sessions=tuple(sorted(sessions.values(), key=lambda session: session.timestamp)),
                questions=tuple(questions),
            )
        )
        del release, raw
    if not personas:
        raise ValueError("MOSAIC annotations must not be empty")
    if len({persona.persona_id for persona in personas}) != len(personas):
        raise ValueError("MOSAIC contains duplicate persona IDs")
    return tuple(personas)


def mosaic_media_path(value: str) -> str:
    """Map the published machine-specific paths onto the release's media root."""
    for prefix in ("clean_output/media/", "dataset/media/"):
        if prefix in value:
            value = value.split(prefix, 1)[1]
            break
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid MOSAIC media path: {value}")
    return path.as_posix()


def mosaic_refusal(prediction: str) -> bool:
    """Use the paper table's substring rule for both refusal rates."""
    normalized = " ".join(prediction.lower().split())
    return any(
        phrase in normalized
        for phrase in (
            "i don't know",
            "i do not know",
            "cannot determine",
            "cannot answer",
            "not enough information",
            "insufficient information",
            "unable to determine",
            "unable to answer",
            "unanswerable",
        )
    )
