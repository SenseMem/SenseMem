"""Reusable tools over public memory operations and existing perception backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from sensemem.exceptions import ModelError, ValidationError
from sensemem.identity import IdentityProfile, PerceptionResult
from sensemem.models.base import SpeechBackend, TranscriptionBackend
from sensemem.prompts import tools as tool_prompts
from sensemem.records import SearchHit
from sensemem.tools import RegisteredTool, Tool

if TYPE_CHECKING:
    from sensemem.memory import Memory


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Perceive(_Arguments):
    memory_id: StrictStr
    kinds: list[Literal["speech", "faces"]] | None = None


class _Perceived(BaseModel):
    result: PerceptionResult


class _Identity(_Arguments):
    identity_id: StrictStr


class _Profile(BaseModel):
    result: IdentityProfile | None


class _Search(_Arguments):
    query: StrictStr
    limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 10


class _Evidence(BaseModel):
    memory_id: str
    content: str
    score: float
    asset_ids: tuple[str, ...]


class _Found(BaseModel):
    hits: tuple[_Evidence, ...]


class _Compile(_Arguments):
    goal: StrictStr


class _Compiled(BaseModel):
    context: str
    source_ids: tuple[str, ...]
    excerpt_source_ids: tuple[str, ...]
    projection_source_ids: tuple[str, ...]
    omitted: int


def memory_tools(memory: Memory) -> tuple[RegisteredTool, ...]:
    """Borrow a memory's public operations as tools without granting writes or owning it.

    Perception may cache source observations. Identity confirmation, policy changes, deletion,
    and arbitrary writes are deliberately separate host operations.
    """

    def perceive(arguments: _Perceive) -> _Perceived:
        return _Perceived(
            result=memory.perceive(
                arguments.memory_id,
                kinds=None if arguments.kinds is None else frozenset(arguments.kinds),
            )
        )

    def identity(arguments: _Identity) -> _Profile:
        return _Profile(result=memory.identity(arguments.identity_id))

    def search(arguments: _Search) -> _Found:
        hits: Sequence[SearchHit] = memory.search(arguments.query, limit=arguments.limit).hits
        return _Found(
            hits=tuple(
                _Evidence(
                    memory_id=hit.id,
                    content=hit.content,
                    score=hit.score,
                    asset_ids=tuple(asset.id for asset in hit.assets),
                )
                for hit in hits
            )
        )

    def compile_context(arguments: _Compile) -> _Compiled:
        bundle = memory.compile(arguments.goal)
        return _Compiled(
            context=bundle.render(),
            source_ids=tuple(hit.id for hit in bundle.hits),
            excerpt_source_ids=tuple(item.source_memory_id for item in bundle.excerpts),
            projection_source_ids=tuple(item.source_memory_id for item in bundle.projections),
            omitted=bundle.omitted,
        )

    return (
        Tool(
            "perceive_memory",
            tool_prompts.PERCEIVE_MEMORY,
            _Perceive,
            _Perceived,
            perceive,
        ),
        Tool(
            "get_identity",
            tool_prompts.GET_IDENTITY,
            _Identity,
            _Profile,
            identity,
        ),
        Tool(
            "search_memories",
            tool_prompts.SEARCH_MEMORIES,
            _Search,
            _Found,
            search,
        ),
        Tool(
            "compile_context",
            tool_prompts.COMPILE_CONTEXT,
            _Compile,
            _Compiled,
            compile_context,
        ),
    )


class _Transcribe(_Arguments):
    memory_id: StrictStr


class _Transcript(BaseModel):
    asset_id: str
    text: str


class _Transcribed(BaseModel):
    model: str
    space: str
    transcripts: tuple[_Transcript, ...]


def transcription_tool(
    memory: Memory,
    backend: SpeechBackend | TranscriptionBackend,
    *,
    name: str = "transcribe",
) -> RegisteredTool:
    """Reuse one ASR implementation as an explicit tool over retained original media.

    Declare the backend as a resource so a registry and the ordinary transcription slot can
    share it under one Memory owner. The result is temporary; this does not write a transcript
    or turn it into a durable fact.
    """
    if not isinstance(backend, SpeechBackend | TranscriptionBackend):
        raise ValidationError("transcription tool requires a speech or transcription backend")

    def transcribe(arguments: _Transcribe) -> _Transcribed:
        record = memory.get(arguments.memory_id)
        assets = tuple(
            asset for asset in record.assets if asset.modality in backend.transcription_capabilities
        )
        if not assets:
            return _Transcribed(
                model=backend.transcription_model,
                space=backend.transcription_space,
                transcripts=(),
            )
        with memory.prepare_input(assets) as prepared:
            if isinstance(backend, TranscriptionBackend):
                texts = backend.transcribe(prepared.assets)
            else:
                texts = tuple(
                    "\n".join(turn.text for turn in analysis.turns)
                    for analysis in backend.analyze(prepared.assets)
                )
            if len(texts) != len(assets):
                raise ModelError(
                    "transcription output count did not match its input",
                    reason="response_invalid",
                    stage="transcription",
                )
        return _Transcribed(
            model=backend.transcription_model,
            space=backend.transcription_space,
            transcripts=tuple(
                _Transcript(asset_id=asset.id, text=text)
                for asset, text in zip(assets, texts, strict=True)
            ),
        )

    return Tool(
        name,
        tool_prompts.TRANSCRIBE,
        _Transcribe,
        _Transcribed,
        transcribe,
        resources=(backend,),
    )
