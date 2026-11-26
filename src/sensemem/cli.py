"""The SenseMem product command line: one process, one operation, one JSON document.

Every command dispatches to the Memory operation of the same name, or to the /v1 route a
running owner already serves. This module owns argument decoding, output encoding, and process
lifecycle. It owns no routing, retrieval, persistence, provider selection, or defaults of its own,
and no error policy beyond mapping the shared error codes onto stable exit statuses.

Composition is never implicit: exactly one of --app, --embedder, or --url must be
given. There is no default and no environment fallback, which is this surface's analogue of
Memory requiring embedder as a keyword with no default.
"""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import math
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import Field, asdict
from datetime import datetime, timedelta
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import ClassVar, Protocol, TextIO, TypeAlias
from urllib.parse import quote, urlencode
from uuid import uuid4

import sensemem.recipes as recipes
from sensemem._resources import Closable, close_quietly
from sensemem.api.client import JsonDocument as _Document
from sensemem.api.client import RemoteFailure, request_json
from sensemem.api.responses import (
    AffectCueResponse,
    MemoryResponse,
    SearchHitResponse,
    bundle_response,
)
from sensemem.content import AssetRef, Blob, ContentAtom, ContentInput
from sensemem.context_bundle import ContextBundle
from sensemem.context_values import (
    AffectCue,
    ContextBudget,
)
from sensemem.control import dump_operation, load_operation
from sensemem.control_values import (
    ExportBundle,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryTrigger,
    RetentionReport,
)
from sensemem.exceptions import (
    IdentityNotFoundError,
    IndexUnavailableError,
    MemoryNotFoundError,
    ModelError,
    ModelOutputTruncatedError,
    SenseMemError,
    StorageError,
    ValidationError,
)
from sensemem.identity import ConsentState
from sensemem.infrastructure.local._lock import DataDirectoryInUseError, DataDirectoryLock
from sensemem.kernel.contracts import declared_capabilities
from sensemem.memory import Memory
from sensemem.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FormationBackend,
    GenerationBackend,
    RerankingBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from sensemem.records import MemoryRecord, SearchHit
from sensemem.settings import MemorySettings
from sensemem.spatial import ObservationContext, RetrievalScope, SpatialContext
from sensemem.stream_values import StreamInput
from sensemem.types import EvidenceBasis, MemoryCategory, Modality, SpatialAnchor


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[object]]]


class _LocalCommand(Protocol):
    def __call__(self, memory: Memory, arguments: argparse.Namespace, /) -> _Document: ...


class _RemoteCommand(Protocol):
    def __call__(self, arguments: argparse.Namespace, /) -> tuple[str, str, _Document | None]: ...


class _RemoteResultTransform(Protocol):
    def __call__(self, document: _Document, /) -> _Document: ...


PROGRAM = "sensemem"
CONFIGURATION_ERROR = "configuration_error"
INTERRUPT_EXIT_CODE = 130
# One exit status per stable error code, so an agent branches on $? without reading anything.
# Keyed off the exception classes rather than string literals; tests/unit/test_cli.py fails when
# a new public exception has no status here instead of letting it collapse into exit 1.
EXIT_CODES: Mapping[str, int] = {
    ValidationError.code: 3,
    MemoryNotFoundError.code: 4,
    ModelError.code: 6,
    StorageError.code: 7,
    IndexUnavailableError.code: 8,
    CONFIGURATION_ERROR: 10,
    ModelOutputTruncatedError.code: 11,
    IdentityNotFoundError.code: 12,
}
# The one exit selected by reason rather than code. It is the CLI's single transport decision:
# an agent that cannot tell "busy, retry with --url" from "the disk broke" cannot be scripted.
DATA_DIR_IN_USE_EXIT_CODE = 9
DOCTOR = "doctor"
_QUERY_METAVAR: Mapping[str, str] = {
    "add": "TEXT",
    "capture": "TEXT",
    "search": "QUERY",
    "ask": "QUESTION",
    "compile": "GOAL",
    "consolidate": "GOAL",
}
_DEFAULT_REMOTE_TIMEOUT_SECONDS = 30.0
# One list, because a slot added to recipes used to need remembering in four separate literals
# here; the flag, the explain document, doctor, and the composition guard all derive from it.
_SLOTS: tuple[str, ...] = (
    "embedder",
    "answerer",
    "former",
    "consolidator",
    "transcriber",
    "vision",
)
_OPTIONAL_SLOTS: tuple[str, ...] = _SLOTS[1:]
# The one slot whose Memory keyword is not its name. MemorySpec and the flag call the
# slot vision; the object it holds is a describer, which is what Memory names its parameter.
_MEMORY_KEYWORDS: Mapping[str, str] = {"vision": "vision_describer"}
_TUNING: tuple[str, ...] = (
    "index_speech",
    "minimum_relevance",
    "ambiguity_margin",
    "decay_half_life_days",
)
_STDIN = "-"

_Atom: TypeAlias = tuple[str, str]
_stdin_consumed = False


class _CompositionError(Exception):
    """A composition or transport condition the CLI itself owns, reported as exit 10.

    Deliberately not a SenseMemError: that taxonomy belongs to the SDK, and adding a class to
    it here would silently change the code set the MCP adapter derives from the class hierarchy.
    """

    def __init__(self, message: str, *, reason: str, subject: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.subject = subject


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit status."""
    global _stdin_consumed
    _stdin_consumed = False
    arguments = _parser().parse_args(None if argv is None else list(argv))
    try:
        document = _dispatch(arguments)
    except KeyboardInterrupt:
        print(f"{PROGRAM}: interrupted", file=sys.stderr)
        return INTERRUPT_EXIT_CODE
    except RemoteFailure as error:
        return _forward(error.envelope)
    except _CompositionError as error:
        return _fail(CONFIGURATION_ERROR, str(error), reason=error.reason, subject=error.subject)
    except SenseMemError as error:
        return _fail(
            error.code,
            str(error) or error.code,
            reason=error.reason,
            retryable=error.retryable,
            stage=error.stage,
            subject=error.subject,
        )
    except Exception as error:
        message = " ".join(str(error).split()) or type(error).__name__
        return _fail("internal_error", message, reason="unexpected")
    json.dump(document, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def _dispatch(arguments: argparse.Namespace) -> _Document:
    _reject_duplicate_content(arguments)
    composition = _describe_composition(arguments)
    if arguments.explain_composition:
        return composition
    if not arguments.quiet:
        print(json.dumps(composition, ensure_ascii=False), file=sys.stderr)
    if arguments.url is not None:
        return _run_remote(arguments)
    if arguments.command == DOCTOR:
        return _doctor(arguments, composition)
    with _open_memory(arguments) as memory:
        return _LOCAL[arguments.command](memory, arguments)


# Composition


def _describe_composition(arguments: argparse.Namespace) -> _Document:
    """Validate composition options and describe the selected runtime before opening it."""
    if arguments.url is None and arguments.timeout is not None:
        raise _CompositionError(
            "--timeout applies only to --url compositions",
            reason="option_not_applicable",
            subject="--timeout",
        )
    if arguments.embedder is None:
        _reject_embedder_only_options(arguments)
    if arguments.app is None and arguments.embedder is None and arguments.url is None:
        # Not argparse's required=True: a missing composition is a configuration condition an
        # agent must be able to recognize by exit status, not a usage typo.
        raise _CompositionError(
            "no composition was given; pass exactly one of --app MODULE:ATTR, --embedder NAME "
            f"({_recipes()}), or --url URL. There is no default and no environment fallback.",
            reason="composition_missing",
        )
    if arguments.url is not None:
        return {
            "source": f"--url {arguments.url}",
            "url": arguments.url,
            "timeout_seconds": _remote_timeout(arguments),
        }
    if arguments.app is not None:
        module_name, attribute = _application_target(arguments.app)
        return {
            "source": f"--app {arguments.app}",
            # The application chose the directory and composed the backends, so both are the
            # application's to report: Memory publishes no accessor for either.
            "data_dir": None,
            "app": {"module": module_name, "attribute": attribute},
        }
    document: _Document = {
        "source": f"--embedder {arguments.embedder}",
        "data_dir": str(_data_dir(arguments)),
    }
    with _composing():
        for slot in _SLOTS:
            name = getattr(arguments, slot)
            if name is None:
                document[slot] = None
                continue
            # Checked while resolving, so --explain, doctor, and every operation agree that a
            # recipe in the wrong slot is one configuration failure rather than a loader result.
            recipes.require_slot(name, slot)
            document[slot] = recipes.describe(name)
    return document


def _reject_embedder_only_options(arguments: argparse.Namespace) -> None:
    """--app and --url compose elsewhere, so a knob meant for a recipe must not be ignored."""
    given = [
        f"--{name.replace('_', '-')}"
        for name in ("data_dir", *_OPTIONAL_SLOTS, *_TUNING)
        if getattr(arguments, name) != _MEMORY_DEFAULTS[name]
    ]
    if given:
        owner = "the application" if arguments.app is not None else "the running owner"
        raise _CompositionError(
            f"{', '.join(given)} apply to --embedder compositions; {owner} owns them here",
            reason="option_not_applicable",
            subject=given[0],
        )


def _open_memory(arguments: argparse.Namespace) -> Memory:
    if arguments.app is not None:
        return _application_memory(arguments.app)
    created: list[Closable] = []
    try:
        with _composing():
            embedder = recipes.embedder(arguments.embedder)
            created.append(embedder)
            answerer: GenerationBackend | None = None
            if arguments.answerer is not None:
                answerer = recipes.answerer(arguments.answerer)
                created.append(answerer)
            former: FormationBackend | None = None
            if arguments.former is not None:
                former = recipes.former(arguments.former)
                created.append(former)
            consolidator: ConsolidationBackend | None = None
            if arguments.consolidator is not None:
                consolidator = recipes.consolidator(arguments.consolidator)
                created.append(consolidator)
            transcriber: SpeechBackend | TranscriptionBackend | None = None
            if arguments.transcriber is not None:
                transcriber = recipes.transcriber(arguments.transcriber)
                created.append(transcriber)
            vision: VisionDescriptionBackend | None = None
            if arguments.vision is not None:
                vision = recipes.vision(arguments.vision)
                created.append(vision)
        # Memory closes the backends it accepted, but it opens the store before it accepts them,
        # so a busy data directory would otherwise leak whatever this process just constructed.
        return Memory(
            _data_dir(arguments),
            embedder=embedder,
            answerer=answerer,
            former=former,
            consolidator=consolidator,
            transcriber=transcriber,
            vision_describer=vision,
            settings=MemorySettings(
                index_speech=arguments.index_speech,
                minimum_relevance=arguments.minimum_relevance,
                ambiguity_margin=arguments.ambiguity_margin,
                decay_half_life_days=arguments.decay_half_life_days,
            ),
        )
    except BaseException:
        close_quietly(*created)
        raise


@contextmanager
def _cwd_import_path() -> Iterator[None]:
    """Allow --app MODULE:ATTR to resolve from cwd without leaving sys.path mutated."""
    added = "" not in sys.path
    if added:
        sys.path.insert(0, "")
    try:
        yield
    finally:
        if added and "" in sys.path:
            sys.path.remove("")


def _application_memory(spec: str) -> Memory:
    module_name, attribute = _application_target(spec)
    with _composing(spec), _cwd_import_path():
        value = getattr(import_module(module_name), attribute)
    if isinstance(value, Memory):
        return value
    if not callable(value):
        raise _CompositionError(
            f"{spec} is neither a Memory nor a zero-argument callable returning one",
            reason="app_invalid",
            subject=spec,
        )
    with _composing(spec):
        built = value()
    if not isinstance(built, Memory):
        raise _CompositionError(
            f"{spec} returned {type(built).__name__}, not a Memory",
            reason="app_invalid",
            subject=spec,
        )
    return built


def _application_target(spec: str) -> tuple[str, str]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise _CompositionError(
            f"--app must be MODULE:ATTR, not {spec}", reason="app_invalid", subject=spec
        )
    return module_name.strip(), attribute.strip()


@contextmanager
def _composing(subject: str | None = None) -> Iterator[None]:
    """Report every composition failure as exit 10, whatever raised it underneath."""
    try:
        yield
    except _CompositionError:
        raise
    except StorageError:
        # A busy or unreadable data directory describes the store, not the composition, and it is
        # the one failure whose exit status tells the caller to retry against --url instead.
        raise
    except ImportError as error:
        raise _CompositionError(
            f"composition needs a package that is not installed: {error.name}",
            reason="missing_dependency",
            subject=error.name,
        ) from error
    except Exception as error:
        message = " ".join(str(error).split()) or type(error).__name__
        raise _CompositionError(message, reason="composition_failed", subject=subject) from error


def _data_dir(arguments: argparse.Namespace) -> Path:
    return Path(arguments.data_dir).expanduser().resolve()


# doctor


def _doctor(arguments: argparse.Namespace, composition: _Document) -> _Document:
    report: _Document = {"source": composition["source"]}
    data_dir: Path | None = None
    state = "owned by the application"
    loaded: dict[str, Closable] = {}
    # --app owns its own backends and doctor never calls the factory, so there is nothing to
    # declare there; a local composition declares what its probed backends can do.
    capabilities: _Document | None = None
    if arguments.app is not None:
        report["app"] = composition["app"]
        # Calling the factory would open the store, so the check stops at "the target resolves".
        report["probe"] = "resolved, not called"
        _application_target_exists(arguments.app)
    else:
        data_dir = _data_dir(arguments)
        state = _data_dir_state(data_dir)
        try:
            for slot in _SLOTS:
                name = getattr(arguments, slot)
                report[slot] = None if name is None else _probe(name, slot, loaded)
            capabilities = _doctor_capabilities(loaded)
        finally:
            close_quietly(*loaded.values())
    return {
        "version": _installed_version(),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "data_dir": None if data_dir is None else str(data_dir),
        "data_dir_state": state,
        # The same document /healthz serves and the MCP server greets an agent with, so the
        # three surfaces cannot describe one composition differently.
        "capabilities": capabilities,
        "composition": report,
    }


def _doctor_capabilities(loaded: Mapping[str, object]) -> _Document | None:
    """Declare what the probed backends can do, without opening the store Memory would.

    Every capability is a backend declaration, so this needs no data_dir and stays true to
    doctor's rule that diagnosing a busy or absent directory must not create or lock anything.
    """
    embedder = loaded.get("embedder")
    if not isinstance(embedder, EmbeddingBackend):
        return None
    answerer = loaded.get("answerer")
    transcriber = loaded.get("transcriber")
    vision = loaded.get("vision")
    former = loaded.get("former")
    consolidator = loaded.get("consolidator")
    reranker = loaded.get("reranker")
    try:
        capabilities = declared_capabilities(
            embedder=embedder,
            answerer=answerer if isinstance(answerer, GenerationBackend) else None,
            transcriber=(
                transcriber
                if isinstance(transcriber, (SpeechBackend, TranscriptionBackend))
                else None
            ),
            vision_describer=vision if isinstance(vision, VisionDescriptionBackend) else None,
            former=former if isinstance(former, FormationBackend) else None,
            consolidator=consolidator if isinstance(consolidator, ConsolidationBackend) else None,
            reranker=reranker if isinstance(reranker, RerankingBackend) else None,
        )
    except SenseMemError:
        # A backend that declares an invalid contract is already reported per slot above; the
        # capability document is a summary, not a second place to fail the command.
        return None
    return capabilities.document()


def _application_target_exists(spec: str) -> None:
    module_name, attribute = _application_target(spec)
    with _composing(spec), _cwd_import_path():
        getattr(import_module(module_name), attribute)


def _load_recipe(slot: str, name: str) -> Closable:
    if slot == "embedder":
        return recipes.embedder(name, load=True)
    if slot == "answerer":
        return recipes.answerer(name, load=True)
    if slot == "former":
        return recipes.former(name, load=True)
    if slot == "consolidator":
        return recipes.consolidator(name, load=True)
    if slot == "transcriber":
        return recipes.transcriber(name, load=True)
    if slot == "vision":
        return recipes.vision(name, load=True)
    raise ValidationError(f"unknown slot: {slot}")


def _probe(name: str, slot: str, loaded: dict[str, Closable]) -> _Document:
    """Construct one recipe with its loader exercised, keeping it for the capability summary."""
    document: _Document = dict(recipes.describe(name))
    document["probe"] = recipes.probe(name)
    try:
        backend = _load_recipe(slot, name)
    except ImportError as error:
        document.update(loader="failed", reason="missing_dependency", detail=error.name)
        return document
    except Exception as error:
        reason = error.reason if isinstance(error, SenseMemError) else "load_failed"
        document.update(
            loader="failed",
            reason=reason,
            detail=" ".join(str(error).split()) or type(error).__name__,
        )
        return document
    loaded[slot] = backend
    document["loader"] = "ok"
    if isinstance(backend, EmbeddingBackend):
        document["embedding_model"] = backend.embedding_model
        document["embedding_space"] = backend.embedding_space
        document["embedding_dimension"] = backend.embedding_dimension
        document["embedding_modalities"] = sorted(
            modality.value for modality in backend.embedding_capabilities
        )
    if isinstance(backend, GenerationBackend):
        document["generation_modalities"] = sorted(
            modality.value for modality in backend.generation_capabilities
        )
    if isinstance(backend, (SpeechBackend, TranscriptionBackend)):
        document["transcription_model"] = backend.transcription_model
        document["transcription_space"] = backend.transcription_space
        document["transcription_modalities"] = sorted(
            modality.value for modality in backend.transcription_capabilities
        )
    return document


def _data_dir_state(data_dir: Path) -> str:
    """Report ownership without creating anything: with no lock file there is no live owner."""
    if not (data_dir / ".sensemem.lock").exists():
        return "absent" if not data_dir.exists() else "free"
    try:
        lock = DataDirectoryLock(data_dir)
    except DataDirectoryInUseError:
        return "in use by another process"
    except OSError as error:
        return f"unreadable: {error.strerror}"
    lock.close()
    return "free"


def _installed_version() -> str:
    return version("sensemem")


# Local execution


def _add(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return _memory_document(
        memory.add(
            _content_input(arguments),
            occurred_at=_optional_time(arguments.occurred_at, "occurred_at"),
            occurred_end=_optional_time(arguments.occurred_end, "occurred_end"),
            metadata=_metadata_value(_json_source(arguments.metadata)),
            memory_type=MemoryCategory(arguments.memory_type),
            context=_observation_context(_json_source(arguments.context)),
        )
    )


def _capture(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return _memory_document(
        memory.capture(
            _content_input(arguments),
            occurred_at=_optional_time(arguments.occurred_at, "occurred_at"),
            occurred_end=_optional_time(arguments.occurred_end, "occurred_end"),
            metadata=_metadata_value(_json_source(arguments.metadata)),
            memory_type=MemoryCategory(arguments.memory_type),
            context=_observation_context(_json_source(arguments.context)),
        )
    )


def _add_many(memory: Memory, arguments: argparse.Namespace) -> _Document:
    items = _jsonl(arguments.source)
    return {
        "memories": [
            _memory_document(record)
            for record in memory.add_many(
                tuple(_parts_input(item["content"]) for item in items),
                occurred_at=[
                    _optional_time(item.get("occurred_at"), "occurred_at") for item in items
                ],
                occurred_end=[
                    _optional_time(item.get("occurred_end"), "occurred_end") for item in items
                ],
                metadata=[_metadata_value(item.get("metadata")) for item in items],
                memory_type=MemoryCategory(arguments.memory_type),
                context=[_observation_context(item.get("context")) for item in items],
            )
        ]
    }


def _add_stream(memory: Memory, arguments: argparse.Namespace) -> _Document:
    memory_type = MemoryCategory(arguments.memory_type)
    inputs = (
        StreamInput(
            _parts_input(item["content"]),
            occurred_at=_optional_time(item.get("occurred_at"), "occurred_at"),
            occurred_end=_optional_time(item.get("occurred_end"), "occurred_end"),
            metadata=_metadata_value(item.get("metadata")),
            memory_type=memory_type,
            context=_observation_context(item.get("context")),
        )
        for item in _jsonl_stream(arguments.source)
    )
    # ponytail: preserve the CLI's one-JSON-document contract by collecting result records; add
    # streaming output only if finite CLI imports outgrow memory. The SDK input path stays lazy.
    return {
        "memories": [
            _memory_document(record)
            for record in memory.add_stream(inputs, capture=arguments.capture)
        ]
    }


def _search(memory: Memory, arguments: argparse.Namespace) -> _Document:
    result = memory.search(
        _content_input(arguments),
        limit=arguments.limit,
        memory_type=_optional_memory_type(arguments),
        reference_at=_optional_time(arguments.reference_at, "reference_at"),
        occurred_from=_optional_time(arguments.occurred_from, "occurred_from"),
        occurred_until=_optional_time(arguments.occurred_until, "occurred_until"),
        scope=_retrieval_scope(_json_source(arguments.scope)),
        explain=arguments.explain,
    )
    return {
        "hits": [_memory_document(hit) for hit in result.hits],
        "trace": None if result.trace is None else asdict(result.trace),
    }


def _ask(memory: Memory, arguments: argparse.Namespace) -> _Document:
    result = memory.ask(
        _content_input(arguments),
        limit=arguments.limit,
        memory_type=_optional_memory_type(arguments),
        reference_at=_optional_time(arguments.reference_at, "reference_at"),
        scope=_retrieval_scope(_json_source(arguments.scope)),
    )
    return {
        "answer": result.answer,
        "hits": [_memory_document(hit) for hit in result.hits],
        "abstained": result.abstained,
        "abstention_reason": (
            None if result.abstention_reason is None else result.abstention_reason.value
        ),
    }


def _budget(arguments: argparse.Namespace) -> ContextBudget:
    return ContextBudget(
        max_chars=arguments.max_chars,
        max_items=arguments.max_items,
        max_media_items=arguments.max_media_items,
        memory_types=(
            None
            if arguments.memory_type is None
            else frozenset(MemoryCategory(value) for value in arguments.memory_type)
        ),
        min_confidence=arguments.min_confidence,
        freshness=_freshness_delta(arguments.freshness_seconds),
        max_latency_ms=arguments.max_latency_ms,
        selected_proofs=arguments.selected_proofs,
    )


def _freshness_delta(seconds: float | None) -> timedelta | None:
    if seconds is None:
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValidationError("freshness_seconds must be a positive finite number")
    try:
        return timedelta(seconds=seconds)
    except OverflowError:
        raise ValidationError("freshness_seconds is outside the supported duration range") from None


def _compile(memory: Memory, arguments: argparse.Namespace) -> _Document:
    bundle = memory.compile(
        _content_input(arguments),
        budget=_budget(arguments),
        reference_at=_optional_time(arguments.reference_at, "reference_at"),
        scope=_retrieval_scope(_json_source(arguments.scope)),
        allow_partial_sources=arguments.allow_partial_sources,
    )
    # ContextBundle.document() is the projection REST and MCP publish too, so a section added
    # to the bundle reaches this document without being spelled here a third time.
    return _bundle_document(bundle)


def _bundle_document(bundle: ContextBundle) -> _Document:
    """Publish the shared safe response projection, including nested media and proofs."""
    dumped: object = bundle_response(bundle).model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def _get(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return _memory_document(memory.get(arguments.memory_id))


def _perceive(memory: Memory, arguments: argparse.Namespace) -> _Document:
    kinds = None if arguments.kind is None else frozenset(arguments.kind)
    return _document(memory.perceive(arguments.memory_id, kinds=kinds))


def _register_identity(memory: Memory, arguments: argparse.Namespace) -> _Document:
    memory.register_identity(
        arguments.identity_id,
        arguments.name,
        relationship=arguments.relationship,
    )
    return {}


def _identity_profile(memory: Memory, arguments: argparse.Namespace) -> _Document:
    profile = memory.identity(arguments.identity_id)
    if profile is None:
        return {"identity": None}
    return {
        "identity": {
            "identity_id": profile.identity_id,
            "name": profile.name,
            "relationship": profile.relationship,
        }
    }


def _forget_identity(memory: Memory, arguments: argparse.Namespace) -> _Document:
    erasure = memory.forget_identity(arguments.identity_id)
    # The audit record is the point: an operator running this needs to see what was destroyed.
    return {
        "identity_id": erasure.identity_id,
        "alias_ids": list(erasure.alias_ids),
        "face_exemplars": erasure.face_exemplars,
        "voice_exemplars": erasure.voice_exemplars,
        "face_observations": erasure.face_observations,
        "speech_segments": erasure.speech_segments,
    }


def _unlink_identity(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {"restored_identity_id": memory.unlink_identity(arguments.alias_id)}


def _record_consent(memory: Memory, arguments: argparse.Namespace) -> _Document:
    record = memory.record_consent(
        arguments.identity_id,
        ConsentState(arguments.state),
        note=arguments.note,
    )
    return {"operation": None if record is None else _operation_document(record)}


def _consent(memory: Memory, arguments: argparse.Namespace) -> _Document:
    state = memory.consent(arguments.identity_id)
    return {"consent": None if state is None else state.value}


def _export(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return _export_document(
        memory.export(
            identity_id=arguments.identity_id,
            memory_ids=arguments.memory_ids or None,
        )
    )


def _export_document(bundle: ExportBundle) -> _Document:
    return {
        "exported_at": _encode_time(bundle.exported_at),
        "identity_id": bundle.identity_id,
        "identities": [_document(profile) for profile in bundle.identities],
        "records": [_memory_document(record) for record in bundle.records],
        "operations": [_operation_document(record) for record in bundle.operations],
    }


def _apply_retention(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return _retention_document(memory.apply_retention(dry_run=arguments.dry_run))


def _retention_document(report: RetentionReport) -> _Document:
    # deleted is derived, not a declared field, so it is the one value asdict cannot supply.
    return {**_document(report), "deleted": report.deleted}


def _settle(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {
        "settled": memory.settle(
            limit=arguments.limit,
            max_attempts=arguments.max_attempts,
            memory_ids=tuple(arguments.memory_ids) or None,
        )
    }


def _pending_captures(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {
        "pending": [
            {
                "memory_id": pending.memory_id,
                "enqueued_at": _encode_time(pending.enqueued_at),
                "attempts": pending.attempts,
                "last_error": pending.last_error,
                "awaiting": pending.awaiting,
            }
            for pending in memory.pending_captures(
                limit=arguments.limit,
                memory_ids=tuple(arguments.memory_ids) or None,
            )
        ]
    }


def _reinforce(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {"reinforced": memory.reinforce(arguments.memory_ids)}


def _consolidation_candidates(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {
        "candidates": [
            {
                "trigger": candidate.trigger.value,
                "memory_ids": list(candidate.memory_ids),
                "evidence_count": candidate.evidence_count,
            }
            for candidate in memory.consolidation_candidates(
                limit=arguments.limit,
                idle=arguments.idle,
            )
        ]
    }


def _consolidate(memory: Memory, arguments: argparse.Namespace) -> _Document:
    if arguments.command == "consolidate-pending":
        report = memory.consolidate_pending(
            limit=arguments.limit, max_rounds=arguments.max_rounds, idle=arguments.idle
        )
    else:
        report = memory.consolidate(
            evidence_ids=arguments.evidence_ids or None,
            query=(
                _content_input(arguments)
                if arguments.content or arguments.content_json is not None
                else None
            ),
            limit=arguments.limit,
            trigger=MemoryTrigger(arguments.trigger),
        )
    return {
        "operations": [_operation_document(record) for record in report.operations],
        "rejected": [
            {"intent": operation.intent.value, "reason": reason}
            for operation, reason in report.rejected
        ],
        "weighed": report.weighed,
        "rounds": report.rounds,
        "skipped": report.skipped,
        "applied": report.applied,
        "model_calls": report.model_calls,
    }


def _apply(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {"operation": _operation_document(memory.apply(_operation_input(arguments.operation)))}


def _operation_input(source: str) -> MemoryOperation:
    """Read one operation from the same JSON the log stores, so a log row replays verbatim."""
    return load_operation(_read_source(source))


def _record_outcome(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {
        "recorded": memory.record_outcome(
            arguments.operation_id,
            MemoryOutcome(arguments.outcome),
            note=arguments.note,
        )
    }


def _forget(memory: Memory, arguments: argparse.Namespace) -> _Document:
    record = memory.forget(arguments.memory_ids)
    return {"operation": None if record is None else _operation_document(record)}


def _rollback(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {"rolled_back": memory.rollback(arguments.operation_id)}


def _operations(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {
        "operations": [
            _operation_document(record) for record in memory.operations(limit=arguments.limit)
        ]
    }


def _operation_document(record: MemoryOperationRecord) -> _Document:
    # claim and proposal are read back from the canonical payload the operation log stores,
    # so the only audit surface for an operation prints exactly what was logged: an identify
    # row carries no evidence or targets, and its claim is the whole of what it did.
    logged = json.loads(dump_operation(record.operation))
    return {
        "operation_id": record.operation_id,
        "intent": record.operation.intent.value,
        "trigger": record.trigger.value,
        "evidence_ids": list(record.operation.evidence_ids),
        "target_ids": list(record.operation.target_ids),
        # A consolidation's subject is what it proposed, so without this its row would report
        # an intent and no statement -- and apply --operation takes a row as operations
        # prints it, which the kernel refuses for a consolidation carrying no proposal.
        "proposal": logged["proposal"],
        # A merge, split, or identity erasure names people rather than records, so without this
        # its row would report an intent and no subject at all.
        "identity": logged["identity"],
        # A naming or consent row asserts something about a person, so without these two its
        # row would report an intent and no statement at all.
        "claim": logged["claim"],
        "consent": logged["consent"],
        "rationale": record.operation.rationale,
        "model_id": record.model_id,
        "recipe": record.recipe,
        "created_ids": list(record.created_ids),
        "changed_ids": list(record.changed_ids),
        "forgotten_ids": list(record.forgotten_ids),
        "superseded": [[memory_id, version] for memory_id, version in record.superseded],
        "applied_at": _encode_time(record.applied_at),
        "rolled_back_at": _encode_optional_time(record.rolled_back_at),
        "outcome": None if record.outcome is None else record.outcome.value,
        "outcome_note": record.outcome_note,
    }


def _list(memory: Memory, arguments: argparse.Namespace) -> _Document:
    page = memory.list(limit=arguments.limit, cursor=arguments.cursor)
    return {
        "items": [_memory_document(record) for record in page.items],
        "next_cursor": page.next_cursor,
    }


def _delete(memory: Memory, arguments: argparse.Namespace) -> _Document:
    return {"deleted": memory.delete(arguments.memory_id)}


def _reindex(memory: Memory, _arguments: argparse.Namespace) -> _Document:
    return {"memories": memory.reindex()}


def _optimize(memory: Memory, _arguments: argparse.Namespace) -> _Document:
    memory.optimize()
    return {}


_LOCAL: Mapping[str, _LocalCommand] = {
    "add": _add,
    "add-many": _add_many,
    "add-stream": _add_stream,
    "capture": _capture,
    "settle": _settle,
    "pending-captures": _pending_captures,
    "search": _search,
    "ask": _ask,
    "compile": _compile,
    "get": _get,
    "perceive": _perceive,
    "register-identity": _register_identity,
    "identity": _identity_profile,
    "record-consent": _record_consent,
    "consent": _consent,
    "forget-identity": _forget_identity,
    "unlink-identity": _unlink_identity,
    "reinforce": _reinforce,
    "consolidation-candidates": _consolidation_candidates,
    "consolidate": _consolidate,
    "consolidate-pending": _consolidate,
    "apply": _apply,
    "record-outcome": _record_outcome,
    "forget": _forget,
    "rollback": _rollback,
    "operations": _operations,
    "export": _export,
    "apply-retention": _apply_retention,
    "list": _list,
    "delete": _delete,
    "reindex": _reindex,
    "optimize": _optimize,
}


def _optional_memory_type(arguments: argparse.Namespace) -> MemoryCategory | None:
    value = arguments.memory_type
    return None if value is None else MemoryCategory(value)


# Remote execution against a running owner


def _run_remote(arguments: argparse.Namespace) -> _Document:
    command = arguments.command
    if command == DOCTOR:
        return _remote_doctor(arguments)
    if command not in _REMOTE:
        raise _CompositionError(
            f"{command} has no /v1 route, so it cannot run against --url; the Python SDK and a "
            "local --app or --embedder composition support it",
            reason="unsupported_in_remote_mode",
            subject=command,
        )
    method, path, body = _REMOTE[command](arguments)
    try:
        document = request_json(arguments.url, method, path, body, _remote_timeout(arguments))
    except RemoteFailure as failure:
        raise _remote_switch_off(command, arguments.url, failure) from failure
    return _REMOTE_RESULT[command](document) if command in _REMOTE_RESULT else document


def _remote_switch_off(command: str, url: str, failure: RemoteFailure) -> Exception:
    """Report a gated route the owner never registered as the configuration condition it is.

    create_app leaves the identity and embodied routes unregistered when the host did not opt
    in, so the owner answers "route does not exist" rather than anything about the subject the
    command named. That code is in no exit table, so forwarding it verbatim exits 1 with a
    message about routing; the switch the host left off is the actual subject.

    A base URL pointing somewhere that is not a SenseMem owner answers the same 404, and no
    probe distinguishes the two -- a /healthz request would answer for whatever is listening
    there. The message names both possibilities rather than asserting the likelier one.
    """
    switch = _REMOTE_SWITCH.get(command)
    if switch is None or failure.envelope.get("code") != "not_found":
        return failure
    return _CompositionError(
        f"the owner at {url} does not serve {command}: it was started without {switch}, or the "
        "URL does not point at a SenseMem server",
        reason="unsupported_in_remote_mode",
        subject=command,
    )


def _remote_add(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    body: _Document = {
        "content": _content_value(arguments),
        "memory_type": arguments.memory_type,
    }
    _put(body, "occurred_at", _remote_time(arguments.occurred_at, "occurred_at"))
    _put(body, "occurred_end", _remote_time(arguments.occurred_end, "occurred_end"))
    _put(body, "metadata", _metadata_value(_json_source(arguments.metadata)))
    _put(
        body,
        "context",
        _observation_context_document(_observation_context(_json_source(arguments.context))),
    )
    return "POST", "/v1/memories", body


def _remote_add_many(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    items = _jsonl(arguments.source)
    body: _Document = {
        "contents": [_remote_content(item["content"]) for item in items],
        "memory_type": arguments.memory_type,
    }
    for field in ("occurred_at", "occurred_end"):
        values = [_remote_time(item.get(field), field) for item in items]
        if any(value is not None for value in values):
            body[field] = values
    metadata = [_metadata_value(item.get("metadata")) for item in items]
    if any(value is not None for value in metadata):
        body["metadata"] = metadata
    contexts = [_observation_context(item.get("context")) for item in items]
    if any(value is not None for value in contexts):
        body["context"] = [_observation_context_document(value) for value in contexts]
    return "POST", "/v1/memories/batch", body


def _remote_settle(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    body: _Document = {"limit": arguments.limit, "max_attempts": arguments.max_attempts}
    _put(body, "memory_ids", list(arguments.memory_ids) or None)
    return "POST", "/v1/settle", body


def _remote_pending_captures(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    query = [("limit", arguments.limit), *(("memory_ids", value) for value in arguments.memory_ids)]
    return "GET", f"/v1/pending_captures?{urlencode(query)}", None


def _remote_register_identity(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    body: _Document = {"identity_id": arguments.identity_id, "name": arguments.name}
    _put(body, "relationship", arguments.relationship)
    return "POST", "/v1/identities", body


def _remote_record_consent(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    body: _Document = {"state": arguments.state}
    _put(body, "note", arguments.note)
    return "POST", f"/v1/identities/{_segment(arguments.identity_id)}/consent", body


def _segment(value: str) -> str:
    """Quote one ID as a whole path segment; / in an ID must not become a route boundary."""
    return quote(value, safe="")


def _remote_export(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    query = (
        [("identity_id", arguments.identity_id)]
        if arguments.identity_id is not None
        else [("memory_ids", value) for value in arguments.memory_ids]
    )
    return "GET", f"/v1/export?{urlencode(query)}", None


def _remote_query(field: str, arguments: argparse.Namespace) -> _Document:
    body: _Document = {field: _content_value(arguments), "limit": arguments.limit}
    _put(body, "memory_type", arguments.memory_type)
    _put(body, "reference_at", _remote_time(arguments.reference_at, "reference_at"))
    if field == "query":
        _put(body, "occurred_from", _remote_time(arguments.occurred_from, "occurred_from"))
        _put(body, "occurred_until", _remote_time(arguments.occurred_until, "occurred_until"))
    _put(
        body,
        "scope",
        _retrieval_scope_document(_retrieval_scope(_json_source(arguments.scope))),
    )
    return body


def _remote_compile(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    body: _Document = {
        "goal": _content_value(arguments),
        "budget": _budget(arguments).document(),
    }
    _put(body, "reference_at", _remote_time(arguments.reference_at, "reference_at"))
    _put(
        body,
        "scope",
        _retrieval_scope_document(_retrieval_scope(_json_source(arguments.scope))),
    )
    if arguments.allow_partial_sources:
        body["allow_partial_sources"] = True
    return "POST", "/v1/context", body


def _remote_list(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    query = {"limit": arguments.limit}
    if arguments.cursor is not None:
        # Opaque by contract: forwarded exactly as it was returned, never parsed.
        query["cursor"] = arguments.cursor
    return "GET", f"/v1/memories?{urlencode(query)}", None


def _remote_reinforce(arguments: argparse.Namespace) -> tuple[str, str, _Document | None]:
    return "POST", "/v1/memories/reinforce", {"memory_ids": list(arguments.memory_ids)}


# Every command a running owner serves over /v1, and the only place that set is written down:
# _run_remote refuses anything absent here. A command missing from it has no route at all, not
# a route the CLI forgot -- consolidate, apply, rollback, operations, and
# the rest of the control plane are SDK-only by design, and so are add-stream,
# reindex, and optimize.
_REMOTE: Mapping[str, _RemoteCommand] = {
    "add": _remote_add,
    "add-many": _remote_add_many,
    # The route takes the same body POST /v1/memories does, so the encoding is the same too.
    "capture": lambda a: ("POST", "/v1/capture", _remote_add(a)[2]),
    "settle": _remote_settle,
    "pending-captures": _remote_pending_captures,
    "search": lambda a: (
        "POST",
        "/v1/memories/search",
        {**_remote_query("query", a), "explain": a.explain},
    ),
    "ask": lambda a: ("POST", "/v1/answers", _remote_query("question", a)),
    "compile": _remote_compile,
    "get": lambda a: ("GET", f"/v1/memories/{_segment(a.memory_id)}", None),
    "perceive": lambda a: ("POST", "/v1/perception", {"memory_id": a.memory_id, "kinds": a.kind}),
    "register-identity": _remote_register_identity,
    "identity": lambda a: ("GET", f"/v1/identities/{_segment(a.identity_id)}", None),
    "record-consent": _remote_record_consent,
    "consent": lambda a: ("GET", f"/v1/identities/{_segment(a.identity_id)}/consent", None),
    "forget-identity": lambda a: ("DELETE", f"/v1/identities/{_segment(a.identity_id)}", None),
    "unlink-identity": lambda a: ("POST", f"/v1/identities/{_segment(a.alias_id)}/unlink", None),
    "reinforce": _remote_reinforce,
    "export": _remote_export,
    "apply-retention": lambda a: ("POST", "/v1/retention", {"dry_run": a.dry_run}),
    "list": _remote_list,
    "delete": lambda a: ("DELETE", f"/v1/memories/{_segment(a.memory_id)}", None),
}
# Which create_app switch registers each gated route, so a 404 from an owner that never opted
# in names the switch instead of reaching the caller as an unexplained exit 1.
_REMOTE_SWITCH: Mapping[str, str] = {
    "perceive": "embodied_operations",
    "register-identity": "identity_operations",
    "identity": "identity_operations",
    "record-consent": "identity_operations",
    "consent": "identity_operations",
    "forget-identity": "identity_operations",
    "unlink-identity": "identity_operations",
    "export": "identity_operations",
    "apply-retention": "identity_operations",
}


def _pending_captures_result(document: _Document) -> _Document:
    raw_items = document["items"]
    assert isinstance(raw_items, list)
    pending: list[object] = []
    for entry in raw_items:
        assert isinstance(entry, dict)
        pending.append({key: value for key, value in entry.items() if key != "cursor"})
    return {"pending": pending}


def _forget_identity_result(document: _Document) -> _Document:
    erasure = document["erasure"]
    assert isinstance(erasure, dict)
    return erasure


def _identity_result(document: _Document) -> _Document:
    return {"identity": _remote_profile(document["identity"])}


# The four responses REST shapes differently from the command's own document. A script must read
# one shape per command whichever transport ran it, so the transport translates back rather than
# printing a second vocabulary under --url.
_REMOTE_RESULT: Mapping[str, _RemoteResultTransform] = {
    "pending-captures": _pending_captures_result,
    "forget-identity": _forget_identity_result,
    "register-identity": lambda _document: {},
    "identity": _identity_result,
}


def _remote_profile(profile: object) -> _Document | None:
    """Project the owner's identity profile onto the three fields identity prints locally."""
    if not isinstance(profile, dict):
        return None
    return {name: profile.get(name) for name in ("identity_id", "name", "relationship")}


def _remote_doctor(arguments: argparse.Namespace) -> _Document:
    return {
        "version": _installed_version(),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "url": arguments.url,
        "health": request_json(arguments.url, "GET", "/healthz", None, _remote_timeout(arguments)),
        "composition": {"source": f"--url {arguments.url}", "owner": "remote"},
    }


def _remote_timeout(arguments: argparse.Namespace) -> float:
    return (
        _DEFAULT_REMOTE_TIMEOUT_SECONDS if arguments.timeout is None else float(arguments.timeout)
    )


# Input


def _reject_duplicate_content(arguments: argparse.Namespace) -> None:
    """One operand, one source. Preferring either would silently discard what the caller typed.

    Checked before the composition runs, so the refusal is identical locally and against --url
    and nothing is constructed or sent first.
    """
    if (
        arguments.command in _QUERY_METAVAR
        and arguments.content_json is not None
        and arguments.content
    ):
        raise ValidationError(
            "--content-json and positional content both supply this operand; pass one or the other",
            subject="--content-json",
        )


def _atoms(arguments: argparse.Namespace) -> tuple[_Atom, ...]:
    """Decode ordered content atoms without shell quoting tricks."""
    values: list[str] = list(arguments.content) or [_STDIN]
    atoms: list[_Atom] = []
    for value in values:
        if value == _STDIN:
            atoms.append(("text", _read_stdin()))
        elif value.startswith("@@"):
            atoms.append(("text", value[1:]))
        elif value.startswith("@"):
            atoms.append(("path", value[1:]))
        else:
            atoms.append(("text", value))
    return tuple(atoms)


def _content_input(arguments: argparse.Namespace) -> ContentInput:
    if arguments.content_json is not None:
        return _parts_input(_json_source(arguments.content_json))
    atoms: list[ContentAtom] = [
        value if kind == "text" else Path(value) for kind, value in _atoms(arguments)
    ]
    return atoms[0] if len(atoms) == 1 else tuple(atoms)


def _content_value(arguments: argparse.Namespace) -> object:
    """The content a /v1 request carries. Local paths deliberately never cross the wire."""
    if arguments.content_json is not None:
        return _remote_content(_json_source(arguments.content_json))
    atoms = _atoms(arguments)
    if any(kind == "path" for kind, _value in atoms):
        raise _CompositionError(
            "a local file path cannot be sent to a remote owner; pass base64 media in a data URL "
            "through --content-json, which REST accepts",
            reason="unsupported_in_remote_mode",
            subject="@PATH",
        )
    if len(atoms) == 1:
        return atoms[0][1]
    return [{"type": "input_text", "text": value} for _kind, value in atoms]


def _remote_content(value: object) -> object:
    """Refuse the one CLI-only part type on every path that reaches a running owner.

    add-many carries the same union one item per JSONL line, so it validates here too rather
    than forwarding a local path and letting the owner reject it as an unknown field.
    """
    items = value if isinstance(value, list) else []
    if any(isinstance(item, dict) and "path" in item for item in items):
        raise _CompositionError(
            "input_file.path is a local-mode part type and cannot be sent to a remote owner",
            reason="unsupported_in_remote_mode",
            subject="input_file.path",
        )
    return value


def _parts_input(value: object) -> ContentInput:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValidationError("content must be a string or a non-empty array of parts")
    return tuple(_part(item) for item in value)


def _part(part: object) -> ContentAtom:
    if not isinstance(part, dict):
        raise ValidationError("each content part must be an object")
    kind = part.get("type")
    if kind == "input_text":
        _fields(part, {"type", "text"})
        text = part.get("text")
        if not isinstance(text, str):
            raise ValidationError("input_text requires a text string")
        return text
    if kind == "input_image":
        _fields(part, {"type", "image_url", "file_id"})
        source, value = _one_source(part, ("image_url", "file_id"))
        if source == "file_id":
            return AssetRef(id=str(value), modality=Modality.IMAGE)
        return _data_blob(str(value), name=None, expected="image/*")
    if kind == "input_file":
        return _file_part(part)
    raise ValidationError(f"unknown content part type: {kind!r}")


def _file_part(part: Mapping[str, object]) -> ContentAtom:
    _fields(part, {"type", "file_url", "file_data", "file_id", "media_type", "filename", "path"})
    media_type = part.get("media_type")
    filename = part.get("filename")
    name = None if filename is None else str(filename)
    source, value = _one_source(part, ("file_url", "file_data", "file_id", "path"))
    if source == "path":
        # The one part type REST and MCP refuse: the CLI runs on the machine that owns the data
        # directory, and Memory already accepts a Path atom.
        return Path(str(value))
    if source == "file_id":
        return _asset_reference(str(value), media_type)
    if source == "file_data":
        if not isinstance(media_type, str) or media_type.endswith("/*"):
            raise ValidationError("file_data requires a concrete media_type")
        return Blob(data=_decode_base64(str(value)), media_type=media_type, name=name)
    return _data_blob(
        str(value), name=name, expected=None if media_type is None else str(media_type)
    )


def _asset_reference(file_id: str, media_type: object) -> AssetRef:
    if media_type is None:
        return AssetRef(id=file_id)
    text = str(media_type)
    if text.endswith("/*"):
        return AssetRef(id=file_id, modality=Modality(text.split("/", 1)[0]))
    return AssetRef(id=file_id, media_type=text)


def _fields(mapping: Mapping[str, object], accepted: set[str]) -> None:
    unexpected = sorted(set(mapping) - accepted)
    if unexpected:
        raise ValidationError(f"unexpected fields: {', '.join(unexpected)}")


def _one_source(mapping: Mapping[str, object], sources: tuple[str, ...]) -> tuple[str, object]:
    present = [name for name in sources if mapping.get(name) is not None]
    if len(present) != 1:
        raise ValidationError(f"exactly one of {', '.join(sources)} is required")
    return present[0], mapping[present[0]]


def _data_blob(value: str, *, name: str | None, expected: str | None) -> Blob:
    header, separator, payload = value.partition(",")
    if not value.startswith("data:") or not separator or not header.endswith(";base64"):
        raise ValidationError(
            "remote URLs are not accepted; supply base64 media in a data URL, or @PATH locally"
        )
    media_type = header.removeprefix("data:").removesuffix(";base64").lower()
    if not media_type or "/" not in media_type:
        raise ValidationError("data URL must declare a media type")
    if expected is not None and not (
        expected == media_type
        or (expected.endswith("/*") and expected.split("/", 1)[0] == media_type.split("/", 1)[0])
    ):
        raise ValidationError("media_type contradicts the data URL")
    return Blob(data=_decode_base64(payload), media_type=media_type, name=name)


def _decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValidationError("media bytes must be valid base64") from error


def _jsonl(source: str) -> tuple[Mapping[str, object], ...]:
    return tuple(_jsonl_stream(source))


def _jsonl_stream(source: str) -> Iterator[Mapping[str, object]]:
    found = False
    for number, line in enumerate(_source_lines(source), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as error:
            raise ValidationError(f"line {number} is not valid JSON") from error
        if not isinstance(item, dict) or "content" not in item:
            raise ValidationError(f"line {number} must be an object with a content field")
        _fields(item, {"content", "occurred_at", "occurred_end", "metadata", "context"})
        found = True
        yield item
    if not found:
        raise ValidationError("no memories were supplied")


def _source_lines(source: str) -> Iterable[str]:
    if source == _STDIN:
        yield from _claim_stdin()
        return
    if source.startswith("@"):
        try:
            with Path(source[1:]).open(encoding="utf-8") as stream:
                yield from stream
        except OSError as error:
            raise ValidationError(f"cannot read {source[1:]}: {error.strerror}") from None
        return
    yield from source.splitlines()


def _metadata_value(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValidationError("metadata must be a JSON object with string keys")
    return value


def _optional_json_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    return value


def _require_json_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    return value


def _require_json_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"{field} must be a finite number")
    return float(value)


def _optional_json_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _require_json_float(value, field)


def _observation_context(value: object) -> ObservationContext | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValidationError("context must be a JSON object")
    _fields(
        value,
        {"basis", "source_id", "confidence", "valid_from", "valid_until", "spatial"},
    )
    raw_basis = value.get("basis", EvidenceBasis.OBSERVATION)
    if isinstance(raw_basis, EvidenceBasis):
        basis = raw_basis
    elif isinstance(raw_basis, str):
        try:
            basis = EvidenceBasis(raw_basis)
        except ValueError:
            raise ValidationError("context.basis is invalid") from None
    else:
        raise ValidationError("context.basis is invalid")
    return ObservationContext(
        basis=basis,
        source_id=_optional_json_str(value.get("source_id"), "context.source_id"),
        confidence=_require_json_float(value.get("confidence", 1.0), "context.confidence"),
        valid_from=_optional_time(value.get("valid_from"), "context.valid_from"),
        valid_until=_optional_time(value.get("valid_until"), "context.valid_until"),
        spatial=_spatial_context(value.get("spatial")),
    )


def _retrieval_scope(value: object) -> RetrievalScope | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValidationError("scope must be a JSON object")
    _fields(value, {"valid_at", "known_at", "near", "radius_m", "place_id", "identity_id"})
    return RetrievalScope(
        valid_at=_optional_time(value.get("valid_at"), "scope.valid_at"),
        known_at=_optional_time(value.get("known_at"), "scope.known_at"),
        near=_spatial_context(value.get("near")),
        radius_m=_optional_json_float(value.get("radius_m"), "scope.radius_m"),
        place_id=_optional_json_str(value.get("place_id"), "scope.place_id"),
        identity_id=_optional_json_str(value.get("identity_id"), "scope.identity_id"),
    )


def _spatial_context(value: object) -> SpatialContext | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValidationError("spatial context must be a JSON object")
    _fields(
        value,
        {
            "frame_id",
            "anchor",
            "x",
            "y",
            "z",
            "orientation_xyzw",
            "position_uncertainty_m",
        },
    )
    raw_anchor = value.get("anchor")
    if isinstance(raw_anchor, SpatialAnchor):
        anchor = raw_anchor
    elif isinstance(raw_anchor, str):
        try:
            anchor = SpatialAnchor(raw_anchor)
        except ValueError:
            raise ValidationError("spatial anchor is invalid") from None
    else:
        raise ValidationError("spatial anchor is invalid")
    raw_orientation = value.get("orientation_xyzw")
    orientation: tuple[float, float, float, float] | None = None
    if raw_orientation is not None:
        if not isinstance(raw_orientation, list | tuple) or len(raw_orientation) != 4:
            raise ValidationError("spatial orientation_xyzw must contain four numbers")
        orientation = (
            _require_json_float(raw_orientation[0], "spatial orientation_xyzw"),
            _require_json_float(raw_orientation[1], "spatial orientation_xyzw"),
            _require_json_float(raw_orientation[2], "spatial orientation_xyzw"),
            _require_json_float(raw_orientation[3], "spatial orientation_xyzw"),
        )
    return SpatialContext(
        frame_id=_require_json_str(value.get("frame_id"), "spatial frame_id"),
        anchor=anchor,
        x=_require_json_float(value.get("x"), "spatial x"),
        y=_require_json_float(value.get("y"), "spatial y"),
        z=_require_json_float(value.get("z", 0.0), "spatial z"),
        orientation_xyzw=orientation,
        position_uncertainty_m=_optional_json_float(
            value.get("position_uncertainty_m"), "spatial position_uncertainty_m"
        ),
    )


def _observation_context_document(context: ObservationContext | None) -> _Document | None:
    # Every declared field but place_id, which the SDK reads from spatial on the way in and
    # never echoes back on this document.
    return (
        None
        if context is None
        else {name: value for name, value in _document(context).items() if name != "place_id"}
    )


def _retrieval_scope_document(scope: RetrievalScope | None) -> _Document | None:
    return None if scope is None else _document(scope)


def _json_source(value: str | None) -> object:
    if value is None:
        return None
    try:
        return json.loads(_read_source(value))
    except ValueError as error:
        raise ValidationError("value must be valid JSON") from error


def _read_source(value: str) -> str:
    if value == _STDIN:
        return _read_stdin()
    if value.startswith("@"):
        try:
            return Path(value[1:]).read_text(encoding="utf-8")
        except OSError as error:
            raise ValidationError(f"cannot read {value[1:]}: {error.strerror}") from None
    return value


def _read_stdin() -> str:
    return _claim_stdin().read()


def _claim_stdin() -> TextIO:
    global _stdin_consumed
    if _stdin_consumed:
        raise ValidationError("standard input can only be read once per invocation")
    if sys.stdin.isatty():
        raise ValidationError("no content was supplied and standard input is a terminal")
    _stdin_consumed = True
    return sys.stdin


def _optional_time(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be an ISO 8601 timestamp")
    text = value.strip()
    if text.endswith(("z", "Z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValidationError(f"{name} must be an ISO 8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{name} must include a timezone")
    return parsed


def _remote_time(value: object, name: str) -> str | None:
    parsed = _optional_time(value, name)
    return None if parsed is None else _encode_time(parsed)


def _put(body: _Document, name: str, value: object) -> None:
    if value is not None:
        body[name] = value


# Output
#
# Shared response models live in api.responses without importing FastAPI, so local and --url
# commands use the same field vocabulary on installations without a web framework.


def _memory_document(record: MemoryRecord | SearchHit) -> _Document:
    schema = (
        AffectCueResponse
        if isinstance(record, AffectCue)
        else SearchHitResponse
        if isinstance(record, SearchHit)
        else MemoryResponse
    )
    dumped: object = schema.model_validate(record).model_dump(mode="json")
    assert isinstance(dumped, dict)
    return dumped


def _encode_time(value: datetime) -> str:
    text = value.isoformat()
    return f"{text[:-6]}Z" if text.endswith("+00:00") else text


def _document(value: _DataclassInstance) -> _Document:
    """One JSON-native document for one frozen SDK value, field for declared field.

    asdict flattens the nested values; this only encodes what json cannot write itself, so a
    field added to an SDK value reaches every command that prints it without a line here naming
    it. The result stays JSON-native because it is also what a --url request body carries.
    """
    encoded = _json_native(asdict(value))
    assert isinstance(encoded, dict)
    return encoded


def _json_native(value: object) -> object:
    if isinstance(value, datetime):
        return _encode_time(value)
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _encode_optional_time(value: datetime | None) -> str | None:
    return None if value is None else _encode_time(value)


# Failure


def _fail(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    retryable: bool | None = None,
    stage: str | None = None,
    subject: str | None = None,
) -> int:
    envelope: _Document = {
        "code": code,
        "reason": reason,
        "retryable": SenseMemError(reason=reason).retryable if retryable is None else retryable,
        "stage": stage,
        # Unlike REST, the CLI runs as the invoking user on the machine that owns data_dir, so a
        # local path or a failing batch position is information the caller already holds.
        "subject": subject,
        "message": message,
        "trace_id": f"trace_{uuid4().hex}",
        "issues": [],
    }
    return _forward(envelope)


def _forward(envelope: _Document) -> int:
    print(json.dumps(envelope, ensure_ascii=False), file=sys.stderr)
    code = envelope.get("code")
    reason = envelope.get("reason")
    return _exit_code(
        code if isinstance(code, str) else "internal_error",
        reason if isinstance(reason, str) else None,
    )


def _exit_code(code: str, reason: str | None) -> int:
    if code == StorageError.code and reason == "data_dir_in_use":
        return DATA_DIR_IN_USE_EXIT_CODE
    return EXIT_CODES.get(code, 1)


# Parser
#
# Every default is read from the SDK signature it will be passed to, so --help shows the real
# value and the CLI cannot drift from it the way the REST adapter's list default once did.


def _default(operation: str, parameter: str) -> object:
    return inspect.signature(getattr(Memory, operation)).parameters[parameter].default


_MEMORY_DEFAULTS: Mapping[str, object] = {
    name: (
        getattr(MemorySettings(), _MEMORY_KEYWORDS.get(name, name))
        if name in _TUNING
        else _default("__init__", _MEMORY_KEYWORDS.get(name, name))
    )
    for name in ("data_dir", *_OPTIONAL_SLOTS, *_TUNING)
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Local multimodal memory operations over one composed Memory instance.",
        epilog="Data is JSON on stdout; diagnostics and failures are JSON on stderr.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"sensemem {_version()}")
    parser.add_argument(
        "--data-dir",
        default=_MEMORY_DEFAULTS["data_dir"],
        metavar="PATH",
        help="local memory directory (default: %(default)s)",
    )
    # Not required=True: a missing composition is reported by _describe_composition as exit 10, because an
    # agent has to tell "you never said which backend" from "you typed the flag wrong".
    composition = parser.add_mutually_exclusive_group()
    composition.add_argument("--app", metavar="MODULE:ATTR", help="an application-composed Memory")
    composition.add_argument("--embedder", metavar="NAME", help=f"one of: {_recipes()}")
    composition.add_argument("--url", metavar="URL", help="address a running owner over /v1")
    parser.add_argument(
        "--timeout",
        type=_positive_seconds,
        metavar="SECONDS",
        help=f"remote request timeout (default: {_DEFAULT_REMOTE_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument("--answerer", metavar="NAME", help="generation recipe, with --embedder")
    parser.add_argument("--former", metavar="NAME", help="formation recipe, with --embedder")
    parser.add_argument(
        "--consolidator", metavar="NAME", help="consolidation recipe, with --embedder"
    )
    parser.add_argument("--transcriber", metavar="NAME", help="speech recipe, with --embedder")
    parser.add_argument(
        "--vision", metavar="NAME", help="visual description recipe, with --embedder"
    )
    # Derived from the SDK default, never hardcoded: _reject_embedder_only_options compares this
    # against _MEMORY_DEFAULTS, so a literal here silently rejects every --app/--url invocation
    # the moment the SDK default moves. BooleanOptionalAction also supplies --no-index-speech.
    parser.add_argument(
        "--index-speech",
        action=argparse.BooleanOptionalAction,
        default=_MEMORY_DEFAULTS["index_speech"],
        help="index transcripts on add (default: %(default)s)",
    )
    parser.add_argument(
        "--minimum-relevance",
        type=float,
        metavar="FLOAT",
        default=_MEMORY_DEFAULTS["minimum_relevance"],
        help="weak-evidence floor (default: %(default)s)",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        metavar="FLOAT",
        default=_MEMORY_DEFAULTS["ambiguity_margin"],
        help="top-two gate when limit=1 (default: %(default)s)",
    )
    parser.add_argument(
        "--decay-half-life-days",
        type=float,
        metavar="FLOAT",
        default=_MEMORY_DEFAULTS["decay_half_life_days"],
        help="opt-in recency decay (default: %(default)s)",
    )
    parser.add_argument(
        "--explain",
        dest="explain_composition",
        action="store_true",
        help="print the composition and exit",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress the stderr banner")
    _commands(parser.add_subparsers(dest="command", required=True, metavar="COMMAND"))
    return parser


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive finite number") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return seconds


def _commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name, help_text in (
        ("add", "store one memory"),
        ("capture", "store one memory without model work; settle makes it searchable"),
    ):
        observation = _content_command(commands, name, help_text)
        observation.add_argument("--occurred-at", metavar="TIME", help="ISO 8601 event start")
        observation.add_argument("--occurred-end", metavar="TIME", help="ISO 8601 event end")
        observation.add_argument(
            "--metadata",
            metavar="JSON",
            help="application metadata object, @PATH, or -",
        )
        observation.add_argument(
            "--context",
            metavar="JSON",
            help="typed observation context, @PATH, or -",
        )
        _memory_type_option(observation, name)
    batch = commands.add_parser("add-many", help="store a JSONL batch in one transaction")
    batch.add_argument("source", nargs="?", default=_STDIN, metavar="JSONL", help="@PATH or -")
    _memory_type_option(batch, "add_many")
    stream = commands.add_parser("add-stream", help="store completed JSONL items incrementally")
    stream.add_argument("source", nargs="?", default=_STDIN, metavar="JSONL", help="@PATH or -")
    stream.add_argument(
        "--capture",
        action="store_true",
        help="commit each item through capture; settle makes them searchable",
    )
    _memory_type_option(stream, "add")
    for operation in ("search", "ask"):
        name = operation.replace("_", "-")
        command = _content_command(commands, name, f"{name} memories")
        command.add_argument(
            "--limit",
            type=int,
            default=_default(operation, "limit"),
            help="maximum hits (default: %(default)s)",
        )
        _memory_type_option(command, operation)
        command.add_argument("--reference-at", metavar="TIME", help="retrieval reference clock")
        command.add_argument("--scope", metavar="JSON", help="temporal/spatial scope, @PATH, or -")
        if operation != "ask":
            command.add_argument(
                "--explain", action="store_true", help="include retrieval diagnostics"
            )
            command.add_argument("--occurred-from", metavar="TIME", help="event overlap start")
            command.add_argument("--occurred-until", metavar="TIME", help="event overlap end")
    _compile_command(commands)
    for name, help_text in (
        ("get", "read one memory"),
        ("perceive", "recognize speech and faces with configured plugins"),
        ("delete", "delete one memory"),
    ):
        commands.add_parser(name, help=help_text).add_argument("memory_id", metavar="MEMORY_ID")
    perception = commands.choices["perceive"]
    perception.add_argument("--kind", action="append", choices=("speech", "faces"))
    identity = commands.add_parser("register-identity", help="name one face/voice identity")
    identity.add_argument("identity_id", metavar="IDENTITY_ID")
    identity.add_argument("name", metavar="NAME")
    identity.add_argument("--relationship", help="how this person relates to the owner")
    profile = commands.add_parser("identity", help="read one identity's name and relationship")
    profile.add_argument("identity_id", metavar="IDENTITY_ID")
    forget = commands.add_parser(
        "forget-identity",
        help="erase a person: their face and voice templates, aliases, and indexed name",
    )
    forget.add_argument("identity_id", metavar="IDENTITY_ID")
    unlink = commands.add_parser("unlink-identity", help="reverse one face/voice merge")
    unlink.add_argument("alias_id", metavar="ALIAS_ID")
    consent = commands.add_parser(
        "record-consent",
        help="record what one person said about being processed as a recognized person",
    )
    consent.add_argument("identity_id", metavar="IDENTITY_ID")
    consent.add_argument("state", choices=[item.value for item in ConsentState])
    consent.add_argument("--note", help="the subject's own words about their decision")
    stated = commands.add_parser("consent", help="read one identity's standing consent state")
    stated.add_argument("identity_id", metavar="IDENTITY_ID")
    export = commands.add_parser(
        "export",
        help="export everything held about one data subject, media by reference",
    )
    subject = export.add_mutually_exclusive_group(required=True)
    subject.add_argument("--identity-id", metavar="IDENTITY_ID", help="one recognized person")
    subject.add_argument(
        "--memory-id",
        dest="memory_ids",
        action="append",
        metavar="MEMORY_ID",
        help="repeatable; export these records instead of one person",
    )
    retention = commands.add_parser(
        "apply-retention",
        help="delete what the declared retention policy says has outlived its purpose",
    )
    retention.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be deleted and delete nothing",
    )
    reinforce = commands.add_parser("reinforce", help="record positive feedback")
    reinforce.add_argument("memory_ids", nargs="+", metavar="MEMORY_ID")
    candidates = commands.add_parser(
        "consolidation-candidates",
        help="ask what needs deliberation, derived from recorded evidence and feedback",
    )
    candidates.add_argument(
        "--limit",
        type=int,
        default=_default("consolidation_candidates", "limit"),
        help="candidate rows (default: %(default)s)",
    )
    candidates.add_argument(
        "--idle",
        action="store_true",
        help="declare an approved idle window, admitting never-weighed lineages",
    )
    consolidate = _content_command(
        commands, "consolidate", "deliberate over evidence and apply memory operations"
    )
    consolidate.add_argument(
        "--evidence-id", dest="evidence_ids", action="append", metavar="MEMORY_ID"
    )
    consolidate.add_argument(
        "--limit",
        type=int,
        default=_default("consolidate", "limit"),
        help="evidence set size (default: %(default)s)",
    )
    consolidate.add_argument(
        "--trigger",
        choices=[item.value for item in MemoryTrigger],
        default=MemoryTrigger(_default("consolidate", "trigger")).value,
    )
    pending = commands.add_parser(
        "consolidate-pending", help="process due consolidation candidates"
    )
    pending.add_argument("--limit", type=int, default=_default("consolidate_pending", "limit"))
    pending.add_argument(
        "--max-rounds", type=int, default=_default("consolidate_pending", "max_rounds")
    )
    pending.add_argument("--idle", action="store_true", help="declare an approved idle window")
    applying = commands.add_parser(
        "apply",
        help="apply one host-supplied memory operation through the kernel, as replay does",
    )
    applying.add_argument(
        "--operation",
        required=True,
        metavar="JSON",
        help="one complete current operation object as `operations` logs it, @PATH, or -",
    )
    outcome = commands.add_parser(
        "record-outcome",
        help="record what later evidence said about one logged operation",
    )
    outcome.add_argument("operation_id", type=int, metavar="OPERATION_ID")
    outcome.add_argument("outcome", choices=[item.value for item in MemoryOutcome])
    outcome.add_argument("--note", help="why later evidence confirmed or refuted it")
    forget = commands.add_parser("forget", help="cognitively forget memories without deleting")
    forget.add_argument("memory_ids", nargs="+", metavar="MEMORY_ID")
    rollback = commands.add_parser("rollback", help="reverse one logged memory operation")
    rollback.add_argument("operation_id", type=int, metavar="OPERATION_ID")
    operations = commands.add_parser("operations", help="list logged memory operations")
    operations.add_argument(
        "--limit",
        type=int,
        default=_default("operations", "limit"),
        help="page size (default: %(default)s)",
    )
    listing = commands.add_parser("list", help="list newest memories")
    listing.add_argument(
        "--limit",
        type=int,
        default=_default("list", "limit"),
        help="page size (default: %(default)s)",
    )
    listing.add_argument("--cursor", help="opaque cursor from a previous page")
    settle = commands.add_parser("settle", help="enrich and index captured memories")
    settle.add_argument(
        "--limit",
        type=int,
        default=_default("settle", "limit"),
        help="maximum captured memories to settle (default: %(default)s)",
    )
    settle.add_argument(
        "--max-attempts",
        type=int,
        default=_default("settle", "max_attempts"),
        help="skip captures that already failed this often (default: %(default)s)",
    )
    settle.add_argument(
        "memory_ids",
        nargs="*",
        metavar="MEMORY_ID",
        help="settle only these memories, ignoring --max-attempts",
    )
    pending = commands.add_parser(
        "pending-captures",
        help="list captured memories waiting to be settled",
    )
    pending.add_argument(
        "--limit",
        type=int,
        default=_default("pending_captures", "limit"),
        help="maximum queued captures to report (default: %(default)s)",
    )
    pending.add_argument(
        "memory_ids",
        nargs="*",
        metavar="MEMORY_ID",
        help="report only these memories; absent from the result means not pending",
    )
    for name, help_text in (
        ("reindex", "rebuild the search index from SQLite"),
        ("optimize", "merge staged index vectors"),
        (DOCTOR, "resolve the composition and exercise each loader"),
    ):
        commands.add_parser(name, help=help_text)


def _compile_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add compile, whose options are the ContextBudget fields plus retrieval scope."""
    budget = ContextBudget()
    command = _content_command(commands, "compile", "compile a bounded context bundle")
    command.add_argument(
        "--max-chars",
        type=int,
        default=budget.max_chars,
        help="evidence character budget (default: %(default)s)",
    )
    command.add_argument(
        "--max-items",
        type=int,
        default=budget.max_items,
        help="maximum included memories (default: %(default)s)",
    )
    command.add_argument(
        "--max-media-items",
        type=int,
        help="maximum grounded media parts; 0 compiles a text-only bundle",
    )
    command.add_argument(
        "--memory-type",
        action="append",
        choices=[item.value for item in MemoryCategory],
        help="repeatable; keep only these memory types",
    )
    command.add_argument(
        "--min-confidence",
        type=float,
        default=budget.min_confidence,
        help="minimum typed confidence (default: %(default)s)",
    )
    command.add_argument(
        "--freshness-seconds",
        type=float,
        help="keep only memories anchored within this many seconds of the reference clock",
    )
    command.add_argument(
        "--max-latency-ms",
        type=int,
        help="deadline after which optional compilation stages are skipped, in milliseconds",
    )
    command.add_argument("--reference-at", metavar="TIME", help="retrieval reference clock")
    command.add_argument(
        "--selected-proofs",
        action="store_true",
        help="deliver complete count and confidence certificates instead of every alternative source",
    )
    command.add_argument("--scope", metavar="JSON", help="temporal/spatial scope, @PATH, or -")
    command.add_argument(
        "--allow-partial-sources",
        action="store_true",
        help="allow verified partial raw-text sources when a complete record does not fit",
    )


def _content_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> argparse.ArgumentParser:
    command = commands.add_parser(name, help=help_text)
    command.add_argument(
        "content",
        nargs="*",
        metavar=_QUERY_METAVAR[name],
        help="text, @PATH for a local file, @@TEXT for a literal @, - for stdin",
    )
    command.add_argument("--content-json", metavar="VALUE", help="REST parts array, @PATH, or -")
    return command


def _memory_type_option(command: argparse.ArgumentParser, operation: str) -> None:
    default = _default(operation, "memory_type")
    command.add_argument(
        "--memory-type",
        choices=[item.value for item in MemoryCategory],
        default=None if default is None else MemoryCategory(default).value,
    )


def _recipes() -> str:
    return ", ".join(recipes.names())


def _version() -> str:
    try:
        return _installed_version()
    except PackageNotFoundError:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
