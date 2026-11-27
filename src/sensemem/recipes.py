"""Named recipes over the model backends this package bundles.

A recipe is a name for a construction the caller could equally write by hand. It is not a plugin
system and not a registry: the table is closed, every entry pins its model identity to a constant
that already lives in this package, and each function returns the constructed object so the caller
owns it, can inspect it, and can throw it away. A backend this package does not bundle is
constructed by the application and passed to Memory directly.

load=True additionally exercises the backend's loader before returning, which is how
sensemem doctor turns a missing runtime into one line instead of a silent ingestion failure.
How deep that probe reaches depends on what each backend publishes; probe() reports which.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Literal, Protocol, overload, runtime_checkable

from sensemem.exceptions import ValidationError
from sensemem.models._openai_defaults import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
)
from sensemem.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FormationBackend,
    GenerationBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from sensemem.models.funasr import (
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_RECIPE,
    FunASRTranscriber,
)
from sensemem.models.openai_completion import OpenAICompletion
from sensemem.models.openai_embedding import OpenAIEmbedder
from sensemem.models.openai_speech import OpenAITranscriber
from sensemem.types import Modality

Slot = Literal["embedder", "answerer", "former", "consolidator", "transcriber", "vision"]


@runtime_checkable
class _SdkClient(Protocol):
    def close(self) -> None: ...


FUNASR = "funasr"
OPENAI = "openai"

# Which slots each recipe may fill. Only openai accepts an openai:<model> suffix, because it
# is the only bundled backend whose model is an argument rather than a pinned local revision.
_SLOTS: Mapping[str, frozenset[str]] = {
    FUNASR: frozenset({"transcriber"}),
    OPENAI: frozenset({"embedder", "answerer", "former", "consolidator", "transcriber", "vision"}),
}
_PARAMETERIZED = frozenset({OPENAI})
_PROBES: Mapping[str, str] = {FUNASR: "import", OPENAI: "client"}
_OPENAI_MODELS: Mapping[str, str] = {
    "embedder": DEFAULT_EMBEDDING_MODEL,
    "answerer": DEFAULT_GENERATION_MODEL,
    # Formation proposes typed memories with the generation model, so it pins the same constant.
    "former": DEFAULT_GENERATION_MODEL,
    # So does the memory-management loop; both reason with the same completion controls.
    "consolidator": DEFAULT_GENERATION_MODEL,
    "transcriber": DEFAULT_TRANSCRIPTION_MODEL,
    # Captioning is a chat completion over image parts, so it pins the generation constant too.
    "vision": DEFAULT_GENERATION_MODEL,
}
# SenseMem never reads a credential. The official SDK performs its own documented lookup, so the
# source of the key is reportable while the value never enters this process's output.
OPENAI_CREDENTIAL = "OPENAI_API_KEY (openai SDK default)"


def names() -> tuple[str, ...]:
    """Every recipe name. openai also accepts an openai:<model> suffix."""
    return tuple(sorted(_SLOTS))


def slots(name: str) -> tuple[str, ...]:
    """The composition slots one recipe may fill."""
    family, _model = _split(name)
    return tuple(sorted(_SLOTS[family]))


def require_slot(name: str, slot: str) -> None:
    """Raise unless one recipe can fill the named composition slot."""
    family, _model = _split(name)
    if slot not in _SLOTS[family]:
        accepted = ", ".join(f"--{value}" for value in slots(name))
        raise ValidationError(f"recipe {name} cannot fill --{slot}; it fills: {accepted}")


def probe(name: str) -> str:
    """What load=True exercises: a runtime import or a client."""
    family, _model = _split(name)
    return _PROBES[family]


def describe(name: str) -> dict[str, object]:
    """Static identity of one recipe. Resolves nothing and constructs nothing."""
    family, model = _split(name)
    if family == FUNASR:
        return {
            "recipe": name,
            "classes": {"transcriber": _qualified(FunASRTranscriber)},
            "slots": list(slots(name)),
            "models": {"transcriber": DEFAULT_FUNASR_MODEL_ID},
            "revision": DEFAULT_FUNASR_RECIPE.model_revision,
            "extra": "local",
        }
    return {
        "recipe": name,
        "classes": {
            slot: _qualified(
                OpenAIEmbedder
                if slot == "embedder"
                else OpenAITranscriber
                if slot == "transcriber"
                else OpenAICompletion
            )
            for slot in _OPENAI_MODELS
        },
        "slots": list(slots(name)),
        "models": {
            slot: default if model is None else model for slot, default in _OPENAI_MODELS.items()
        },
        "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
        "credential": OPENAI_CREDENTIAL,
        "extra": "openai",
    }


def embedder(name: str, *, load: bool = False) -> EmbeddingBackend:
    """Return the embedding backend one recipe names; the caller owns and closes it."""
    _family, model = _resolve_slot(name, "embedder")
    selected = _OPENAI_MODELS["embedder"] if model is None else model
    return _owned_openai(OpenAIEmbedder, embedding_model=selected)


def answerer(name: str, *, load: bool = False) -> GenerationBackend:
    """Return the generation backend one recipe names; the caller owns and closes it."""
    _family, model = _resolve_slot(name, "answerer")
    selected = _OPENAI_MODELS["answerer"] if model is None else model
    return _owned_openai(OpenAICompletion, generation_model=selected)


def former(name: str, *, load: bool = False) -> FormationBackend:
    """Return the formation backend one recipe names; the caller owns and closes it."""
    _family, model = _resolve_slot(name, "former")
    selected = _OPENAI_MODELS["former"] if model is None else model
    return _owned_openai(OpenAICompletion, generation_model=selected)


def consolidator(name: str, *, load: bool = False) -> ConsolidationBackend:
    """Return the consolidation backend one recipe names; the caller owns and closes it."""
    _family, model = _resolve_slot(name, "consolidator")
    selected = _OPENAI_MODELS["consolidator"] if model is None else model
    return _owned_openai(OpenAICompletion, generation_model=selected)


def vision(name: str, *, load: bool = False) -> VisionDescriptionBackend:
    """Return the visual-description backend one recipe names; the caller owns and closes it."""
    _family, model = _resolve_slot(name, "vision")
    selected = _OPENAI_MODELS["vision"] if model is None else model
    # vision_capabilities are the generation ones, and Memory rejects a describer that declares
    # anything but image or video. Image alone is the declarative default too: video is captioned
    # from four decoded stills, so it costs four image parts per memory.
    return _owned_openai(
        OpenAICompletion,
        generation_model=selected,
        generation_capabilities=frozenset({Modality.IMAGE}),
    )


def transcriber(name: str, *, load: bool = False) -> SpeechBackend | TranscriptionBackend:
    """Return the speech backend one recipe names; the caller owns and closes it."""
    family, model = _resolve_slot(name, "transcriber")
    if family == FUNASR:
        if load:
            # FunASR publishes no loader, so the deepest honest probe is the deferred import an
            # under-declared dependency breaks. probe() reports that, so nothing overstates it.
            import_module("funasr")
        return FunASRTranscriber(DEFAULT_FUNASR_RECIPE)
    selected = _OPENAI_MODELS["transcriber"] if model is None else model
    return _owned_openai(OpenAITranscriber, transcription_model=selected)


# What the SDK client takes rather than the adapter; everything else is an adapter control.
_CONNECTION = ("base_url", "api_key", "timeout", "max_retries")


@overload
def _owned_openai(backend_type: type[OpenAIEmbedder], **controls: object) -> OpenAIEmbedder: ...


@overload
def _owned_openai(backend_type: type[OpenAICompletion], **controls: object) -> OpenAICompletion: ...


@overload
def _owned_openai(
    backend_type: type[OpenAITranscriber], **controls: object
) -> OpenAITranscriber: ...


def _owned_openai(
    backend_type: type[OpenAIEmbedder] | type[OpenAICompletion] | type[OpenAITranscriber],
    **controls: object,
) -> OpenAIEmbedder | OpenAICompletion | OpenAITranscriber:
    """Build one request adapter and transfer the created client's lifetime to it."""
    connection: dict[str, object] = {}
    for name in _CONNECTION:
        value = controls.pop(name, None)
        if value is not None:
            connection[name] = value
    client = _openai_client(**connection)
    try:
        constructor: object = backend_type
        assert callable(constructor)
        built = constructor(client, owns_client=True, **controls)
        assert isinstance(built, (OpenAIEmbedder, OpenAICompletion, OpenAITranscriber))
        return built
    except BaseException:
        client.close()
        raise


def _resolve_slot(name: str, slot: Slot) -> tuple[str, str | None]:
    family, model = _split(name)
    require_slot(name, slot)
    return family, model


def _openai_client(**config: object) -> _SdkClient:
    factory: object = import_module("openai").OpenAI
    assert callable(factory)
    client = factory(**config)
    assert isinstance(client, _SdkClient)
    return client


def _split(name: str) -> tuple[str, str | None]:
    family, separator, model = name.partition(":")
    if family not in _SLOTS or (separator and family not in _PARAMETERIZED):
        raise ValidationError(f"unknown recipe: {name}; known recipes: {', '.join(names())}")
    if separator and not model.strip():
        raise ValidationError(f"recipe {family} requires a model identifier after the colon")
    return family, model.strip() or None


def _qualified(value: type[object]) -> str:
    return f"{value.__module__}.{value.__qualname__}"
