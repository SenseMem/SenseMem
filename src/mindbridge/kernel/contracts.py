"""Backend contracts and capability routing.

A backend declares atomic modalities and a space; this module validates the declaration once at
wiring time, folds it into an immutable `Backends` value, and answers the routing questions the
planes ask against declared capability rather than provider names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from mindbridge._resources import Closable, present_resources, unique_resources
from mindbridge._validation import nonempty_text
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.kernel.content import PreparedContent, prepared_modalities
from mindbridge.kernel.validation import positive_int
from mindbridge.models.base import (
    ConsolidationBackend,
    EmbeddingBackend,
    FaceBackend,
    FormationBackend,
    GenerationBackend,
    GenerationRoutingBackend,
    RerankingBackend,
    SpeechBackend,
    TranscriptionBackend,
    VisionDescriptionBackend,
    _modalities,
    embedding_routes,
)
from mindbridge.types import MemoryCapabilities, Modality

_LOGGER = logging.getLogger(__name__)


_NO_TRANSCRIPTION_SPACE = "none:asr-v1"


_NO_FACE_SPACE = "none:face-v1"


def fallback_unsupported(
    prepared: PreparedContent,
    supported: frozenset[Modality],
    operation: str,
    *,
    rescuable: frozenset[Modality] = frozenset(),
) -> frozenset[Modality]:
    unsupported = prepared_modalities(prepared) - supported
    fallback = {Modality.AUDIO}
    if (
        any(asset.transcript is not None for asset in prepared.assets)
        or "[speech identities:" in prepared.text
    ):
        fallback.add(Modality.VIDEO)
    if prepared.visual_description:
        fallback.update((Modality.IMAGE, Modality.VIDEO))
    # This runs before any derived text exists, so it cannot be seen yet -- only its possibility.
    # `rescuable` is the modality set a transcript or a visual description could still rescue,
    # which is why a video reaching a video-less embedder is no longer fatal when the transcriber
    # can read it. Empty by default, so a caller that does not pass it keeps the strict behaviour.
    fallback.update(rescuable)
    fatal = set(unsupported - fallback)
    if Modality.TEXT not in supported:
        fatal.update(unsupported & fallback)
    if fatal:
        names = ", ".join(sorted(modality.value for modality in fatal))
        raise ModelError(
            f"configured {operation} model does not support: {names}",
            reason="unsupported_modality",
        )
    if unsupported:
        # Not fatal, but the whole capability of this modality now rides on a transcript or a
        # description that may never arrive. Silent fallback is how media encoding has died
        # unnoticed here before.
        _LOGGER.warning(
            "configured %s model does not support %s; falling back to derived text",
            operation,
            ", ".join(sorted(modality.value for modality in unsupported)),
        )
    return unsupported


def _transcription_contract(
    transcriber: SpeechBackend | TranscriptionBackend | None,
) -> tuple[frozenset[Modality], str]:
    if transcriber is None:
        return frozenset(), _NO_TRANSCRIPTION_SPACE
    return _modalities(transcriber.transcription_capabilities, "transcription"), nonempty_text(
        transcriber.transcription_space,
        "transcription space",
    )


def _generation_contract(answerer: GenerationBackend | None) -> frozenset[Modality]:
    return (
        frozenset()
        if answerer is None
        else _modalities(answerer.generation_capabilities, "generation")
    )


def _vision_contract(
    describer: VisionDescriptionBackend | None,
) -> tuple[frozenset[Modality], str, str]:
    if describer is None:
        return frozenset(), "none", "none"
    capabilities = _modalities(describer.vision_capabilities, "vision")
    if not capabilities or capabilities - {Modality.IMAGE, Modality.VIDEO}:
        raise ValidationError("vision capabilities must contain image or video")
    return (
        capabilities,
        nonempty_text(describer.vision_model, "vision model"),
        nonempty_text(describer.vision_space, "vision space") + ":typed-v1",
    )


def _formation_contract(
    former: FormationBackend | None,
) -> tuple[frozenset[Modality], str, str]:
    if former is None:
        return frozenset(), "none", "none"
    capabilities = _modalities(former.formation_capabilities, "formation")
    if not capabilities:
        raise ValidationError("formation capabilities must not be empty")
    return (
        capabilities,
        nonempty_text(former.formation_model, "formation model"),
        nonempty_text(former.formation_space, "formation space"),
    )


def _consolidation_contract(
    consolidator: ConsolidationBackend | None,
) -> tuple[str, str]:
    if consolidator is None:
        return "none", "none"
    return (
        nonempty_text(consolidator.consolidation_model, "consolidation model"),
        nonempty_text(consolidator.consolidation_recipe, "consolidation recipe"),
    )


def _face_contract(
    analyzer: FaceBackend | None,
) -> tuple[frozenset[Modality], str, str, str]:
    if analyzer is None:
        return frozenset(), "none", _NO_FACE_SPACE, _NO_FACE_SPACE
    capabilities = _modalities(analyzer.face_capabilities, "face")
    if not capabilities or capabilities - {Modality.IMAGE, Modality.VIDEO}:
        raise ValidationError("face capabilities must contain image or video")
    return (
        capabilities,
        nonempty_text(analyzer.face_model, "face model"),
        _embedding_space(analyzer.face_space),
        _embedding_space(analyzer.face_analysis_space),
    )


def _embedding_contract(
    embedder: EmbeddingBackend,
) -> tuple[frozenset[Modality], str, str, int]:
    return (
        _modalities(embedder.embedding_capabilities, "embedding"),
        nonempty_text(embedder.embedding_model, "embedding model"),
        _embedding_space(embedder.embedding_space),
        positive_int(embedder.embedding_dimension, "embedder.embedding_dimension"),
    )


def _embedding_space(value: object) -> str:
    space = nonempty_text(value, "embedding space")
    if "'" in space or "\\" in space or any(ord(character) < 32 for character in space):
        raise ValidationError("embedding space contains characters unsupported by Zvec")
    return space


def declared_capabilities(
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    vision_describer: VisionDescriptionBackend | None = None,
    face_analyzer: FaceBackend | None = None,
    former: FormationBackend | None = None,
    consolidator: ConsolidationBackend | None = None,
    reranker: RerankingBackend | None = None,
) -> MemoryCapabilities:
    """Use the ordinary assembly boundary to inspect plugins without opening storage."""
    return resolve_backends(
        embedder=embedder,
        answerer=answerer,
        transcriber=transcriber,
        vision_describer=vision_describer,
        face_analyzer=face_analyzer,
        former=former,
        consolidator=consolidator,
        reranker=reranker,
    ).capabilities


@dataclass(frozen=True, slots=True)
class Backends:
    """The injected model backends and the contracts they declared, validated once."""

    embedder: EmbeddingBackend
    answerer: GenerationBackend | None
    transcriber: SpeechBackend | TranscriptionBackend | None
    vision_describer: VisionDescriptionBackend | None
    face_analyzer: FaceBackend | None
    former: FormationBackend | None
    consolidator: ConsolidationBackend | None
    embedding_capabilities: frozenset[Modality]
    embedding_model: str
    space_id: str
    embedding_dimension: int
    generation_capabilities: frozenset[Modality]
    transcription_capabilities: frozenset[Modality]
    transcription_space: str
    vision_capabilities: frozenset[Modality]
    vision_model: str
    vision_space: str
    face_capabilities: frozenset[Modality]
    face_model: str
    face_space: str
    face_analysis_space: str
    formation_capabilities: frozenset[Modality]
    formation_model: str
    formation_space: str
    consolidation_model: str
    consolidation_recipe: str
    reranker: RerankingBackend | None = None
    reranking_capabilities: frozenset[Modality] = frozenset()

    capabilities: MemoryCapabilities = field(init=False)

    def __post_init__(self) -> None:
        native_embedding = embedding_routes(self.embedder, self.embedding_capabilities)
        if self.embedding_capabilities and not native_embedding:
            raise ValidationError("embedding routes must not be empty for a configured embedder")
        routes = (
            self.answerer.generation_routes
            if isinstance(self.answerer, GenerationRoutingBackend)
            else (self.generation_capabilities,)
            if self.answerer is not None
            else ()
        )
        if self.generation_capabilities and not routes:
            raise ValidationError("generation routes must not be empty for a configured answerer")
        object.__setattr__(
            self,
            "capabilities",
            MemoryCapabilities(
                embedding=self.embedding_capabilities,
                embedding_model=self.embedding_model,
                embedding_space=self.space_id,
                embedding_dimension=self.embedding_dimension,
                embedding_routes=native_embedding,
                generation=self.generation_capabilities,
                generation_routes=routes,
                transcription=self.transcription_capabilities,
                vision=self.vision_capabilities,
                face=self.face_capabilities,
                formation=self.formation_capabilities,
                reranking=self.reranking_capabilities,
                reranking_model=None if self.reranker is None else self.reranker.reranking_model,
                generation_model=getattr(self.answerer, "generation_model", None),
                transcription_space=None if self.transcriber is None else self.transcription_space,
                vision_model=None if self.vision_describer is None else self.vision_model,
                face_model=None if self.face_analyzer is None else self.face_model,
                formation_model=None if self.former is None else self.formation_model,
                consolidation_model=None if self.consolidator is None else self.consolidation_model,
                speaker_recognition=isinstance(self.transcriber, SpeechBackend),
            ),
        )

    def resources(self) -> tuple[Closable, ...]:
        """Every distinct backend that may hold a client, in closing order."""
        return unique_resources(
            present_resources(
                self.embedder,
                self.transcriber,
                self.vision_describer,
                self.face_analyzer,
                self.former,
                self.consolidator,
                self.answerer,
                self.reranker,
            )
        )


def resolve_backends(
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None,
    transcriber: SpeechBackend | TranscriptionBackend | None,
    vision_describer: VisionDescriptionBackend | None,
    face_analyzer: FaceBackend | None,
    former: FormationBackend | None,
    consolidator: ConsolidationBackend | None,
    reranker: RerankingBackend | None = None,
) -> Backends:
    """Validate each backend's declared contract and fold the results into one value."""
    for name, backend, protocols in (
        ("embedder", embedder, (EmbeddingBackend,)),
        ("answerer", answerer, (GenerationBackend,)),
        ("transcriber", transcriber, (SpeechBackend, TranscriptionBackend)),
        ("vision_describer", vision_describer, (VisionDescriptionBackend,)),
        ("face_analyzer", face_analyzer, (FaceBackend,)),
        ("former", former, (FormationBackend,)),
        ("consolidator", consolidator, (ConsolidationBackend,)),
        ("reranker", reranker, (RerankingBackend,)),
    ):
        if backend is None and name != "embedder":
            continue
        if not isinstance(backend, protocols):
            expected = " or ".join(protocol.__name__ for protocol in protocols)
            raise ValidationError(f"plugins.{name} must implement {expected}")
    if reranker is not None:
        nonempty_text(reranker.reranking_model, "reranking model")
    reranking_capabilities = (
        frozenset()
        if reranker is None
        else _modalities(reranker.reranking_capabilities, "reranking")
    )
    embedding_capabilities, embedding_model, space_id, embedding_dimension = _embedding_contract(
        embedder
    )
    generation_capabilities = _generation_contract(answerer)
    transcription_capabilities, transcription_space = _transcription_contract(transcriber)
    vision_capabilities, vision_model, vision_space = _vision_contract(vision_describer)
    face_capabilities, face_model, face_space, face_analysis_space = _face_contract(face_analyzer)
    formation_capabilities, formation_model, formation_space = _formation_contract(former)
    consolidation_model, consolidation_recipe = _consolidation_contract(consolidator)
    return Backends(
        embedder=embedder,
        answerer=answerer,
        transcriber=transcriber,
        vision_describer=vision_describer,
        face_analyzer=face_analyzer,
        former=former,
        consolidator=consolidator,
        reranker=reranker,
        reranking_capabilities=reranking_capabilities,
        embedding_capabilities=embedding_capabilities,
        embedding_model=embedding_model,
        space_id=space_id,
        embedding_dimension=embedding_dimension,
        generation_capabilities=generation_capabilities,
        transcription_capabilities=transcription_capabilities,
        transcription_space=transcription_space,
        vision_capabilities=vision_capabilities,
        vision_model=vision_model,
        vision_space=vision_space,
        face_capabilities=face_capabilities,
        face_model=face_model,
        face_space=face_space,
        face_analysis_space=face_analysis_space,
        formation_capabilities=formation_capabilities,
        formation_model=formation_model,
        formation_space=formation_space,
        consolidation_model=consolidation_model,
        consolidation_recipe=consolidation_recipe,
    )
