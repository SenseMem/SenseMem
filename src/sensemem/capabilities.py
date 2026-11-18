"""Capabilities values and pure operations for the public SDK."""

from __future__ import annotations

from dataclasses import dataclass, fields
from itertools import combinations

from sensemem._validation import positive_int
from sensemem._value_validation import _text
from sensemem.exceptions import ValidationError
from sensemem.types import Modality


@dataclass(frozen=True, slots=True)
class EmbeddingRoute:
    """One exact native input combination, with its repeated-media constraint."""

    modalities: frozenset[Modality]
    max_assets_per_modality: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.modalities, frozenset)
            or not self.modalities
            or any(
                not isinstance(value, Modality) or value is Modality.OMNI
                for value in self.modalities
            )
        ):
            raise ValidationError("embedding route must contain atomic modalities")
        if self.max_assets_per_modality is not None:
            positive_int(self.max_assets_per_modality, "max_assets_per_modality")

    @classmethod
    def combinations(
        cls, modalities: frozenset[Modality], *, max_assets_per_modality: int | None = None
    ) -> tuple[EmbeddingRoute, ...]:
        """Declare every non-empty subset when a backend has no joint-input restrictions."""
        ordered = sorted(modalities, key=lambda value: value.value)
        return tuple(
            cls(frozenset(values), max_assets_per_modality=max_assets_per_modality)
            for size in range(1, len(ordered) + 1)
            for values in combinations(ordered, size)
        )

    def supports(self, modalities: frozenset[Modality], *, assets_per_modality: int = 1) -> bool:
        """Match the exact combination and maximum asset count of any one media kind."""
        return modalities == self.modalities and (
            self.max_assets_per_modality is None
            or assets_per_modality <= self.max_assets_per_modality
        )


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    """What one composition can actually do, declared by its backends rather than inferred.

    Routing is by declared capability, so this is the same information `Memory` routes on. It is
    published because a caller and an agent both need to know which modalities an operation will
    accept before sending one, and because the embedding identity is what decides whether stored
    vectors and a new backend belong to the same space.
    """

    embedding: frozenset[Modality]
    embedding_model: str
    embedding_space: str
    embedding_dimension: int
    embedding_routes: tuple[EmbeddingRoute, ...] = ()
    generation: frozenset[Modality] = frozenset()
    generation_routes: tuple[frozenset[Modality], ...] = ()
    transcription: frozenset[Modality] = frozenset()
    vision: frozenset[Modality] = frozenset()
    face: frozenset[Modality] = frozenset()
    formation: frozenset[Modality] = frozenset()
    reranking: frozenset[Modality] = frozenset()
    reranking_model: str | None = None
    generation_model: str | None = None
    transcription_space: str | None = None
    vision_model: str | None = None
    face_model: str | None = None
    formation_model: str | None = None
    consolidation_model: str | None = None
    # A `TranscriptionBackend` yields text; only a `SpeechBackend` resolves speakers, and the two
    # occupy the same slot, so whether speech perception will work is not visible from the modalities.
    speaker_recognition: bool = False

    def __post_init__(self) -> None:
        _text(self.embedding_model, "embedding_model")
        _text(self.embedding_space, "embedding_space")
        if not isinstance(self.embedding_dimension, int) or self.embedding_dimension <= 0:
            raise ValidationError("embedding_dimension must be a positive integer")
        for name in (
            "embedding",
            "generation",
            "transcription",
            "vision",
            "face",
            "formation",
            "reranking",
        ):
            values = getattr(self, name)
            if not isinstance(values, frozenset) or any(
                not isinstance(value, Modality) or value is Modality.OMNI for value in values
            ):
                raise ValidationError(f"{name} capabilities must be a frozenset of modalities")
        embedding_routes = tuple(self.embedding_routes) or EmbeddingRoute.combinations(
            self.embedding
        )
        if (
            any(not isinstance(route, EmbeddingRoute) for route in embedding_routes)
            or frozenset(modality for route in embedding_routes for modality in route.modalities)
            != self.embedding
        ):
            raise ValidationError(
                "embedding_routes must cover exactly the declared embedding modalities"
            )
        object.__setattr__(self, "embedding_routes", tuple(dict.fromkeys(embedding_routes)))
        routes = tuple(self.generation_routes) or ((self.generation,) if self.generation else ())
        if (
            any(
                not isinstance(route, frozenset)
                or not route
                or any(not isinstance(modality, Modality) for modality in route)
                or not route <= self.generation
                for route in routes
            )
            or frozenset(modality for route in routes for modality in route) != self.generation
        ):
            raise ValidationError(
                "generation_routes must cover exactly the declared generation modalities"
            )
        object.__setattr__(self, "generation_routes", tuple(dict.fromkeys(routes)))

    def supports_embedding(
        self, modalities: frozenset[Modality], *, assets_per_modality: int = 1
    ) -> bool:
        """Whether one native embedding route accepts this combination, before atomic fallback."""
        if (
            not isinstance(modalities, frozenset)
            or not modalities
            or any(
                not isinstance(value, Modality) or value is Modality.OMNI for value in modalities
            )
        ):
            raise ValidationError(
                "modalities must be a non-empty frozenset of atomic Modality values"
            )
        if (
            isinstance(assets_per_modality, bool)
            or not isinstance(assets_per_modality, int)
            or assets_per_modality < 0
        ):
            raise ValidationError("assets_per_modality must be a non-negative integer")
        return any(
            route.supports(modalities, assets_per_modality=assets_per_modality)
            for route in self.embedding_routes
        )

    def supports_generation(self, modalities: frozenset[Modality]) -> bool:
        """Whether one native route covers the full modality combination, before conversion."""
        if (
            not isinstance(modalities, frozenset)
            or not modalities
            or any(
                not isinstance(modality, Modality) or modality is Modality.OMNI
                for modality in modalities
            )
        ):
            raise ValidationError(
                "modalities must be a non-empty frozenset of atomic Modality values"
            )
        return any(modalities <= route for route in self.generation_routes)

    @property
    def operations(self) -> frozenset[str]:
        """Optional operations this composition can serve, named by the backend each needs.

        The mapping from an operation to its backend was prose in three reference pages, so a
        caller had to know that `ask` needs generation and `consolidate` needs a consolidation
        backend. It is derived, never declared: the backends above are the only source.
        """
        available = {
            "ask": bool(self.generation),
            "perceive": (self.speaker_recognition and bool(self.transcription)) or bool(self.face),
            "transcribe": bool(self.transcription),
            "describe_vision": bool(self.vision),
            "formation": bool(self.formation),
            "consolidate": self.consolidation_model is not None,
            "consolidate_pending": self.consolidation_model is not None,
        }
        return frozenset(name for name, ready in available.items() if ready)

    def document(self, *, served: frozenset[str] | None = None) -> dict[str, object]:
        """Return the JSON-ready capability document every surface publishes.

        REST serves it from `/healthz`, the MCP server embeds it in its instructions, and
        `sensemem doctor` prints it, so the three cannot describe the same composition
        differently -- beyond the one narrowing below. Modality sets and operation names are
        sorted so the document is stable.

        `served` names the operations the publishing surface can actually route to, and
        `operations` is intersected with it. The backends are declared the same way everywhere,
        but the control plane is SDK-only by design, so a network surface that published the
        whole derivation would tell an agent it has a `consolidate` it cannot find. The SDK and
        `sensemem doctor` pass nothing and publish the derivation itself.
        """
        values: dict[str, object] = {
            declared.name: (
                sorted(item.value for item in value)
                if isinstance(value := getattr(self, declared.name), frozenset)
                else value
            )
            for declared in fields(self)
        }
        operations = self.operations
        values["generation_routes"] = [
            sorted(value.value for value in route) for route in self.generation_routes
        ]
        values["embedding_routes"] = [
            {
                "modalities": sorted(value.value for value in route.modalities),
                "max_assets_per_modality": route.max_assets_per_modality,
            }
            for route in self.embedding_routes
        ]
        values["operations"] = sorted(operations if served is None else operations & served)
        return values
