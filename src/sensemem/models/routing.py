"""Explicit compositions of model capabilities, with no registry or product-state access."""

from collections.abc import Generator, Sequence
from datetime import datetime
from functools import partial

from sensemem._resources import unique_resources
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models._concurrency import ordered_calls
from sensemem.models.base import (
    EmbeddingBackend,
    EmbedTask,
    GenerationBackend,
    GenerationRoutingBackend,
    ModelInput,
    RecallPlanningBackend,
    StreamingGenerationBackend,
    embedding_routes,
)
from sensemem.types import AnswerPolicy, AnswerResult, Modality, SearchHit


class GenerationRouter:
    """Select the narrowest native backend covering both question and supplied evidence.

    Caller order breaks ties. Model rejection never silently changes the selected backend.
    Every distinct child is owned and closed once by this composition.
    """

    def __init__(self, backends: Sequence[GenerationBackend]) -> None:
        self.backends: tuple[GenerationBackend, ...] = tuple(
            child
            for backend in backends
            for child in (backend.backends if isinstance(backend, GenerationRouter) else (backend,))
        )
        if not self.backends or any(
            not isinstance(item, GenerationBackend) for item in self.backends
        ):
            raise ValidationError("generation routes must contain GenerationBackend values")
        self.generation_capabilities = frozenset(
            modality for item in self.backends for modality in item.generation_capabilities
        )
        self._coverage = tuple(
            (
                item.generation_routes
                if isinstance(item, GenerationRoutingBackend)
                else (item.generation_capabilities,)
            )
            for item in self.backends
        )
        self.generation_routes = tuple(
            dict.fromkeys(route for routes in self._coverage for route in routes)
        )
        self.generation_model = "capability-routing"

    def select(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        convertible: frozenset[Modality] = frozenset(),
    ) -> GenerationBackend:
        required = question.modalities | frozenset(
            modality for hit in hits for modality in ModelInput(hit.content, hit.assets).modalities
        )
        candidates = []
        for item, routes in zip(self.backends, self._coverage, strict=True):
            if not any(
                required <= route or (Modality.TEXT in route and required - route <= convertible)
                for route in routes
            ):
                continue
            if isinstance(item, GenerationRoutingBackend):
                try:
                    item = item.select(question, hits, convertible=convertible)
                except ModelError as error:
                    if error.reason != "unsupported_modality":
                        raise
                    continue
            if required <= item.generation_capabilities or (
                Modality.TEXT in item.generation_capabilities
                and required - item.generation_capabilities <= convertible
            ):
                candidates.append(item)
        if not candidates:
            raise ModelError("no generation route covers this input", reason="unsupported_modality")
        return min(
            candidates,
            key=lambda item: (
                len(required - item.generation_capabilities),
                len(item.generation_capabilities - required),
            ),
        )

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return self.select(question, hits).answer(
            question,
            hits,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
        )

    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]:
        backend = self.select(question, hits)
        if isinstance(backend, StreamingGenerationBackend):
            return (
                yield from backend.stream_answer(
                    question,
                    hits,
                    answer_policy=answer_policy,
                    exhaustive=exhaustive,
                )
            )
        result = backend.answer(question, hits, answer_policy=answer_policy, exhaustive=exhaustive)
        yield result.answer
        return result

    def plan_recall(
        self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
    ) -> str | None:
        for backend in self.backends:
            if (
                isinstance(backend, RecallPlanningBackend)
                and Modality.TEXT in backend.generation_capabilities
            ):
                return backend.plan_recall(
                    question,
                    reference_at=reference_at,
                    corpus_digest=corpus_digest,
                    attempted=attempted,
                )
        return None

    def resources(self) -> tuple[GenerationBackend, ...]:
        return self.backends

    def close(self) -> None:
        failures = []
        for backend in unique_resources(self.backends):
            try:
                backend.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ModelError("failed to close generation routes") from failures[0]


class EmbeddingRouter:
    """Compose specialized encoders which explicitly share one vector space and dimension."""

    def __init__(self, backends: Sequence[EmbeddingBackend]) -> None:
        self.backends = tuple(backends)
        if not self.backends or any(
            not isinstance(item, EmbeddingBackend) for item in self.backends
        ):
            raise ValidationError("embedding routes must contain EmbeddingBackend values")
        first = self.backends[0]
        if any(
            (item.embedding_space, item.embedding_dimension)
            != (first.embedding_space, first.embedding_dimension)
            for item in self.backends
        ):
            raise ValidationError("embedding routes must share one vector space and dimension")
        self.embedding_space = first.embedding_space
        self.embedding_model = "capability-routing"
        self.embedding_dimension = first.embedding_dimension
        declared = tuple(item.embedding_capabilities for item in self.backends)
        self.embedding_capabilities = frozenset(
            modality for modalities in declared for modality in modalities
        )
        self._coverage = tuple(
            embedding_routes(item, modalities)
            for item, modalities in zip(self.backends, declared, strict=True)
        )
        self.embedding_routes = tuple(
            dict.fromkeys(route for routes in self._coverage for route in routes)
        )

    def _candidates(self, value: ModelInput) -> tuple[EmbeddingBackend, ...]:
        modalities, asset_count = value.modalities, value.assets_per_modality
        return tuple(
            item
            for item, routes in zip(self.backends, self._coverage, strict=True)
            if any(route.supports(modalities, assets_per_modality=asset_count) for route in routes)
        )

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        grouped: dict[int, list[tuple[int, ModelInput]]] = {}
        for position, value in enumerate(inputs):
            candidates = self._candidates(value)
            if not candidates:
                raise ModelError(
                    "no embedding route covers this input", reason="unsupported_modality"
                )
            backend = min(
                candidates, key=lambda item: len(item.embedding_capabilities - value.modalities)
            )
            grouped.setdefault(self.backends.index(backend), []).append((position, value))
        groups = tuple(grouped.items())
        batches = ordered_calls(
            tuple(
                partial(self.backends[index].embed, tuple(value for _, value in rows), task=task)
                for index, rows in groups
            )
        )
        result: dict[int, tuple[float, ...]] = {}
        for (_, rows), vectors in zip(groups, batches, strict=True):
            if len(rows) != len(vectors):
                raise ModelError(
                    "embedding route returned invalid output", reason="response_invalid"
                )
            result.update(
                (position, vector) for (position, _), vector in zip(rows, vectors, strict=True)
            )
        return tuple(result[position] for position in range(len(inputs)))

    def resources(self) -> tuple[EmbeddingBackend, ...]:
        return self.backends

    def close(self) -> None:
        failures = []
        for backend in unique_resources(self.backends):
            try:
                backend.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise ModelError("failed to close embedding routes") from failures[0]
