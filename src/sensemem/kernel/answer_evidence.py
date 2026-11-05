"""Compose any generation client with independent evidence computation plugins."""

from collections.abc import Generator, Sequence
from dataclasses import replace
from datetime import datetime

from sensemem._resources import unique_resources
from sensemem.evidence import answer_evidence_payloads, retain_source_complete_hits
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models.base import (
    GenerationBackend,
    GenerationMediaBackend,
    GenerationRoutingBackend,
    MediaSelectionBackend,
    ModelInput,
    RecallPlanningBackend,
    SourceMetadataBackend,
    StreamingGenerationBackend,
)
from sensemem.types import AnswerPolicy, AnswerResult, Modality, SearchHit


class EvidenceGeneration:
    def __init__(
        self,
        backend: GenerationBackend,
        *,
        selector: MediaSelectionBackend | None = None,
        enricher: SourceMetadataBackend | None = None,
        media_max_items: int = 2,
    ) -> None:
        self._backend, self._selector, self._enricher = backend, selector, enricher
        self._media_max_items = media_max_items
        if (
            isinstance(media_max_items, bool)
            or not isinstance(media_max_items, int)
            or not 1 <= media_max_items <= 8
        ):
            raise ValidationError("media_max_items must be an integer between 1 and 8")
        self.generation_capabilities = backend.generation_capabilities
        self.generation_model = getattr(backend, "generation_model", None)
        self.generation_routes = (
            backend.generation_routes
            if isinstance(backend, GenerationRoutingBackend)
            else (backend.generation_capabilities,)
        )

    def select(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        convertible: frozenset[Modality] = frozenset(),
    ) -> GenerationBackend:
        if not isinstance(self._backend, GenerationRoutingBackend):
            return self
        backend = self._backend.select(question, hits, convertible=convertible)
        return EvidenceGeneration(
            backend,
            selector=self._selector,
            enricher=self._enricher,
            media_max_items=self._media_max_items,
        )

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        question, hits = self.prepare(question, hits)
        return self._backend.answer(
            question, hits, answer_policy=answer_policy, exhaustive=exhaustive
        )

    def stream_answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]:
        question, hits = self.prepare(question, hits)
        if isinstance(self._backend, StreamingGenerationBackend):
            return (
                yield from self._backend.stream_answer(
                    question,
                    hits,
                    answer_policy=answer_policy,
                    exhaustive=exhaustive,
                )
            )
        result = self._backend.answer(
            question, hits, answer_policy=answer_policy, exhaustive=exhaustive
        )
        yield result.answer
        return result

    def prepare(
        self, question: ModelInput, hits: Sequence[SearchHit]
    ) -> tuple[ModelInput, tuple[SearchHit, ...]]:
        assets = (
            self._backend.grounding_assets(question, hits)
            if self._enricher is not None and isinstance(self._backend, GenerationMediaBackend)
            else None
        )
        sources = question.evidence_sources or tuple(hits)
        supplied_assets = (*question.assets, *(asset for hit in hits for asset in hit.assets))
        metadata = question.source_metadata
        if self._enricher is not None:
            metadata = self._enricher.enrich(supplied_assets if assets is None else assets)
        question = replace(question, source_metadata=metadata, evidence_sources=sources)
        if self._selector is None:
            return question, tuple(hits)
        question_ids = {asset.id for asset in question.assets}
        available = {
            asset.id: asset
            for hit in hits
            for asset in hit.assets
            if asset.modality in self.generation_capabilities and asset.id not in question_ids
        }
        indexed = tuple(available.values())[:64]
        selected: tuple[int, ...] = ()
        if indexed:
            payload: dict[str, object] = {
                "question": question.text,
                "question_media_count": len(question.assets),
                "hits": answer_evidence_payloads(hits),
                "available_media": [
                    {
                        "media_index": index,
                        "modality": asset.modality.value if asset.modality else None,
                        "evidence_indices": [
                            position
                            for position, hit in enumerate(hits)
                            if any(item.id == asset.id for item in hit.assets)
                        ],
                    }
                    for index, asset in enumerate(indexed)
                ],
                "max_items": self._media_max_items,
            }
            try:
                selected = self._selector.select_media(
                    payload, count=len(indexed), max_items=self._media_max_items
                )
            except ModelError as error:
                if error.reason == "payload_too_large":
                    raise
                selected = ()
            if (
                not isinstance(selected, tuple)
                or len(selected) > self._media_max_items
                or any(
                    type(index) is not int or not 0 <= index < len(indexed) for index in selected
                )
                or len(set(selected)) != len(selected)
            ):
                selected = ()
        chosen = {indexed[index].id for index in selected} | question_ids
        routed = []
        for hit in hits:
            retained = tuple(asset for asset in hit.assets if asset.id in chosen)
            if retained or hit.content.strip():
                routed.append(
                    replace(
                        hit, assets=retained, modality=ModelInput(hit.content, retained).modality
                    )
                )
        return question, retain_source_complete_hits(routed, hits)

    def plan_recall(
        self, question: str, *, reference_at: datetime, corpus_digest: str, attempted: str = ""
    ) -> str | None:
        if isinstance(self._backend, RecallPlanningBackend):
            return self._backend.plan_recall(
                question,
                reference_at=reference_at,
                corpus_digest=corpus_digest,
                attempted=attempted,
            )
        return None

    def resources(
        self,
    ) -> tuple[GenerationBackend | MediaSelectionBackend | SourceMetadataBackend, ...]:
        return tuple(
            item for item in (self._backend, self._selector, self._enricher) if item is not None
        )

    def close(self) -> None:
        resources = self.resources()
        failures = []
        for resource in unique_resources(resources):
            try:
                resource.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise failures[0]
