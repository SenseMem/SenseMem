"""The answer plane: ground evidence, route generation, and validate streamed answers.

Retrieval ranks candidates and Recall owns bounded planning and reads. Answering coordinates
answer attempts and sends their evidence through the selected generation backend.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from time import perf_counter
from typing import cast

from opentelemetry import trace
from opentelemetry.trace import Tracer

from sensemem._telemetry import (
    ASYNC_QUEUE_TIME,
    EXPANSION_EDGES,
    EXPANSION_ROWS,
    MODEL_TTFT,
    OPERATION_TTFT,
    RECALL_QUERY_BUDGET_DROPPED,
    RECALL_QUERY_CANDIDATES,
    RECALL_QUERY_COUNT,
    RECALL_QUERY_FAILURES,
    RECALL_QUERY_GROUNDED,
    RECALL_QUERY_LIMITED,
    RECALL_QUERY_NEW_CANDIDATES,
    RECALL_QUERY_OMITTED,
    RECALL_QUERY_SELECTED,
    _record_generation_inputs,
    _record_retrieval_results,
    current_model_request_count,
    mark_model_requests,
)
from sensemem.content import ContentInput
from sensemem.context_values import ContextBudget
from sensemem.exceptions import ModelError, SenseMemError, StorageError, ValidationError
from sensemem.kernel import grounding
from sensemem.kernel.content import PreparedContent, memory_modality, prepared_modalities
from sensemem.kernel.contracts import Backends
from sensemem.kernel.derived import has_stream_description, has_stream_transcript
from sensemem.kernel.execution import ASYNC_QUEUE_TIME_MS, Execution
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.identity import Identities
from sensemem.kernel.lifecycle import Lifecycle, OperationAssets
from sensemem.kernel.materialization import Materializer
from sensemem.kernel.recall import Recall, RecallReads, RecallRequest
from sensemem.kernel.retrieval import RERANK_CANDIDATES, Retrieval
from sensemem.kernel.routing import (
    derived_modalities,
    has_derived_text,
    route_input,
    with_descriptions,
)
from sensemem.kernel.runtime import Storage, translate_storage_errors
from sensemem.kernel.settings import ResolvedSettings
from sensemem.kernel.speech import Speech
from sensemem.kernel.temporal import (
    query_temporal_range,
    resolved_reference_at,
    temporal_context,
)
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import (
    optional_memory_type,
    pushed_memory_types,
    validate_answer_policy,
    validate_limit,
    validated_retrieval_scope,
)
from sensemem.kernel.vision import Vision
from sensemem.models.base import (
    GenerationBackend,
    GenerationRoutingBackend,
    ModelInput,
    StreamingGenerationBackend,
)
from sensemem.prompts import recall as recall_prompts
from sensemem.recall import (
    RecallResult,
)
from sensemem.records import AnswerChunk, AnswerResult, SearchHit
from sensemem.spatial import RetrievalScope
from sensemem.types import AnswerPolicy, MemoryCategory, Modality

# How many media rows an exhaustive plan may ground on, as a multiple of the window the caller
# asked for. Media is the one kind of evidence whose grounding costs writes -- face and speech
# recognition run on each row before the answer call, and again on a replan -- so a set read that
# matched every clip in the corpus is bounded here rather than by the character budget alone.
_RECALL_MEDIA_FACTOR = 2


@dataclass(frozen=True, slots=True)
class AnswerContext(RecallRequest):
    """Recall inputs plus generation routing and answer policy for one answering round."""

    answerer: GenerationBackend
    routed: ModelInput
    answer_policy: AnswerPolicy


class Answering(Traced):
    """Ground a question, optionally run its recall program, and generate the answer."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        backends: Backends,
        settings: ResolvedSettings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        vision: Vision,
        retrieval: Retrieval,
        execution: Execution,
        identities: Identities,
        recall: Recall,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._vision = vision
        self._retrieval = retrieval
        self._execution = execution
        self._identities = identities
        self._recall = recall

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        answer_policy: AnswerPolicy = "strict",
    ) -> AnswerResult:
        # Draining `ask_stream()` keeps one implementation of the whole path, so a buffered and
        # a streamed answer cannot drift apart in grounding, abstention, or reinforcement.
        stream = self.ask_stream(
            question,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            answer_policy=answer_policy,
        )
        while True:
            try:
                next(stream)
            except StopIteration as complete:
                return cast(AnswerResult, complete.value)

    def ask_stream(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryCategory | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        answer_policy: AnswerPolicy = "strict",
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        validate_limit(limit, maximum=100)
        validate_answer_policy(answer_policy)
        answerer = self._backends.answerer
        if answerer is None:
            raise ModelError(
                "answer backend is not configured",
                reason="backend_not_configured",
                stage="generate",
            )
        return self._ask_chunks(
            question,
            answerer=answerer,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            answer_policy=answer_policy,
        )

    def _ask_chunks(
        self,
        question: ContentInput,
        *,
        answerer: GenerationBackend,
        limit: int,
        memory_type: MemoryCategory | None,
        reference_at: datetime | None,
        scope: RetrievalScope | None,
        answer_policy: AnswerPolicy,
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        answer = yield from self._ask_operation(
            question,
            answerer=answerer,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
            answer_policy=answer_policy,
        )
        # The operation and its span close before the result is handed over. A generator stays
        # suspended at whichever yield the caller stops on, so a terminal chunk yielded inside
        # the operation would pin it open for every caller that reads the result and stops --
        # the ordinary way to consume this -- and `close()` waits on open operations.
        yield AnswerChunk(result=answer)
        return answer

    def _ask_operation(
        self,
        question: ContentInput,
        *,
        answerer: GenerationBackend,
        limit: int,
        memory_type: MemoryCategory | None,
        reference_at: datetime | None,
        scope: RetrievalScope | None,
        answer_policy: AnswerPolicy,
    ) -> Generator[AnswerChunk, None, AnswerResult]:
        started = perf_counter()
        operation_ttft_recorded = False
        with (
            self._trace("sensemem.ask", kind="operation") as span,
            self._lifecycle.operation(model_work=True) as assets,
        ):
            queue_time_ms = ASYNC_QUEUE_TIME_MS.get()
            if queue_time_ms is not None:
                span.set_attribute(ASYNC_QUEUE_TIME, queue_time_ms)
            # `search()` opens a `sensemem.search` operation span; `ask` reaches the same
            # retrieval plane directly. Keep its stage around every prerequisite through ranked
            # hits so the reported leg matches what answering actually waits for.
            with self._trace("sensemem.retrieve", kind="stage"):
                with self._trace("sensemem.content.prepare", kind="stage"):
                    prepared = self._materializer.materialize_content(question, assets)
                explicit_reference = resolved_reference_at(reference_at)
                reference, temporal_text = temporal_context(
                    prepared.text,
                    explicit_reference or datetime.now(timezone.utc),
                    infer_reference=explicit_reference is None,
                )
                temporal_range = query_temporal_range(temporal_text, reference)
                scope = validated_retrieval_scope(scope)
                memory_type = optional_memory_type(memory_type)
                ranked_depth = (
                    RERANK_CANDIDATES
                    if self._settings.evidence_budget is not None
                    else min(100, limit * 3)
                )
                prepared_search = partial(
                    self._retrieval.search_prepared,
                    prepared,
                    # Without a budget the answer grounds on `limit` hits, so ranking three times
                    # that is enough headroom for the modality round robin. With one, the budget is
                    # what decides depth, so the whole rerank pool has to be ranked for it to spend.
                    limit=ranked_depth,
                    operation=assets,
                    memory_types=pushed_memory_types(memory_type),
                    reference_at=reference,
                    temporal_range=temporal_range,
                    occurred_from=None,
                    occurred_until=None,
                    scope=scope,
                    require_unambiguous=limit == 1,
                    capture_trace=False,
                    capture_visual_matches=True,
                )
                speech_assets = self._speech.answer_speech_assets(prepared.assets)
                if speech_assets and all(
                    Modality(asset.modality) in self._backends.embedding_capabilities
                    for asset in speech_assets
                ):
                    with self._execution.overlap_speech(
                        self._speech.recognize, speech_assets, assets
                    ) as identities:
                        outcome = prepared_search()
                        hits = outcome.hits
                        _record_retrieval_results(hits)
                        identities.result()
                else:
                    if speech_assets:
                        self._speech.recognize(speech_assets, assets)
                    outcome = prepared_search()
                    hits = outcome.hits
                    _record_retrieval_results(hits)
            routed = self._route_generation(prepared, assets)
            image_ids = {asset.id for asset in routed.assets if asset.modality is Modality.IMAGE}
            round_context = AnswerContext(
                answerer=answerer,
                prepared=prepared,
                # Every question the answerer reads as text gets the reference time, not just the
                # ones whose phrasing a parser recognized. "How long ago did grandpa visit?"
                # narrows no retrieval window and names no date, yet it is exactly the question
                # the reader cannot answer without knowing when it is being asked. Applied after
                # routing, because that is what decides whether the reader gets text at all -- a
                # spoken question becomes text there -- and because routing appends speech
                # identities and transcripts, which would otherwise land after the line
                # documented as final.
                routed=replace(
                    routed,
                    image_retrieval_matches=tuple(
                        match
                        for match in outcome.image_retrieval_matches
                        if match.query_asset_id in image_ids
                    ),
                ),
                assets=assets,
                ranked=hits,
                ranked_depth=ranked_depth,
                limit=limit,
                reference=reference,
                temporal_range=temporal_range,
                scope=scope,
                memory_type=memory_type,
                answer_policy=answer_policy,
            )
            # `closing` rather than plain iteration: an abandoned stream throws `GeneratorExit`
            # at the yield below, and without this the inner generator would only be closed when
            # the cleared frame drops its last reference. That defers closing the provider's
            # response and ends `sensemem.model.generation` after its own parent, which loses
            # the call's model usage and emits a malformed trace.
            with closing(self._answer_rounds(round_context)) as deltas:
                while True:
                    try:
                        delta = next(deltas)
                    except StopIteration as complete:
                        result, hits = complete.value
                        break
                    if delta.strip() and not operation_ttft_recorded:
                        operation_ttft_ms = (perf_counter() - started) * 1_000.0
                        if queue_time_ms is not None:
                            operation_ttft_ms += queue_time_ms
                        span.set_attribute(OPERATION_TTFT, operation_ttft_ms)
                        operation_ttft_recorded = True
                    yield AnswerChunk(text=delta)
            used_ids = {hit.id for hit in result.hits}
            grounding = tuple(hit for hit in hits if hit.id in used_ids)
            # Reinforcement is part of answering, so it stays inside the operation and runs
            # before the terminal chunk: a caller that sees the result sees a settled store.
            if not result.abstained:
                self._reinforce_answered(grounding)
            # An abstention is the answering plane reporting that the evidence did not support
            # an answer, which is the same durable signal as an empty search: the memory the
            # question needed is missing or unreachable.
            self._retrieval.note_query_failure(
                prepared.text,
                failed=result.abstained,
                operation="ask",
                memory_types=pushed_memory_types(memory_type),
                scope=scope,
                reference_at=reference,
            )
            return AnswerResult(
                answer=result.answer,
                hits=grounding,
                abstained=result.abstained,
                abstention_reason=result.abstention_reason,
            )

    def _route_generation(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
        *,
        capabilities: frozenset[Modality] | None = None,
    ) -> ModelInput:
        supported = self._backends.generation_capabilities if capabilities is None else capabilities
        if self._speech.answer_speech_assets(prepared.assets):
            prepared = self._speech.with_speech_identities(prepared, operation)
        missing = prepared_modalities(prepared) - supported
        if missing & self._speech.transcript_fallback(prepared.assets):
            prepared = self._speech.with_audio_transcripts(prepared, operation)
        visuals = tuple(
            asset
            for asset in prepared.assets
            if Modality(asset.modality) in missing & {Modality.IMAGE, Modality.VIDEO}
            and not has_stream_description(prepared.text, (asset,))
        )
        if visuals:
            self._vision.query_descriptions(
                replace(prepared, assets=visuals),
                operation,
                required=frozenset(
                    asset.asset_id
                    for asset in visuals
                    if not has_derived_text(prepared.text, asset.asset_id, Modality(asset.modality))
                ),
            )
            identifiers = {asset.asset_id for asset in visuals}
            prepared = with_descriptions(
                prepared,
                {
                    key: value
                    for key, value in operation.query_descriptions.items()
                    if key in identifiers
                },
            )
        return route_input(prepared, self._hydrator.model_input(prepared), supported, "generation")

    def _route_generation_hits(
        self,
        hits: Sequence[SearchHit],
        operation: OperationAssets,
        *,
        capabilities: frozenset[Modality] | None = None,
    ) -> tuple[SearchHit, ...]:
        asset_ids = tuple(asset.id for hit in hits for asset in hit.assets)
        with (
            self._trace("sensemem.storage.hydrate", kind="stage"),
            translate_storage_errors("hydrate media for answer generation"),
        ):
            stored = self._store.media.read_assets(asset_ids)
        by_id = {asset.asset_id: asset for asset in stored}
        prepared_hits = []
        for hit in hits:
            try:
                assets = tuple(by_id[asset.id] for asset in hit.assets)
            except KeyError:
                raise StorageError(
                    "memory references missing media metadata", reason="asset_unavailable"
                ) from None
            prepared_hits.append(
                PreparedContent(
                    text=hit.content,
                    assets=assets,
                    modality=memory_modality(assets),
                    canonical_parts=tuple((part.kind, part.value) for part in hit.parts),
                    audio_transcript=has_stream_transcript(hit.content, assets),
                    visual_description=has_stream_description(hit.content, assets),
                )
            )
        self._identities.perceive(
            tuple(asset for prepared in prepared_hits for asset in prepared.assets),
            operation,
        )
        routed = []
        for hit, prepared in zip(hits, prepared_hits, strict=True):
            if self._vision.answer_face_assets(prepared.assets):
                prepared = self._vision.with_face_identities(prepared, operation)
            model_input = self._route_generation(prepared, operation, capabilities=capabilities)
            routed.append(
                replace(
                    hit,
                    content=model_input.text,
                    assets=model_input.assets,
                    modality=model_input.modality,
                    parts=model_input.parts,
                )
            )
        return tuple(routed)

    def _answer_rounds(
        self,
        context: AnswerContext,
    ) -> Generator[str, None, tuple[AnswerResult, tuple[SearchHit, ...]]]:
        """Answer this question, replanning once when the first round's own answer asked for it.

        A round another round may replace is held rather than streamed. Two complete answers on
        one wire with no boundary between them is not something a caller can render, and the
        terminal result has to be the text that was streamed -- so the deltas of a replannable
        round are buffered and yielded only if that round is the one that stands. Every other
        round, the common case and always the last one, streams as the provider produces it.
        """
        attempted = ""
        reads = self._recall.reads(context)
        for attempt in range(self._settings.recall_rounds if self._settings.recall_planning else 1):
            held: list[str] = []
            replannable = self._replan_possible(context, attempt=attempt)
            with closing(self._recall_round(context, reads, attempted=attempted)) as deltas:
                while True:
                    try:
                        delta = next(deltas)
                    except StopIteration as complete:
                        result, hits = complete.value
                        break
                    if replannable:
                        held.append(delta)
                    else:
                        yield delta
            if not (result.abstained and replannable):
                yield from held
                break
            attempted = grounding.attempt_note(
                hits, result, include_evidence=self._settings.recall_query_expansion
            )
        return result, hits

    def _recall_round(
        self,
        context: AnswerContext,
        reads: RecallReads,
        *,
        attempted: str,
    ) -> Generator[str, None, tuple[AnswerResult, tuple[SearchHit, ...]]]:
        """Ground one attempt at this question and yield its answer deltas.

        With `recall_planning` off there is exactly one attempt and this is the sequence `ask`
        has always run: the ranking's own grounding window, the routed question with the
        reference clock last, then the answer.
        """
        program = self._recall.execute(context, reads, attempted=attempted)
        semantic = (
            program is not None
            and self._settings.recall_query_expansion
            and any(reads.semantic_routes())
        )
        query_context = (
            self._query_media_context(context, reads) if self._settings.evidence_expansion else ()
        )
        native_grounding = (
            self._ground_query_media_recall(program, reads, context, query_context)
            if query_context
            else None
        )
        if native_grounding is not None:
            hits, note = native_grounding
        else:
            hits, note = self._grounded_recall(program, context, defer_tail=semantic)
            if program is not None and semantic:
                hits, note = self._ground_semantic_recall(program, reads, context, hits, note)
        hits, note = self._close_answer_evidence(hits, note, context, program)
        sequence = program is not None and program.plan.shape == "sequence"
        if sequence:
            # Ordering is a presentation step after every budget/route decision. Predicate
            # rows alone omit ranked anchors and semantic bridges, which may share their event
            # timestamp: SQLite's insertion order must cover the entire selected union.
            hits = reads.chronological(hits)
            order_note = recall_prompts.CHRONOLOGICAL_EVIDENCE_NOTE
            note = recall_prompts.append_note(note, order_note)
        answerer = context.answerer
        if isinstance(answerer, GenerationRoutingBackend):
            convertible = (
                self._backends.vision_capabilities
                | self._backends.transcription_capabilities
                | derived_modalities(
                    (context.routed, *(ModelInput(hit.content, hit.assets) for hit in hits))
                )
            )
            answerer = answerer.select(context.routed, hits, convertible=convertible)
        routed = self._route_generation(
            context.prepared, context.assets, capabilities=answerer.generation_capabilities
        )
        routed = replace(
            routed,
            image_retrieval_matches=tuple(
                match
                for match in context.routed.image_retrieval_matches
                if any(asset.id == match.query_asset_id for asset in routed.assets)
            ),
        )
        question = (
            grounding.with_reference_time(
                grounding.with_recall_note(routed, note), context.reference
            )
            if routed.text
            else routed
        )
        question = replace(question, evidence_sources=tuple(hits))
        routed_hits = (
            self._route_generation_hits(
                hits,
                context.assets,
                capabilities=answerer.generation_capabilities,
            )
            if hits
            else ()
        )
        if question.image_retrieval_matches:
            delivered = {(hit.id, asset.id) for hit in routed_hits for asset in hit.assets}
            question = replace(
                question,
                image_retrieval_matches=tuple(
                    match
                    for match in question.image_retrieval_matches
                    if (match.memory_id, match.source_asset_id) in delivered
                ),
            )
        self._speech.persist_transcripts(context.assets)
        with closing(
            self._answer_chunks(
                question,
                routed_hits,
                answerer=answerer,
                answer_policy=context.answer_policy,
                # This existing backend flag selects the program-order prompt, not a promise
                # of completeness. Sequence hits are ordered even for a pure similarity plan;
                # calling them rank order would contradict the question's chronology note.
                exhaustive=program is not None and (program.plan.exhaustive or sequence),
            )
        ) as deltas:
            result = yield from deltas
        return result, hits

    def _close_answer_evidence(
        self,
        hits: tuple[SearchHit, ...],
        note: str | None,
        context: AnswerContext,
        program: RecallResult | None,
    ) -> tuple[tuple[SearchHit, ...], str | None]:
        if not any(hit.context is not None and hit.context.evidence_ids for hit in hits):
            return hits, note
        # Deletion must not separate source hydration from operation-long media protection.
        with self._write_lock:
            closures, _ = self._hydrator.evidence_closures(
                hits,
                budget=ContextBudget(
                    max_items=RERANK_CANDIDATES,
                    memory_types=pushed_memory_types(context.memory_type),
                ),
                reference_at=context.reference,
                scope=context.scope,
            )
            budget = self._settings.evidence_budget
            if program is not None and (
                program.plan.exhaustive or self._settings.recall_query_expansion
            ):
                budget = (
                    self._settings.recall_set_budget
                    if budget is None
                    else min(budget, self._settings.recall_set_budget)
                )
            closed = grounding.close_grounding(
                hits, closures, grounding.grounding_hits(context.ranked, context.limit), budget
            )
            leased_ids = {asset.asset_id for asset in context.assets.leased}
            asset_ids = tuple({asset.id for hit in closed for asset in hit.assets} - leased_ids)
            if asset_ids:
                with translate_storage_errors("hydrate source media for answer generation"):
                    stored = self._store.media.read_assets(asset_ids)
                self._lifecycle.lease_assets(stored, context.assets.leased)
        delivered = {hit.id for hit in closed}
        if program is not None and program.plan.exhaustive and note is not None:
            original_ids = {hit.id for hit in hits}
            previous_omitted = sum(hit.id not in original_ids for hit in program.exhaustive)
            omitted = sum(hit.id not in delivered for hit in program.exhaustive)
            note = recall_prompts.update_recall_coverage(
                note, program, previous_omitted=previous_omitted, omitted=omitted
            )
        withheld = sum(
            hit.id not in delivered
            for hit in hits
            if hit.context is not None and hit.context.evidence_ids
        )
        if withheld:
            warning = recall_prompts.withheld_evidence_note(withheld)
            note = recall_prompts.append_note(note, warning)
        return closed, note

    def _query_media_context(
        self, context: AnswerContext, reads: RecallReads
    ) -> tuple[SearchHit, ...]:
        """Follow exact original query assets only from already scoped, hydrated ranked rows."""
        query_assets = {
            (asset.asset_id, asset.sha256)
            for asset in context.prepared.assets
            if asset.sha256 is not None
        }
        if not query_assets:
            return ()
        anchors = tuple(
            hit
            for hit in context.ranked[:RERANK_CANDIDATES]
            if any((asset.id, asset.sha256) in query_assets for asset in hit.assets)
        )[: context.limit]
        if not anchors:
            return ()
        with self._trace("sensemem.expand.query_media", kind="stage") as span:
            neighbors = reads.neighbors(
                tuple(hit.id for hit in anchors),
                before=1,
                after=1,
                max_rows=self._settings.evidence_expansion_max_rows,
            ).rows
            # The generic neighbor primitive applies scope but carries no type predicate.
            # The caller's memory type still constrains this supplemental read.
            neighbors = tuple(
                hit
                for hit in neighbors
                if context.memory_type is None or hit.memory_type is context.memory_type
            )
            by_id = {hit.id: hit for hit in (*anchors, *neighbors)}
            span.set_attributes(
                {EXPANSION_ROWS: len(by_id), EXPANSION_EDGES: (f"query_asset:{len(anchors)}",)}
            )
            return tuple(by_id.values())

    def _ground_query_media_recall(
        self,
        program: RecallResult | None,
        reads: RecallReads,
        context: AnswerContext,
        query_context: tuple[SearchHit, ...],
    ) -> tuple[tuple[SearchHit, ...], str | None] | None:
        """Give exact-source context budget before semantic additions and replaceable rank tail."""
        floor = grounding.grounding_hits(context.ranked, context.limit)
        floor_ids = {hit.id for hit in floor}
        routes = reads.semantic_routes() if self._settings.recall_query_expansion else ()
        exhaustive = program is not None and program.plan.exhaustive
        budget = self._settings.evidence_budget
        if exhaustive or any(routes):
            budget = (
                self._settings.recall_set_budget
                if budget is None
                else min(budget, self._settings.recall_set_budget)
            )
        note = None
        if program is not None and exhaustive:
            assert budget is not None
            required, omitted = grounding.budget_recall(
                replace(program, ranked=()),
                budget,
                media_limit=_RECALL_MEDIA_FACTOR * context.limit,
                max_rows=self._settings.recall_set_max_rows,
                required=floor,
            )
            note = recall_prompts.recall_note(program, omitted=omitted)
        else:
            required = floor
        required_ids = {hit.id for hit in required}
        matched = {hit.id for hit in program.exhaustive} if program is not None else set()
        # Predicate rows keep their existing caps. All further supplemental media share what
        # is left; the original ranked floor is never charged against this supplemental cap.
        media_limit = max(
            0,
            _RECALL_MEDIA_FACTOR * context.limit
            - sum(bool(hit.assets) for hit in required if hit.id not in floor_ids),
        )
        hits, native_added = grounding.pack_grounding(
            required,
            tuple(hit for hit in query_context if hit.id not in matched or hit.id in required_ids),
            (),
            budget,
            media_limit=media_limit,
            max_added=self._settings.evidence_expansion_max_rows,
        )
        if not native_added:
            return None
        media_limit -= sum(bool(hit.assets) for hit in hits if hit.id not in required_ids)
        if native_added:
            native_note = recall_prompts.query_media_note(native_added)
            note = recall_prompts.append_note(note, native_note)
        offered = grounding.semantic_hits(routes, {hit.id for hit in hits} | matched, context.limit)
        before_semantic = {hit.id for hit in hits}
        hits, semantic_added = grounding.pack_grounding(
            hits, offered, (), budget, media_limit=media_limit
        )
        media_limit -= sum(bool(hit.assets) for hit in hits if hit.id not in before_semantic)
        if semantic_added:
            semantic_note = recall_prompts.semantic_evidence_note(semantic_added, included=True)
            note = recall_prompts.append_note(note, semantic_note)
        tail = context.ranked
        if program is not None and exhaustive:
            tail = grounding.recall_ranked_tail(
                program, tail, budgeted=self._settings.evidence_budget is not None
            )
        if budget is not None and (self._settings.evidence_budget is not None or exhaustive):
            hits = (*hits, *grounding.budgeted_hits(tail, hits, budget))
        remaining = self._settings.evidence_expansion_max_rows - native_added
        if remaining:
            expansion = self._recall.expand(hits, context)
            expansion = replace(
                expansion,
                hits=tuple(
                    hit
                    for hit in expansion.hits
                    if hit.id not in matched
                    and (context.memory_type is None or hit.memory_type is context.memory_type)
                ),
            )
            hits, linked = grounding.pack_grounding(
                hits,
                expansion.hits,
                (),
                budget,
                media_limit=media_limit,
                max_added=remaining,
            )
            if linked:
                linked_note = recall_prompts.expansion_note(
                    linked,
                    expansion.edges,
                    chronological=program is not None and program.plan.shape == "sequence",
                )
                note = recall_prompts.append_note(note, linked_note)
        return hits, note

    def _ground_semantic_recall(
        self,
        program: RecallResult,
        reads: RecallReads,
        context: AnswerContext,
        required: tuple[SearchHit, ...],
        note: str | None,
    ) -> tuple[tuple[SearchHit, ...], str | None]:
        """Pack query-local evidence before the replaceable original ranked tail.

        Scores from distinct queries are not comparable. Each query can add at most one
        `limit`-sized window, after dedup; the character and media caps may narrow it further.
        The same limits apply to point and exhaustive plans, without licensing set totals.
        """
        routes = reads.semantic_routes()
        candidates = {hit.id for route in routes for hit in route}
        matched = {hit.id for hit in program.exhaustive}
        # Predicate rows keep their own row/media caps and completeness accounting.
        offered = grounding.semantic_hits(
            routes, {hit.id for hit in required} | matched, context.limit
        )
        budget = self._settings.recall_set_budget
        if self._settings.evidence_budget is not None:
            budget = min(budget, self._settings.evidence_budget)
        with self._trace("sensemem.recall.ground", kind="stage") as span:
            hits, added = grounding.pack_grounding(
                required, offered, (), budget, media_limit=_RECALL_MEDIA_FACTOR * context.limit
            )
            required_ids = {hit.id for hit in required}
            novel = candidates - required_ids
            span.set_attributes(
                {
                    # Query counts are cumulative per ask; candidate/grounding counts describe
                    # this round. Counts only: no query, evidence or provider failure text.
                    RECALL_QUERY_COUNT: reads.queries,
                    RECALL_QUERY_FAILURES: reads.failures,
                    RECALL_QUERY_LIMITED: len(reads.limited),
                    RECALL_QUERY_CANDIDATES: len(candidates),
                    RECALL_QUERY_NEW_CANDIDATES: len(
                        candidates - {hit.id for hit in context.ranked}
                    ),
                    RECALL_QUERY_GROUNDED: added,
                    RECALL_QUERY_SELECTED: len(offered),
                    RECALL_QUERY_BUDGET_DROPPED: len(offered) - added,
                    RECALL_QUERY_OMITTED: len(novel) - added,
                }
            )
        if not added:
            return self._grounded_recall(program, context)
        semantic_note = recall_prompts.semantic_evidence_note(
            added, included=program.plan.shape == "sequence"
        )
        note = recall_prompts.append_note(note, semantic_note)
        tail = context.ranked
        if program.plan.exhaustive:
            tail = grounding.recall_ranked_tail(
                program, tail, budgeted=self._settings.evidence_budget is not None
            )
        if program.plan.exhaustive or self._settings.evidence_budget is not None:
            hits = (*hits, *grounding.budgeted_hits(tail, hits, budget))
        if self._settings.evidence_expansion:
            expansion = self._recall.expand(hits, context)
            hits, linked = grounding.pack_grounding(
                hits, expansion.hits, (), budget, media_limit=_RECALL_MEDIA_FACTOR * context.limit
            )
            if linked:
                linked_note = recall_prompts.expansion_note(
                    linked, expansion.edges, chronological=program.plan.shape == "sequence"
                )
                note = recall_prompts.append_note(note, linked_note)
        return hits, note

    def _grounded_recall(
        self,
        program: RecallResult | None,
        context: AnswerContext,
        *,
        defer_tail: bool = False,
    ) -> tuple[tuple[SearchHit, ...], str | None]:
        """Widen the grounding window by whatever the question's shape asks for beyond it.

        A ranking question keeps the window the ranking earned. A question that asked for every
        matching record adds that set to it, up to `recall_set_budget_chars`, and carries a line
        saying what the set is -- because a reader cannot see the predicate that produced its
        evidence, and a count over a silently truncated set is a wrong answer stated confidently.

        The set is added to the ranked window, never substituted for it. Grounding a plan on its
        exhaustive rows alone cost every question whose reads returned few rows the window it
        already had: measured on ATM-Hard, 12 grounded records down to 1, 2, 4 and 5, and one
        down to a refusal the unplanned path had answered. A plan says what else is relevant; it
        never says the ranking was wrong.
        """
        if program is not None:
            # A selective predicate must not strand spare budget or hide ranked follow-ups
            # that omit its literal terms. Query-local rows still pass semantic row/media caps.
            program = replace(
                program,
                ranked=(
                    ()
                    if defer_tail
                    else grounding.recall_ranked_tail(
                        program,
                        context.ranked,
                        budgeted=self._settings.evidence_budget is not None,
                    )
                ),
            )
        if program is None or not program.plan.exhaustive:
            if defer_tail:
                return grounding.grounding_hits(context.ranked, context.limit), None
            budget = self._settings.evidence_budget
            if not self._settings.evidence_expansion:
                return grounding.grounding_hits(
                    context.ranked, context.limit, budget_chars=budget
                ), None
            required = grounding.grounding_hits(context.ranked, context.limit)
            expansion = self._recall.expand(required, context)
            hits, added = grounding.pack_grounding(
                required,
                expansion.hits,
                context.ranked,
                budget,
                media_limit=_RECALL_MEDIA_FACTOR * context.limit,
            )
            return hits, (
                recall_prompts.expansion_note(
                    added,
                    expansion.edges,
                    chronological=program is not None and program.plan.shape == "sequence",
                )
                if added
                else None
            )
        # Both settings constrain supplemental packing, after charging the mandatory ranked
        # window. A set plan may narrow that allowance, never widen it. Neither setting trims
        # the mandatory window or promises a provider input limit.
        budget = (
            self._settings.recall_set_budget
            if self._settings.evidence_budget is None
            else min(self._settings.recall_set_budget, self._settings.evidence_budget)
        )
        hits, omitted = grounding.budget_recall(
            program,
            budget,
            media_limit=_RECALL_MEDIA_FACTOR * context.limit,
            max_rows=self._settings.recall_set_max_rows,
            # Reserve the limit-sized window, including its modality floor, before additions.
            required=grounding.grounding_hits(context.ranked, context.limit),
        )
        note = recall_prompts.recall_note(program, omitted=omitted)
        if self._settings.evidence_expansion and not defer_tail:
            # A plan already decided what else to read, so expansion follows its set rather than
            # competing with it: the rows go after everything the plan grounded, on whatever the
            # budget has left. Planning and expansion are separate mechanisms and this is the
            # only place they meet.
            expansion = self._recall.expand(hits, context)
            packed, added = grounding.pack_grounding(
                hits,
                expansion.hits,
                (),
                budget,
                media_limit=_RECALL_MEDIA_FACTOR * context.limit,
            )
            if added:
                linked_note = recall_prompts.expansion_note(
                    added, expansion.edges, chronological=program.plan.shape == "sequence"
                )
                return packed, recall_prompts.append_note(note, linked_note)
        return hits, note

    def _replan_possible(self, context: AnswerContext, *, attempt: int) -> bool:
        """Whether a later round could replace this one, which is what makes it unstreamable.

        Another round needs an abstention. Experimental semantic recall may also investigate a
        strict refusal before releasing it; ordinary planning retains its best-effort-only rule.
        """
        return (
            self._settings.recall_planning
            and (context.answer_policy == "best_effort" or self._settings.recall_query_expansion)
            and attempt + 1 < self._settings.recall_rounds
        )

    def _answer_chunks(  # noqa: C901 - streaming and non-streaming validation share one span
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answerer: GenerationBackend,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> Generator[str, None, AnswerResult]:
        """Yield answer deltas in provider order and return the validated grounded result.

        A backend without `stream_answer` yields its whole answer as one delta, so every caller
        sees the same shape and `ask_stream()` never requires a streaming provider.
        """
        model = getattr(answerer, "generation_model", None)
        with self._model_trace(
            "generation",
            "chat",
            model=model if isinstance(model, str) else None,
            batch_size=1,
            modalities=(
                modality
                for value in (question, *(ModelInput(hit.content, hit.assets) for hit in hits))
                for modality in value.modalities
            ),
        ):
            mark_model_requests(1)
            _record_generation_inputs(tuple(hits))
            buffered = False
            try:
                if isinstance(answerer, StreamingGenerationBackend):
                    started = perf_counter()
                    parts: list[str] = []
                    stream = answerer.stream_answer(
                        question, hits, answer_policy=answer_policy, exhaustive=exhaustive
                    )
                    result: object = None
                    try:
                        while True:
                            try:
                                part = next(stream)
                            except StopIteration as completed:
                                result = completed.value
                                break
                            if not isinstance(part, str):
                                raise ModelError(
                                    "generation model returned an invalid answer chunk",
                                    reason="response_invalid",
                                )
                            if part:
                                if not parts and current_model_request_count():
                                    trace.get_current_span().set_attribute(
                                        MODEL_TTFT, perf_counter() - started
                                    )
                                parts.append(part)
                                yield part
                    finally:
                        # This iterator owns the provider's streaming response. An abandoned
                        # `ask_stream()` unwinds to here; close explicitly so releasing the
                        # response does not depend on interpreter reference collection.
                        stream.close()
                    answer = "".join(parts)
                    if not answer.strip():
                        raise ModelError(
                            "generation model returned an invalid answer", reason="response_invalid"
                        )
                else:
                    result = answerer.answer(
                        question, hits, answer_policy=answer_policy, exhaustive=exhaustive
                    )
                    buffered = True
                if not isinstance(result, AnswerResult) or (
                    not buffered and answer_policy == "strict" and result.answer != answer
                ):
                    raise ModelError(
                        "generation model returned an invalid answer", reason="response_invalid"
                    )
            except ValidationError as error:
                raise ModelError(
                    "generation model returned an invalid answer", reason="response_invalid"
                ) from error
            except SenseMemError:
                raise
            except Exception as error:
                raise ModelError(
                    "failed to generate a grounded answer", reason="model_failed"
                ) from error
            if buffered:
                yield result.answer
            return result

    def _reinforce_answered(self, hits: Sequence[SearchHit]) -> None:
        """Count backend-declared evidence use after a non-abstained answer.

        The caller intersects the backend's result with the selected canonical hits. Declared
        use is not verified factual support. Bookkeeping failure must not discard the generated
        answer; disabling `reinforce_on_answer` keeps later rankings independent of this update.
        """
        if not hits or not self._settings.reinforce_on_answer:
            return
        with suppress(Exception), self._write_lock:
            self._store.records.reinforce_memories(
                tuple(hit.id for hit in hits),
                accessed_at=datetime.now(timezone.utc),
            )
