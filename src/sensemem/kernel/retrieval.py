"""The retrieval plane: candidates from the index, truth from SQLite, one ranking."""

from __future__ import annotations

import builtins
import logging
import re
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from itertools import chain

from opentelemetry.trace import Tracer

from sensemem._telemetry import mark_model_requests
from sensemem.exceptions import ModelError, SenseMemError
from sensemem.infrastructure.local.scoring import max_cosine_part_scores, max_cosine_scores
from sensemem.infrastructure.local.store import IndexCandidate, StoredEmbedding, StoredMemory
from sensemem.infrastructure.local.zvec_index import IndexHit
from sensemem.kernel.content import PreparedContent
from sensemem.kernel.contracts import Backends
from sensemem.kernel.embedding import DOCUMENT_TASK, MAX_RETRIEVAL_KEYS, Embedding
from sensemem.kernel.evidence_windows import query_evidence_windows
from sensemem.kernel.execution import Execution
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.lifecycle import Lifecycle, OperationAssets
from sensemem.kernel.materialization import Materializer
from sensemem.kernel.projection import Projection
from sensemem.kernel.ranking import (
    IndexCandidates,
    SearchOutcome,
    extend_hydration_traces,
    extend_memory_type_traces,
    extend_missing_memory_traces,
    extend_ranked_traces,
    lexical_query_terms,
    merge_index_hits,
    merge_lexical_routes,
    merge_temporal_lexical_hits,
    parent_index_ids,
    parent_index_signals,
    qualified_candidates,
    rank_memories,
    retrieval_is_ambiguous,
    search_outcome,
)
from sensemem.kernel.routing import has_derived_text, with_descriptions
from sensemem.kernel.runtime import (
    Index,
    Storage,
    translate_index_errors,
    translate_storage_errors,
)
from sensemem.kernel.settings import Settings
from sensemem.kernel.speech import Speech
from sensemem.kernel.temporal import (
    intersect_occurrence_range,
    occurrence_overlaps,
    query_temporal_range,
    resolved_reference_at,
    search_occurrence_range,
    temporal_context,
)
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import (
    optional_memory_type,
    pushed_memory_types,
    strict_bool,
    validate_limit,
    validated_retrieval_scope,
)
from sensemem.kernel.vision import Vision
from sensemem.models.base import EmbedTask, ModelInput
from sensemem.models.reranking import reranking_scores
from sensemem.types import (
    ContentInput,
    IndexQuantization,
    MemoryType,
    Modality,
    RetrievalCandidateTrace,
    RetrievalMode,
    RetrievalScope,
    SearchResult,
    VisualRetrievalMatch,
)

RERANK_CANDIDATES = 100

_LOGGER = logging.getLogger(__name__)


# Everything a recall query's normalization drops before two of them are called near-equal:
# case, accents, punctuation, and whitespace runs. Deliberately lexical -- an embedding-based
# near-duplicate check would make the trigger depend on the model, so "did the user ask this
# again" would change meaning when the embedder was replaced.
_QUERY_NOISE = re.compile(r"[^\w\s]+", re.UNICODE)


_QUERY_SPACE = re.compile(r"\s+")


# Depth of every index route, deliberately not a function of the requested `limit`. Each route
# is truncated to it and a memory's dense relevance is the maximum over the routes that reached
# it, so a `limit`-derived depth made the score -- and the lexical term weights computed over the
# candidate pool -- properties of the request: at limit 20 against limit 100, 27-39 % of the
# candidates present in both runs reported a different dense relevance and 78-100 % a different
# lexical bonus, over adjacent-rank cosine gaps whose median is 0.0026-0.0077. Every public
# `limit` is capped at 100. Unquantized routes keep that fixed pool; quantized routes retain
# twice as many parents before authoritative FP32 scoring, without repeating that expansion
# in native grouping. The depth remains independent of the requested result count. Earlier
# widening to `limit * 3` cost 2.3x p50 on a common limit-8 query; the loop below widens further
# only when authoritative scope checks leave too few survivors.
_ROUTE_CANDIDATES = RERANK_CANDIDATES
_QUANTIZED_ROUTE_CANDIDATES = RERANK_CANDIDATES * 2


_MAX_QUERY_RETRIEVAL_KEYS = 7


@dataclass(frozen=True, slots=True)
class _QueryRepresentations:
    prepared: PreparedContent
    text: str
    vectors: tuple[tuple[float, ...], ...]
    image_queries: tuple[tuple[str, tuple[float, ...]], ...]
    lexical_queries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CandidatePool:
    native: IndexCandidates
    index_ids: tuple[str, ...]
    hydrated: tuple[IndexCandidate, ...]
    documents: tuple[IndexCandidate, ...]
    limit: int


class Retrieval(Traced):
    """One shared retrieval plane: index candidates, authoritative hydration, ranking."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        index: Index,
        backends: Backends,
        settings: Settings,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        vision: Vision,
        embedding: Embedding,
        projection: Projection,
        execution: Execution,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._write_lock = storage.write_lock
        self._index = index
        self._backends = backends
        self._settings = settings
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._vision = vision
        self._embedding = embedding
        self._projection = projection
        self._execution = execution

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
        explain: bool = False,
    ) -> SearchResult:
        with (
            self._trace("sensemem.search", kind="operation"),
            self._lifecycle.operation(model_work=True) as assets,
        ):
            validate_limit(limit, maximum=100)
            explain = strict_bool(explain, "explain")
            occurred_from, occurred_until = search_occurrence_range(
                occurred_from,
                occurred_until,
            )
            scope = validated_retrieval_scope(scope)
            with self._trace("sensemem.content.prepare", kind="stage"):
                prepared = self._materializer.prepare(query, assets)
            explicit_reference = resolved_reference_at(reference_at)
            reference, temporal_text = temporal_context(
                prepared.text,
                explicit_reference or datetime.now(timezone.utc),
                infer_reference=explicit_reference is None,
            )
            outcome = self.search_prepared(
                prepared,
                limit=limit,
                operation=assets,
                memory_types=pushed_memory_types(optional_memory_type(memory_type)),
                reference_at=reference,
                temporal_range=query_temporal_range(temporal_text, reference),
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                scope=scope,
                require_unambiguous=limit == 1,
                capture_trace=explain,
            )
            self._speech.persist_transcripts(assets)
            self.note_query_failure(prepared.text, failed=not outcome.hits)
            return SearchResult(hits=outcome.hits, trace=outcome.trace)

    def search_prepared(  # noqa: C901 - one shared retrieval plane
        self,
        prepared: PreparedContent,
        *,
        limit: int,
        operation: OperationAssets,
        memory_types: frozenset[MemoryType] | None,
        reference_at: datetime,
        temporal_range: tuple[datetime, datetime] | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        scope: RetrievalScope | None,
        require_unambiguous: bool,
        capture_trace: bool,
        capture_visual_matches: bool = False,
    ) -> SearchOutcome:
        trace_candidates: builtins.list[RetrievalCandidateTrace] | None = (
            [] if capture_trace else None
        )
        query = self._prepare_query(
            prepared, operation, capture_visual_matches=capture_visual_matches
        )
        prepared = query.prepared
        vectors, focused_text, image_queries = query.vectors, query.text, query.image_queries
        pool = self._candidate_pool(
            vectors,
            lexical_queries=query.lexical_queries,
            limit=limit,
            memory_types=memory_types,
            temporal_range=temporal_range,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )
        candidates, index_ids, candidate_limit = pool.native, pool.index_ids, pool.limit
        if not index_ids:
            return search_outcome(
                (),
                trace_candidates,
                candidate_limit=candidate_limit,
                exhaustive=candidates.exhausted,
            )
        hydrated_documents, documents = pool.hydrated, pool.documents
        index_ids_by_memory = (
            parent_index_ids(hydrated_documents) if trace_candidates is not None else {}
        )
        extend_hydration_traces(
            trace_candidates,
            candidates,
            index_ids,
            hydrated_documents,
            documents,
            index_ids_by_memory,
            retrieval_mode=self._settings.retrieval_mode,
        )
        (
            dense_relevance,
            dense_confidence,
            lexical_relevance_by_rank,
            lexical_matches,
            matched_dense_index_ids,
        ) = parent_index_signals(candidates, documents)
        # `lexical_relevance_by_rank` rather than `lexical_matches`, which holds the same memory
        # ids in a set. Both mappings are filled in one pass over `documents`, so this preserves
        # native lexical-route insertion order instead of following the interpreter's string hash
        # seed. The visible effect is narrow -- `read_memories` does not
        # order by its argument, so the ranking and the ranked part of the trace never depended on
        # this, and the ranking sorts on `(-final_score, memory_id)` regardless -- but
        # `extend_missing_memory_traces` walks these ids directly, so a stale-index candidate's
        # position in `search(..., explain=True)` did. A trace is a debugging surface; two runs of one
        # query on one library should print it in one order.
        parent_ids = tuple(dict.fromkeys((*dense_relevance, *lexical_relevance_by_rank)))
        if not parent_ids:
            return search_outcome(
                (),
                trace_candidates,
                candidate_limit=candidate_limit,
                exhaustive=candidates.exhausted,
            )
        with ExitStack() as candidate_lease, ExitStack() as scoring_snapshot:
            document_parts: Iterator[tuple[str, str, bytes]] | None = None
            document_vectors: Iterator[tuple[str, tuple[float, ...]]] | None = None
            with self._write_lock:
                with (
                    self._trace("sensemem.storage.hydrate", kind="stage"),
                    translate_storage_errors("hydrate search results"),
                ):
                    memories = self._store.records.read_memories(
                        parent_ids,
                        valid_at=None if scope is None else scope.valid_at,
                        known_at=None if scope is None else scope.known_at,
                        near=None if scope is None else scope.near,
                        radius_m=None if scope is None else scope.radius_m,
                        place_id=None if scope is None else scope.place_id,
                        identity_id=None if scope is None else scope.identity_id,
                        active_only=True,
                    )
                kept = (
                    None
                    if memory_types is None
                    else frozenset(memory_type.value for memory_type in memory_types)
                )
                eligible_memories = tuple(
                    memory for memory in memories if kept is None or memory.memory_type in kept
                )
                image_matches = (
                    self._image_matches(image_queries, eligible_memories) if image_queries else ()
                )
                # Deletion must not separate authoritative hydration from media protection.
                # Candidates keep short leases; only winners retain them through the operation.
                candidate_lease.enter_context(
                    self._lifecycle.temporary_asset_lease(
                        tuple(asset for memory in eligible_memories for asset in memory.assets)
                    )
                )
                score_ids = tuple(
                    memory.memory_id
                    for memory in eligible_memories
                    if self._settings.index_quantization is not IndexQuantization.NONE
                    or (
                        memory.memory_id in lexical_matches
                        and memory.memory_id not in dense_relevance
                    )
                )
                if score_ids and vectors:
                    with translate_storage_errors("score search candidates"):
                        # BEGIN is deferred: consume the first row under the deletion lock to
                        # establish the WAL snapshot. Keep the stream bounded across SQL batches
                        # and release its reader immediately after scoring, before lexical work.
                        if self._settings.index_quantization is not IndexQuantization.NONE:
                            parts = scoring_snapshot.enter_context(
                                closing(
                                    self._store.index.iter_memory_embedding_parts(
                                        score_ids,
                                        space_id=self._backends.space_id,
                                        task=DOCUMENT_TASK,
                                    )
                                )
                            )
                            first_part = next(parts, None)
                            document_parts = (
                                iter(()) if first_part is None else chain((first_part,), parts)
                            )
                        else:
                            stored_vectors = scoring_snapshot.enter_context(
                                closing(
                                    self._store.index.iter_memory_embedding_vectors(
                                        score_ids,
                                        space_id=self._backends.space_id,
                                        task=DOCUMENT_TASK,
                                    )
                                )
                            )
                            first_vector = next(stored_vectors, None)
                            document_vectors = (
                                iter(())
                                if first_vector is None
                                else chain((first_vector,), stored_vectors)
                            )
            with self._trace("sensemem.retrieval.rank", kind="stage"):
                extend_missing_memory_traces(
                    trace_candidates,
                    parent_ids,
                    memories,
                    index_ids_by_memory,
                    dense_relevance,
                    dense_confidence,
                    lexical_relevance_by_rank,
                    lexical_matches,
                    retrieval_mode=self._settings.retrieval_mode,
                )
                if kept is not None:
                    extend_memory_type_traces(
                        trace_candidates,
                        memories,
                        kept,
                        index_ids_by_memory,
                        dense_relevance,
                        dense_confidence,
                        lexical_relevance_by_rank,
                        lexical_matches,
                        retrieval_mode=self._settings.retrieval_mode,
                    )
                memories = eligible_memories
                if score_ids and vectors:
                    with (
                        self._trace("sensemem.retrieval.score_completion", kind="stage"),
                        translate_storage_errors("score search candidates"),
                    ):
                        if document_parts is not None:
                            part_scores = max_cosine_part_scores(
                                vectors,
                                document_parts,
                                preferred_embeddings=matched_dense_index_ids,
                            )
                            for memory_id, (
                                embedding_id,
                                relevance,
                                confidence,
                            ) in part_scores.items():
                                dense_relevance[memory_id] = relevance
                                dense_confidence[memory_id] = confidence
                                matched_dense_index_ids[memory_id] = embedding_id
                        elif document_vectors is not None:
                            completed = max_cosine_scores(vectors, document_vectors)
                            for memory_id, (relevance, confidence) in completed.items():
                                dense_relevance[memory_id] = relevance
                                dense_confidence[memory_id] = confidence
                with translate_storage_errors("score search candidates"):
                    scoring_snapshot.close()
                ranked, ranked_traces = rank_memories(
                    memories,
                    dense_relevance=dense_relevance,
                    dense_confidence=dense_confidence,
                    lexical_relevance_by_rank=lexical_relevance_by_rank,
                    lexical_matches=lexical_matches,
                    lexical_queries=query.lexical_queries,
                    index_ids_by_memory=index_ids_by_memory,
                    retrieval_mode=self._settings.retrieval_mode,
                    reference_at=reference_at,
                    temporal_range=temporal_range,
                    decay_half_life=self._settings.decay_half_life,
                    capture_trace=trace_candidates is not None,
                )
                ranked = qualified_candidates(
                    ranked,
                    minimum_relevance=self._settings.minimum_relevance,
                    trace_candidates=trace_candidates,
                    ranked_traces=ranked_traces,
                )
                ranked.sort(key=lambda item: (-item[1], item[0].memory_id))
                ambiguous = require_unambiguous and retrieval_is_ambiguous(
                    ranked,
                    margin=self._settings.ambiguity_margin,
                    temporal_range=temporal_range,
                )
                if not ambiguous:
                    ranked = self._rerank_candidates(focused_text, prepared, ranked, ranked_traces)
                extend_ranked_traces(
                    trace_candidates,
                    ranked_traces,
                    ranked,
                    limit=limit,
                    ambiguous=ambiguous,
                )
                visible = () if ambiguous else ranked[:limit]
                self._lifecycle.lease_assets(
                    tuple(
                        asset
                        for memory, _score, _confidence, _lexical in visible
                        for asset in memory.assets
                    ),
                    operation.leased,
                )
                operation.persisted.update(
                    asset.asset_id
                    for memory, _score, _confidence, _lexical in visible
                    for asset in memory.assets
                )
        hits = tuple(
            self._hydrator.search_hit(memory, score)
            for memory, score, _confidence, _lexical in visible
        )
        return search_outcome(
            hits,
            trace_candidates,
            candidate_limit=candidate_limit,
            exhaustive=candidates.exhausted,
            ambiguous=ambiguous,
            matched_dense_index_ids={
                hit.id: matched_dense_index_ids[hit.id]
                for hit in hits
                if hit.id in matched_dense_index_ids
            },
            image_retrieval_matches=image_matches,
        )

    def _prepare_query(
        self,
        prepared: PreparedContent,
        operation: OperationAssets,
        *,
        capture_visual_matches: bool,
    ) -> _QueryRepresentations:
        """Resolve required derivations and encode native, atomic and lexical query routes."""
        prepared = (
            self._speech.with_audio_transcripts(prepared, operation)
            if self._settings.retrieval_mode is RetrievalMode.LEXICAL
            else self._embedding.content(prepared, operation)
        )
        required_visuals = tuple(
            asset
            for asset in prepared.assets
            if Modality(asset.modality) in {Modality.IMAGE, Modality.VIDEO}
            and (
                self._settings.retrieval_mode is RetrievalMode.LEXICAL
                or Modality(asset.modality) not in self._backends.embedding_capabilities
            )
        )
        descriptions = (
            self._vision.query_descriptions(
                prepared
                if self._settings.query_vision
                else replace(prepared, assets=required_visuals),
                operation,
                required=frozenset(
                    asset.asset_id
                    for asset in required_visuals
                    if not has_derived_text(prepared.text, asset.asset_id, Modality(asset.modality))
                ),
            )
            if self._settings.query_vision or required_visuals
            else ()
        )
        if descriptions and required_visuals:
            required_ids = {asset.asset_id for asset in required_visuals}
            prepared = with_descriptions(
                prepared,
                {
                    key: value
                    for key, value in operation.query_descriptions.items()
                    if key in required_ids
                },
            )
        vectors: tuple[tuple[float, ...], ...] = ()
        image_queries: tuple[tuple[str, tuple[float, ...]], ...] = ()
        text_parts = tuple(value for kind, value in prepared.canonical_parts if kind == "text")
        focused_text = text_parts[0] if len(text_parts) > 1 else prepared.text
        # Lexical-only retrieval must not spend a remote query-embedding call.
        if self._settings.retrieval_mode is not RetrievalMode.LEXICAL:
            focused = replace(
                prepared,
                text=focused_text,
                canonical_parts=(("text", focused_text),) if focused_text else (),
            )
            complete = self._embedding.inputs(prepared, maximum_keys=_MAX_QUERY_RETRIEVAL_KEYS)
            aggregate = self._embedding.route(prepared)
            model_inputs = tuple(
                dict.fromkeys(
                    (
                        *((aggregate,) if aggregate in complete else complete),
                        *self._embedding.inputs(
                            focused,
                            maximum_keys=_MAX_QUERY_RETRIEVAL_KEYS,
                        ),
                    )
                )
            )
            encoded = self._embedding.embed(model_inputs, task=EmbedTask.QUERY)
            vectors = tuple(dict.fromkeys(encoded))
            if capture_visual_matches:
                image_queries = tuple(
                    (value.assets[0].id, vector)
                    for value, vector in zip(model_inputs, encoded, strict=True)
                    if not value.text
                    and len(value.assets) == 1
                    and value.assets[0].modality is Modality.IMAGE
                )
            if descriptions and Modality.TEXT in self._backends.embedding_capabilities:
                try:
                    described_vectors = self._embedding.embed(
                        tuple(ModelInput(text=value) for value in descriptions),
                        task=EmbedTask.QUERY,
                    )
                except ModelError as error:
                    _LOGGER.warning("query description embedding failed: %s", error.reason)
                else:
                    vectors = tuple(dict.fromkeys((*vectors, *described_vectors)))
        lexical_queries = tuple(
            dict.fromkeys(
                value for value in (focused_text, *descriptions) if lexical_query_terms(value)
            )
        )
        if self._settings.retrieval_mode is RetrievalMode.DENSE:
            lexical_queries = ()
        return _QueryRepresentations(
            prepared, focused_text, vectors, image_queries, lexical_queries
        )

    def _candidate_pool(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        lexical_queries: Sequence[str],
        limit: int,
        memory_types: frozenset[MemoryType] | None,
        temporal_range: tuple[datetime, datetime] | None,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        scope: RetrievalScope | None,
    ) -> _CandidatePool:
        """Widen native candidates until authoritative scope checks admit enough parents."""
        # Quantized native order can hide a more relevant parent just below the ordinary
        # route window. Keep a wider, request-limit-independent pool for SQLite FP32 scoring.
        candidate_limit = (
            _ROUTE_CANDIDATES
            if self._settings.index_quantization is IndexQuantization.NONE
            else _QUANTIZED_ROUTE_CANDIDATES
        )
        candidate_ceiling = max(candidate_limit, limit * (MAX_RETRIEVAL_KEYS + 1))
        seen_index_ids: set[str] = set()
        # Both are static per document, so the index answers them itself instead of the loop
        # below widening its window until enough survivors of a post-filter turn up. That
        # widening is bounded, and for a one-person or one-room predicate the bound is reached
        # long before the matches are: the pushdown is what makes those scopes recall what the
        # store holds. SQLite still applies the same predicates below and decides.
        scope_place_id = None if scope is None else scope.place_id
        scope_identity_id = self._pushed_identity(scope)
        with self._write_lock:
            # A read asks for the current truth, so it closes an `add_stream` group that is still
            # open on this thread instead of skipping the drain with it: the consumer that searches
            # between two yields must see the item it was just handed.
            self._projection.drain(force=True)
        while True:
            with translate_index_errors("search memories"):
                preferred_range = (
                    None
                    if temporal_range is None
                    else intersect_occurrence_range(
                        temporal_range,
                        occurred_from,
                        occurred_until,
                    )
                )
                candidates = self._index_candidates(
                    vectors,
                    lexical_queries=lexical_queries,
                    limit=candidate_limit,
                    memory_types=memory_types,
                    occurred_from=occurred_from,
                    occurred_until=occurred_until,
                    place_id=scope_place_id,
                    identity_id=scope_identity_id,
                    retrieval_mode=self._settings.retrieval_mode,
                    preferred_range=preferred_range,
                )
            index_ids = tuple(
                dict.fromkeys(hit.id for hit in (*candidates.dense, *candidates.lexical))
            )
            if not index_ids:
                return _CandidatePool(candidates, index_ids, (), (), candidate_limit)
            with (
                self._trace("sensemem.storage.hydrate", kind="stage"),
                translate_storage_errors("hydrate search candidates"),
            ):
                hydrated_documents = self._store.index.read_index_candidates(index_ids)
            documents = hydrated_documents
            if occurred_from is not None or occurred_until is not None:
                documents = tuple(
                    document
                    for document in documents
                    if occurrence_overlaps(
                        document.occurred_at,
                        document.occurred_end,
                        occurred_from,
                        occurred_until,
                    )
                )
            # The survivor count exists only to decide whether to widen, so neither it nor the
            # slate it counts is built where it can no longer change that decision. Every other
            # reason to stop is already known here, and asking first made a second scope pass
            # over the whole candidate window part of the price of every search that was never
            # going to widen -- which is every search whose routes came back short, and every
            # one that had already reached the ceiling.
            if candidates.exhausted or candidate_limit >= candidate_ceiling:
                break
            current_index_ids = set(index_ids)
            if current_index_ids <= seen_index_ids:
                break
            candidate_parent_ids = tuple(
                dict.fromkeys(document.memory_id for document in documents)
            )
            with translate_storage_errors("apply search scope"):
                # Only the number of survivors decides whether to widen, so count them under
                # the same predicates instead of hydrating content, media assets and typed
                # context rows and calling `len()` on the records.
                active_count = self._store.records.count_memories(
                    candidate_parent_ids,
                    valid_at=None if scope is None else scope.valid_at,
                    known_at=None if scope is None else scope.known_at,
                    near=None if scope is None else scope.near,
                    radius_m=None if scope is None else scope.radius_m,
                    # Passed here for consistency with every other scope axis, which all
                    # reach both reads. This one is the survivor count that drives candidate
                    # widening; no constructed corpus (30 or 120 memories) could make its
                    # absence change a result, so it is unproven rather than proven needed.
                    # Kept because omitting one axis at one of two sites is the anomaly a
                    # reader would have to explain, and because narrowing a count can only
                    # widen the search. The hydration site below is mutation-covered.
                    place_id=None if scope is None else scope.place_id,
                    identity_id=None if scope is None else scope.identity_id,
                    active_only=True,
                )
            if active_count >= limit:
                break
            seen_index_ids.update(current_index_ids)
            candidate_limit = min(candidate_limit * 2, candidate_ceiling)
        return _CandidatePool(candidates, index_ids, hydrated_documents, documents, candidate_limit)

    def _rerank_candidates(
        self,
        query: str,
        prepared: PreparedContent,
        ranked: builtins.list[tuple[StoredMemory, float, float, bool]],
        traces: builtins.dict[str, partial[RetrievalCandidateTrace]] | None,
    ) -> builtins.list[tuple[StoredMemory, float, float, bool]]:
        """Score authoritative candidates in supported modalities without changing gates."""
        reranker = self._backends.reranker
        if reranker is None:
            return ranked
        capabilities = self._backends.reranking_capabilities
        query_text = query if Modality.TEXT in capabilities else ""
        query_assets = tuple(
            asset for asset in prepared.assets if Modality(asset.modality) in capabilities
        )
        if not query_text.strip() and not query_assets:
            return ranked
        candidates = [
            (index, row)
            for index, row in enumerate(ranked)
            if (Modality.TEXT in capabilities and row[0].content.strip())
            or any(Modality(asset.modality) in capabilities for asset in row[0].assets)
        ]
        if not candidates:
            return ranked
        assets = tuple(
            asset
            for _index, row in candidates
            for asset in row[0].assets
            if Modality(asset.modality) in capabilities
        )
        with self._lifecycle.temporary_asset_lease(assets):
            query_input = ModelInput(
                text=query_text,
                assets=tuple(self._hydrator.asset_ref(asset) for asset in query_assets),
            )
            documents = tuple(
                ModelInput(
                    text="\n\n".join(
                        window.text for window in query_evidence_windows(query, row[0].content)
                    )
                    if Modality.TEXT in capabilities
                    else "",
                    assets=tuple(
                        self._hydrator.asset_ref(asset)
                        for asset in row[0].assets
                        if Modality(asset.modality) in capabilities
                    ),
                )
                for _index, row in candidates
            )
            modalities = set(query_input.modalities)
            modalities.update(kind for document in documents for kind in document.modalities)
            with self._model_trace(
                "reranking",
                "rerank",
                model=reranker.reranking_model,
                batch_size=len(documents),
                modalities=tuple(sorted(modalities, key=lambda value: value.value)),
            ) as span:
                span.set_attribute("sensemem.reranking.scored_candidates", len(documents))
                mark_model_requests(1)
                try:
                    values = reranker.rerank(query_input, documents)
                except SenseMemError:
                    raise
                except Exception:
                    raise ModelError("reranking model failed", reason="model_failed") from None
                scores = reranking_scores(values, len(documents))
        by_id = {
            row[0].memory_id: score for (_index, row), score in zip(candidates, scores, strict=True)
        }
        ordered = sorted(
            (row for _index, row in candidates),
            key=lambda row: (-by_id[row[0].memory_id], -row[1], row[0].memory_id),
        )
        result = list(ranked)
        for (index, _row), replacement in zip(candidates, ordered, strict=True):
            result[index] = replacement
        if traces is not None:
            for memory_id, score in by_id.items():
                traces[memory_id] = partial(traces[memory_id], reranking_score=score)
        return result

    def _image_matches(
        self,
        queries: Sequence[tuple[str, tuple[float, ...]]],
        memories: Sequence[StoredMemory],
    ) -> tuple[VisualRetrievalMatch, ...]:
        """Preserve image-only provenance within the authoritative, scoped candidate pool."""
        visual = tuple(
            memory
            for memory in memories
            if any(asset.modality == "image" for asset in memory.assets)
        )
        if not visual:
            return ()
        with translate_storage_errors("read image retrieval keys"):
            documents = self._store.index.read_memory_index_documents(
                tuple(memory.memory_id for memory in visual)
            )
        by_memory: builtins.dict[str, builtins.list[StoredEmbedding]] = {}
        for document in documents:
            embedding = document.embedding
            if embedding.space_id == self._backends.space_id and embedding.task == DOCUMENT_TASK:
                by_memory.setdefault(embedding.memory_id, []).append(embedding)
        parts = tuple(
            (memory.memory_id, asset_id, embedding)
            for memory in visual
            for asset_id, embedding in self._embedding.image_parts(
                memory, by_memory.get(memory.memory_id, ())
            )
        )
        result: builtins.list[VisualRetrievalMatch] = []
        for query_id, query in queries:
            scores = max_cosine_scores(
                (query,), ((embedding.embedding_id, embedding.values) for _, _, embedding in parts)
            )
            ordered = sorted(
                parts,
                key=lambda part: (-scores[part[2].embedding_id][1], part[0], part[1]),
            )[:3]
            result.extend(
                VisualRetrievalMatch(
                    query_asset_id=query_id,
                    memory_id=memory_id,
                    source_asset_id=asset_id,
                    rank=rank,
                    cosine_similarity=2.0 * scores[embedding.embedding_id][1] - 1.0,
                )
                for rank, (memory_id, asset_id, embedding) in enumerate(ordered, 1)
            )
        return tuple(result)

    def _pushed_identity(self, scope: RetrievalScope | None) -> str | None:
        """Resolve a scoped identity to the ID the index projection stores, or `None`.

        A merge re-points every projected row onto the survivor, so a request naming a merged
        alias has to be resolved before it can be pushed down. An unknown identity is pushed
        unresolved: it matches nothing in the index, which is what SQLite decides too.
        """
        if scope is None or scope.identity_id is None:
            return None
        with translate_storage_errors("resolve a scoped identity"):
            return (
                self._store.identities.resolve_identity_id(scope.identity_id) or scope.identity_id
            )

    def _index_candidates(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        lexical_queries: Sequence[str],
        limit: int,
        memory_types: frozenset[MemoryType] | None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        preferred_range: tuple[datetime, datetime] | None = None,
    ) -> IndexCandidates:
        # The index filter takes one type, so a set becomes one route per type rather than a
        # post-filter over a shared window. Each type then gets the full `limit` of depth, which
        # is what lets a request for a rare type reach records a common type outranks.
        type_values: tuple[str | None, ...] = (
            (None,)
            if memory_types is None
            else tuple(sorted(memory_type.value for memory_type in memory_types))
        )
        ranges = (
            ((occurred_from, occurred_until),)
            if preferred_range is None
            else (
                preferred_range,
                (occurred_from, occurred_until),
            )
        )
        calls: list[Callable[[], tuple[IndexHit, ...]]] = []
        sizes = []
        for occurred_from, occurred_until in ranges:
            dense_calls = tuple(
                partial(
                    self._index.search,
                    vector,
                    limit=limit,
                    space_id=self._backends.space_id,
                    task=DOCUMENT_TASK,
                    memory_type=value,
                    occurred_from=occurred_from,
                    occurred_until=occurred_until,
                    place_id=place_id,
                    identity_id=identity_id,
                )
                for value in type_values
                for vector in vectors
                if retrieval_mode is not RetrievalMode.LEXICAL
            )
            lexical_calls = (
                tuple(
                    partial(
                        self._index.lexical_search,
                        query,
                        limit=limit,
                        space_id=self._backends.space_id,
                        task=DOCUMENT_TASK,
                        memory_type=value,
                        occurred_from=occurred_from,
                        occurred_until=occurred_until,
                        place_id=place_id,
                        identity_id=identity_id,
                    )
                    for value in type_values
                    for query in lexical_queries
                )
                if retrieval_mode is not RetrievalMode.DENSE
                else ()
            )
            sizes.append((len(dense_calls), len(lexical_calls)))
            calls.extend((*dense_calls, *lexical_calls))
        with self._trace("sensemem.index.search", kind="stage") as span:
            span.set_attribute("sensemem.index.route_count", len(calls))
            if not calls:
                return IndexCandidates(dense=(), lexical=(), exhausted=True)
            routes = self._execution.index_routes(calls)
        groups = []
        offset = 0
        for dense_count, lexical_count in sizes:
            dense_routes = routes[offset : offset + dense_count]
            offset += dense_count
            lexical_routes = routes[offset : offset + lexical_count]
            offset += lexical_count
            groups.append(
                IndexCandidates(
                    dense=merge_index_hits(*dense_routes),
                    lexical=merge_lexical_routes(lexical_routes),
                    exhausted=all(len(route) < limit for route in (*dense_routes, *lexical_routes)),
                )
            )
        if len(groups) == 1:
            return groups[0]
        preferred, fallback = groups
        return IndexCandidates(
            dense=merge_index_hits(preferred.dense, fallback.dense),
            lexical=merge_temporal_lexical_hits(preferred.lexical, fallback.lexical),
            exhausted=preferred.exhausted and fallback.exhausted,
        )

    def note_query_failure(self, text: str, *, failed: bool) -> None:
        """Record one empty recall, which is the whole of the QUERY_FAILURE signal.

        The stored query is the owner's own words about their own memory, in their own memory
        domain, and the table is capped on write so it stays a signal buffer rather than a
        growing query log. A non-text query records nothing: there is nothing to compare two of
        them by.
        """
        if not failed:
            return
        folded = unicodedata.normalize("NFKC", text).casefold()
        normalized = _QUERY_SPACE.sub(" ", _QUERY_NOISE.sub(" ", folded)).strip()
        if not normalized:
            return
        with self._write_lock, translate_storage_errors("record a query failure"):
            self._store.control.record_query_failure(
                text,
                normalized,
                failed_at=datetime.now(timezone.utc),
                keep=self._settings.query_failure_history,
            )
