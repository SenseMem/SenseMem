"""Shared bounded recall: planning, authoritative reads, and linked evidence expansion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from time import perf_counter

from opentelemetry.trace import Tracer

from sensemem._telemetry import (
    EXPANSION_DROPPED_EDGES,
    EXPANSION_EDGES,
    EXPANSION_LINKED_ROWS,
    EXPANSION_ROWS,
    RECALL_COMPLETE,
    RECALL_EXHAUSTIVE_ROWS,
    RECALL_FALLBACK,
    RECALL_NON_SELECTIVE_STEPS,
    RECALL_OPS,
    RECALL_REPLAN,
    RECALL_SHAPE,
    mark_model_requests,
)
from sensemem.context_values import ContextBudget, ContextUnknown, ContextUnknownKind
from sensemem.exceptions import ModelError
from sensemem.infrastructure.local.store._codec import datetime_text
from sensemem.infrastructure.local.store.local_store import LocalStore
from sensemem.infrastructure.local.store.rows import RecallRead
from sensemem.kernel import grounding
from sensemem.kernel.content import PreparedContent
from sensemem.kernel.contracts import generation_model_name
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.lifecycle import OperationAssets
from sensemem.kernel.retrieval import RERANK_CANDIDATES, Retrieval
from sensemem.kernel.settings import ResolvedSettings
from sensemem.kernel.temporal import (
    rows_time_span,
)
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import (
    pushed_memory_types,
)
from sensemem.models.base import (
    RecallPlanningBackend,
)
from sensemem.prompts import context as context_prompts
from sensemem.prompts import recall as recall_prompts
from sensemem.recall import (
    RecallProgram,
    RecallResult,
    RecallRows,
    execute_recall_program,
    fallback_recall_program,
    parse_recall_program,
)
from sensemem.records import SearchHit
from sensemem.spatial import RetrievalScope
from sensemem.types import MemoryCategory, Modality


@dataclass(frozen=True, slots=True)
class RecallRequest:
    """Immutable inputs shared by bounded answering and context-compilation reads."""

    prepared: PreparedContent
    assets: OperationAssets
    # The original limit-sized window stays the grounding floor. Semantic expansion can replace
    # its supplemental ranked tail without comparing scores from different queries.
    ranked: tuple[SearchHit, ...]
    ranked_depth: int
    limit: int
    reference: datetime
    temporal_range: tuple[datetime, datetime] | None
    scope: RetrievalScope | None
    memory_type: MemoryCategory | None


class RecallReads:
    """Run a recall plan's reads against one store, inside the caller's open operation.

    The exhaustive ops read SQLite through the store's own primitives and are hydrated the way
    every other hit is. They carry no relevance score, because they earned none: they are in the
    evidence set because they match the question's predicate, not because they ranked.
    """

    def __init__(
        self,
        store: LocalStore,
        hydrator: Hydrator,
        context: RecallRequest,
        retrieval: Retrieval | None = None,
        *,
        max_queries: int,
    ) -> None:
        self._store = store
        self._hydrator = hydrator
        self._context = context
        self._retrieval = retrieval
        self._max_queries = max_queries
        original = context.prepared.text.strip()
        self._cache = {original: context.ranked}
        self._depths = {original: context.ranked_depth}
        self._semantic_depths: dict[str, int] = {}
        self._unavailable: set[str] = set()
        self.queries = 0
        self.failures = 0
        self.limited: set[str] = set()

    def similar(self, query: str, *, k: int) -> RecallRows:
        if self._retrieval is None:
            rows = self._context.ranked[:k]
            return RecallRows(rows, len(rows))
        query = query.strip()
        if k > self._depths.get(query, 0):
            if self.queries >= self._max_queries:
                self.limited.add(query)
                self._unavailable.add(query)
                return RecallRows(self._cache.get(query, ())[:k], k)
            self.queries += 1
            # Query strings are untrusted model output, interpreted only as text. A rewrite
            # cannot replace the original media, reference clock, temporal intent or scope.
            prepared = (
                self._context.prepared
                if query == self._context.prepared.text.strip()
                else PreparedContent(query, (), Modality.TEXT, (("text", query),))
            )
            try:
                rows = self._retrieval.search_prepared(
                    prepared,
                    limit=RERANK_CANDIDATES,
                    operation=self._context.assets,
                    memory_types=pushed_memory_types(self._context.memory_type),
                    reference_at=self._context.reference,
                    temporal_range=self._context.temporal_range,
                    occurred_from=None,
                    occurred_until=None,
                    scope=self._context.scope,
                    require_unambiguous=False,
                    capture_trace=False,
                ).hits
            except ModelError:
                # An optional expansion failure must not cost the evidence already earned.
                # Cache the failure too: repeated steps and replans do not retry a paid call.
                self.failures += 1
                self._unavailable.add(query)
                rows = self._cache.get(query, ())
            else:
                self._semantic_depths.setdefault(query, 0)
            self._cache[query] = rows
            self._depths[query] = RERANK_CANDIDATES
        if query in self._semantic_depths:
            self._semantic_depths[query] = max(self._semantic_depths[query], k)
        rows = self._cache[query][:k]
        return RecallRows(rows, len(rows))

    def semantic_routes(self) -> tuple[tuple[SearchHit, ...], ...]:
        """Retain successful reads across rounds, in discovery order at requested depth.

        A later query may depend on an earlier read's bridge fact. Keep both routes eligible
        for the same bounded packing pass even when the later plan names only the new query.
        """
        return tuple(self._cache[query][:depth] for query, depth in self._semantic_depths.items())

    def mark_unread(self, program: RecallResult) -> RecallResult:
        """A failed anchor cannot make its dependent exhaustive reads complete."""
        if not self._unavailable:
            return program
        unread: set[int] = set()
        executed = list(program.executed)
        for index, step in enumerate(program.plan.steps):
            if (
                (step.op == "similar" and (step.query or "").strip() in self._unavailable)
                or step.of in unread
                or step.time_of in unread
            ):
                unread.add(index)
                executed[index] = replace(executed[index], bounded=True)
        return replace(program, executed=tuple(executed))

    def match(
        self,
        terms: Sequence[str],
        *,
        any_of: bool,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        modality: Modality | None,
        memory_type: MemoryCategory | None,
        max_rows: int,
        entity_id: str | None = None,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.match_memories(
                tuple(terms),
                any_of=any_of,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                modality=None if modality is None else modality.value,
                memory_type=self._memory_type(memory_type),
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
                entity_id=entity_id,
            )
        )

    def window(
        self,
        *,
        occurred_from: datetime,
        occurred_until: datetime,
        modality: Modality | None,
        memory_type: MemoryCategory | None,
        max_rows: int,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.memories_in_window(
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                modality=None if modality is None else modality.value,
                memory_type=self._memory_type(memory_type),
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
            )
        )

    def neighbors(
        self,
        memory_ids: Sequence[str],
        *,
        before: int,
        after: int,
        max_rows: int,
    ) -> RecallRows:
        scope = self._context.scope or RetrievalScope()
        return self._hits(
            self._store.recall.neighbor_memories(
                tuple(memory_ids),
                before=before,
                after=after,
                max_rows=max_rows,
                valid_at=scope.valid_at,
                known_at=scope.known_at,
                near=scope.near,
                radius_m=scope.radius_m,
                place_id=scope.place_id,
                identity_id=scope.identity_id,
            )
        )

    def entity(self, name: str, *, max_rows: int) -> RecallRows:
        """Read what is known about one person, by identity when the name resolves to one.

        A name no identity carries is not an error: the store may hold the person only as words
        in other people's memories, so the read degrades to matching the name as text.
        """
        identity_id = self._store.recall.identity_id_for_name(
            name,
            identity_id=None if self._context.scope is None else self._context.scope.identity_id,
        )
        return self.match(
            () if identity_id is not None else (name,),
            any_of=True,
            occurred_from=None,
            occurred_until=None,
            modality=None,
            memory_type=None,
            max_rows=max_rows,
            entity_id=identity_id,
        )

    def related(
        self,
        memory_ids: Sequence[str],
        *,
        ceiling: int,
        max_rows: int,
    ) -> grounding.EvidenceExpansion:
        """Read what the grounded evidence is structurally linked to, under the caller's scope."""
        scope = self._context.scope or RetrievalScope()
        read = self._store.recall.related_memories(
            tuple(memory_ids),
            ceiling=ceiling,
            max_rows=max_rows,
            valid_at=scope.valid_at,
            known_at=scope.known_at,
            near=scope.near,
            radius_m=scope.radius_m,
            place_id=scope.place_id,
            identity_id=scope.identity_id,
        )
        return grounding.EvidenceExpansion(
            hits=self._hits(read).rows,
            edges=read.edges,
            dropped=read.dropped,
        )

    def time_span(self, rows: Sequence[SearchHit]) -> tuple[datetime, datetime] | None:
        """Bound a later recall read by these rows' event times and stated dates."""
        return rows_time_span(rows, self._context.reference)

    def chronological(self, rows: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
        """Keep SQLite's corpus order when timestamps tie across different reads."""
        by_id = {hit.id: hit for hit in rows}
        return tuple(
            by_id[memory_id] for memory_id in self._store.recall.order_memory_ids(tuple(by_id))
        )

    def _memory_type(self, requested: MemoryCategory | None) -> str | None:
        """The caller's own memory_type wins: a plan may not widen the question's scope."""
        chosen = self._context.memory_type or requested
        return None if chosen is None else chosen.value

    def _hits(self, read: RecallRead) -> RecallRows:
        """Hydrate one primitive's rows, keeping the count its predicate selected.

        Metric scope and the caller's memory type may reduce the selected rows. Keep the
        selection count so filtering cannot hide that SQL's bound truncated the read.
        """
        return RecallRows(
            tuple(
                self._hydrator.search_hit(memory, 0.0)
                for memory in read
                if self._context.memory_type is None
                or memory.memory_type == self._context.memory_type.value
            ),
            read.selected,
        )


class Recall(Traced):
    """Plan and execute reads for answering and compilation without generating answers."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        store: LocalStore,
        settings: ResolvedSettings,
        hydrator: Hydrator,
        retrieval: Retrieval,
        planner: RecallPlanningBackend | None,
    ) -> None:
        super().__init__(tracer)
        self._store = store
        self._settings = settings
        self._hydrator = hydrator
        self._retrieval = retrieval
        self._planner = planner

    def reads(self, context: RecallRequest) -> RecallReads:
        return RecallReads(
            self._store,
            self._hydrator,
            context,
            self._retrieval if self._settings.recall_query_expansion else None,
            max_queries=self._settings.recall_max_queries,
        )

    def for_context(
        self,
        prepared: PreparedContent,
        ranked: tuple[SearchHit, ...],
        *,
        candidate_limit: int,
        budget: ContextBudget,
        reference_at: datetime,
        temporal_range: tuple[datetime, datetime] | None,
        scope: RetrievalScope | None,
        assets: OperationAssets,
        started_at: float,
    ) -> tuple[tuple[SearchHit, ...], tuple[ContextUnknown, ...]]:
        """Run the configured bounded recall program without generating an answer.

        The compiler retains ownership of final source closure, consent, filters, and budgets.
        Planned rows precede the original ranking. The compiler uses that order for selection;
        no original hit occupies a mandatory budget slot or receives a cross-query score.
        """
        if not self._settings.recall_planning:
            return ranked, ()
        if self._planner is None:
            return ranked, (
                ContextUnknown(
                    kind=ContextUnknownKind.STAGE_SKIPPED,
                    detail=context_prompts.RECALL_PLANNER_UNAVAILABLE,
                ),
            )
        if (
            budget.max_latency_ms is not None
            and (perf_counter() - started_at) * 1_000 >= budget.max_latency_ms
        ):
            return ranked, (
                ContextUnknown(
                    kind=ContextUnknownKind.STAGE_SKIPPED,
                    detail=context_prompts.RECALL_DEADLINE,
                ),
            )
        # The planner is an optional protocol implemented by the configured generation backend;
        # this operation uses its planning method only, never answer() or stream_answer().
        context = RecallRequest(
            prepared=prepared,
            assets=assets,
            ranked=ranked,
            ranked_depth=candidate_limit,
            limit=min(100, budget.max_items),
            reference=reference_at,
            temporal_range=temporal_range,
            scope=scope,
            memory_type=(
                next(iter(budget.memory_types))
                if budget.memory_types and len(budget.memory_types) == 1
                else None
            ),
        )
        reads = self.reads(context)
        program = self.execute(context, reads, attempted="")
        assert program is not None
        if program.plan == fallback_recall_program(prepared.text, k=context.limit):
            return ranked, (
                ContextUnknown(
                    kind=ContextUnknownKind.STAGE_SKIPPED,
                    detail=recall_prompts.NO_DISTINCT_READ,
                ),
            )
        unknowns = [
            ContextUnknown(
                kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                detail=recall_prompts.unsupported_read(index, step.op),
            )
            for index, step in enumerate(program.executed)
            if step.rows == 0
        ]
        offered = {
            hit.id: hit
            for hit in (
                *program.exhaustive,
                *grounding.semantic_hits(reads.semantic_routes(), set(), context.limit),
            )
        }
        if self._settings.evidence_expansion:
            for hit in self.expand(tuple(offered.values()) or ranked[:1], context).hits:
                offered.setdefault(hit.id, hit)
        supplemental, _added = grounding.pack_grounding(
            (),
            tuple(offered.values()),
            (),
            self._settings.recall_set_budget,
            media_limit=budget.max_media_items
            if budget.max_media_items is not None
            else budget.max_items,
            max_added=self._settings.recall_set_max_rows,
        )
        if (
            not program.complete
            or reads.failures
            or reads.limited
            or len(supplemental) < len(offered)
        ):
            unknowns.append(
                ContextUnknown(
                    kind=ContextUnknownKind.CANDIDATES_EXHAUSTED,
                    detail=context_prompts.RECALL_READS_INCOMPLETE,
                )
            )
        candidates = {hit.id: hit for hit in (*supplemental, *ranked)}
        return tuple(candidates.values()), tuple(unknowns)

    def execute(
        self, context: RecallRequest, reads: RecallReads, *, attempted: str
    ) -> RecallResult | None:
        """Plan and run this question's reads, or None when planning is off.

        Inside its own stage: the sensemem.retrieve leg closes on the ranked hits, and the
        planning call and the exhaustive reads happen after it -- a model call and a set of table
        scans that a trace without this span attributes to nothing, leaving the operation's own
        duration as the only evidence they ran at all.
        """
        if not self._settings.recall_planning:
            return None
        with self._trace("sensemem.recall", kind="stage") as span:
            question, k = context.prepared.text, context.limit
            plan = self.plan(
                question,
                reference_at=context.reference,
                k=k,
                attempted=attempted,
            )
            program = execute_recall_program(
                plan,
                reads,
                limit=k,
                # The digest the planner was already handed, recomputed only after a commit, so
                # the selectivity of every step is decided against one cached read per question
                # rather than a count per step.
                active_records=self._store.recall.recall_digest().records,
            )
            program = reads.mark_unread(program)
            span.set_attributes(
                {
                    RECALL_SHAPE: plan.shape,
                    RECALL_OPS: tuple(step.op for step in plan.steps),
                    RECALL_EXHAUSTIVE_ROWS: len(program.exhaustive),
                    RECALL_COMPLETE: program.complete,
                    RECALL_NON_SELECTIVE_STEPS: program.non_selective_steps,
                    RECALL_REPLAN: bool(attempted),
                    # Every way planning can fail resolves to this same plan, so equality with it
                    # is the only signal that says the planner did not decide this question --
                    # which is what a result claiming the feature ran has to be read against.
                    RECALL_FALLBACK: plan == fallback_recall_program(question, k=k),
                }
            )
            return program

    def expand(
        self, hits: Sequence[SearchHit], context: RecallRequest
    ) -> grounding.EvidenceExpansion:
        """Read the records this evidence is structurally linked to, at no model cost.

        Every linked row is offered, including one the ranking also found. Discarding those was
        tried, on the theory that a wider budget reaches them anyway, and it removed the whole
        benefit: with no character budget the candidate pool is min(100, limit * 3) while the
        window it grounds is limit, so a gold row at rank 20 is in the pool and will never be in
        the window. Gating on pool membership dropped exactly those rows -- complete support fell
        to the narrow window's own 0.7949 on ATM-Bench and the paired comparison went to 117 ties
        -- and kept the rows past rank 36, which are the ones that do not matter.

        What the measurement left is a simpler reading of this feature. Expansion does not reach
        evidence the ranking cannot; it reaches further down the same ranking by a different
        selector, and it is worth enabling where that selector is cheaper than cosine depth. The
        rows a wider window would have contained anyway are removed by grounding.pack_grounding's own
        dedup, which is the only place that can know what the window will hold.
        """
        if not hits:
            return grounding.EvidenceExpansion(hits=(), edges={}, dropped=())
        with self._trace("sensemem.expand", kind="stage") as span:
            expansion = RecallReads(
                self._store,
                self._hydrator,
                context,
                max_queries=self._settings.recall_max_queries,
            ).related(
                tuple(hit.id for hit in hits),
                ceiling=grounding.expansion_ceiling(
                    context.limit,
                    self._store.recall.recall_digest().records,
                ),
                max_rows=self._settings.evidence_expansion_max_rows,
            )
            reachable = {hit.id for hit in context.ranked}
            span.set_attributes(
                {
                    EXPANSION_ROWS: len(expansion.hits),
                    # How many of the linked rows the ranking also proposed. Kept as a signal
                    # rather than a filter: gating on it was measured harmful, and the number is
                    # still what says whether a corpus's captures track its ranking.
                    EXPANSION_LINKED_ROWS: sum(1 for hit in expansion.hits if hit.id in reachable),
                    EXPANSION_EDGES: tuple(
                        f"{edge}:{count}" for edge, count in expansion.edges.items() if count
                    ),
                    EXPANSION_DROPPED_EDGES: expansion.dropped,
                }
            )
            return expansion

    def plan(
        self,
        question: str,
        *,
        reference_at: datetime,
        k: int,
        attempted: str = "",
    ) -> RecallProgram:
        """Ask the answerer how to read for this question, or keep today's single search.

        Every way this can go wrong lands on the same fallback -- a backend that cannot plan, a
        planner that fails, an empty completion, a plan the kernel will not run -- so the cost
        of a bad planner is one model call, never a different answer. Model output is data: it
        reaches parse_recall_program, which decides what SenseMem does.
        """
        planner = self._planner
        if planner is None or not question.strip():
            return fallback_recall_program(question, k=k)
        model = generation_model_name(planner)
        with self._model_trace(
            "generation",
            "plan",
            model=model,
            batch_size=1,
            modalities=(Modality.TEXT,),
        ):
            mark_model_requests(1)
            digest = self.corpus_digest()
            try:
                payload = planner.plan_recall(
                    question,
                    reference_at=reference_at,
                    corpus_digest=digest,
                    attempted=attempted,
                )
            except ModelError:
                # Planning is not answering. A provider that refused this call has said nothing
                # about whether it can answer, and the fallback plan is the one ask has always
                # run, so the question continues instead of failing on a hint.
                return fallback_recall_program(question, k=k)
        if payload is None:
            return fallback_recall_program(question, k=k)
        return parse_recall_program(payload, reference_at=reference_at) or fallback_recall_program(
            question, k=k
        )

    def corpus_digest(self) -> str:
        """Describe the active corpus in one line, so a plan cannot ask for what is not there."""
        digest = self._store.recall.recall_digest()
        return recall_prompts.corpus_digest(
            digest.records,
            earliest=None if digest.earliest is None else datetime_text(digest.earliest),
            latest=None if digest.latest is None else datetime_text(digest.latest),
            modalities=digest.modalities,
            named_identities=digest.named_identities,
            max_queries=(
                self._settings.recall_max_queries if self._settings.recall_query_expansion else None
            ),
        )
