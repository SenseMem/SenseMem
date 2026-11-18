"""The compile plane: closed, budgeted context with optional goal-directed recall.

Selection and structuring are pure and live in `sensemem.context`; this module owns the
store-backed reads that close evidence over lineage, consent, and actors before that pure step
runs.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from time import perf_counter

from opentelemetry.trace import Tracer

from sensemem.content import ContentInput
from sensemem.context import (
    EvidenceClosure,
    NamedActorLink,
    _rejection,
    bundle_cost,
    compile_context,
)
from sensemem.context_bundle import ContextBundle
from sensemem.context_selection import ContextSelector
from sensemem.context_values import (
    ContextBudget,
    ContextExcerpt,
    ContextProof,
    ContextProofScoring,
    ContextSupplement,
    ContextUnknown,
    ContextUnknownKind,
)
from sensemem.evidence import AffectLinkKey, affect_link_key, functional_lineage
from sensemem.evidence_policy import required_support
from sensemem.exceptions import ValidationError
from sensemem.kernel.content import PreparedContent, verified_context_excerpt
from sensemem.kernel.contracts import Backends
from sensemem.kernel.corroboration import SupportSummary, independent_support, naming_conditions
from sensemem.kernel.hydration import Hydrator
from sensemem.kernel.lifecycle import Lifecycle
from sensemem.kernel.materialization import Materializer
from sensemem.kernel.proofs import certificates, select_certificates
from sensemem.kernel.recall import Recall
from sensemem.kernel.retrieval import RERANK_CANDIDATES, Retrieval
from sensemem.kernel.runtime import Storage, translate_storage_errors
from sensemem.kernel.settings import ResolvedSettings
from sensemem.kernel.speech import Speech
from sensemem.kernel.temporal import query_temporal_range, resolved_reference_at, temporal_context
from sensemem.kernel.tracing import Traced
from sensemem.kernel.validation import (
    modality_names,
    pushed_memory_types,
    scope_description,
    validated_retrieval_scope,
)
from sensemem.prompts import context as context_prompts
from sensemem.records import SearchHit
from sensemem.spatial import RetrievalScope
from sensemem.types import Modality, RecordKind

# How much deeper `compile()` ranks when the budget carries a bound only the compiler can
# apply -- `min_confidence` or `freshness`. Those remove candidates after retrieval, so the
# window has to hold what they will remove for `candidates_exhausted` to mean the store ran
# out rather than the window did.
_POST_FILTER_WINDOW = 3


# Context compilation may complete an affirmative functional claim with low-relevance competing
# values.  This is an obligation lookup, never a second relevance route, so a finite cap must
# refuse rather than silently make a prefix look complete.
_LINEAGE_COMPLETION_REPRESENTATIVES = 8
_AFFECT_LINK_COMPLETION_LIMIT = 128


def _functional_representatives(
    hits: Sequence[SearchHit],
) -> dict[str, dict[str, list[str]]]:
    """Group candidate IDs by functional lineage and value, preserving rank order."""
    values_by_lineage: dict[str, dict[str, list[str]]] = {}
    for hit in hits:
        functional = functional_lineage(hit.context)
        if functional is None:
            continue
        lineage_id, value = functional
        values_by_lineage.setdefault(lineage_id, {}).setdefault(value, []).append(hit.id)
    return values_by_lineage


def _incomplete_functional_lineages(
    competing_values: Mapping[str, Mapping[str, Sequence[str]]],
    candidates: Mapping[str, SearchHit],
    closure_ids: Collection[str],
    budget: ContextBudget,
    reference_at: datetime,
    capped_lineages: Collection[str],
) -> set[str]:
    """Return lineages with a value that has no filter-eligible evidence closure."""
    incomplete = set(capped_lineages)
    for lineage_id, values in competing_values.items():
        if any(
            not any(
                memory_id in closure_ids
                and _rejection(candidates[memory_id], budget, reference_at) is None
                for memory_id in representatives
            )
            for representatives in values.values()
        ):
            incomplete.add(lineage_id)
    return incomplete


def _consent_withheld_required_closure(
    closures: Sequence[EvidenceClosure],
    consent_closure_ids: Collection[str],
    values_by_lineage: Mapping[str, Mapping[str, Sequence[str]]],
    candidates: Mapping[str, SearchHit],
    budget: ContextBudget,
    reference_at: datetime,
) -> bool:
    """Whether consent removed a closure without leaving a same-value alternative."""
    for closure in closures:
        if closure.anchor.id in consent_closure_ids:
            continue
        functional = functional_lineage(closure.anchor.context)
        if functional is None:
            return True
        lineage_id, value = functional
        if not any(
            representative_id in consent_closure_ids
            and _rejection(candidates[representative_id], budget, reference_at) is None
            for representative_id in values_by_lineage[lineage_id][value]
        ):
            return True
    return False


def _deliverable_functional_representatives(
    hits: Sequence[SearchHit],
    closure_ids: Collection[str],
    budget: ContextBudget,
    reference_at: datetime,
) -> set[str]:
    """Choose the first deliverable candidate for every functional value."""
    selected_ids: set[str] = set()
    selected_values: set[tuple[str, str]] = set()
    for hit in hits:
        functional = functional_lineage(hit.context)
        if functional is None or functional in selected_values:
            continue
        if hit.id not in closure_ids or _rejection(hit, budget, reference_at) is not None:
            continue
        selected_values.add(functional)
        selected_ids.add(hit.id)
    return selected_ids


def _closure_contains_functional_lineage(
    closure: EvidenceClosure,
    lineage_ids: Collection[str],
) -> bool:
    """Whether one closure carries a claim in a lineage this compilation withheld.

    Completion finds unsafe lineages only for ranked functional anchors.  A separately ranked
    derived assertion may still cite one as support, and closed selection would otherwise
    materialize that support through the derived closure.  This check deliberately makes no new
    discovery beneath arbitrary support: it only enforces the concrete obligations completion
    already found for this compilation.
    """
    return any(
        functional is not None and functional[0] in lineage_ids
        for member in closure.members
        if (functional := functional_lineage(member.context)) is not None
    )


class Compilation(Traced):
    """Close evidence over lineage, consent, and actors, then compile a context bundle."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        storage: Storage,
        settings: ResolvedSettings,
        backends: Backends,
        lifecycle: Lifecycle,
        hydrator: Hydrator,
        materializer: Materializer,
        speech: Speech,
        retrieval: Retrieval,
        recall: Recall,
        context_selector: ContextSelector,
    ) -> None:
        super().__init__(tracer)
        self._store = storage.store
        self._proof_scoring: ContextProofScoring = (
            "independent_pair" if settings.independent_evidence else "capture_groups"
        )
        self._backends = backends
        self._lifecycle = lifecycle
        self._hydrator = hydrator
        self._materializer = materializer
        self._speech = speech
        self._retrieval = retrieval
        self._recall = recall
        self._context_selector = context_selector

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        allow_partial_sources: bool = False,
        supplements: Sequence[ContextSupplement] = (),
    ) -> ContextBundle:
        with (
            self._trace("sensemem.compile", kind="operation"),
            self._lifecycle.operation(model_work=True) as assets,
        ):
            # Taken before anything else, so `budget.max_latency_ms` bounds the compilation a
            # caller waits for rather than the selection pass alone.
            started_at = perf_counter()
            budget = ContextBudget() if budget is None else budget
            if not isinstance(budget, ContextBudget):
                raise ValidationError("budget must be a ContextBudget")
            if not isinstance(allow_partial_sources, bool):
                raise ValidationError("allow_partial_sources must be a boolean")
            try:
                supplied_context = tuple(supplements)
            except TypeError:
                raise ValidationError("supplements must contain ContextSupplement values") from None
            if any(not isinstance(item, ContextSupplement) for item in supplied_context):
                raise ValidationError("supplements must contain ContextSupplement values")
            scope = validated_retrieval_scope(scope)
            with self._trace("sensemem.content.prepare", kind="stage"):
                prepared = self._materializer.materialize_content(goal, assets)
            explicit_reference = resolved_reference_at(reference_at)
            reference, temporal_text = temporal_context(
                prepared.text,
                explicit_reference or datetime.now(timezone.utc),
                infer_reference=explicit_reference is None,
            )
            # `validate_limit` bounds what a caller may ask a public search to return, not how deep the
            # kernel ranks, so `_search_prepared` has no ceiling of its own and a bundle may rank
            # past one hundred. Three candidates per slot is the same headroom `ask()` gives its
            # modality round robin -- widened when the budget carries a bound the index cannot
            # answer, because those are applied to the window after it comes back and a window
            # sized for the bundle alone would report exhaustion the store does not have.
            candidate_limit = max(RERANK_CANDIDATES, budget.max_items * 3) * (
                _POST_FILTER_WINDOW
                if budget.min_confidence > 0.0 or budget.freshness is not None
                else 1
            )
            outcome = self._retrieval.search_prepared(
                prepared,
                limit=candidate_limit,
                operation=assets,
                # `memory_types` is index-pushable one type at a time, so the kernel opens one
                # route per type. Without that a request for a rare type competes for the same
                # window as every common one and comes back empty while the store holds records.
                memory_types=pushed_memory_types(budget.memory_types),
                reference_at=reference,
                temporal_range=query_temporal_range(temporal_text, reference),
                occurred_from=None,
                occurred_until=None,
                scope=scope,
                require_unambiguous=False,
                capture_trace=False,
            )
            # The same transcript cache `search()` writes for a spoken query. It is a cache of
            # the query's own audio, not a memory: `compile()` stores nothing it retrieved.
            self._speech.persist_transcripts(assets)
            ranked, recall_unknowns = self._recall.for_context(
                prepared,
                outcome.hits,
                candidate_limit=candidate_limit,
                budget=budget,
                reference_at=reference,
                temporal_range=query_temporal_range(temporal_text, reference),
                scope=scope,
                assets=assets,
                started_at=started_at,
            )
            completion_hits, capped_lineages = self._competing_lineage_hits(
                ranked,
                scope=scope,
            )
            candidate_by_id = {hit.id: hit for hit in ranked}
            for hit in completion_hits:
                candidate_by_id.setdefault(hit.id, hit)
            candidates = tuple(candidate_by_id.values())
            closures, unavailable = self._evidence_closures(
                candidates,
                budget=budget,
                reference_at=reference,
                scope=scope,
                started_at=started_at,
            )
            candidates, closures, affect_unavailable = self._complete_affect_links(
                candidates,
                closures,
                budget=budget,
                reference_at=reference,
                scope=scope,
                started_at=started_at,
            )
            candidate_by_id = {hit.id: hit for hit in candidates}
            unavailable = (*unavailable, *affect_unavailable)
            closure_ids = frozenset(closure.anchor.id for closure in closures)
            values_by_lineage = _functional_representatives(candidates)
            competing_values = {
                lineage_id: values
                for lineage_id, values in values_by_lineage.items()
                if len(values) > 1
            }
            incomplete_lineages = _incomplete_functional_lineages(
                competing_values,
                candidate_by_id,
                closure_ids,
                budget,
                reference,
                capped_lineages,
            )
            if incomplete_lineages:
                # A functional assertion is never affirmative by itself once its indexed
                # lineage completion found an unrenderable competitor or reached its cap.  The
                # raw observations supporting it remain ordinary evidence, but every claim in
                # the unsafe lineage is withheld before the compiler can fall back to flat rank.
                withheld_ids = frozenset(
                    hit.id
                    for hit in candidates
                    if (functional := functional_lineage(hit.context)) is not None
                    and functional[0] in incomplete_lineages
                )
                ranked_withheld = sum(1 for hit in ranked if hit.id in withheld_ids)
                candidates = tuple(hit for hit in candidates if hit.id not in withheld_ids)
                closures = tuple(
                    closure
                    for closure in closures
                    if not _closure_contains_functional_lineage(closure, incomplete_lineages)
                )
                unavailable = (
                    *unavailable,
                    ContextUnknown(
                        kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                        detail=(context_prompts.competing_evidence_required(ranked_withheld)),
                    ),
                )
            closure_hits = tuple(
                {
                    hit.id: hit
                    for hit in (
                        *candidates,
                        *(hit for closure in closures for hit in closure.members),
                    )
                }.values()
            )
            provisional = self._provisional_identities(closure_hits, scope=scope)
            permitted, named, withheld, restrained = self._consented_actors(
                closure_hits,
                self._named_actors(closure_hits, scope=scope),
                observed_identity_ids=frozenset(
                    identity_id
                    for identity_ids in provisional.values()
                    for identity_id in identity_ids
                ),
            )
            permitted_ids = {hit.id for hit in permitted}
            closures_before_consent = closures
            closures = tuple(
                closure
                for closure in closures
                if all(member.id in permitted_ids for member in closure.members)
            )
            hits = tuple(hit for hit in candidates if hit.id in permitted_ids)
            consent_closure_ids = frozenset(closure.anchor.id for closure in closures)
            withheld_closures = _consent_withheld_required_closure(
                closures_before_consent,
                consent_closure_ids,
                values_by_lineage,
                candidate_by_id,
                budget,
                reference,
            )
            consent_incomplete_lineages = {
                lineage_id
                for lineage_id, values in competing_values.items()
                if lineage_id not in incomplete_lineages
                if any(
                    not any(
                        memory_id in permitted_ids and memory_id in consent_closure_ids
                        for memory_id in representatives
                    )
                    for representatives in values.values()
                )
            }
            if consent_incomplete_lineages:
                # Consent cannot turn a previously known competitor into a silent agreement.  Do
                # not name the withheld identity or value here: the only safe presentation is
                # that the affirmative claim's competing evidence is unavailable.
                consent_withheld_ids = frozenset(
                    hit.id
                    for hit in hits
                    if (functional := functional_lineage(hit.context)) is not None
                    and functional[0] in consent_incomplete_lineages
                )
                hits = tuple(hit for hit in hits if hit.id not in consent_withheld_ids)
                closures = tuple(
                    closure
                    for closure in closures
                    if not _closure_contains_functional_lineage(
                        closure, consent_incomplete_lineages
                    )
                )
                unavailable = (
                    *unavailable,
                    ContextUnknown(
                        kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                        detail=context_prompts.COMPETING_CONSENT_WITHHELD,
                    ),
                )
            consent_closure_ids = frozenset(closure.anchor.id for closure in closures)
            consent_incomplete_attributions = {
                key
                for hit in candidates
                if (key := affect_link_key(hit)) is not None
                and (hit.id not in permitted_ids or hit.id not in consent_closure_ids)
            }
            if consent_incomplete_attributions:
                hits = tuple(
                    hit
                    for hit in hits
                    if affect_link_key(hit) not in consent_incomplete_attributions
                )
                closures = tuple(
                    closure
                    for closure in closures
                    if all(
                        affect_link_key(member) not in consent_incomplete_attributions
                        for member in closure.members
                    )
                )
                unavailable = (
                    *unavailable,
                    ContextUnknown(
                        kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                        detail=context_prompts.AFFECT_CONSENT_WITHHELD,
                    ),
                )
            consent_closure_ids = frozenset(closure.anchor.id for closure in closures)
            deliverable_functional_ids = _deliverable_functional_representatives(
                hits, consent_closure_ids, budget, reference
            )
            hits = tuple(
                hit
                for hit in hits
                if (functional := functional_lineage(hit.context)) is None
                or hit.id in deliverable_functional_ids
            )
            closures = tuple(
                closure
                for closure in closures
                if functional_lineage(closure.anchor.context) is None
                or closure.anchor.id in deliverable_functional_ids
            )
            excerpts = (
                self._context_excerpts(
                    hits,
                    dict(outcome.matched_dense_index_ids),
                )
                if allow_partial_sources
                else {}
            )
            if withheld_closures:
                unavailable = (
                    *unavailable,
                    ContextUnknown(
                        kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                        detail=context_prompts.EVIDENCE_CONSENT_WITHHELD,
                    ),
                )
            bundle = compile_context(
                prepared.text,
                hits,
                budget=budget,
                reference_at=reference,
                started_at=started_at,
                unknowns=(
                    *self._request_unknowns(prepared, scope, hits),
                    *recall_unknowns,
                    *withheld,
                    *unavailable,
                ),
                candidate_limit=candidate_limit,
                provisional={
                    memory_id: tuple(
                        identity_id for identity_id in identity_ids if identity_id not in restrained
                    )
                    for memory_id, identity_ids in self._provisional_identities(
                        permitted, scope=scope
                    ).items()
                    if any(identity_id not in restrained for identity_id in identity_ids)
                },
                named=named,
                resolve_co_derived_event_ids=partial(self._read_co_derived_event_ids, scope=scope),
                closures=closures,
                excerpts=excerpts,
                supplements=supplied_context,
                context_selector=self._context_selector,
                proof_scoring=self._proof_scoring,
            )
            # QUERY_FAILURE hook: a bundle with no evidence in it is the compiler reporting that
            # the goal found nothing, which is the same signal as an empty search.
            self._retrieval.note_query_failure(
                prepared.text,
                failed=not bundle.hits and not bundle.excerpts and not bundle.projections,
                operation="compile",
                memory_types=budget.memory_types,
                scope=scope,
                reference_at=reference,
            )
            return bundle

    def _context_excerpts(
        self,
        hits: Sequence[SearchHit],
        matched_dense_index_ids: Mapping[str, str],
    ) -> dict[str, ContextExcerpt]:
        """Verify compiler-only excerpts for scoped, consented raw candidates."""
        eligible = tuple(
            hit
            for hit in hits
            if hit.id in matched_dense_index_ids
            and hit.modality is Modality.TEXT
            and not hit.assets
            and (hit.context is None or hit.context.kind is RecordKind.OBSERVATION)
        )
        if not eligible:
            return {}
        embedding_ids = tuple(matched_dense_index_ids[hit.id] for hit in eligible)
        with translate_storage_errors("read text selectors"):
            selectors = self._store.index.read_text_selectors(embedding_ids)
        excerpts: dict[str, ContextExcerpt] = {}
        for hit in eligible:
            embedding_id = matched_dense_index_ids[hit.id]
            for stored in selectors.get(embedding_id, ()):
                excerpt = verified_context_excerpt(hit, embedding_id, stored)
                if excerpt is not None:
                    excerpts[hit.id] = excerpt
                    break
        return excerpts

    def _competing_lineage_hits(
        self,
        anchors: Sequence[SearchHit],
        *,
        scope: RetrievalScope | None,
    ) -> tuple[tuple[SearchHit, ...], frozenset[str]]:
        """Complete ranked functional claims with bounded, scope-correct competitors.

        Retrieval chooses the anchors.  This indexed SQLite read only asks whether a ranked
        functional lineage has a differently valued claim the context compiler must carry or
        withhold.  It assigns zero relevance to the completed rows, so they cannot become a new
        relevance route or displace an ordinary result except as an anchor's own obligation.
        """
        lineage_ids = tuple(
            dict.fromkeys(
                functional[0]
                for anchor in anchors
                if (functional := functional_lineage(anchor.context)) is not None
            )
        )
        if not lineage_ids:
            return (), frozenset()
        with translate_storage_errors("complete competing lineage evidence"):
            memories, incomplete = self._store.semantics.read_functional_lineage_representatives(
                lineage_ids,
                per_lineage_limit=_LINEAGE_COMPLETION_REPRESENTATIVES,
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
                near=None if scope is None else scope.near,
                radius_m=None if scope is None else scope.radius_m,
                place_id=None if scope is None else scope.place_id,
                identity_id=None if scope is None else scope.identity_id,
            )
        return tuple(self._hydrator.search_hit(memory, 0.0) for memory in memories), incomplete

    def _evidence_closures(
        self,
        anchors: Sequence[SearchHit],
        *,
        budget: ContextBudget,
        reference_at: datetime,
        scope: RetrievalScope | None,
        started_at: float | None = None,
    ) -> tuple[tuple[EvidenceClosure, ...], tuple[ContextUnknown, ...]]:
        """Hydrate finite, scoped transitive provenance for ranked compilation anchors."""
        started_at = perf_counter() if started_at is None else started_at
        if budget.selected_proofs and (scope is None or scope.known_at is None):
            return self._selected_closures(
                anchors,
                budget=budget,
                reference_at=reference_at,
                scope=scope,
                started_at=started_at,
            )
        return self._hydrator.evidence_closures(
            anchors, budget=budget, reference_at=reference_at, scope=scope, started_at=started_at
        )

    def _complete_affect_links(  # noqa: C901 - exact completion follows transitive delivery obligations
        self,
        candidates: Sequence[SearchHit],
        closures: Sequence[EvidenceClosure],
        *,
        budget: ContextBudget,
        reference_at: datetime,
        scope: RetrievalScope | None,
        started_at: float,
    ) -> tuple[tuple[SearchHit, ...], tuple[EvidenceClosure, ...], tuple[ContextUnknown, ...]]:
        """Complete every delivered attribution with its scoped affirmations and denials.

        The exact read is an evidence obligation for an already ranked claim, including a claim
        reached through provenance. Unrelated objects and roles never become relevance routes.
        """
        by_id = {hit.id: hit for hit in candidates}
        by_anchor = {closure.anchor.id: closure for closure in closures}
        groups: dict[AffectLinkKey, set[str]] = {}
        checked: set[AffectLinkKey] = set()
        incomplete: set[AffectLinkKey] = set()
        unknowns: list[ContextUnknown] = []
        while True:
            for closure in by_anchor.values():
                for member in closure.members:
                    key = affect_link_key(member)
                    if key is not None:
                        groups.setdefault(key, set()).add(member.id)
            focal_ids = {key[0] for key in groups}
            fallback_cues = {
                member.id: member
                for closure in by_anchor.values()
                for member in closure.members
                if member.id in focal_ids and member.id not in by_id
            }
            if fallback_cues:
                # An attribution may not fit while its cue does. Reuse its already hydrated
                # focus as a separate anchor; do not discover other affects or infer a cause.
                by_id.update(fallback_cues)
                cue_closures, cue_unavailable = self._evidence_closures(
                    tuple(fallback_cues.values()),
                    budget=budget,
                    reference_at=reference_at,
                    scope=scope,
                    started_at=started_at,
                )
                by_anchor.update({closure.anchor.id: closure for closure in cue_closures})
                unknowns.extend(cue_unavailable)
            pending = groups.keys() - checked
            if not pending:
                break
            checked.update(pending)
            if len(checked) > _AFFECT_LINK_COMPLETION_LIMIT or (
                budget.max_latency_ms is not None
                and (perf_counter() - started_at) * 1000 > budget.max_latency_ms
            ):
                incomplete.update(pending)
                continue
            with translate_storage_errors("complete affect attribution evidence"):
                records = self._store.semantics.affect_links(
                    tuple(sorted({key[0] for key in pending})),
                    limit=_AFFECT_LINK_COMPLETION_LIMIT,
                    scope=scope,
                )
            if records.selected > len(records):
                incomplete.update(pending)
                continue
            peers: list[SearchHit] = []
            found: dict[AffectLinkKey, set[str]] = {}
            for record in records:
                hit = self._hydrator.search_hit(record, 0.0)
                key = affect_link_key(hit)
                if key is None or key not in pending:
                    continue
                found.setdefault(key, set()).add(hit.id)
                if _rejection(hit, budget, reference_at) is not None:
                    incomplete.add(key)
                if hit.id not in by_id:
                    by_id[hit.id] = hit
                if hit.id not in by_anchor:
                    peers.append(by_id[hit.id])
            for key in pending:
                if not groups[key].issubset(found.get(key, set())):
                    incomplete.add(key)
                groups[key].update(found.get(key, set()))
            peer_closures, unavailable = self._evidence_closures(
                peers,
                budget=budget,
                reference_at=reference_at,
                scope=scope,
                started_at=started_at,
            )
            unknowns.extend(unavailable)
            by_anchor.update({closure.anchor.id: closure for closure in peer_closures})
        for key, peer_ids in groups.items():
            if any(memory_id not in by_anchor for memory_id in peer_ids):
                incomplete.add(key)
        withheld = {
            closure.anchor.id
            for closure in by_anchor.values()
            if any(affect_link_key(member) in incomplete for member in closure.members)
        }
        withheld.update(hit.id for hit in by_id.values() if affect_link_key(hit) in incomplete)
        if incomplete:
            unknowns.append(
                ContextUnknown(
                    kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                    detail=(context_prompts.incomplete_affect(len(incomplete))),
                )
            )
        return (
            tuple(hit for hit in by_id.values() if hit.id not in withheld),
            tuple(closure for closure in by_anchor.values() if closure.anchor.id not in withheld),
            tuple(unknowns),
        )

    def _selected_closures(  # noqa: C901 - enforce every intermediate's delivery obligations
        self,
        anchors: Sequence[SearchHit],
        *,
        budget: ContextBudget,
        reference_at: datetime,
        scope: RetrievalScope | None,
        started_at: float,
    ) -> tuple[tuple[EvidenceClosure, ...], tuple[ContextUnknown, ...]]:
        """Complete count and confidence obligations before atomically pricing their union."""
        closures: list[EvidenceClosure] = []
        unavailable = 0
        truncated = False
        for anchor in anchors:
            if (
                budget.max_latency_ms is not None
                and (perf_counter() - started_at) * 1000 > budget.max_latency_ms
            ):
                unavailable += 1
                continue
            with translate_storage_errors("read delivery certificates"):
                nodes, records, clipped = self._store.semantics.delivery_evidence(
                    anchor.id, scope=scope
                )
            truncated |= clipped
            hits = {
                record.memory_id: self._hydrator.search_hit(
                    record, anchor.score if record.memory_id == anchor.id else 0.0
                )
                for record in records
            }
            ranked = {hit.id: hit for hit in anchors}
            eligible = {
                memory_id: node
                for memory_id, node in nodes.items()
                if memory_id in hits
                and _rejection(hits[memory_id], budget, reference_at) is None
                and (
                    memory_id not in ranked
                    or (
                        hits[memory_id].context == ranked[memory_id].context
                        and hits[memory_id].content == ranked[memory_id].content
                    )
                )
            }
            costs = {memory_id: bundle_cost(hits[memory_id]) for memory_id in eligible}
            pending = [anchor.id]
            examined: set[str] = set()
            members: set[str] = {anchor.id}
            selected: dict[ContextProof, None] = {}
            complete = True
            while pending:
                memory_id = pending.pop()
                if memory_id in examined:
                    continue
                examined.add(memory_id)
                if memory_id not in eligible or len(members) > budget.max_items:
                    complete = False
                    break
                node = eligible[memory_id]
                if node.root_group is not None and not node.assessments:
                    continue
                if (
                    budget.max_latency_ms is not None
                    and (perf_counter() - started_at) * 1000 > budget.max_latency_ms
                ):
                    complete = False
                    break
                full_support = (
                    independent_support(nodes, memory_id)
                    if self._proof_scoring == "independent_pair"
                    else SupportSummary(0, 0.0, False)
                )
                context = hits[memory_id].context
                support = (
                    1
                    if context is None
                    else required_support(context.kind, context.basis, context.identity_id)
                )
                required = SupportSummary(
                    count=max(support, full_support.count),
                    confidence=0.0 if context is None else context.confidence,
                    truncated=full_support.truncated or clipped,
                )
                witnesses, clipped_search = certificates(eligible, memory_id)
                truncated |= clipped_search or full_support.truncated
                bought = select_certificates(
                    witnesses,
                    required,
                    costs,
                    naming_conditions(nodes, memory_id),
                    scoring=self._proof_scoring,
                )
                if not bought:
                    complete = False
                    break
                selected.update(dict.fromkeys(bought))
                for proof in bought:
                    for step in proof.nodes:
                        members.add(step.memory_id)
                        if step.memory_id not in examined:
                            pending.append(step.memory_id)
            if not complete or len(members) > budget.max_items:
                unavailable += 1
                continue
            closures.append(
                EvidenceClosure(
                    hits[anchor.id],
                    (
                        hits[anchor.id],
                        *(hits[memory_id] for memory_id in sorted(members - {anchor.id})),
                    ),
                    tuple(selected),
                )
            )
        unknowns = []
        if unavailable:
            unknowns.append(
                ContextUnknown(
                    kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                    detail=context_prompts.uncertified_assertions(unavailable),
                )
            )
        if truncated:
            unknowns.append(
                ContextUnknown(
                    kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                    detail=context_prompts.CERTIFICATE_SEARCH_TRUNCATED,
                )
            )
        return tuple(closures), tuple(unknowns)

    def _read_co_derived_event_ids(
        self,
        cues: Sequence[str],
        *,
        scope: RetrievalScope | None,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve, for the selected affect entries, the events their observations also formed.

        One batched read per compilation, under the same bitemporal and spatial scope the hits
        were hydrated with, so the hop cannot surface an event the search itself would have
        dropped. The compiler calls this after selection, so the read is bounded by the budget
        rather than by the candidate window. The edge is shared evidence -- co-occurrence inside
        one capture -- and never a cause.
        """
        if not cues:
            return {}
        with translate_storage_errors("read co-derived events"):
            return self._store.identities.read_co_derived_event_ids(
                cues,
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
                near=None if scope is None else scope.near,
                radius_m=None if scope is None else scope.radius_m,
                place_id=None if scope is None else scope.place_id,
                identity_id=None if scope is None else scope.identity_id,
            )

    def _consented_actors(
        self,
        hits: Sequence[SearchHit],
        named: Mapping[str, tuple[NamedActorLink, ...]],
        *,
        observed_identity_ids: frozenset[str] = frozenset(),
    ) -> tuple[
        tuple[SearchHit, ...],
        dict[str, tuple[NamedActorLink, ...]],
        tuple[ContextUnknown, ...],
        frozenset[str],
    ]:
        """Drop the named actors whose subject withheld or withdrew consent, and say so.

        Only the naming assertion is dropped, not the person's memories: consent governs being
        treated as a recognized person, not whether the evening happened. The bundle then says
        a person is missing from `actors` rather than silently compiling a scene with a hole in
        it, which is what makes the omission auditable instead of a retrieval mystery.

        Both routes to an actor are filtered, because both publish the same person. A bound
        `ENTITY` hit renders as its own line, and a resolved identity edge on any other memory
        -- the face in a photo, the speaker in a clip -- becomes a `NamedActor` whether or not
        the assertion that names them ranked at all.
        """
        bound = {
            hit.context.identity_id
            for hit in hits
            if hit.context is not None
            and hit.context.identity_id is not None
            and hit.context.kind is RecordKind.ENTITY
        }
        # Every identity the enrichment could name, not only the ones a retrieved `ENTITY` hit
        # carried: a bundle that reached a person's clip but not their naming assertion used to
        # find nothing bound here, consult no consent at all, and then name them from the clip.
        bound.update(link[0] for links in named.values() for link in links)
        bound.update(observed_identity_ids)
        if not bound:
            return tuple(hits), dict(named), (), frozenset()
        with translate_storage_errors("read identity consent"):
            restrained = self._store.identities.restrained_identities() & bound
        if not restrained:
            return tuple(hits), dict(named), (), frozenset()
        kept = tuple(
            hit
            for hit in hits
            if hit.context is None
            or hit.context.kind is not RecordKind.ENTITY
            or hit.context.identity_id not in restrained
        )
        permitted = {
            memory_id: kept_links
            for memory_id, links in named.items()
            if (kept_links := tuple(link for link in links if link[0] not in restrained))
        }
        return (
            kept,
            permitted,
            (
                ContextUnknown(
                    kind=ContextUnknownKind.CONSENT_WITHHELD,
                    detail=(context_prompts.consent_withheld_actors(len(restrained))),
                ),
            ),
            restrained,
        )

    def _provisional_identities(
        self,
        hits: Sequence[SearchHit],
        *,
        scope: RetrievalScope | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve which people the candidate evidence observed but nobody has named.

        Deterministic kernel policy, like every other identity decision: a person is
        provisional exactly while no visible naming assertion names them.
        """
        if not hits:
            return {}
        with translate_storage_errors("read provisional identities"):
            return self._store.identities.provisional_identities(
                tuple(hit.id for hit in hits),
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
            )

    def _named_actors(
        self,
        hits: Sequence[SearchHit],
        *,
        scope: RetrievalScope | None = None,
    ) -> dict[str, tuple[NamedActorLink, ...]]:
        """Resolve which people the candidate evidence's identity edge already names.

        Deterministic kernel policy, the positive counterpart of `_provisional_identities`: a
        person is named exactly while a visible naming assertion names them, whether or not
        that assertion itself ranked into the bundle.
        """
        if not hits:
            return {}
        with translate_storage_errors("read named actors"):
            return self._store.identities.named_actors(
                tuple(hit.id for hit in hits),
                valid_at=None if scope is None else scope.valid_at,
                known_at=None if scope is None else scope.known_at,
            )

    def _request_unknowns(
        self,
        prepared: PreparedContent,
        scope: RetrievalScope | None,
        hits: Sequence[SearchHit],
    ) -> tuple[ContextUnknown, ...]:
        """Name what the request asked for that the compiler cannot see in the hits alone."""
        unknowns: list[ContextUnknown] = []
        unsupported = sorted(
            {
                asset.modality
                for asset in prepared.assets
                if Modality(asset.modality) not in self._backends.embedding_capabilities
            }
        )
        if unsupported:
            unknowns.append(
                ContextUnknown(
                    kind=ContextUnknownKind.MODALITY_UNSUPPORTED,
                    detail=(
                        context_prompts.unsupported_goal_modalities(
                            unsupported, modality_names(self._backends.embedding_capabilities)
                        )
                    ),
                )
            )
        if not hits and (described := scope_description(scope)) is not None:
            unknowns.append(ContextUnknown(kind=ContextUnknownKind.SCOPE_EMPTY, detail=described))
        return tuple(unknowns)
