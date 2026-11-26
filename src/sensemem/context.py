"""Pure context selection behind Memory.compile.

The compiler selects and structures already-retrieved evidence. It calls no model, resolves no
conflict, and writes nothing itself: every value here is a deterministic function of the ranked
hits, the budget, the reference clock, and -- for the deadline alone -- the wall clock.
Authoritative visibility and scope were applied by the retrieval path that produced the hits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from time import perf_counter
from types import MappingProxyType
from typing import Protocol, TypeAlias

from sensemem.context_bundle import ContextBundle
from sensemem.context_rendering import (
    _MAX_CUE_IDS,
    render_excerpt_line,
    render_hit_line,
    render_named_actor_line,
    render_projection_line,
    render_provisional_actor_line,
    render_supplement_line,
)
from sensemem.context_selection import (
    ContextCandidate,
    ContextSelection,
    ContextSelectionRequest,
    ContextSelector,
    DefaultContextSelector,
)
from sensemem.context_values import (
    AffectCue,
    ContextBudget,
    ContextConflict,
    ContextExcerpt,
    ContextProjection,
    ContextProof,
    ContextProofScoring,
    ContextSupplement,
    ContextUnknown,
    ContextUnknownKind,
    NamedActor,
    ProvisionalActor,
)
from sensemem.evidence import AffectLinkKey, affect_link_key, functional_lineage
from sensemem.exceptions import ValidationError
from sensemem.prompts import context as context_prompts
from sensemem.records import SearchHit
from sensemem.types import CATEGORY_BY_KIND, MemoryCategory, Modality, RecordKind

# One (identity_id, name, naming_assertion_id) edge a memory carries to a currently named
# identity. naming_assertion_id is None only when a caller cannot cite it cheaply.
NamedActorLink: TypeAlias = tuple[str, str, str | None]

# Text-equivalent cost of one grounded media part, at the usual four characters per token. The
# modalities are an order of magnitude apart in what a model charges for them: an image part
# measured near five hundred tokens against this stack where a ten-second video part measured
# near three thousand, so one flat number would either starve text or overrun on video. These
# are coarse by design; the budget is the caller's knob, not these constants.
_ASSET_EVIDENCE_CHARS: Mapping[Modality, int] = MappingProxyType(
    {
        Modality.IMAGE: 2_000,
        Modality.AUDIO: 4_000,
        Modality.VIDEO: 12_000,
    }
)
_DEFAULT_ASSET_EVIDENCE_CHARS = 4_000
# Bundle sections, keyed by the ContextBundle field they fill.
_SECTIONS: tuple[str, ...] = (
    "actors",
    "relationships",
    "scene",
    "facts",
    "episodes",
    "procedures",
    "affect",
    "traits",
)
# A typed kind decides the section first; an untyped or generic record falls back to its type.
_KIND_SECTIONS: Mapping[RecordKind, str] = MappingProxyType(
    {
        RecordKind.ENTITY: "actors",
        RecordKind.RELATION: "relationships",
        RecordKind.STATE: "scene",
        RecordKind.AFFECT: "affect",
        RecordKind.TRAIT: "traits",
    }
)
_TYPE_SECTIONS: Mapping[MemoryCategory, str] = MappingProxyType(
    {
        MemoryCategory.SEMANTIC: "facts",
        MemoryCategory.EPISODIC: "episodes",
        MemoryCategory.PROCEDURAL: "procedures",
    }
)
# Which memory types can reach each section at all. A kind-keyed section is fed by whatever type
# a formed record of that kind is stored as, a type-keyed one by its own type. This is what lets a
# memory_types filter name the sections it emptied instead of only the types it dropped: kind
# affect is stored episodic, so a {semantic} request cannot fill affect however it ranks.
_SECTION_MEMORY_TYPES: Mapping[str, frozenset[MemoryCategory]] = MappingProxyType(
    {
        name: frozenset(
            {
                CATEGORY_BY_KIND.get(kind, MemoryCategory.SEMANTIC)
                for kind, section in _KIND_SECTIONS.items()
                if section == name
            }
            | {memory_type for memory_type, section in _TYPE_SECTIONS.items() if section == name}
        )
        for name in _SECTIONS
    }
)
# Why a candidate the retrieval kernel returned never became a bundle line. Reported as counts
# under unknowns so a caller reading a thin bundle knows which bound produced it.
_EXCLUSION_REASONS = ("memory_types", "min_confidence", "freshness")
# One asserted value: the value, the memory asserting it, and that memory's subject and predicate.
_Claim: TypeAlias = tuple[str, str, str | None, str | None]


# Resolves affect memory IDs to the event IDs their own observations also formed. The compiler
# calls it once, for the affect entries the budget actually bought, and never past the deadline.
class _CoDerivedEventResolver(Protocol):
    def __call__(self, memory_ids: Sequence[str], /) -> Mapping[str, Sequence[str]]: ...


@dataclass(frozen=True, slots=True)
class EvidenceClosure:
    """One ranked assertion and every scoped record required to support it.

    This remains an internal compiler value.  anchor has its retrieval score; dependencies
    use zero because they were not query-ranked.  Selection admits members atomically.
    """

    anchor: SearchHit
    members: tuple[SearchHit, ...]
    proofs: tuple[ContextProof, ...] = ()

    def __post_init__(self) -> None:
        members = tuple(dict.fromkeys(self.members))
        if self.anchor.id not in {member.id for member in members}:
            raise ValidationError("an evidence closure must contain its anchor")
        object.__setattr__(self, "members", members)


def evidence_cost(hit: SearchHit) -> int:
    """Return what grounding one hit costs, in characters of text-equivalent evidence."""
    return len(hit.content) + _asset_cost(hit)


def _asset_cost(hit: SearchHit) -> int:
    # AssetRef.modality is optional on the public type; an unresolved asset is charged the
    # default rather than being treated as free.
    return sum(
        _DEFAULT_ASSET_EVIDENCE_CHARS
        if asset.modality is None
        else _ASSET_EVIDENCE_CHARS.get(asset.modality, _DEFAULT_ASSET_EVIDENCE_CHARS)
        for asset in hit.assets
    )


def bundle_cost(hit: SearchHit) -> int:
    """Return what one hit costs a bundle: its rendered line plus its media parts.

    The rendered line, not the record text: the - [id] frame and the confidence suffix are
    characters the consumer pays for, so charging only len(content) would hand a caller a
    budget that render() overruns. Media parts are charged their text equivalent, which is far
    more than the zero characters they render as, so the number is an upper bound on the text.
    """
    return len(render_hit_line(hit)) + 1 + _asset_cost(hit)


def excerpt_cost(excerpt: ContextExcerpt) -> int:
    """Return the exact rendered cost of one partial source."""
    return len(render_excerpt_line(excerpt)) + 1


def projection_cost(projection: ContextProjection) -> int:
    """Price the complete labelled text delivery and its omitted-media declaration."""
    return len(render_projection_line(projection)) + 1


def _text_projection(hit: SearchHit) -> ContextProjection | None:
    context = hit.context
    if (
        not hit.assets
        or not any(part.kind != "asset" for part in hit.parts)
        or (
            context is not None
            and (context.kind is not RecordKind.OBSERVATION or context.evidence_ids)
        )
    ):
        return None
    return ContextProjection(
        source_memory_id=hit.id,
        parts=hit.parts,
        omitted_assets=tuple(
            {asset.id: replace(asset, path=None) for asset in hit.assets}.values()
        ),
        score=hit.score,
        created_at=hit.created_at,
        occurred_at=hit.occurred_at,
        occurred_end=hit.occurred_end,
        memory_type=hit.memory_type,
        context=context,
        place_id=hit.place_id,
    )


def named_actor_cost(actor: NamedActor) -> int:
    """Return what one named-actor line costs, priced exactly as render_named_actor_line
    renders it, so the price and the text can never drift apart.
    """
    return len(render_named_actor_line(actor)) + 1


def provisional_actor_cost(actor: ProvisionalActor) -> int:
    """Return what one provisional-actor line costs, priced exactly as
    render_provisional_actor_line renders it.
    """
    return len(render_provisional_actor_line(actor)) + 1


def _actor_cost(entry: NamedActor | ProvisionalActor) -> int:
    if isinstance(entry, NamedActor):
        return named_actor_cost(entry)
    return provisional_actor_cost(entry)


def _frame_cost(
    goal: str,
    reference_at: datetime,
    budget: ContextBudget,
    proof_scoring: ContextProofScoring = "independent_pair",
) -> int:
    """Return an upper bound on the fixed header render() writes above the first section."""
    return (
        len(f"# Context: {goal}")
        + len(context_prompts.CONTEXT_READING_GUIDE)
        + len(f"Reference time: {reference_at.isoformat()}")
        # chars cannot have more digits than max_chars, nor included than max_items.
        + len(
            f"Budget: {budget.max_chars}/{budget.max_chars} chars,"
            f" {budget.max_items}/{budget.max_items} items"
        )
        + (
            len(context_prompts.proof_scoring_line(proof_scoring)) + 1
            if budget.selected_proofs
            else 0
        )
        + 4
    )


def _heading_cost(section: str) -> int:
    """Return what one section heading costs: a blank line, ## , the name, two newlines."""
    return len(section) + 5


def _select_supplements(
    supplements: Sequence[ContextSupplement],
    budget: ContextBudget,
    reference_at: datetime,
    overhead: int,
) -> tuple[tuple[ContextSupplement, ...], int, tuple[ContextUnknown, ...]]:
    """Admit temporary evidence in submission order and reserve its shared budget."""
    selected: list[ContextSupplement] = []
    exclusions: dict[str, int] = {}
    for supplement in supplements:
        reason = None
        if supplement.expires_at is not None and supplement.expires_at <= reference_at:
            reason = "expired"
        elif supplement.observed_at is not None and supplement.observed_at > reference_at:
            reason = "future"
        elif budget.freshness is not None and (
            supplement.observed_at is None
            or reference_at - supplement.observed_at > budget.freshness
        ):
            reason = "freshness"
        cost = len(render_supplement_line(supplement)) + 1
        if not selected:
            cost += _heading_cost("temporary context (untrusted)")
        if reason is None and (
            len(selected) >= budget.max_items or overhead + cost > budget.max_chars
        ):
            reason = "budget"
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        selected.append(supplement)
        overhead += cost
    unknowns = tuple(
        ContextUnknown(
            kind=(
                ContextUnknownKind.BUDGET_EXCLUDED
                if reason == "budget"
                else ContextUnknownKind.EVIDENCE_UNAVAILABLE
            ),
            detail=context_prompts.supplements_excluded(count, reason),
        )
        for reason, count in exclusions.items()
    )
    return tuple(selected), overhead, unknowns


def compile_context(  # noqa: C901 - filters, closed selection, and deadlines share one bundle budget
    goal: str,
    hits: Sequence[SearchHit],
    *,
    budget: ContextBudget,
    reference_at: datetime,
    started_at: float | None = None,
    unknowns: Sequence[ContextUnknown] = (),
    candidate_limit: int | None = None,
    provisional: Mapping[str, Sequence[str]] = MappingProxyType({}),
    named: Mapping[str, Sequence[NamedActorLink]] = MappingProxyType({}),
    resolve_co_derived_event_ids: _CoDerivedEventResolver | None = None,
    closures: Sequence[EvidenceClosure] | None = None,
    excerpts: Mapping[str, ContextExcerpt] = MappingProxyType({}),
    supplements: Sequence[ContextSupplement] = (),
    context_selector: ContextSelector | None = None,
    proof_scoring: ContextProofScoring = "independent_pair",
) -> ContextBundle:
    """Partition, filter, and budget ranked hits into one bundle.

    started_at is a time.perf_counter() reading taken before retrieval, so budget
    .max_latency_ms bounds the whole compilation rather than this function. unknowns carries
    what the caller already knows and the compiler cannot see, such as an empty spatial scope.
    candidate_limit is the depth retrieval was asked to rank, so a bundle that lost evidence
    and filled that window can say the ranking may continue past it.

    provisional maps a memory ID to the recognized people it observed whom no visible naming
    assertion names. The kernel resolves that, deterministically, before calling; the compiler
    only keeps the entries whose evidence actually made it into the bundle, so the bundle never
    reports a person the reader cannot see the evidence for.

    named maps a memory ID to the identities its identity edge resolves to that a currently
    visible naming assertion names -- the positive counterpart of provisional, resolved the
    same way by the kernel before calling. An identity named reports is never also reported
    as provisional, even if a stale provisional entry still names it. Both a named and a
    provisional actor render a line, and that line is priced and fit into whatever max_chars
    has left after the ranked hits: it is included when it fits and dropped, not appended for
    free, when it does not.

    resolve_co_derived_event_ids resolves affect memory IDs to the event IDs formed from the same
    observations, under the same visibility the hits were hydrated with. It is called once, with
    the affect entries the budget actually bought, and not at all once the deadline has passed:
    the hop is optional enrichment, like conflict detection. The edge is co-occurrence inside
    one capture, not causation, and the events themselves are neither fetched nor charged.
    """
    started_at = perf_counter() if started_at is None else started_at
    excluded: dict[str, int] = {}
    candidates: list[SearchHit] = []
    withheld_attributions: set[AffectLinkKey] = set()
    for hit in hits:
        reason = _rejection(hit, budget, reference_at)
        if reason is None:
            candidates.append(hit)
        else:
            excluded[reason] = excluded.get(reason, 0) + 1
            key = affect_link_key(hit)
            if key is not None:
                withheld_attributions.add(key)
    if withheld_attributions:
        candidates = [
            hit for hit in candidates if affect_link_key(hit) not in withheld_attributions
        ]
        closures = (
            None
            if closures is None
            else tuple(
                closure
                for closure in closures
                if all(
                    affect_link_key(member) not in withheld_attributions
                    for member in closure.members
                )
            )
        )
        unknowns = (
            *unknowns,
            ContextUnknown(
                kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                detail=context_prompts.AFFECT_FILTERED,
            ),
        )
    overhead = _frame_cost(goal, reference_at, budget, proof_scoring)
    if overhead > budget.max_chars:
        # max_chars bounds the header too, so a budget it alone overruns has no bundle that
        # satisfies it: _select would buy nothing and the four framing lines would still be
        # rendered over the limit. Refused rather than returned oversized, because a caller
        # sizing a model window is better served by being told the floor than by a bundle that
        # quietly breaks it. The floor moves with the goal, which the header repeats.
        raise ValidationError(
            f"budget max_chars must be at least {overhead} to fit this goal's context header",
            subject="budget",
        )
    selected_supplements, overhead, supplement_unknowns = _select_supplements(
        supplements, budget, reference_at, overhead
    )
    unknowns = (*unknowns, *supplement_unknowns)
    # Explicit temporary evidence takes its slots first; memory evidence still enters only as
    # complete closures. A zero remaining item budget skips selection, without manufacturing a
    # ContextBudget whose public positive-item constraint would be false.
    remaining_items = budget.max_items - len(selected_supplements)
    selection_budget = replace(budget, max_items=max(1, remaining_items))
    # An affect entry becomes an AffectCue: the same hit, plus the events its evidence also
    # formed. The conversion happens before the pricing, not after it, because a cue renders its
    # basis, cue modality, valence, arousal, and evidence IDs on the line: pricing the plain hit
    # and rendering the cue is what let a bundle overrun the max_chars it reported.
    candidates = [_affect_cue(hit, ()) if _section(hit) == "affect" else hit for hit in candidates]
    excerpt_candidates = {
        candidate.id: excerpt
        for candidate in candidates
        if (excerpt := excerpts.get(candidate.id)) is not None
    }
    projection_candidates = {
        candidate.id: projection
        for candidate in candidates
        if (projection := _text_projection(candidate)) is not None
    }
    candidate_by_id = {candidate.id: candidate for candidate in reversed(candidates)}
    closure_by_anchor = {
        closure.anchor.id: EvidenceClosure(
            anchor=candidate_by_id[closure.anchor.id],
            members=tuple(
                candidate_by_id[member.id]
                if member.id in candidate_by_id
                else _affect_cue(member, ())
                if _section(member) == "affect"
                else member
                for member in closure.members
            ),
            proofs=closure.proofs,
        )
        for closure in closures or ()
        if closure.anchor.id in candidate_by_id
    }
    has_conflicts = bool(_conflicts(candidates, frozenset(hit.id for hit in candidates)))
    all_singletons = len(closure_by_anchor) == len(candidates) and all(
        len(closure.members) == 1 for closure in closure_by_anchor.values()
    )
    closed_mode = closures is not None and (
        any(closure.proofs for closure in closure_by_anchor.values())
        or any(affect_link_key(candidate) is not None for candidate in candidates)
        or not (all_singletons and not has_conflicts)
    )
    selected_closures, peer_withheld = (
        _expand_conflict_closures(candidates, closure_by_anchor) if closed_mode else ((), 0)
    )
    if peer_withheld:
        # Fail-closed is the rule; failing silently is not. The anchor is in no section, not in
        # omitted, and not in the caller's unknowns, so it has to be named here.
        unknowns = (
            *unknowns,
            ContextUnknown(
                kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                detail=(context_prompts.conflicting_evidence_unavailable(peer_withheld)),
            ),
        )
    context_selector = DefaultContextSelector() if context_selector is None else context_selector
    sections, selected_excerpts, selected_projections, selected_proofs, unproposed = _select(
        selected_closures
        if closed_mode and remaining_items
        else tuple(EvidenceClosure(hit, (hit,)) for hit in candidates)
        if remaining_items
        else (),
        goal=goal,
        budget=selection_budget,
        reference_at=reference_at,
        overhead=overhead,
        excerpts=excerpt_candidates,
        projections=projection_candidates,
        closed_evidence=closed_mode,
        context_selector=context_selector,
    )
    # The deadline is checked here, between section assembly and the optional enrichment that
    # follows it. Nothing already computed is discarded and no stage is cut in half, so a bundle
    # under a deadline is a prefix of the one without it, never a different one.
    skipped = _past_deadline(budget, started_at)
    # The hop is resolved only for the entries that were bought, so the read it costs is bounded
    # by the budget rather than by how deep retrieval ranked.
    hops: Mapping[str, Sequence[str]] = MappingProxyType({})
    if resolve_co_derived_event_ids is not None and not skipped and sections["affect"]:
        hops = resolve_co_derived_event_ids(tuple(hit.id for hit in sections["affect"]))
    if hops:
        # A resolved hop lengthens a line _select has already priced, so selection runs again
        # over the cues it will render. An affect entry the second pass buys instead carries no
        # hop, the same as one resolved to nothing: the hop is enrichment, and the read it costs
        # stays bounded by what the first pass bought.
        candidates = [
            _affect_cue(hit, hops[hit.id]) if isinstance(hit, AffectCue) and hit.id in hops else hit
            for hit in candidates
        ]
        candidate_by_id = {candidate.id: candidate for candidate in candidates}
        selected_closures = tuple(
            EvidenceClosure(
                anchor=candidate_by_id[closure.anchor.id],
                members=tuple(candidate_by_id.get(member.id, member) for member in closure.members),
                proofs=closure.proofs,
            )
            for closure in selected_closures
        )
        # Affect hops annotate rather than support records, so closed selection stays unchanged.
        sections, selected_excerpts, selected_projections, selected_proofs, unproposed = _select(
            selected_closures
            if closed_mode and remaining_items
            else tuple(EvidenceClosure(hit, (hit,)) for hit in candidates)
            if remaining_items
            else (),
            goal=goal,
            budget=selection_budget,
            reference_at=reference_at,
            overhead=overhead,
            excerpts=excerpt_candidates,
            projections=projection_candidates,
            closed_evidence=closed_mode,
            context_selector=context_selector,
        )
    if any(
        affect_link_key(hit) is not None for section in sections.values() for hit in section
    ) and _past_deadline(budget, started_at):
        skipped = True
        withheld_anchors = {
            closure.anchor.id
            for closure in selected_closures
            if any(affect_link_key(member) is not None for member in closure.members)
        }
        candidates = [
            hit
            for hit in candidates
            if hit.id not in withheld_anchors and affect_link_key(hit) is None
        ]
        selected_closures = tuple(
            closure for closure in selected_closures if closure.anchor.id not in withheld_anchors
        )
        sections, selected_excerpts, selected_projections, selected_proofs, unproposed = _select(
            selected_closures
            if closed_mode and remaining_items
            else tuple(EvidenceClosure(hit, (hit,)) for hit in candidates)
            if remaining_items
            else (),
            goal=goal,
            budget=selection_budget,
            reference_at=reference_at,
            overhead=overhead,
            excerpts=excerpt_candidates,
            projections=projection_candidates,
            closed_evidence=closed_mode,
            context_selector=context_selector,
        )
        unknowns = (
            *unknowns,
            ContextUnknown(
                kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                detail=context_prompts.AFFECT_DEADLINE,
            ),
        )
    affect = tuple(entry for entry in sections["affect"] if isinstance(entry, AffectCue))
    # Rank order, not section order: _lineage_conflict pairs each value with the highest-ranked
    # memory asserting it, so a lineage whose claims land in different sections must still be read
    # by score. ContextBundle.hits sorts the same way.
    included = tuple(
        sorted(
            (hit for name in _SECTIONS for hit in sections[name]),
            key=lambda hit: (-hit.score, hit.id),
        )
    )
    omitted = (
        len(selected_closures)
        - len(
            {
                closure.anchor.id
                for closure in selected_closures
                if closure.anchor.id
                in (
                    {member.id for section in sections.values() for member in section}
                    | {excerpt.source_memory_id for excerpt in selected_excerpts}
                    | {projection.source_memory_id for projection in selected_projections}
                )
            }
        )
        if closed_mode
        else len(candidates) - len(included) - len(selected_excerpts) - len(selected_projections)
    )
    # Conflict detection reads every candidate the filters kept, not only what the budget bought,
    # so dropping one side of a disagreement for want of a slot cannot make it disappear.
    conflicts = (
        ()
        if skipped
        else _conflicts(
            sorted(candidates, key=lambda hit: (-hit.score, hit.id)),
            frozenset(hit.id for hit in included),
        )
    )
    selected_evidence: tuple[SearchHit | ContextExcerpt | ContextProjection, ...] = (
        *included,
        *selected_excerpts,
        *selected_projections,
    )
    selector_omitted = len(
        unproposed
        - {hit.id for hit in included}
        - {excerpt.source_memory_id for excerpt in selected_excerpts}
        - {projection.source_memory_id for projection in selected_projections}
    )
    if selector_omitted:
        unknowns = (
            *unknowns,
            ContextUnknown(
                kind=ContextUnknownKind.EVIDENCE_UNAVAILABLE,
                detail=context_prompts.selector_excluded(selector_omitted),
            ),
        )
    occurred_from, occurred_until = _occurred_range(selected_evidence)
    elapsed_ms = _elapsed_ms(started_at)
    # Actor lines are priced and fit in after the ranked hits, into whatever _select left of
    # max_chars: _bundle_chars already bounds that at or below max_chars, so this can
    # never push the total over it. A named identity is never also reported provisional, even
    # if a stale provisional entry still names it.
    base_chars = _bundle_chars(
        overhead, sections, (*selected_excerpts, *selected_projections), selected_proofs
    )
    # Omitted native media cannot establish that the reader was supplied an identity
    # observation. Only existing full/exact-excerpt delivery can populate actor links.
    actor_evidence: tuple[SearchHit | ContextExcerpt, ...] = (*included, *selected_excerpts)
    named_actors = _named_actors(actor_evidence, named)
    provisional_actors = _provisional_actors(
        actor_evidence,
        provisional,
        exclude=frozenset(actor.identity_id for actor in named_actors),
    )
    actor_candidates: tuple[NamedActor | ProvisionalActor, ...] = (
        *named_actors,
        *provisional_actors,
    )
    # render() writes ## Actors as soon as the section holds anything, so a bundle whose
    # typed ENTITY hits left it empty owes that heading the moment a synthesized actor fills
    # it. _bundle_chars prices only the sections _select filled, so it is charged here or
    # nowhere -- and reserved before the fit, so the heading cannot push the total past
    # max_chars either.
    actors_heading = _heading_cost("actors") if actor_candidates and not sections["actors"] else 0
    fitted_actors, actor_chars, actor_excluded = _fit_actors(
        actor_candidates,
        budget.max_chars - base_chars - actors_heading,
    )
    if not fitted_actors:
        # Nothing fit, so no heading is written and none is owed.
        actors_heading = 0
    return ContextBundle(
        goal=goal,
        reference_at=reference_at,
        budget=budget,
        actors=(*sections["actors"], *fitted_actors),
        relationships=sections["relationships"],
        scene=sections["scene"],
        episodes=sections["episodes"],
        facts=sections["facts"],
        procedures=sections["procedures"],
        affect=affect,
        traits=sections["traits"],
        conflicts=conflicts,
        unknowns=_unknowns(
            unknowns,
            budget,
            sections,
            selected_excerpts,
            selected_projections,
            excluded,
            omitted - selector_omitted,
            skipped=skipped,
            exhausted=(
                candidate_limit is not None
                and len(hits) >= candidate_limit
                and bool(excluded or omitted)
            ),
            candidate_limit=candidate_limit,
            actor_excluded=actor_excluded,
        ),
        occurred_from=occurred_from,
        occurred_until=occurred_until,
        frames=_frames(selected_evidence),
        places=tuple(sorted({hit.place_id for hit in selected_evidence if hit.place_id})),
        omitted=omitted,
        chars=base_chars + actors_heading + actor_chars,
        elapsed_ms=elapsed_ms,
        deadline_exceeded=(
            budget.max_latency_ms is not None and elapsed_ms > budget.max_latency_ms
        ),
        excerpts=selected_excerpts,
        projections=selected_projections,
        proofs=selected_proofs,
        proof_scoring=proof_scoring,
        supplements=tuple(selected_supplements),
        omitted_supplements=len(supplements) - len(selected_supplements),
    )


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _past_deadline(budget: ContextBudget, started_at: float) -> bool:
    """Whether the declared deadline had already passed at this checkpoint."""
    return budget.max_latency_ms is not None and _elapsed_ms(started_at) > budget.max_latency_ms


def _rejection(hit: SearchHit, budget: ContextBudget, reference_at: datetime) -> str | None:
    """Return which budget bound excludes this hit, or None when it is a candidate."""
    if budget.memory_types is not None and hit.memory_type not in budget.memory_types:
        return "memory_types"
    # A record with no typed context is evidence in its own right, not a low-confidence inference.
    confidence = 1.0 if hit.context is None else hit.context.confidence
    if confidence < budget.min_confidence:
        return "min_confidence"
    if budget.freshness is not None and reference_at - _anchor(hit) > budget.freshness:
        return "freshness"
    return None


def _anchor(hit: SearchHit) -> datetime:
    """Freshness anchors on event end, then event start, then when the record was written."""
    return hit.occurred_end or hit.occurred_at or hit.created_at


def _select(
    closures: Sequence[EvidenceClosure],
    *,
    goal: str,
    budget: ContextBudget,
    reference_at: datetime,
    overhead: int,
    excerpts: Mapping[str, ContextExcerpt],
    projections: Mapping[str, ContextProjection],
    closed_evidence: bool,
    context_selector: ContextSelector,
) -> tuple[
    dict[str, tuple[SearchHit, ...]],
    tuple[ContextExcerpt, ...],
    tuple[ContextProjection, ...],
    tuple[ContextProof, ...],
    frozenset[str],
]:
    """Validate a selector's anchor proposal and admit only original complete evidence."""
    by_id: dict[str, EvidenceClosure] = {}
    for closure in closures:
        by_id.setdefault(closure.anchor.id, closure)
    candidates = []
    for closure in by_id.values():
        sections: dict[str, list[SearchHit]] = {name: [] for name in _SECTIONS}
        for member in closure.members:
            sections[_section(member)].append(member)
        excerpt = excerpts.get(closure.anchor.id) if len(closure.members) == 1 else None
        projection = projections.get(closure.anchor.id) if len(closure.members) == 1 else None
        candidates.append(
            ContextCandidate(
                hit=closure.anchor,
                section=_section(closure.anchor),
                chars=bundle_cost(closure.anchor),
                section_chars=_heading_cost(_section(closure.anchor)),
                required_ids=tuple(member.id for member in closure.members),
                closure_chars=_bundle_chars(
                    0,
                    {name: tuple(section) for name, section in sections.items()},
                    proofs=closure.proofs,
                ),
                closure_media_items=sum(len(member.assets) for member in closure.members),
                excerpt_chars=None if excerpt is None else excerpt_cost(excerpt),
                projection_chars=None if projection is None else projection_cost(projection),
            )
        )
    selection = context_selector.select(
        ContextSelectionRequest(
            goal=goal,
            candidates=tuple(candidates),
            budget=budget,
            reference_at=reference_at,
            reserved_chars=overhead,
            excerpt_section_chars=_heading_cost("partial sources"),
            projection_section_chars=_heading_cost("text-only sources"),
            closed_evidence=closed_evidence,
        )
    )
    if not isinstance(selection, ContextSelection) or not isinstance(selection.diversify, bool):
        raise ValidationError("context selector must return ContextSelection")
    proposed = selection.anchor_ids
    if not isinstance(proposed, tuple) or any(not isinstance(item, str) for item in proposed):
        raise ValidationError("context selector must return a tuple of candidate anchor IDs")
    if len(set(proposed)) != len(proposed):
        raise ValidationError("context selector returned duplicate candidate anchor IDs")
    if any(memory_id not in by_id for memory_id in proposed):
        raise ValidationError("context selector returned an unknown candidate anchor ID")
    return (
        *_select_closures(
            tuple(by_id[memory_id] for memory_id in proposed),
            budget,
            overhead,
            {**excerpts, **projections},
            diversify=selection.diversify and not closed_evidence,
        ),
        frozenset(by_id).difference(proposed),
    )


def _expand_conflict_closures(  # noqa: C901 - fixed-point conflict closure is one graph pass
    candidates: Sequence[SearchHit],
    closures: Mapping[str, EvidenceClosure],
) -> tuple[tuple[EvidenceClosure, ...], int]:
    """Make candidate-bounded conflicting representatives co-required with an anchor.

    Returns the expanded closures and how many anchors were withheld because a conflicting
    representative had no closure of its own.
    """
    representatives: dict[str, tuple[str, ...]] = {}
    for conflict in _conflicts(candidates, frozenset(hit.id for hit in candidates)):
        representatives[conflict.lineage_id] = conflict.memory_ids
    attribution_peers: dict[AffectLinkKey, list[str]] = {}
    for candidate in candidates:
        key = affect_link_key(candidate)
        if key is not None:
            attribution_peers.setdefault(key, []).append(candidate.id)

    def required_peers(hit: SearchHit) -> tuple[str, ...]:
        context = hit.context
        lineage_peers = (
            ()
            if context is None or context.lineage_id is None
            else representatives.get(context.lineage_id, ())
        )
        key = affect_link_key(hit)
        return (*lineage_peers, *(() if key is None else attribution_peers.get(key, ())))

    expanded = []
    withheld = 0
    for anchor in candidates:
        closure = closures.get(anchor.id)
        if closure is None:
            continue
        members = {member.id: member for member in closure.members}
        proofs = dict.fromkeys(closure.proofs)
        # Conflict edges are equivalence edges rather than provenance: following them does not
        # create a bad cycle.  Iterate to a fixed point so a dependency which is itself one side
        # of an eligible candidate conflict cannot enter alone.
        pending = [peer_id for member in members.values() for peer_id in required_peers(member)]
        examined: set[str] = set()
        while pending:
            peer_id = pending.pop()
            if peer_id in examined:
                continue
            examined.add(peer_id)
            peer = closures.get(peer_id)
            if peer is None:
                # This representative was eligible in the candidate window but its own required
                # support failed.  A sibling may not make the conflict look complete by itself.
                members = {}
                break
            for member in peer.members:
                if member.id not in members:
                    members[member.id] = member
                pending.extend(required_peers(member))
            proofs.update(dict.fromkeys(peer.proofs))
        if members:
            expanded.append(EvidenceClosure(anchor, tuple(members.values()), tuple(proofs)))
        else:
            withheld += 1
    return tuple(expanded), withheld


@dataclass(slots=True)
class _Admission:
    """One authoritative ledger for record, proof, media, and partial-source admission."""

    budget: ContextBudget
    overhead: int
    partial_sources: Mapping[str, ContextExcerpt | ContextProjection]
    sections: dict[str, list[SearchHit]] = field(
        default_factory=lambda: {name: [] for name in _SECTIONS}
    )
    taken: set[str] = field(default_factory=set)
    media: int = 0
    selected_partials: dict[str, ContextExcerpt | ContextProjection] = field(default_factory=dict)
    selected_proofs: dict[ContextProof, None] = field(default_factory=dict)

    def admit(self, closure: EvidenceClosure) -> bool:
        marginal = tuple(member for member in closure.members if member.id not in self.taken)
        media_cost = sum(len(member.assets) for member in marginal)
        replacing = {member.id for member in marginal if member.id in self.selected_partials}
        proposed_sections = {name: list(section) for name, section in self.sections.items()}
        for member in marginal:
            proposed_sections[_section(member)].append(member)
        proposed_partials = tuple(
            source
            for memory_id, source in self.selected_partials.items()
            if memory_id not in replacing
        )
        item_count = len(self.taken) + len(marginal) + len(proposed_partials)
        cost = _bundle_chars(
            self.overhead,
            {name: tuple(section) for name, section in proposed_sections.items()},
            proposed_partials,
            tuple(dict.fromkeys((*self.selected_proofs, *closure.proofs))),
        )
        full_fits = (
            item_count <= self.budget.max_items
            and (
                self.budget.max_media_items is None
                or self.media + media_cost <= self.budget.max_media_items
            )
            and cost <= self.budget.max_chars
        )
        if full_fits:
            self.sections = proposed_sections
            for memory_id in replacing:
                self.selected_partials.pop(memory_id)
            self.taken.update(member.id for member in marginal)
            self.media += media_cost
            self.selected_proofs.update(dict.fromkeys(closure.proofs))
            return True
        if (
            len(closure.members) != 1
            or closure.anchor.id in self.taken
            or closure.anchor.id in self.selected_partials
        ):
            return False
        source = self.partial_sources.get(closure.anchor.id)
        if source is None:
            return False
        proposed_partials = (*self.selected_partials.values(), source)
        if len(self.taken) + len(proposed_partials) > self.budget.max_items:
            return False
        if (
            _bundle_chars(
                self.overhead,
                {name: tuple(section) for name, section in self.sections.items()},
                proposed_partials,
                tuple(self.selected_proofs),
            )
            > self.budget.max_chars
        ):
            return False
        self.selected_partials[closure.anchor.id] = source
        return True


def _select_closures(
    closures: Sequence[EvidenceClosure],
    budget: ContextBudget,
    overhead: int,
    partial_sources: Mapping[str, ContextExcerpt | ContextProjection] = MappingProxyType({}),
    *,
    diversify: bool = False,
) -> tuple[
    dict[str, tuple[SearchHit, ...]],
    tuple[ContextExcerpt, ...],
    tuple[ContextProjection, ...],
    tuple[ContextProof, ...],
]:
    """Own budget admission, including diversity rounds and partial-source replacement."""
    admission = _Admission(budget, overhead, partial_sources)
    represented: set[str] = set()
    head = -(-budget.max_items // 2)
    sections_available = {_section(closure.anchor) for closure in closures}
    rounds = (
        ((head, False), (budget.max_items, True), (budget.max_items, False))
        if diversify and budget.max_items - head >= len(sections_available)
        else ((budget.max_items, False),)
    )
    for ceiling, floor in rounds:
        for closure in closures:
            section = _section(closure.anchor)
            if diversify:
                if len(admission.taken) + len(admission.selected_partials) >= ceiling:
                    break
                if (
                    closure.anchor.id in admission.taken
                    or closure.anchor.id in admission.selected_partials
                ):
                    continue
                if floor and section in represented:
                    continue
            if admission.admit(closure):
                represented.add(section)
    return (
        {name: tuple(section) for name, section in admission.sections.items()},
        tuple(
            source
            for source in admission.selected_partials.values()
            if isinstance(source, ContextExcerpt)
        ),
        tuple(
            source
            for source in admission.selected_partials.values()
            if isinstance(source, ContextProjection)
        ),
        tuple(admission.selected_proofs),
    )


def _affect_cue(hit: SearchHit, event_ids: Sequence[str]) -> AffectCue:
    """Return one affect hit as the evidence-first entry the bundle publishes.

    The cue's own evidence stays where retrieval hydrated it, on context.evidence_ids, so the
    entry cannot disagree with itself. The resolver returns event_ids already deduplicated and
    ordered; at most _MAX_CUE_IDS are kept, so one capture that formed many events cannot
    grow a line the budget never charged for.
    """
    return AffectCue(
        **{field.name: getattr(hit, field.name) for field in fields(SearchHit)},
        event_ids=tuple(event_ids)[:_MAX_CUE_IDS],
    )


def _bundle_chars(
    overhead: int,
    sections: Mapping[str, tuple[SearchHit, ...]],
    partial_sources: Sequence[ContextExcerpt | ContextProjection] = (),
    proofs: Sequence[ContextProof] = (),
) -> int:
    """Return what this bundle charged against max_chars, priced exactly as _select did."""
    excerpts = tuple(source for source in partial_sources if isinstance(source, ContextExcerpt))
    projections = tuple(
        source for source in partial_sources if isinstance(source, ContextProjection)
    )
    return (
        overhead
        + sum(
            _heading_cost(name) + sum(bundle_cost(hit) for hit in section)
            for name, section in sections.items()
            if section
        )
        + (
            _heading_cost("partial sources") + sum(excerpt_cost(excerpt) for excerpt in excerpts)
            if excerpts
            else 0
        )
        + (
            _heading_cost("text-only sources")
            + sum(projection_cost(projection) for projection in projections)
            if projections
            else 0
        )
        + (
            _heading_cost("selected proofs") + sum(len(proof.render()) + 1 for proof in proofs)
            if proofs
            else 0
        )
    )


def _section(hit: SearchHit) -> str:
    context = hit.context
    if context is not None and context.kind in _KIND_SECTIONS:
        return _KIND_SECTIONS[context.kind]
    return _TYPE_SECTIONS[hit.memory_type]


def _unknowns(
    supplied: Sequence[ContextUnknown],
    budget: ContextBudget,
    sections: Mapping[str, tuple[SearchHit, ...]],
    excerpts: Sequence[ContextExcerpt],
    projections: Sequence[ContextProjection],
    excluded: Mapping[str, int],
    omitted: int,
    *,
    skipped: bool,
    exhausted: bool = False,
    candidate_limit: int | None = None,
    actor_excluded: int = 0,
) -> tuple[ContextUnknown, ...]:
    """Name what the request implied and this bundle does not carry, in one stable order."""
    found = list(supplied)
    if projections:
        media_count = sum(
            part.kind == "asset" for projection in projections for part in projection.parts
        )
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.BUDGET_EXCLUDED,
                detail=(context_prompts.text_only_delivery(len(projections), media_count)),
            )
        )
    found.extend(
        ContextUnknown(
            kind=ContextUnknownKind.BUDGET_EXCLUDED,
            detail=context_prompts.candidates_excluded(excluded[reason], reason),
        )
        for reason in _EXCLUSION_REASONS
        if excluded.get(reason)
    )
    if omitted:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.BUDGET_EXCLUDED,
                detail=(
                    context_prompts.budget_excluded(
                        omitted, max_items=budget.max_items, max_chars=budget.max_chars
                    )
                ),
            )
        )
    if actor_excluded:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.BUDGET_EXCLUDED,
                detail=context_prompts.actors_excluded(actor_excluded, budget.max_chars),
            )
        )
    if exhausted:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.CANDIDATES_EXHAUSTED,
                detail=(context_prompts.candidate_window_exhausted(candidate_limit)),
            )
        )
    found.extend(_section_empty(budget, sections, excerpts, projections))
    if skipped:
        found.append(
            ContextUnknown(
                kind=ContextUnknownKind.STAGE_SKIPPED,
                detail=(context_prompts.enrichment_deadline(budget.max_latency_ms)),
            )
        )
    return tuple(sorted(found, key=lambda item: (item.kind.value, item.detail)))


def _section_empty(
    budget: ContextBudget,
    sections: Mapping[str, tuple[SearchHit, ...]],
    excerpts: Sequence[ContextExcerpt],
    projections: Sequence[ContextProjection],
) -> tuple[ContextUnknown, ...]:
    """Name each section a memory_types request left empty, and why it is empty.

    Only a request that named types gets these: without one an empty section means the store
    holds nothing for it, which is not a statement about this request. With one, an empty
    section is either a section the filter can never fill -- its records carry a type the
    request excluded -- or one the filter admits and no record of that type reached the bundle
    at all (feeds.isdisjoint(present)). The second variant is deliberately narrower than "this
    section is empty": a kind-keyed section (actors, scene, affect, traits) that is empty
    while a *different* section already carries a record of the same underlying MemoryCategory
    stays silent, because flagging it would be noise in most bundles -- the type was not missing,
    it just formed into another kind.
    """
    requested = budget.memory_types
    if requested is None:
        return ()
    present = frozenset(hit.memory_type for section in sections.values() for hit in section)
    present |= frozenset(excerpt.memory_type for excerpt in excerpts)
    present |= frozenset(projection.memory_type for projection in projections)
    found: list[ContextUnknown] = []
    for name in _SECTIONS:
        if sections[name]:
            continue
        feeds = _SECTION_MEMORY_TYPES[name]
        if feeds.isdisjoint(requested):
            detail = context_prompts.section_type_excluded(
                name, requested=_type_names(requested), feeds=_type_names(feeds)
            )
        elif feeds <= requested and feeds.isdisjoint(present):
            detail = context_prompts.section_no_memory(name, _type_names(feeds))
        else:
            continue
        found.append(ContextUnknown(kind=ContextUnknownKind.SECTION_EMPTY, detail=detail))
    return tuple(found)


def _type_names(memory_types: frozenset[MemoryCategory]) -> str:
    return ", ".join(memory_type.value for memory_type in sorted(memory_types))


def _conflicts(
    hits: Sequence[SearchHit],
    included_ids: frozenset[str],
) -> tuple[ContextConflict, ...]:
    lineages: dict[str, list[_Claim]] = {}
    for hit in hits:
        context = hit.context
        if context is None or (functional := functional_lineage(context)) is None:
            continue
        lineage_id, value = functional
        lineages.setdefault(lineage_id, []).append(
            (value, hit.id, context.subject, context.predicate)
        )
    # A lineage no included memory takes part in is not this bundle's disagreement to report.
    return tuple(
        conflict
        for lineage_id, claims in lineages.items()
        if any(memory_id in included_ids for _value, memory_id, _subject, _predicate in claims)
        and (conflict := _lineage_conflict(lineage_id, claims)) is not None
    )


def _lineage_conflict(lineage_id: str, claims: Sequence[_Claim]) -> ContextConflict | None:
    claimed: dict[str, str] = {}
    for value, memory_id, _subject, _predicate in claims:
        claimed.setdefault(value, memory_id)
    if len(claimed) < 2:
        return None
    _value, _memory_id, subject, predicate = claims[0]
    return ContextConflict(
        lineage_id=lineage_id,
        subject=subject,
        predicate=predicate,
        values=tuple(claimed),
        memory_ids=tuple(claimed.values()),
    )


def _occurred_range(
    hits: Sequence[SearchHit | ContextExcerpt | ContextProjection],
) -> tuple[datetime | None, datetime | None]:
    spans = tuple(
        (hit.occurred_at, hit.occurred_end or hit.occurred_at)
        for hit in hits
        if hit.occurred_at is not None
    )
    if not spans:
        return None, None
    return min(start for start, _end in spans), max(end for _start, end in spans)


def _frames(hits: Sequence[SearchHit | ContextExcerpt | ContextProjection]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                hit.context.spatial.frame_id
                for hit in hits
                if hit.context is not None and hit.context.spatial is not None
            }
        )
    )


def _provisional_actors(
    hits: Sequence[SearchHit | ContextExcerpt],
    provisional: Mapping[str, Sequence[str]],
    *,
    exclude: frozenset[str] = frozenset(),
) -> tuple[ProvisionalActor, ...]:
    """Name every unnamed person the included evidence observed, in identity order.

    These are appended to the ranked actors rather than competing with them for an item slot:
    an unnamed person earned no hit slot, and dropping them would leave an agent unable to say
    that somebody it does not recognize is in the room. exclude drops an identity named
    already reported, so a stale provisional entry can never list the same identity twice.
    """
    if not provisional:
        return ()
    observed: dict[str, list[str]] = {}
    for hit in hits:
        memory_id = _evidence_id(hit)
        for identity_id in provisional.get(memory_id, ()):
            if identity_id in exclude:
                continue
            memory_ids = observed.setdefault(identity_id, [])
            if memory_id not in memory_ids:
                memory_ids.append(memory_id)
    return tuple(
        ProvisionalActor(identity_id=identity_id, memory_ids=tuple(memory_ids))
        for identity_id, memory_ids in sorted(observed.items())
    )


def _named_actors(
    hits: Sequence[SearchHit | ContextExcerpt],
    named: Mapping[str, Sequence[NamedActorLink]],
) -> tuple[NamedActor, ...]:
    """Name every identity the included evidence's identity edge names, in identity order.

    Aggregated across every included memory that carries the edge, the same way
    _provisional_actors aggregates an unnamed person's observing memories, and dropped
    entirely when the evidence that carried the edge did not make it into the bundle.
    """
    if not named:
        return ()
    names: dict[str, str] = {}
    assertions: dict[str, str | None] = {}
    observed: dict[str, list[str]] = {}
    for hit in hits:
        memory_id = _evidence_id(hit)
        for identity_id, name, naming_assertion_id in named.get(memory_id, ()):
            names[identity_id] = name
            assertions[identity_id] = naming_assertion_id
            memory_ids = observed.setdefault(identity_id, [])
            if memory_id not in memory_ids:
                memory_ids.append(memory_id)
    return tuple(
        NamedActor(
            identity_id=identity_id,
            name=names[identity_id],
            memory_ids=tuple(memory_ids),
            naming_assertion_id=assertions[identity_id],
        )
        for identity_id, memory_ids in sorted(observed.items())
    )


def _evidence_id(hit: SearchHit | ContextExcerpt) -> str:
    return hit.id if isinstance(hit, SearchHit) else hit.source_memory_id


def _fit_actors(
    candidates: Sequence[NamedActor | ProvisionalActor],
    remaining: int,
) -> tuple[tuple[NamedActor | ProvisionalActor, ...], int, int]:
    """Fit actor lines into what is left of max_chars after the ranked hits.

    Priced and greedy exactly like _select prices a hit: one oversized entry does not close
    the bundle to the rest, so a cheaper actor can still fit. Returns the entries that fit,
    what they cost, and how many did not fit.
    """
    fitted: list[NamedActor | ProvisionalActor] = []
    spent = 0
    excluded = 0
    for entry in candidates:
        cost = _actor_cost(entry)
        if cost <= remaining - spent:
            fitted.append(entry)
            spent += cost
        else:
            excluded += 1
    return tuple(fitted), spent, excluded
