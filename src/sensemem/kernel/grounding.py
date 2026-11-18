"""Pure answer-evidence selection, packing, and request annotations.

These functions read no storage and call no model. The answer plane owns execution;
this module owns the deterministic policy over the evidence it is given.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from math import ceil

from sensemem.context import EvidenceClosure, evidence_cost
from sensemem.models.base import ModelInput
from sensemem.prompts import recall as recall_prompts
from sensemem.recall import (
    RECALL_NON_SELECTIVE_MIN_ROWS,
    RECALL_NON_SELECTIVE_SHARE,
    RecallResult,
)
from sensemem.records import AnswerResult, SearchHit
from sensemem.types import Modality

_RECALL_ATTEMPT_CHARS = 6_000


@dataclass(frozen=True, slots=True)
class EvidenceExpansion:
    """The linked rows one expansion read, and which edge each of them came through."""

    hits: tuple[SearchHit, ...]
    edges: Mapping[str, int]
    dropped: tuple[str, ...]


def expansion_ceiling(limit: int, active_records: int) -> int:
    """The per-edge bound, which is the bound recall programs already apply to a predicate.

    Reused rather than restated: an edge that links to a fifth of the corpus and a `match` that
    selects a fifth of it fail the same way, by describing the store instead of the question.
    """
    return max(
        RECALL_NON_SELECTIVE_MIN_ROWS * limit,
        ceil(RECALL_NON_SELECTIVE_SHARE * active_records),
    )


def pack_grounding(
    required: Sequence[SearchHit],
    expansion: Sequence[SearchHit],
    ranked: Sequence[SearchHit],
    budget_chars: int | None,
    *,
    media_limit: int,
    max_added: int | None = None,
) -> tuple[tuple[SearchHit, ...], int]:
    """Fill one budget with the window, then the rest of the ranking, then its linked rows.

    Linked evidence spends only the ranking's unused allowance: it may not displace evidence
    the query already earned. Without a character budget there is no ranked tail. Media and
    row caps apply only to additions; the mandatory window and ranked tail remain protected.
    """
    selected = list(required)
    taken = {hit.id for hit in selected}
    spent = sum(evidence_cost(hit) for hit in selected)
    # Without a budget the window is `limit` hits and there is no tail: walking the ranking here
    # would hand the reader the whole rerank pool, which is not what either path ever grounded.
    tail = ranked if budget_chars is not None else ()
    # Attribute only new admissions to expansion, not linked rows already bought by the ranking.
    added = 0
    media = 0
    for source, from_expansion in ((tail, False), (expansion, True)):
        for hit in source:
            if hit.id in taken:
                continue
            cost = evidence_cost(hit)
            if (
                (budget_chars is not None and spent + cost > budget_chars)
                or (from_expansion and hit.assets and media >= media_limit)
                or (from_expansion and max_added is not None and added >= max_added)
            ):
                # A packing pass, not a prefix: an oversized row, or one over the media cap, must
                # not block a smaller or text-only row behind it from using what is left.
                continue
            selected.append(hit)
            taken.add(hit.id)
            spent += cost
            added += 1 if from_expansion else 0
            media += 1 if from_expansion and hit.assets else 0
    return tuple(selected), added


def close_grounding(
    hits: Sequence[SearchHit],
    closures: Sequence[EvidenceClosure],
    required: Sequence[SearchHit],
    budget_chars: int | None,
) -> tuple[SearchHit, ...]:
    """Admit assertions with their full provenance, preserving the mandatory raw window.

    A soft budget may retain an oversized raw observation. It cannot admit a derived assertion
    while leaving its sources out. Shared sources are charged once; existing evidence keeps its
    selected order and newly hydrated sources follow it.
    """
    required_ids = {hit.id for hit in required}
    selected = {
        hit.id: hit
        for hit in hits
        if hit.id in required_ids and (hit.context is None or not hit.context.evidence_ids)
    }
    used = sum(evidence_cost(hit) for hit in selected.values())
    for closure in closures:
        additions = tuple(hit for hit in closure.members if hit.id not in selected)
        cost = sum(evidence_cost(hit) for hit in additions)
        if budget_chars is not None and used + cost > budget_chars:
            continue
        selected.update((hit.id, hit) for hit in additions)
        used += cost
    original_ids = {hit.id for hit in hits}
    return (
        *(hit for hit in hits if hit.id in selected),
        *(hit for hit in selected.values() if hit.id not in original_ids),
    )


def semantic_hits(
    routes: Sequence[Sequence[SearchHit]], excluded: set[str], limit: int
) -> tuple[SearchHit, ...]:
    """Offer each semantic route's next novel row before any route spends another seat."""
    taken = set(excluded)
    offered: list[SearchHit] = []
    route_iters = tuple(iter(route) for route in routes)
    for _ in range(limit):
        for route in route_iters:
            for hit in route:
                if hit.id in taken:
                    continue
                offered.append(hit)
                taken.add(hit.id)
                break
    return tuple(offered)


def recall_ranked_tail(
    program: RecallResult, ranked: tuple[SearchHit, ...], *, budgeted: bool
) -> tuple[SearchHit, ...]:
    """Fill an explicit evidence budget; otherwise retain the plan's ranked depth."""
    if not budgeted:
        depth = max((step.k for step in program.plan.steps if step.op == "similar"), default=0)
        ranked = ranked[:depth]
    matched = {hit.id for hit in program.exhaustive}
    return tuple(hit for hit in ranked if hit.id not in matched)


def budget_recall(
    program: RecallResult,
    budget_chars: int,
    *,
    media_limit: int,
    max_rows: int,
    required: Sequence[SearchHit],
) -> tuple[tuple[SearchHit, ...], int]:
    """Union the program's set with the window the ranking earned, inside one budget.

    Matched rows remain chronological; similarity rows retain rank, with ID deduplication.
    The unplanned path's `required` window survives every bound. Character, media, and row caps
    constrain only supplemental evidence. The returned shortfall counts omitted matched rows;
    ranking omissions cannot make an otherwise complete predicate read incomplete.
    """
    exhaustive_ids = {hit.id for hit in program.exhaustive}
    ranked = {hit.id: hit for hit in (*program.ranked, *required) if hit.id not in exhaustive_ids}
    required_ids = {hit.id for hit in required}
    selected: list[SearchHit] = []
    # Required rows may come after optional matches in chronological order. Reserve their cost
    # before admitting any additions, then count an overlapping required row only once.
    spent = sum(evidence_cost(hit) for hit in required)
    media = 0
    omitted = 0
    for index, hit in enumerate((*program.exhaustive, *ranked.values())):
        exhaustive_row = index < len(program.exhaustive)
        cost = evidence_cost(hit)
        if hit.id not in required_ids and (
            (spent + cost > budget_chars)
            or (exhaustive_row and hit.assets and media >= media_limit)
            or (exhaustive_row and index >= max_rows)
        ):
            omitted += 1 if exhaustive_row else 0
            continue
        selected.append(hit)
        if hit.id not in required_ids:
            spent += cost
        media += 1 if exhaustive_row and hit.assets and hit.id not in required_ids else 0
    return tuple(selected), omitted


def with_recall_note(question: ModelInput, note: str | None) -> ModelInput:
    """Attach the component's rendering without changing the input's media or order."""
    if note is None or not question.text:
        return question
    return replace(question, text=recall_prompts.with_recall_note(question.text, note))


def with_reference_time(question: ModelInput, reference_at: datetime) -> ModelInput:
    """Attach the trusted answering clock while retaining the original input assets."""
    return replace(question, text=recall_prompts.with_reference_time(question.text, reference_at))


def attempt_note(
    hits: Sequence[SearchHit], result: AnswerResult, *, include_evidence: bool = False
) -> str:
    """Select bounded replanning excerpts before delegating their presentation."""
    reason = None if result.abstention_reason is None else result.abstention_reason.value
    if include_evidence:
        evidence: list[dict[str, object]] = []
        seen: set[str] = set()
        records_read = len({hit.id for hit in hits})
        status = "insufficient_evidence" if result.abstained else "answered"
        for hit in hits:
            if hit.id in seen:
                continue
            seen.add(hit.id)
            evidence.append(
                {
                    "alias": f"R{len(evidence) + 1}",
                    "source_id": None if hit.context is None else hit.context.source_id,
                    "occurred_at": None if hit.occurred_at is None else hit.occurred_at.isoformat(),
                    "excerpt": hit.content[:1_500],
                    "excerpt_truncated": len(hit.content) > 1_500,
                }
            )
            rendered = recall_prompts.attempt_evidence(
                status=status, reason=reason, records_read=records_read, evidence=evidence
            )
            if len(rendered) > _RECALL_ATTEMPT_CHARS:
                evidence.pop()
                break
        return recall_prompts.attempt_evidence(
            status=status, reason=reason, records_read=records_read, evidence=evidence
        )
    dates = sorted(
        hit.occurred_at.date().isoformat() for hit in hits if hit.occurred_at is not None
    )
    return recall_prompts.attempt_summary(len(hits), dates, reason)


def grounding_hits(
    hits: Sequence[SearchHit],
    limit: int,
    *,
    budget_chars: int | None = None,
) -> tuple[SearchHit, ...]:
    """Ground on the ranking's own order, guaranteeing one hit per modality it contains.

    A missing modality takes the lowest-ranked replaceable seat. The top hit is protected,
    and a promotion cannot remove a modality already represented. When modalities exceed the
    available slots, the ranking decides which fit; no equal-share rotation replaces it.
    """
    best_by_modality: dict[Modality, SearchHit] = {}
    for hit in hits:
        best_by_modality.setdefault(hit.modality, hit)
    selected = list(hits[:limit])
    represented = {hit.modality for hit in selected}
    promoted = [hit for modality, hit in best_by_modality.items() if modality not in represented]
    for hit in promoted[: max(limit - 1, 0)]:
        if len(selected) >= limit:
            counts = Counter(chosen.modality for chosen in selected)
            spare = next(
                (
                    index
                    for index in range(len(selected) - 1, 0, -1)
                    if counts[selected[index].modality] > 1
                ),
                None,
            )
            if spare is None:
                break
            del selected[spare]
        selected.append(hit)
    if budget_chars is None:
        return tuple(selected)
    return (*selected, *budgeted_hits(hits, selected, budget_chars))


def budgeted_hits(
    hits: Sequence[SearchHit],
    selected: Sequence[SearchHit],
    budget_chars: int,
) -> tuple[SearchHit, ...]:
    """Extend a grounding set down the ranking while the evidence fits one budget.

    The guaranteed hits are never dropped, so this only ever widens what the answer sees. Media
    is charged per modality because an image or video part costs the model far more than its
    record's text; the provider adapter still enforces its own byte ceiling.
    """
    taken = {hit.id for hit in selected}
    used = sum(evidence_cost(hit) for hit in selected)
    # Deduplicate IDs only: similar prose can describe different events, times, or measurements.
    extra: list[SearchHit] = []
    for hit in hits:
        if hit.id in taken:
            continue
        cost = evidence_cost(hit)
        if used + cost > budget_chars:
            # The selected prefix remains mandatory, but this supplemental scan is a packing
            # pass. An oversized lower-ranked hit must not prevent a later, smaller record from
            # using the remaining character budget.
            continue
        used += cost
        extra.append(hit)
    return tuple(extra)
