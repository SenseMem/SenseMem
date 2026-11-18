"""Pure, replaceable proposals for selecting already-authorized context evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from sensemem.context_values import ContextBudget
from sensemem.records import SearchHit


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextCandidate:
    """One eligible anchor and the kernel-calculated cost of its required evidence.

    ``chars`` prices the anchor's rendered line, including its media; ``section_chars``
    prices its section heading. ``closure_chars`` includes every required record, heading,
    and proof when admitted alone. Shared records and proofs are charged once by the kernel,
    so that standalone cost is an upper bound on the marginal cost of a later admission.
    ``required_ids`` includes the anchor and any evidence or conflicting assertions it needs.
    These values describe evidence; a selector cannot replace or edit it.
    """

    hit: SearchHit
    section: str
    chars: int
    section_chars: int
    required_ids: tuple[str, ...]
    closure_chars: int
    closure_media_items: int
    excerpt_chars: int | None = None
    projection_chars: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextSelectionRequest:
    """Authorized candidates and the remaining evidence budget for one selection pass.

    ``reserved_chars`` already includes the context frame and admitted temporary context;
    ``budget.max_items`` is the number of memory slots left after that temporary context.
    ``closed_evidence`` means admission must preserve multi-record or proof-bearing closures.
    A compiler may request selection again after optional evidence annotations change costs.
    """

    goal: str
    candidates: tuple[ContextCandidate, ...]
    budget: ContextBudget
    reference_at: datetime
    reserved_chars: int
    excerpt_section_chars: int
    projection_section_chars: int
    closed_evidence: bool


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """Anchor priorities and a section-diversity preference, without admitting evidence."""

    anchor_ids: tuple[str, ...]
    diversify: bool = False


@runtime_checkable
class ContextSelector(Protocol):
    """Propose anchor priority and diversity; the compiler alone admits evidence.

    Returning a subset deliberately excludes other anchors; empty ``anchor_ids`` excludes
    all of them. The kernel retains consent, complete evidence, conflicts, and every budget
    check. Selection priority does not change the retrieval scores on the resulting hits.
    Implementations are deterministic computations over the request and perform no model or
    storage work. ``Memory`` owns the selector and closes it with its other plugins.
    """

    def select(self, request: ContextSelectionRequest) -> ContextSelection: ...

    def close(self) -> None: ...


class DefaultContextSelector:
    """Keep retrieval rank and request section diversity from the authoritative compiler."""

    def select(self, request: ContextSelectionRequest) -> ContextSelection:
        return ContextSelection(
            tuple(candidate.hit.id for candidate in request.candidates),
            diversify=not request.closed_evidence,
        )

    def close(self) -> None:
        """Release no resources; the default selector is stateless."""
