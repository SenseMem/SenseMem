"""Context bundle values and pure operations for the public SDK."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

from sensemem._context_proofs import proof_support
from sensemem._naming_conditions import is_naming_condition
from sensemem.context_rendering import compact_context, render_context
from sensemem.context_values import (
    AffectCue,
    ContextBudget,
    ContextConflict,
    ContextExcerpt,
    ContextPresentation,
    ContextProjection,
    ContextProof,
    ContextProofScoring,
    ContextSupplement,
    ContextUnknown,
    NamedActor,
    ProvisionalActor,
)
from sensemem.evidence_policy import required_support
from sensemem.exceptions import ValidationError
from sensemem.prompts import context as context_prompts
from sensemem.records import SearchHit
from sensemem.types import EvidenceBasis, RecordKind


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextBundle:
    """One bounded, structured context view compiled for a goal."""

    goal: str
    reference_at: datetime
    budget: ContextBudget
    actors: tuple[SearchHit | NamedActor | ProvisionalActor, ...]
    relationships: tuple[SearchHit, ...]
    scene: tuple[SearchHit, ...]
    episodes: tuple[SearchHit, ...]
    facts: tuple[SearchHit, ...]
    procedures: tuple[SearchHit, ...]
    # Evidence-first: an affect entry carries its provenance and its one co-occurrence hop.
    affect: tuple[AffectCue, ...]
    traits: tuple[SearchHit, ...]
    conflicts: tuple[ContextConflict, ...]
    unknowns: tuple[ContextUnknown, ...]
    occurred_from: datetime | None
    occurred_until: datetime | None
    frames: tuple[str, ...]
    places: tuple[str, ...]
    omitted: int
    chars: int
    elapsed_ms: int
    deadline_exceeded: bool
    excerpts: tuple[ContextExcerpt, ...] = ()
    projections: tuple[ContextProjection, ...] = ()
    proofs: tuple[ContextProof, ...] = ()
    proof_scoring: ContextProofScoring = "independent_pair"
    supplements: tuple[ContextSupplement, ...] = ()
    omitted_supplements: int = 0

    def __post_init__(self) -> None:  # noqa: C901 - validate the complete certificate delivery contract
        supplements = tuple(self.supplements)
        if any(not isinstance(item, ContextSupplement) for item in supplements):
            raise ValidationError("supplements must contain ContextSupplement values")
        object.__setattr__(self, "supplements", supplements)
        if (
            isinstance(self.omitted_supplements, bool)
            or not isinstance(self.omitted_supplements, int)
            or self.omitted_supplements < 0
        ):
            raise ValidationError("omitted_supplements must be a non-negative integer")
        # The affect section is the one whose entries carry more than a hit does, and every
        # transport reads that extra without asking. A plain hit here would serialize as an
        # affect cue with an empty hop, so the type is enforced rather than defaulted.
        if any(not isinstance(cue, AffectCue) for cue in self.affect):
            raise ValidationError("affect entries must be AffectCue")
        if any(not isinstance(excerpt, ContextExcerpt) for excerpt in self.excerpts):
            raise ValidationError("excerpts must contain ContextExcerpt values")
        if {hit.id for hit in self.hits} & {excerpt.source_memory_id for excerpt in self.excerpts}:
            raise ValidationError("a bundle cannot contain a full hit and excerpt of one source")
        if any(not isinstance(item, ContextProjection) for item in self.projections):
            raise ValidationError("projections must contain ContextProjection values")
        projected_ids = {item.source_memory_id for item in self.projections}
        if len(projected_ids) != len(self.projections) or projected_ids & (
            {hit.id for hit in self.hits} | {item.source_memory_id for item in self.excerpts}
        ):
            raise ValidationError("a projected source must appear exactly once in a bundle")
        if any(not isinstance(proof, ContextProof) for proof in self.proofs):
            raise ValidationError("proofs must contain ContextProof values")
        if self.proof_scoring not in ("capture_groups", "independent_pair"):
            raise ValidationError("proof_scoring is invalid")
        if self.proofs and not self.budget.selected_proofs:
            raise ValidationError("delivery certificates require selected_proofs")
        hits = {hit.id: hit for hit in self.hits}
        by_anchor: dict[str, list[ContextProof]] = {}
        source_groups: dict[str, str | None] = {}
        for proof in self.proofs:
            by_anchor.setdefault(proof.anchor_id, []).append(proof)
            for node in proof.nodes:
                if self.proof_scoring == "capture_groups" and node.source_group_id is None:
                    raise ValidationError("capture-group proofs require stored source_group_id")
                if (
                    node.memory_id in source_groups
                    and source_groups[node.memory_id] != node.source_group_id
                ):
                    raise ValidationError("a proof node must have one consistent source_group_id")
                source_groups[node.memory_id] = node.source_group_id
                if node.memory_id not in hits:
                    raise ValidationError("every proof node must be delivered as a full hit")
                context = hits[node.memory_id].context
                if node.sources and (
                    context is None or not set(node.sources) <= set(context.evidence_ids)
                ):
                    raise ValidationError(
                        "proof sources must belong to the authoritative evidence union"
                    )
                if node.capture_id is not None:
                    if context is not None and (
                        context.evidence_ids
                        or (
                            context.kind is not RecordKind.OBSERVATION
                            and context.basis
                            not in {EvidenceBasis.USER_STATEMENT, EvidenceBasis.RESPONSE_FEEDBACK}
                        )
                    ):
                        raise ValidationError("a derived assertion cannot be a capture leaf")
                    if node.capture_id != (
                        node.memory_id if context is None else context.source_id or node.memory_id
                    ):
                        raise ValidationError("proof capture_id must match its stored capture")
        for memory_id, witnesses in by_anchor.items():
            context = hits[memory_id].context
            conditions = frozenset(
                key
                for key, hit in hits.items()
                if context is not None
                and hit.context is not None
                and is_naming_condition(
                    hit.context.kind,
                    hit.context.predicate,
                    hit.context.basis,
                    hit.context.identity_id,
                    context.kind,
                    context.identity_id,
                    context.basis,
                )
            )
            count, confidence = proof_support(
                tuple(witnesses), conditions, scoring=self.proof_scoring
            )
            if context is not None:
                if confidence + 1e-12 < context.confidence:
                    raise ValidationError(
                        "selected witnesses must justify the delivered confidence"
                    )
                minimum = required_support(context.kind, context.basis, context.identity_id)
                if minimum > 1 and count < minimum:
                    raise ValidationError(
                        "selected trait or naming evidence requires corroborating witnesses"
                    )
        if self.proofs:
            for hit in self.hits:
                if hit.context is not None and hit.context.evidence_ids and hit.id not in by_anchor:
                    raise ValidationError(
                        "every delivered derived assertion requires its own certificate"
                    )

    @property
    def hits(self) -> tuple[SearchHit, ...]:
        """Every included hit in rank order, without duplicates.

        A provisional actor is not a hit: it is a person the evidence observed, not evidence.
        """
        merged = {
            entry.id: entry
            for _heading, section in self._sections()
            for entry in section
            if isinstance(entry, SearchHit)
        }
        return tuple(sorted(merged.values(), key=lambda hit: (-hit.score, hit.id)))

    def render(self) -> str:
        """Return the context text from the pure renderer."""
        return render_context(self)

    def compact(self) -> ContextPresentation:
        """Return the context presentation from the pure renderer."""
        return compact_context(self)

    def document(self) -> dict[str, object]:
        """Return the field-keyed projection REST, MCP, and the CLI each publish.

        Values stay as they are declared -- the transports validate them through their own
        models -- except budget, which flattens freshness to seconds, and rendered, which
        is render() rather than a stored field. Driven by fields(self) so a new section
        reaches all three surfaces instead of being spelled a fourth time in each of them.
        """
        values: dict[str, object] = {
            declared.name: getattr(self, declared.name) for declared in fields(self)
        }
        values["budget"] = self.budget.document()
        values["rendered"] = self.render()
        return values

    def _sections(
        self,
    ) -> tuple[tuple[str, tuple[SearchHit | NamedActor | ProvisionalActor, ...]], ...]:
        """Return the sections in their fixed rendering order."""
        return tuple(
            zip(
                context_prompts.SECTION_HEADINGS,
                (
                    self.actors,
                    self.relationships,
                    self.scene,
                    self.facts,
                    self.episodes,
                    self.procedures,
                    self.affect,
                    self.traits,
                ),
                strict=True,
            )
        )
