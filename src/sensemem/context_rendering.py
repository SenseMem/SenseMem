"""Context rendering values and pure operations for the public SDK."""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from typing import TYPE_CHECKING

from sensemem.prompts import context as context_prompts

if TYPE_CHECKING:
    from sensemem.context_bundle import ContextBundle
from sensemem.context_values import (
    AffectCue,
    ContextConflict,
    ContextExcerpt,
    ContextPresentation,
    ContextProjection,
    ContextSupplement,
    ContextSymbol,
    ContextSymbolCoverage,
    ContextSymbolNamespace,
    ContextSymbolRole,
    NamedActor,
    ProvisionalActor,
    _symbol,
)
from sensemem.records import SearchHit


def render_context(bundle: ContextBundle) -> str:
    """Return the bundle as deterministic sectioned text carrying [id] provenance."""
    included_ids = {hit.id for hit in bundle.hits}
    return context_prompts.render_bundle(
        goal=bundle.goal,
        reference_at=bundle.reference_at.isoformat(),
        chars=bundle.chars,
        max_chars=bundle.budget.max_chars,
        included=len(bundle.hits)
        + len(bundle.excerpts)
        + len(bundle.projections)
        + len(bundle.supplements),
        max_items=bundle.budget.max_items,
        sections=tuple(
            (heading, tuple(_section_line(entry) for entry in section))
            for heading, section in bundle._sections()
        ),
        excerpts=tuple(render_excerpt_line(item) for item in bundle.excerpts),
        projections=tuple(render_projection_line(item) for item in bundle.projections),
        proofs=tuple(proof.render() for proof in bundle.proofs),
        proof_scoring=bundle.proof_scoring if bundle.budget.selected_proofs else None,
        supplements=tuple(render_supplement_line(item) for item in bundle.supplements),
        conflicts=tuple(_conflict_line(item, included_ids) for item in bundle.conflicts),
        unknowns=tuple((item.kind.value, item.detail) for item in bundle.unknowns),
        omitted=bundle.omitted,
        omitted_supplements=bundle.omitted_supplements,
    )


def compact_context(bundle: ContextBundle) -> ContextPresentation:
    """Return an optional compact view without changing selection or budget accounting.

    Only identifier fields held structurally by the formatter are aliased. Stored content,
    the goal, conflict values, and diagnostic prose are never searched or rewritten. The
    existing render(), hits, and transport document remain the stable-ID presentation.
    """
    symbols = _context_symbols(bundle)
    text = _compact_context_text(bundle, symbols)
    return ContextPresentation(text=text, symbols=symbols, chars=len(text))


def render_supplement_line(supplement: ContextSupplement) -> str:
    """Render external provenance without introducing a memory citation alias."""
    return context_prompts.supplement_line(
        supplement.source,
        supplement.content,
        observed_at=None if supplement.observed_at is None else supplement.observed_at.isoformat(),
        expires_at=None if supplement.expires_at is None else supplement.expires_at.isoformat(),
    )


_MAX_CUE_IDS = 8


def _entry_symbol_references(
    entry: SearchHit | NamedActor | ProvisionalActor,
) -> tuple[tuple[ContextSymbolNamespace, str, ContextSymbolRole], ...]:
    if isinstance(entry, SearchHit):
        references = [
            (ContextSymbolNamespace.MEMORY, entry.id, ContextSymbolRole.HIT),
            *(
                (ContextSymbolNamespace.MEMORY, evidence_id, ContextSymbolRole.EVIDENCE)
                for evidence_id in (() if entry.context is None else entry.context.evidence_ids)
            ),
        ]
        if isinstance(entry, AffectCue):
            references.extend(
                (ContextSymbolNamespace.MEMORY, event_id, ContextSymbolRole.AFFECT_EVENT)
                for event_id in entry.event_ids
            )
        if entry.context is not None:
            link = entry.context.affect_link
            if link is not None:
                references.append(
                    (ContextSymbolNamespace.MEMORY, link.affect_id, ContextSymbolRole.AFFECT_FOCUS)
                )
                object_ref = link.object_ref
                references.append(
                    (
                        ContextSymbolNamespace.MEMORY
                        if object_ref.memory_id is not None
                        else ContextSymbolNamespace.IDENTITY,
                        object_ref.memory_id or object_ref.identity_id or "",
                        ContextSymbolRole.AFFECT_OBJECT,
                    )
                )
            references.extend(
                (ContextSymbolNamespace.MEMORY, witness.source_id, ContextSymbolRole.EVIDENCE)
                for witness in entry.context.affect_witnesses
            )
        return tuple(references)
    references = [
        (ContextSymbolNamespace.IDENTITY, entry.identity_id, ContextSymbolRole.ACTOR),
        *(
            (ContextSymbolNamespace.MEMORY, memory_id, ContextSymbolRole.OBSERVED_IN)
            for memory_id in entry.memory_ids
        ),
    ]
    if isinstance(entry, NamedActor) and entry.naming_assertion_id is not None:
        references.append(
            (
                ContextSymbolNamespace.MEMORY,
                entry.naming_assertion_id,
                ContextSymbolRole.NAMING_ASSERTION,
            )
        )
    return tuple(references)


def _context_symbols(bundle: ContextBundle) -> tuple[ContextSymbol, ...]:  # noqa: C901 - one ordered structural symbol scan
    full_memory_ids = {hit.id for hit in bundle.hits}
    partial_by_memory_id = {excerpt.source_memory_id: excerpt for excerpt in bundle.excerpts}
    projected_memory_ids = {item.source_memory_id for item in bundle.projections}
    ordered: list[tuple[ContextSymbolNamespace, str]] = []
    roles: dict[tuple[ContextSymbolNamespace, str], list[ContextSymbolRole]] = {}

    for _heading, section in bundle._sections():
        for entry in section:
            for namespace, stable_id, role in _entry_symbol_references(entry):
                _add_context_symbol_reference(ordered, roles, namespace, stable_id, role)
    for excerpt in bundle.excerpts:
        _add_context_symbol_reference(
            ordered,
            roles,
            ContextSymbolNamespace.MEMORY,
            excerpt.source_memory_id,
            ContextSymbolRole.EXCERPT,
        )
    for projection in bundle.projections:
        _add_context_symbol_reference(
            ordered,
            roles,
            ContextSymbolNamespace.MEMORY,
            projection.source_memory_id,
            ContextSymbolRole.PROJECTION,
        )
    for conflict in bundle.conflicts:
        for memory_id in conflict.memory_ids:
            _add_context_symbol_reference(
                ordered,
                roles,
                ContextSymbolNamespace.MEMORY,
                memory_id,
                ContextSymbolRole.CONFLICT,
            )
    for proof in bundle.proofs:
        for node in proof.nodes:
            for memory_id in (
                node.memory_id,
                *node.sources,
                *((node.capture_id,) if node.capture_id else ()),
                *((node.source_group_id,) if node.source_group_id else ()),
            ):
                _add_context_symbol_reference(
                    ordered,
                    roles,
                    ContextSymbolNamespace.MEMORY,
                    memory_id,
                    ContextSymbolRole.EVIDENCE,
                )

    next_number = {ContextSymbolNamespace.MEMORY: 1, ContextSymbolNamespace.IDENTITY: 1}
    result: list[ContextSymbol] = []
    for namespace, stable_id in ordered:
        number = next_number[namespace]
        next_number[namespace] += 1
        coverage = _context_symbol_coverage(
            namespace,
            stable_id,
            full_memory_ids,
            partial_by_memory_id,
            projected_memory_ids,
        )
        result.append(
            ContextSymbol(
                symbol=f"{'m' if namespace is ContextSymbolNamespace.MEMORY else 'i'}{number}",
                namespace=namespace,
                stable_id=stable_id,
                coverage=coverage,
                roles=tuple(roles[(namespace, stable_id)]),
                selector=(
                    partial_by_memory_id[stable_id].selector
                    if coverage is ContextSymbolCoverage.PARTIAL
                    else None
                ),
            )
        )
    return tuple(result)


def _add_context_symbol_reference(
    ordered: list[tuple[ContextSymbolNamespace, str]],
    roles: dict[tuple[ContextSymbolNamespace, str], list[ContextSymbolRole]],
    namespace: ContextSymbolNamespace,
    stable_id: str,
    role: ContextSymbolRole,
) -> None:
    key = (namespace, stable_id)
    if key not in roles:
        ordered.append(key)
        roles[key] = []
    if role not in roles[key]:
        roles[key].append(role)


def _context_symbol_coverage(
    namespace: ContextSymbolNamespace,
    stable_id: str,
    full_memory_ids: Container[str],
    partial_by_memory_id: Mapping[str, ContextExcerpt],
    projected_memory_ids: Container[str],
) -> ContextSymbolCoverage:
    if namespace is ContextSymbolNamespace.IDENTITY:
        return ContextSymbolCoverage.REFERENCE_ONLY
    if stable_id in full_memory_ids:
        return ContextSymbolCoverage.FULL
    if stable_id in partial_by_memory_id:
        return ContextSymbolCoverage.PARTIAL
    if stable_id in projected_memory_ids:
        return ContextSymbolCoverage.TEXT_ONLY
    return ContextSymbolCoverage.REFERENCE_ONLY


def _compact_context_text(bundle: ContextBundle, symbols: Sequence[ContextSymbol]) -> str:
    included_ids = {hit.id for hit in bundle.hits}
    return context_prompts.render_bundle(
        goal=bundle.goal,
        reference_at=bundle.reference_at.isoformat(),
        chars=bundle.chars,
        max_chars=bundle.budget.max_chars,
        included=len(bundle.hits)
        + len(bundle.excerpts)
        + len(bundle.projections)
        + len(bundle.supplements),
        max_items=bundle.budget.max_items,
        sections=tuple(
            (heading, tuple(_compact_section_line(entry, symbols) for entry in section))
            for heading, section in bundle._sections()
        ),
        excerpts=tuple(_compact_excerpt_line(item, symbols) for item in bundle.excerpts),
        projections=tuple(
            _render_projection_line(
                item, _symbol(item.source_memory_id, ContextSymbolNamespace.MEMORY, symbols)
            )
            for item in bundle.projections
        ),
        proofs=tuple(proof.render(symbols=symbols) for proof in bundle.proofs),
        proof_scoring=bundle.proof_scoring if bundle.budget.selected_proofs else None,
        supplements=tuple(render_supplement_line(item) for item in bundle.supplements),
        conflicts=tuple(
            _compact_conflict_line(item, included_ids, symbols) for item in bundle.conflicts
        ),
        unknowns=tuple((item.kind.value, item.detail) for item in bundle.unknowns),
        omitted=bundle.omitted,
        omitted_supplements=bundle.omitted_supplements,
        compact=True,
    )


def _compact_section_line(
    entry: SearchHit | NamedActor | ProvisionalActor, symbols: Sequence[ContextSymbol]
) -> str:
    if isinstance(entry, SearchHit):
        return _compact_hit_line(entry, symbols)
    return context_prompts.actor_line(
        _symbol(entry.identity_id, ContextSymbolNamespace.IDENTITY, symbols),
        tuple(
            _symbol(memory_id, ContextSymbolNamespace.MEMORY, symbols)
            for memory_id in entry.memory_ids
        ),
        name=entry.name if isinstance(entry, NamedActor) else None,
        naming_assertion=(
            _symbol(entry.naming_assertion_id, ContextSymbolNamespace.MEMORY, symbols)
            if isinstance(entry, NamedActor) and entry.naming_assertion_id
            else None
        ),
    )


def _compact_hit_line(hit: SearchHit, symbols: Sequence[ContextSymbol]) -> str:
    confidence = 1.0 if hit.context is None else hit.context.confidence
    marks = (
        f"confidence {confidence:.2f}{_compact_basis_and_evidence_marks(hit, symbols)}"
        f"{_affect_marks(hit)}{_affect_link_marks(hit, symbols)}"
        f"{_validity(hit)}{_compact_affect_hop(hit, symbols)}"
    )
    anchor = _symbol(hit.id, ContextSymbolNamespace.MEMORY, symbols)
    return context_prompts.hit_line(anchor, _render_source_content(hit), marks)


def _compact_excerpt_line(excerpt: ContextExcerpt, symbols: Sequence[ContextSymbol]) -> str:
    """Render an exact partial source without presenting it as a complete memory."""
    context = excerpt.context
    return context_prompts.excerpt_line(
        _symbol(excerpt.source_memory_id, ContextSymbolNamespace.MEMORY, symbols),
        ", ".join(
            f"{piece.role}:{piece.start_codepoint}-{piece.end_codepoint}"
            for piece in excerpt.selector.pieces
        ),
        excerpt.content,
        confidence=1.0 if context is None else context.confidence,
        occurred_at=None if excerpt.occurred_at is None else excerpt.occurred_at.isoformat(),
        occurred_end=None if excerpt.occurred_end is None else excerpt.occurred_end.isoformat(),
        valid_from=None
        if context is None or context.valid_from is None
        else context.valid_from.isoformat(),
        valid_until=None
        if context is None or context.valid_until is None
        else context.valid_until.isoformat(),
    )


def _compact_basis_and_evidence_marks(
    hit: SearchHit,
    symbols: Sequence[ContextSymbol],
) -> str:
    context = hit.context
    if context is None or not context.evidence_ids:
        return ""
    marks = "" if isinstance(hit, AffectCue) else f"; basis {context.basis.value}"
    if not isinstance(hit, AffectCue):
        marks += f"; evidence {_compact_id_list(context.evidence_ids, symbols)}"
    return marks


def _compact_affect_hop(hit: SearchHit, symbols: Sequence[ContextSymbol]) -> str:
    if not isinstance(hit, AffectCue):
        return ""
    marks = ""
    evidence_ids = () if hit.context is None else hit.context.evidence_ids
    if evidence_ids:
        marks += f"; from {_compact_id_list(evidence_ids, symbols)}"
    if hit.event_ids:
        marks += f"; co-occurring events {_compact_id_list(hit.event_ids, symbols)}"
    return marks


def _compact_id_list(
    memory_ids: Sequence[str],
    symbols: Sequence[ContextSymbol],
) -> str:
    shown = ", ".join(
        f"[{_symbol(memory_id, ContextSymbolNamespace.MEMORY, symbols)}]"
        for memory_id in memory_ids[:_MAX_CUE_IDS]
    )
    remaining = len(memory_ids) - _MAX_CUE_IDS
    return shown if remaining <= 0 else f"{shown}, +{remaining} more"


def _compact_conflict_line(
    conflict: ContextConflict, included: Container[str], symbols: Sequence[ContextSymbol]
) -> str:
    label = " ".join(part for part in (conflict.subject, conflict.predicate) if part)
    return context_prompts.conflict_line(
        label or conflict.lineage_id,
        tuple(
            (
                value,
                _symbol(memory_id, ContextSymbolNamespace.MEMORY, symbols),
                memory_id in included,
            )
            for value, memory_id in zip(conflict.values, conflict.memory_ids, strict=True)
        ),
    )


def _section_line(entry: SearchHit | NamedActor | ProvisionalActor) -> str:
    if isinstance(entry, SearchHit):
        return render_hit_line(entry)
    if isinstance(entry, NamedActor):
        return render_named_actor_line(entry)
    return render_provisional_actor_line(entry)


def render_named_actor_line(actor: NamedActor) -> str:
    """Return the actor line charged against the context budget."""
    return context_prompts.actor_line(
        actor.identity_id,
        actor.memory_ids,
        name=actor.name,
        naming_assertion=actor.naming_assertion_id,
    )


def render_provisional_actor_line(actor: ProvisionalActor) -> str:
    """Return the provisional actor line charged against the context budget."""
    return context_prompts.actor_line(actor.identity_id, actor.memory_ids, name=None)


def render_excerpt_line(excerpt: ContextExcerpt) -> str:
    """Render an exact partial source without presenting it as a complete memory."""
    context = excerpt.context
    return context_prompts.excerpt_line(
        excerpt.source_memory_id,
        ", ".join(
            f"{piece.role}:{piece.start_codepoint}-{piece.end_codepoint}"
            for piece in excerpt.selector.pieces
        ),
        excerpt.content,
        confidence=1.0 if context is None else context.confidence,
        occurred_at=None if excerpt.occurred_at is None else excerpt.occurred_at.isoformat(),
        occurred_end=None if excerpt.occurred_end is None else excerpt.occurred_end.isoformat(),
        valid_from=None
        if context is None or context.valid_from is None
        else context.valid_from.isoformat(),
        valid_until=None
        if context is None or context.valid_until is None
        else context.valid_until.isoformat(),
    )


def render_projection_line(projection: ContextProjection) -> str:
    """Render all text roles and omitted-media positions using the compiler's priced text."""
    return _render_projection_line(projection, projection.source_memory_id)


def _render_projection_line(projection: ContextProjection, source: str) -> str:
    assets = {asset.id: asset for asset in projection.omitted_assets}
    sections = []
    for part in projection.parts:
        modality = "media"
        if part.kind == "asset":
            asset_modality = assets[part.value].modality
            if asset_modality is not None:
                modality = asset_modality.value
        sections.append(context_prompts.projection_part(part.kind, part.value, modality=modality))
    return context_prompts.projection_line(
        source,
        sections,
        occurred_at=None if projection.occurred_at is None else projection.occurred_at.isoformat(),
        occurred_end=None
        if projection.occurred_end is None
        else projection.occurred_end.isoformat(),
    )


def render_hit_line(hit: SearchHit) -> str:
    """Return the one line render() writes for this hit.

    The compiler charges its length against ContextBudget.max_chars, so the price and the text
    come from the same function and cannot drift.
    """
    confidence = 1.0 if hit.context is None else hit.context.confidence
    marks = (
        f"confidence {confidence:.2f}{_basis_and_evidence_marks(hit)}"
        f"{_affect_marks(hit)}{_affect_link_marks(hit)}{_validity(hit)}{_affect_hop(hit)}"
    )
    # One hit is one line, so stored newlines collapse rather than break the section shape.
    return context_prompts.hit_line(hit.id, _render_source_content(hit), marks)


def _render_source_content(hit: SearchHit) -> str:
    first_asset = next(
        (position for position, part in enumerate(hit.parts) if part.kind == "asset"),
        len(hit.parts),
    )
    if not any(part.kind != "asset" for part in hit.parts[first_asset:]):
        return " ".join(hit.content.split())
    pieces: list[str] = []
    offset = 0
    for part in hit.parts:
        if part.kind == "asset":
            pieces.append(f"[asset:{part.value}]")
            continue
        position = hit.content.find(part.value, offset)
        if position >= 0:
            pieces.append(hit.content[offset:position])
            offset = position + len(part.value)
        pieces.append(part.value)
    pieces.append(hit.content[offset:])
    return " ".join(" ".join(pieces).split())


def _basis_and_evidence_marks(hit: SearchHit) -> str:
    """Render typed provenance without claiming that a source entails its assertion."""
    context = hit.context
    if context is None or not context.evidence_ids:
        return ""
    marks = "" if isinstance(hit, AffectCue) else f"; basis {context.basis.value}"
    if context.evidence_ids and not isinstance(hit, AffectCue):
        marks += f"; evidence {_id_list(context.evidence_ids)}"
    return marks


def _affect_marks(hit: SearchHit) -> str:
    """Render an affect cue's provenance inline, so an inference never reads as a statement."""
    context = hit.context
    if not isinstance(hit, AffectCue) or context is None:
        return ""
    marks = f"; basis {context.basis.value}"
    if context.cue_modality is not None:
        marks += f"; cue {context.cue_modality.value}"
    if context.valence is not None:
        marks += f"; valence {context.valence:.2f}"
    if context.arousal is not None:
        marks += f"; arousal {context.arousal:.2f}"
    return marks


def _affect_link_marks(hit: SearchHit, symbols: Sequence[ContextSymbol] | None = None) -> str:
    context = hit.context
    if context is None:
        return ""

    def label(identifier: str, namespace: ContextSymbolNamespace) -> str:
        return identifier if symbols is None else _symbol(identifier, namespace, symbols)

    marks = ""
    link = context.affect_link
    if link is not None:
        focus = label(link.affect_id, ContextSymbolNamespace.MEMORY)
        object_ref = link.object_ref
        object_id = object_ref.memory_id or object_ref.identity_id
        assert object_id is not None
        namespace = (
            ContextSymbolNamespace.MEMORY
            if object_ref.memory_id is not None
            else ContextSymbolNamespace.IDENTITY
        )
        target = label(object_id, namespace)
        marks += f"; affect [{focus}] {link.role} {link.polarity} [{target}]"
    for witness in context.affect_witnesses[:_MAX_CUE_IDS]:
        source = label(witness.source_id, ContextSymbolNamespace.MEMORY)
        quote = " ".join(witness.quote.split())
        reporter = " ".join(witness.reporter.split())
        marks += f'; report by {reporter} from [{source}]: "{quote}"'
    if len(context.affect_witnesses) > _MAX_CUE_IDS:
        marks += f"; +{len(context.affect_witnesses) - _MAX_CUE_IDS} reports"
    return marks


def _affect_hop(hit: SearchHit) -> str:
    """Render the cue's own evidence and the events it co-occurs with, as IDs only."""
    if not isinstance(hit, AffectCue):
        return ""
    marks = ""
    evidence_ids = () if hit.context is None else hit.context.evidence_ids
    if evidence_ids:
        marks += f"; from {_id_list(evidence_ids)}"
    if hit.event_ids:
        # Named for what the edge is: both records were formed from one observation. It is not
        # a claim that the event caused the feeling.
        marks += f"; co-occurring events {_id_list(hit.event_ids)}"
    return marks


def _id_list(memory_ids: Sequence[str]) -> str:
    """Render an ID list, truncated, so one line's marks stay bounded by a constant.

    A capture with a hundred observations must not turn one memory into a paragraph the budget
    never charged for. The count of what was dropped is kept, so a truncated line still says
    that more exists.
    """
    shown = ", ".join(f"[{memory_id}]" for memory_id in memory_ids[:_MAX_CUE_IDS])
    remaining = len(memory_ids) - _MAX_CUE_IDS
    return shown if remaining <= 0 else f"{shown}, +{remaining} more"


def _validity(hit: SearchHit) -> str:
    """Render a typed memory's world-validity bounds inline, so a stale fact reads as stale.

    RecordSemantics refuses an end without a start, so the only open form is an open-ended one.
    """
    context = hit.context
    if context is None or context.valid_from is None:
        return ""
    start = context.valid_from.isoformat()
    if context.valid_until is None:
        return f"; valid from {start}"
    return f"; valid {start} → {context.valid_until.isoformat()}"


def _conflict_line(conflict: ContextConflict, included: Container[str]) -> str:
    label = " ".join(part for part in (conflict.subject, conflict.predicate) if part)
    return context_prompts.conflict_line(
        label or conflict.lineage_id,
        tuple(
            (value, memory_id, memory_id in included)
            for value, memory_id in zip(conflict.values, conflict.memory_ids, strict=True)
        ),
    )
