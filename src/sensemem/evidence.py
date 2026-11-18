"""Compact, source-qualified evidence for answer generation.

Labels identify records within one answer context, not independent witnesses.
Counts and overlaps describe omitted supporting records without serializing an
unbounded provenance graph into the model's answer context.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias, cast
from unicodedata import normalize

from sensemem.records import SearchHit
from sensemem.semantics import RecordSemantics
from sensemem.types import EvidenceBasis, RecordKind

_MAX_EXACT_OMITTED_SOURCES = 64
AffectLinkKey: TypeAlias = tuple[str, str, str | None, str | None]


def evidence_aliases(hits: Sequence[SearchHit]) -> dict[str, str]:
    """Assign one answer-local E label to each record in first-occurrence order."""
    aliases: dict[str, str] = {}
    for hit in hits:
        if hit.id not in aliases:
            aliases[hit.id] = f"E{len(aliases) + 1}"
    return aliases


def evidence_identity_ids(content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the speaker and face identity IDs one memory's stored evidence names.

    Speech and face evidence is stored as a labelled JSON section of the memory's own content,
    so the identifiers a naming claim may cite are already in what the adapter is shown; there
    is nothing to plumb through and nothing to invent.
    """
    speakers: dict[str, None] = {}
    faces: dict[str, None] = {}
    for section in content.split("\n\n"):
        marker, separator, body = section.partition("\n")
        if not separator:
            continue
        if marker.startswith("[speech identities:"):
            found, group, field = speakers, "segments", "speaker_id"
        elif marker.startswith("[face identities:"):
            found, group, field = faces, "identities", "identity_id"
        else:
            continue
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, RecursionError):
            continue
        items = parsed.get(group) if isinstance(parsed, dict) else None
        for item in items if isinstance(items, list) else ():
            value = item.get(field) if isinstance(item, dict) else None
            if isinstance(value, str) and value.strip():
                found[value] = None
    return tuple(speakers), tuple(faces)


def functional_lineage(context: RecordSemantics | None) -> tuple[str, str] | None:
    """Return the claim key whose competing values make delivery non-affirmative.

    This matches the compiler and slow-loop contract: relations and inferred traits can
    accumulate values, while STATE and a user-stated TRAIT have one functional value at a scoped
    point in time.
    """
    if (
        context is None
        or context.lineage_id is None
        or context.value is None
        or not (
            context.kind is RecordKind.STATE
            or (context.kind is RecordKind.TRAIT and context.basis is EvidenceBasis.USER_STATEMENT)
        )
    ):
        return None
    return context.lineage_id, context.value


def affect_link_key(hit: SearchHit) -> AffectLinkKey | None:
    """Identify the attribution whose affirmations and denials must travel together."""
    link = None if hit.context is None else hit.context.affect_link
    if link is None:
        return None
    return link.affect_id, link.role, link.object_ref.memory_id, link.object_ref.identity_id


def retain_source_complete_hits(
    hits: Sequence[SearchHit], sources: Sequence[SearchHit]
) -> tuple[SearchHit, ...]:
    """Withhold assertions whose supplied sources a transport packing pass removed.

    A source's text may survive media omission, but it no longer supplies its full evidence.
    Propagate that loss through dependent assertions without dropping usable raw records.
    References already outside the supplied pool remain the adapter's explicit S labels.
    """
    original = {hit.id: hit for hit in sources}
    unavailable = original.keys() - {hit.id for hit in hits}
    for hit in hits:
        source = original.get(hit.id)
        if source is not None and {asset.id for asset in source.assets} - {
            asset.id for asset in hit.assets
        }:
            unavailable.add(hit.id)
    withheld: set[str] = set()
    while True:
        unavailable_attributions = {
            key
            for memory_id, hit in original.items()
            if memory_id in unavailable and (key := affect_link_key(hit)) is not None
        }
        rejected = {
            hit.id
            for hit in hits
            if (
                hit.context is not None
                and any(source_id in unavailable for source_id in hit.context.evidence_ids)
            )
            or affect_link_key(hit) in unavailable_attributions
        } - withheld
        if not rejected:
            break
        withheld.update(rejected)
        unavailable.update(rejected)
    return tuple(hit for hit in hits if hit.id not in withheld)


def answer_constraint_payloads(
    hits: Sequence[SearchHit], payloads: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Source-qualified candidates for applying state, trait and host policy evidence.

    This does not interpret prose, infer preferences, or authorize instructions. Conditions
    and exceptions stay in the referenced evidence text. Values with overlapping validity
    are flagged as competing candidates; recording order never selects a winner.
    """
    if len(hits) != len(payloads):
        raise ValueError("constraint payloads must align with selected evidence")
    selected: list[tuple[RecordSemantics, dict[str, object]]] = []
    for hit, payload in zip(hits, payloads, strict=True):
        context = hit.context
        if (
            context is None
            or context.kind not in {RecordKind.STATE, RecordKind.TRAIT, RecordKind.RESPONSE_POLICY}
            or context.subject is None
            or context.predicate is None
        ):
            continue
        candidate = {
            "evidence_label": payload["evidence_label"],
            **_context_payload(context),
            "current_version": context.visible and context.retired_at is None,
            "conditions_and_exceptions": "read the referenced evidence text",
        }
        selected.append((context, candidate))
    by_key: dict[tuple[str, str], list[tuple[RecordSemantics, dict[str, object]]]] = {}
    for context, candidate in selected:
        key = tuple(
            normalize("NFKC", value).strip().casefold()
            for value in (
                context.subject or "",
                context.predicate or "",
            )
        )
        group = by_key.setdefault((key[0], key[1]), [])
        for other, previous in group:
            if (
                context.value is None
                or other.value is None
                or context.value == other.value
                or not candidate["current_version"]
                or not previous["current_version"]
                or (
                    context.valid_until is not None
                    and other.valid_from is not None
                    and context.valid_until <= other.valid_from
                )
                or (
                    other.valid_until is not None
                    and context.valid_from is not None
                    and other.valid_until <= context.valid_from
                )
            ):
                continue
            cast(list[object], candidate.setdefault("competing_evidence_labels", [])).append(
                previous["evidence_label"]
            )
            cast(list[object], previous.setdefault("competing_evidence_labels", [])).append(
                candidate["evidence_label"]
            )
        group.append((context, candidate))
    return [candidate for _context, candidate in selected]


def answer_evidence_payloads(
    hits: Sequence[SearchHit],
    *,
    omitted_media: Mapping[str, Mapping[str, int]] | None = None,
) -> list[dict[str, object]]:
    """Project selected records without retrieving or inventing source content.

    Selection order is preserved. Event time, validity and recording time retain
    their distinct meanings; neither record order nor a missing source label
    establishes chronology, corroboration or source visibility. The adapter may
    identify media excluded from the actual request; those counts describe missing
    input, not missing information in the record's retained text.
    """
    qualified = any(hit.context is not None for hit in hits)
    selected = evidence_aliases(hits)
    external_labels: dict[str, str] = {}

    def source_label(source_id: str) -> str:
        if source_id in selected:
            return selected[source_id]
        if source_id not in external_labels:
            external_labels[source_id] = f"S{len(external_labels) + 1}"
        return external_labels[source_id]

    omitted_sources = {
        hit.id: frozenset(sid for sid in hit.context.evidence_ids if sid not in selected)
        for hit in hits
        if hit.context is not None
    }
    exact_sources = _small_source_set(tuple(omitted_sources.values()))
    shared_sources = {} if exact_sources else _shared_sources(omitted_sources, selected)

    payloads: list[dict[str, object]] = []
    for hit in hits:
        payload = _record_payload(hit, selected[hit.id] if qualified else None)
        if omitted_media is not None and (media_counts := omitted_media.get(hit.id)):
            payload["media_omitted"] = dict(media_counts)
        if hit.context is not None:
            details = _context_payload(hit.context)
            details.update(
                _source_payload(
                    hit.context,
                    selected,
                    source_label,
                    exact_sources=exact_sources,
                    omitted_count=len(omitted_sources[hit.id]),
                    shared=shared_sources.get(hit.id, ()),
                )
            )
            details.update(_affect_payload(hit.context, selected, source_label))
            payload["context"] = details
        payloads.append(payload)
    return payloads


def _affect_payload(
    context: RecordSemantics,
    selected: Mapping[str, str],
    source_label: Callable[[str], str],
) -> dict[str, object]:
    """Alias attribution endpoints and reporting witnesses without claiming omitted content."""
    details: dict[str, object] = {}
    link = context.affect_link
    if link is not None:
        object_ref: dict[str, object] = {}
        if link.object_ref.memory_id is not None:
            object_ref["memory_label"] = source_label(link.object_ref.memory_id)
            if link.object_ref.memory_id not in selected:
                object_ref["not_in_context"] = True
        else:
            object_ref["identity_id"] = link.object_ref.identity_id
        details["affect_link"] = {
            "affect_label": source_label(link.affect_id),
            "role": link.role,
            "object_ref": object_ref,
            "polarity": link.polarity,
            **({"affect_not_in_context": True} if link.affect_id not in selected else {}),
        }
    if context.affect_witnesses:
        details["affect_witnesses"] = [
            {
                "source_label": source_label(witness.source_id),
                "reporter": witness.reporter,
                "quote": witness.quote,
                **({"source_not_in_context": True} if witness.source_id not in selected else {}),
            }
            for witness in context.affect_witnesses
        ]
    return details


def _record_payload(hit: SearchHit, label: str | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "content": hit.content,
        "memory_type": hit.memory_type.value,
    }
    if hit.modality.value != "text":
        payload["source_modality"] = hit.modality.value
    if hit.occurred_at is not None:
        payload["occurred_at"] = hit.occurred_at.isoformat()
    else:
        payload["created_at"] = hit.created_at.isoformat()
    if hit.occurred_end is not None:
        payload["occurred_end"] = hit.occurred_end.isoformat()
    payload["metadata"] = dict(hit.metadata)
    if label is not None:
        payload["evidence_label"] = label
    if hit.place_id is not None:
        payload["place_id"] = hit.place_id
    return payload


def _context_payload(context: RecordSemantics) -> dict[str, object]:
    details: dict[str, object] = {
        "kind": context.kind.value,
        "basis": context.basis.value,
        "confidence": context.confidence,
        "recorded_at": context.recorded_at.isoformat(),
    }
    for key, time_value in (
        ("valid_from", context.valid_from),
        ("valid_until", context.valid_until),
    ):
        if time_value is not None:
            details[key] = time_value.isoformat()
    for key, value in (
        ("subject", context.subject),
        ("predicate", context.predicate),
        ("value", context.value),
    ):
        if value is not None:
            details[key] = value
    if context.cue_modality is not None:
        details["cue_modality"] = context.cue_modality.value
    if context.valence is not None:
        details["valence"] = context.valence
    if context.arousal is not None:
        details["arousal"] = context.arousal
    return details


def _source_payload(
    context: RecordSemantics,
    selected: Mapping[str, str],
    source_label: Callable[[str], str],
    *,
    exact_sources: bool,
    omitted_count: int,
    shared: Sequence[dict[str, object]],
) -> dict[str, object]:
    # A source_id describes provenance, whereas supersedes_id describes a
    # correction. Do not combine correction edges with supporting ones.
    details: dict[str, object] = {}
    details.update(
        _support_payload(
            context.evidence_ids,
            selected,
            source_label,
            exact_sources=exact_sources,
            omitted_count=omitted_count,
            shared=shared,
        )
    )
    if context.source_id is not None:
        details["source_label"] = source_label(context.source_id)
        if context.source_id not in selected:
            details["source_not_in_context"] = True
    if context.supersedes_id is not None:
        details["supersedes_label"] = source_label(context.supersedes_id)
    return details


def _support_payload(
    evidence_ids: Sequence[str],
    selected: Mapping[str, str],
    source_label: Callable[[str], str],
    *,
    exact_sources: bool,
    omitted_count: int,
    shared: Sequence[dict[str, object]],
) -> dict[str, object]:
    if exact_sources:
        source_ids = tuple(dict.fromkeys(evidence_ids))
        if not source_ids:
            return {}
        details: dict[str, object] = {
            "evidence_labels": [source_label(sid) for sid in source_ids],
            "supporting_record_count": len(source_ids),
        }
        absent = [source_label(sid) for sid in source_ids if sid not in selected]
        if absent:
            details["sources_not_in_context"] = absent
        return details
    details = {}
    visible_sources = tuple(dict.fromkeys(sid for sid in evidence_ids if sid in selected))
    if visible_sources or omitted_count:
        details["supporting_record_count"] = len(visible_sources) + omitted_count
    if visible_sources:
        details["evidence_labels"] = [selected[sid] for sid in visible_sources]
    if omitted_count:
        details["sources_not_in_context_count"] = omitted_count
    if shared:
        details["shared_sources_not_in_context"] = list(shared)
    return details


def _small_source_set(source_sets: Sequence[frozenset[str]]) -> bool:
    seen: set[str] = set()
    for source_ids in source_sets:
        for source_id in source_ids:
            seen.add(source_id)
            if len(seen) > _MAX_EXACT_OMITTED_SOURCES:
                return False
    return True


def _shared_sources(
    omitted: Mapping[str, frozenset[str]],
    selected: Mapping[str, str],
) -> dict[str, list[dict[str, object]]]:
    """Bound the rendered provenance by selected records, not lifetime fanout.

    Pairwise overlap does not determine higher-order unions or prove independent
    observations. Computation still depends on the sizes of the stored source sets.
    """
    shared: dict[str, list[dict[str, object]]] = {}
    items = tuple(omitted.items())
    for position, (memory_id, source_ids) in enumerate(items):
        if not source_ids:
            continue
        for other_id, other_sources in items[position + 1 :]:
            overlap = len(source_ids & other_sources)
            if overlap:
                shared.setdefault(memory_id, []).append(
                    {"evidence_label": selected[other_id], "count": overlap}
                )
                shared.setdefault(other_id, []).append(
                    {"evidence_label": selected[memory_id], "count": overlap}
                )
    return shared
