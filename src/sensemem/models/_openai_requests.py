"""Pure request serialization, media fitting and tool schemas for OpenAI calls."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import chain
from typing import Literal, cast

from opentelemetry import trace

from sensemem.content import AssetRef
from sensemem.control import dump_query_failure
from sensemem.control_values import MemoryTrigger, QueryFailure
from sensemem.evidence import (
    answer_constraint_payloads,
    answer_evidence_payloads,
    evidence_aliases,
    evidence_identity_ids,
    retain_source_complete_hits,
)
from sensemem.exceptions import ModelError, ValidationError
from sensemem.models import _openai_media as media
from sensemem.models._evidence_operations import (
    EVIDENCE_OPERATION_PARAMETERS,
    EVIDENCE_OPERATION_TOOL_NAME,
    EvidenceCatalogue,
    make_evidence_catalogue,
)
from sensemem.models._openai_protocol import (
    _json_text,
    _require_capabilities,
    output_limit_parameter,
)
from sensemem.models._openai_schemas import SchemaName, response_schema
from sensemem.models._openai_telemetry import _record_grounding_fit, _record_grounding_request
from sensemem.models.base import EventObservation, FormationInput, ModelInput
from sensemem.prompts import evidence_operations as evidence_prompts
from sensemem.prompts import generation as prompts
from sensemem.records import MemoryRecord, SearchHit
from sensemem.types import AbstentionReason, AnswerPolicy, Modality

_MAX_GROUNDED_TEXT_BYTES = 4 * 1024 * 1024


def _formation_response_schema(count: int) -> dict[str, object]:
    """Constrain batch coverage without requiring the model to propose any new fact."""
    schema = response_schema("formation")
    properties = cast(dict[str, object], schema["properties"])
    items = cast(dict[str, object], properties["items"])
    items["minItems"] = items["maxItems"] = count
    item = cast(dict[str, object], items["items"])
    fields = cast(dict[str, object], item["properties"])
    fields["observation_id"] = {
        "type": "string",
        "enum": [f"observation_{index}" for index in range(count)],
    }
    return schema


def _transcription_request(
    model: str,
    file: object,
    *,
    prompt: str | None,
    keywords: Sequence[str],
    languages: Sequence[str],
) -> dict[str, object]:
    request: dict[str, object] = {"model": model, "file": file}
    if prompt is not None:
        request["prompt"] = prompt
    context = {}
    if keywords:
        context["keywords"] = list(keywords)
    if languages:
        context["languages"] = list(languages)
    if context:
        request["extra_body"] = context
    return request


def _event_content(
    observations: Sequence[EventObservation],
    *,
    end_of_stream: bool,
    image_max_pixels: int | None,
    audio_data_url: bool,
) -> list[dict[str, object]]:
    assets = tuple(asset for value in observations for asset in value.content.assets)
    media._require_consistent_assets(assets)
    verified_assets = media._require_asset_integrity(assets)
    image_parts = media._image_transport_parts(assets, image_max_pixels)
    media._require_verified_inline_size(
        tuple(verified for verified in verified_assets if verified[0].id not in image_parts)
    )
    parts: list[dict[str, object]] = [
        {"type": "text", "text": _json_text({"end_of_stream": end_of_stream})}
    ]
    cache: dict[str, str] = {}
    media_position = 0
    for position, value in enumerate(observations):
        media_aliases = tuple(
            f"media_{index}"
            for index in range(media_position, media_position + len(value.content.assets))
        )
        media_position += len(media_aliases)
        parts.append(
            {
                "type": "text",
                "text": _json_text(
                    {
                        "chunk_index": position,
                        "occurred_at": value.occurred_at.isoformat(),
                        "occurred_end": value.occurred_end.isoformat(),
                        "content": value.content.text,
                        "media_count": len(value.content.assets),
                        "media_order": list(media_aliases),
                        **_ordered_content_payload(
                            value.content,
                            dict(
                                zip(
                                    (asset.id for asset in value.content.assets),
                                    media_aliases,
                                    strict=True,
                                )
                            ),
                        ),
                    }
                ),
            }
        )
        for asset in value.content.assets:
            if asset.id in image_parts:
                parts.extend(image_parts[asset.id])
            else:
                parts.append(
                    media._generation_asset_part(asset, cache, audio_data_url=audio_data_url)
                )
    media._require_inline_media_parts(parts)
    return parts


def _formation_content(
    inputs: Sequence[FormationInput],
    *,
    image_parts: Mapping[str, tuple[dict[str, object], ...]] | None = None,
    audio_data_url: bool = False,
) -> str | list[dict[str, object]]:
    payloads = []
    shared = _formation_history_groups(inputs)
    aliases = _formation_aliases(inputs, shared=shared)
    groups = {
        f"history_group_{position}": [
            _formation_history_payload(record, aliases[position][record.id], aliases[position])
            for record in inputs[position].history
        ]
        for position, group in enumerate(shared)
        if group == position
    }
    media_position = 0
    for position, value in enumerate(inputs):
        media_aliases = tuple(
            f"media_{index}"
            for index in range(media_position, media_position + len(value.content.assets))
        )
        media_position += len(media_aliases)
        payload = _formation_input_payload(
            value,
            observation_id=f"observation_{position}",
            media_aliases=media_aliases,
        )
        if value.history or value.history_truncated:
            group = shared[position]
            if group is None:
                payload["read_only_history"] = [
                    _formation_history_payload(
                        record, aliases[position][record.id], aliases[position]
                    )
                    for record in value.history
                ]
            else:
                payload["read_only_history_group"] = f"history_group_{group}"
            payload["history_truncated"] = value.history_truncated
        payloads.append(payload)
    if not any(value.content.assets for value in inputs):
        request: dict[str, object] = {"observations": payloads}
        if groups:
            request["read_only_history_groups"] = groups
        return _json_text(request)
    parts: list[dict[str, object]] = []
    if groups:
        parts.append({"type": "text", "text": _json_text({"read_only_history_groups": groups})})
    cache: dict[str, str] = {}
    for value, payload in zip(inputs, payloads, strict=True):
        parts.append({"type": "text", "text": _json_text({"observation": payload})})
        for asset in value.content.assets:
            if image_parts is not None and asset.id in image_parts:
                parts.extend(image_parts[asset.id])
            else:
                parts.append(
                    media._generation_asset_part(asset, cache, audio_data_url=audio_data_url)
                )
    return parts


def _formation_history_groups(inputs: Sequence[FormationInput]) -> tuple[int | None, ...]:
    observations = {value.memory_id: f"observation_{index}" for index, value in enumerate(inputs)}
    representatives: list[int] = []
    projections: dict[int, str] = {}
    matches: list[int | None] = []
    for position, value in enumerate(inputs):
        if not value.history:
            matches.append(None)
            continue
        local = {
            memory_id: observations.get(memory_id, f"history_{index}")
            for index, memory_id in enumerate(sorted(record.id for record in value.history))
        }
        projection = _json_text(
            [
                _formation_history_payload(record, local[record.id], local)
                for record in value.history
            ]
        )
        # Equality covers hidden provenance; the wire comparison also preserves ISO spellings
        # that Python's datetime equality treats as the same instant.
        representative = next(
            (
                other
                for other in representatives
                if value.history == inputs[other].history
                and value.history_truncated == inputs[other].history_truncated
                and projection == projections[other]
            ),
            position,
        )
        if representative == position:
            representatives.append(position)
            projections[position] = projection
        matches.append(representative)
    counts = Counter(matches)
    return tuple(match if match is not None and counts[match] > 1 else None for match in matches)


def _formation_aliases(
    inputs: Sequence[FormationInput], *, shared: Sequence[int | None] | None = None
) -> tuple[dict[str, str], ...]:
    if shared is None:
        shared = _formation_history_groups(inputs)
    observations = {value.memory_id: f"observation_{index}" for index, value in enumerate(inputs)}
    result: list[dict[str, str]] = []
    history_position = 0
    for position, value in enumerate(inputs):
        aliases: dict[str, str] = {}
        histories = {record.id: record for record in value.history}
        for memory_id in sorted(histories):
            if memory_id in observations:
                aliases[memory_id] = observations[memory_id]
            else:
                aliases[memory_id] = f"history_{history_position}"
                history_position += 1
        representative = shared[position]
        if representative is not None and representative != position:
            # Reserve the old private range so later unshared histories keep their aliases.
            aliases = result[representative]
        result.append(aliases)
    return tuple(result)


def _formation_history_payload(
    record: MemoryRecord, alias: str, aliases: Mapping[str, str]
) -> dict[str, object]:
    context = record.context
    return {
        "observation_id": alias,
        "content": record.content,
        **_ordered_content_payload(record, {}),
        "recorded_at": record.created_at.isoformat(),
        "occurred_at": None if record.occurred_at is None else record.occurred_at.isoformat(),
        "active": record.forgotten_at is None
        and (context is None or (context.visible and context.retired_at is None)),
        "context": None
        if context is None
        else {
            "kind": context.kind.value,
            "basis": context.basis.value,
            "identity_id": context.identity_id,
            "subject": context.subject,
            "predicate": context.predicate,
            "value": context.value,
            "confidence": context.confidence,
            "recorded_at": context.recorded_at.isoformat(),
            "visible": context.visible,
            "retired": context.retired_at is not None,
            "valid_from": None if context.valid_from is None else context.valid_from.isoformat(),
            "valid_until": None if context.valid_until is None else context.valid_until.isoformat(),
            "has_correction": context.supersedes_id is not None,
            "supersedes_observation_id": aliases.get(context.supersedes_id or ""),
            "support_observation_ids": [
                aliases[item] for item in context.evidence_ids if item in aliases
            ],
            "support_not_in_history_count": sum(
                item not in aliases for item in context.evidence_ids
            ),
            "affect_link": None
            if context.affect_link is None
            else {
                "affect_id": aliases.get(context.affect_link.affect_id),
                "role": context.affect_link.role,
                "object_ref": {
                    "memory_id": aliases.get(context.affect_link.object_ref.memory_id or ""),
                    "identity_id": context.affect_link.object_ref.identity_id,
                },
                "polarity": context.affect_link.polarity,
            },
            "affect_witnesses": [
                {
                    "source_id": aliases[witness.source_id],
                    "reporter": witness.reporter,
                    "quote": witness.quote,
                }
                for witness in context.affect_witnesses
                if witness.source_id in aliases
            ],
            "affect_witnesses_not_in_history_count": sum(
                witness.source_id not in aliases for witness in context.affect_witnesses
            ),
        },
    }


def _ordered_content_payload(
    value: ModelInput | MemoryRecord | SearchHit, labels: Mapping[str, str]
) -> dict[str, object]:
    first_asset = next(
        (position for position, part in enumerate(value.parts) if part.kind == "asset"),
        len(value.parts),
    )
    if not any(part.kind != "asset" for part in value.parts[first_asset:]) and not any(
        part.kind not in {"text", "asset"} for part in value.parts
    ):
        return {}
    parts = [
        (
            {"kind": "asset", "media": labels[part.value]}
            if part.value in labels
            else {"kind": "asset", "available": False}
        )
        if part.kind == "asset"
        else {"kind": part.kind, "text": part.value}
        for part in value.parts
    ]
    return {"content_parts": parts}


def _formation_input_payload(
    value: FormationInput,
    *,
    observation_id: str,
    media_aliases: Sequence[str],
) -> dict[str, object]:
    context = value.context
    spatial = context.spatial
    return {
        "observation_id": observation_id,
        "content": value.content.text,
        **_ordered_content_payload(
            value.content,
            dict(zip((asset.id for asset in value.content.assets), media_aliases, strict=True)),
        ),
        "media_order": list(media_aliases),
        "occurred_at": None if value.occurred_at is None else value.occurred_at.isoformat(),
        "occurred_end": None if value.occurred_end is None else value.occurred_end.isoformat(),
        "created_at": None if value.created_at is None else value.created_at.isoformat(),
        "context": {
            "basis": context.basis.value,
            "confidence": context.confidence,
            "valid_from": (None if context.valid_from is None else context.valid_from.isoformat()),
            "valid_until": (
                None if context.valid_until is None else context.valid_until.isoformat()
            ),
            # Formation may reuse the source coordinate vocabulary, but the provider does not need
            # the observation's metric pose to know which frame an inferred spatial value belongs
            # to. Keep those coordinates local.
            "spatial": (
                None
                if spatial is None
                else {"frame_id": spatial.frame_id, "anchor": spatial.anchor.value}
            ),
        },
    }


def _consolidation_content(
    evidence: Sequence[MemoryRecord],
    *,
    trigger: MemoryTrigger,
    capabilities: frozenset[Modality],
    audio_data_url: bool = False,
    query_failure: QueryFailure | None = None,
) -> str | list[dict[str, object]]:
    """Number the evidence, and attach the media the generation model declares it can read."""
    attached = tuple(
        tuple(asset for asset in record.assets if asset.modality in capabilities)
        for record in evidence
    )
    payloads = []
    evidence_indices = {record.id: index for index, record in enumerate(evidence)}
    media_position = 0
    for index, (record, assets) in enumerate(zip(evidence, attached, strict=True)):
        aliases = tuple(
            f"media_{position}" for position in range(media_position, media_position + len(assets))
        )
        media_position += len(aliases)
        payloads.append(
            _consolidation_evidence_payload(
                record,
                index,
                evidence_indices=evidence_indices,
                media_aliases=aliases,
                media_labels=dict(zip((asset.id for asset in assets), aliases, strict=True)),
            )
        )
    if not any(attached):
        return _json_text(
            {
                "trigger": trigger.value,
                "evidence": payloads,
                "query_failure": None
                if query_failure is None
                else json.loads(dump_query_failure(query_failure)),
            }
        )
    # One part is emitted per attachment below, so the same asset cited by two evidence items
    # is carried twice. Size the request the way `form()` does, by emitted part, not by asset ID.
    carried = tuple(chain.from_iterable(attached))
    media._require_consistent_assets(carried)
    media._require_inline_size(carried)
    parts: list[dict[str, object]] = [
        {
            "type": "text",
            "text": _json_text(
                {
                    "trigger": trigger.value,
                    "query_failure": None
                    if query_failure is None
                    else json.loads(dump_query_failure(query_failure)),
                }
            ),
        }
    ]
    cache: dict[str, str] = {}
    for payload, assets in zip(payloads, attached, strict=True):
        parts.append({"type": "text", "text": _json_text({"evidence": payload})})
        parts.extend(
            media._generation_asset_part(asset, cache, audio_data_url=audio_data_url)
            for asset in assets
        )
    return parts


def _consolidation_evidence_payload(
    record: MemoryRecord,
    index: int,
    *,
    evidence_indices: Mapping[str, int],
    media_aliases: Sequence[str] = (),
    media_labels: Mapping[str, str] | None = None,
) -> dict[str, object]:
    context = record.context
    speakers, faces = evidence_identity_ids(record.content)
    payload: dict[str, object] = {
        "index": index,
        "memory_id": record.id,
        "content": record.content,
        **_ordered_content_payload(record, {} if media_labels is None else media_labels),
        "memory_type": record.memory_type.value,
        "modality": record.modality.value,
        "media_order": list(media_aliases),
        "occurred_at": (None if record.occurred_at is None else record.occurred_at.isoformat()),
        "created_at": record.created_at.isoformat(),
        "target_only": context is not None
        and (not context.visible or context.retired_at is not None),
        # Who this evidence contains. An identify claim may only name one of these, so the
        # model needs to see them rather than infer an identifier it cannot know.
        "speaker_ids": list(speakers),
        "face_identity_ids": list(faces),
    }
    if context is not None:
        payload["context"] = {
            "kind": context.kind.value,
            "basis": context.basis.value,
            "confidence": context.confidence,
            "visible": context.visible,
            "retired_at": None if context.retired_at is None else context.retired_at.isoformat(),
            "subject": context.subject,
            "predicate": context.predicate,
            "value": context.value,
            "valid_from": (None if context.valid_from is None else context.valid_from.isoformat()),
            "valid_until": (
                None if context.valid_until is None else context.valid_until.isoformat()
            ),
            "source_count": len(context.evidence_ids),
            "evidence_indices": [
                evidence_indices[source]
                for source in context.evidence_ids
                if source in evidence_indices
            ],
            "sources_not_shown_count": sum(
                source not in evidence_indices for source in context.evidence_ids
            ),
        }
    return payload


def _vision_content(
    inputs: Sequence[ModelInput],
    *,
    video_frames: int = 4,
    image_parts: Mapping[str, tuple[dict[str, object], ...]] | None = None,
    capture_metadata: Mapping[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Number each visual, inline it, and follow it with whatever context arrived as text.

    A video is replaced by its ordered stills, so one visual can carry several image parts, and
    the count has to be in the marker rather than left to the reader. Asked for "one description
    per numbered visual" over a single clip's four unlabelled stills, a measured endpoint returned
    four descriptions on every attempt, which the output contract rejected whole -- the request was
    billed and the caption was lost. Naming the count is what makes one caption per clip the
    obvious reading.

    `ModelInput.text` is the visual's own already-derived context -- in practice the clip's
    diarised transcript, which the write path computed before describing. It is what a durable
    fact about a person can be read from at all: four stills say two people are at a table, and
    only the words say which of them is called Lily and which is allergic to peanuts. It follows
    the stills so that the labelled description is still written from pixels, and it is labelled
    as a transcript so the caption never quotes it as visible text.
    """
    parts: list[dict[str, object]] = []
    cache: dict[str, str] = {}
    capture = media._source_capture_metadata(
        tuple(asset for value in inputs for asset in value.assets), cached=capture_metadata
    )
    for position, value in enumerate(inputs, start=1):
        groups = tuple(
            (
                image_parts[asset.id]
                if image_parts is not None and asset.id in image_parts
                else media._vision_asset_parts(asset, cache, video_frames=video_frames)
            )
            for asset in value.assets
        )
        visual_count = sum(len(group) for group in groups)
        marker = (
            f"Visual {position}:"
            if visual_count <= 1
            else f"Visual {position}, as {visual_count} ordered stills:"
        )
        parts.append({"type": "text", "text": marker})
        for index, (asset, group) in enumerate(zip(value.assets, groups, strict=True), start=1):
            if asset.id in capture:
                parts.append(
                    {
                        "type": "text",
                        "text": _json_text(
                            {
                                "untrusted_source_metadata": {
                                    "attachment": f"Visual {position} attachment {index}",
                                    **capture[asset.id],
                                }
                            }
                        ),
                    }
                )
            parts.extend(group)
        if value.text:
            parts.append(
                {
                    "type": "text",
                    "text": f"Visual {position} transcript, speaker labels as diarised:\n"
                    + value.text,
                }
            )
    return parts


def _json_retry_request(request: dict[str, object], instruction: str) -> dict[str, object]:
    """Append a correction to the existing user content without replaying rejected model text."""
    messages = cast(list[dict[str, object]], request["messages"])
    user = messages[-1]
    content = user["content"]
    corrected_content: object = (
        f"{content}\n\n{instruction}"
        if isinstance(content, str)
        else [*cast(list[dict[str, object]], content), {"type": "text", "text": instruction}]
    )
    corrected_user = {**user, "content": corrected_content}
    return {**request, "messages": [*messages[:-1], corrected_user]}


def _embedding_samples(
    inputs: Sequence[ModelInput],
    *,
    sample_video: bool = False,
    sampled: list[int] | None = None,
    sampled_assets: frozenset[str] = frozenset(),
    image_parts: Mapping[str, tuple[dict[str, object], ...]] | None = None,
) -> list[list[dict[str, object]]]:
    cache: dict[str, str] = {}
    samples: list[list[dict[str, object]]] = []
    for index, value in enumerate(inputs):
        parts, was_sampled = media._input_parts(
            value,
            cache,
            sample_video=sample_video,
            sampled_assets=sampled_assets,
            image_parts=image_parts,
        )
        if was_sampled and sampled is not None:
            sampled.append(index)
        samples.append([{"role": "user", "content": parts}])
    return samples


def _require_answer_text_size(parts: Iterable[str]) -> None:
    if sum(len(part.encode("utf-8")) for part in parts) > _MAX_GROUNDED_TEXT_BYTES:
        raise ModelError(
            "answer input text exceeds 4 MiB; reduce the question or grounding evidence",
            reason="payload_too_large",
            stage="generate",
        )


def _answer_wire_size(request: Mapping[str, object], *, stream: bool) -> int:
    """Match the official SDK's top-level extra_body merge and UTF-8 JSON encoding."""
    body = dict(request)
    extra = cast(Mapping[str, object], body.pop("extra_body", {}))
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    body.update(extra)
    return len(
        json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    )


def _evidence_operation_recovery(request: Mapping[str, object]) -> dict[str, object]:
    messages = cast(list[dict[str, object]], request["messages"])
    system = messages[0]
    return {
        **{
            key: value
            for key, value in request.items()
            if key not in {"tools", "tool_choice", "parallel_tool_calls"}
        },
        "messages": [
            {
                **system,
                "content": evidence_prompts.operation_system_prompt(
                    cast(str, system["content"]), recovery=True
                ),
            },
            *messages[1:],
        ],
    }


def _answer_parts(
    question: ModelInput,
    hits: Sequence[SearchHit],
    texts: Sequence[str],
    *,
    media_slack_bytes: int,
    minimum_video_seconds: float | None,
    sampled_parts: Mapping[str, tuple[dict[str, object], ...]] | None = None,
    audio_data_url: bool = False,
) -> list[dict[str, object]]:
    """Interleave the request text with its media, each asset announced by its label.

    The label text part sits immediately before the asset's own parts, so `attachment 3` is
    defined where the media is and not by counting pictures: a short video may arrive as several
    still frames, and without the marker the reader has no way to tell where one attachment ends
    and the next begins. The labels are derived here from the same question and hits that
    `_answer_text_parts` derives them from, by the same pure function, so the payloads and the
    markers agree by construction and no caller can hand either a mismatched mapping.
    """
    media_labels = media._media_labels(question, hits)
    cache: dict[str, str] = {}
    parts: list[dict[str, object]] = [{"type": "text", "text": texts[0]}]
    seen_assets: set[str] = set()

    def emit(asset: AssetRef) -> None:
        nonlocal media_slack_bytes
        if asset.id in seen_assets:
            return
        seen_assets.add(asset.id)
        original_size = media._encoded_size(cast(int, asset.size_bytes))
        sampled = None if sampled_parts is None else sampled_parts.get(asset.id)
        asset_parts = (
            sampled
            if sampled is not None
            else media._generation_asset_parts(
                asset,
                cache,
                minimum_video_seconds,
                original_size + media_slack_bytes,
                audio_data_url=audio_data_url,
            )
        )
        represented = (
            asset_parts
            if asset.modality is Modality.VIDEO and asset_parts[0]["type"] == "image_url"
            else sampled
        )
        label = media._transport_label(media_labels[asset.id], represented, asset.modality)
        parts.append({"type": "text", "text": label})
        parts.extend(asset_parts)
        if (
            sampled is None
            and asset.modality is Modality.VIDEO
            and asset_parts[0]["type"] == "image_url"
        ):
            media_slack_bytes += original_size - media._image_parts_size(asset_parts)

    for asset in question.assets:
        emit(asset)
    for hit, text in zip(hits, texts[1:], strict=True):
        parts.append({"type": "text", "text": text})
        for asset in hit.assets:
            emit(asset)
    return parts


def _answer_text_parts(
    question: ModelInput,
    hits: Sequence[SearchHit],
    evidence_payloads: Sequence[Mapping[str, object]],
    *,
    capture_metadata: Mapping[str, dict[str, object]] | None = None,
    source_hits: Sequence[SearchHit] | None = None,
) -> tuple[str, ...]:
    if len(evidence_payloads) != len(hits):
        raise ValueError("answer evidence payloads must align with hits")
    media_labels = media._media_labels(question, hits)
    assets = (*question.assets, *(asset for hit in hits for asset in hit.assets))
    capture = (
        media._source_capture_metadata(assets, cached=capture_metadata)
        if source_hits is None
        else dict(capture_metadata or {})
    )
    capture.update(question.source_metadata)
    original_assets = {} if source_hits is None else {hit.id: hit.assets for hit in source_hits}

    def source_metadata(assets: Sequence[AssetRef]) -> dict[str, object]:
        values = [
            {
                **(
                    {"attachment": media_labels[asset.id]}
                    if asset.id in media_labels
                    else {
                        "media_omitted": True,
                        "source_asset_id": asset.id,
                        "source_modality": cast(Modality, asset.modality).value,
                    }
                ),
                **capture[asset.id],
            }
            for asset in assets
            if asset.id in capture
        ]
        return {"untrusted_source_metadata": values} if values else {}

    constraints = answer_constraint_payloads(hits, evidence_payloads)
    image_matches: dict[str, list[dict[str, object]]] = {}
    by_id = {hit.id: hit for hit in hits}
    for match in question.image_retrieval_matches:
        hit = by_id.get(match.memory_id)
        if hit is None or not any(asset.id == match.source_asset_id for asset in hit.assets):
            continue
        image_matches.setdefault(hit.id, []).append(
            {
                "query_media": media_labels[match.query_asset_id],
                "source_media": media_labels[match.source_asset_id],
                "basis": "image_embedding_cosine",
                "rank": match.rank,
                "cosine_similarity": match.cosine_similarity,
            }
        )
    return (
        _json_text(
            {
                "question": question.text,
                **_ordered_content_payload(question, media_labels),
                "media": [media_labels[asset.id] for asset in question.assets],
                "media_modalities": {
                    media_labels[asset.id]: cast(Modality, asset.modality).value
                    for asset in question.assets
                },
                **source_metadata(question.assets),
                **({"constraint_candidates": constraints} if constraints else {}),
            }
        ),
        *(
            _json_text(
                {
                    "memory": {
                        **payload,
                        **_ordered_content_payload(hit, media_labels),
                        "media": [media_labels[asset.id] for asset in hit.assets],
                        "media_modalities": {
                            media_labels[asset.id]: cast(Modality, asset.modality).value
                            for asset in hit.assets
                        },
                        **source_metadata(original_assets.get(hit.id, hit.assets)),
                        **(
                            {"image_retrieval_matches": image_matches[hit.id]}
                            if hit.id in image_matches
                            else {}
                        ),
                    }
                }
            )
            for hit, payload in zip(hits, evidence_payloads, strict=True)
        ),
    )


@dataclass(frozen=True, slots=True)
class OpenAIRequestBuilder:
    """Serialize SenseMem inputs without making model calls or changing runtime policy."""

    model: str
    response_format: Literal["json_object", "json_schema"]
    capabilities: frozenset[Modality]
    seed: int | None
    temperature: float | None
    max_tokens: int | None
    extra_body: Mapping[str, object] | None
    store: bool
    video_limit: int | None
    min_video_seconds: float | None
    image_max_pixels: int | None
    request_max_bytes: int | None
    answer_style: Literal["default", "concise"]
    answer_format: Literal["text", "structured"]
    evidence_operation_tool: bool
    source_metadata_key: str

    def json_request(
        self,
        system_prompt: str,
        content: str | list[dict[str, object]],
        *,
        schema: SchemaName,
        formation_count: int | None = None,
    ) -> dict[str, object]:
        """Build one JSON-object chat completion under the configured generation controls."""
        request: dict[str, object] = {
            "model": self.model,
            "store": self.store,
            "messages": [
                {
                    "role": "system",
                    "content": prompts._json_system_prompt(
                        system_prompt, strict_schema=self.response_format == "json_schema"
                    ),
                },
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.response_format == "json_schema":
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema,
                    "strict": True,
                    "schema": (
                        _formation_response_schema(formation_count)
                        if schema == "formation" and formation_count is not None
                        else response_schema(schema)
                    ),
                },
            }
        if self.seed is not None:
            request["seed"] = self.seed
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request[output_limit_parameter(self.model)] = self.max_tokens
        if self.extra_body is not None:
            request["extra_body"] = dict(self.extra_body)
        return request

    def evidence_operation_fits(self, request: Mapping[str, object], *, stream: bool) -> bool:
        texts: list[str] = []
        for message in cast(list[dict[str, object]], request["messages"]):
            content = message["content"]
            if isinstance(content, str):
                texts.append(content)
            else:
                texts.extend(
                    cast(str, part["text"])
                    for part in cast(list[dict[str, object]], content)
                    if part["type"] == "text"
                )
        if "tools" in request:
            texts.append(_json_text(request["tools"]))
        try:
            _require_answer_text_size(texts)
        except ModelError as error:
            if error.reason != "payload_too_large":
                raise
            return False
        return self.request_max_bytes is None or (
            _answer_wire_size(request, stream=stream) <= self.request_max_bytes
        )

    def evidence_operation_request(
        self,
        prepared: tuple[dict[str, object], tuple[SearchHit, ...], frozenset[Modality]],
        *,
        stream: bool,
    ) -> tuple[dict[str, object], EvidenceCatalogue] | None:
        if not self.evidence_operation_tool:
            return None
        request, grounded, _modalities = prepared
        aliases = evidence_aliases(grounded)
        visible: dict[str, frozenset[str]] = {label: frozenset() for label in aliases.values()}
        messages = [dict(message) for message in cast(list[dict[str, object]], request["messages"])]
        messages[0]["content"] = evidence_prompts.operation_system_prompt(
            cast(str, messages[0]["content"])
        )
        content = messages[-1]["content"]
        if isinstance(content, str):
            payload = json.loads(content)
            for hit, memory in zip(grounded, payload["hits"], strict=True):
                memory["evidence_label"] = aliases[hit.id]
            messages[-1]["content"] = _json_text(payload)
        else:
            original = cast(list[dict[str, object]], content)
            supplied: set[str] = set()
            current: str | None = None
            for part in original:
                if part["type"] == "text":
                    marker = re.fullmatch(
                        r"(attachment [1-9][0-9]*)(?: \(.*\))?", cast(str, part["text"])
                    )
                    current = None if marker is None else marker[1]
                elif (
                    part["type"] in media._GENERATION_MODALITY_BY_PART_TYPE and current is not None
                ):
                    supplied.add(current)
            parts: list[dict[str, object]] = []
            index = 0
            for part in original:
                if part["type"] == "text" and cast(str, part["text"]).startswith('{"memory":'):
                    payload = json.loads(cast(str, part["text"]))
                    memory = payload["memory"]
                    label = aliases[grounded[index].id]
                    index += 1
                    memory["evidence_label"] = label
                    visible[label] |= frozenset(memory.get("media_modalities", {})).intersection(
                        supplied
                    )
                    part = {**part, "text": _json_text(payload)}
                parts.append(part)
            messages[-1]["content"] = parts
        candidate = {
            **request,
            "messages": messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": EVIDENCE_OPERATION_TOOL_NAME,
                        "description": evidence_prompts.EVIDENCE_OPERATION_TOOL_DESCRIPTION,
                        "parameters": EVIDENCE_OPERATION_PARAMETERS,
                    },
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        # This optional view may only use space left by the complete original fit. Reserve
        # the tool-free recovery notice without dropping or admitting any original media.
        if not self.evidence_operation_fits(
            candidate, stream=stream
        ) or not self.evidence_operation_fits(_evidence_operation_recovery(request), stream=stream):
            return None
        return candidate, make_evidence_catalogue(
            grounded,
            visible_media=visible,
            source_metadata_key=self.source_metadata_key,
        )

    def answer_request(  # noqa: C901 - preserve fitted evidence when the optional view exceeds bounds
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        omission_source_hits: Sequence[SearchHit] | None = None,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
        stream: bool = False,
        image_source_context: bool = False,
    ) -> tuple[dict[str, object], tuple[SearchHit, ...], frozenset[Modality]] | AbstentionReason:
        if not isinstance(question, ModelInput):
            raise ValidationError("question must be a ModelInput value")
        if isinstance(hits, (str, bytes)):
            raise ValidationError("hits must contain SearchHit values")
        retrieved = tuple(hits)
        if any(not isinstance(hit, SearchHit) for hit in retrieved):
            raise ValidationError("hits must contain SearchHit values")
        sampled_parts: dict[str, tuple[dict[str, object], ...]] = {}
        capture_metadata: dict[str, dict[str, object]] = {}
        grounded = media._fit_grounding_media(
            question,
            hits,
            video_limit=self.video_limit,
            sampled_parts=sampled_parts,
            sample_videos=Modality.IMAGE in self.capabilities,
            image_max_pixels=self.image_max_pixels,
            capture_metadata=capture_metadata,
        )
        _record_grounding_fit(retrieved, grounded)
        # `best_effort` asks the model even with nothing to ground on: the caller wants a guess
        # rather than a refusal, and `_forced_abstention` still reports that there was no
        # evidence behind it.
        if not grounded and answer_policy == "strict":
            return (
                AbstentionReason.NO_EVIDENCE
                if not (question.evidence_sources or retrieved)
                else AbstentionReason.INSUFFICIENT_EVIDENCE
            )

        source_hits = (
            (question.evidence_sources or retrieved)
            if omission_source_hits is None
            else omission_source_hits
        )
        # Keep metadata from this call's original inputs through both transport fits.
        # Omission provenance may also name media an upstream policy already withheld.
        source_assets = question.assets + tuple(asset for hit in retrieved for asset in hit.assets)
        capture_metadata = media._source_capture_metadata(source_assets, cached=capture_metadata)
        original_assets = {asset.id for hit in grounded for asset in hit.assets}
        original_parts = dict(sampled_parts)
        rendered = self.render_answer_request(
            question,
            grounded,
            source_hits,
            sampled_parts,
            capture_metadata,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
            metadata_source_hits=retrieved,
        )
        if (
            self.request_max_bytes is not None
            and _answer_wire_size(rendered[0], stream=stream) > self.request_max_bytes
        ):
            grounded = self.bound_answer_media(
                question,
                grounded,
                source_hits,
                sampled_parts,
                capture_metadata,
                answer_policy=answer_policy,
                exhaustive=exhaustive,
                stream=stream,
                metadata_source_hits=retrieved,
            )
            _record_grounding_fit(retrieved, grounded)
            if not grounded and answer_policy == "strict":
                return AbstentionReason.INSUFFICIENT_EVIDENCE
            rendered = self.render_answer_request(
                question,
                grounded,
                source_hits,
                sampled_parts,
                capture_metadata,
                answer_policy=answer_policy,
                exhaustive=exhaustive,
                metadata_source_hits=retrieved,
            )
        if image_source_context:
            # Fit complete original evidence first. The optional view adds field/prompt overhead;
            # preserve that fitted request if the overhead exceeds either text or wire bounds.
            try:
                contextual = self.render_answer_request(
                    question,
                    grounded,
                    source_hits,
                    sampled_parts,
                    capture_metadata,
                    answer_policy=answer_policy,
                    exhaustive=exhaustive,
                    metadata_source_hits=retrieved,
                    image_source_context=True,
                )
            except ModelError as error:
                if error.reason != "payload_too_large":
                    raise
            else:
                if self.request_max_bytes is None or (
                    _answer_wire_size(contextual[0], stream=stream) <= self.request_max_bytes
                ):
                    rendered = contextual
        request, modalities, assets, text_parts = rendered
        unique_assets = media._require_consistent_assets(assets)
        for modality, name in (
            (Modality.VIDEO, "video_sampled_assets"),
            (Modality.IMAGE, "image_transcoded_assets"),
        ):
            trace.get_current_span().set_attribute(
                f"sensemem.grounding.{name}",
                sum(
                    asset.modality is modality and asset.id in sampled_parts
                    for asset in unique_assets
                ),
            )
        if self.request_max_bytes is not None:
            span = trace.get_current_span()
            span.set_attribute(
                "sensemem.grounding.request_bytes", _answer_wire_size(request, stream=stream)
            )
            span.set_attribute("sensemem.grounding.request_max_bytes", self.request_max_bytes)
            span.set_attribute(
                "sensemem.grounding.budget_omitted_assets",
                len(original_assets - {asset.id for asset in unique_assets}),
            )
            for modality, name in (
                (Modality.IMAGE, "budget_transcoded_images"),
                (Modality.VIDEO, "budget_sampled_videos"),
            ):
                span.set_attribute(
                    f"sensemem.grounding.{name}",
                    sum(
                        asset.modality is modality
                        and asset.id in sampled_parts
                        and sampled_parts[asset.id] != original_parts.get(asset.id)
                        for asset in unique_assets
                    ),
                )
        _record_grounding_request(grounded, assets, text_parts)
        return request, grounded, modalities

    def bound_answer_media(  # noqa: C901 - deterministic admission against the complete wire body
        self,
        question: ModelInput,
        hits: tuple[SearchHit, ...],
        source_hits: Sequence[SearchHit],
        sampled_parts: dict[str, tuple[dict[str, object], ...]],
        capture_metadata: dict[str, dict[str, object]],
        *,
        answer_policy: AnswerPolicy,
        exhaustive: bool,
        stream: bool,
        metadata_source_hits: Sequence[SearchHit],
    ) -> tuple[SearchHit, ...]:
        maximum = cast(int, self.request_max_bytes)

        def size(selected: tuple[SearchHit, ...]) -> int:
            request, _, _, _ = self.render_answer_request(
                question,
                selected,
                source_hits,
                sampled_parts,
                capture_metadata,
                answer_policy=answer_policy,
                exhaustive=exhaustive,
                metadata_source_hits=metadata_source_hits,
            )
            return _answer_wire_size(request, stream=stream)

        if size(hits) <= maximum:
            return hits
        admitted = {asset.id for asset in question.assets}

        def selected() -> tuple[SearchHit, ...]:
            return retain_source_complete_hits(
                tuple(
                    replace(
                        hit,
                        assets=assets,
                        modality=ModelInput(assets=assets).modality if assets else Modality.TEXT,
                    )
                    for hit in hits
                    for assets in (tuple(asset for asset in hit.assets if asset.id in admitted),)
                    if assets or hit.content.strip()
                ),
                hits,
            )

        def shrink(asset: AssetRef, overflow: int, *, mandatory: bool = False) -> bool:
            # Reserve space for the changed attachment label and JSON part wrappers. The full
            # body is measured again; this allowance is not the final admission decision.
            available = media._transport_size(asset, sampled_parts) - overflow - 256
            if mandatory and available <= 0:
                # Several mandatory assets can jointly exceed the cap by more than any one
                # asset. Make progress on each rather than requiring one to absorb all excess.
                available = media._transport_size(asset, sampled_parts) // 2
            if available <= 0 or Modality.IMAGE not in self.capabilities:
                return False
            if asset.modality is Modality.IMAGE:
                try:
                    url = media._bounded_image_url(
                        asset,
                        self.image_max_pixels or 64_000_000,
                        max_encoded_bytes=min(available, media._MAX_INLINE_MODEL_ITEM_BYTES),
                    )
                except ModelError as error:
                    if error.reason != "payload_too_large":
                        raise
                    return False
                if url is None:
                    return False
                sampled_parts[asset.id] = ({"type": "image_url", "image_url": {"url": url}},)
                return True
            if asset.modality is Modality.VIDEO:
                frames = media._video_frame_urls(
                    asset, count=16, max_frame_bytes=max(1, available // 16 - 256)
                )
                if frames is not None:
                    sampled_parts[asset.id] = tuple(
                        {"type": "image_url", "image_url": {"url": frame}} for frame in frames
                    )
                    return True
            return False

        current = selected()
        current_size = size(current)
        # Mandatory question assets can be resized/sampled, but never removed.
        while current_size > maximum:
            previous_size = current_size
            for asset in reversed(media._require_consistent_assets(question.assets)):
                if current_size <= maximum:
                    break
                previous_parts = sampled_parts.get(asset.id)
                if shrink(asset, current_size - maximum, mandatory=True):
                    updated_size = size(current)
                    if updated_size < current_size:
                        current_size = updated_size
                    elif previous_parts is None:
                        sampled_parts.pop(asset.id, None)
                    else:
                        sampled_parts[asset.id] = previous_parts
            if current_size >= previous_size:
                break
        if current_size > maximum:
            raise ModelError(
                "question, mandatory media and source text/metadata exceed generation request_max_bytes",
                reason="payload_too_large",
                stage="generate",
            )
        for asset in media._require_consistent_assets(
            tuple(asset for hit in hits for asset in hit.assets)
        ):
            if asset.id in admitted:
                continue
            admitted.add(asset.id)
            candidate = selected()
            candidate_size = size(candidate)
            if candidate_size <= maximum:
                current = candidate
                continue
            # Preserve a native video that fits on its own. Lack of room after earlier ranked
            # assets causes an explicit omission, not an unnecessary loss of motion/audio.
            sample_allowed = True
            if asset.modality is Modality.VIDEO and asset.id not in sampled_parts:
                alone = tuple(
                    replace(
                        hit,
                        assets=assets,
                        modality=ModelInput(assets=assets).modality if assets else Modality.TEXT,
                    )
                    for hit in hits
                    for assets in (
                        tuple(
                            value
                            for value in hit.assets
                            if value.id in {*(value.id for value in question.assets), asset.id}
                        ),
                    )
                    if assets or hit.content.strip()
                )
                sample_allowed = size(alone) > maximum
            if (
                sample_allowed
                and shrink(asset, candidate_size - maximum)
                and size(candidate) <= maximum
            ):
                current = candidate
            else:
                admitted.remove(asset.id)
        return current

    def render_answer_request(  # noqa: C901 - keep the fitted media view and its prose together
        self,
        question_input: ModelInput,
        grounded: tuple[SearchHit, ...],
        source_hits: Sequence[SearchHit],
        sampled_parts: dict[str, tuple[dict[str, object], ...]],
        capture_metadata: dict[str, dict[str, object]],
        *,
        answer_policy: AnswerPolicy,
        exhaustive: bool,
        metadata_source_hits: Sequence[SearchHit] | None = None,
        image_source_context: bool = False,
    ) -> tuple[dict[str, object], frozenset[Modality], tuple[AssetRef, ...], tuple[str, ...]]:
        omitted_media = media._omitted_grounding_media(question_input, source_hits, grounded)
        assets = question_input.assets + tuple(asset for hit in grounded for asset in hit.assets)
        evidence_payloads = answer_evidence_payloads(
            grounded,
            omitted_media=omitted_media,
        )
        if self.answer_format == "structured":
            aliases = evidence_aliases(grounded)
            for hit, payload in zip(grounded, evidence_payloads, strict=True):
                payload["evidence_label"] = aliases[hit.id]
        # Fitting or provider fallback can leave only text from an original media record.
        source_modalities = {hit.id: hit.modality.value for hit in source_hits}
        for hit, payload in zip(grounded, evidence_payloads, strict=True):
            if source_modalities[hit.id] != "text":
                payload["source_modality"] = source_modalities[hit.id]
            if (
                image_source_context
                and source_modalities[hit.id] == "image"
                and hit.content.strip()
                and any(asset.modality is Modality.IMAGE for asset in hit.assets)
                and not omitted_media.get(hit.id, {}).get("image", 0)
            ):
                payload["auxiliary_source_text"] = hit.content
                payload["content"] = ""
        constraints = answer_constraint_payloads(grounded, evidence_payloads)
        metadata_only = metadata_source_hits is not None and bool(capture_metadata)
        text_parts = (
            _answer_text_parts(
                question_input,
                grounded,
                evidence_payloads,
                capture_metadata=capture_metadata,
                source_hits=metadata_source_hits,
            )
            if assets or metadata_only
            else (
                _json_text(
                    {
                        "question": question_input.text,
                        "hits": evidence_payloads,
                        **({"constraint_candidates": constraints} if constraints else {}),
                    }
                ),
            )
        )
        system_prompt = prompts._answer_system_prompt(
            grounded,
            omitted_media,
            evidence_payloads,
            answer_policy=answer_policy,
            exhaustive=exhaustive,
            answer_style=self.answer_style,
            answer_format=self.answer_format,
            source_hits=source_hits,
            image_source_context=any(
                "auxiliary_source_text" in payload for payload in evidence_payloads
            ),
        )
        # This is a byte safety bound on all answer text, independent of the kernel's soft
        # evidence packing budget and the provider's tokenizer and context window.
        _require_answer_text_size(
            chain(
                (system_prompt,),
                text_parts,
                (
                    media._transport_label(
                        label,
                        sampled_parts.get(asset_id),
                        next(asset.modality for asset in assets if asset.id == asset_id),
                    )
                    for asset_id, label in media._media_labels(question_input, grounded).items()
                ),
            )
        )
        required = {Modality.TEXT}
        required.update(cast(Modality, asset.modality) for asset in assets)
        _require_capabilities("generation", frozenset(required), self.capabilities)
        unique_assets = media._require_consistent_assets(assets)
        verified_assets = media._require_asset_integrity(unique_assets)
        media._require_verified_inline_size(
            tuple(value for value in verified_assets if value[0].id not in sampled_parts)
        )
        content: str | list[dict[str, object]] = (
            _answer_parts(
                question_input,
                grounded,
                text_parts,
                media_slack_bytes=media._MAX_INLINE_MODEL_BYTES
                - sum(media._transport_size(asset, sampled_parts) for asset in unique_assets),
                minimum_video_seconds=(
                    self.min_video_seconds if Modality.IMAGE in self.capabilities else None
                ),
                sampled_parts=sampled_parts,
                audio_data_url=self.model.rsplit("/", 1)[-1] == "qwen3.8-omni-flash",
            )
            if assets or metadata_only
            else text_parts[0]
        )
        if not isinstance(content, str):
            media._require_inline_media_parts(content)
            _require_answer_text_size(
                chain(
                    (system_prompt,),
                    (cast(str, part["text"]) for part in content if part["type"] == "text"),
                )
            )
        modalities = media._generation_modalities(content)
        request: dict[str, object] = (
            self.json_request(system_prompt, content, schema="answer")
            if self.answer_format == "structured"
            else {
                "model": self.model,
                "store": self.store,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            }
        )
        if self.seed is not None:
            request["seed"] = self.seed
        if self.temperature is not None:
            request["temperature"] = self.temperature
        # Injected here, not at a call site, so streaming and non-streaming carry the same controls.
        if self.max_tokens is not None:
            request[output_limit_parameter(self.model)] = self.max_tokens
        if self.extra_body is not None:
            request["extra_body"] = dict(self.extra_body)
        return request, modalities, assets, text_parts
