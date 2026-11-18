"""Translate provider usage and provenance into SenseMem telemetry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from opentelemetry import trace

from sensemem._telemetry import (
    EMBEDDING_VIDEO_SAMPLED,
    GROUNDING_ASSET_IDS,
    GROUNDING_HITS_DROPPED,
    GROUNDING_MEDIA_ELIDED,
    GROUNDING_MEDIA_FALLBACK,
    GROUNDING_RECORD_IDS,
    GROUNDING_TEXT_BYTES,
    nested_model_span,
    record_model_provenance,
    record_model_request_id,
    record_model_usage,
)
from sensemem.content import AssetRef
from sensemem.records import SearchHit
from sensemem.types import Modality

_EvidenceOperationRejection = Literal[
    "protocol_finish",
    "protocol_type",
    "protocol_count",
    "protocol_name",
    "protocol_arguments",
    "schema",
    "invalid_evidence",
    "response_invalid",
]


@dataclass(frozen=True, slots=True)
class _ModelUsage:
    token_based: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_by_modality: Mapping[str, int] = field(default_factory=dict)
    output_by_modality: Mapping[str, int] = field(default_factory=dict)
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    audio_seconds: float | None = None


def _record_openai_usage(
    response: object,
    *,
    input_modalities: frozenset[Modality],
    output_modalities: frozenset[Modality],
    request_count: int = 1,
) -> None:
    _record_openai_provenance(response)
    _record_usage_batch(
        (
            _model_usage(
                response,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
            ),
        ),
        request_count=request_count,
    )


def _record_openai_provenance(response: object) -> None:
    """Keep only stable response identity, never provider request or response bodies."""
    model = _member(response, "model")
    fingerprint = _member(response, "system_fingerprint")
    record_model_provenance(
        response_model=model if isinstance(model, str) else None,
        system_fingerprint=fingerprint if isinstance(fingerprint, str) else None,
    )
    _record_provider_request_id(response)


def _record_provider_request_id(value: object) -> None:
    request_id = getattr(value, "_request_id", None) or getattr(value, "request_id", None)
    if not request_id:
        headers = getattr(getattr(value, "response", None), "headers", None)
        if headers is not None:
            request_id = headers.get("x-request-id")
    if isinstance(request_id, str):
        record_model_request_id(request_id)


def _record_usage_batch(
    usages: Sequence[_ModelUsage | None],
    *,
    request_count: int,
) -> None:
    available = tuple(usage for usage in usages if usage is not None)
    token_usages = tuple(usage for usage in available if usage.token_based)
    reported = tuple(usage for usage in token_usages if usage.total_tokens is not None)
    missing = request_count - len(available)
    expected = len(token_usages) + missing
    input_tokens = _sum_reported(token_usages, "input_tokens")
    output_tokens = _sum_reported(token_usages, "output_tokens")
    total_tokens = _sum_reported(reported, "total_tokens")
    cached_input_tokens = _sum_reported(token_usages, "cached_input_tokens")
    reasoning_output_tokens = _sum_reported(token_usages, "reasoning_output_tokens")
    input_by_modality = _sum_modalities(token_usages, "input_by_modality")
    output_by_modality = _sum_modalities(token_usages, "output_by_modality")
    audio_seconds = (
        sum(cast(float, usage.audio_seconds) for usage in available)
        if len(available) == request_count
        and all(usage.audio_seconds is not None for usage in available)
        else None
    )
    record_model_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        input_by_modality=input_by_modality,
        output_by_modality=output_by_modality,
        request_count=request_count,
        expected_requests=expected,
        reported_requests=len(reported),
        input_tokens_complete=_component_complete(token_usages, "input_tokens", expected),
        output_tokens_complete=_component_complete(token_usages, "output_tokens", expected),
        cached_input_tokens_complete=_component_complete(
            token_usages, "cached_input_tokens", expected
        ),
        reasoning_output_tokens_complete=_component_complete(
            token_usages, "reasoning_output_tokens", expected
        ),
        audio_seconds=audio_seconds or None,
    )


def _model_usage(
    response: object,
    *,
    input_modalities: frozenset[Modality],
    output_modalities: frozenset[Modality],
) -> _ModelUsage | None:
    usage = _member(response, "usage")
    if usage is None:
        return None
    input_tokens = _count(usage, "input_tokens", "prompt_tokens")
    output_tokens = _count(usage, "output_tokens", "completion_tokens")
    total_tokens = _count(usage, "total_tokens")
    seconds = _number(usage, "seconds")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None if seconds is None else _ModelUsage(False, audio_seconds=seconds)
    if not output_modalities and output_tokens is None:
        output_tokens = 0
    if input_tokens is None and total_tokens is not None and output_tokens is not None:
        input_tokens = max(0, total_tokens - output_tokens)
    if output_tokens is None and total_tokens is not None and input_tokens is not None:
        output_tokens = max(0, total_tokens - input_tokens)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if (
        total_tokens is not None
        and input_tokens is not None
        and output_tokens is not None
        and total_tokens != input_tokens + output_tokens
    ):
        return None
    input_details = _member(
        usage,
        "input_token_details",
        "input_tokens_details",
        "prompt_tokens_details",
    )
    output_details = _member(
        usage,
        "output_token_details",
        "output_tokens_details",
        "completion_tokens_details",
    )
    return _ModelUsage(
        True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_by_modality=_modality_tokens(input_tokens, input_details, input_modalities),
        output_by_modality=_modality_tokens(output_tokens, output_details, output_modalities),
        cached_input_tokens=_count(input_details, "cached_tokens"),
        reasoning_output_tokens=_count(output_details, "reasoning_tokens"),
        audio_seconds=seconds,
    )


def _modality_tokens(
    total: int | None,
    details: object,
    requested: frozenset[Modality],
) -> Mapping[str, int]:
    aliases = {
        "text": ("text_tokens",),
        "image": ("image_tokens", "vision_tokens"),
        "video": ("video_tokens",),
        "audio": ("audio_tokens",),
    }
    exact = {
        modality: count
        for modality, names in aliases.items()
        if (count := _count(details, *names)) is not None
    }
    if total is None:
        return exact
    if sum(exact.values()) > total:
        return {"unattributed": total}
    remainder = total - sum(exact.values())
    missing = {modality.value for modality in requested} - exact.keys()
    if remainder and len(missing) == 1:
        exact[missing.pop()] = remainder
    elif remainder:
        exact["unattributed"] = remainder
    return exact


def _sum_reported(usages: Sequence[_ModelUsage], name: str) -> int | None:
    values = tuple(value for usage in usages if (value := getattr(usage, name)) is not None)
    return sum(cast(tuple[int, ...], values)) if values else None


def _component_complete(usages: Sequence[_ModelUsage], name: str, expected: int) -> bool:
    return bool(
        expected
        and len(usages) == expected
        and all(getattr(usage, name) is not None for usage in usages)
    )


def _sum_modalities(
    usages: Sequence[_ModelUsage],
    name: str,
) -> Mapping[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        for modality, count in cast(Mapping[str, int], getattr(usage, name)).items():
            totals[modality] = totals.get(modality, 0) + count
    return totals


def _member(value: object, *names: str) -> object:
    for name in names:
        candidate = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _count(value: object, *names: str) -> int | None:
    candidate = _member(value, *names)
    return (
        candidate
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0
        else None
    )


def _number(value: object, *names: str) -> float | None:
    candidate = _member(value, *names)
    if (
        isinstance(candidate, bool)
        or not isinstance(candidate, int | float)
        or not math.isfinite(candidate)
        or candidate < 0
    ):
        return None
    return float(candidate)


def _record_video_sampling(count: int) -> None:
    """Publish how many inputs reached the model as stills instead of video."""
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(EMBEDDING_VIDEO_SAMPLED, count)


def _evidence_operation_event(
    status: Literal[
        "submitted", "executed", "needs_evidence", "rejected", "response_invalid", "recovery"
    ],
    *,
    rejection: _EvidenceOperationRejection | None = None,
) -> None:
    # Reuse the caller's tracer provider. These stages make no model request, and carry no
    # proposal, source, answer or provider body; names alone suffice for aggregate counts.
    names = (status,) if rejection is None else (status, f"rejected_{rejection}")
    for name in names:
        with nested_model_span(
            f"sensemem.evidence_operation.{name}",
            attributes={"sensemem.span.kind": "stage", "sensemem.stage": "generate"},
        ):
            pass


def _record_grounding_request(
    grounded: Sequence[SearchHit], assets: Sequence[AssetRef], text_parts: Sequence[str]
) -> None:
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attributes(
            {
                GROUNDING_RECORD_IDS: tuple(hit.id for hit in grounded),
                GROUNDING_ASSET_IDS: tuple(dict.fromkeys(asset.id for asset in assets)),
                GROUNDING_TEXT_BYTES: sum(len(part.encode("utf-8")) for part in text_parts),
                GROUNDING_MEDIA_FALLBACK: False,
            }
        )


def _record_grounding_fit(
    retrieved: Sequence[SearchHit],
    grounded: Sequence[SearchHit],
) -> None:
    """Record how much retrieved evidence the inline budget removed, so no loss stays silent."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    grounded_assets = {hit.id: {asset.id for asset in hit.assets} for hit in grounded}
    span.set_attribute(
        GROUNDING_MEDIA_ELIDED,
        sum(
            1
            for hit in retrieved
            if {asset.id for asset in hit.assets} - grounded_assets.get(hit.id, set())
        ),
    )
    span.set_attribute(GROUNDING_HITS_DROPPED, len(retrieved) - len(grounded))
