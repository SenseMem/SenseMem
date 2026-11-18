"""Validate provider outputs and translate them into SenseMem values.

These parsers read no storage and call no model. Provider bodies never become public
error details; malformed structures, refusals and truncation retain distinct failures.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Literal, cast

from opentelemetry import trace

from sensemem._telemetry import FORMATION_PROPOSALS_DROPPED, GEN_AI_FINISH_REASONS
from sensemem.control_values import MemoryIntent, MemoryOperation
from sensemem.events import EventSegment
from sensemem.evidence import evidence_aliases, evidence_identity_ids
from sensemem.exceptions import ModelError, ModelOutputTruncatedError, ValidationError
from sensemem.identity import IdentityClaim
from sensemem.models._answer_output import parse_answer_draft, render_answer_draft
from sensemem.models._evidence_operations import EVIDENCE_OPERATION_TOOL_NAME
from sensemem.models._openai_telemetry import _EvidenceOperationRejection, _member
from sensemem.models.base import FormationInput
from sensemem.perception import SpeakerName, VisualDescription
from sensemem.prompts import generation as prompts
from sensemem.records import AnswerResult, MemoryRecord, SearchHit
from sensemem.semantics import AffectLink, AffectObjectRef, AffectWitness, FormationProposal
from sensemem.spatial import SpatialContext
from sensemem.types import (
    AbstentionReason,
    AnswerPolicy,
    EvidenceBasis,
    Modality,
    RecordKind,
    SpatialAnchor,
)

_CONSOLIDATION_FIELDS = frozenset(
    {"intent", "evidence", "targets", "proposal", "claim", "rationale"}
)


_FORMATION_FIELDS = frozenset(
    {
        "kind",
        "content",
        "subject",
        "predicate",
        "value",
        "confidence",
        "valid_from",
        "valid_until",
        "spatial",
        "cue_modality",
        "valence",
        "arousal",
        "evidence_observation_ids",
        "local_id",
        "affect_link",
        "affect_witnesses",
    }
)


_MAX_FORMATION_PROPOSALS = 64


_CLAIM_FIELDS = frozenset({"identity_id", "name", "relationship"})


_MAX_CONSOLIDATION_OPERATIONS = 16


_TRUNCATED_ANSWER_ERROR = (
    "generation stopped at the output token limit; raise generation_max_tokens or lower the "
    "answer limit"
)


class _InvalidStructuredResponse(ModelError):
    """A successful provider response whose structured payload can be retried once."""

    _cleanup_failed = False


def _invalid_json_response(
    subject: str, stage: str, detail: str = ""
) -> _InvalidStructuredResponse:
    return _InvalidStructuredResponse(
        f"{subject} response was invalid{detail}",
        reason="response_invalid",
        stage=stage,
    )


def _invalid_consolidation_response(detail: str = "") -> ModelError:
    return _invalid_json_response("consolidation", "consolidate", detail)


def _invalid_formation_response() -> ModelError:
    return _invalid_json_response("formation", "form")


def _model_refused(subject: str, stage: str) -> ModelError:
    return ModelError(f"{subject} was refused by the model", reason="model_refused", stage=stage)


def _record_finish_reason(finish_reason: object) -> None:
    """Record the provider stop reason so truncation is countable instead of inferred."""
    if isinstance(finish_reason, str) and finish_reason:
        trace.get_current_span().set_attribute(GEN_AI_FINISH_REASONS, (finish_reason,))


def _json_completion_text(response: object, *, subject: str, stage: str) -> str:
    """Return the single JSON-object completion's text, or fail the way its caller reports."""
    choices = getattr(response, "choices", None)
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or getattr(choices[0], "index", None) != 0
    ):
        raise _invalid_json_response(subject, stage)
    finish_reason = getattr(choices[0], "finish_reason", None)
    _record_finish_reason(finish_reason)
    message = getattr(choices[0], "message", None)
    if getattr(message, "refusal", None) or finish_reason == "content_filter":
        raise _model_refused(subject, stage)
    if finish_reason == "length":
        raise ModelOutputTruncatedError(
            f"{subject} stopped at the output token limit; raise generation_max_tokens",
            stage=stage,
        )
    content = getattr(message, "content", None)
    if finish_reason != "stop" or not isinstance(content, str) or not content.strip():
        raise _invalid_json_response(subject, stage)
    return content.strip()


def _answer_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or getattr(choices[0], "index", None) != 0
    ):
        raise ModelError(
            "generation response was invalid", reason="response_invalid", stage="generate"
        )
    finish_reason = getattr(choices[0], "finish_reason", None)
    _record_finish_reason(finish_reason)
    if getattr(getattr(choices[0], "message", None), "refusal", None):
        raise _model_refused("generation", "generate")
    if finish_reason == "length":
        raise ModelOutputTruncatedError(_TRUNCATED_ANSWER_ERROR, stage="generate")
    if finish_reason == "content_filter":
        raise _model_refused("generation", "generate")
    answer = getattr(getattr(choices[0], "message", None), "content", None)
    if finish_reason != "stop" or not isinstance(answer, str) or not answer.strip():
        raise ModelError(
            "generation response was invalid", reason="response_invalid", stage="generate"
        )
    return answer.strip()


def _evidence_operation_response(  # noqa: C901 - preserve ordered native protocol admission
    response: object,
    *,
    tools_allowed: bool,
    hits: Sequence[SearchHit],
) -> tuple[str | None, str | None, _EvidenceOperationRejection | None]:
    """Admit ordinary text or return native arguments with the first protocol rejection."""
    choices = getattr(response, "choices", None)
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or getattr(choices[0], "index", None) != 0
    ):
        raise ModelError(
            "generation response was invalid", reason="response_invalid", stage="generate"
        )
    choice = choices[0]
    message = getattr(choice, "message", None)
    reason = getattr(choice, "finish_reason", None)
    if getattr(message, "refusal", None) or reason in {"length", "content_filter"}:
        _answer_text(response)
    calls = getattr(message, "tool_calls", None)
    if not calls:
        answer = _answer_text(response)
        if tools_allowed:
            labels = "|".join(re.escape(label) for label in evidence_aliases(hits).values())
            item = rf"(?:{labels}|\"(?:{labels})\"|'(?:{labels})')"
            alias_list = re.fullmatch(
                rf"\s*(?:\[\s*)?{item}(?:[\s,;\uff0c\uff1b\u3001]+{item})*\s*(?:\])?\s*",
                answer,
            )
            _require_no_evidence_operation_text(answer)
            if alias_list:
                raise ModelError(
                    "generation response contained unaccepted protocol text",
                    reason="response_invalid",
                    stage="generate",
                )
        return answer, None, None
    function = (
        _member(calls[0], "function")
        if isinstance(calls, list | tuple) and len(calls) == 1
        else None
    )
    arguments = _member(function, "arguments")
    rejection: _EvidenceOperationRejection | None = None
    if reason != "tool_calls":
        rejection = "protocol_finish"
    elif not tools_allowed or not isinstance(calls, list | tuple):
        rejection = "protocol_type"
    elif len(calls) != 1:
        rejection = "protocol_count"
    elif _member(calls[0], "type") != "function":
        rejection = "protocol_type"
    elif _member(function, "name") != EVIDENCE_OPERATION_TOOL_NAME:
        rejection = "protocol_name"
    elif not isinstance(arguments, str):
        rejection = "protocol_arguments"
    return None, arguments if isinstance(arguments, str) else None, rejection


def _require_no_evidence_operation_text(answer: str) -> None:
    if re.search(
        r"<(tool_calls|function_calls)>\s*<invoke\s+name=([\"'])"
        r"(?:functions\.)?submit_evidence_operation\2\s*>"
        r".*?</invoke>\s*</\1>",
        answer,
        re.DOTALL,
    ):
        raise ModelError(
            "generation response contained unaccepted protocol text",
            reason="response_invalid",
            stage="generate",
        )


def _evidence_operation_recovery_answer(response: object) -> str:
    """Accept recovery text only, rejecting protocol imitations before native calls."""
    answer = _answer_text(response)
    choice = cast(list[object], getattr(response, "choices", None))[0]
    _require_no_evidence_operation_text(answer)
    if getattr(getattr(choice, "message", None), "tool_calls", None):
        raise ModelError(
            "generation response was invalid", reason="response_invalid", stage="generate"
        )
    return answer


def _event_segments(content: str, count: int) -> tuple[EventSegment, ...]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        raise _invalid_json_response("event segmentation", "segment_events") from None
    if not isinstance(payload, dict) or set(payload) != {"events"}:
        raise _invalid_json_response("event segmentation", "segment_events")
    events = payload["events"]
    if not isinstance(events, list) or len(events) > count:
        raise _invalid_json_response("event segmentation", "segment_events")
    result = []
    previous = 0
    for event in events:
        if not isinstance(event, dict) or set(event) != {"end_chunk", "summary", "complete"}:
            raise _invalid_json_response("event segmentation", "segment_events")
        end, summary, complete = event["end_chunk"], event["summary"], event["complete"]
        if (
            isinstance(end, bool)
            or not isinstance(end, int)
            or not previous < end <= count
            or not isinstance(summary, str)
            or not summary.strip()
            or not isinstance(complete, bool)
            or (not complete and (event is not events[-1] or end != count))
        ):
            raise _invalid_json_response("event segmentation", "segment_events")
        try:
            result.append(EventSegment(end_chunk=end, summary=summary, complete=complete))
        except ValidationError:
            raise _invalid_json_response("event segmentation", "segment_events") from None
        previous = end
    return tuple(result)


def _formation_results(  # noqa: C901 - validates the strict batch witness envelope
    content: str,
    inputs: Sequence[FormationInput],
    *,
    source_aliases: Sequence[Mapping[str, str]],
) -> tuple[tuple[FormationProposal, ...], ...]:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        raise _invalid_formation_response() from None
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise _invalid_formation_response()
    items = payload["items"]
    if not isinstance(items, list) or len(items) != len(inputs):
        raise _invalid_formation_response()
    expected = tuple(f"observation_{index}" for index, _value in enumerate(inputs))
    ids_by_alias = dict(zip(expected, (value.memory_id for value in inputs), strict=True))
    ids_by_alias.update(
        (alias, memory_id) for aliases in source_aliases for memory_id, alias in aliases.items()
    )
    dropped = 0
    by_id: dict[str, tuple[FormationProposal, ...]] = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"observation_id", "proposals"}:
            raise _invalid_formation_response()
        observation_id = item["observation_id"]
        values = item["proposals"]
        if (
            not isinstance(observation_id, str)
            or observation_id in by_id
            or not isinstance(values, list)
            or len(values) > _MAX_FORMATION_PROPOSALS
        ):
            raise _invalid_formation_response()
        if observation_id not in expected:
            raise _invalid_formation_response()
        target_index = int(observation_id.removeprefix("observation_"))
        target = inputs[target_index]
        allowed = set(expected) | {
            source_aliases[target_index][record.id]
            for record in target.history
            if record.forgotten_at is None
            and (
                record.context is None
                or (record.context.visible and record.context.retired_at is None)
            )
        }
        # A malformed proposal is one derived opinion the model expressed badly. It must not cost
        # the caller the observation it was derived from: `add` has already committed that source
        # by the time formation runs, so raising here fails a write that in fact succeeded, and
        # the retry re-runs formation and fails again -- the memory could never be added at all.
        # Structural damage to the envelope still raises below; only individual proposals are
        # skipped, and the count is published so the loss is never silent.
        parsed: list[FormationProposal | None] = []
        local_ids = Counter(
            value["local_id"]
            for value in values
            if isinstance(value, dict) and isinstance(value.get("local_id"), str)
        )
        witness_sources = {
            alias: tuple(part.value for part in source.content.parts if part.kind == "text")
            for alias, source in zip(expected, inputs, strict=True)
        }
        witness_sources.update(
            (
                source_aliases[target_index][record.id],
                tuple(part.value for part in record.parts if part.kind == "text"),
            )
            for record in target.history
            if record.context is None or record.context.kind is RecordKind.OBSERVATION
        )
        identity_ids = {
            identity_id
            for record in target.history
            if source_aliases[target_index][record.id] in allowed
            for identity_id in (
                *(
                    (record.context.identity_id,)
                    if record.context and record.context.identity_id
                    else ()
                ),
                *(item for group in evidence_identity_ids(record.content) for item in group),
            )
        }
        identity_ids.update(
            item
            for source in inputs
            for group in evidence_identity_ids(source.content.text)
            for item in group
        )
        for value in values:
            if not isinstance(value, dict) or "evidence_observation_ids" not in value:
                parsed.append(None)
                continue
            aliases = value["evidence_observation_ids"]
            if (
                not isinstance(aliases, list)
                or not aliases
                or any(not isinstance(alias, str) or alias not in allowed for alias in aliases)
                or len(set(aliases)) != len(aliases)
                or observation_id not in aliases
            ):
                # Wrong witnesses make this one proposal ungroundable, not the whole response.
                parsed.append(None)
                continue
            evidence_ids = tuple(ids_by_alias[alias] for alias in aliases)
            proposal = _formation_proposal(value, evidence_ids)
            if proposal is not None:
                try:
                    if proposal.local_id is not None and local_ids[proposal.local_id] != 1:
                        raise ValueError
                    link = proposal.affect_link
                    if link is not None:
                        references = (link.affect_id, link.object_ref.memory_id)
                        if any(
                            reference is not None
                            and reference not in allowed
                            and not (
                                reference.startswith("proposal:")
                                and local_ids[reference.removeprefix("proposal:")] == 1
                            )
                            for reference in references
                        ) or (
                            link.object_ref.identity_id is not None
                            and link.object_ref.identity_id not in identity_ids
                        ):
                            raise ValueError
                        link = replace(
                            link,
                            affect_id=ids_by_alias.get(link.affect_id, link.affect_id),
                            object_ref=replace(
                                link.object_ref,
                                memory_id=ids_by_alias.get(
                                    link.object_ref.memory_id, link.object_ref.memory_id
                                )
                                if link.object_ref.memory_id is not None
                                else None,
                            ),
                        )
                    witnesses = []
                    for witness in proposal.affect_witnesses:
                        if (
                            witness.source_id not in aliases
                            or not any(
                                witness.quote in part
                                for part in witness_sources.get(witness.source_id, ())
                            )
                            or re.search(
                                rf"(?<![A-Za-z0-9_]){re.escape(witness.reporter)}(?![A-Za-z0-9_])",
                                witness.quote,
                                re.IGNORECASE,
                            )
                            is None
                        ):
                            raise ValueError
                        witnesses.append(
                            replace(witness, source_id=ids_by_alias[witness.source_id])
                        )
                    proposal = replace(
                        proposal, affect_link=link, affect_witnesses=tuple(witnesses)
                    )
                except (ValueError, ValidationError):
                    proposal = None
            parsed.append(proposal)
        dropped += sum(1 for proposal in parsed if proposal is None)
        by_id[observation_id] = tuple(proposal for proposal in parsed if proposal is not None)
    if set(by_id) != set(expected):
        raise _invalid_formation_response()
    # Recorded even at zero: absence of the attribute would otherwise be indistinguishable from
    # formation never having run, which is the failure this signal exists to make visible.
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(FORMATION_PROPOSALS_DROPPED, dropped)
    return tuple(by_id[memory_id] for memory_id in expected)


def _formation_proposal(
    value: object,
    evidence_ids: tuple[str, ...] | None = None,
) -> FormationProposal | None:
    """Build one proposal, or return None when the model shaped this one wrongly."""
    if not isinstance(value, dict):
        return None
    fields = set(value)
    if not {"kind", "content", "confidence"} <= fields or fields - _FORMATION_FIELDS:
        return None
    # The wire boundary checks shape; the value constructors own field validation.
    try:
        return FormationProposal(
            kind=cast(RecordKind, value["kind"]),
            basis=EvidenceBasis.MODEL_INFERENCE,
            content=cast(str, value["content"]),
            subject=cast(str | None, value.get("subject")),
            predicate=cast(str | None, value.get("predicate")),
            value=cast(str | None, value.get("value")),
            confidence=cast(float, value["confidence"]),
            valid_from=_formation_datetime(value.get("valid_from")),
            valid_until=_formation_datetime(value.get("valid_until")),
            spatial=_formation_spatial(value.get("spatial")),
            cue_modality=cast(Modality | None, value.get("cue_modality")),
            valence=cast(float | None, value.get("valence")),
            arousal=cast(float | None, value.get("arousal")),
            evidence_ids=evidence_ids,
            local_id=cast(str | None, value.get("local_id")),
            affect_link=_formation_affect_link(value.get("affect_link")),
            affect_witnesses=_formation_affect_witnesses(value.get("affect_witnesses", [])),
        )
    except (TypeError, ValueError, ValidationError):
        return None


def _formation_affect_link(value: object) -> AffectLink | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"affect_id", "role", "object_ref", "polarity"}:
        raise ValueError
    reference = value["object_ref"]
    if not isinstance(reference, dict) or set(reference) != {"memory_id", "identity_id"}:
        raise ValueError
    return AffectLink(
        affect_id=cast(str, value["affect_id"]),
        role=cast(Literal["target", "elicitor"], value["role"]),
        object_ref=AffectObjectRef(
            memory_id=cast(str | None, reference["memory_id"]),
            identity_id=cast(str | None, reference["identity_id"]),
        ),
        polarity=cast(Literal["affirmed", "denied"], value["polarity"]),
    )


def _formation_affect_witnesses(value: object) -> tuple[AffectWitness, ...]:
    if not isinstance(value, list) or any(
        not isinstance(witness, dict) or set(witness) != {"source_id", "reporter", "quote"}
        for witness in value
    ):
        raise ValueError
    return tuple(
        AffectWitness(
            source_id=witness["source_id"], reporter=witness["reporter"], quote=witness["quote"]
        )
        for witness in value
    )


def _formation_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _formation_spatial(value: object) -> SpatialContext | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError
    required = {"frame_id", "anchor", "x", "y"}
    allowed = required | {"z", "orientation_xyzw", "position_uncertainty_m"}
    if not required <= set(value) or set(value) - allowed:
        raise ValueError
    orientation = value.get("orientation_xyzw")
    if orientation is not None:
        if isinstance(orientation, (str, bytes)) or not isinstance(orientation, Sequence):
            raise ValueError
        orientation = tuple(orientation)
    return SpatialContext(
        frame_id=cast(str, value["frame_id"]),
        anchor=cast(SpatialAnchor, value["anchor"]),
        x=cast(float, value["x"]),
        y=cast(float, value["y"]),
        z=cast(float, value.get("z", 0.0)),
        orientation_xyzw=cast(tuple[float, float, float, float] | None, orientation),
        position_uncertainty_m=cast(float | None, value.get("position_uncertainty_m")),
    )


def _consolidation_results(
    content: str,
    evidence: Sequence[MemoryRecord],
) -> tuple[MemoryOperation, ...]:
    """Validate operations without copying model text into public errors."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        raise _invalid_consolidation_response() from None
    if not isinstance(payload, dict) or set(payload) != {"operations"}:
        raise _invalid_consolidation_response()
    items = payload["operations"]
    if not isinstance(items, list) or len(items) > _MAX_CONSOLIDATION_OPERATIONS:
        raise _invalid_consolidation_response()
    return tuple(_consolidation_operation(item, evidence) for item in items)


def _consolidation_operation(
    value: object,
    evidence: Sequence[MemoryRecord],
) -> MemoryOperation:
    if not isinstance(value, dict):
        raise _invalid_consolidation_response()
    fields = set(value)
    unknown = fields - _CONSOLIDATION_FIELDS
    if unknown:
        # Never guess a misspelled field or echo arbitrary model-provided keys into logs.
        raise _invalid_consolidation_response(f": operation names {len(unknown)} unknown field(s)")
    if "intent" not in fields:
        raise _invalid_consolidation_response(": operation names no intent")
    supplied = value.get("proposal")
    proposal = None if supplied is None else _formation_proposal(supplied)
    if supplied is not None and proposal is None:
        # A bad shape is reported by returning None, not by raising, so this used to fall
        # through as "no proposal at all": a consolidate then failed on the missing proposal
        # with the bare message, naming nothing about the proposal the model actually wrote.
        raise _invalid_consolidation_response(": proposal is not a valid formation proposal")
    claimed = value.get("claim")
    cited = _cited_ids(value.get("evidence"), evidence)
    try:
        return MemoryOperation(
            intent=cast(MemoryIntent, value["intent"]),
            evidence_ids=cited,
            target_ids=_cited_ids(value.get("targets"), evidence),
            proposal=proposal,
            claim=None if claimed is None else _cited_claim(claimed, evidence),
            rationale=cast(str | None, value.get("rationale")),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _invalid_consolidation_response() from error


def _cited_claim(value: object, evidence: Sequence[MemoryRecord]) -> IdentityClaim:
    """Resolve one identity claim, rejecting an ID the evidence rendering never showed.

    An index outside the evidence list already fails the whole response; an identity nobody in
    the shown evidence has is the same class of miscount, so it fails the same way rather than
    reaching the kernel to be rejected there.
    """
    if not isinstance(value, dict) or set(value) - _CLAIM_FIELDS:
        raise _invalid_consolidation_response()
    shown = {
        identity_id
        for record in evidence
        for group in evidence_identity_ids(record.content)
        for identity_id in group
    }
    if value.get("identity_id") not in shown:
        raise _invalid_consolidation_response()
    try:
        return IdentityClaim(
            identity_id=cast(str, value["identity_id"]),
            name=cast(str, value.get("name")),
            relationship=cast(str | None, value.get("relationship")),
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise _invalid_consolidation_response() from error


def _cited_ids(value: object, evidence: Sequence[MemoryRecord]) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _invalid_consolidation_response()
    resolved: list[str] = []
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(evidence):
            raise _invalid_consolidation_response()
        resolved.append(evidence[index].id)
    return tuple(resolved)


def _vision_captions(content: str, count: int) -> tuple[str, ...]:
    """Accept only one non-empty caption per visual, in order, and nothing else."""
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, RecursionError):
        raise _invalid_json_response("vision description", "describe") from None
    values = payload.get("descriptions") if isinstance(payload, Mapping) else None
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(not isinstance(value, str) or not value.strip() for value in values)
    ):
        raise _invalid_json_response("vision description", "describe")
    return tuple(_caption_lines(cast(str, value)) for value in values)


def _caption_lines(caption: str) -> str:
    """Keep the caption's labelled lines and nothing that could split it into two sections.

    A caption crosses a memory's text document, whose sections are separated by a blank line and
    introduced by a marker line. A single newline inside the caption is therefore free -- and is
    what carries the labelled structure -- while a blank line inside it would end the section and
    leave the rest of the caption unlabelled, so blank lines are dropped rather than preserved.
    Whitespace inside a line is collapsed for the same reason it always was: it is index terms.
    """
    lines = (" ".join(line.split()) for line in caption.splitlines())
    return "\n".join(line for line in lines if line)


def _visual_description(value: str) -> VisualDescription:
    """Decode this provider recipe's labelled wire output into explicit proposals."""
    visible, facts, names = [], [], []
    for line in value.splitlines():
        if not line.startswith("Fact:"):
            visible.append(line)
            continue
        fact = line.removeprefix("Fact:").strip()
        if not fact:
            continue
        facts.append(fact)
        match = re.fullmatch(
            r"speakers?[ _-]?(?P<index>\d{1,3})\s+is\s+called\s+"
            r"(?P<name>[^\s.;,]+(?: [^\s.;,]+){0,3})\s*[.;]?",
            fact,
            re.IGNORECASE,
        )
        if match is not None and int(match["index"]) > 0:
            names.append(SpeakerName(f"speaker_{int(match['index'])}", match["name"]))
    return VisualDescription("\n".join(visible), tuple(facts), tuple(names))


def _parse_media_requests(content: str, available: int, limit: int) -> tuple[int, ...]:
    payload = json.loads(content)
    if not isinstance(payload, dict) or set(payload) != {"requests"}:
        raise ValueError("invalid media diagnosis envelope")
    requests = payload["requests"]
    if not isinstance(requests, list) or len(requests) > limit:
        raise ValueError("media diagnosis exceeds its request bound")
    indices: list[int] = []
    for item in requests:
        if not isinstance(item, dict) or set(item) != {"media_index", "reason"}:
            raise ValueError("invalid media request")
        index = item["media_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < available
            or index in indices
        ):
            raise ValueError("media request index must be unique and in range")
        if item["reason"] not in {"detail_missing", "conflicting_evidence"}:
            raise ValueError("media request needs an allowed gap reason")
        indices.append(index)
    return tuple(indices)


def _embedding_vectors(
    response: object,
    count: int,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != count:
        raise ModelError("embedding response was invalid", reason="response_invalid", stage="embed")
    ordered: dict[int, tuple[float, ...]] = {}
    for item in data:
        index = getattr(item, "index", None)
        values = getattr(item, "embedding", None)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= count
            or index in ordered
            or not isinstance(values, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int | float) for value in values
            )
        ):
            raise ModelError(
                "embedding response was invalid", reason="response_invalid", stage="embed"
            )
        ordered[index] = _embedding_vector(values, dimension)
    return tuple(ordered[index] for index in range(count))


def _embedding_vector(values: Sequence[float], dimension: int) -> tuple[float, ...]:
    if len(values) != dimension:
        # Name both widths: a declared dimension copied from another model's configuration fails
        # every write, and "invalid" alone sent an operator hunting through the server instead.
        raise ModelError(
            "embedding response was invalid: the model returned "
            f"{len(values)} values but the configured dimension is {dimension}",
            reason="response_invalid",
            stage="embed",
        )
    vector = tuple(values)
    if any(not math.isfinite(value) for value in vector):
        raise ModelError("embedding response was invalid", reason="response_invalid", stage="embed")
    return vector


def _finish_answer(
    answer: str,
    retrieved: Sequence[SearchHit],
    grounded: tuple[SearchHit, ...],
    *,
    answer_policy: AnswerPolicy,
    answer_format: Literal["text", "structured"],
    source_metadata_key: str,
) -> AnswerResult:
    """Render admitted output with the same grounding and abstention policy for every caller."""
    forced = _forced_abstention(answer_policy, retrieved, grounded)
    if answer_format == "text":
        return _answer_result(answer, grounded, answer_policy=answer_policy, forced=forced)
    draft = parse_answer_draft(answer)
    result = render_answer_draft(draft, grounded, source_metadata_key=source_metadata_key)
    if draft.kind == "text":
        result = _answer_result(
            result.answer,
            result.hits,
            answer_policy=answer_policy,
            forced=result.abstention_reason or forced,
        )
    reason = result.abstention_reason or forced
    return replace(
        result,
        answer=prompts.UNKNOWN_ANSWER
        if reason is not None and answer_policy == "strict"
        else result.answer,
        abstained=reason is not None,
        abstention_reason=reason,
    )


def _answer_result(
    answer: str,
    hits: tuple[SearchHit, ...],
    *,
    answer_policy: AnswerPolicy = "strict",
    forced: AbstentionReason | None = None,
) -> AnswerResult:
    reason = _abstention_reason(answer) or forced
    if answer_policy == "best_effort":
        # Here the marker is a confidence flag above a real answer, so it is cut out of the prose
        # rather than replacing it. A model that emitted the marker alone declined anyway, and the
        # refusal sentence is all that is left to report.
        committed = _without_marker(answer)
        return AnswerResult(
            answer=committed or prompts.UNKNOWN_ANSWER,
            hits=hits,
            abstained=reason is not None,
            abstention_reason=reason,
        )
    return AnswerResult(
        # The marker is an instrument, not a sentence. Callers read `answer` to show or speak it,
        # so a refusal reports the prose it reported before the marker existed; `abstained` and
        # `abstention_reason` carry the machine-readable signal. Streaming yields raw deltas and
        # so still surfaces the token -- a consumer that streams should render on `abstained`.
        answer=prompts.UNKNOWN_ANSWER if reason is not None and _marker_in(answer) else answer,
        hits=hits,
        abstained=reason is not None,
        abstention_reason=reason,
    )


def _forced_abstention(
    answer_policy: AnswerPolicy,
    retrieved: Sequence[SearchHit],
    grounded: Sequence[SearchHit],
) -> AbstentionReason | None:
    """Report the insufficiency a `best_effort` guess was made under.

    Under `strict` an ungrounded question never reaches the model, so the reason is returned
    before the call. Under `best_effort` the model answers anyway, and the reason is known here
    from the same two sequences rather than from whether the model remembered its marker.
    """
    if answer_policy == "strict" or grounded:
        return None
    return (
        AbstentionReason.NO_EVIDENCE
        if not tuple(retrieved)
        else AbstentionReason.INSUFFICIENT_EVIDENCE
    )


def _without_marker(answer: str) -> str:
    """Drop the low-confidence marker from a committed answer, wherever the model put it."""
    committed = answer.replace(prompts._ABSTENTION_MARKER, "").strip()
    return (
        "" if _normalized_answer(committed) == prompts._ABSTENTION_MARKER.strip("[]") else committed
    )


def _marker_in(answer: str) -> bool:
    normalized = _normalized_answer(answer)
    return (
        prompts._ABSTENTION_MARKER in normalized
        or normalized == prompts._ABSTENTION_MARKER.strip("[]")
    )


def _abstention_reason(answer: str) -> AbstentionReason | None:
    """Report a refusal from the marker the grounded prompt requires, not from prose.

    The bracketed marker counts anywhere, because the brackets are what make it a machine token
    rather than a word. The bare token counts only as the entire answer: evidence itself can
    contain `insufficient_evidence` -- a runbook sentence, a logged status -- and quoting it in a
    real answer must not read as a refusal. The English sentinel is still honored for a model that
    ignores the marker, but only anchored at the start: a hedge inside a real answer is not a
    refusal, and over-reporting would corrupt the meter as badly as the exact-equality check it
    replaces.
    """
    normalized = _normalized_answer(answer)
    bare = prompts._ABSTENTION_MARKER.strip("[]")
    if prompts._ABSTENTION_MARKER in normalized or normalized == bare:
        return AbstentionReason.INSUFFICIENT_EVIDENCE
    unknown = _normalized_answer(prompts.UNKNOWN_ANSWER).rstrip(".")
    return AbstentionReason.INSUFFICIENT_EVIDENCE if normalized.startswith(unknown) else None


def _normalized_answer(answer: str) -> str:
    """Drop the formatting a model varies without changing what it said."""
    return " ".join(answer.replace("\u2019", "'").split()).casefold().strip("\"'*_ ")
