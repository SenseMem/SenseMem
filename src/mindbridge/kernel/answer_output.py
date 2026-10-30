"""Pure, evidence-bound rendering of an explicitly structured answer.

The wire envelope is ``{"draft": <variant>}``. Every listed field is required:

* text: kind, answer, evidence_labels, abstained.
* source_ids: kind, evidence_labels.
* date: kind, source_value, granularity, format, evidence. Evidence has evidence_label, quote,
  timestamp_field; exactly one of its last two fields is non-null.
* calculation scalar/sum/difference: kind, operation, format, operands. Each operand has
  evidence_label, quote, value, unit.
* calculation count_distinct: kind, operation, members. Each member has
  evidence_label, quote, key.
* calculation duration_days: kind, operation, format, interval_semantics, intervals.
  Each interval has evidence_label, quote, start_date, end_date.

Objects reject additional fields. Non-text lists contain 1..128 entries; text
evidence_labels may be empty. Aliases start at E1 in first-record order. This
boundary establishes literal support and performs arithmetic; fact relevance,
answer-set completeness and requested date granularity remain explicit model
decisions, never inferred from a question's wording or a benchmark label.
Omitted parser formats retain ISO dates and unit-bearing numbers; the wire schema
requires an explicit format. UTC presentation requires an explicitly zoned source timestamp.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, localcontext
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic import ValidationError as PydanticValidationError

from mindbridge.exceptions import ModelError
from mindbridge.types import AbstentionReason, AnswerResult, SearchHit

_Label = Annotated[str, StringConstraints(pattern=r"^E[1-9][0-9]*$")]
_Text = Annotated[str, StringConstraints(min_length=1)]
_Quote = Annotated[str, StringConstraints(min_length=1, max_length=1_000)]
_Number = Annotated[
    str,
    StringConstraints(
        pattern=r"^[+-]?(?:[0-9]{1,64}|[0-9]{1,3}(?:,[0-9]{3}){1,21})(?:\.[0-9]{1,32})?$"
    ),
]
_Labels = Annotated[tuple[_Label, ...], Field(max_length=128)]
_RequiredLabels = Annotated[tuple[_Label, ...], Field(min_length=1, max_length=128)]
DateGranularity: TypeAlias = Literal["year", "month", "day", "minute", "second"]


class _DraftValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TextDraft(_DraftValue):
    kind: Literal["text"]
    answer: _Text
    evidence_labels: _Labels
    abstained: bool


class SourceIdsDraft(_DraftValue):
    kind: Literal["source_ids"]
    evidence_labels: _RequiredLabels


class DateEvidence(_DraftValue):
    evidence_label: _Label
    quote: _Quote | None
    timestamp_field: Literal["occurred_at", "occurred_end", "created_at"] | None

    @model_validator(mode="after")
    def _one_witness(self) -> Self:
        if (self.quote is None) == (self.timestamp_field is None):
            raise ValueError("a date needs exactly one quote or visible timestamp witness")
        return self


class DateDraft(_DraftValue):
    kind: Literal["date"]
    source_value: _Text
    granularity: DateGranularity
    format: Literal["iso8601", "utc_datetime"] = "iso8601"
    evidence: DateEvidence


class NumericOperand(_DraftValue):
    evidence_label: _Label
    quote: _Quote
    value: _Number
    unit: Annotated[str, StringConstraints(min_length=1, max_length=64)]


class NumericDraft(_DraftValue):
    kind: Literal["calculation"]
    operation: Literal["scalar", "sum", "difference"]
    format: Literal["with_unit", "number"] = "with_unit"
    operands: Annotated[tuple[NumericOperand, ...], Field(min_length=1, max_length=128)]


class CountMember(_DraftValue):
    evidence_label: _Label
    quote: _Quote
    key: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class CountDraft(_DraftValue):
    kind: Literal["calculation"]
    operation: Literal["count_distinct"]
    members: Annotated[tuple[CountMember, ...], Field(min_length=1, max_length=128)]


class DateInterval(_DraftValue):
    evidence_label: _Label
    quote: _Quote
    start_date: _Text
    end_date: _Text


class DurationDraft(_DraftValue):
    kind: Literal["calculation"]
    operation: Literal["duration_days"]
    format: Literal["with_unit", "number"] = "with_unit"
    interval_semantics: Literal["half_open", "closed"]
    intervals: Annotated[tuple[DateInterval, ...], Field(min_length=1, max_length=128)]


AnswerDraft: TypeAlias = (
    TextDraft | SourceIdsDraft | DateDraft | NumericDraft | CountDraft | DurationDraft
)


class _Envelope(_DraftValue):
    draft: AnswerDraft


def _invalid(message: str) -> ModelError:
    return ModelError(message, reason="response_invalid", stage="generate")


def parse_answer_draft(payload: str) -> AnswerDraft:
    """Validate unknown model JSON once, without exposing it in an exception body."""
    try:
        return _Envelope.model_validate_json(payload).draft
    except PydanticValidationError:
        raise _invalid("generation model returned an invalid structured answer") from None


def evidence_aliases(hits: Sequence[SearchHit]) -> dict[str, str]:
    """Use the E1-based, first-occurrence rule of answer_evidence_payloads."""
    aliases: dict[str, str] = {}
    for hit in hits:
        if hit.id not in aliases:
            aliases[hit.id] = f"E{len(aliases) + 1}"
    return aliases


def _hit(label: str, visible: Mapping[str, SearchHit]) -> SearchHit:
    hit = visible.get(label)
    if hit is None:
        raise _invalid("structured answer references evidence outside the answer context")
    return hit


def _quote(label: str, quote: str, visible: Mapping[str, SearchHit]) -> tuple[SearchHit, int]:
    hit = _hit(label, visible)
    start = hit.content.find(quote)
    if start < 0 or hit.content.find(quote, start + 1) >= 0:
        raise _invalid("structured answer requires a uniquely located visible evidence quote")
    return hit, start


def _quoted_token(
    value: str, hit: SearchHit, start: int, quote: str, *, word_boundary: bool = False
) -> bool:
    # Lookarounds must see the original neighbours, which a shortened quote can hide.
    boundary = r"\w" if word_boundary else r"\d"
    pattern = f"(?<!{boundary})" + re.escape(value) + f"(?!{boundary})"
    return any(
        start <= match.start() and match.end() <= start + len(quote)
        for match in re.finditer(pattern, hit.content)
    )


def _number_spans(value: str, quote: str, unit: str) -> tuple[tuple[int, int], ...]:
    # A literal unit can touch a number (GBP12.50 or 1280m). Keep the numeric boundary
    # inside that optional unit too: a model cannot call the last digit or a minus sign a unit.
    pattern = (
        r"(?<![\w.,+\-\u2212])(?:"
        + re.escape(unit)
        + r"\s*)?(?<![0-9.,+\-\u2212])(?P<value>"
        + re.escape(value)
        + r")(?![0-9]|[.,][0-9])(?:\s*"
        + re.escape(unit)
        + r")?(?!\w|[.,][0-9])"
    )
    return tuple(match.span("value") for match in re.finditer(pattern, quote))


def _parsed_date(value: str) -> tuple[date | datetime, int]:
    """Parse only the ISO precision the source text actually supplied."""
    try:
        if re.fullmatch(r"\d{4}", value):
            parsed: date | datetime = date(int(value), 1, 1)
            precision = 0
        elif re.fullmatch(r"\d{4}-\d{2}", value):
            parsed = date.fromisoformat(value + "-01")
            precision = 1
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            parsed = date.fromisoformat(value)
            precision = 2
        else:
            if not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?",
                value,
            ):
                raise ValueError("ISO date and minute precision required")
            parsed = datetime.fromisoformat(value)
            precision = 4 if re.search(r"[T ]\d{2}:\d{2}:\d{2}", value) else 3
    except ValueError:
        raise _invalid("structured answer contains an invalid ISO date") from None
    return parsed, precision


def _date_value(
    value: str, granularity: DateGranularity, date_format: Literal["iso8601", "utc_datetime"]
) -> str:
    """Coarsen only to an explicitly requested granularity, never silently add time."""
    parsed, precision = _parsed_date(value)
    precisions = {"year": 0, "month": 1, "day": 2, "minute": 3, "second": 4}
    if precisions[granularity] > precision:
        raise _invalid("structured answer requests unsupported date precision")
    if date_format == "utc_datetime":
        if granularity not in ("minute", "second"):
            raise _invalid("UTC datetime format requires minute or second granularity")
        assert isinstance(parsed, datetime)
        if parsed.utcoffset() is None:
            raise _invalid("UTC datetime format requires an explicitly zoned source timestamp")
        try:
            utc = parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            raise _invalid("structured UTC datetime is outside the supported calendar") from None
        return utc.isoformat(sep=" ", timespec=granularity + "s").removesuffix("+00:00")
    if granularity == "year":
        return f"{parsed.year:04d}"
    if granularity == "month":
        return f"{parsed.year:04d}-{parsed.month:02d}"
    if granularity == "day":
        return f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}"
    assert isinstance(parsed, datetime)
    return parsed.isoformat(timespec="minutes" if granularity == "minute" else "seconds")


def _render_date(draft: DateDraft, visible: Mapping[str, SearchHit]) -> str:
    witness = draft.evidence
    if witness.quote is not None:
        hit, start = _quote(witness.evidence_label, witness.quote, visible)
        if not _quoted_token(draft.source_value, hit, start, witness.quote):
            raise _invalid("structured date is not present in its evidence quote")
    else:
        hit = _hit(witness.evidence_label, visible)
        field = witness.timestamp_field
        moment = (
            hit.occurred_at
            if field == "occurred_at"
            else hit.occurred_end
            if field == "occurred_end"
            else hit.created_at
            if hit.occurred_at is None
            else None
        )
        if moment is None or draft.source_value != moment.isoformat():
            raise _invalid("structured date does not match a visible evidence timestamp")
    return _date_value(draft.source_value, draft.granularity, draft.format)


def _calculate_numeric(draft: NumericDraft, visible: Mapping[str, SearchHit]) -> str:
    if (draft.operation == "scalar" and len(draft.operands) != 1) or (
        draft.operation == "difference" and len(draft.operands) != 2
    ):
        raise _invalid("structured calculation has the wrong operand count")
    values: list[Decimal] = []
    used: set[tuple[str, int]] = set()
    unit = draft.operands[0].unit
    for operand in draft.operands:
        hit, start = _quote(operand.evidence_label, operand.quote, visible)
        spans = _number_spans(operand.value, operand.quote, operand.unit)
        if len(spans) != 1 or operand.unit != unit:
            raise _invalid("structured calculation requires unambiguous values and equal units")
        begin, end = spans[0]
        absolute_span = start + begin, start + end
        if absolute_span not in _number_spans(operand.value, hit.content, operand.unit):
            raise _invalid("structured calculation requires a complete visible source value")
        for text, left, right in (
            (operand.quote, begin, end),
            (hit.content, *absolute_span),
        ):
            before, after = text[:left].rstrip(), text[right:].lstrip()
            if not (
                (
                    after.startswith(unit)
                    and (
                        len(after) == len(unit)
                        or not (after[len(unit)].isalnum() or after[len(unit)] == "_")
                    )
                )
                or (
                    before.endswith(unit)
                    and (
                        len(before) == len(unit)
                        or not (before[-len(unit) - 1].isalnum() or before[-len(unit) - 1] == "_")
                    )
                )
            ):
                raise _invalid("structured calculation unit is not attached to its quoted value")
        position = (hit.id, absolute_span[0])
        if position in used:
            raise _invalid("structured calculation repeats one evidence value")
        used.add(position)
        values.append(Decimal(operand.value.replace(",", "")))
    with localcontext() as context:
        context.prec = 128
        value = (
            values[0] - values[1] if draft.operation == "difference" else sum(values, Decimal(0))
        )
    number = format(value, "f")
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number if draft.format == "number" else f"{number} {unit}"


def _count_distinct(draft: CountDraft, visible: Mapping[str, SearchHit]) -> str:
    keys: set[str] = set()
    for member in draft.members:
        hit, start = _quote(member.evidence_label, member.quote, visible)
        if not _quoted_token(member.key, hit, start, member.quote, word_boundary=True):
            raise _invalid("structured count key is not present in its evidence quote")
        keys.add(member.key)
    return str(len(keys))


def _duration_days(draft: DurationDraft, visible: Mapping[str, SearchHit]) -> str:
    intervals: list[tuple[date, date]] = []
    for interval in draft.intervals:
        hit, start = _quote(interval.evidence_label, interval.quote, visible)
        if not all(
            re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
            and _quoted_token(value, hit, start, interval.quote)
            for value in (interval.start_date, interval.end_date)
        ):
            raise _invalid("structured interval dates require literal visible evidence")
        try:
            start_date = date.fromisoformat(interval.start_date)
            end_date = date.fromisoformat(interval.end_date)
            if end_date < start_date:
                raise _invalid("structured interval is reversed")
            if draft.interval_semantics == "closed":
                end_date += timedelta(days=1)
        except (ValueError, OverflowError):
            raise _invalid("structured interval contains an invalid date") from None
        intervals.append((start_date, end_date))
    days = 0
    until: date | None = None
    for start_date, end_date in sorted(intervals):
        days += max(0, (end_date - max(start_date, until or start_date)).days)
        until = max(until or end_date, end_date)
    return str(days) if draft.format == "number" else f"{days} days"


def render_answer_draft(
    draft: AnswerDraft,
    hits: Sequence[SearchHit],
    *,
    source_metadata_key: str = "source_id",
) -> AnswerResult:
    """Bind aliases to the actual final context and render only witnessed values.

    Call with the adapter's final, fitted hits, not its pre-packing candidates.
    The caller chooses the application source metadata field. Ordinary text is
    not reinterpreted as IDs, numbers, dates, executable expressions or policy.
    """
    aliases = evidence_aliases(hits)
    visible = {aliases[hit.id]: hit for hit in reversed(hits)}
    selected: list[str] = []
    abstained = False
    if isinstance(draft, TextDraft):
        selected.extend(draft.evidence_labels)
        answer = draft.answer.strip()
        if not answer:
            raise _invalid("structured answer text must not be empty")
        abstained = draft.abstained
    elif isinstance(draft, SourceIdsDraft):
        selected.extend(draft.evidence_labels)
        source_ids: list[str] = []
        for label in selected:
            source = _hit(label, visible).metadata.get(source_metadata_key)
            if not isinstance(source, str) or not source.strip():
                raise _invalid("structured source answer requires application source metadata")
            if source not in source_ids:
                source_ids.append(source)
        answer = ", ".join(source_ids)
    elif isinstance(draft, DateDraft):
        selected.append(draft.evidence.evidence_label)
        answer = _render_date(draft, visible)
    elif isinstance(draft, NumericDraft):
        selected.extend(operand.evidence_label for operand in draft.operands)
        answer = _calculate_numeric(draft, visible)
    elif isinstance(draft, CountDraft):
        selected.extend(member.evidence_label for member in draft.members)
        answer = _count_distinct(draft, visible)
    else:
        selected.extend(interval.evidence_label for interval in draft.intervals)
        answer = _duration_days(draft, visible)
    used_hits = tuple(_hit(label, visible) for label in dict.fromkeys(selected))
    return AnswerResult(
        answer=answer,
        hits=used_hits,
        abstained=abstained,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE if abstained else None,
    )
