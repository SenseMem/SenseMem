"""Structured answers bind to the delivered evidence, not merely a known source ID."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pytest

from sensemem.exceptions import ModelError
from sensemem.models._answer_output import (
    evidence_aliases,
    parse_answer_draft,
    render_answer_draft,
)
from sensemem.types import AbstentionReason, AnswerResult, SearchHit

NOW = datetime(2026, 2, 14, 16, 35, 47, tzinfo=timezone.utc)


def _hit(
    identifier: str,
    content: str,
    *,
    metadata: Mapping[str, object] | None = None,
    occurred_at: datetime | None = None,
) -> SearchHit:
    return SearchHit(
        id=identifier,
        content=content,
        score=0.8,
        created_at=NOW,
        occurred_at=occurred_at,
        metadata={} if metadata is None else metadata,
    )


def _render(draft: Mapping[str, object], hits: Sequence[SearchHit]) -> AnswerResult:
    return render_answer_draft(parse_answer_draft(json.dumps({"draft": draft})), hits)


def _operand(label: str, quote: str, value: str, unit: str = "GBP") -> dict[str, str]:
    return {"evidence_label": label, "quote": quote, "value": value, "unit": unit}


def test_source_answers_bind_first_occurrence_aliases_to_application_metadata() -> None:
    first = _hit("internal-one", "A picture.", metadata={"source_id": "photo-alpha"})
    second = _hit("internal-two", "Another picture.", metadata={"source_id": "photo-beta"})
    assert evidence_aliases((first, first, second)) == {
        "internal-one": "E1",
        "internal-two": "E2",
    }
    answer = _render(
        {"kind": "source_ids", "evidence_labels": ["E2", "E1", "E2"]},
        (first, first, second),
    )
    assert answer.answer == "photo-beta, photo-alpha"
    assert answer.hits == (second, first)


@pytest.mark.parametrize("label", ["E3", "S1", "attachment 1", "internal-one", "photo-alpha"])
def test_source_answers_reject_unknown_or_non_evidence_aliases(label: str) -> None:
    with pytest.raises(ModelError) as error:
        _render(
            {"kind": "source_ids", "evidence_labels": [label]},
            (_hit("internal-one", "A picture.", metadata={"source_id": "photo-alpha"}),),
        )
    assert error.value.reason == "response_invalid"
    assert error.value.stage == "generate"


def test_source_answer_cannot_fall_back_to_internal_or_attachment_identifiers() -> None:
    with pytest.raises(ModelError, match="application source metadata"):
        _render({"kind": "source_ids", "evidence_labels": ["E1"]}, (_hit("record", "Source."),))


def test_source_field_is_chosen_by_host_and_duplicate_application_sources_coalesce() -> None:
    hits = (
        _hit("one", "Source.", metadata={"document_key": "public"}),
        _hit("two", "Another source.", metadata={"document_key": "public"}),
    )
    draft = parse_answer_draft('{"draft":{"kind":"source_ids","evidence_labels":["E1","E2"]}}')
    assert render_answer_draft(draft, hits, source_metadata_key="document_key").answer == "public"


def test_text_is_not_reinterpreted_as_a_date_or_executable_expression() -> None:
    answer = _render(
        {
            "kind": "text",
            "answer": "2026-02-14T16:35:47+00:00; 5 + 6",
            "evidence_labels": [],
            "abstained": False,
        },
        (),
    )
    assert answer.answer == "2026-02-14T16:35:47+00:00; 5 + 6"


def test_explicit_text_abstention_preserves_the_answer_result_contract() -> None:
    answer = _render(
        {
            "kind": "text",
            "answer": "Insufficient evidence.",
            "evidence_labels": [],
            "abstained": True,
        },
        (),
    )
    assert answer.abstained
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    ("granularity", "expected"),
    [
        ("year", "2026"),
        ("month", "2026-02"),
        ("day", "2026-02-14"),
        ("minute", "2026-02-14T16:35+00:00"),
        ("second", "2026-02-14T16:35:47+00:00"),
    ],
)
def test_date_precision_is_explicit_and_preserves_requested_time(
    granularity: str, expected: str
) -> None:
    answer = _render(
        {
            "format": "iso8601",
            "kind": "date",
            "source_value": NOW.isoformat(),
            "granularity": granularity,
            "evidence": {"evidence_label": "E1", "quote": None, "timestamp_field": "occurred_at"},
        },
        (_hit("event", "A visit.", occurred_at=NOW),),
    )
    assert answer.answer == expected


def test_date_requires_a_visible_quote_or_displayed_timestamp() -> None:
    hit = _hit("event", "The inspection is on 2026-03-02.", occurred_at=NOW)
    draft: dict[str, object] = {
        "format": "iso8601",
        "kind": "date",
        "source_value": "2026-03-02",
        "granularity": "day",
        "evidence": {
            "evidence_label": "E1",
            "quote": "The inspection is on 2026-03-02.",
            "timestamp_field": None,
        },
    }
    assert _render(draft, (hit,)).answer == "2026-03-02"
    draft["source_value"] = "2026-03-03"
    with pytest.raises(ModelError, match="not present"):
        _render(draft, (hit,))
    draft["source_value"] = NOW.isoformat()
    draft["evidence"] = {"evidence_label": "E1", "quote": None, "timestamp_field": "created_at"}
    with pytest.raises(ModelError, match="visible evidence timestamp"):
        _render(draft, (hit,))


@pytest.mark.parametrize("value", ["2026", "2026-02", "2026-02-14", "2026-02-14T16"])
def test_date_does_not_invent_finer_precision(value: str) -> None:
    quote = f"Recorded date: {value}."
    with pytest.raises(ModelError):
        _render(
            {
                "format": "iso8601",
                "kind": "date",
                "source_value": value,
                "granularity": "second",
                "evidence": {"evidence_label": "E1", "quote": quote, "timestamp_field": None},
            },
            (_hit("date", quote),),
        )


@pytest.mark.parametrize(
    ("source_value", "granularity", "expected"),
    [
        ("2042-07-08T00:25:47+05:30", "minute", "2042-07-07 18:55"),
        ("2042-07-08T00:25:47+05:30", "second", "2042-07-07 18:55:47"),
        ("2042-07-08T23:25:47-07:00", "minute", "2042-07-09 06:25"),
        ("2042-07-08T16:25Z", "minute", "2042-07-08 16:25"),
    ],
)
def test_explicit_utc_datetime_format_converts_the_witnessed_source_value(
    source_value: str, granularity: str, expected: str
) -> None:
    quote = f"Boarding: {source_value}."
    result = _render(
        {
            "kind": "date",
            "source_value": source_value,
            "granularity": granularity,
            "format": "utc_datetime",
            "evidence": {"evidence_label": "E1", "quote": quote, "timestamp_field": None},
        },
        (_hit("boarding", quote),),
    )
    assert result.answer == expected


def test_utc_datetime_format_does_not_guess_a_missing_timezone() -> None:
    source_value = "2042-07-08T16:25:00"
    with pytest.raises(ModelError, match="explicitly zoned"):
        _render(
            {
                "kind": "date",
                "source_value": source_value,
                "granularity": "minute",
                "format": "utc_datetime",
                "evidence": {
                    "evidence_label": "E1",
                    "quote": source_value,
                    "timestamp_field": None,
                },
            },
            (_hit("boarding", source_value),),
        )


@pytest.mark.parametrize("granularity", ["year", "month", "day"])
def test_utc_datetime_format_rejects_coarse_granularity(granularity: str) -> None:
    with pytest.raises(ModelError, match="minute or second"):
        _render(
            {
                "kind": "date",
                "source_value": NOW.isoformat(),
                "granularity": granularity,
                "format": "utc_datetime",
                "evidence": {
                    "evidence_label": "E1",
                    "quote": None,
                    "timestamp_field": "occurred_at",
                },
            },
            (_hit("boarding", "Boarding.", occurred_at=NOW),),
        )


def test_preformatted_date_answer_cannot_replace_the_source_value() -> None:
    source_value = "2042-07-08T16:25:00+00:00"
    with pytest.raises(ModelError, match="not present"):
        _render(
            {
                "kind": "date",
                "source_value": "2042-07-08 16:25",
                "granularity": "minute",
                "format": "utc_datetime",
                "evidence": {
                    "evidence_label": "E1",
                    "quote": source_value,
                    "timestamp_field": None,
                },
            },
            (_hit("boarding", source_value),),
        )


def test_numeric_number_format_only_changes_the_rendered_unit() -> None:
    quote = "Balance: GBP-20.50."
    result = _render(
        {
            "kind": "calculation",
            "operation": "scalar",
            "format": "number",
            "operands": [_operand("E1", quote, "-20.50")],
        },
        (_hit("balance", quote),),
    )
    assert result.answer == "-20.5"


@pytest.mark.parametrize(
    ("source", "quote"), [("Fare: 20 USD.", "20 USD"), ("Fare: 20 GBPXYZ.", "20 GBP")]
)
def test_numeric_number_format_still_requires_the_complete_quoted_source_unit(
    source: str, quote: str
) -> None:
    with pytest.raises(ModelError):
        _render(
            {
                "kind": "calculation",
                "operation": "scalar",
                "format": "number",
                "operands": [_operand("E1", quote, "20")],
            },
            (_hit("fare", source),),
        )


def test_decimal_sum_and_difference_use_exact_supported_money_and_units() -> None:
    hits = (
        _hit("fare", "The fare was 1,200.10 GBP."),
        _hit("tax", "Tax: GBP 0.20."),
    )
    operands = [
        _operand("E1", hits[0].content, "1,200.10"),
        _operand("E2", hits[1].content, "0.20"),
    ]
    assert (
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "sum",
                "operands": operands,
            },
            hits,
        ).answer
        == "1200.3 GBP"
    )
    assert (
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "difference",
                "operands": operands,
            },
            hits,
        ).answer
        == "1199.9 GBP"
    )


@pytest.mark.parametrize(
    ("quote", "value", "unit", "expected"),
    [
        ("Fare: 12.50GBP.", "12.50", "GBP", "12.5 GBP"),
        ("Fare: GBP12.50.", "12.50", "GBP", "12.5 GBP"),
        ("Span: 1280m.", "1280", "m", "1280 m"),
        ("Balance: GBP-20.50.", "-20.50", "GBP", "-20.5 GBP"),
        ("Change: +20GBP.", "+20", "GBP", "20 GBP"),
    ],
)
def test_numeric_units_can_touch_the_complete_literal_value(
    quote: str, value: str, unit: str, expected: str
) -> None:
    result = _render(
        {
            "format": "with_unit",
            "kind": "calculation",
            "operation": "scalar",
            "operands": [_operand("E1", quote, value, unit)],
        },
        (_hit("receipt", quote),),
    )
    assert result.answer == expected


@pytest.mark.parametrize(
    ("quote", "value", "unit"),
    [
        ("Fare: 19 GBP.", "999", "GBP"),
        ("Fare: 19 GBP.", "9", "GBP"),
        ("Fare: -19 GBP.", "19", "GBP"),
        ("Fare: 19 GBP and tax 20 USD.", "20", "GBP"),
        ("Fare: 19 GBP.", "19", "USD"),
        ("Fare: 19 GBPX.", "19", "GBP"),
        ("Fare: 200GBP.", "20", "GBP"),
        ("Fare: 20.50GBP.", "20", "GBP"),
        ("Fare: 20,500GBP.", "20", "GBP"),
        ("Fare: -20GBP.", "20", "GBP"),
        ("Fare: GBP-20.", "20", "GBP-"),
        ("Fare: 200.", "20", "0"),
        ("Fare: 120.", "20", "1"),
        ("Code: x20GBP.", "20", "GBP"),
        ("Code: GBP20x.", "20", "GBP"),
    ],
)
def test_numeric_values_need_actual_quote_support_and_their_own_unit(
    quote: str, value: str, unit: str
) -> None:
    with pytest.raises(ModelError):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "scalar",
                "operands": [_operand("E1", quote, value, unit)],
            },
            (_hit("receipt", quote),),
        )


@pytest.mark.parametrize(
    ("source", "quote", "value", "unit"),
    [
        ("Fare: 120 GBP.", "20 GBP", "20", "GBP"),
        ("Fare: -20 GBP.", "20 GBP", "20", "GBP"),
        ("Fare: +20 GBP.", "20 GBP", "20", "GBP"),
        ("Fare: \u221220 GBP.", "20 GBP", "20", "GBP"),
        ("Fare: 20.50 GBP.", "50 GBP", "50", "GBP"),
        ("Fare: 1,200 GBP.", "200 GBP", "200", "GBP"),
        ("Fare: 20 GBPXYZ.", "20 GBP", "20", "GBP"),
        ("Fare: XYZGBP20.", "GBP20", "20", "GBP"),
    ],
)
def test_shortened_numeric_quotes_cannot_hide_original_token_boundaries(
    source: str, quote: str, value: str, unit: str
) -> None:
    with pytest.raises(ModelError):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "scalar",
                "operands": [_operand("E1", quote, value, unit)],
            },
            (_hit("receipt", source),),
        )


def test_shortened_quote_of_a_complete_numeric_token_remains_valid() -> None:
    result = _render(
        {
            "format": "with_unit",
            "kind": "calculation",
            "operation": "scalar",
            "operands": [_operand("E1", "GBP-20.50", "-20.50")],
        },
        (_hit("receipt", "Current balance: GBP-20.50, after the refund."),),
    )
    assert result.answer == "-20.5 GBP"


def test_known_id_is_not_sufficient_proof_and_invalid_results_have_no_fallback() -> None:
    with pytest.raises(ModelError, match="visible evidence quote"):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "scalar",
                "operands": [_operand("E1", "The total is 999 GBP.", "999")],
            },
            (_hit("known", "The total is 19 GBP."),),
        )


def test_requoting_the_same_value_cannot_double_count_it() -> None:
    quote = "The fare is 25 GBP."
    with pytest.raises(ModelError, match="repeats one evidence value"):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "sum",
                "operands": [_operand("E1", quote, "25"), _operand("E1", "25 GBP", "25")],
            },
            (_hit("receipt", quote),),
        )


def test_mixed_units_and_ambiguous_quote_occurrences_fail() -> None:
    hits = (_hit("a", "Total: 25 GBP."), _hit("b", "Tax: 5 USD."))
    with pytest.raises(ModelError, match="equal units"):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "sum",
                "operands": [
                    _operand("E1", hits[0].content, "25"),
                    _operand("E2", hits[1].content, "5", "USD"),
                ],
            },
            hits,
        )
    with pytest.raises(ModelError, match="uniquely located"):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "scalar",
                "operands": [_operand("E1", "25 GBP", "25")],
            },
            (_hit("a", "25 GBP deposit; 25 GBP later."),),
        )


def test_count_distinct_requires_quoted_keys_and_counts_exact_keys() -> None:
    quote = "Observed badges: alder, birch, alder."
    members = [
        {"evidence_label": "E1", "quote": quote, "key": key} for key in ("alder", "birch", "alder")
    ]
    assert (
        _render(
            {"kind": "calculation", "operation": "count_distinct", "members": members},
            (_hit("badges", quote),),
        ).answer
        == "2"
    )
    members.append({"evidence_label": "E1", "quote": quote, "key": "cedar"})
    with pytest.raises(ModelError, match="not present"):
        _render(
            {"kind": "calculation", "operation": "count_distinct", "members": members},
            (_hit("badges", quote),),
        )


@pytest.mark.parametrize("source", ["Visited Parkland.", "Visited StonePark."])
def test_shortened_count_quote_cannot_split_a_source_word(source: str) -> None:
    with pytest.raises(ModelError, match="not present"):
        _render(
            {
                "kind": "calculation",
                "operation": "count_distinct",
                "members": [{"evidence_label": "E1", "quote": "Park", "key": "Park"}],
            },
            (_hit("visit", source),),
        )


@pytest.mark.parametrize("source", ["Date: 12026-03-01.", "Date: 2026-03-010."])
def test_shortened_date_quote_cannot_hide_an_adjacent_source_digit(source: str) -> None:
    with pytest.raises(ModelError, match="not present"):
        _render(
            {
                "format": "iso8601",
                "kind": "date",
                "source_value": "2026-03-01",
                "granularity": "day",
                "evidence": {
                    "evidence_label": "E1",
                    "quote": "2026-03-01",
                    "timestamp_field": None,
                },
            },
            (_hit("date", source),),
        )


@pytest.mark.parametrize(
    "source",
    ["Stay: 12026-03-01 through 2026-03-05.", "Stay: 2026-03-01 through 2026-03-050."],
)
def test_shortened_interval_quote_cannot_hide_an_adjacent_source_digit(source: str) -> None:
    with pytest.raises(ModelError, match="literal visible evidence"):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "duration_days",
                "interval_semantics": "half_open",
                "intervals": [
                    {
                        "evidence_label": "E1",
                        "quote": "2026-03-01 through 2026-03-05",
                        "start_date": "2026-03-01",
                        "end_date": "2026-03-05",
                    }
                ],
            },
            (_hit("stay", source),),
        )


@pytest.mark.parametrize(("semantics", "expected"), [("half_open", "5 days"), ("closed", "6 days")])
def test_interval_union_uses_explicit_endpoint_semantics(semantics: str, expected: str) -> None:
    quotes = ("Stay: 2026-03-01 through 2026-03-04.", "Stay: 2026-03-03 through 2026-03-06.")
    intervals = [
        {"evidence_label": f"E{i + 1}", "quote": quote, "start_date": start, "end_date": end}
        for i, (quote, start, end) in enumerate(
            zip(quotes, ("2026-03-01", "2026-03-03"), ("2026-03-04", "2026-03-06"), strict=True)
        )
    ]
    assert (
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "duration_days",
                "interval_semantics": semantics,
                "intervals": intervals,
            },
            tuple(_hit(str(i), quote) for i, quote in enumerate(quotes)),
        ).answer
        == expected
    )


def test_duration_number_format_returns_the_validated_union_as_an_integer() -> None:
    quote = "Stay: 2026-03-01 through 2026-03-04."
    result = _render(
        {
            "kind": "calculation",
            "operation": "duration_days",
            "format": "number",
            "interval_semantics": "half_open",
            "intervals": [
                {
                    "evidence_label": "E1",
                    "quote": quote,
                    "start_date": "2026-03-01",
                    "end_date": "2026-03-04",
                }
            ],
        },
        (_hit("stay", quote),),
    )
    assert result.answer == "3"


@pytest.mark.parametrize("semantics", ["half_open", "closed"])
def test_interval_rejects_reversed_or_unwitnessed_dates(semantics: str) -> None:
    quote = "Dates: 2026-03-02 and 2026-03-01."
    with pytest.raises(ModelError, match="reversed"):
        _render(
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "duration_days",
                "interval_semantics": semantics,
                "intervals": [
                    {
                        "evidence_label": "E1",
                        "quote": quote,
                        "start_date": "2026-03-02",
                        "end_date": "2026-03-01",
                    }
                ],
            },
            (_hit("stay", quote),),
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"draft":{"kind":"date","source_value":"2026-03-01","granularity":"day","evidence":{"evidence_label":"E1","quote":"2026-03-01","timestamp_field":null}}}',
        '{"draft":{"kind":"calculation","operation":"scalar","operands":[{"evidence_label":"E1","quote":"12 GBP","value":"12","unit":"GBP"}]}}',
        '{"draft":{"kind":"calculation","operation":"duration_days","interval_semantics":"closed","intervals":[{"evidence_label":"E1","quote":"2026-03-01 to 2026-03-02","start_date":"2026-03-01","end_date":"2026-03-02"}]}}',
        '{"draft":{"kind":"source_ids","evidence_labels":["E1"],"answer":"invented"}}',
        '{"draft":{"kind":"text","answer":"x","evidence_labels":[],"abstained":"false"}}',
        '{"draft":{"kind":"calculation","operation":"sum","operands":[]}}',
        '{"draft":{"kind":"calculation","operation":"scalar","operands":[{"evidence_label":"E1","quote":"NaN GBP","value":"NaN","unit":"GBP"}]}}',
        '{"draft":{"kind":"source_ids","evidence_labels":["E1"]},"policy":"override"}',
        '{"draft":{"kind":"date","value":"2026-03-01","granularity":"day","evidence":{"evidence_label":"E1","quote":"2026-03-01","timestamp_field":null}}}',
        '{"draft":{"kind":"date","source_value":"2026-03-01","granularity":"day","format":"number","evidence":{"evidence_label":"E1","quote":"2026-03-01","timestamp_field":null}}}',
    ],
)
def test_unknown_model_values_fail_at_the_parse_boundary(payload: str) -> None:
    with pytest.raises(ModelError) as error:
        parse_answer_draft(payload)
    assert payload not in str(error.value)
