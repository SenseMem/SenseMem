"""The optional answer protocol validates JSON before exposing public answer text."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from datetime import datetime, timezone
from typing import Literal, cast

import httpx2 as httpx
import pytest
from test_openai_sdk import _sdk_client

from mindbridge import OpenAIModels
from mindbridge.exceptions import ModelError, ModelOutputTruncatedError, ValidationError
from mindbridge.models import openai_sdk as adapter
from mindbridge.models.base import ModelInput
from mindbridge.types import AnswerResult, SearchHit

NOW = datetime(2026, 4, 9, 11, 24, 35, tzinfo=timezone.utc)


def _hit(identifier: str, content: str, source: str) -> SearchHit:
    return SearchHit(
        id=identifier,
        content=content,
        score=0.8,
        created_at=NOW,
        occurred_at=NOW,
        metadata={"source_id": source, "document_ref": f"doc-{source}"},
    )


def _reply(content: str, *, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-structured-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
        },
    )


def _consume(stream: Generator[str, None, AnswerResult]) -> tuple[list[str], AnswerResult]:
    chunks: list[str] = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as done:
            return chunks, cast(AnswerResult, done.value)


@pytest.mark.parametrize("stream", [False, True])
def test_structured_budget_counts_the_schema_and_preserves_post_fit_source_aliases(
    stream: bool,
) -> None:
    hit = _hit("internal", "Visible source text.", "public-source")
    probe = OpenAIModels(
        generation_answer_format="structured", generation_request_max_bytes=100_000
    )
    prepared = probe._answer_request(ModelInput(text="Which source?"), (hit,), stream=stream)
    assert isinstance(prepared, tuple)
    exact = adapter._answer_wire_size(prepared[0], stream=stream)
    requests: list[bytes] = []
    envelope = '{"draft":{"kind":"source_ids","evidence_labels":["E1"]}}'

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.content)
        body = json.loads(request.content)
        assert body["response_format"]["json_schema"]["name"] == "answer"
        payload = json.loads(body["messages"][1]["content"])
        assert payload["hits"][0]["evidence_label"] == "E1"
        if stream:
            chunk = {
                "choices": [{"index": 0, "delta": {"content": envelope}, "finish_reason": "stop"}]
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return _reply(envelope)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        model = OpenAIModels(
            _sdk_client(transport),
            generation_answer_format="structured",
            generation_request_max_bytes=exact,
        )
        if stream:
            chunks, answer = _consume(model.stream_answer(ModelInput(text="Which source?"), (hit,)))
            assert chunks == ["public-source"]
        else:
            answer = model.answer(ModelInput(text="Which source?"), (hit,))
        assert answer.answer == "public-source"
        assert len(requests[0]) == exact
        too_small = OpenAIModels(
            _sdk_client(transport),
            generation_answer_format="structured",
            generation_request_max_bytes=exact - 1,
        )
        with pytest.raises(ModelError) as failure:
            if stream:
                _consume(too_small.stream_answer(ModelInput(text="Which source?"), (hit,)))
            else:
                too_small.answer(ModelInput(text="Which source?"), (hit,))
        assert failure.value.reason == "payload_too_large"
        assert len(requests) == 1


def test_structured_sources_use_post_fit_aliases_and_the_host_source_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fitted(
        question: ModelInput, hits: Sequence[SearchHit], **options: object
    ) -> tuple[SearchHit, ...]:
        return tuple(hits[1:])

    monkeypatch.setattr(adapter, "_fit_grounding_media", fitted)

    def respond(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], json.loads(request.content))
        calls.append(body)
        messages = cast(list[dict[str, object]], body["messages"])
        evidence = json.loads(cast(str, messages[1]["content"]))["hits"]
        assert len(evidence) == 1
        assert evidence[0]["evidence_label"] == "E1"
        assert evidence[0]["metadata"]["source_id"] == "survivor"
        return _reply('{"draft":{"kind":"source_ids","evidence_labels":["E1"]}}')

    removed = _hit("internal-removed", "Removed before transport.", "removed")
    survivor = _hit("internal-survivor", "Delivered evidence.", "survivor")
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        model = OpenAIModels(
            _sdk_client(transport),
            generation_answer_format="structured",
            generation_source_metadata_key="document_ref",
        )
        answer = model.answer(ModelInput(text="Which source?"), (removed, survivor))
    assert answer.answer == "doc-survivor"
    assert answer.hits == (survivor,)
    format_ = cast(dict[str, object], calls[0]["response_format"])
    assert format_["type"] == "json_schema"
    schema = cast(dict[str, object], format_["json_schema"])
    assert schema["strict"] is True
    assert schema["name"] == "answer"


@pytest.mark.parametrize("format_", ["json_object", "json_schema"])
def test_structured_numeric_answer_checks_visible_quote_not_only_evidence_id(
    format_: Literal["json_object", "json_schema"],
) -> None:
    reply = json.dumps(
        {
            "draft": {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "sum",
                "operands": [
                    {
                        "evidence_label": "E1",
                        "quote": "Fare: 999 GBP.",
                        "value": "999",
                        "unit": "GBP",
                    }
                ],
            }
        }
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: _reply(reply))) as transport:
        model = OpenAIModels(
            _sdk_client(transport),
            generation_answer_format="structured",
            generation_response_format=format_,
        )
        with pytest.raises(ModelError, match="visible evidence quote") as error:
            model.answer(
                ModelInput(text="What was the total?"), (_hit("fare", "Fare: 12 GBP.", "receipt"),)
            )
    assert error.value.reason == "response_invalid"
    assert "999" not in str(error.value)


@pytest.mark.parametrize(
    ("source", "draft"),
    [
        (
            source,
            {
                "format": "with_unit",
                "kind": "calculation",
                "operation": "scalar",
                "operands": [
                    {"evidence_label": "E1", "quote": "20 GBP", "value": "20", "unit": "GBP"}
                ],
            },
        )
        for source in ("Fare: 120 GBP.", "Fare: -20 GBP.")
    ]
    + [
        (
            "Visited Parkland.",
            {
                "kind": "calculation",
                "operation": "count_distinct",
                "members": [{"evidence_label": "E1", "quote": "Park", "key": "Park"}],
            },
        )
    ],
)
def test_structured_transport_rejects_quotes_that_split_original_source_tokens(
    source: str, draft: dict[str, object]
) -> None:
    reply = json.dumps({"draft": draft})
    with httpx.Client(transport=httpx.MockTransport(lambda request: _reply(reply))) as transport:
        model = OpenAIModels(_sdk_client(transport), generation_answer_format="structured")
        with pytest.raises(ModelError) as error:
            model.answer(
                ModelInput(text="What does the evidence establish?"),
                (_hit("source", source, "public"),),
            )
    assert error.value.reason == "response_invalid"
    assert error.value.stage == "generate"


def test_structured_date_uses_explicit_granularity_and_final_timestamps() -> None:
    reply = json.dumps(
        {
            "draft": {
                "format": "iso8601",
                "kind": "date",
                "source_value": NOW.isoformat(),
                "granularity": "day",
                "evidence": {
                    "evidence_label": "E1",
                    "quote": None,
                    "timestamp_field": "occurred_at",
                },
            }
        }
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: _reply(reply))) as transport:
        model = OpenAIModels(_sdk_client(transport), generation_answer_format="structured")
        assert (
            model.answer(
                ModelInput(text="On which date?"), (_hit("visit", "A visit.", "source"),)
            ).answer
            == "2026-04-09"
        )


@pytest.mark.parametrize(
    ("source_value", "granularity", "expected"),
    [
        ("2042-07-08T00:25:47+05:30", "minute", "2042-07-07 18:55"),
        ("2042-07-08T00:25:47+05:30", "second", "2042-07-07 18:55:47"),
        ("2042-07-08T16:25", "minute", None),
        ("2042-07-08T16:25Z", "day", None),
    ],
)
def test_structured_transport_utc_datetime_keeps_the_original_witness(
    source_value: str, granularity: str, expected: str | None
) -> None:
    reply = json.dumps(
        {
            "draft": {
                "kind": "date",
                "source_value": source_value,
                "granularity": granularity,
                "format": "utc_datetime",
                "evidence": {
                    "evidence_label": "E1",
                    "quote": source_value,
                    "timestamp_field": None,
                },
            }
        }
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: _reply(reply))) as transport:
        model = OpenAIModels(_sdk_client(transport), generation_answer_format="structured")
        hits = (_hit("boarding", f"Boarding: {source_value}.", "source"),)
        if expected is None:
            with pytest.raises(ModelError) as error:
                model.answer(ModelInput(text="Give UTC boarding time."), hits)
            assert error.value.reason == "response_invalid"
        else:
            assert model.answer(ModelInput(text="Give UTC boarding time."), hits).answer == expected


@pytest.mark.parametrize("duration", [False, True], ids=["numeric", "duration"])
def test_structured_transport_number_format_omits_only_the_rendered_unit(duration: bool) -> None:
    source = "Stay: 2026-03-01 through 2026-03-04." if duration else "Fare: 12.50 GBP."
    draft: dict[str, object] = {"kind": "calculation", "format": "number"}
    if duration:
        draft.update(
            operation="duration_days",
            interval_semantics="half_open",
            intervals=[
                {
                    "evidence_label": "E1",
                    "quote": source,
                    "start_date": "2026-03-01",
                    "end_date": "2026-03-04",
                }
            ],
        )
    else:
        draft.update(
            operation="scalar",
            operands=[{"evidence_label": "E1", "quote": source, "value": "12.50", "unit": "GBP"}],
        )
    reply = json.dumps({"draft": draft})
    with httpx.Client(transport=httpx.MockTransport(lambda request: _reply(reply))) as transport:
        model = OpenAIModels(_sdk_client(transport), generation_answer_format="structured")
        result = model.answer(
            ModelInput(text="Give the number only."), (_hit("source", source, "public"),)
        )
    assert result.answer == ("3" if duration else "12.5")


@pytest.mark.parametrize("policy", ["strict", "best_effort"])
def test_structured_abstention_keeps_existing_answer_policy(
    policy: Literal["strict", "best_effort"],
) -> None:
    reply = '{"draft":{"kind":"text","answer":"Probably amber.","evidence_labels":["E1"],"abstained":true}}'
    with httpx.Client(transport=httpx.MockTransport(lambda request: _reply(reply))) as transport:
        model = OpenAIModels(_sdk_client(transport), generation_answer_format="structured")
        result = model.answer(
            ModelInput(text="What color?"),
            (_hit("color", "Unclear color.", "photo"),),
            answer_policy=policy,
        )
    assert result.abstained
    assert result.answer == (adapter.UNKNOWN_ANSWER if policy == "strict" else "Probably amber.")


@pytest.mark.parametrize(
    ("answer", "explicit_abstention", "expected"),
    [
        ("[insufficient_evidence] Probably amber.", True, "Probably amber."),
        ("[insufficient_evidence] Probably amber.", False, "Probably amber."),
        ("[insufficient_evidence]", True, adapter.UNKNOWN_ANSWER),
    ],
)
def test_structured_best_effort_text_cleans_the_marker_without_losing_abstention(
    answer: str, explicit_abstention: bool, expected: str
) -> None:
    reply = json.dumps(
        {
            "draft": {
                "kind": "text",
                "answer": answer,
                "evidence_labels": ["E1"],
                "abstained": explicit_abstention,
            }
        }
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: _reply(reply))) as transport:
        model = OpenAIModels(_sdk_client(transport), generation_answer_format="structured")
        result = model.answer(
            ModelInput(text="What color?"),
            (_hit("color", "Unclear color.", "photo"),),
            answer_policy="best_effort",
        )
    assert result.answer == expected
    assert result.abstained
    assert result.abstention_reason is not None
    assert result.abstention_reason.value == "insufficient_evidence"


@pytest.mark.parametrize("ending", ["stop", "length", "invalid", "forged"])
def test_structured_stream_buffers_validates_and_closes_before_exposing_text(ending: str) -> None:
    reply = '{"draft":{"kind":"source_ids","evidence_labels":["E1"]}}'
    if ending == "invalid":
        reply = '{"draft":'
    elif ending == "forged":
        reply = '{"draft":{"kind":"source_ids","evidence_labels":["E2"]}}'

    class Body(httpx.SyncByteStream):
        closed = False

        def __iter__(self) -> Generator[bytes, None, None]:
            for text in (reply[:12], reply[12:]):
                chunk = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "reader",
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "reader",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "length" if ending == "length" else "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

        def close(self) -> None:
            self.closed = True

    body = Body()
    response = httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})
    with httpx.Client(transport=httpx.MockTransport(lambda request: response)) as transport:
        model = OpenAIModels(_sdk_client(transport), generation_answer_format="structured")
        stream = model.stream_answer(
            ModelInput(text="Which source?"), (_hit("internal", "Evidence.", "public-source"),)
        )
        if ending == "stop":
            assert next(stream) == "public-source"
            assert body.closed
            chunks, result = _consume(stream)
            assert chunks == []
            assert result.answer == "public-source"
        else:
            expected = ModelOutputTruncatedError if ending == "length" else ModelError
            with pytest.raises(expected):
                next(stream)
            assert body.closed


def test_default_text_answer_does_not_add_schema_or_reinterpret_json() -> None:
    calls: list[dict[str, object]] = []
    text = '{"draft":{"kind":"text","answer":"a literal JSON example"}}'

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return _reply(text)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        result = OpenAIModels(_sdk_client(transport)).answer(
            ModelInput(text="Show the text."), (_hit("example", "Evidence.", "source"),)
        )
    assert "response_format" not in calls[0]
    assert result.answer == text


@pytest.mark.parametrize("value", ["json", "", [], True])
def test_unknown_structured_format_is_rejected_before_transport(value: object) -> None:
    with pytest.raises(ValidationError, match="generation_answer_format"):
        OpenAIModels(generation_answer_format=cast(Literal["text", "structured"], value))
