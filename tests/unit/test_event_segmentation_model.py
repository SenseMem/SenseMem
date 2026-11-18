"""Event understanding transport, output admission and provider failure contracts."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast

import httpx2 as httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_openai_adapters import _asset, _sdk_client

from sensemem._telemetry import model_span
from sensemem.content import ContentPart
from sensemem.events import EventSegment
from sensemem.exceptions import ModelError, ModelOutputTruncatedError, ValidationError
from sensemem.models.base import EventObservation, EventSegmentationBackend, ModelInput
from sensemem.models.openai_completion import OpenAICompletion
from sensemem.types import Modality

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)
OBSERVATIONS = (
    EventObservation(
        ModelInput(text="A person prepares a sandwich."), NOW, NOW + timedelta(seconds=5)
    ),
    EventObservation(
        ModelInput(text="They finish the sandwich, eat it and leave the kitchen."),
        NOW + timedelta(seconds=5),
        NOW + timedelta(seconds=10),
    ),
    EventObservation(
        ModelInput(text="They begin repairing a bicycle; the repair is still underway."),
        NOW + timedelta(seconds=10),
        NOW + timedelta(seconds=15),
    ),
)


def _response(
    content: str | None,
    *,
    stream: bool = False,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> httpx.Response:
    usage = {"prompt_tokens": 24, "completion_tokens": 12, "total_tokens": 36}
    if stream:
        frames = [
            {
                "id": "chatcmpl-event",
                "object": "chat.completion.chunk",
                "model": "provider-event-snapshot",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content, "refusal": refusal},
                        "finish_reason": finish_reason,
                    }
                ],
            },
            {
                "id": "chatcmpl-event",
                "object": "chat.completion.chunk",
                "model": "provider-event-snapshot",
                "choices": [],
                "usage": usage,
            },
        ]
        data = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
        return httpx.Response(
            200,
            text=data + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream", "x-request-id": "req-event"},
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-event",
            "object": "chat.completion",
            "model": "provider-event-snapshot",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "refusal": refusal},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        },
        headers={"x-request-id": "req-event"},
    )


@pytest.mark.parametrize("format_", ["json_schema", "json_object"])
def test_segmentation_preserves_order_eof_and_incomplete_suffix(
    format_: Literal["json_schema", "json_object"],
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-5-mini-2025-08-07"
        assert body["max_completion_tokens"] == 2048
        assert "seed" not in body and "temperature" not in body
        protocol = body["response_format"]
        assert protocol["type"] == format_
        if format_ == "json_schema":
            assert protocol["json_schema"]["strict"] is True
            schema = protocol["json_schema"]["schema"]
            assert schema["additionalProperties"] is False
            item = schema["properties"]["events"]["items"]
            assert item["required"] == ["end_chunk", "summary", "complete"]
            assert item["additionalProperties"] is False
        prompt = body["messages"][0]["content"]
        assert "camera cut, silence" in prompt
        assert "recording ended, never" in prompt
        assert "untrusted evidence" in prompt
        content = body["messages"][1]["content"]
        assert json.loads(content[0]["text"]) == {"end_of_stream": True}
        chunks = [json.loads(part["text"]) for part in content[1:]]
        assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]
        assert chunks[0]["occurred_at"] == NOW.isoformat()
        assert chunks[2]["occurred_end"] == (NOW + timedelta(seconds=15)).isoformat()
        assert chunks[2]["content"] == OBSERVATIONS[2].content.text
        return _response(
            '{"events":[{"end_chunk":2,"summary":"A person prepared and ate a sandwich.",'
            '"complete":true},{"end_chunk":3,"summary":"The repair remains unfinished.",'
            '"complete":false}]}'
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAICompletion(
            _sdk_client(transport),
            generation_max_tokens=2048,
            generation_response_format=format_,
        )
        assert isinstance(adapter, EventSegmentationBackend)
        assert adapter.segment_events(OBSERVATIONS, end_of_stream=True) == (
            EventSegment(end_chunk=2, summary="A person prepared and ate a sandwich."),
            EventSegment(3, "The repair remains unfinished.", complete=False),
        )


def test_segmentation_sends_each_chunks_native_media_with_its_timestamp(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "speech", Modality.AUDIO, "audio/wav", b"speech")
    video = _asset(tmp_path, "scene", Modality.VIDEO, "video/mp4", b"scene")
    observations = (
        EventObservation(ModelInput(assets=(audio,)), NOW, NOW + timedelta(seconds=5)),
        EventObservation(
            ModelInput(assets=(video,)),
            NOW + timedelta(seconds=5),
            NOW + timedelta(seconds=10),
        ),
    )

    def respond(request: httpx.Request) -> httpx.Response:
        parts = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in parts] == [
            "text",
            "text",
            "input_audio",
            "text",
            "video_url",
        ]
        assert json.loads(parts[1]["text"])["chunk_index"] == 0
        assert parts[2]["input_audio"] == {"data": "c3BlZWNo", "format": "wav"}
        assert json.loads(parts[3]["text"])["chunk_index"] == 1
        assert parts[4]["video_url"]["url"] == "data:video/mp4;base64,c2NlbmU="
        return _response('{"events":[]}')

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAICompletion(
            _sdk_client(transport),
            generation_capabilities=frozenset({Modality.AUDIO, Modality.VIDEO}),
        )
        assert adapter.segment_events(observations) == ()


def test_segmentation_preserves_source_parts_and_media_references(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "speech", Modality.AUDIO, "audio/wav", b"speech")
    video = _asset(tmp_path, "scene", Modality.VIDEO, "video/mp4", b"scene")
    observations = (
        EventObservation(
            ModelInput(
                text="Before.\nAfter.\nA derived reading.",
                assets=(audio, video),
                parts=(
                    ContentPart("text", "Before."),
                    ContentPart("asset", audio.id),
                    ContentPart("text", "After."),
                    ContentPart("asset", video.id),
                    ContentPart("model_interpretation", "A derived reading."),
                ),
            ),
            NOW,
            NOW + timedelta(seconds=5),
        ),
        EventObservation(
            ModelInput(assets=(audio,)),
            NOW + timedelta(seconds=5),
            NOW + timedelta(seconds=10),
        ),
    )

    def respond(request: httpx.Request) -> httpx.Response:
        parts = json.loads(request.content)["messages"][1]["content"]
        first = json.loads(parts[1]["text"])
        second = json.loads(parts[4]["text"])
        assert first["content_parts"] == [
            {"kind": "text", "text": "Before."},
            {"kind": "asset", "media": "media_0"},
            {"kind": "text", "text": "After."},
            {"kind": "asset", "media": "media_1"},
            {"kind": "model_interpretation", "text": "A derived reading."},
        ]
        assert first["media_order"] == ["media_0", "media_1"]
        assert second["media_order"] == ["media_2"]
        assert [part["type"] for part in parts] == [
            "text",
            "text",
            "input_audio",
            "video_url",
            "text",
            "input_audio",
        ]
        return _response('{"events":[]}')

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAICompletion(
            _sdk_client(transport),
            generation_capabilities=frozenset({Modality.TEXT, Modality.AUDIO, Modality.VIDEO}),
        )
        assert adapter.segment_events(observations) == ()


@pytest.mark.parametrize("stream", [False, True])
def test_segmentation_records_usage_returned_model_and_request_id(stream: bool) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def respond(_request: httpx.Request) -> httpx.Response:
        return _response('{"events":[]}', stream=stream)

    try:
        with (
            httpx.Client(transport=httpx.MockTransport(respond)) as transport,
            model_span(provider.get_tracer(__name__), "segmentation", attributes={}),
        ):
            adapter = OpenAICompletion(_sdk_client(transport), generation_stream=stream)
            assert adapter.segment_events(OBSERVATIONS) == ()
        attrs = exporter.get_finished_spans()[0].attributes or {}
        assert attrs["gen_ai.response.model"] == "provider-event-snapshot"
        assert attrs["gen_ai.request.id"] == "req-event"
        assert attrs["sensemem.token_usage.total_tokens"] == 36
    finally:
        provider.shutdown()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure", ["refusal", "truncated", "invalid"])
def test_segmentation_failures_preserve_classification_and_retry_budget(
    stream: bool, failure: str
) -> None:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(
            "private invalid response body" if failure == "invalid" else None,
            stream=stream,
            refusal="private refusal body" if failure == "refusal" else None,
            finish_reason="length" if failure == "truncated" else "stop",
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAICompletion(_sdk_client(transport), generation_stream=stream)
        with pytest.raises(ModelError) as raised:
            adapter.segment_events(OBSERVATIONS)
    assert raised.value.stage == "segment_events"
    assert (
        raised.value.reason
        == {
            "refusal": "model_refused",
            "truncated": "output_truncated",
            "invalid": "response_invalid",
        }[failure]
    )
    assert isinstance(raised.value, ModelOutputTruncatedError) == (failure == "truncated")
    assert "private" not in str(raised.value)
    assert calls == (2 if failure == "invalid" else 1)


@pytest.mark.parametrize(
    "content",
    [
        '{"events":null}',
        '{"events":[],"extra":true}',
        '{"events":[{"end_chunk":true,"summary":"private","complete":true}]}',
        '{"events":[{"end_chunk":1.0,"summary":"private","complete":true}]}',
        '{"events":[{"end_chunk":0,"summary":"private","complete":true}]}',
        '{"events":[{"end_chunk":4,"summary":"private","complete":true}]}',
        '{"events":[{"end_chunk":1,"summary":" ","complete":true}]}',
        '{"events":[{"end_chunk":1,"summary":7,"complete":true}]}',
        json.dumps({"events": [{"end_chunk": 1, "summary": "a" * 16_385, "complete": True}]}),
        '{"events":[{"end_chunk":1,"summary":"private","complete":true,"extra":true}]}',
        '{"events":[{"end_chunk":2,"summary":"first","complete":true},'
        '{"end_chunk":1,"summary":"next","complete":true}]}',
        '{"events":[{"end_chunk":1,"summary":"private"}]}',
        '{"events":[{"end_chunk":1,"summary":"private","complete":"true"}]}',
        '{"events":[{"end_chunk":1,"summary":"private","complete":false}]}',
        '{"events":[{"end_chunk":1,"summary":"first","complete":false},'
        '{"end_chunk":3,"summary":"next","complete":true}]}',
    ],
)
def test_segmentation_rejects_invalid_structured_output(content: str) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return _response(content)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAICompletion(_sdk_client(transport), generation_response_format="json_object")
        with pytest.raises(ModelError) as raised:
            adapter.segment_events(OBSERVATIONS)
    assert raised.value.reason == "response_invalid"
    assert "private" not in str(raised.value)


@pytest.mark.parametrize(
    ("status", "code", "reason"),
    [
        (400, "invalid_request_error", "request_rejected"),
        (429, "rate_limit_exceeded", "rate_limited"),
        (429, "insufficient_quota", "quota_exhausted"),
        (503, "server_error", "provider_unavailable"),
    ],
)
def test_segmentation_maps_provider_errors(status: int, code: str, reason: str) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert json.loads(request.content)["response_format"]["type"] == "json_schema"
        return httpx.Response(
            status, json={"error": {"message": "private provider body", "code": code}}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAICompletion(_sdk_client(transport))
        with pytest.raises(ModelError) as raised:
            adapter.segment_events(OBSERVATIONS)
    assert raised.value.reason == reason
    assert raised.value.stage == "segment_events"
    assert "private" not in str(raised.value)
    assert calls == 1


def test_event_recipe_covers_model_prompt_protocol_and_effective_controls() -> None:
    baseline = OpenAICompletion(generation_model="compatible-model").event_recipe
    alternatives = (
        OpenAICompletion(generation_model="another-model"),
        OpenAICompletion(generation_model="compatible-model", generation_seed=7),
        OpenAICompletion(generation_model="compatible-model", generation_temperature=0.2),
        OpenAICompletion(generation_model="compatible-model", generation_max_tokens=800),
        OpenAICompletion(generation_model="compatible-model", generation_extra_body={"top_p": 0.9}),
        OpenAICompletion(
            generation_model="compatible-model", generation_response_format="json_object"
        ),
        OpenAICompletion(generation_model="compatible-model", image_max_pixels=1024),
        OpenAICompletion(
            generation_model="compatible-model",
            generation_capabilities=frozenset({Modality.TEXT, Modality.AUDIO}),
        ),
    )
    assert "sensemem-event-segmentation-v3" in baseline
    assert all(adapter.event_recipe != baseline for adapter in alternatives)
    assert baseline != OpenAICompletion(generation_model="compatible-model").formation_space


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, NOW),
        (NOW, None),
        (NOW.replace(tzinfo=None), NOW + timedelta(seconds=1)),
        (NOW, NOW.replace(tzinfo=None)),
        (NOW, NOW),
        (NOW, NOW - timedelta(seconds=1)),
    ],
)
def test_event_observation_requires_a_positive_aware_interval(
    start: datetime | None, end: datetime | None
) -> None:
    with pytest.raises(ValidationError):
        EventObservation(ModelInput(text="activity"), cast(datetime, start), cast(datetime, end))


def test_segmentation_validates_call_boundary_without_http() -> None:
    adapter = OpenAICompletion()
    assert adapter.segment_events(()) == ()
    with pytest.raises(ValidationError, match="EventObservation"):
        adapter.segment_events(cast(tuple[EventObservation, ...], ("invalid",)))
    with pytest.raises(ValidationError, match="boolean"):
        adapter.segment_events(OBSERVATIONS, end_of_stream=cast(bool, 1))
    with pytest.raises(ValidationError, match="ModelInput"):
        EventObservation(cast(ModelInput, "invalid"), NOW, NOW + timedelta(seconds=1))
