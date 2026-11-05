"""Bounded native media diagnosis through the public OpenAI adapter."""

import json
from dataclasses import replace
from pathlib import Path

import httpx2 as httpx
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_openai_sdk import _asset, _sdk_client, _search_hit

from sensemem import (
    AbstentionReason,
    EvidenceGeneration,
    ModelError,
    ModelInput,
    OpenAIModels,
    ValidationError,
)
from sensemem._telemetry import MODEL_REQUEST_COUNT, TOKEN_TOTAL, model_span, operation_span
from sensemem.types import EvidenceBasis, MemoryContext, MemoryKind, Modality


def _selected(
    client: OpenAI, *, generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT})
) -> EvidenceGeneration:
    model = OpenAIModels(client, generation_capabilities=generation_capabilities)
    return EvidenceGeneration(model, selector=model, media_max_items=1)


@pytest.mark.parametrize("stream", (False, True))
def test_unselected_media_only_evidence_remains_insufficient(tmp_path: Path, stream: bool) -> None:
    hit = replace(
        _search_hit(),
        content="",
        modality=Modality.IMAGE,
        assets=(_asset(tmp_path, "source", Modality.IMAGE, "image/png", b"png"),),
    )

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"content": '{"requests":[]}'}, "finish_reason": "stop"}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _selected(
            _sdk_client(client), generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        )
        if stream:
            result_stream = model.stream_answer(ModelInput(text="what is shown?"), (hit,))
            while True:
                try:
                    next(result_stream)
                except StopIteration as complete:
                    result = complete.value
                    break
        else:
            result = model.answer(ModelInput(text="what is shown?"), (hit,))
    assert result.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_media_selection_preserves_an_assertion_only_with_its_source(
    tmp_path: Path, stream: bool, selected: bool
) -> None:
    source = replace(
        _search_hit(),
        id="source",
        content="",
        modality=Modality.IMAGE,
        assets=(_asset(tmp_path, "source", Modality.IMAGE, "image/png", b"png"),),
    )
    assertion = replace(
        _search_hit(),
        id="assertion",
        content="The key in the photo is amber.",
        context=MemoryContext(
            kind=MemoryKind.RELATION,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.9,
            valid_from=None,
            valid_until=None,
            recorded_at=source.created_at,
            evidence_ids=(source.id,),
        ),
    )
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            diagnosis = (
                '{"requests":[{"media_index":0,"reason":"detail_missing"}]}'
                if selected
                else '{"requests":[]}'
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"index": 0, "message": {"content": diagnosis}, "finish_reason": "stop"}
                    ]
                },
            )
        assert selected
        assert b"The key in the photo is amber" in request.content
        assert b"data:image/png;base64," in request.content
        if stream:
            chunk = {
                "choices": [{"index": 0, "delta": {"content": "Amber."}, "finish_reason": "stop"}]
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": "Amber."}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _selected(
            _sdk_client(client), generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        )
        if stream:
            generator = model.stream_answer(ModelInput(text="What color?"), (assertion, source))
            while True:
                try:
                    next(generator)
                except StopIteration as complete:
                    result = complete.value
                    break
        else:
            result = model.answer(ModelInput(text="What color?"), (assertion, source))
    assert calls == (2 if selected else 1)
    assert result.abstained is not selected
    assert {hit.id for hit in result.hits} == ({source.id, assertion.id} if selected else set())


@pytest.mark.parametrize(
    "diagnosis,expected",
    [
        ('{"requests":[]}', 0),
        ('{"requests":[{"media_index":1,"reason":"detail_missing"}]}', 1),
        ('{"requests":[{"media_index":1,"reason":"conflicting_evidence"}]}', 1),
        ('{"requests":[{"media_index":99,"reason":"detail_missing"}]}', 0),
        ('{"requests":[{"media_index":true,"reason":"detail_missing"}]}', 0),
        ('{"requests":[{"media_index":1,"reason":"identify_this_person"}]}', 0),
        (
            '{"requests":[{"media_index":0,"reason":"detail_missing"},{"media_index":1,"reason":"detail_missing"}]}',
            0,
        ),
        ("not json", 0),
    ],
)
def test_diagnosis_is_bounded_and_keeps_text_omissions_and_request_cost(
    tmp_path: Path,
    diagnosis: str,
    expected: int,
) -> None:
    assets = tuple(
        _asset(tmp_path, f"picture-{index}", Modality.IMAGE, "image/png", f"png-{index}".encode())
        for index in range(2)
    )
    hit = replace(
        _search_hit(),
        content="Two contrasting descriptions; the label is absent.",
        assets=assets,
        modality=Modality.IMAGE,
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            assert isinstance(payload["messages"][1]["content"], str)
            assert payload["max_completion_tokens"] == 8192
            assert "max_tokens" not in payload
            assert "data:image" not in request.content.decode()
        else:
            content = payload["messages"][1]["content"]
            parts = content if isinstance(content, list) else []
            images = [part for part in parts if part["type"] == "image_url"]
            assert len(images) == expected
            assert "media_omitted" in str(content)
            assert hit.content in str(content)
            if images:
                assert images[0]["image_url"]["url"].endswith("cG5nLTE=")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": diagnosis if len(sent) == 1 else "grounded answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("media-test")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _selected(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        with (
            operation_span(tracer, "operation", attributes={}),
            model_span(tracer, "answer", attributes={}),
        ):
            model.answer(ModelInput(text="What does the missing label say?"), (hit,))
    attributes = {span.name: span.attributes for span in exporter.get_finished_spans()}
    provider.shutdown()
    assert len(sent) == 2
    assert attributes["operation"] is not None and attributes["operation"][MODEL_REQUEST_COUNT] == 2
    assert attributes["operation"][TOKEN_TOTAL] == 30
    assert attributes["sensemem.model.media_diagnosis"] is not None
    assert (
        attributes["sensemem.model.media_diagnosis"]["sensemem.grounding.media_diagnosis.selected"]
        == expected
    )


def test_text_only_answers_skip_diagnosis() -> None:
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        _selected(_sdk_client(client)).answer(ModelInput(text="question"), (_search_hit(),))
    assert len(sent) == 1


def test_diagnosis_text_ceiling_includes_system_prompt_before_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"pixels")
    hit = replace(_search_hit(), assets=(image,), modality=Modality.IMAGE)
    monkeypatch.setattr("sensemem.models.openai_sdk._MEDIA_DIAGNOSTIC_PROMPT", "界" * 500)
    monkeypatch.setattr("sensemem.models.openai_sdk._MAX_GROUNDED_TEXT_BYTES", 1_000)

    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("oversized diagnosis text reached the provider")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        adapter = _selected(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        with pytest.raises(ModelError) as failure:
            adapter.answer(ModelInput(text="question"), (hit,))
    assert failure.value.reason == "payload_too_large"
    assert failure.value.stage == "generate"


@pytest.mark.parametrize("safety_rejection", [False, True])
def test_provider_rejection_does_not_diagnose_again_or_lose_original_omissions(
    tmp_path: Path,
    safety_rejection: bool,
) -> None:
    assets = tuple(
        _asset(tmp_path, f"picture-{i}", Modality.IMAGE, "image/png", b"image") for i in range(2)
    )
    original = _search_hit()
    hit = replace(
        original,
        assets=assets,
        modality=Modality.IMAGE,
        context=MemoryContext(
            kind=MemoryKind.OBSERVATION,
            basis=EvidenceBasis.OBSERVATION,
            confidence=1.0,
            valid_from=None,
            valid_until=None,
            recorded_at=original.created_at,
        ),
    )
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            content = '{"requests":[{"media_index":0,"reason":"detail_missing"}]}'
        elif len(sent) == 2:
            message = (
                "image input may contain inappropriate content"
                if safety_rejection
                else "image input is too short"
            )
            return httpx.Response(400, json={"error": {"message": message}})
        else:
            assert "response_format" not in payload
            evidence = json.loads(payload["messages"][1]["content"])["hits"][0]
            assert evidence["source_modality"] == "image"
            assert evidence["media_omitted"] == {"image": 2}
            assert "media_modalities" not in evidence
            assert "time order" in payload["messages"][0]["content"]
            content = "answer"
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _selected(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        if safety_rejection:
            with pytest.raises(ModelError) as failure:
                model.answer(ModelInput(text="question"), (hit,), exhaustive=True)
            assert failure.value.reason == "model_refused"
        else:
            model.answer(ModelInput(text="question"), (hit,), exhaustive=True)
    assert len(sent) == (2 if safety_rejection else 3)


@pytest.mark.parametrize("limit", [True, 0, 9])
def test_media_controls_reject_invalid_values(limit: int) -> None:
    with pytest.raises(ValidationError):
        EvidenceGeneration(OpenAIModels(), media_max_items=limit)
