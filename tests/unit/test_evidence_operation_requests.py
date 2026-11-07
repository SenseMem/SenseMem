"""Native witnessed operations preserve fitted evidence and expose only terminal text."""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx2 as httpx
import pytest
from _feature_support import TinyEmbedder
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_image_source_prose import _image, _memories
from test_openai_sdk import _sdk_client, _search_hit

import sensemem.configuration as configuration
import sensemem.models.openai_sdk as adapter
import sensemem.recipes as recipes
from sensemem import Memory, Modality, ModelError, ModelInput, OpenAIModels, SenseMemConfig
from sensemem._telemetry import (
    MODEL_REQUEST_COUNT,
    TOKEN_COMPLETE,
    TOKEN_TOTAL,
    model_span,
    operation_span,
)
from sensemem.models._evidence_operations import (
    EVIDENCE_OPERATION_INSTRUCTION,
    EVIDENCE_OPERATION_TOOL_NAME,
)
from sensemem.types import AbstentionReason, AnswerPolicy

QUESTION = "What is the TOTAL?"
QUOTE = "Invoice TOTAL 58.40 EUR."


def _arguments(*, native: str | None = None, label: str = "E1", uncertain: bool = False) -> str:
    anchor = {
        "evidence_label": label,
        "quote": QUOTE if native is None else None,
        "attachment": native,
        "observation": QUOTE if native is not None else None,
    }
    return json.dumps(
        {
            "operation": "scalar",
            "member_unit": "entity",
            "quantifier": "exact",
            "measure": "total",
            "format": "with_unit",
            "scope": {
                "kind": "single_source",
                "basis": "model_assessed",
                "status": "uncertain" if uncertain else "adequate",
                "question_quote": "TOTAL",
                "evidence_labels": [label],
                "reason": "Named invoice field is supported."
                if not uncertain
                else "The named invoice is uncertain.",
            },
            "predicates": [],
            "inventory": {"status": "complete", "anchors": [anchor]},
            "members": [{"id": "invoice", "anchor": anchor, "conditions": [], "identity": None}],
            "identities": [],
            "values": [
                {
                    "id": "total",
                    "member_id": "invoice",
                    "role": "total",
                    "label": "TOTAL",
                    "value": "58.40",
                    "unit": "EUR",
                    "extent": "whole",
                    "part_key": None,
                    "anchor": anchor,
                }
            ],
            "contributions": [
                {
                    "member_id": "invoice",
                    "basis": "whole",
                    "value_ids": ["total"],
                    "parts_complete": False,
                }
            ],
        }
    )


def _reply(
    request: httpx.Request,
    *,
    arguments: str | None = None,
    answer: str = "Ordinary answer.",
    usage: bool = True,
    missing_finish: bool = False,
    finish: str | None = None,
) -> httpx.Response:
    reason = finish or ("tool_calls" if arguments is not None else "stop")
    message: dict[str, object] = {
        "content": "Provisional answer 999." if arguments is not None else answer
    }
    function = {"name": EVIDENCE_OPERATION_TOOL_NAME, "arguments": arguments}
    if arguments is not None:
        message["tool_calls"] = [{"id": "native-call", "type": "function", "function": function}]
    reported = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    if not json.loads(request.content).get("stream"):
        choice = {
            "index": 0,
            "message": message,
            **({"finish_reason": reason} if not missing_finish else {}),
        }
        return httpx.Response(
            200, json={"choices": [choice], **({"usage": reported} if usage else {})}
        )
    deltas: list[dict[str, object]] = [{"content": message["content"]}]
    if arguments is not None:
        deltas.extend(
            [
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "native-call",
                            "type": "function",
                            "function": {
                                "name": EVIDENCE_OPERATION_TOOL_NAME,
                                "arguments": arguments[:17],
                            },
                        }
                    ]
                },
                {"tool_calls": [{"index": 0, "function": {"arguments": arguments[17:]}}]},
            ]
        )
    chunks = [
        {
            "id": "completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fixture-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    if not missing_finish:
        chunks.append(
            {
                "id": "completion",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            }
        )
    if usage:
        chunks.append(
            {
                "id": "completion",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture-model",
                "choices": [],
                "usage": reported,
            }
        )
    return httpx.Response(
        200,
        content="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


def _parts(payload: dict[str, object]) -> list[dict[str, object]]:
    return cast(
        list[dict[str, object]], cast(list[dict[str, object]], payload["messages"])[1]["content"]
    )


def _provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _events(exporter: InMemorySpanExporter) -> Counter[str]:
    return Counter(
        span.name.removeprefix("sensemem.evidence_operation.")
        for span in exporter.get_finished_spans()
        if span.name.startswith("sensemem.evidence_operation.")
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_valid_native_submission_executes_without_a_second_reader(
    tmp_path: Path, native: bool, stream: bool
) -> None:
    hit = replace(_search_hit(), content=QUOTE, metadata={"source_id": "invoice-photo"})
    if native:
        hit = replace(
            hit,
            content="Venue: Cedar House. OCR: TOTAL 99.50 EUR.",
            assets=(_image(tmp_path, "invoice"),),
            modality=Modality.IMAGE,
        )
    question = ModelInput(QUESTION, evidence_sources=(hit,))
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, arguments=_arguments(native="attachment 1" if native else None))

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_model="fixture-model",
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_evidence_operation_tool=True,
            generation_image_source_prose="context_when_supplied",
        )
        with (
            operation_span(provider.get_tracer("test"), "operation", attributes={}),
            model_span(provider.get_tracer("test"), "model", attributes={}),
        ):
            if stream:
                assert list(model.stream_answer(question, (hit,))) == ["58.4 EUR"]
            else:
                result = model.answer(question, (hit,))
                assert result.answer == "58.4 EUR" and not result.abstained
                assert result.hits == (hit,)
    provider.shutdown()
    assert len(sent) == 1 and sent[0]["tool_choice"] == "auto"
    assert sent[0]["parallel_tool_calls"] is False
    assert "response_format" not in sent[0]
    assert _memories(sent[0])[0]["evidence_label"] == "E1"
    if native:
        assert _memories(sent[0])[0]["auxiliary_source_text"] == hit.content
        assert sum(part["type"] == "image_url" for part in _parts(sent[0])) == 1
    assert _events(exporter) == {"submitted": 1, "executed": 1}
    for span in exporter.get_finished_spans():
        assert span.attributes is not None
        assert span.attributes.get(MODEL_REQUEST_COUNT, 0) == (
            1 if span.name in {"model", "operation"} else 0
        )
        assert "58.40" not in json.dumps(dict(span.attributes))
    parent = next(span for span in exporter.get_finished_spans() if span.name == "model")
    assert parent.attributes is not None and parent.attributes[TOKEN_TOTAL] == 12


def test_public_config_ask_stream_reaches_actual_native_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, arguments=_arguments())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(configuration, "_build_embedding", lambda _config: TinyEmbedder())
        monkeypatch.setattr(recipes, "_openai_client", lambda **_controls: _sdk_client(client))
        config = SenseMemConfig.model_validate(
            {
                "data_dir": tmp_path / "memory",
                "embedding": {"provider": "openai"},
                "generation": {
                    "provider": "openai",
                    "model": "fixture-model",
                    "evidence_operation_tool": True,
                },
            }
        )
        with Memory.from_config(config) as memory:
            memory.add(QUOTE, metadata={"source_id": "invoice"})
            chunks = list(memory.ask_stream(QUESTION))
            assert "".join(chunk.text or "" for chunk in chunks) == "58.4 EUR"
            assert chunks[-1].result is not None and not chunks[-1].result.abstained
            assert memory.get(chunks[-1].result.hits[0].id).content == QUOTE
    assert len(sent) == 1 and sent[0]["stream"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}
    assert (
        cast(dict[str, object], cast(list[dict[str, object]], sent[0]["tools"])[0]["function"])[
            "name"
        ]
        == EVIDENCE_OPERATION_TOOL_NAME
    )


@pytest.mark.parametrize("policy", ["strict", "best_effort"])
@pytest.mark.parametrize("stream", [False, True])
def test_honest_needs_evidence_is_policy_result_not_protocol_recovery(
    policy: AnswerPolicy, stream: bool
) -> None:
    hit = replace(_search_hit(), content=QUOTE)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, arguments=_arguments(uncertain=True))

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if stream:
                values = list(
                    model.stream_answer(ModelInput(QUESTION), (hit,), answer_policy=policy)
                )
                text = "".join(values)
            else:
                result = model.answer(ModelInput(QUESTION), (hit,), answer_policy=policy)
                assert (
                    result.abstained
                    and result.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
                )
                text = result.answer
    provider.shutdown()
    assert (
        text == adapter.UNKNOWN_ANSWER
        if policy == "strict"
        else text.startswith("Supported result in the supplied evidence: 58.4 EUR.")
    )
    assert len(sent) == 1 and _events(exporter) == {"submitted": 1, "needs_evidence": 1}


@pytest.mark.parametrize("stream", [False, True])
def test_rejected_native_proposal_gets_only_one_original_evidence_read(stream: bool) -> None:
    hit = replace(_search_hit(), content=QUOTE)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(
            request,
            arguments='{"private_proposal":"untrusted secret"}' if len(sent) == 1 else None,
            answer="Recovered answer.",
        )

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if stream:
                assert list(model.stream_answer(ModelInput(QUESTION), (hit,))) == [
                    "Recovered answer."
                ]
            else:
                assert model.answer(ModelInput(QUESTION), (hit,)).answer == "Recovered answer."
    provider.shutdown()
    assert len(sent) == 2 and "tools" not in sent[1] and "tool_choice" not in sent[1]
    assert "parallel_tool_calls" not in sent[1]
    assert "untrusted secret" not in json.dumps(sent[1]) and "999" not in json.dumps(sent[1])
    assert _memories(sent[1])[0]["content"] == hit.content
    assert "evidence_label" not in _memories(sent[1])[0]
    assert adapter._EVIDENCE_OPERATION_RECOVERY in cast(
        str, cast(list[dict[str, object]], sent[1]["messages"])[0]["content"]
    )
    assert _events(exporter) == {
        "submitted": 1,
        "rejected": 1,
        "rejected_schema": 1,
        "recovery": 1,
    }
    attributes = next(
        span for span in exporter.get_finished_spans() if span.name == "model"
    ).attributes
    assert (
        attributes is not None
        and attributes[MODEL_REQUEST_COUNT] == 2
        and attributes[TOKEN_TOTAL] == 24
    )


@pytest.mark.parametrize("stream", [False, True])
def test_first_generic_invalid_response_recovers_without_inventing_a_submission(
    stream: bool,
) -> None:
    hit = replace(_search_hit(), content=QUOTE)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, missing_finish=len(sent) == 1, answer="Recovered.")

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if stream:
                assert list(model.stream_answer(ModelInput(QUESTION), (hit,))) == ["Recovered."]
            else:
                assert model.answer(ModelInput(QUESTION), (hit,)).answer == "Recovered."
    provider.shutdown()
    assert len(sent) == 2
    assert _events(exporter) == {"response_invalid": 1, "recovery": 1}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "first_answer",
    [
        "E1",
        "E1, E2",
        '["E1", "E2"]',
        '<tool_calls><invoke name="submit_evidence_operation"><parameter name="operation">'
        "scalar</parameter></invoke></tool_calls>",
        '<tool_calls><invoke name="functions.submit_evidence_operation">'
        '<parameter name="operation">scalar</parameter></invoke></tool_calls>',
    ],
)
def test_no_call_protocol_text_gets_one_original_evidence_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, first_answer: str
) -> None:
    hits = (
        replace(
            _search_hit(),
            id="image-source",
            content="The entrance is E1.",
            metadata={"source_id": "entrance-photo"},
            assets=(_image(tmp_path, "entrance"),),
            modality=Modality.IMAGE,
        ),
        replace(_search_hit(), id="text-source", content="The matching wing is E1."),
    )
    sent: list[dict[str, object]] = []
    final_answer = first_answer if first_answer.startswith("<tool_calls>") else "E1"

    def unexpected_execution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("text must never be executed as a native proposal")

    monkeypatch.setattr(adapter, "execute_evidence_operation", unexpected_execution)

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, answer=first_answer if len(sent) == 1 else final_answer)

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_evidence_operation_tool=True,
            generation_image_source_prose="context_when_supplied",
        )
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if stream:
                assert list(model.stream_answer(ModelInput("Which entrance?"), hits)) == [
                    final_answer
                ]
            else:
                assert model.answer(ModelInput("Which entrance?"), hits).answer == final_answer
    provider.shutdown()
    assert len(sent) == 2
    assert all(key not in sent[1] for key in ("tools", "tool_choice", "parallel_tool_calls"))
    assert _memories(sent[1]) == [
        {key: value for key, value in memory.items() if key != "evidence_label"}
        for memory in _memories(sent[0])
    ]
    assert [part for part in _parts(sent[0]) if part["type"] == "image_url"] == [
        part for part in _parts(sent[1]) if part["type"] == "image_url"
    ]
    messages = cast(list[dict[str, object]], sent[1]["messages"])
    assert [message["role"] for message in messages] == ["system", "user"]
    assert EVIDENCE_OPERATION_INSTRUCTION not in cast(str, messages[0]["content"])
    assert "<tool_calls>" not in json.dumps(sent[1])
    assert _events(exporter) == {"response_invalid": 1, "recovery": 1}
    attributes = next(
        span for span in exporter.get_finished_spans() if span.name == "model"
    ).attributes
    assert attributes is not None
    assert attributes[MODEL_REQUEST_COUNT] == 2 and attributes[TOKEN_TOTAL] == 24


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "answer",
    [
        "The entrance is E1.",
        "E10",
        "E1, E99",
        '<event code="E1">Arrival</event>',
        '<tool_calls><invoke name="lookup">Literal example</invoke></tool_calls>',
    ],
)
def test_no_call_factual_alias_or_unrelated_xml_text_is_not_rejected(
    stream: bool, answer: str
) -> None:
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, answer=answer)

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if stream:
                assert list(model.stream_answer(ModelInput(QUESTION), (_search_hit(),))) == [answer]
            else:
                assert model.answer(ModelInput(QUESTION), (_search_hit(),)).answer == answer
    provider.shutdown()
    assert len(sent) == 1 and not _events(exporter)


@pytest.mark.parametrize("stream", [False, True])
def test_no_call_protocol_text_does_not_override_content_filter(stream: bool) -> None:
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, answer="E1", finish="content_filter")

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with (
            model_span(provider.get_tracer("test"), "model", attributes={}),
            pytest.raises(ModelError) as failure,
        ):
            if stream:
                list(model.stream_answer(ModelInput(QUESTION), (_search_hit(),)))
            else:
                model.answer(ModelInput(QUESTION), (_search_hit(),))
    provider.shutdown()
    assert failure.value.reason == "model_refused"
    assert len(sent) == 1 and not _events(exporter)
    attributes = next(
        span for span in exporter.get_finished_spans() if span.name == "model"
    ).attributes
    assert attributes is not None
    assert attributes[MODEL_REQUEST_COUNT] == 1 and attributes[TOKEN_TOTAL] == 12


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("final_finish", ["tool_calls", "stop"])
@pytest.mark.parametrize("first_text", [False, True])
def test_bad_final_response_never_causes_a_third_reader(
    stream: bool, final_finish: str, first_text: bool
) -> None:
    hit = replace(_search_hit(), content=QUOTE)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(
            request,
            arguments=None if len(sent) == 1 and first_text else "{}",
            answer="E1",
            finish=final_finish if len(sent) == 2 else None,
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with pytest.raises(ModelError) as failure:
            if stream:
                list(model.stream_answer(ModelInput(QUESTION), (hit,)))
            else:
                model.answer(ModelInput(QUESTION), (hit,))
    assert failure.value.reason == "response_invalid" and len(sent) == 2


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget", ["wire", "text"])
def test_tool_and_alias_overhead_never_refits_or_drops_original_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool, budget: str
) -> None:
    hits = tuple(
        replace(
            _search_hit(),
            id=f"source-{index}",
            content=QUOTE,
            assets=(_image(tmp_path, f"image-{index}"),),
            modality=Modality.IMAGE,
        )
        for index in range(2)
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        sdk = _sdk_client(client)
        for enabled in (False, True):
            maximum = None
            if sent:
                if budget == "wire":
                    maximum = adapter._answer_wire_size(sent[0], stream=stream)
                else:
                    messages = cast(list[dict[str, object]], sent[0]["messages"])
                    byte_count = len(cast(str, messages[0]["content"]).encode()) + sum(
                        len(cast(str, part["text"]).encode())
                        for part in _parts(sent[0])
                        if part["type"] == "text"
                    )
                    monkeypatch.setattr(adapter, "_MAX_GROUNDED_TEXT_BYTES", byte_count)
            model = OpenAIModels(
                sdk,
                generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
                generation_evidence_operation_tool=enabled,
                generation_request_max_bytes=maximum,
            )
            if stream:
                assert (
                    "".join(model.stream_answer(ModelInput(QUESTION), hits)) == "Ordinary answer."
                )
            else:
                assert model.answer(ModelInput(QUESTION), hits).answer == "Ordinary answer."
    assert len(sent) == 2 and sent[0] == sent[1]
    assert sum(part["type"] == "image_url" for part in _parts(sent[1])) == 2


@pytest.mark.parametrize("foreign", ["missing", "wrong_source"])
def test_foreign_or_omitted_native_anchor_is_rejected_without_losing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: str
) -> None:
    images = (_image(tmp_path, "first"), _image(tmp_path, "omitted"))
    hits = tuple(
        replace(
            _search_hit(), id=f"source-{i}", content=QUOTE, assets=(image,), modality=Modality.IMAGE
        )
        for i, image in enumerate(images)
    )
    monkeypatch.setattr(
        adapter, "_MAX_INLINE_MODEL_BYTES", adapter._encoded_size(cast(int, images[0].size_bytes))
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        arguments = _arguments(
            native="attachment 99" if foreign == "missing" else "attachment 1",
            label="E1" if foreign == "missing" else "E2",
        )
        return _reply(request, arguments=arguments if len(sent) == 1 else None)

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_evidence_operation_tool=True,
        )
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            assert model.answer(ModelInput(QUESTION), hits).answer == "Ordinary answer."
    provider.shutdown()
    assert len(sent) == 2 and _events(exporter) == {
        "submitted": 1,
        "rejected": 1,
        "rejected_invalid_evidence": 1,
        "recovery": 1,
    }
    for payload in sent:
        assert _memories(payload)[1]["media_omitted"] == {"image": 1}
        assert [memory["content"] for memory in _memories(payload)] == [QUOTE, QUOTE]
        assert sum(part["type"] == "image_url" for part in _parts(payload)) == 1


def test_shared_native_asset_remains_bound_to_both_actual_source_aliases(tmp_path: Path) -> None:
    image = _image(tmp_path, "shared")
    hits = tuple(
        replace(
            _search_hit(), id=f"source-{i}", content=QUOTE, assets=(image,), modality=Modality.IMAGE
        )
        for i in range(2)
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, arguments=_arguments(native="attachment 1", label="E2"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_evidence_operation_tool=True,
        )
        result = model.answer(ModelInput(QUESTION), hits)
    assert result.answer == "58.4 EUR" and result.hits == (hits[1],)
    assert len(sent) == 1
    assert [_memories(sent[0])[i]["evidence_label"] for i in range(2)] == ["E1", "E2"]
    assert sum(part["type"] == "image_url" for part in _parts(sent[0])) == 1


@pytest.mark.parametrize("stream", [False, True])
def test_no_call_accepts_unreported_usage_without_a_recovery(stream: bool) -> None:
    hit = replace(_search_hit(), content=QUOTE)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, usage=False)

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if stream:
                assert (
                    "".join(model.stream_answer(ModelInput(QUESTION), (hit,))) == "Ordinary answer."
                )
            else:
                assert model.answer(ModelInput(QUESTION), (hit,)).answer == "Ordinary answer."
    provider.shutdown()
    assert len(sent) == 1 and not _events(exporter)
    attributes = exporter.get_finished_spans()[0].attributes
    assert (
        attributes is not None
        and attributes[MODEL_REQUEST_COUNT] == 1
        and attributes[TOKEN_COMPLETE] is False
    )


@pytest.mark.parametrize("stream", [False, True])
def test_existing_media_rejection_keeps_original_body_and_counts_both_attempts(
    tmp_path: Path, stream: bool
) -> None:
    hit = replace(
        _search_hit(), content=QUOTE, assets=(_image(tmp_path, "image"),), modality=Modality.IMAGE
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            return httpx.Response(400, json={"error": {"message": "The image is too short."}})
        return _reply(request)

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_image_source_prose="context_when_supplied",
            generation_evidence_operation_tool=True,
        )
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if stream:
                assert (
                    "".join(model.stream_answer(ModelInput(QUESTION), (hit,))) == "Ordinary answer."
                )
            else:
                assert model.answer(ModelInput(QUESTION), (hit,)).answer == "Ordinary answer."
    provider.shutdown()
    assert len(sent) == 2 and "tools" not in sent[1]
    assert _memories(sent[1])[0]["content"] == QUOTE
    assert "auxiliary_source_text" not in _memories(sent[1])[0]
    assert not _events(exporter)
    attributes = exporter.get_finished_spans()[0].attributes
    assert (
        attributes is not None
        and attributes[MODEL_REQUEST_COUNT] == 2
        and attributes[TOKEN_COMPLETE] is False
    )


@pytest.mark.parametrize("broken", [False, True])
def test_native_stream_closes_once_and_iterator_failure_does_not_start_recovery(
    tmp_path: Path, broken: bool
) -> None:
    del tmp_path
    close_count = 0
    requests = 0

    class TrackedStream(httpx.SyncByteStream):
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __iter__(self) -> Iterator[bytes]:
            if broken:
                yield self.body.partition(b"data: [DONE]")[0]
                raise RuntimeError("private stream failure")
            yield self.body

        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            stream=TrackedStream(_reply(request, arguments=_arguments()).content),
            headers={"content-type": "text/event-stream"},
        )

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            if broken:
                with pytest.raises(ModelError) as failure:
                    list(
                        model.stream_answer(
                            ModelInput(QUESTION), (replace(_search_hit(), content=QUOTE),)
                        )
                    )
                assert failure.value.stage == "generate" and "private stream failure" not in str(
                    failure.value
                )
            else:
                assert list(
                    model.stream_answer(
                        ModelInput(QUESTION), (replace(_search_hit(), content=QUOTE),)
                    )
                ) == ["58.4 EUR"]
    provider.shutdown()
    assert requests == close_count == 1
    assert _events(exporter) == ({} if broken else {"submitted": 1, "executed": 1})


@pytest.mark.parametrize("ending", ["invalid_chunk", "missing_finish", "completed"])
def test_native_stream_cleanup_failure_never_starts_protocol_recovery(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    requests = 0

    class FailingCloseStream:
        close_calls = 0

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(content="Provisional answer."),
                        finish_reason="stop" if ending != "missing_finish" else None,
                    )
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            )
            if ending == "invalid_chunk":
                yield SimpleNamespace(choices=[SimpleNamespace(index=1)])

        def close(self) -> None:
            self.close_calls += 1
            raise httpx.ReadError("private cleanup failure")

    responses = FailingCloseStream()

    def create(**_request: object) -> FailingCloseStream:
        nonlocal requests
        requests += 1
        return responses

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(OpenAIModels, "_client", lambda _self, _operation: client)
    model = OpenAIModels(generation_evidence_operation_tool=True)
    provider, exporter = _provider()
    with (
        model_span(provider.get_tracer("test"), "model", attributes={}),
        pytest.raises(ModelError) as failure,
    ):
        list(model.stream_answer(ModelInput(QUESTION), (replace(_search_hit(), content=QUOTE),)))
    provider.shutdown()
    assert failure.value.reason == (
        "connection_failed" if ending == "completed" else "response_invalid"
    )
    assert "private cleanup failure" not in str(failure.value)
    assert requests == responses.close_calls == 1
    assert _events(exporter) == {}
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes[MODEL_REQUEST_COUNT] == 1 and attributes[TOKEN_TOTAL] == 12


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [401, 500])
def test_provider_request_errors_do_not_enter_protocol_recovery(stream: bool, status: int) -> None:
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(status, json={"error": {"message": "private provider failure"}})

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with (
            model_span(provider.get_tracer("test"), "model", attributes={}),
            pytest.raises(ModelError) as failure,
        ):
            if stream:
                list(
                    model.stream_answer(
                        ModelInput(QUESTION), (replace(_search_hit(), content=QUOTE),)
                    )
                )
            else:
                model.answer(ModelInput(QUESTION), (replace(_search_hit(), content=QUOTE),))
    provider.shutdown()
    assert requests == 1 and _events(exporter) == {}
    assert "private provider failure" not in str(failure.value)


@pytest.mark.parametrize(
    "category",
    ["protocol_finish", "protocol_type", "protocol_count", "protocol_name", "protocol_arguments"],
)
def test_native_protocol_rejections_have_only_static_safe_categories(category: str) -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests != 1:
            return _reply(request, answer="Recovered.")
        response = _reply(request, arguments=_arguments()).json()
        choice = response["choices"][0]
        calls = choice["message"]["tool_calls"]
        if category == "protocol_finish":
            choice["finish_reason"] = "stop"
        elif category == "protocol_type":
            del calls[0]["type"]
        elif category == "protocol_count":
            calls.append(calls[0])
        elif category == "protocol_name":
            calls[0]["function"]["name"] = "private function name"
        else:
            calls[0]["function"]["arguments"] = {"private": "invalid arguments"}
        return httpx.Response(200, json=response)

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            assert (
                model.answer(ModelInput(QUESTION), (replace(_search_hit(), content=QUOTE),)).answer
                == "Recovered."
            )
    provider.shutdown()
    assert requests == 2
    assert _events(exporter) == {
        "submitted": 1,
        "rejected": 1,
        f"rejected_{category}": 1,
        "recovery": 1,
    }
    for span in exporter.get_finished_spans():
        assert span.attributes is not None
        assert "private" not in span.name + json.dumps(dict(span.attributes))
        if span.name.startswith("sensemem.evidence_operation."):
            assert span.attributes.get(MODEL_REQUEST_COUNT, 0) == 0
    parent = next(span for span in exporter.get_finished_spans() if span.name == "model")
    assert parent.attributes is not None
    assert parent.attributes[MODEL_REQUEST_COUNT] == 2 and parent.attributes[TOKEN_TOTAL] == 24


@pytest.mark.parametrize("category", ["protocol_finish", "protocol_type", "protocol_name"])
def test_completed_native_sse_preserves_protocol_rejection_category(category: str) -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests != 1:
            return _reply(request, answer="Recovered.")
        response = _reply(
            request,
            arguments=_arguments(),
            finish="stop" if category == "protocol_finish" else None,
        )
        body = response.content
        if category == "protocol_type":
            body = body.replace(b'"type": "function", ', b"")
        elif category == "protocol_name":
            body = body.replace(EVIDENCE_OPERATION_TOOL_NAME.encode(), b"private_function")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider, exporter = _provider()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(_sdk_client(client), generation_evidence_operation_tool=True)
        with model_span(provider.get_tracer("test"), "model", attributes={}):
            assert list(
                model.stream_answer(ModelInput(QUESTION), (replace(_search_hit(), content=QUOTE),))
            ) == ["Recovered."]
    provider.shutdown()
    assert requests == 2
    assert _events(exporter) == {
        "submitted": 1,
        "rejected": 1,
        f"rejected_{category}": 1,
        "recovery": 1,
    }
