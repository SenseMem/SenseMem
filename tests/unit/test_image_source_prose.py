"""Final image prose is a wire view over the original, fully fitted evidence."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import httpx2 as httpx
import pytest
from PIL import Image
from test_openai_sdk import NOW, _asset, _sdk_client, _search_hit

import sensemem.configuration as configuration
import sensemem.models.openai_sdk as adapter
import sensemem.recipes as recipes
from sensemem import AssetRef, Modality, ModelInput, OpenAIModels
from sensemem.types import EvidenceBasis, MemoryContext, MemoryKind

_Policy = Literal["include", "context_when_supplied"]


def _image(directory: Path, name: str) -> AssetRef:
    path = directory / f"{name}.png"
    Image.new("RGB", (1, 1), "blue").save(path)
    data = path.read_bytes()
    return AssetRef(
        name, Modality.IMAGE, "image/png", len(data), hashlib.sha256(data).hexdigest(), path=path
    )


def _reply(request: httpx.Request, answer: str = "Viewed.") -> httpx.Response:
    if json.loads(request.content).get("stream"):
        chunk = {
            "id": "fixture-completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "fixture-model",
            "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": "stop"}],
        }
        return httpx.Response(
            200,
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": {"content": answer}, "finish_reason": "stop"}]},
    )


def _memories(payload: dict[str, object]) -> list[dict[str, object]]:
    content = cast(list[dict[str, object]], payload["messages"])[1]["content"]
    if isinstance(content, str):
        return cast(list[dict[str, object]], json.loads(content)["hits"])
    return [
        cast(dict[str, object], json.loads(cast(str, part["text"]))["memory"])
        for part in cast(list[dict[str, object]], content)
        if part["type"] == "text" and cast(str, part["text"]).startswith('{"memory":')
    ]


@pytest.mark.parametrize("policy", ["include", "context_when_supplied"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("answer_format", ["text", "structured"])
def test_configured_final_view_preserves_other_sources_metadata_and_original_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: _Policy,
    streaming: bool,
    answer_format: Literal["text", "structured"],
) -> None:
    image, mixed_image = _image(tmp_path, "image"), _image(tmp_path, "mixed-image")
    video = _asset(tmp_path, "video", Modality.VIDEO, "video/mp4", b"first clip")
    omitted_video = _asset(tmp_path, "mixed-video", Modality.VIDEO, "video/mp4", b"second clip")
    context = MemoryContext(
        kind=MemoryKind.OBSERVATION,
        basis=EvidenceBasis.OBSERVATION,
        confidence=1.0,
        valid_from=None,
        valid_until=None,
        recorded_at=NOW,
        subject="typed subject",
    )
    hits = (
        replace(
            _search_hit(),
            content=(
                "The toolbox is blue.\nVenue: Willow House, 25 Cedar Lane.\n"
                "Description: a red toolbox. OCR: ITEM-10."
            ),
            assets=(image,),
            modality=Modality.IMAGE,
            metadata={"source_id": "public-image", "application_field": "kept"},
            context=context,
        ),
        replace(_search_hit(), id="text", content="Original text source."),
        replace(
            _search_hit(),
            id="video",
            content="Original video prose.",
            assets=(video,),
            modality=Modality.VIDEO,
        ),
        replace(
            _search_hit(),
            id="mixed",
            content="Original mixed prose.",
            assets=(mixed_image, omitted_video),
            modality=Modality.OMNI,
        ),
    )
    question = ModelInput(
        "What is shown?",
        evidence_sources=hits,
        source_metadata={image.id: {"capture_time": {"local_datetime": "2026-01-02T12:00:00"}}},
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        answer = (
            "Viewed."
            if answer_format == "text"
            else json.dumps(
                {
                    "draft": {
                        "kind": "text",
                        "answer": "Viewed.",
                        "evidence_labels": ["E1"],
                        "abstained": False,
                    }
                }
            )
        )
        return _reply(request, answer)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(recipes, "_openai_client", lambda **_controls: _sdk_client(client))
        model = configuration._build_generation(
            configuration.OpenAIGenerationConfig(
                provider="openai",
                model="fixture-model",
                modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
                image_source_prose=policy,
                answer_format=answer_format,
                video_limit=1,
            )
        )
        assert isinstance(model, OpenAIModels)
        try:
            if streaming:
                stream = model.stream_answer(question, hits)
                assert "".join(stream) == "Viewed."
            else:
                result = model.answer(question, hits)
                assert result.answer == "Viewed."
                assert result.hits[0].content == hits[0].content
        finally:
            model.close()

    assert len(sent) == 1
    memories = _memories(sent[0])
    assert [memory["content"] for memory in memories] == [
        "" if policy == "context_when_supplied" else hits[0].content,
        hits[1].content,
        hits[2].content,
        hits[3].content,
    ]
    if policy == "context_when_supplied":
        assert memories[0]["auxiliary_source_text"] == hits[0].content
        assert adapter._IMAGE_SOURCE_CONTEXT_PROMPT in json.dumps(sent[0])
    else:
        assert "auxiliary_source_text" not in memories[0]
        assert adapter._IMAGE_SOURCE_CONTEXT_PROMPT not in json.dumps(sent[0])
    assert all("auxiliary_source_text" not in memory for memory in memories[1:])
    assert memories[0]["metadata"] == dict(hits[0].metadata)
    assert cast(dict[str, object], memories[0]["context"])["subject"] == "typed subject"
    assert memories[0]["created_at"] == NOW.isoformat()
    assert all("inspection_memory_id" not in memory for memory in memories)
    assert all("inspection_assets" not in memory for memory in memories)
    assert "2026-01-02T12:00:00" in json.dumps(memories[0])
    assert memories[3]["source_modality"] == "omni"
    assert memories[3]["media_modalities"] == {"attachment 3": "image"}
    assert memories[3]["media_omitted"] == {"video": 1}
    assert question.evidence_sources == hits and question.evidence_sources[0] is hits[0]
    assert "Venue: Willow House, 25 Cedar Lane." in hits[0].content
    assert (
        image.path is not None
        and hashlib.sha256(image.path.read_bytes()).hexdigest() == image.sha256
    )


@pytest.mark.parametrize("budget", ["inline", "wire"])
def test_projection_neither_admits_more_media_nor_loses_omitted_image_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: str
) -> None:
    images = (_image(tmp_path, "first"), _image(tmp_path, "second"))
    hits = tuple(
        replace(
            _search_hit(),
            id=f"source-{index}",
            content=f"Original prose {index}.",
            assets=(image,),
            modality=Modality.IMAGE,
        )
        for index, image in enumerate(images)
    )
    question = ModelInput("Read the sources.", evidence_sources=hits)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request)

    original_inline_limit = adapter._MAX_INLINE_MODEL_BYTES
    monkeypatch.setattr(
        adapter, "_MAX_INLINE_MODEL_BYTES", adapter._encoded_size(cast(int, images[0].size_bytes))
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        sdk = _sdk_client(client)
        OpenAIModels(
            sdk, generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        ).answer(question, hits)
        maximum = adapter._answer_wire_size(sent[0], stream=False) if budget == "wire" else None
        if budget == "wire":
            monkeypatch.setattr(adapter, "_MAX_INLINE_MODEL_BYTES", original_inline_limit)
        for policy in ("include", "context_when_supplied"):
            result = OpenAIModels(
                sdk,
                generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
                generation_image_source_prose=policy,
                generation_request_max_bytes=maximum,
            ).answer(question, hits)
            assert result.hits[0].assets == (images[0],)
            assert result.hits[1].assets == () and result.hits[1].content == hits[1].content
    assert [_memories(payload)[1]["content"] for payload in sent] == [hits[1].content] * 3
    for payload in sent:
        content = cast(
            list[dict[str, object]],
            cast(list[dict[str, object]], payload["messages"])[1]["content"],
        )
        assert sum(part["type"] == "image_url" for part in content) == 1
    assert _memories(sent[1])[0]["content"] == hits[0].content
    if budget == "inline":
        assert _memories(sent[2])[0]["content"] == ""
        assert _memories(sent[2])[0]["auxiliary_source_text"] == hits[0].content
    else:
        assert sent[2] == sent[1]
    assert "auxiliary_source_text" not in _memories(sent[2])[1]


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_image_rejection_falls_back_to_original_text(
    tmp_path: Path, streaming: bool
) -> None:
    image = _image(tmp_path, "rejected")
    hit = replace(_search_hit(), assets=(image,), modality=Modality.IMAGE)
    question = ModelInput("What color?", evidence_sources=(hit,))
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            assert _memories(sent[0])[0]["content"] == ""
            assert _memories(sent[0])[0]["auxiliary_source_text"] == hit.content
            assert adapter._IMAGE_SOURCE_CONTEXT_PROMPT in json.dumps(sent[0])
            return httpx.Response(400, json={"error": {"message": "The image is too short."}})
        assert _memories(sent[1])[0]["content"] == hit.content
        assert _memories(sent[1])[0]["media_omitted"] == {"image": 1}
        assert "auxiliary_source_text" not in _memories(sent[1])[0]
        assert adapter._IMAGE_SOURCE_CONTEXT_PROMPT not in json.dumps(sent[1])
        return _reply(request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_image_source_prose="context_when_supplied",
        )
        if streaming:
            assert "".join(model.stream_answer(question, (hit,))) == "Viewed."
        else:
            result = model.answer(question, (hit,))
            assert result.hits[0].content == hit.content and result.hits[0].assets == ()
    assert len(sent) == 2 and hit.assets == (image,)


def test_partially_supplied_multi_image_source_keeps_full_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images = (_image(tmp_path, "album-first"), _image(tmp_path, "album-second"))
    hit = replace(
        _search_hit(),
        content="The first image shows a toolbox; the second shows its serial number.",
        assets=images,
        modality=Modality.IMAGE,
    )
    question = ModelInput("What is the serial number?", evidence_sources=(hit,))
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request)

    monkeypatch.setattr(
        adapter, "_MAX_INLINE_MODEL_BYTES", adapter._encoded_size(cast(int, images[0].size_bytes))
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_image_source_prose="context_when_supplied",
        ).answer(question, (hit,))

    assert result.hits[0].assets == (images[0],)
    memory = _memories(sent[0])[0]
    assert memory["media"] == ["attachment 1"]
    assert memory["media_omitted"] == {"image": 1}
    assert memory["content"] == hit.content
    assert "auxiliary_source_text" not in memory
    assert adapter._IMAGE_SOURCE_CONTEXT_PROMPT not in json.dumps(sent[0])
    assert hit.assets == images and question.evidence_sources == (hit,)


def test_final_policy_does_not_change_recall_planning_wire() -> None:
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request, "{}")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        for policy in ("include", "context_when_supplied"):
            model = OpenAIModels(
                _sdk_client(client),
                generation_image_source_prose=policy,
            )
            assert (
                model.plan_recall("What color?", reference_at=NOW, corpus_digest="fixture corpus")
                == "{}"
            )
    assert len(sent) == 2 and sent[0] == sent[1]


@pytest.mark.parametrize("content", ["", " \t\n "])
def test_blank_image_prose_keeps_the_complete_ordinary_request(
    tmp_path: Path, content: str
) -> None:
    hit = replace(
        _search_hit(),
        content=content,
        assets=(_image(tmp_path, "blank-prose"),),
        modality=Modality.IMAGE,
    )
    question = ModelInput("What is shown?", evidence_sources=(hit,))
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply(request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        for policy in ("include", "context_when_supplied"):
            result = OpenAIModels(
                _sdk_client(client),
                generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
                generation_image_source_prose=policy,
            ).answer(question, (hit,))
            assert result.hits == (hit,)

    assert len(sent) == 2 and sent[0] == sent[1]
    assert _memories(sent[1])[0]["content"] == content
    assert "auxiliary_source_text" not in _memories(sent[1])[0]
    assert adapter._IMAGE_SOURCE_CONTEXT_PROMPT not in json.dumps(sent[1])
    assert question.evidence_sources == (hit,)


@pytest.mark.parametrize("budget", ["wire", "text"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("answer_format", ["text", "structured"])
def test_context_overhead_falls_back_to_the_complete_fitted_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget: str,
    streaming: bool,
    answer_format: Literal["text", "structured"],
) -> None:
    hit = replace(
        _search_hit(),
        content="Venue: Willow House, 25 Cedar Lane.\nDescription: a blue toolbox.",
        assets=(_image(tmp_path, "bounded-context"),),
        modality=Modality.IMAGE,
    )
    question = ModelInput("Where was the toolbox?", evidence_sources=(hit,))
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        answer = (
            "Viewed."
            if answer_format == "text"
            else json.dumps(
                {
                    "draft": {
                        "kind": "text",
                        "answer": "Viewed.",
                        "evidence_labels": ["E1"],
                        "abstained": False,
                    }
                }
            )
        )
        return _reply(request, answer)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        sdk = _sdk_client(client)

        def answer(policy: _Policy, maximum: int | None = None) -> None:
            model = OpenAIModels(
                sdk,
                generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
                generation_image_source_prose=policy,
                generation_answer_format=answer_format,
                generation_request_max_bytes=maximum,
            )
            if streaming:
                assert "".join(model.stream_answer(question, (hit,))) == "Viewed."
            else:
                result = model.answer(question, (hit,))
                assert result.answer == "Viewed." and result.hits == (hit,)

        answer("include")
        answer("context_when_supplied")
        assert _memories(sent[1])[0]["auxiliary_source_text"] == hit.content
        assert adapter._answer_wire_size(sent[1], stream=streaming) > adapter._answer_wire_size(
            sent[0], stream=streaming
        )
        if budget == "wire":
            maximum = adapter._answer_wire_size(sent[0], stream=streaming)
        else:
            messages = cast(list[dict[str, object]], sent[0]["messages"])
            user_parts = cast(list[dict[str, object]], messages[1]["content"])
            text_bytes = len(cast(str, messages[0]["content"]).encode()) + sum(
                len(cast(str, part["text"]).encode())
                for part in user_parts
                if part["type"] == "text"
            )
            monkeypatch.setattr(adapter, "_MAX_GROUNDED_TEXT_BYTES", text_bytes)
            maximum = None
        answer("context_when_supplied", maximum)

    assert len(sent) == 3 and sent[2] == sent[0]
    assert _memories(sent[2])[0]["content"] == hit.content
    assert "auxiliary_source_text" not in _memories(sent[2])[0]
    assert adapter._IMAGE_SOURCE_CONTEXT_PROMPT not in json.dumps(sent[2])
    assert hit.assets[0].path is not None and hit.assets[0].path.exists()
    assert question.evidence_sources == (hit,)
