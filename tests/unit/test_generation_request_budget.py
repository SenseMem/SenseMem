"""Complete outbound byte limits without model calls or benchmark-specific inputs."""

import base64
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import cast

import httpx2 as httpx
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mindbridge.models.openai_sdk as adapter
from mindbridge import AssetRef, Modality, ModelError, OpenAIModels, SearchHit
from mindbridge.models.base import ModelInput

ATOMIC_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


def _asset(tmp_path: Path, name: str, modality: Modality, data: bytes) -> AssetRef:
    path = tmp_path / name
    path.write_bytes(data)
    return AssetRef(
        id=name,
        modality=modality,
        media_type={
            Modality.IMAGE: "image/jpeg",
            Modality.VIDEO: "video/mp4",
            Modality.AUDIO: "audio/wav",
        }[modality],
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        name=name,
        path=path,
    )


def _hit(identifier: str, assets: tuple[AssetRef, ...] = ()) -> SearchHit:
    return SearchHit(
        id=identifier,
        content=f"Complete source {identifier}: 蓝色工具箱。",
        score=0.9,
        created_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        assets=assets,
        modality=ModelInput(assets=assets).modality if assets else Modality.TEXT,
    )


def _jpeg() -> bytes:
    from PIL import Image

    output = BytesIO()
    Image.effect_noise((512, 512), 100).convert("RGB").save(output, format="JPEG", quality=95)
    return output.getvalue()


def _reply() -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": {"content": "蓝色。"}, "finish_reason": "stop"}]},
    )


@pytest.mark.parametrize("stream", [False, True])
def test_budget_matches_real_sdk_wire_bytes_including_extra_body_and_stream(stream: bool) -> None:
    extra = {"reasoning": {"effort": "low"}, "provider_note": "中文🧰"}
    hit = _hit("one")
    captured: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        if stream:
            chunk = {
                "choices": [{"index": 0, "delta": {"content": "蓝色。"}, "finish_reason": "stop"}]
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return _reply()

    model = OpenAIModels(generation_request_max_bytes=100_000, generation_extra_body=extra)
    prepared = model._answer_request("箱子是什么颜色?", (hit,), stream=stream)
    assert isinstance(prepared, tuple)
    exact = adapter._answer_wire_size(prepared[0], stream=stream)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        sdk = OpenAI(
            api_key="test", base_url="https://models.example/v1", http_client=client, max_retries=0
        )
        model = OpenAIModels(sdk, generation_request_max_bytes=exact, generation_extra_body=extra)
        if stream:
            assert tuple(model.stream_answer("箱子是什么颜色?", (hit,))) == ("蓝色。",)
        else:
            assert model.answer("箱子是什么颜色?", (hit,)).answer == "蓝色。"
        assert len(captured[0]) == exact
        payload = json.loads(captured[0])
        assert payload["reasoning"] == {"effort": "low"}
        assert "extra_body" not in payload
        model = OpenAIModels(
            sdk, generation_request_max_bytes=exact - 1, generation_extra_body=extra
        )
        with pytest.raises(ModelError) as failure:
            if stream:
                tuple(model.stream_answer("箱子是什么颜色?", (hit,)))
            else:
                model.answer("箱子是什么颜色?", (hit,))
        assert failure.value.reason == "payload_too_large"
        assert len(captured) == 1


@pytest.mark.parametrize("count", [1, 2])
def test_budget_resizes_required_question_image_without_removing_text_or_source(
    tmp_path: Path,
    count: int,
) -> None:
    pictures = tuple(
        _asset(tmp_path, f"query-{index}.jpg", Modality.IMAGE, _jpeg()) for index in range(count)
    )
    question = ModelInput(text="完整问题原文🧰" * 20, assets=pictures)
    hit = _hit("one")
    captured: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return _reply()

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            OpenAI(api_key="test", http_client=client, max_retries=0),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_request_max_bytes=8_000,
        )
        result = model.answer(question, (hit,))
    assert result.hits[0].content == hit.content
    assert len(captured[0]) <= 8_000
    payload = json.loads(captured[0])
    content = payload["messages"][1]["content"]
    assert json.loads(content[0]["text"])["question"] == question.text
    assert sum(part["type"] == "image_url" for part in content) == count
    assert "image re-encoded or resized" in content[1]["text"]
    for picture in pictures:
        assert picture.path is not None
        assert hashlib.sha256(picture.path.read_bytes()).hexdigest() == picture.sha256


def test_required_question_audio_is_never_dropped_to_meet_the_limit(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "query.wav", Modality.AUDIO, b"audio" * 4_000)

    def unreachable(_request: httpx.Request) -> httpx.Response:
        pytest.fail("an over-budget mandatory question must fail before HTTP")

    with httpx.Client(transport=httpx.MockTransport(unreachable)) as client:
        model = OpenAIModels(
            OpenAI(api_key="test", http_client=client, max_retries=0),
            generation_capabilities=ATOMIC_MODALITIES,
            generation_request_max_bytes=4_000,
        )
        with pytest.raises(ModelError) as failure:
            model.answer(ModelInput(text="Full question", assets=(audio,)), (_hit("one"),))
    assert failure.value.reason == "payload_too_large"
    assert failure.value.stage == "generate"


def test_ranked_omission_keeps_gps_and_map_metadata_without_absent_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio = _asset(tmp_path, "first.wav", Modality.AUDIO, b"audio" * 800)
    image = _asset(tmp_path, "second.jpg", Modality.IMAGE, _jpeg())
    hits = (_hit("first", (audio,)), _hit("second", (image,)))
    gps = {
        "gps": {"latitude": 51.5, "longitude": -0.1},
        "capture_time": {"value": "2026-01-01T10:00:00"},
    }
    monkeypatch.setattr(adapter, "image_capture_metadata", lambda _path: gps)
    probe = OpenAIModels(
        generation_request_max_bytes=1_000_000, generation_capabilities=ATOMIC_MODALITIES
    )
    stripped = (hits[0], replace(hits[1], assets=(), modality=Modality.TEXT))
    request, _, _, _ = probe._render_answer_request(
        ModelInput(text="What is here?"),
        stripped,
        hits,
        {},
        {image.id: gps},
        answer_policy="strict",
        exhaustive=False,
        metadata_source_hits=hits,
    )
    maximum = adapter._answer_wire_size(request, stream=False) + 100
    captured: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return _reply()

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        provider.get_tracer("test").start_as_current_span("answer"),
    ):
        model = OpenAIModels(
            OpenAI(api_key="test", http_client=client, max_retries=0),
            generation_capabilities=ATOMIC_MODALITIES,
            generation_request_max_bytes=maximum,
        )
        result = model.answer("What is here?", hits)
    provider.shutdown()
    assert len(captured[0]) <= maximum
    assert result.hits[0].assets == (audio,)
    assert result.hits[1].assets == ()
    texts = [
        json.loads(part["text"])
        for part in json.loads(captured[0])["messages"][1]["content"]
        if part["type"] == "text" and part["text"].startswith("{")
    ]
    memory = texts[-1]["memory"]
    assert memory["content"] == hits[1].content
    assert memory["media"] == []
    fields = memory["untrusted_source_metadata"][0]
    assert fields["gps"] == gps["gps"]
    assert fields["media_omitted"] is True and fields["source_asset_id"] == image.id
    assert "attachment" not in fields
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["mindbridge.grounding.request_bytes"] == len(captured[0])
    assert attributes["mindbridge.grounding.budget_omitted_assets"] == 1
    # Map enrichment is retained in the same source-bound metadata-only payload.
    parts = adapter._answer_text_parts(
        ModelInput(text="Where?"),
        stripped,
        ({}, {}),
        source_hits=hits,
        capture_metadata={
            image.id: {
                **gps,
                "offline_osm_candidates": {
                    "candidates": [{"name": "fixture"}],
                    "global_complete": False,
                },
            }
        },
    )
    metadata = json.loads(parts[-1])["memory"]["untrusted_source_metadata"][0]
    assert metadata["offline_osm_candidates"]["candidates"][0]["name"] == "fixture"
    assert "attachment" not in metadata


def test_video_is_sampled_only_when_native_video_cannot_fit_on_its_own(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = _asset(tmp_path, "clip.mp4", Modality.VIDEO, b"video" * 10_000)
    hit = _hit("clip", (video,))
    image = BytesIO()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(image, format="JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(image.getvalue()).decode()
    counts: list[tuple[int, int | None]] = []

    def frames(
        _asset: AssetRef, *, count: int, max_frame_bytes: int | None = None
    ) -> tuple[str, ...]:
        counts.append((count, max_frame_bytes))
        return (url,) * count

    monkeypatch.setattr(adapter, "_video_frame_urls", frames)
    model = OpenAIModels(
        generation_capabilities=ATOMIC_MODALITIES, generation_request_max_bytes=25_000
    )
    prepared = model._answer_request("What happens?", (hit,))
    assert isinstance(prepared, tuple)
    assert adapter._answer_wire_size(prepared[0], stream=False) <= 25_000
    content = cast(
        list[dict[str, object]],
        cast(list[dict[str, object]], prepared[0]["messages"])[1]["content"],
    )
    assert sum(part["type"] == "image_url" for part in content) == 16
    assert any(
        "motion, audio and exact timing are not supplied" in str(part.get("text"))
        for part in content
    )
    assert counts[0][0] == 16 and counts[0][1] is not None
    counts.clear()
    model = OpenAIModels(
        generation_capabilities=ATOMIC_MODALITIES, generation_request_max_bytes=100_000
    )
    prepared = model._answer_request("What happens?", (hit,))
    assert isinstance(prepared, tuple)
    assert not counts
    # Native video fits by itself, but earlier ranked audio leaves too little room. Keep
    # native semantics and omit it explicitly instead of converting it to stills.
    audio = _asset(tmp_path, "earlier.wav", Modality.AUDIO, b"audio" * 4_000)
    model = OpenAIModels(
        generation_capabilities=ATOMIC_MODALITIES, generation_request_max_bytes=80_000
    )
    prepared = model._answer_request("What happens?", (_hit("earlier", (audio,)), hit))
    assert isinstance(prepared, tuple)
    assert prepared[1][0].assets == (audio,)
    assert prepared[1][1].assets == ()
    assert not counts


def test_sampled_video_frames_obey_the_explicit_frame_byte_limit() -> None:
    import av
    from PIL import Image

    from_image = cast(Callable[[Image.Image], av.VideoFrame], av.VideoFrame.from_image)
    frame = from_image(Image.effect_noise((512, 512), 100).convert("RGB"))
    url = adapter._jpeg_frame_url(frame, max_encoded_bytes=3_000)
    assert len(url) <= 3_000
    with Image.open(BytesIO(base64.b64decode(url.partition(",")[2]))) as image:
        assert image.width * image.height < 512 * 512


def test_none_keeps_existing_body_and_all_nonanswer_recipe_identities() -> None:
    default = OpenAIModels()
    explicit = OpenAIModels(generation_request_max_bytes=None)
    bounded = OpenAIModels(generation_request_max_bytes=100_000)
    hit = _hit("one")
    assert default._answer_request("What?", (hit,)) == explicit._answer_request("What?", (hit,))
    assert default.embedding_space == bounded.embedding_space
    assert default.vision_space == bounded.vision_space
    assert default.formation_space == bounded.formation_space


def test_fitting_request_keeps_exact_old_body_without_reading_previously_omitted_gps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _asset(tmp_path, "omitted.jpg", Modality.IMAGE, _jpeg())
    original = (_hit("one", (image,)),)
    fitted = (replace(original[0], assets=(), modality=Modality.TEXT),)
    read: list[Path] = []

    def metadata(path: Path) -> dict[str, object]:
        read.append(path)
        return {"gps": {"latitude": 51.5, "longitude": -0.1}}

    monkeypatch.setattr(adapter, "image_capture_metadata", metadata)
    unbounded = OpenAIModels(generation_capabilities=ATOMIC_MODALITIES)
    old = unbounded._answer_request("Where?", fitted, omission_source_hits=original)
    bounded = OpenAIModels(
        generation_capabilities=ATOMIC_MODALITIES, generation_request_max_bytes=1_000_000
    )
    new = bounded._answer_request("Where?", fitted, omission_source_hits=original)
    assert isinstance(old, tuple) and isinstance(new, tuple)
    assert json.dumps(old[0], ensure_ascii=False) == json.dumps(new[0], ensure_ascii=False)
    assert not read
    # A real overflow also cannot add metadata from media the pre-existing fit already removed.
    too_small = OpenAIModels(
        generation_capabilities=ATOMIC_MODALITIES, generation_request_max_bytes=1
    )
    with pytest.raises(ModelError):
        too_small._answer_request("Where?", fitted, omission_source_hits=original)
    assert not read


def test_multiple_mandatory_images_fit_when_full_text_uses_most_of_the_budget(
    tmp_path: Path,
) -> None:
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (1, 1)).save(output, format="JPEG")
    tiny = tuple(
        _asset(tmp_path, f"tiny-{index}.jpg", Modality.IMAGE, output.getvalue())
        for index in range(2)
    )
    large = tuple(
        _asset(tmp_path, f"large-{index}.jpg", Modality.IMAGE, _jpeg()) for index in range(2)
    )
    text = ("The complete original question. " * 250).strip()
    hit = _hit("one")
    baseline = OpenAIModels(generation_capabilities=ATOMIC_MODALITIES)
    text_only = baseline._answer_request(text, (hit,))
    minimum = baseline._answer_request(ModelInput(text=text, assets=tiny), (hit,))
    assert isinstance(text_only, tuple) and isinstance(minimum, tuple)
    fixed = adapter._answer_wire_size(text_only[0], stream=False)
    maximum = max(fixed * 10 // 7, adapter._answer_wire_size(minimum[0], stream=False) + 256)
    assert fixed / maximum >= 0.65
    model = OpenAIModels(
        generation_capabilities=ATOMIC_MODALITIES, generation_request_max_bytes=maximum
    )
    prepared = model._answer_request(ModelInput(text=text, assets=large), (hit,))
    assert isinstance(prepared, tuple)
    assert adapter._answer_wire_size(prepared[0], stream=False) <= maximum
    content = cast(
        list[dict[str, object]],
        cast(list[dict[str, object]], prepared[0]["messages"])[1]["content"],
    )
    assert sum(part["type"] == "image_url" for part in content) == 2
    assert json.loads(cast(str, content[0]["text"]))["question"] == text
