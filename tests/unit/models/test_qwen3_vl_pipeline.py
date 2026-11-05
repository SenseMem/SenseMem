"""Bounded media preparation overlaps serialized inference without losing positions or resources."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import sensemem.models.qwen3_vl_reranking as reranking
from sensemem import AssetRef, Modality, ModelInput, Qwen3VLReranker
from sensemem._telemetry import MODEL_REQUEST_COUNT
from sensemem.exceptions import ModelError, ValidationError

REVISION = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"


@dataclass
class Image:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


class Runtime:
    def __init__(self) -> None:
        self.images: list[Image] = []
        self.videos: list[tuple[str, dict[str, object]]] = []
        self.batches: list[list[reranking._Prompt]] = []
        self.shutdown = Event()

    def fetch_image(self, url: str) -> Image:
        image = Image()
        self.images.append(image)
        return image

    def fetch_video(self, url: str, kwargs: dict[str, object]) -> reranking._Video:
        self.videos.append((url, kwargs))
        return SimpleNamespace(nbytes=16), {"fps": 4, "frames_indices": [0, 4], "duration": 2}

    def classify(self, rows: list[reranking._Prompt]) -> list[object]:
        self.batches.append(rows)
        return [SimpleNamespace(outputs=SimpleNamespace(probs=[0.5])) for _ in rows]


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    runtime = Runtime()
    llm = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(encode=lambda *args, **kwargs: [1]),
        classify=lambda rows, **kwargs: runtime.classify(rows),
        llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=runtime.shutdown.set)),
    )
    monkeypatch.setattr(
        reranking,
        "import_module",
        lambda name: SimpleNamespace(LLM=lambda **kwargs: llm) if name == "vllm" else runtime,
    )
    return runtime


def asset(tmp_path: Path, name: str = "a", modality: Modality = Modality.VIDEO) -> AssetRef:
    return AssetRef(
        name,
        path=tmp_path / name,
        modality=modality,
        media_type="video/mp4" if modality is Modality.VIDEO else "image/png",
        sha256=name * 64,
        size_bytes=1,
    )


def test_prepares_next_chunk_during_inference_and_stops_one_chunk_ahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: Runtime
) -> None:
    entered = Event()
    next_ready = Event()
    fetch_image = runtime.fetch_image

    def fetch(url: str) -> Image:
        if url.endswith("b"):
            assert entered.wait(5)
            next_ready.set()
        elif url.endswith("c"):
            assert len(runtime.batches) >= 1
        return fetch_image(url)

    def classify(rows: list[reranking._Prompt]) -> list[object]:
        if not runtime.batches:
            entered.set()
            assert next_ready.wait(5)
            assert len(runtime.images) <= 2
        runtime.batches.append(rows)
        score = len(runtime.batches) / 10
        return [SimpleNamespace(outputs=SimpleNamespace(probs=[score])) for _ in rows]

    monkeypatch.setattr(runtime, "fetch_image", fetch)
    monkeypatch.setattr(runtime, "classify", classify)
    model = Qwen3VLReranker.load("org/model", revision=REVISION, batch_size=1)
    try:
        documents = tuple(
            ModelInput(assets=(asset(tmp_path, name, Modality.IMAGE),)) for name in "abc"
        )
        assert model.rerank(ModelInput(text="question"), documents) == (0.1, 0.2, 0.3)
        assert len(runtime.batches) == 3
        assert all(image.closed for image in runtime.images)
    finally:
        model.close()


def test_other_requests_prepare_while_inference_is_locked_and_close_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: Runtime
) -> None:
    entered = Event()
    other_ready = Event()
    release = Event()
    fetch_image = runtime.fetch_image

    def fetch(url: str) -> Image:
        other_ready.set()
        return fetch_image(url)

    def classify(rows: list[reranking._Prompt]) -> list[object]:
        assert not runtime.shutdown.is_set()
        if not runtime.batches:
            runtime.batches.append(rows)
            entered.set()
            assert release.wait(5)
        else:
            runtime.batches.append(rows)
        return [SimpleNamespace(outputs=SimpleNamespace(probs=[0.5]))]

    monkeypatch.setattr(runtime, "fetch_image", fetch)
    monkeypatch.setattr(runtime, "classify", classify)
    model = Qwen3VLReranker.load("org/model", revision=REVISION)
    query = ModelInput(text="question")
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(model.rerank, query, (ModelInput(text="first"),))
            assert entered.wait(5)
            second = executor.submit(
                model.rerank,
                query,
                (ModelInput(assets=(asset(tmp_path, modality=Modality.IMAGE),)),),
            )
            try:
                assert other_ready.wait(5)
                closing = executor.submit(model.close)
                with model._lock:
                    assert model._lock.wait_for(lambda: model._engine is None, timeout=5)
                assert not closing.done()
                assert not runtime.shutdown.is_set()
                with pytest.raises(ModelError, match="closed"):
                    model.rerank(query, (ModelInput(text="late"),))
            finally:
                release.set()
            assert first.result(timeout=5) == second.result(timeout=5) == (0.5,)
            closing.result(timeout=5)
        assert runtime.shutdown.is_set()
        assert all(image.closed for image in runtime.images)
    finally:
        release.set()
        model.close()


@pytest.mark.parametrize("fail_inference", [False, True])
def test_releases_active_and_prefetched_images_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: Runtime, fail_inference: bool
) -> None:
    next_ready = Event()
    fetch_image = runtime.fetch_image

    def fetch(url: str) -> Image:
        if url.endswith("c") and not fail_inference:
            raise RuntimeError("private decode failure")
        image = fetch_image(url)
        if url.endswith("b"):
            next_ready.set()
        return image

    def classify(rows: list[reranking._Prompt]) -> list[object]:
        assert next_ready.wait(5)
        if fail_inference:
            raise RuntimeError("private inference failure")
        return [SimpleNamespace(outputs=SimpleNamespace(probs=[0.5]))]

    monkeypatch.setattr(runtime, "fetch_image", fetch)
    monkeypatch.setattr(runtime, "classify", classify)
    model = Qwen3VLReranker.load("org/model", revision=REVISION, batch_size=1)
    try:
        with pytest.raises(ModelError, match="reranking model failed"):
            model.rerank(
                ModelInput(text="question"),
                (
                    ModelInput(assets=(asset(tmp_path, "a", Modality.IMAGE),)),
                    ModelInput(
                        assets=tuple(asset(tmp_path, name, Modality.IMAGE) for name in "bc")
                    ),
                ),
            )
        assert all(image.closed for image in runtime.images)
    finally:
        model.close()


@pytest.mark.parametrize(("budget", "decodes"), [(0, 2), (15, 2), (16, 1)])
def test_video_cache_reuses_sha_across_paths_and_queries_with_timing(
    tmp_path: Path, runtime: Runtime, budget: int, decodes: int
) -> None:
    video = asset(tmp_path)
    model = Qwen3VLReranker.load(
        "org/model", revision=REVISION, video_frames=4, video_cache_bytes=budget
    )
    try:
        for index in range(2):
            model.rerank(
                ModelInput(text=f"question {index}"),
                (ModelInput(assets=(replace(video, id=str(index), path=tmp_path / str(index)),)),),
            )
        assert len(runtime.videos) == decodes
        assert all(
            kwargs == {"num_frames": 4, "video_backend": "opencv"} for _, kwargs in runtime.videos
        )
        first = cast(reranking._Video, runtime.batches[0][0]["multi_modal_data"]["video"][0])
        second = cast(reranking._Video, runtime.batches[1][0]["multi_modal_data"]["video"][0])
        if decodes == 1:
            assert first is second
        assert first[1] == {"fps": 4, "frames_indices": [0, 4], "duration": 2}
    finally:
        model.close()
    assert model._engine is None


def test_video_cache_evicts_least_recently_used_and_separates_sampling(
    tmp_path: Path, runtime: Runtime
) -> None:
    cache = reranking._VideoCache(32)
    for name, frames in [("a", 4), ("b", 4), ("a", 4), ("c", 4), ("b", 4), ("b", 8)]:
        cache.load(asset(tmp_path, name), frames, runtime)
    assert [Path(url).name for url, _ in runtime.videos] == ["a", "b", "c", "b", "b"]
    assert cache._bytes == 32
    cache.clear()
    assert cache._bytes == 0
    assert not cache._entries


def test_chunked_calls_report_actual_inference_count(runtime: Runtime) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    model = Qwen3VLReranker.load("org/model", revision=REVISION, batch_size=2)
    try:
        with provider.get_tracer(__name__).start_as_current_span("rerank"):
            assert (
                model.rerank(
                    ModelInput(text="question"), tuple(ModelInput(text=str(i)) for i in range(5))
                )
                == (0.5,) * 5
            )
        assert [len(batch) for batch in runtime.batches] == [2, 2, 1]
        attributes = exporter.get_finished_spans()[0].attributes
        assert attributes is not None
        assert attributes[MODEL_REQUEST_COUNT] == 3
    finally:
        model.close()
        provider.shutdown()


@pytest.mark.parametrize("count", [0, 2])
def test_each_chunk_rejects_wrong_score_count(
    monkeypatch: pytest.MonkeyPatch, runtime: Runtime, count: int
) -> None:
    def classify(rows: list[reranking._Prompt]) -> list[object]:
        runtime.batches.append(rows)
        return [SimpleNamespace(outputs=SimpleNamespace(probs=[0.5])) for _ in range(count)]

    monkeypatch.setattr(runtime, "classify", classify)
    model = Qwen3VLReranker.load("org/model", revision=REVISION, batch_size=1)
    try:
        with pytest.raises(ModelError) as error:
            model.rerank(ModelInput(text="question"), (ModelInput(text="a"), ModelInput(text="b")))
        assert error.value.reason == "response_invalid"
        assert len(runtime.batches) == 1
    finally:
        model.close()


@pytest.mark.parametrize("value", [-1, True, 1.5, "1024"])
def test_cache_budget_is_strict(value: object) -> None:
    with pytest.raises(ValidationError, match="video_cache_bytes"):
        Qwen3VLReranker(model_id="org/model", revision=REVISION, video_cache_bytes=value)  # type: ignore[arg-type]
