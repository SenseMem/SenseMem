"""vLLM score wire protocol, failure mapping, and resource ownership."""

import base64
import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import httpx2 as httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import sensemem.models.vllm_reranking as reranking
from sensemem import AssetRef, Modality, ModelError, ModelInput, ValidationError, VLLMReranker
from sensemem._telemetry import (
    MODEL_REQUEST_COUNT,
    TOKEN_COMPLETE,
    TOKEN_EXPECTED_REQUEST_COUNT,
    TOKEN_REPORTED_REQUEST_COUNT,
    TOKEN_TOTAL,
)


def _reply(*scores: float) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "score-fixture",
            "model": "served-reranker",
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
            "data": sorted(
                [{"index": index, "score": score} for index, score in enumerate(scores)],
                key=lambda row: row["score"],
                reverse=True,
            ),
        },
    )


def _asset(tmp_path: Path, modality: Modality) -> AssetRef:
    path = tmp_path / modality.value
    path.write_bytes(modality.value.encode())
    return AssetRef(
        modality.value,
        modality=modality,
        path=path,
        media_type={
            Modality.IMAGE: "image/png",
            Modality.VIDEO: "video/mp4",
            Modality.AUDIO: "audio/wav",
        }[modality],
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_text_requests_batch_all_documents_and_restore_server_indices() -> None:
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://fixture.test/proxy/score"
        assert request.headers["authorization"] == "Bearer fixture-key"
        assert request.extensions["timeout"] == dict.fromkeys(
            ("connect", "read", "write", "pool"), 7.0
        )
        payload = json.loads(request.content)
        sent.append(payload)
        return _reply(0.2, 0.9) if len(sent) == 1 else _reply(-1.5)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    adapter = VLLMReranker(
        client,
        model_id="org/model",
        base_url="https://fixture.test/proxy/",
        api_key="fixture-key",
        batch_size=2,
        timeout=7,
    )
    try:
        assert adapter.rerank(
            ModelInput(text="query"), [ModelInput(text=str(i)) for i in range(3)]
        ) == (0.2, 0.9, -1.5)
        assert sent == [
            {"model": "org/model", "queries": "query", "documents": ["0", "1"]},
            {"model": "org/model", "queries": "query", "documents": ["2"]},
        ]
    finally:
        adapter.close()


def test_failed_batch_waits_for_other_media_readers_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = _asset(tmp_path, Modality.IMAGE)
    entered = threading.Barrier(2)
    failed = threading.Event()
    release = threading.Event()
    sent: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        document = json.loads(request.content)["documents"][0]
        if document == "fail":
            sent.append("fail")
            entered.wait(timeout=5)
            return httpx.Response(503)
        sent.append("media")
        entered.wait(timeout=5)
        assert release.wait(5)
        assert image.path is not None and image.path.read_bytes() == b"image"
        assert not client.is_closed
        return _reply(0.8)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    adapter = VLLMReranker(
        client,
        model_id="org/model",
        base_url="https://fixture.test",
        batch_size=1,
        request_concurrency=2,
    )
    response_result = reranking._response_scores

    def observe_failure(
        response: httpx.Response, count: int, usage: reranking._RequestUsage
    ) -> tuple[float, ...]:
        try:
            return response_result(response, count, usage)
        finally:
            if response.status_code == 503:
                failed.set()

    monkeypatch.setattr(reranking, "_response_scores", observe_failure)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                adapter.rerank,
                ModelInput(text="query"),
                [ModelInput(text="fail"), ModelInput(assets=(image,)), ModelInput(text="unsent")],
            )
            try:
                assert failed.wait(5)
                assert not result.done()
            finally:
                release.set()
            with pytest.raises(ModelError) as raised:
                result.result(timeout=5)
            assert raised.value.reason == "provider_unavailable"
        assert sorted(sent) == ["fail", "media"]
    finally:
        adapter.close()
    adapter.close()
    assert client.is_closed
    with pytest.raises(ModelError, match="closed"):
        adapter.rerank(ModelInput(text="query"), [])


def test_fusion_refines_only_top_media_candidates_and_preserves_text_scores(tmp_path: Path) -> None:
    image = _asset(tmp_path, Modality.IMAGE)
    video = _asset(tmp_path, Modality.VIDEO)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            assert payload["documents"] == ["plain", "image", "video", "other"]
            return _reply(0.7, 0.8, 0.8, 0.1)
        assert payload["documents"] == [
            {
                "content": [
                    {"type": "text", "text": "image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + base64.b64encode(b"image").decode()
                        },
                    },
                ]
            }
        ]
        return _reply(0.2)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(
            client,
            model_id="org/model",
            base_url="https://fixture.test",
            media_candidates=1,
            media_weight=0.5,
        )
        try:
            result = adapter.rerank(
                ModelInput(text="query"),
                (
                    ModelInput(text="plain"),
                    ModelInput(text="image", assets=(image,)),
                    ModelInput(text="video", assets=(video,)),
                    ModelInput(text="other", assets=(image,)),
                ),
            )
            assert result == pytest.approx((0.7, 0.5, 0.8, 0.1))
        finally:
            adapter.close()
    assert len(sent) == 2


@pytest.mark.parametrize("last_response", ["complete", "missing", "failed"])
def test_usage_accumulates_text_and_media_batches_including_failures(
    tmp_path: Path, last_response: str
) -> None:
    image = _asset(tmp_path, Modality.IMAGE)
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        if calls == 3 and last_response == "failed":
            return httpx.Response(503)
        response = _reply(*(0.5 for _ in payload["documents"]))
        if calls == 3 and last_response == "missing":
            body = json.loads(response.content)
            body.pop("usage")
            return httpx.Response(200, json=body)
        return response

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(
            client,
            model_id="org/model",
            base_url="https://fixture.test",
            batch_size=2,
            media_candidates=1,
        )
        try:
            with provider.get_tracer("test").start_as_current_span("reranking"):
                query = ModelInput(text="query")
                documents = (ModelInput(text="plain"),) * 2 + (ModelInput(assets=(image,)),)
                if last_response == "failed":
                    with pytest.raises(ModelError, match="reranking request failed"):
                        adapter.rerank(query, documents)
                else:
                    assert adapter.rerank(query, documents) == (0.5, 0.5, 0.5)
        finally:
            adapter.close()
    assert calls == 3
    attributes = exporter.get_finished_spans()[0].attributes or {}
    assert attributes[MODEL_REQUEST_COUNT] == attributes[TOKEN_EXPECTED_REQUEST_COUNT] == 3
    reported = 3 if last_response == "complete" else 2
    assert attributes[TOKEN_REPORTED_REQUEST_COUNT] == reported
    assert attributes[TOKEN_TOTAL] == 12 * reported
    assert attributes[TOKEN_COMPLETE] == (last_response == "complete")
    provider.shutdown()


def test_fusion_retains_query_media_when_candidates_have_only_text(tmp_path: Path) -> None:
    image = _asset(tmp_path, Modality.IMAGE)
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            assert payload["queries"] == "Retrieve candidates relevant to the query media."
            return _reply(0.3, 0.9)
        assert isinstance(payload["queries"], dict)
        assert payload["documents"] == ["second"]
        return _reply(0.8)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(
            client,
            model_id="org/model",
            base_url="https://fixture.test",
            media_candidates=1,
            media_weight=1.0,
        )
        try:
            assert adapter.rerank(
                ModelInput(assets=(image,)),
                (
                    ModelInput(text="first"),
                    ModelInput(text="second"),
                ),
            ) == pytest.approx((0.3, 0.8))
        finally:
            adapter.close()


def test_multimodal_query_and_mixed_documents_use_content_without_chat_roles(
    tmp_path: Path,
) -> None:
    image = _asset(tmp_path, Modality.IMAGE)
    video = _asset(tmp_path, Modality.VIDEO)
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"image").decode()},
    }
    video_part = {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(b"video").decode()},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        assert str(request.url) == "https://fixture.test/score"
        assert json.loads(request.content) == {
            "model": "org/model",
            "queries": {"content": [video_part, image_part]},
            "documents": [
                "plain text",
                {"content": [image_part]},
                {"content": [{"type": "text", "text": "caption"}, video_part, image_part]},
            ],
        }
        return _reply(0.1, 0.8, 0.5)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(client, model_id="org/model", base_url="https://fixture.test")
        try:
            assert adapter.rerank(
                ModelInput(assets=(video, image)),
                [
                    ModelInput(text="plain text"),
                    ModelInput(assets=(image,)),
                    ModelInput(text="caption", assets=(video, image)),
                ],
            ) == (0.1, 0.8, 0.5)
        finally:
            adapter.close()


@pytest.mark.parametrize("concurrency", [2, 32])
def test_shared_http_backend_allows_concurrent_requests(concurrency: int) -> None:
    entered = threading.Barrier(concurrency)

    def respond(_: httpx.Request) -> httpx.Response:
        entered.wait(timeout=5)
        return _reply(0.8)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(client, model_id="org/model", base_url="https://fixture.test")
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                results = [
                    executor.submit(
                        adapter.rerank, ModelInput(text="query"), [ModelInput(text="doc")]
                    )
                    for _ in range(concurrency)
                ]
                assert [result.result(timeout=10) for result in results] == [(0.8,)] * concurrency
        finally:
            adapter.close()


@pytest.mark.parametrize("failure", [False, True])
def test_close_waits_for_active_request_and_rejects_new_calls(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    entered = threading.Event()
    release = threading.Event()
    closing = threading.Event()

    def respond(_: httpx.Request) -> httpx.Response:
        entered.set()
        assert release.wait(timeout=5)
        assert not client.is_closed
        return httpx.Response(503) if failure else _reply(0.8)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(client, model_id="org/model", base_url="https://fixture.test")
        wait_for = adapter._condition.wait_for

        def observe_wait(predicate: Callable[[], bool], timeout: float | None = None) -> bool:
            closing.set()
            return wait_for(predicate, timeout)

        monkeypatch.setattr(adapter._condition, "wait_for", observe_wait)
        with ThreadPoolExecutor(max_workers=2) as executor:
            request = executor.submit(
                adapter.rerank, ModelInput(text="query"), [ModelInput(text="doc")]
            )
            try:
                assert entered.wait(timeout=5)
                shutdown = executor.submit(adapter.close)
                assert closing.wait(timeout=5)
                assert not client.is_closed and not shutdown.done()
                with pytest.raises(ModelError, match="closed"):
                    adapter.rerank(ModelInput(text="new query"), [ModelInput(text="doc")])
            finally:
                release.set()
            if failure:
                with pytest.raises(ModelError, match="request failed"):
                    request.result(timeout=5)
            else:
                assert request.result(timeout=5) == (0.8,)
            shutdown.result(timeout=5)
        assert client.is_closed
        adapter.close()


@pytest.mark.parametrize(
    "body",
    [
        "not JSON",
        {},
        {"data": None},
        {"data": []},
        {"data": [{"index": 0, "score": 0.5}]},
        {"data": [{"index": 0, "score": 0.5}, {"index": 0, "score": 0.8}]},
        {"data": [{"index": 0, "score": 0.5}, {"index": 2, "score": 0.8}]},
        {"data": [{"index": -1, "score": 0.5}, {"index": 1, "score": 0.8}]},
        {
            "data": [
                {"index": True, "score": 0.5},
                {"index": 0, "score": 0.8},
            ]
        },
        {"data": [{"index": "0", "score": 0.5}, {"index": 1, "score": 0.8}]},
        *[
            {
                "data": [
                    {"index": 0, "score": score},
                    {"index": 1, "score": 0.8},
                ]
            }
            for score in [True, "0.5", None, float("nan"), float("inf")]
        ],
    ],
)
def test_invalid_responses_cannot_supply_scores(body: object) -> None:
    content = body if isinstance(body, str) else json.dumps(body)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    ) as client:
        adapter = VLLMReranker(client, model_id="org/model", base_url="https://fixture.test")
        try:
            with pytest.raises(ModelError) as raised:
                adapter.rerank(
                    ModelInput(text="query"), [ModelInput(text="first"), ModelInput(text="second")]
                )
            assert raised.value.reason == "response_invalid"
            assert raised.value.stage == "reranking"
        finally:
            adapter.close()


@pytest.mark.parametrize(
    ("status", "reason", "retryable"),
    [
        (400, "request_rejected", False),
        (401, "auth_failed", False),
        (403, "auth_failed", False),
        (404, "request_rejected", False),
        (408, "timeout", True),
        (413, "payload_too_large", False),
        (429, "rate_limited", True),
        (503, "provider_unavailable", True),
    ],
)
def test_http_failures_are_classified_without_exposing_provider_bodies(
    status: int, reason: str, retryable: bool
) -> None:
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status, text="private provider body", headers={"x-request-id": "request-fixture"}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(
            client, model_id="org/model", base_url="https://fixture.test", api_key="secret"
        )
        try:
            with pytest.raises(ModelError) as raised:
                adapter.rerank(
                    ModelInput(text="private query"), [ModelInput(text="private content")]
                )
            assert raised.value.reason == reason
            assert raised.value.retryable is retryable
            assert raised.value.stage == "reranking"
            assert "private" not in str(raised.value) and "secret" not in str(raised.value)
            assert raised.value.__cause__ is None
            assert calls == 1
        finally:
            adapter.close()


@pytest.mark.parametrize(
    ("error", "reason"), [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "connection_failed")]
)
def test_transport_failures_are_classified(error: type[httpx.RequestError], reason: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise error("private provider detail", request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(client, model_id="org/model", base_url="https://fixture.test")
        try:
            with pytest.raises(ModelError) as raised:
                adapter.rerank(ModelInput(text="query"), [ModelInput(text="document")])
            assert raised.value.reason == reason
            assert raised.value.stage == "reranking"
            assert "private" not in str(raised.value)
        finally:
            adapter.close()


def test_empty_and_unsupported_inputs_do_not_make_requests(tmp_path: Path) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        raise AssertionError("input validation must run before transport")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = VLLMReranker(
            client,
            model_id="org/model",
            base_url="https://fixture.test",
            modalities={Modality.TEXT},
        )
        try:
            assert adapter.rerank(ModelInput(text="query"), []) == ()
            with pytest.raises(ValidationError, match="unsupported modality"):
                adapter.rerank(
                    ModelInput(assets=(_asset(tmp_path, Modality.IMAGE),)),
                    [ModelInput(text="document")],
                )
            with pytest.raises(ValidationError, match="unsupported modality"):
                adapter.rerank(
                    ModelInput(text="query"),
                    [ModelInput(assets=(_asset(tmp_path, Modality.AUDIO),))],
                )
        finally:
            adapter.close()


def test_missing_media_preserves_asset_error_at_reranking_stage(tmp_path: Path) -> None:
    image = _asset(tmp_path, Modality.IMAGE)
    assert image.path is not None
    image.path.unlink()
    with httpx.Client(transport=httpx.MockTransport(lambda _: _reply(0.5))) as client:
        adapter = VLLMReranker(client, model_id="org/model", base_url="https://fixture.test")
        try:
            with pytest.raises(ModelError) as raised:
                adapter.rerank(ModelInput(text="query"), [ModelInput(assets=(image,))])
            assert raised.value.reason == "asset_unavailable"
            assert raised.value.subject == image.id
            assert raised.value.stage == "reranking"
        finally:
            adapter.close()


@pytest.mark.parametrize(
    "controls",
    [
        {"timeout": 0},
        {"timeout": True},
        {"timeout": float("nan")},
        {"batch_size": 0},
        {"request_concurrency": 0},
        {"request_concurrency": True},
        {"request_concurrency": "2"},
        {"modalities": []},
        {"modalities": [Modality.AUDIO]},
        {"base_url": "fixture.test"},
        {"base_url": "http://fixture.test:bad"},
        {"base_url": "https://fixture.test?key=secret"},
    ],
)
def test_constructor_rejects_invalid_controls(controls: dict[str, object]) -> None:
    values: dict[str, object] = {
        "model_id": "org/model",
        "base_url": "https://fixture.test",
        **controls,
    }
    with pytest.raises(ValidationError):
        VLLMReranker(**values)  # type: ignore[arg-type]


def test_batches_overlap_and_refill_before_the_first_batch_finishes() -> None:
    release = threading.Event()
    refilled = threading.Event()
    finished: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        index = int(json.loads(request.content)["documents"][0])
        if index == 0:
            assert release.wait(5)
        elif index == 2:
            refilled.set()
            assert release.wait(5)
        finished.append(index)
        return _reply(float(index))

    client = httpx.Client(transport=httpx.MockTransport(respond))
    adapter = VLLMReranker(
        client,
        model_id="org/model",
        base_url="https://fixture.test",
        batch_size=1,
        request_concurrency=2,
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                adapter.rerank,
                ModelInput(text="query"),
                [ModelInput(text=str(i)) for i in range(5)],
            )
            try:
                assert refilled.wait(5)
                assert finished == [1]
                assert not result.done()
            finally:
                release.set()
            assert result.result(timeout=5) == (0.0, 1.0, 2.0, 3.0, 4.0)
    finally:
        adapter.close()


def test_request_limit_is_shared_across_callers(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Barrier(3)
    submitted = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = peak = submissions = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if not release.is_set():
            entered.wait(timeout=5)
        assert release.wait(5)
        with lock:
            active -= 1
        return _reply(0.8)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    adapter = VLLMReranker(
        client, model_id="org/model", base_url="https://fixture.test", request_concurrency=2
    )
    submit = adapter._workers.submit

    def observe_submit(fn: Callable[..., httpx.Response], *args: object) -> Future[httpx.Response]:
        nonlocal submissions
        future = submit(fn, *args)
        with lock:
            submissions += 1
            if submissions == 3:
                submitted.set()
        return future

    monkeypatch.setattr(adapter._workers, "submit", observe_submit)
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = [
                executor.submit(adapter.rerank, ModelInput(text="query"), [ModelInput(text="doc")])
                for _ in range(3)
            ]
            try:
                entered.wait(timeout=5)
                assert submitted.wait(5)
                assert active == peak == 2
            finally:
                release.set()
            assert [result.result(timeout=5) for result in results] == [(0.8,)] * 3
        assert peak == 2
    finally:
        adapter.close()
