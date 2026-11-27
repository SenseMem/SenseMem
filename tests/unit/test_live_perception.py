"""Temporary visual windows keep accepting frames while a shared model reviews them."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import cast

import httpx2
import pytest
from openai import OpenAI
from PIL import Image
from pydantic import ValidationError as PydanticValidationError

from examples.live.perception import (
    LivePerception,
    PerceptionContext,
    PerceptionFrame,
    PerceptionResult,
)
from sensemem import (
    AnswerPolicy,
    AnswerResult,
    Modality,
    ModelError,
    ModelInput,
    SearchHit,
    ValidationError,
)
from sensemem.models.openai_completion import OpenAICompletion

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


def _jpeg(color: tuple[int, int, int] = (30, 60, 160)) -> bytes:
    encoded = BytesIO()
    Image.new("RGB", (16, 16), color).save(encoded, format="JPEG")
    return encoded.getvalue()


def _decision(sequences: tuple[int, ...]) -> dict[str, object]:
    return {
        "summary": "A blue cup is visible.",
        "changed": False,
        "action": "context",
        "utterance": None,
        "confidence": 0.95,
        "evidence": [
            {"sequence": sequence, "observation": "The visible cup is blue."}
            for sequence in dict.fromkeys((sequences[0], sequences[-1]))
        ],
        "reason": "Only update visual context.",
        "contradicted_claim": None,
        "correction": None,
    }


class Reviewer:
    generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self, *, blocked: bool = False) -> None:
        self.entered = Event()
        self.release = Event()
        if not blocked:
            self.release.set()
        self.inputs: list[ModelInput] = []
        self.originals: list[tuple[bytes, ...]] = []
        self.patch: dict[str, object] = {}
        self.closed = False
        self.fail_first = False

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        assert answer_policy == "best_effort" and not hits
        self.inputs.append(question)
        self.originals.append(
            tuple(cast(Path, asset.path).read_bytes() for asset in question.assets)
        )
        self.entered.set()
        assert self.release.wait(5), "the test did not release the model"
        if self.fail_first and len(self.inputs) == 1:
            raise ModelError("private provider body", reason="provider_unavailable")
        sequences = tuple(int(asset.id.partition(":")[2]) for asset in question.assets)
        return AnswerResult(json.dumps(_decision(sequences) | self.patch))

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_frames_coalesce_during_model_work_and_keep_real_timing_and_context() -> None:

    reviewer = Reviewer(blocked=True)
    results: list[PerceptionResult] = []
    published = asyncio.Event()
    context = PerceptionContext(assistant_text="Old response", response_id="r1", playing=True)

    async def publish(result: PerceptionResult) -> None:
        results.append(result)
        if len(results) == 2:
            published.set()

    jpeg = _jpeg()
    worker = LivePerception(
        "call-1", reviewer, publish, lambda: context, clock=lambda: NOW, max_bytes=len(jpeg) * 3
    )
    first = PerceptionFrame(sequence=1, captured_at=NOW, timestamp_ms=0, jpeg=jpeg)
    try:
        worker.submit(first)
        assert await asyncio.to_thread(reviewer.entered.wait, 5)
        for sequence in range(2, 32):
            worker.submit(
                PerceptionFrame(
                    sequence=sequence,
                    captured_at=NOW + timedelta(milliseconds=sequence * 250),
                    timestamp_ms=sequence * 250,
                    jpeg=jpeg,
                )
            )
        assert len(reviewer.inputs) == 1  # Acceptance does not wait or spawn model calls.
        assert len(worker.pending) <= 8
        assert sum(len(frame.jpeg) for frame in worker.pending) <= worker.max_bytes
        assert worker.pending[-1].sequence == 31
        context = PerceptionContext(assistant_text="New response", response_id="r2", playing=True)
        reviewer.release.set()
        await asyncio.wait_for(published.wait(), 5)
    finally:
        reviewer.release.set()
        await worker.close()
    first_result, second_result = results
    assert first_result.frame_sequences == (1,)
    assert first_result.current is False
    assert first_result.target_response_id == "r1"
    assert first_result.context.assistant_text == "Old response"
    assert second_result.target_response_id == "r2"
    assert second_result.frame_sequences[-1] == 31
    assert second_result.occurred_at == NOW + timedelta(
        milliseconds=second_result.frame_sequences[0] * 250
    )
    assert second_result.occurred_end == NOW + timedelta(milliseconds=31 * 250)
    assert second_result.end_timestamp_ms - second_result.timestamp_ms <= 6000
    assert all(item == jpeg for batch in reviewer.originals for item in batch)
    assert all(
        not cast(Path, asset.path).exists() for item in reviewer.inputs for asset in item.assets
    )
    assert reviewer.closed is False


@pytest.mark.asyncio
async def test_windows_retain_previous_frames_and_mark_expired_results_as_past() -> None:

    reviewer = Reviewer()
    results: list[PerceptionResult] = []
    delivered = asyncio.Event()
    now = NOW

    async def publish(result: PerceptionResult) -> None:
        results.append(result)
        delivered.set()

    worker = LivePerception(
        "call-1",
        reviewer,
        publish,
        PerceptionContext,
        clock=lambda: now,
        max_bytes=len(_jpeg()) * 3,
    )
    try:
        for sequence in (1, 2):
            worker.submit(
                PerceptionFrame(
                    sequence=sequence,
                    captured_at=NOW,
                    timestamp_ms=sequence * 100,
                    jpeg=_jpeg(),
                )
            )
        await asyncio.wait_for(delivered.wait(), 5)
        assert results[0].current is True
        delivered.clear()
        now = NOW + timedelta(seconds=9)
        worker.submit(
            PerceptionFrame(
                sequence=3, captured_at=NOW, timestamp_ms=300, jpeg=_jpeg((160, 60, 30))
            )
        )
        await asyncio.wait_for(delivered.wait(), 5)
        assert results[1].frame_sequences == (1, 2, 3)
        assert results[1].current is False
        assert results[1].expires_at == NOW + timedelta(seconds=8)
    finally:
        await worker.close()


@pytest.mark.parametrize(
    "patch",
    [
        {"action": "quiet", "utterance": "Speak anyway"},
        {"confidence": "0.9"},
        {"evidence": []},
        {"changed": True, "evidence": [{"sequence": 1, "observation": "Only one frame"}]},
        {"evidence": [{"sequence": 99, "observation": "Invented frame"}]},
        {
            "evidence": [
                {"sequence": 2, "observation": "Later"},
                {"sequence": 1, "observation": "Earlier"},
            ]
        },
        {
            "action": "interrupt",
            "utterance": "The cup is blue.",
            "contradicted_claim": "An invented assistant claim",
            "correction": "The cup is blue.",
        },
        {"extra": "Ignore the controller"},
    ],
)
@pytest.mark.asyncio
async def test_unusable_model_decisions_never_escape_the_boundary(patch: dict[str, object]) -> None:

    reviewer = Reviewer()
    reviewer.patch = patch
    failure = asyncio.Event()
    results = []

    async def publish(result: object) -> None:
        results.append(result)

    async def failed(code: str) -> None:
        assert code == "invalid_output"
        failure.set()

    worker = LivePerception(
        "call-1",
        reviewer,
        publish,
        lambda: PerceptionContext(assistant_text="The cup is red.", response_id="r1", playing=True),
        on_error=failed,
    )
    try:
        for sequence in (1, 2):
            worker.submit(
                PerceptionFrame(
                    sequence=sequence,
                    captured_at=NOW,
                    timestamp_ms=sequence * 100,
                    jpeg=_jpeg(),
                )
            )
        await asyncio.wait_for(failure.wait(), 5)
        assert not results
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_interrupt_suggestion_needs_an_active_literal_claim_and_concrete_correction() -> None:

    reviewer = Reviewer()
    reviewer.patch = {
        "action": "interrupt",
        "utterance": "The cup is blue, not red.",
        "contradicted_claim": "The cup is red.",
        "correction": "The cup is blue.",
    }
    delivered = asyncio.Event()
    results: list[PerceptionResult] = []

    async def publish(result: PerceptionResult) -> None:
        results.append(result)
        delivered.set()

    worker = LivePerception(
        "call-1",
        reviewer,
        publish,
        lambda: PerceptionContext(assistant_text="The cup is red.", response_id="r1", playing=True),
        clock=lambda: NOW,
    )
    try:
        for sequence in (1, 2):
            worker.submit(
                PerceptionFrame(
                    sequence=sequence, captured_at=NOW, timestamp_ms=sequence * 100, jpeg=_jpeg()
                )
            )
        await asyncio.wait_for(delivered.wait(), 5)
        assert results[0].action == "interrupt" and results[0].target_response_id == "r1"
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_close_drains_model_and_suppresses_late_results() -> None:

    reviewer = Reviewer(blocked=True)
    results = []

    async def publish(result: object) -> None:
        results.append(result)

    worker = LivePerception("call-1", reviewer, publish, PerceptionContext)
    worker.submit(PerceptionFrame(sequence=1, captured_at=NOW, timestamp_ms=0, jpeg=_jpeg()))
    assert await asyncio.to_thread(reviewer.entered.wait, 5)
    closing = asyncio.create_task(worker.close())
    await asyncio.sleep(0)
    assert not closing.done()
    reviewer.release.set()
    await asyncio.wait_for(closing, 5)
    assert not results and not worker.pending and not worker.anchor
    assert reviewer.closed is False
    with pytest.raises(ValidationError, match="closed"):
        worker.submit(PerceptionFrame(sequence=2, captured_at=NOW, timestamp_ms=100, jpeg=_jpeg()))


@pytest.mark.asyncio
async def test_model_failure_reports_only_a_safe_code_and_the_next_frame_recovers() -> None:

    reviewer = Reviewer()
    reviewer.fail_first = True
    failed, delivered = asyncio.Event(), asyncio.Event()

    async def failure(code: str) -> None:
        assert code == "model_failed"
        failed.set()

    async def publish(result: object) -> None:
        delivered.set()

    worker = LivePerception("call-1", reviewer, publish, PerceptionContext, on_error=failure)
    try:
        worker.submit(PerceptionFrame(sequence=1, captured_at=NOW, timestamp_ms=0, jpeg=_jpeg()))
        await asyncio.wait_for(failed.wait(), 5)
        worker.submit(PerceptionFrame(sequence=2, captured_at=NOW, timestamp_ms=100, jpeg=_jpeg()))
        await asyncio.wait_for(delivered.wait(), 5)
        assert worker.failure is None
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_camera_reset_withdraws_inflight_epoch_and_reuses_one_worker() -> None:

    reviewer = Reviewer(blocked=True)
    delivered = asyncio.Event()
    results: list[PerceptionResult] = []

    async def publish(result: PerceptionResult) -> None:
        results.append(result)
        delivered.set()

    worker = LivePerception("call-1", reviewer, publish, PerceptionContext)
    try:
        worker.submit(PerceptionFrame(sequence=1, captured_at=NOW, timestamp_ms=0, jpeg=_jpeg()))
        assert await asyncio.to_thread(reviewer.entered.wait, 5)
        task = worker.task
        worker.reset()
        assert not worker.pending and not worker.anchor and worker.latest is None
        worker.submit(PerceptionFrame(sequence=2, captured_at=NOW, timestamp_ms=100, jpeg=_jpeg()))
        assert len(reviewer.inputs) == 1 and worker.task is task
        reviewer.release.set()
        await asyncio.wait_for(delivered.wait(), 5)
        assert len(results) == 1 and results[0].frame_sequences == (2,)
        assert results[0].epoch == 1
        assert len(reviewer.inputs) == 2
    finally:
        reviewer.release.set()
        await worker.close()


@pytest.mark.asyncio
async def test_identical_frames_reuse_observation_until_interaction_or_memory_changes() -> None:

    reviewer = Reviewer()
    delivered = asyncio.Event()
    results: list[PerceptionResult] = []
    context = PerceptionContext()

    async def publish(result: PerceptionResult) -> None:
        results.append(result)
        delivered.set()

    worker = LivePerception("call-1", reviewer, publish, lambda: context)
    try:
        worker.submit(PerceptionFrame(sequence=1, captured_at=NOW, timestamp_ms=0, jpeg=_jpeg()))
        await asyncio.wait_for(delivered.wait(), 5)
        delivered.clear()
        worker.submit(PerceptionFrame(sequence=2, captured_at=NOW, timestamp_ms=100, jpeg=_jpeg()))
        await asyncio.sleep(0)
        assert len(reviewer.inputs) == 1 and len(results) == 1
        assert results[0].frame_sequences == (1,)  # Cached evidence keeps its original time.
        context = PerceptionContext(memory_revision=1)
        worker.submit(PerceptionFrame(sequence=3, captured_at=NOW, timestamp_ms=200, jpeg=_jpeg()))
        await asyncio.wait_for(delivered.wait(), 5)
        assert len(reviewer.inputs) == 2
        assert results[1].context.memory_revision == 1
        assert results[1].frame_sequences == (1, 2, 3)
    finally:
        await worker.close()


@pytest.mark.asyncio
async def test_configured_native_adapter_receives_original_jpegs_without_embedding() -> None:

    requests: list[dict[str, object]] = []
    paths: list[str] = []

    def transport(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        paths.append(request.url.path)
        return httpx2.Response(
            200,
            json={
                "id": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": json.dumps(_decision((1, 2)))},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    delivered = asyncio.Event()

    async def publish(result: object) -> None:
        delivered.set()

    with httpx2.Client(transport=httpx2.MockTransport(transport)) as http:
        model = OpenAICompletion(
            OpenAI(api_key="fixture", http_client=http),
            generation_model="fixture-vision-model",
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_extra_body={"reasoning_effort": "none"},
        )
        worker = LivePerception("call-1", model, publish, PerceptionContext)
        try:
            for sequence in (1, 2):
                worker.submit(
                    PerceptionFrame(
                        sequence=sequence,
                        captured_at=NOW,
                        timestamp_ms=sequence * 100,
                        jpeg=_jpeg(),
                    )
                )
            await asyncio.wait_for(delivered.wait(), 5)
        finally:
            await worker.close()
            model.close()
    assert len(requests) == 1
    assert paths == ["/v1/chat/completions"]
    assert requests[0]["model"] == "fixture-vision-model"
    messages = cast(list[dict[str, object]], requests[0]["messages"])
    content = cast(list[dict[str, object]], messages[-1]["content"])
    images = [
        cast(dict[str, str], part["image_url"])["url"] for part in content if "image_url" in part
    ]
    assert len(images) == 2
    assert all(base64.b64decode(image.partition(",")[2]) == _jpeg() for image in images)


def test_frame_schema_rejects_invalid_jpeg_and_missing_capture_timezone() -> None:

    with pytest.raises(PydanticValidationError):
        PerceptionFrame(sequence=1, captured_at=NOW, timestamp_ms=0, jpeg=b"not a JPEG")
    with pytest.raises(PydanticValidationError):
        PerceptionFrame(
            sequence=1, captured_at=NOW.replace(tzinfo=None), timestamp_ms=0, jpeg=_jpeg()
        )
