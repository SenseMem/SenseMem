"""The live example shares its native Omni adapter and retains original A/V inputs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AsyncEventStream,
    AsyncMemory,
    Blob,
    DefaultQueryPolicy,
    EmbedTask,
    EventChunk,
    EventObservation,
    EventSegment,
    EventSegmentationBackend,
    GenerationBackend,
    GenerationRouter,
    MemoryComposition,
    MemoryPlugins,
    MemorySettings,
    MemorySpec,
    Modality,
    ModelInput,
    QueryRepresentationPlan,
    QueryRequest,
    SearchHit,
    ValidationError,
)
from sensemem.models.openai_completion import OpenAICompletion

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)
NATIVE_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO})


@pytest.fixture(autouse=True)
def example_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))


class TrackingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.closed = False
        self.query_batches: list[tuple[ModelInput, ...]] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.QUERY:
            self.query_batches.append(tuple(inputs))
        return super().embed(inputs, task)

    def close(self) -> None:
        self.closed = True


class RecordingOmni(OpenAICompletion):
    def __init__(self) -> None:
        super().__init__(
            generation_model="qwen3.8-omni-flash",
            generation_capabilities=NATIVE_MODALITIES,
            generation_extra_body={"reasoning_effort": "none"},
        )
        self.calls: list[tuple[tuple[EventObservation, ...], bool]] = []

    def segment_events(
        self,
        observations: Sequence[EventObservation],
        *,
        end_of_stream: bool = False,
    ) -> tuple[EventSegment, ...]:
        self.calls.append((tuple(observations), end_of_stream))
        return (EventSegment(2, "The activity is complete."),) if len(observations) == 2 else ()


def _config(directory: Path) -> MemorySpec:
    return MemorySpec.model_validate({"data_dir": directory, "embedding": {"provider": "openai"}})


@pytest.mark.asyncio
async def test_runtime_shares_native_omni_and_preserves_original_av(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.live import events

    embedder = TrackingEmbedder()
    model = RecordingOmni()
    composition = MemoryComposition(
        tmp_path,
        MemoryPlugins(embedder=embedder, answerer=model),
        MemorySettings(),
    )
    config = _config(tmp_path)
    resolved: list[MemorySpec] = []

    def resolve(value: MemorySpec) -> MemoryComposition:
        resolved.append(value)
        return composition

    monkeypatch.setattr(events, "build_memory_composition", resolve)
    memory, segmenter = events.open_event_memory(config)
    blobs = (
        Blob(b"original microphone", "audio/wav"),
        Blob(b"original camera image", "image/png"),
        Blob(b"original camera video with sound", "video/webm"),
    )

    async def source() -> AsyncIterator[EventChunk]:
        yield EventChunk((blobs[0], blobs[1]), NOW, NOW + timedelta(seconds=3))
        yield EventChunk(blobs[2], NOW + timedelta(seconds=3), NOW + timedelta(seconds=6))

    try:
        assert isinstance(memory, AsyncMemory)
        assert isinstance(segmenter, EventSegmentationBackend)
        assert segmenter is model
        assert resolved == [config]
        assert composition.plugins.transcriber is None
        assert composition.plugins.vision_describer is None
        commits = [
            commit
            async for commit in AsyncEventStream(memory, segmenter=segmenter, capture=True).consume(
                source()
            )
        ]
        assert len(commits) == 1
        assert [len(observations) for observations, _eof in model.calls] == [1, 2]
        observations, eof = model.calls[-1]
        assert eof is False
        assert [(value.occurred_at, value.occurred_end) for value in observations] == [
            (NOW, NOW + timedelta(seconds=3)),
            (NOW + timedelta(seconds=3), NOW + timedelta(seconds=6)),
        ]
        assert all(not value.content.text for value in observations)
        record = commits[0].record
        assert [
            (await memory.read_asset(record.id, asset.id)).data
            for value in observations
            for asset in value.content.assets
        ] == [blob.data for blob in blobs]
        assert {asset.modality for asset in record.assets} == {
            Modality.AUDIO,
            Modality.IMAGE,
            Modality.VIDEO,
        }
        assert record.occurred_at == NOW
        assert record.occurred_end == NOW + timedelta(seconds=6)
    finally:
        await memory.close()
    assert embedder.closed is True


async def test_review_embeds_only_recall_key_and_keeps_every_original_for_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.live import events
    from examples.live import prompts as demo_prompts

    class OrdinaryPolicy(DefaultQueryPolicy):
        def __init__(self) -> None:
            self.requests: list[str] = []
            self.closed = 0

        def plan(self, request: QueryRequest) -> QueryRepresentationPlan:
            self.requests.append(request.content.text)
            return super().plan(request)

        def close(self) -> None:
            self.closed += 1

    class Reviewer(RecordingOmni):
        def __init__(self) -> None:
            super().__init__()
            self.reviewed: list[tuple[str, tuple[bytes, ...]]] = []

        def stream_answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> Generator[str, None, AnswerResult]:
            originals = []
            for asset in question.assets:
                assert asset.path is not None
                originals.append(asset.path.read_bytes())
            self.reviewed.append((question.text, tuple(originals)))
            result = AnswerResult("reviewed", hits=tuple(hits))
            yield result.answer
            return result

    embedder, model, ordinary = TrackingEmbedder(), Reviewer(), OrdinaryPolicy()
    composition = MemoryComposition(
        tmp_path,
        MemoryPlugins(embedder=embedder, answerer=model, query_policy=ordinary),
        MemorySettings(minimum_relevance=0, ambiguity_margin=0),
    )
    monkeypatch.setattr(events, "build_memory_composition", lambda _value: composition)
    memory, _segmenter = events.open_event_memory(_config(tmp_path))
    originals = tuple(
        Blob(f"original AV chunk {index}".encode(), "video/mp4") for index in range(24)
    )
    recall_key = "Keys moved into the white bowl.\nPlease remember where I put the keys."
    full_event = '{"summary": "' + "A long chronological native event. " * 600 + '"}'
    question = demo_prompts.proactive_question(full_event, recall_key)
    try:
        await memory.add("Earlier keys were in the blue toolbox")
        result = await memory.ask((question, *originals), limit=4, answer_policy="best_effort")
        assert not result.abstained
        assert embedder.query_batches == [(ModelInput(text=recall_key),)]
        assert ordinary.requests == []
        assert len(model.reviewed) == 1
        assert model.reviewed[0][0].startswith(question)
        assert model.reviewed[0][1] == tuple(blob.data for blob in originals)
        await memory.search(("ordinary camera query", originals[0]))
        assert ordinary.requests == ["ordinary camera query"]
        assert any(value.assets for batch in embedder.query_batches[1:] for value in batch)
    finally:
        await memory.close()
    await memory.close()
    assert ordinary.closed == 1 and embedder.closed


@pytest.mark.parametrize(
    "answerer",
    [None, GenerationRouter((OpenAICompletion(),)), OpenAICompletion()],
    ids=["missing", "answer-only", "text-only"],
)
def test_runtime_rejects_missing_native_capabilities_and_closes_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answerer: GenerationBackend | None,
) -> None:
    from examples.live import events

    embedder = TrackingEmbedder()
    composition = MemoryComposition(
        tmp_path,
        MemoryPlugins(embedder=embedder, answerer=answerer),
        MemorySettings(),
    )
    monkeypatch.setattr(events, "build_memory_composition", lambda _value: composition)
    with pytest.raises(ValidationError, match="the live example requires"):
        events.open_event_memory(_config(tmp_path))
    assert embedder.closed is True
    assert not tmp_path.joinpath("memory.sqlite3").exists()
