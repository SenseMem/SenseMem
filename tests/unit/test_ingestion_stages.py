"""Configured stages run on every ingestion route under ordinary durability and leases."""

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from test_formation_history import HistoryFormer

from sensemem import (
    Blob,
    EmbedTask,
    IndexQuantization,
    Memory,
    MemoryCategory,
    MemoryRecord,
    MemorySettings,
    ModelError,
    ModelInput,
    StageResult,
    StorageError,
    StreamInput,
)
from sensemem.infrastructure.local.store import (
    LocalStore,
    SpeechPlan,
    StaleOperationError,
    StoredEmbedding,
    StoredMemory,
)
from sensemem.infrastructure.local.store.records import MemoryRecords
from sensemem.kernel.content import prepare_memory
from sensemem.kernel.derived import with_stream_interpretation


class RecordingStage:
    name = "interpret"
    space = "interpret:v1"

    def __init__(self, process: Callable[[ModelInput], StageResult] | None = None) -> None:
        self.inputs: list[ModelInput] = []
        self.closed = 0
        self.process_input = process

    def process(self, input: ModelInput) -> StageResult:
        self.inputs.append(input)
        return (
            StageResult("Derived interpretation")
            if self.process_input is None
            else self.process_input(input)
        )

    def close(self) -> None:
        self.closed += 1


class RecordingEmbedder(TinyEmbedder):
    def __init__(self) -> None:
        self.inputs: list[ModelInput] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        self.inputs.extend(inputs)
        return super().embed(inputs, task=task)


@pytest.mark.parametrize("route", ["add", "many", "stream", "capture", "capture_stream"])
def test_every_write_route_processes_once_before_formation(tmp_path: Path, route: str) -> None:
    stage = RecordingStage()
    embedder = RecordingEmbedder()
    former = HistoryFormer()
    with Memory(
        tmp_path,
        embedder=embedder,
        ingestion_stages=(stage,),
        former=former,
    ) as memory:
        records: tuple[MemoryRecord, ...]
        if route == "add":
            records = (memory.add("First source"),)
        elif route == "many":
            records = memory.add_many(("First source", "Second source"))
        elif route == "stream":
            records = tuple(memory.add_stream(("First source", "Second source")))
        elif route == "capture":
            records = (memory.capture("First source"), memory.capture("Second source"))
        else:
            records = tuple(memory.add_stream(("First source", "Second source"), capture=True))
        if route in {"capture", "capture_stream"}:
            assert stage.inputs == []
            assert memory.settle() == len(records)
        assert len(stage.inputs) == len(records)
        assert any("Derived interpretation" in value.text for value in embedder.inputs)
        for record in records:
            stored = memory.get(record.id)
            assert stored.parts[0] == record.parts[0]
            assert stored.parts[0].value in {input.text for input in stage.inputs}
            assert stored.parts[-1].kind == "model_interpretation"
            assert "not original evidence" in stored.parts[-1].value
            assert "Derived interpretation" in stored.content
            assert memory.add(stored.parts[0].value).id == record.id
        assert len(stage.inputs) == len(records)
        assert len(former.inputs) == len(records)
        assert all(
            value.content.parts[-1].kind == "model_interpretation" for value in former.inputs
        )
    assert stage.closed == 1


def test_text_capture_batch_does_not_bypass_stages(tmp_path: Path) -> None:
    stage = RecordingStage()
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(stage,)) as memory:
        records = tuple(memory.capture(value) for value in ("First source", "Second source"))
        assert memory.settle() == 2
        assert {value.text for value in stage.inputs} == {"First source", "Second source"}
        assert all("Derived interpretation" in memory.get(record.id).content for record in records)


def test_stage_order_keeps_identity_and_leased_native_assets(tmp_path: Path) -> None:
    source = Blob(b"original bytes", "image/png", "source.png")
    with Memory(tmp_path / "ordinary", embedder=TinyEmbedder()) as ordinary:
        baseline = ordinary.add(("Original source", source))
    first = RecordingStage()
    second = RecordingStage()
    second.name = "second"
    second.space = "second:v1"
    with Memory(
        tmp_path / "processed", embedder=TinyEmbedder(), ingestion_stages=(first, second)
    ) as memory:
        record = memory.add(("Original source", source))
        assert record.id == baseline.id
        assert record.parts[:2] == baseline.parts
        assert tuple(asset.id for asset in record.assets) == tuple(
            asset.id for asset in baseline.assets
        )
        assert "Derived interpretation" in second.inputs[0].text
        assert first.inputs[0].assets[0].path is not None
        assert first.inputs[0].assets[0].path.read_bytes() == source.data
        assert [part.kind for part in record.parts] == [
            "text",
            "asset",
            "model_interpretation",
            "model_interpretation",
        ]


@pytest.mark.parametrize("failure", ["exception", "wrong_type", "invalid_value", "oversized"])
def test_stage_failure_is_safe_and_does_not_commit_batch(tmp_path: Path, failure: str) -> None:
    def fail(input: ModelInput) -> StageResult:
        if input.text == "First source":
            return StageResult()
        if failure == "exception":
            raise RuntimeError("private provider body")
        if failure == "wrong_type":
            return cast(StageResult, "private provider body")
        if failure == "invalid_value":
            return StageResult(cast(str, None))
        return StageResult("private provider body" * 5_000)

    stage = RecordingStage(fail)
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(stage,)) as memory:
        with pytest.raises(ModelError) as raised:
            memory.add_many(("First source", "Second source"))
        assert "private provider body" not in str(raised.value)
        assert raised.value.stage == "ingestion.process"
        assert memory.list().items == ()


def test_failed_settlement_keeps_capture_retryable(tmp_path: Path) -> None:
    def fail(input: ModelInput) -> StageResult:
        del input
        raise RuntimeError("private provider body")

    stage = RecordingStage(fail)
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(stage,)) as memory:
        record = memory.capture("Original source")
        with pytest.raises(ModelError):
            memory.settle()
        assert memory.get(record.id).content == "Original source"
        assert memory.pending_captures()[0].attempts == 1
        stage.process_input = None
        assert memory.settle() == 1
        assert "Derived interpretation" in memory.get(record.id).content
        assert memory.pending_captures() == ()


def test_identity_retry_reuses_stage_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = MemoryRecords.write_memories
    writes = 0

    def stale_once(
        self: MemoryRecords,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding] = (),
        *,
        formation_pending_at: datetime | None = None,
        speech_plans: Sequence[SpeechPlan] = (),
        replace_embeddings: bool = False,
    ) -> tuple[bool, ...]:
        nonlocal writes
        writes += 1
        if writes == 1:
            raise StaleOperationError("retry current identity state")
        return write(
            self,
            memories,
            embeddings,
            formation_pending_at=formation_pending_at,
            speech_plans=speech_plans,
            replace_embeddings=replace_embeddings,
        )

    monkeypatch.setattr(MemoryRecords, "write_memories", stale_once)
    stage = RecordingStage()
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(stage,)) as memory:
        record = memory.add("Original source")
        assert writes == 2
        assert len(stage.inputs) == 1
        assert len([part for part in record.parts if part.kind == "model_interpretation"]) == 1


@pytest.mark.parametrize("change", ["space", "name", "remove", "append"])
def test_nonempty_store_rejects_changed_stage_recipes(tmp_path: Path, change: str) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(RecordingStage(),)) as memory:
        memory.add("Original source")
    with LocalStore(tmp_path) as store:
        recipe = store.get_metadata("processing.ingestion_recipe")
        index_recipe = store.get_metadata("index.recipe")
    sentinel = tmp_path / "zvec" / "preserved"
    sentinel.write_text("existing derived index", encoding="utf-8")
    stage = RecordingStage()
    if change == "space":
        stage.space = "interpret:v2"
    if change == "name":
        stage.name = "renamed"
    additional = RecordingStage()
    additional.name = "additional"
    stages = () if change == "remove" else (stage, additional) if change == "append" else (stage,)
    with pytest.raises(StorageError, match="replay sources into a fresh data directory"):
        Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            ingestion_stages=stages,
            settings=MemorySettings(index_quantization=IndexQuantization.FP16),
        )
    assert stage.inputs == []
    assert sentinel.read_text(encoding="utf-8") == "existing derived index"
    with LocalStore(tmp_path) as store:
        assert store.get_metadata("processing.ingestion_recipe") == recipe
        assert store.get_metadata("index.recipe") == index_recipe
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(RecordingStage(),)) as memory:
        assert len(memory.list().items) == 1


def test_unchanged_recipe_reopens_without_processing_or_reembedding(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(RecordingStage(),)) as memory:
        record = memory.add("Original source")
    stage = RecordingStage()
    embedder = RecordingEmbedder()
    with Memory(tmp_path, embedder=embedder, ingestion_stages=(stage,)) as memory:
        assert memory.add("Original source") == record
        assert stage.inputs == []
        assert embedder.inputs == []
    with LocalStore(tmp_path) as store:
        assert json.loads(store.get_metadata("processing.ingestion_recipe") or "null") == [
            {"name": "interpret", "space": "interpret:v1"}
        ]


def test_empty_store_can_change_stage_recipes(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(RecordingStage(),)):
        pass
    stage = RecordingStage()
    stage.space = "interpret:v2"
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(stage,)) as memory:
        assert "Derived interpretation" in memory.add("Original source").content


def test_stages_cannot_be_added_after_unprocessed_observations_exist(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        memory.add("Original source")
    with pytest.raises(StorageError, match="ingestion stage recipes changed"):
        Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(RecordingStage(),))


@pytest.mark.parametrize("with_stages", [False, True])
def test_nonempty_store_requires_its_ingestion_recipe(tmp_path: Path, with_stages: bool) -> None:
    stages = (RecordingStage(),) if with_stages else ()
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=stages) as memory:
        memory.add("Original source")
    with LocalStore(tmp_path) as store:
        store.update_metadata({"processing.ingestion_recipe": None})
    sentinel = tmp_path / "zvec" / "preserved"
    sentinel.write_text("existing derived index", encoding="utf-8")

    with (
        pytest.raises(StorageError, match="replay sources into a fresh data directory"),
        Memory(tmp_path, embedder=TinyEmbedder()),
    ):
        pass

    assert sentinel.read_text(encoding="utf-8") == "existing derived index"
    with LocalStore(tmp_path) as store:
        assert store.get_metadata("processing.ingestion_recipe") is None
        assert len(store.records.list_memories(limit=100)) == 1


def test_empty_stage_result_keeps_only_original_content(tmp_path: Path) -> None:
    stage = RecordingStage(lambda input: StageResult())
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(stage,)) as memory:
        record = memory.add("Original source")
        assert record.content == "Original source"
        assert len(record.parts) == 1
        assert record.parts[0].kind == "text"
        assert memory.add("Original source") == record
        assert len(stage.inputs) == 1


@pytest.mark.parametrize("capture", [False, True])
def test_transcript_refresh_replaces_pipeline_interpretations_without_feedback(
    tmp_path: Path, capture: bool
) -> None:
    first = RecordingStage(lambda input: StageResult("First stage interpretation"))
    second = RecordingStage(lambda input: StageResult("Second stage interpretation"))
    second.name = "second"
    # An original quote resembling our generated header must remain original evidence.
    quoted = (
        '[processing interpretation {"name": "interpret", "space": "interpret:v1"}'
        ": not original evidence]\nFirst stage interpretation"
    )
    source = (quoted, Blob(b"refreshable audio", "audio/wav"))
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(first, second)) as memory:
        original = next(
            memory.add_stream(
                (
                    StreamInput(
                        source, transcript="Old transcript", interpretation="Stream synopsis"
                    ),
                ),
                capture=capture,
            )
        )
        if capture:
            assert memory.settle() == 1
        refreshed = next(
            memory.add_stream(
                (StreamInput(source, transcript="Updated transcript"),), capture=capture
            )
        )
        if capture:
            assert memory.settle() == 1
        record = memory.get(refreshed.id)
        assert record.id == original.id
        assert record.parts[:2] == original.parts[:2]
        assert record.parts[0].value == quoted
        interpretations = tuple(
            part for part in record.parts if part.kind == "model_interpretation"
        )
        assert len(interpretations) == 3
        assert sum("First stage interpretation" in part.value for part in interpretations) == 1
        assert sum("Second stage interpretation" in part.value for part in interpretations) == 1
        assert sum("Stream synopsis" in part.value for part in interpretations) == 1
        assert len(first.inputs) == len(second.inputs) == 2
        # Inspect typed derivations; the original quote is intentionally equal to a stage label.
        assert [
            part.value for part in first.inputs[1].parts if part.kind == "model_interpretation"
        ] == ["[model interpretation: not original evidence]\nStream synopsis"]
        assert "Updated transcript" in first.inputs[1].text
        assert "Old transcript" not in first.inputs[1].text
        assert (
            sum(
                "First stage interpretation" in part.value
                for part in second.inputs[1].parts
                if part.kind == "model_interpretation"
            )
            == 1
        )
        assert not any(
            "Second stage interpretation" in part.value
            for part in second.inputs[1].parts
            if part.kind == "model_interpretation"
        )


def test_retry_cache_changes_only_when_effective_stage_input_changes(tmp_path: Path) -> None:
    stage = RecordingStage(lambda input: StageResult(f"Interpreted {input.text}"))
    with Memory(tmp_path, embedder=TinyEmbedder(), ingestion_stages=(stage,)) as memory:
        with memory._lifecycle.operation() as operation:
            prepared = prepare_memory(
                memory._materializer.materialize_content("Original source", operation),
                occurred_at=None,
                occurred_end=None,
                metadata=None,
                memory_type=MemoryCategory.SEMANTIC,
            )
            initial = replace(
                prepared, content=with_stream_interpretation(prepared.content, "First synopsis")
            )
            first = memory._ingestion._processing.process_memory(initial, operation)
            assert memory._ingestion._processing.process_memory(first, operation) == first
            updated = replace(
                prepared, content=with_stream_interpretation(prepared.content, "Updated synopsis")
            )
            refreshed = memory._ingestion._processing.process_memory(updated, operation)
        assert refreshed.memory_id == first.memory_id
        assert len(stage.inputs) == 2
        assert "Updated synopsis" in refreshed.content.parts[-1].value
        assert "First synopsis" not in refreshed.content.parts[-1].value
