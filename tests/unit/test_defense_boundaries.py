"""Failures, validation and normalization have one explicit owner per boundary."""

import sqlite3
import struct
from collections.abc import Callable, Sequence
from contextlib import closing
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from test_automatic_routing import ALL, TEXT, Answerer, Describer, Embedder, Transcriber

from sensemem import (
    AnswerResult,
    Blob,
    EmbedTask,
    Memory,
    MemoryConfig,
    MemoryIntent,
    MemoryOperation,
    MemoryPlugins,
    ModelError,
    ModelInput,
    ObservationContext,
    SearchHit,
    SentenceTransformersEmbedder,
    SpatialAnchor,
    SpatialContext,
    StorageError,
    ValidationError,
)
from sensemem.infrastructure.local.assets import AssetStoreError
from sensemem.infrastructure.local.store import LocalStoreCapacityError, StoredDataError
from sensemem.kernel import runtime


def test_invalid_answer_evidence_is_rejected_at_the_model_boundary(tmp_path: Path) -> None:
    class InvalidAnswerer(Answerer):
        def answer(
            self, question: ModelInput, hits: Sequence[SearchHit], **kwargs: object
        ) -> AnswerResult:
            return AnswerResult("invalid", hits=cast(tuple[SearchHit, ...], ("not a hit",)))

    with Memory(tmp_path, embedder=TinyEmbedder(), answerer=InvalidAnswerer()) as memory:
        memory.add("blue toolbox")
        with pytest.raises(ModelError) as failed:
            memory.ask("blue toolbox")
        assert failed.value.reason == "response_invalid"


class _FailedVision(Describer):
    def __init__(self, error: ModelError) -> None:
        super().__init__()
        self.error = error

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        self.calls += 1
        raise self.error


@pytest.mark.parametrize("operation", ("add", "capture", "search", "ask", "compile"))
def test_required_visual_conversion_preserves_failure(tmp_path: Path, operation: str) -> None:
    error = ModelError("private body", reason="quota_exhausted", stage="describe")
    vision = _FailedVision(error)
    with Memory(
        tmp_path,
        embedder=Embedder(TEXT),
        vision_describer=vision,
        answerer=Answerer(),
        config=MemoryConfig(minimum_relevance=0),
    ) as memory:
        image = Blob(b"query", "image/png")
        call: Callable[[], object]
        if operation == "capture":
            record = memory.capture(image)
            call = partial(memory.settle, memory_ids=(record.id,))
        elif operation == "add":
            call = partial(memory.add, image)
        else:
            memory.add("ALPHA-947 registration desk")
            call = partial(getattr(memory, operation), ("Where?", image))
        with pytest.raises(ModelError) as failure:
            call()
        assert failure.value is error
        assert vision.calls == 1


def test_optional_query_failure_is_reused_as_required_answer_failure(tmp_path: Path) -> None:
    error = ModelError("private body", reason="quota_exhausted")
    vision = _FailedVision(error)
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=Answerer(),
        vision_describer=vision,
        config=MemoryConfig(query_vision=True, minimum_relevance=0),
    ) as memory:
        memory.add("ALPHA-947 registration desk")
        with pytest.raises(ModelError) as failure:
            memory.ask(Blob(b"query", "image/png"))
        assert failure.value is error
        assert vision.calls == 1, "escalating a cached failure must not buy a second attempt"


def test_successful_transcription_makes_video_description_optional(tmp_path: Path) -> None:
    vision = _FailedVision(ModelError("private body", reason="quota_exhausted"))
    with Memory(
        tmp_path,
        embedder=Embedder(TEXT),
        answerer=Answerer(),
        transcriber=Transcriber(),
        vision_describer=vision,
        config=MemoryConfig(query_vision=True, minimum_relevance=0),
    ) as memory:
        source = memory.add(Blob(b"stored clip", "video/mp4"))
        assert "ALPHA-947" in source.content
        assert memory.ask(Blob(b"query clip", "video/mp4")).answer


def test_storage_translation_does_not_disguise_programming_failures() -> None:
    error = TypeError("programming failure")
    with pytest.raises(TypeError) as failure, runtime.translate_storage_errors("read records"):
        raise error
    assert failure.value is error


@pytest.mark.parametrize(
    "error",
    (
        sqlite3.OperationalError("database unavailable"),
        OSError("disk failed"),
        AssetStoreError("asset unavailable"),
        StoredDataError("invalid persisted value"),
    ),
)
def test_actual_storage_failures_keep_the_storage_error_contract(error: Exception) -> None:
    with pytest.raises(StorageError) as failure, runtime.translate_storage_errors("read records"):
        raise error
    assert failure.value.reason == "io_failed"
    assert failure.value.__cause__ is error


def test_storage_capacity_is_separate_from_io_failure() -> None:
    with pytest.raises(StorageError) as failure, runtime.translate_storage_errors("read records"):
        raise LocalStoreCapacityError("pool exhausted")
    assert failure.value.reason == "capacity_exceeded"


def test_store_open_does_not_reclassify_an_attribute_bug(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_path: Path) -> None:
        raise AttributeError("initialization bug")

    monkeypatch.setattr(runtime, "LocalStore", broken)
    with pytest.raises(AttributeError, match="initialization bug"):
        runtime.open_store(tmp_path)


def test_persisted_metadata_validation_survives_narrow_exception_mapping(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add("stored observation")
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
            connection.execute("PRAGMA ignore_check_constraints=ON")
            connection.execute("UPDATE memory_records SET metadata_json = 'broken'")
            connection.commit()
        with pytest.raises(StorageError) as failure:
            memory.get(record.id)
        assert failure.value.reason == "io_failed"
        assert isinstance(failure.value.__cause__, StoredDataError)


@pytest.mark.parametrize("damage", ("context", "spatial", "operation"))
def test_persisted_semantics_and_log_decoding_are_storage_failures(
    tmp_path: Path, damage: str
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add(
            "stored observation",
            context=ObservationContext(
                spatial=SpatialContext(frame_id="map", anchor=SpatialAnchor.SUBJECT, x=0, y=0)
            ),
        )
        call: Callable[[], object] = partial(memory.get, record.id)
        if damage == "operation":
            memory.apply(MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(record.id,)))
            call = memory.operations
        statements = {
            "context": "UPDATE memory_versions SET confidence = 2.0",
            "spatial": "UPDATE memory_semantics SET spatial_anchor = 'invalid'",
            "operation": "UPDATE memory_operations SET effects_json = 'broken'",
        }
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
            connection.execute("PRAGMA ignore_check_constraints=ON")
            connection.execute(statements[damage])
            connection.commit()
        with pytest.raises(StorageError) as failure:
            call()
        assert failure.value.reason == "io_failed"
        assert isinstance(failure.value.__cause__, StoredDataError)


class _RawEncoder:
    def __init__(self, values: tuple[float, ...] = (3.0, 4.0)) -> None:
        self.values = values

    def supports(self, modality: str | tuple[str, ...]) -> bool:
        return modality == "text"

    def get_embedding_dimension(self) -> int:
        return 2

    def encode_query(
        self,
        sentences: list[object],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        truncate_dim: int | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(tolist=lambda: [list(self.values) for _ in sentences])

    encode_document = encode_query


def test_capability_probe_failures_remain_failures() -> None:
    class Broken(_RawEncoder):
        def supports(self, modality: str | tuple[str, ...]) -> bool:
            raise RuntimeError("probe failed")

    with pytest.raises(RuntimeError, match="probe failed"):
        SentenceTransformersEmbedder(Broken(), model_id="fixture", revision="a" * 40)


def test_capability_probe_requires_an_actual_boolean() -> None:
    class Invalid(_RawEncoder):
        def supports(self, modality: str | tuple[str, ...]) -> bool:
            return cast(bool, None)

    with pytest.raises(ModelError, match="boolean"):
        SentenceTransformersEmbedder(Invalid(), model_id="fixture", revision="a" * 40)


def test_embedding_normalization_is_owned_by_the_kernel(tmp_path: Path) -> None:
    encoder = _RawEncoder()
    embedder = SentenceTransformersEmbedder(encoder, model_id="fixture", revision="a" * 40)
    assert embedder.embed((ModelInput(text="raw"),)) == ((3.0, 4.0),)
    with Memory(tmp_path, embedder=embedder, config=MemoryConfig(minimum_relevance=0)) as memory:
        record = memory.add("source observation")
        assert memory.search("source").hits[0].id == record.id
        with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
            blobs = connection.execute("SELECT vector FROM embeddings").fetchall()
        assert blobs
        assert all(struct.unpack("<2f", blob[0]) == pytest.approx((0.6, 0.8)) for blob in blobs)


@pytest.mark.parametrize("values", ((0.0, 0.0), (float("nan"), 1.0), (float("inf"), 1.0)))
def test_invalid_vectors_never_become_durable(tmp_path: Path, values: tuple[float, ...]) -> None:
    class Invalid(Embedder):
        embedding_dimension = 2
        embedding_space = "invalid-fixture:2"

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            return tuple(values for _ in inputs)

    with Memory(tmp_path, embedder=Invalid(TEXT)) as memory:
        with pytest.raises(ModelError) as failure:
            memory.add("source")
        assert failure.value.reason == "response_invalid"
        assert not memory.list().items


def test_capability_document_uses_the_assembled_snapshot(tmp_path: Path) -> None:
    backend = Embedder(TEXT)
    with Memory(tmp_path, embedder=backend, config=MemoryConfig(minimum_relevance=0)) as memory:
        backend.embedding_capabilities = frozenset()
        assert memory.capabilities.embedding == TEXT
        memory.add("source")
        assert memory.search("source").hits


@pytest.mark.parametrize(
    ("setting", "config"),
    (
        ("speaker_similarity", MemoryConfig(speaker_similarity=float("nan"))),
        ("speaker_margin", MemoryConfig(speaker_margin=float("nan"))),
        ("face_similarity", MemoryConfig(face_similarity=float("nan"))),
        ("face_margin", MemoryConfig(face_margin=float("nan"))),
    ),
)
def test_thresholds_are_rejected_at_assembly_before_storage(
    tmp_path: Path, setting: str, config: MemoryConfig
) -> None:
    with pytest.raises(ValidationError, match=setting):
        Memory.from_plugins(
            tmp_path,
            plugins=MemoryPlugins(embedder=TinyEmbedder()),
            config=config,
        )
    assert not (tmp_path / "state.sqlite3").exists()
