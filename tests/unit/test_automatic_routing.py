"""Automatic routing and bounded concurrency through the public SDK."""

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier, Event, Lock
from typing import cast

import pytest

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    Blob,
    EmbedTask,
    FormationInput,
    FormationProposal,
    IndexUnavailableError,
    Memory,
    MemoryConfig,
    Modality,
    ModelError,
    ModelInput,
    RetrievalMode,
    SearchHit,
)
from sensemem.kernel.ingestion import Ingestion
from sensemem.models.base import SpeechAnalysis, SpeechTurn
from sensemem.models.sentence_transformers import SentenceTransformersEmbedder

TEXT = frozenset({Modality.TEXT})
ALL = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


class Embedder:
    embedding_model = "audit-fixture"
    embedding_space = "audit-fixture:4"
    embedding_dimension = 4

    def __init__(self, capabilities: frozenset[Modality] = TEXT) -> None:
        self.embedding_capabilities = capabilities
        self.batches: list[tuple[ModelInput, ...]] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        if task is EmbedTask.DOCUMENT:
            self.batches.append(tuple(inputs))
        return ((1.0, 0.0, 0.0, 0.0),) * len(inputs)

    def close(self) -> None:
        pass


class Describer:
    vision_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    vision_model = "audit-describer"
    vision_space = "audit-describer:v1"

    def __init__(self) -> None:
        self.calls = 0

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        self.calls += 1
        return ("ALPHA-947 registration desk",) * len(inputs)

    def close(self) -> None:
        pass


class Transcriber:
    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "audit-transcriber"
    transcription_space = "audit-transcriber:v1"

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        return ("ALPHA-947 registration desk",) * len(assets)

    def close(self) -> None:
        pass


class Speech:
    transcription_capabilities = Transcriber.transcription_capabilities
    transcription_model = "audit-speech"
    transcription_space = "audit-speech:v1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        return tuple(
            SpeechAnalysis((SpeechTurn(0, 100, "ALPHA-947 registration desk"),), ()) for _ in assets
        )

    def close(self) -> None:
        pass


class Answerer:
    generation_capabilities = TEXT

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return AnswerResult(answer="registration desk", hits=tuple(hits))

    def close(self) -> None:
        pass


class Former:
    formation_capabilities = TEXT
    formation_model = "audit-former"
    formation_space = "audit-former:v1"

    def __init__(self) -> None:
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        return ((),) * len(inputs)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("mode", (RetrievalMode.HYBRID, RetrievalMode.DENSE))
def test_image_description_rescues_a_text_embedder_query(
    tmp_path: Path, mode: RetrievalMode
) -> None:
    describer = Describer()
    with Memory(
        tmp_path,
        embedder=Embedder(),
        vision_describer=describer,
        config=MemoryConfig(
            query_vision=True, retrieval_mode=mode, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        # The same composition accepts a visual write through derived text.
        memory.add(("ALPHA-947", Blob(b"stored-image", "image/png")))
        before = describer.calls
        assert memory.search(("ALPHA-947", Blob(b"query-image", "image/png"))).hits
        assert describer.calls == before + 1


def test_query_caption_reaches_a_text_answerer(tmp_path: Path) -> None:
    describer = Describer()
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=Answerer(),
        vision_describer=describer,
        config=MemoryConfig(query_vision=True, minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add("ALPHA-947 registration desk")
        assert memory.ask(("ALPHA-947", Blob(b"query-image", "image/png"))).answer
        assert describer.calls == 1


@pytest.mark.parametrize("media_question", (False, True))
def test_video_transcript_rescues_add_search_and_ask(tmp_path: Path, media_question: bool) -> None:
    with Memory(
        tmp_path,
        embedder=Embedder(),
        answerer=Answerer(),
        transcriber=Transcriber(),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        record = memory.add(Blob(b"stored-video", "video/mp4"))
        assert memory.search(Blob(b"query-video", "video/mp4")).hits[0].id == record.id
        question = (
            ("ALPHA-947", Blob(b"query-video", "video/mp4")) if media_question else "ALPHA-947"
        )
        assert memory.ask(question).answer


def test_speech_plugins_share_the_video_transcription_route(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=Embedder(), transcriber=Speech()) as memory:
        assert memory.add(Blob(b"video", "video/mp4")).content
        # Audio does have a working fallback in this same composition.
        assert memory.add(Blob(b"audio", "audio/wav")).content


def test_captioned_images_reach_a_text_former(tmp_path: Path) -> None:
    former = Former()
    with Memory(
        tmp_path, embedder=Embedder(), vision_describer=Describer(), former=former
    ) as memory:
        memory.add("ALPHA-947")
        assert former.calls == 1
        image = memory.add(Blob(b"image", "image/png"))
        assert "ALPHA-947" in image.content
        assert former.calls == 2
        # The capture queue also reports no remaining formation work.
        assert memory.pending_captures() == ()


def test_speech_indexing_does_not_lock_the_instance_during_embedding(tmp_path: Path) -> None:
    entered, release, query_entered = Event(), Event(), Event()

    class BlockingEmbedder(Embedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            if task is EmbedTask.DOCUMENT and any(value.assets for value in inputs):
                entered.set()
                assert release.wait(10)
            if task is EmbedTask.QUERY:
                query_entered.set()
            return super().embed(inputs, task)

    with Memory(
        tmp_path,
        embedder=BlockingEmbedder(ALL),
        transcriber=Speech(),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add("ALPHA-947")
        with ThreadPoolExecutor(max_workers=2) as pool:
            adding = pool.submit(memory.add, Blob(b"audio", "audio/wav"))
            try:
                assert entered.wait(5)
                acquired = memory._write_lock.acquire(blocking=False)
                if acquired:
                    memory._write_lock.release()
                assert acquired
                searching = pool.submit(memory.search, "ALPHA-947")
                assert query_entered.wait(5)
                assert searching.result(timeout=5).hits
            finally:
                release.set()
            assert adding.result(timeout=5).id
            assert searching.result(timeout=5).hits


def test_settle_batches_text_captures(tmp_path: Path) -> None:
    embedder = Embedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        for text in ("first", "second", "third"):
            memory.capture(text)
        assert memory.settle() == 3
        assert [len(batch) for batch in embedder.batches] == [3]


def test_waiting_settlement_resumes_from_committed_enrichment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release, waiting = Event(), Event(), Event()
    lock = Lock()
    claim_count = 0
    original = Ingestion._claim_settlement

    @contextmanager
    def claim(plane: Ingestion, ids: Sequence[str]) -> Iterator[None]:
        nonlocal claim_count
        with lock:
            claim_count += 1
            if claim_count == 2:
                waiting.set()
        with original(plane, ids):
            yield

    class BlockingEmbedder(Embedder):
        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            entered.set()
            assert release.wait(10)
            return super().embed(inputs, task)

    class RetryFormer(Former):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            self.calls += 1
            assert all("ALPHA-947" in value.content.text for value in inputs)
            if self.calls == 1:
                raise ModelError("retry formation")
            return ((),) * len(inputs)

    former = RetryFormer()
    monkeypatch.setattr(Ingestion, "_claim_settlement", claim)
    with (
        Memory(
            tmp_path, embedder=BlockingEmbedder(), vision_describer=Describer(), former=former
        ) as memory,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        record = memory.capture(Blob(b"image", "image/png"))
        first = pool.submit(memory.settle, memory_ids=(record.id,))
        assert entered.wait(5)
        second = pool.submit(memory.settle, memory_ids=(record.id,))
        try:
            assert waiting.wait(5)
        finally:
            release.set()
        with pytest.raises(ModelError):
            first.result(timeout=10)
        assert second.result(timeout=10) == 1
        assert former.calls == 2
        assert memory.pending_captures() == ()


def test_batch_settlement_retries_index_failure_without_reembedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = Embedder()
    with Memory(tmp_path, embedder=embedder) as memory:
        records = (memory.capture("first"), memory.capture("second"))
        drain = memory._projection.drain

        def unavailable(*, force: bool = False) -> None:
            raise IndexUnavailableError("injected unavailable index")

        monkeypatch.setattr(memory._projection, "drain", unavailable)
        with pytest.raises(IndexUnavailableError):
            memory.settle()
        assert len(memory.pending_captures()) == 2
        monkeypatch.setattr(memory._projection, "drain", drain)
        assert memory.settle() == 2
        assert len(embedder.batches) == 1
        assert {record.id for record in records} == {hit.id for hit in memory.search("first").hits}


def test_settle_disjoint_record_sets_progress_concurrently(tmp_path: Path) -> None:
    entered, release, second_entered = Event(), Event(), Event()

    class BlockingEmbedder(Embedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            if any(value.text == "first" for value in inputs):
                entered.set()
                assert release.wait(10)
            if any(value.text == "second" for value in inputs):
                second_entered.set()
            return super().embed(inputs, task)

    with Memory(tmp_path, embedder=BlockingEmbedder()) as memory:
        first = memory.capture("first")
        second = memory.capture("second")
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_job = pool.submit(memory.settle, memory_ids=(first.id,))
            try:
                assert entered.wait(5)
                second_job = pool.submit(memory.settle, memory_ids=(second.id,))
                assert second_entered.wait(5)
            finally:
                release.set()
            assert first_job.result(timeout=5) == 1
            assert second_job.result(timeout=5) == 1
            assert second_entered.is_set()


@pytest.mark.parametrize("write_text", (False, True))
def test_enabling_unused_speech_on_reopen(tmp_path: Path, write_text: bool) -> None:
    with Memory(tmp_path, embedder=Embedder()) as memory:
        if write_text:
            memory.add("only text")
    with Memory(tmp_path, embedder=Embedder(), transcriber=Transcriber()) as memory:
        assert memory.add(Blob(b"audio", "audio/wav")).content


def test_atomic_embeddings_are_composed_when_joint_inputs_are_unsupported(tmp_path: Path) -> None:
    class Matrix:
        def __init__(self, count: int) -> None:
            self.count = count

        def tolist(self) -> object:
            return [[1.0, 0.0, 0.0, 0.0] for _ in range(self.count)]

    class AtomicEncoder:
        def supports(self, modality: str | tuple[str, ...]) -> bool:
            return isinstance(modality, str) and modality in {"text", "image"}

        def get_embedding_dimension(self) -> int:
            return 4

        def encode_query(self, sentences: list[object], **kwargs: object) -> Matrix:
            return Matrix(len(sentences))

        def encode_document(self, sentences: list[object], **kwargs: object) -> Matrix:
            return Matrix(len(sentences))

    embedder = SentenceTransformersEmbedder(
        AtomicEncoder(), model_id="audit-atomic-only", revision="a" * 40
    )
    with Memory(tmp_path, embedder=embedder) as memory:
        assert memory.capabilities.embedding == {Modality.TEXT, Modality.IMAGE}
        assert memory.capabilities.supports_embedding(frozenset({Modality.TEXT}))
        assert memory.capabilities.supports_embedding(frozenset({Modality.IMAGE}))
        assert not memory.capabilities.supports_embedding(
            frozenset({Modality.TEXT, Modality.IMAGE})
        )
        assert not memory.capabilities.supports_embedding(
            frozenset({Modality.IMAGE}), assets_per_modality=2
        )
        memory.add("text alone")
        memory.add(Blob(b"image-alone", "image/png"))
        assert memory.add(("text and image", Blob(b"joint-image", "image/png"))).id


def test_index_routes_fill_released_capacity_before_the_slowest_finishes(tmp_path: Path) -> None:
    entered, release, fast_done, fifth_entered = Event(), Event(), Event(), Event()

    def slow() -> int:
        entered.set()
        assert release.wait(10)
        return 0

    def fast() -> int:
        fast_done.set()
        return 1

    def fifth() -> int:
        fifth_entered.set()
        return 4

    with (
        Memory(tmp_path, embedder=Embedder()) as memory,
        ThreadPoolExecutor(max_workers=1) as caller,
    ):
        job = caller.submit(memory._execution.index_routes, (slow, fast, fast, fast, fifth))
        try:
            assert entered.wait(5)
            assert fast_done.wait(5)
            assert fifth_entered.wait(5)
        finally:
            release.set()
        assert job.result(timeout=5) == (0, 1, 1, 1, 4)


def test_temporal_and_global_index_routes_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = Barrier(2, timeout=5)
    with Memory(
        tmp_path,
        embedder=Embedder(),
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.DENSE, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        target = memory.add("ALPHA-947", occurred_at=datetime(2024, 6, 1, tzinfo=timezone.utc))
        search = cast(Callable[..., object], memory._index.search)

        def overlapping(*args: object, **kwargs: object) -> object:
            barrier.wait()
            return search(*args, **kwargs)

        monkeypatch.setattr(memory._index, "search", overlapping)
        assert memory.search("ALPHA-947 in 2024").hits[0].id == target.id
