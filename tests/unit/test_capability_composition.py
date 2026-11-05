"""Specialized native backends compose through the SDK and preserve ownership."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest
from test_automatic_routing import ALL, TEXT, Answerer, Describer, Embedder, Speech

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    Blob,
    EmbeddingRouter,
    EmbedTask,
    EvidenceGeneration,
    FaceAnalysis,
    GenerationRouter,
    Memory,
    MemoryConfig,
    MemoryIntent,
    Modality,
    ModelInput,
    SearchHit,
    SpeakerEmbedding,
    SpeakerName,
    SpeechAnalysis,
    SpeechTurn,
    ValidationError,
    VisualDescription,
)


class RecordingAnswerer(Answerer):
    def __init__(self, modalities: frozenset[Modality]) -> None:
        self.generation_capabilities = modalities
        self.inputs: list[ModelInput] = []
        self.evidence: list[tuple[SearchHit, ...]] = []
        self.closed = 0

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.inputs.append(question)
        self.evidence.append(tuple(hits))
        return super().answer(question, hits, answer_policy=answer_policy, exhaustive=exhaustive)

    def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("modality", (Modality.TEXT, Modality.IMAGE, Modality.VIDEO))
def test_generation_selects_the_native_route_for_the_question(
    tmp_path: Path, modality: Modality
) -> None:
    backends = tuple(
        RecordingAnswerer(TEXT | {value})
        for value in (
            Modality.TEXT,
            Modality.IMAGE,
            Modality.VIDEO,
        )
    )
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=GenerationRouter(backends),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add("ALPHA-947")
        question = (
            "ALPHA-947"
            if modality is Modality.TEXT
            else (
                "ALPHA-947",
                Blob(b"question", "image/png" if modality is Modality.IMAGE else "video/mp4"),
            )
        )
        assert memory.ask(question).answer
        selected = {Modality.TEXT: 0, Modality.IMAGE: 1, Modality.VIDEO: 2}[modality]
        assert [len(item.inputs) for item in backends] == [
            int(index == selected) for index in range(3)
        ]
    assert [item.closed for item in backends] == [1, 1, 1]


def test_text_question_routes_on_visual_evidence_too(tmp_path: Path) -> None:
    text, visual = RecordingAnswerer(TEXT), RecordingAnswerer(TEXT | {Modality.IMAGE})
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=GenerationRouter((text, visual)),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        record = memory.add(("ALPHA-947", Blob(b"source", "image/png")))
        assert memory.ask("ALPHA-947").hits[0].id == record.id
        assert not text.inputs and len(visual.inputs) == 1


@pytest.mark.parametrize("outer_evidence", (False, True))
def test_routing_survives_evidence_composition_and_converts_only_missing_media(
    tmp_path: Path,
    outer_evidence: bool,
) -> None:
    text = RecordingAnswerer(TEXT)
    image = RecordingAnswerer(TEXT | {Modality.IMAGE})
    video = RecordingAnswerer(TEXT | {Modality.VIDEO})
    routes = (text, image, video)
    answerer = (
        EvidenceGeneration(GenerationRouter(routes))
        if outer_evidence
        else GenerationRouter(tuple(EvidenceGeneration(backend) for backend in routes))
    )
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=answerer,
        vision_describer=Describer(),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add(("ALPHA-947", Blob(b"clip", "video/mp4")))
        assert memory.ask(("ALPHA-947", Blob(b"question", "image/png"))).answer
        assert image.inputs[0].modalities == {Modality.TEXT, Modality.IMAGE}
        assert image.inputs[0].evidence_sources[0].assets[0].modality is Modality.VIDEO
        assert all(
            not hit.assets and "registration desk" in hit.content for hit in image.evidence[0]
        )
        assert not text.inputs and not video.inputs


def test_specialized_embeddings_compose_atomic_keys(tmp_path: Path) -> None:
    text, images = Embedder(), Embedder(frozenset({Modality.IMAGE}))
    with Memory(
        tmp_path,
        embedder=EmbeddingRouter((text, images)),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        record = memory.add(("ALPHA-947", Blob(b"source", "image/png")))
        assert memory.search(("ALPHA-947", Blob(b"query", "image/png"))).hits[0].id == record.id
    assert all(not item.assets for batch in text.batches for item in batch)
    assert all(not item.text for batch in images.batches for item in batch)


def test_embedding_composition_refuses_incompatible_spaces() -> None:
    class OtherSpace(Embedder):
        embedding_space = "other-space:4"

    with pytest.raises(ValidationError, match="share one vector space"):
        EmbeddingRouter((Embedder(), OtherSpace()))


def test_nested_generation_routes_select_leaf_and_close_shared_resources_once() -> None:
    backend = RecordingAnswerer(TEXT)
    visual = RecordingAnswerer(TEXT | {Modality.IMAGE})
    router = GenerationRouter((GenerationRouter((backend, visual)), EvidenceGeneration(backend)))
    assert router.select(ModelInput("text"), ()) is backend
    router.close()
    assert backend.closed == visual.closed == 1


def test_independent_face_and_speech_inference_overlap(tmp_path: Path) -> None:
    barrier = Barrier(2)

    class ConcurrentSpeech(Speech):
        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            barrier.wait(timeout=5)
            return super().analyze(assets)

    class Faces:
        face_capabilities = frozenset({Modality.VIDEO})
        face_model = "face-concurrency-fixture"
        face_space = "face-concurrency-fixture:2"
        face_analysis_space = "face-concurrency-fixture:v1"

        def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
            barrier.wait(timeout=5)
            return (FaceAnalysis(()),) * len(assets)

        def close(self) -> None:
            pass

    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=Answerer(),
        transcriber=ConcurrentSpeech(),
        face_analyzer=Faces(),
        config=MemoryConfig(index_speech=False, minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add(("ALPHA-947", Blob(b"video", "video/mp4")))
        assert memory.ask("ALPHA-947").answer


def test_enrichment_is_independent_of_the_generation_provider(tmp_path: Path) -> None:
    class Enricher:
        def enrich(self, assets: Sequence[AssetRef]) -> dict[str, dict[str, object]]:
            return {asset.id: {"untrusted": True, "fixture": "source context"} for asset in assets}

        def close(self) -> None:
            pass

    backend = RecordingAnswerer(TEXT | {Modality.IMAGE})
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=EvidenceGeneration(backend, enricher=Enricher()),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add(("ALPHA-947", Blob(b"source", "image/png")))
        assert memory.ask("ALPHA-947").answer
        assert next(iter(backend.inputs[0].source_metadata.values()))["fixture"] == "source context"


class NamedSpeech(Speech):
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        self.calls += 1
        return tuple(
            SpeechAnalysis(
                (SpeechTurn(0, 100, "I am Lily", "local"),),
                (SpeakerEmbedding("local", (1.0, 0.0)),),
            )
            for _ in assets
        )


@pytest.mark.parametrize("structured", (False, True))
def test_only_explicit_visual_name_proposals_can_name_a_speaker(
    tmp_path: Path, structured: bool
) -> None:
    class Plain(Describer):
        def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
            return ("Fact: speaker_1 is called Lily",) * len(inputs)

    class Structured(Plain):
        def describe_analysis(self, inputs: Sequence[ModelInput]) -> tuple[VisualDescription, ...]:
            return (
                VisualDescription(
                    "a person", ("speaker_1 is called Lily",), (SpeakerName("speaker_1", "Lily"),)
                ),
            ) * len(inputs)

    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        transcriber=NamedSpeech(),
        vision_describer=Structured() if structured else Plain(),
    ) as memory:
        record = memory.add(Blob(b"video", "video/mp4"))
        speaker = memory.perceive(record.id, kinds=frozenset({"speech"})).speech[0]
        assert speaker.speaker_name == ("Lily" if structured else None)
        assert (
            any(row.operation.intent is MemoryIntent.IDENTIFY for row in memory.operations())
            is structured
        )


def test_concurrent_speech_plans_revalidate_without_repeating_inference(tmp_path: Path) -> None:
    barrier, lock = Barrier(2, timeout=5), Lock()

    class ConcurrentEmbedder(Embedder):
        def __init__(self) -> None:
            super().__init__(ALL)
            self.entered = 0

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            with lock:
                self.entered += 1
                wait = self.entered <= 2
            if wait:
                barrier.wait()
            return super().embed(inputs, task)

    speech = NamedSpeech()
    with Memory(tmp_path, embedder=ConcurrentEmbedder(), transcriber=speech) as memory:
        with ThreadPoolExecutor(max_workers=2) as pool:
            records = tuple(
                pool.map(memory.add, (Blob(b"one", "audio/wav"), Blob(b"two", "audio/wav")))
            )
        assert (
            len(
                {
                    memory.perceive(record.id, kinds=frozenset({"speech"})).speech[0].speaker_id
                    for record in records
                }
            )
            == 1
        )
        assert speech.calls == 2
