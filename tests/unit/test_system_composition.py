"""Cross-plane contracts for perception, recall, generation, and concurrency."""

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from test_automatic_routing import ALL, Answerer, Describer, Embedder, Transcriber
from test_capability_composition import RecordingAnswerer
from test_memory_api import _FakeSpeech

from sensemem import (
    AssetRef,
    Blob,
    EmbedTask,
    FaceAnalysis,
    FaceEmbedding,
    FormationInput,
    FormationProposal,
    GenerationRouter,
    Memory,
    MemoryConfig,
    MemoryKind,
    MemoryNotFoundError,
    Modality,
    ModelError,
    ModelInput,
    RetrievalMode,
    SentenceTransformersEmbedder,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
    StorageError,
    StreamInput,
)


@pytest.mark.parametrize("media_type", ("audio/wav", "video/mp4"))
@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("diarized", (False, True))
def test_lexical_media_query_uses_the_same_transcript_as_ingestion(
    tmp_path: Path,
    media_type: str,
    native: bool,
    diarized: bool,
) -> None:
    with Memory(
        tmp_path,
        embedder=Embedder(ALL) if native else Embedder(),
        transcriber=_FakeSpeech() if diarized else Transcriber(),
        answerer=Answerer(),
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.LEXICAL, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        source = memory.add(Blob(b"source", media_type))
        assert memory.search(Blob(b"query", media_type)).hits[0].id == source.id
        assert memory.ask(Blob(b"question", media_type)).hits[0].id == source.id
        assert memory.list().items == (memory.get(source.id),)


def test_router_uses_stored_captions_when_the_perception_plugin_is_not_loaded(
    tmp_path: Path,
) -> None:
    with Memory(tmp_path, embedder=Embedder(ALL), vision_describer=Describer()) as memory:
        source = memory.add(Blob(b"source", "image/png"))
    reader = RecordingAnswerer(frozenset({Modality.TEXT}))
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        answerer=GenerationRouter((reader,)),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        assert memory.ask("ALPHA-947").hits[0].id == source.id
        assert "registration desk" in reader.evidence[0][0].content
        assert reader.evidence[0][0].assets == ()


@pytest.mark.parametrize("diarized", (False, True))
def test_unplugged_speech_keeps_derived_evidence_and_recipe_guards(
    tmp_path: Path, diarized: bool
) -> None:
    settings = MemoryConfig(minimum_relevance=0, ambiguity_margin=0)
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        transcriber=_FakeSpeech() if diarized else Transcriber(),
        config=settings,
    ) as memory:
        source = memory.add(Blob(b"speech", "audio/wav"))
        identity_id = memory.perceive(source.id).speech[0].speaker_id if diarized else None
    with Memory(tmp_path, embedder=Embedder(ALL), answerer=Answerer(), config=settings) as memory:
        assert not memory.capabilities.transcription
        assert memory.get(source.id) == source
        assert memory.search("red wrench" if diarized else "ALPHA-947").hits[0].id == source.id
        assert memory.ask("red wrench" if diarized else "ALPHA-947").hits[0].id == source.id
        assert memory.compile("red wrench" if diarized else "ALPHA-947").hits[0].id == source.id
        if identity_id is not None:
            memory.register_identity(identity_id, "Alice")
            assert '"speaker_name":"Alice"' in memory.get(source.id).content
        memory.add("new text without a speech plugin")

    class Incompatible(Transcriber):
        transcription_space = "incompatible:recipe"

    with pytest.raises(StorageError, match=r"transcription\.space_id"):
        Memory(tmp_path, embedder=Embedder(ALL), transcriber=Incompatible())
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        transcriber=_FakeSpeech() if diarized else Transcriber(),
        config=settings,
    ) as memory:
        assert memory.get(source.id).assets == source.assets


@pytest.mark.parametrize("media_type", ("image/png", "video/mp4"))
def test_lexical_visual_query_prepares_text_even_for_a_native_embedder(
    tmp_path: Path,
    media_type: str,
) -> None:
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        vision_describer=Describer(),
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.LEXICAL, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        source = memory.add(Blob(b"source", media_type))
        assert memory.search(Blob(b"query", media_type)).hits[0].id == source.id
        assert memory.compile(Blob(b"context query", media_type)).hits[0].id == source.id


def test_partial_visual_derivation_does_not_hide_another_missing_asset(tmp_path: Path) -> None:
    class RetryDescriber(Describer):
        def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
            self.calls += 1
            if self.calls == 2:
                raise ModelError("transient fixture failure")
            return ("ALPHA-947 registration desk",) * len(inputs)

    describer = RetryDescriber()
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        vision_describer=describer,
        answerer=Answerer(),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        first, second = Blob(b"first", "image/png"), Blob(b"second", "image/png")
        memory.add(first)
        combined = memory.add((first, second))
        assert memory.ask("ALPHA-947", limit=10).answer
        assert describer.calls == 3
        # Answer-time recovery is temporary; it does not rewrite the source observation.
        assert memory.get(combined.id).content == combined.content


def test_atomic_query_routing_retains_all_text_parts(tmp_path: Path) -> None:
    class Matrix:
        def __init__(self, values: list[object]) -> None:
            self.values = values

        def tolist(self) -> object:
            return [[0.0, 1.0] if "blue" in str(value) else [1.0, 0.0] for value in self.values]

    class Encoder:
        def supports(self, modality: str | tuple[str, ...]) -> bool:
            return modality in ("text", "image")

        def get_embedding_dimension(self) -> int:
            return 2

        def encode_query(self, sentences: list[object], **kwargs: object) -> Matrix:
            return Matrix(sentences)

        def encode_document(self, sentences: list[object], **kwargs: object) -> Matrix:
            return Matrix(sentences)

    embedder = SentenceTransformersEmbedder(Encoder(), model_id="system-atomic", revision="a" * 40)
    with Memory(
        tmp_path,
        embedder=embedder,
        config=MemoryConfig(
            retrieval_mode=RetrievalMode.DENSE, minimum_relevance=0.8, ambiguity_margin=0
        ),
    ) as memory:
        source = memory.add("blue code")
        assert (
            memory.search(("primary", "blue code", Blob(b"query", "image/png"))).hits[0].id
            == source.id
        )


@pytest.mark.parametrize("independent", (False, True))
@pytest.mark.parametrize("mutation", ("add", "delete"))
def test_naming_reindexes_concurrent_speech_without_blocking_search(
    tmp_path: Path,
    independent: bool,
    mutation: str,
) -> None:
    entered, release = Event(), Event()

    class Blocking(Embedder):
        blocked = False

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            if (
                task is EmbedTask.DOCUMENT
                and not self.blocked
                and any(value.assets and "Lily" in value.text for value in inputs)
            ):
                self.blocked = True
                entered.set()
                assert release.wait(10)
            return super().embed(inputs, task)

    with Memory(
        tmp_path,
        embedder=Blocking(ALL),
        transcriber=_FakeSpeech(),
        config=MemoryConfig(
            independent_evidence=independent, minimum_relevance=0, ambiguity_margin=0
        ),
    ) as memory:
        first = memory.add(Blob(b"first", "audio/wav"))
        second = memory.add(Blob(b"survivor", "audio/wav"))
        speaker_id = memory.perceive(first.id, kinds=frozenset({"speech"})).speech[0].speaker_id
        assert speaker_id is not None
        with ThreadPoolExecutor(max_workers=3) as pool:
            naming = pool.submit(memory.register_identity, speaker_id, "Lily")
            try:
                assert entered.wait(5)
                assert pool.submit(memory.search, "red wrench").result(timeout=5).hits
                if mutation == "add":
                    second = pool.submit(memory.add, Blob(b"second", "audio/wav")).result(timeout=5)
                else:
                    assert pool.submit(memory.delete, first.id).result(timeout=5)
            finally:
                release.set()
            naming.result(timeout=10)
        if mutation == "add":
            assert '"speaker_name":"Lily"' in memory.get(first.id).content
        else:
            with pytest.raises(MemoryNotFoundError):
                memory.get(first.id)
        assert '"speaker_name":"Lily"' in memory.get(second.id).content


def test_identity_link_reindex_keeps_unrelated_search_available(tmp_path: Path) -> None:
    entered, release = Event(), Event()

    class Faces:
        face_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
        face_model, face_space, face_analysis_space = (
            "system-face",
            "system-face:2",
            "system-face:v1",
        )

        def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
            return tuple(
                FaceAnalysis(
                    (
                        FaceEmbedding(
                            "face",
                            (0.0, 1.0),
                            (0.1, 0.1, 0.4, 0.5),
                            100 if asset.modality is Modality.VIDEO else None,
                        ),
                    )
                )
                for asset in assets
            )

        def close(self) -> None:
            pass

    class Blocking(Embedder):
        armed = False

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            if self.armed and task is EmbedTask.DOCUMENT:
                entered.set()
                assert release.wait(10)
            return super().embed(inputs, task)

    embedder = Blocking(ALL)
    with Memory(
        tmp_path,
        embedder=embedder,
        transcriber=_FakeSpeech(),
        face_analyzer=Faces(),
        answerer=RecordingAnswerer(ALL),
        config=MemoryConfig(identity_link_min_assets=1, minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add(("ALPHA-947", Blob(b"face", "image/png")))
        memory.ask("ALPHA-947")
        voice = memory.add(("ALPHA-947", Blob(b"voice and face", "video/mp4")))
        embedder.armed = True
        with ThreadPoolExecutor(max_workers=2) as pool:
            linking = pool.submit(memory.ask, "ALPHA-947", limit=10)
            try:
                assert entered.wait(5)
                assert pool.submit(memory.search, "ALPHA-947").result(timeout=5).hits
            finally:
                release.set()
            assert linking.result(timeout=10).answer
        assert (
            memory.perceive(voice.id, kinds=frozenset({"speech"})).speech[0].speaker_id
            == memory.perceive(voice.id).faces[0].identity_id
        )


def test_unrelated_audio_fallback_does_not_describe_native_query_images(tmp_path: Path) -> None:
    describer = Describer()
    with Memory(
        tmp_path,
        embedder=Embedder(frozenset({Modality.TEXT, Modality.IMAGE})),
        transcriber=Transcriber(),
        vision_describer=describer,
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add("ALPHA-947")
        assert memory.search(
            ("ALPHA-947", Blob(b"image", "image/png"), Blob(b"voice", "audio/wav"))
        ).hits
        assert describer.calls == 0


def test_video_question_keeps_both_speech_and_visual_representations(tmp_path: Path) -> None:
    class Words(Transcriber):
        def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
            return ("The speaker says hello.",) * len(assets)

    reader = RecordingAnswerer(frozenset({Modality.TEXT}))
    with Memory(
        tmp_path,
        embedder=Embedder(ALL),
        transcriber=Words(),
        vision_describer=Describer(),
        answerer=reader,
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        memory.add("ALPHA-947")
        assert memory.ask(("ALPHA-947", Blob(b"video", "video/mp4"))).answer
        assert "speaker says hello" in reader.inputs[0].text
        assert "registration desk" in reader.inputs[0].text


def test_empty_transcript_does_not_silently_satisfy_media_fallback(tmp_path: Path) -> None:
    class Silence(Transcriber):
        def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
            return ("",) * len(assets)

    with Memory(tmp_path, embedder=Embedder(), transcriber=Silence()) as memory:
        with pytest.raises(ModelError) as failure:
            memory.add(("meeting", Blob(b"silent-video", "video/mp4")))
        assert failure.value.reason == "unsupported_modality"
        assert memory.list().items == ()


def test_perception_capture_formation_recall_and_deletion_share_one_source(tmp_path: Path) -> None:
    class Former:
        formation_capabilities = frozenset({Modality.TEXT})
        formation_model, formation_space = "system-code-former", "system-code-former:v1"

        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            assert all(
                not value.content.assets
                and "registration desk" in value.content.text
                and "spoken red wrench" in value.content.text
                for value in inputs
            )
            return tuple(
                (
                    FormationProposal(
                        kind=MemoryKind.STATE,
                        content="The registration desk code is ALPHA-947",
                        subject="registration desk",
                        predicate="code",
                        value="ALPHA-947",
                        confidence=0.9,
                    ),
                )
                for _ in inputs
            )

        def close(self) -> None:
            pass

    with Memory(
        tmp_path,
        embedder=Embedder(),
        transcriber=_FakeSpeech(),
        vision_describer=Describer(),
        former=Former(),
        answerer=Answerer(),
        config=MemoryConfig(minimum_relevance=0, ambiguity_margin=0),
    ) as memory:
        source = memory.capture((Blob(b"image", "image/png"), Blob(b"speech", "audio/wav")))
        assert memory.settle(memory_ids=(source.id,)) == 1
        formed = next(
            record
            for record in memory.list().items
            if record.context is not None and record.context.kind is MemoryKind.STATE
        )
        assert formed.context is not None and formed.context.evidence_ids == (source.id,)
        hits = memory.search(Blob(b"query", "image/png")).hits
        assert {source.id, formed.id} <= {hit.id for hit in hits}
        assert source.id in {hit.id for hit in memory.compile("ALPHA-947").hits}
        assert memory.ask("ALPHA-947").hits
        assert memory.pending_captures() == ()
        memory.delete(source.id)
        assert memory.search("ALPHA-947").hits == ()


def test_identity_reindex_does_not_assign_an_unattributed_turn(tmp_path: Path) -> None:
    class MixedSpeech(_FakeSpeech):
        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            return (
                SpeechAnalysis(
                    (
                        SpeechTurn(0, 100, "known voice", "0"),
                        SpeechTurn(100, 200, "unattributed words"),
                    ),
                    (SpeakerEmbedding("0", (1.0, 0.0)),),
                ),
            ) * len(assets)

    with Memory(tmp_path, embedder=Embedder(ALL), transcriber=MixedSpeech()) as memory:
        source = memory.add(Blob(b"two turns", "audio/wav"))
        speaker_id = memory.perceive(source.id, kinds=frozenset({"speech"})).speech[0].speaker_id
        assert speaker_id is not None
        memory.register_identity(speaker_id, "Lily")
        payload = json.loads(memory.get(source.id).content.split("\n", 1)[1])
        assert payload["segments"][0]["speaker_name"] == "Lily"
        assert payload["segments"][1]["speaker_id"] is None
        assert payload["segments"][1]["speaker_name"] is None
        assert memory.perceive(source.id, kinds=frozenset({"speech"})).speech[1].speaker_id is None


def test_supplied_audio_transcript_does_not_cover_a_second_video_asset(tmp_path: Path) -> None:
    class Counting(Transcriber):
        def __init__(self) -> None:
            self.calls: list[tuple[AssetRef, ...]] = []

        def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
            self.calls.append(tuple(assets))
            return super().transcribe(assets)

    transcriber = Counting()
    with Memory(tmp_path, embedder=Embedder(), transcriber=transcriber) as memory:
        (source,) = memory.add_stream(
            (
                StreamInput(
                    (Blob(b"audio", "audio/wav"), Blob(b"video", "video/mp4")),
                    transcript="provided words",
                ),
            )
        )
        assert "provided words" in source.content and "ALPHA-947" in source.content
        assert [[asset.modality for asset in batch] for batch in transcriber.calls] == [
            [Modality.VIDEO]
        ]
