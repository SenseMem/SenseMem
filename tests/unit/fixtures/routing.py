"""Shared routing, composition, and visual retrieval fake backends."""

from collections.abc import Sequence

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AssetRef,
    EmbedTask,
    Modality,
    ModelInput,
    SearchHit,
    SpeechAnalysis,
    SpeechTurn,
)

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


class VisualBackend:
    embedding_model = "visual-test"
    embedding_space = "visual-test:2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})
    generation_model = "visual-reader-test"
    generation_capabilities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(self) -> None:
        self.embedding_calls = 0
        self.questions: list[ModelInput] = []

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        self.embedding_calls += 1
        return tuple(
            (1.0, 0.0)
            if value.assets and (value.assets[0].name or "").startswith(("near", "query"))
            else (0.8, 0.6)
            for value in inputs
        )

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        assert isinstance(question, ModelInput)
        self.questions.append(question)
        return AnswerResult(answer="Read the supplied sources.", hits=tuple(hits))

    def close(self) -> None:
        return None
