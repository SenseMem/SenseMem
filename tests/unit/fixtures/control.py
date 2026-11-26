"""Shared control-plane, perception, and independent-evidence fakes for unit tests."""

from collections.abc import Sequence
from pathlib import Path

from _feature_support import TinyEmbedder

from sensemem import (
    AssetRef,
    EvidenceBasis,
    FaceAnalysis,
    FaceEmbedding,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemoryOperationRecord,
    MemoryPlugins,
    MemoryRecord,
    MemorySettings,
    MemoryTrigger,
    Modality,
    ObservationContext,
    QueryFailure,
    RecordKind,
    SpeakerEmbedding,
    SpeechAnalysis,
    SpeechTurn,
)


class FakeSpeech:
    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "fake-funasr"
    transcription_space = "fake-funasr:speech:test"

    def __init__(self) -> None:
        self.calls: list[tuple[AssetRef, ...]] = []
        self.closed = False

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        batch = tuple(assets)
        self.calls.append(batch)
        return tuple(
            SpeechAnalysis(
                turns=(SpeechTurn(0, 900, "spoken red wrench", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            )
            for _asset in batch
        )

    def close(self) -> None:
        self.closed = True


class FakeFace:
    face_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    face_model = "fake-sface"
    face_space = "fake-sface:2:test"
    face_analysis_space = "fake-yunet-sface:test"

    def __init__(self) -> None:
        self.calls: list[tuple[AssetRef, ...]] = []
        self.closed = False

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        batch = tuple(assets)
        self.calls.append(batch)
        return tuple(
            FaceAnalysis(
                (
                    FaceEmbedding(
                        "face-0",
                        (0.0, 1.0),
                        (0.1, 0.1, 0.4, 0.5),
                        100 if asset.modality is Modality.VIDEO else None,
                    ),
                )
            )
            for asset in batch
        )

    def close(self) -> None:
        self.closed = True


class OnePersonSpeech:
    """Hears the same single speaker in every clip, so one identity owns every segment."""

    transcription_capabilities = frozenset({Modality.AUDIO, Modality.VIDEO})
    transcription_model = "one-person-speech"
    transcription_space = "one-person-speech:test"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
        return tuple(
            SpeechAnalysis(
                turns=(SpeechTurn(0, 900, "I am Li, I live next door", "0"),),
                speakers=(SpeakerEmbedding("0", (1.0, 0.0)),),
            )
            for _asset in assets
        )

    def close(self) -> None:
        pass


class OnePersonFace:
    """Sees the same single face in every clip, so it corroborates with the voice."""

    face_capabilities = frozenset({Modality.IMAGE, Modality.VIDEO})
    face_model = "one-person-face"
    face_space = "one-person-face:2:test"
    face_analysis_space = "one-person-face-analysis:test"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis((FaceEmbedding("face-0", (0.0, 1.0), (0.1, 0.1, 0.4, 0.5), None),))
            for _asset in assets
        )

    def close(self) -> None:
        pass


class ScriptedConsolidator:
    """Replays a fixed script of operations, resolving placeholders against shown evidence."""

    consolidation_model = "consolidator-test"
    consolidation_recipe = "consolidator-test:v1"

    def __init__(self, *scripts: Sequence[MemoryOperation]) -> None:
        self._scripts = [tuple(script) for script in scripts]
        self.calls: list[tuple[tuple[str, ...], MemoryTrigger]] = []
        self.failures: list[QueryFailure | None] = []

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
        query_failure: QueryFailure | None = None,
    ) -> tuple[MemoryOperation, ...]:
        self.calls.append((tuple(record.id for record in evidence), trigger))
        self.failures.append(query_failure)
        if not self._scripts:
            return ()
        return self._scripts.pop(0)

    def close(self) -> None:
        pass


def open_independent_memory(path: Path, independent: bool = True) -> Memory:
    return Memory.from_plugins(
        path,
        plugins=MemoryPlugins(embedder=TinyEmbedder()),
        settings=MemorySettings(independent_evidence=independent, minimum_relevance=0),
    )


def add_root_memory(memory: Memory, name: str, capture: str | None = None) -> str:
    return memory.add(
        name,
        context=ObservationContext(basis=EvidenceBasis.OBSERVATION, source_id=capture),
    ).id


def derive_memory(
    memory: Memory,
    sources: tuple[str, ...],
    name: str,
    *,
    trait: bool = False,
    basis: EvidenceBasis = EvidenceBasis.MODEL_INFERENCE,
    forget: bool = False,
) -> MemoryOperationRecord:
    return memory.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=sources,
            target_ids=sources if forget else (),
            proposal=FormationProposal(
                kind=RecordKind.TRAIT if trait else RecordKind.RELATION,
                basis=basis,
                content=name,
                subject="Ana",
                predicate="disposition" if trait else "event",
                value=name,
                confidence=0.6,
            ),
        )
    )
