"""Unified entry points retain typed results and share the underlying execution paths."""

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from _feature_support import TinyEmbedder
from test_memory_api import _FakeFace, _FakeSpeech

from sensemem import (
    AssetRef,
    AsyncMemory,
    Blob,
    ConsolidationReport,
    FaceAnalysis,
    IdentityNotFoundError,
    Memory,
    MemoryConfig,
    MemoryOperation,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ModelError,
    PerceptionKind,
    PerceptionResult,
    SearchResult,
    SpeechAnalysis,
    ValidationError,
)


def test_search_has_one_return_type_and_diagnostics_preserve_hits(tmp_path: Path) -> None:
    with Memory(
        tmp_path, embedder=TinyEmbedder(), config=MemoryConfig(minimum_relevance=0)
    ) as memory:
        memory.add_many(("blue toolbox", "spare key"))
        reference = datetime.now(timezone.utc)
        plain = memory.search("toolbox", reference_at=reference)
        explained = memory.search("toolbox", reference_at=reference, explain=True)
        assert isinstance(plain, SearchResult)
        assert plain.trace is None
        assert explained.trace is not None
        assert explained.hits == plain.hits
        with pytest.raises(ValidationError, match="explain"):
            memory.search("toolbox", explain=cast(bool, 1))


@pytest.mark.asyncio
async def test_async_search_and_perception_preserve_the_same_values(tmp_path: Path) -> None:
    async with AsyncMemory(
        Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            transcriber=_FakeSpeech(),
            config=MemoryConfig(minimum_relevance=0),
        )
    ) as memory:
        record = await memory.add("blue toolbox")
        found = await memory.search("toolbox", explain=True)
        assert found.hits[0].id == record.id and found.trace is not None
        assert await memory.perceive(record.id) == PerceptionResult()
        captured = await memory.capture(Blob(b"speaker", "audio/wav"))
        perceived = await memory.perceive(captured.id)
        identity_id = perceived.speech[0].speaker_id
        assert identity_id is not None
        await memory.register_identity(identity_id, "Alice")
        erased = await memory.forget_identity(identity_id)
        assert erased.identity_id == identity_id and erased.voice_exemplars > 0
        assert await memory.identity(identity_id) is None
        assert await memory.get(captured.id) is not None


def test_perception_selects_configured_plugins_and_explicit_kinds(tmp_path: Path) -> None:
    speech = _FakeSpeech()
    with Memory(tmp_path, embedder=TinyEmbedder(), transcriber=speech) as memory:
        record = memory.capture(Blob(b"voice and image", "video/mp4"))
        result = memory.perceive(record.id)
        assert result.speech and result.faces == ()
        identity_id = result.speech[0].speaker_id
        assert identity_id is not None
        memory.register_identity(identity_id, "Alice")
        profile = memory.identity(identity_id)
        assert profile is not None and profile.name == "Alice"
        with pytest.raises(ModelError, match="face backend"):
            memory.perceive(record.id, kinds=frozenset({"faces"}))
        with pytest.raises(IdentityNotFoundError):
            memory.register_identity("missing", "Nobody")


@pytest.mark.parametrize("kinds", (frozenset(), frozenset({"unknown"})))
def test_perception_rejects_invalid_selection(tmp_path: Path, kinds: frozenset[str]) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add("text")
        with pytest.raises(ValidationError, match="kinds"):
            memory.perceive(record.id, kinds=cast(frozenset[PerceptionKind], kinds))


@pytest.mark.parametrize("kind", ("speech", "faces"))
def test_explicit_perception_rejects_a_modality_its_plugin_cannot_handle(
    tmp_path: Path, kind: PerceptionKind
) -> None:
    class Speech(_FakeSpeech):
        transcription_capabilities = frozenset({Modality.AUDIO})

    class Faces(_FakeFace):
        face_capabilities = frozenset({Modality.IMAGE})

    with Memory(
        tmp_path, embedder=TinyEmbedder(), transcriber=Speech(), face_analyzer=Faces()
    ) as memory:
        record = memory.capture(Blob(b"video", "video/mp4"))
        result = memory.perceive(record.id)
        assert not result.speech and not result.faces
        assert {entry.reason for entry in result.coverage} == {"unsupported_modality"}
        with pytest.raises(ModelError) as failure:
            memory.perceive(record.id, kinds=frozenset({kind}))
        assert failure.value.reason == "unsupported_modality"


def test_unified_perception_overlaps_voice_and_face_work(tmp_path: Path) -> None:
    barrier = Barrier(2)

    class Speech(_FakeSpeech):
        def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]:
            barrier.wait(timeout=5)
            return super().analyze(assets)

    class Faces(_FakeFace):
        def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
            barrier.wait(timeout=5)
            return super().analyze(assets)

    with Memory(
        tmp_path, embedder=TinyEmbedder(), transcriber=Speech(), face_analyzer=Faces()
    ) as memory:
        record = memory.capture(Blob(b"person", "video/mp4"))
        result = memory.perceive(record.id)
        assert result.speech and result.faces
        assert result.speech[0].asset_id == result.faces[0].asset_id


class _SilentConsolidator:
    consolidation_model = "unified-api-fixture"
    consolidation_recipe = "unified-api-fixture:v1"

    def consolidate(
        self, evidence: Sequence[MemoryRecord], *, trigger: MemoryTrigger
    ) -> tuple[MemoryOperation, ...]:
        return ()

    def close(self) -> None:
        pass


def test_consolidation_modes_share_report_fields_and_termination(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), consolidator=_SilentConsolidator()) as memory:
        first, second = memory.add_many(("first observation", "second observation"))
        explicit = memory.consolidate(evidence_ids=(first.id, second.id))
        assert isinstance(explicit, ConsolidationReport)
        assert explicit.weighed == 2 and explicit.rounds == explicit.model_calls == 1
        assert explicit.applied == 0 and explicit.rejected == ()
        memory.reinforce((first.id, second.id))
        automatic = memory.consolidate_pending(max_rounds=2)
        assert isinstance(automatic, ConsolidationReport)
        assert automatic.rounds == 1 and automatic.model_calls == automatic.weighed == 2
        assert memory.consolidate_pending() == ConsolidationReport()
