"""A new interpretation cannot turn one original observation into independent evidence."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AssetRef,
    Blob,
    EmbedTask,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemorySettings,
    Modality,
    ModelInput,
    RecordKind,
    StreamInput,
    ValidationError,
)


@pytest.mark.parametrize("independent", [False, True])
def test_reinterpreting_one_recording_cannot_reinforce_a_trait(
    tmp_path: Path, independent: bool
) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        settings=MemorySettings(independent_evidence=independent),
    ) as memory:
        recording = Blob(b"one original recording", "audio/wav")
        first = next(
            memory.add_stream([StreamInput(recording, interpretation="Kira waits quietly.")])
        )
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(first.id,),
                proposal=FormationProposal(
                    kind=RecordKind.TRAIT,
                    content="Kira is patient.",
                    subject="Kira",
                    predicate="disposition",
                    value="patient",
                    confidence=0.6,
                ),
            )
        )
        claim_id = operation.created_ids[0]
        rewritten = next(
            memory.add_stream([StreamInput(recording, interpretation="Kira waits patiently.")])
        )
        assert rewritten.id == first.id
        assert rewritten.assets == first.assets
        assert "waits patiently" in rewritten.content and "waits quietly" not in rewritten.content
        with pytest.raises(ValidationError):
            memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.REINFORCE,
                    target_ids=(claim_id,),
                    evidence_ids=(rewritten.id,),
                )
            )
        context = memory.get(claim_id).context
        assert context is not None and not context.visible
        assert context.confidence == pytest.approx(0.6)
        assert context.evidence_ids == (first.id,)

        separate = next(
            memory.add_stream(
                [
                    StreamInput(
                        Blob(b"another original recording", "audio/wav"),
                        interpretation="Kira waits patiently.",
                    )
                ]
            )
        )
        assert separate.id != first.id
        memory.apply(
            MemoryOperation(
                intent=MemoryIntent.REINFORCE,
                target_ids=(claim_id,),
                evidence_ids=(separate.id,),
            )
        )
        context = memory.get(claim_id).context
        assert context is not None and context.visible
        assert context.confidence == pytest.approx(0.84)
        assert set(context.evidence_ids) == {first.id, separate.id}


@pytest.mark.parametrize("capture", [False, True])
def test_explicit_refresh_preserves_other_derived_sections(tmp_path: Path, capture: bool) -> None:
    content = ("Original field note.", Blob(b"audio", "audio/wav"), Blob(b"image", "image/png"))
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        original = memory.add(content)
        record = next(
            memory.add_stream(
                [
                    StreamInput(
                        content,
                        transcript="The valve is open.",
                        description="A blue valve.",
                        interpretation="The pump is running.",
                    )
                ],
                capture=capture,
            )
        )
        assert record.id == original.id
        if capture:
            assert memory.settle() == 1
        refreshed = next(
            memory.add_stream(
                [StreamInput(content, interpretation="The pump is stopped.")], capture=capture
            )
        )
        assert refreshed.id == original.id
        assert "Original field note." in refreshed.content
        assert "The valve is open." in refreshed.content
        assert "A blue valve." in refreshed.content
        assert "The pump is running." not in refreshed.content
        assert "The pump is stopped." in refreshed.content
        assert {part.kind for part in refreshed.parts} == {
            "text",
            "asset",
            "audio_transcript",
            "visual_description",
            "model_interpretation",
        }
        if capture:
            assert memory.pending_captures()[0].awaiting == "enrichment"
            assert memory.settle() == 1
        # Ordinary duplicate writes preserve the latest derived content and owe no new work.
        assert memory.add(content).content == refreshed.content
        assert memory.capture(content).content == refreshed.content
        assert memory.pending_captures() == ()


class RewriteEmbedder(TinyEmbedder):
    embedding_dimension = 3
    embedding_space = "derivation-refresh:3:v1"

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (1.0, 0.0, 0.0)
            if "obsoleteviolet" in value.text
            else (0.0, 1.0, 0.0)
            if "currentamber" in value.text
            else (0.0, 0.0, 1.0)
            for value in inputs
        )


class TranscriptEmbedder(RewriteEmbedder):
    embedding_capabilities = frozenset({Modality.TEXT})


class Transcriber:
    transcription_capabilities = frozenset({Modality.AUDIO})
    transcription_model = "refresh-transcriber"
    transcription_space = "refresh-transcriber:v1"

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]:
        return ("obsoleteviolet automatic recognition.",) * len(assets)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("capture", [False, True])
def test_explicit_transcript_replaces_automatic_recognition(tmp_path: Path, capture: bool) -> None:
    config = MemorySettings(minimum_relevance=0.5, ambiguity_margin=0)
    content = ("Actual recorded context.", Blob(b"one original recording", "audio/wav"))
    with Memory(
        tmp_path, embedder=TranscriptEmbedder(), transcriber=Transcriber(), settings=config
    ) as memory:
        original = memory.add(content)
        assert "obsoleteviolet" in original.content
        refreshed = next(
            memory.add_stream(
                [StreamInput(content, transcript="currentamber reviewed recognition.")],
                capture=capture,
            )
        )
        assert refreshed.id == original.id
        assert "Actual recorded context." in refreshed.content
        assert "obsoleteviolet" not in refreshed.content
        assert "currentamber" in refreshed.content
        if capture:
            assert memory.settle() == 1
        assert memory.search("obsoleteviolet").hits == ()
        assert original.id in {hit.id for hit in memory.search("currentamber").hits}

    with Memory(
        tmp_path, embedder=TranscriptEmbedder(), transcriber=Transcriber(), settings=config
    ) as memory:
        assert memory.search("obsoleteviolet").hits == ()
        assert "obsoleteviolet" not in memory.get(original.id).content


@pytest.mark.parametrize("capture", [False, True])
def test_refresh_replaces_old_retrieval_keys_and_survives_reopen(
    tmp_path: Path, capture: bool
) -> None:
    config = MemorySettings(minimum_relevance=0.5, ambiguity_margin=0)
    content = Blob(b"one inspected machine", "image/png")
    old = "\n\n".join(f"obsoleteviolet interpretation paragraph {index}." for index in range(250))
    with Memory(tmp_path, embedder=RewriteEmbedder(), settings=config) as memory:
        record = next(memory.add_stream([StreamInput(content, interpretation=old)]))
        assert record.id in {hit.id for hit in memory.search("obsoleteviolet").hits}
        refreshed = next(
            memory.add_stream(
                [StreamInput(content, interpretation="currentamber inspection.")], capture=capture
            )
        )
        assert refreshed.id == record.id
        assert memory.search("obsoleteviolet").hits == ()
        if capture:
            assert memory.search("currentamber").hits == ()
            assert memory.pending_captures()[0].awaiting == "enrichment"
            assert memory.settle() == 1
        assert record.id in {hit.id for hit in memory.search("currentamber").hits}

    with Memory(tmp_path, embedder=RewriteEmbedder(), settings=config) as memory:
        assert memory.search("obsoleteviolet").hits == ()
        assert record.id in {hit.id for hit in memory.search("currentamber").hits}
        assert "obsoleteviolet" not in memory.get(record.id).content


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("withdrawal", ["forget", "retired", "correct_retired"])
def test_refresh_preserves_source_withdrawal_and_claim_visibility(
    tmp_path: Path, capture: bool, withdrawal: str
) -> None:
    content = Blob(b"retained original recording", "audio/wav")
    config = MemorySettings(minimum_relevance=0)
    with Memory(tmp_path, embedder=TinyEmbedder(), settings=config) as memory:
        source = next(
            memory.add_stream([StreamInput(content, interpretation="Dara inspected a motor.")])
        )
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                target_ids=(source.id,) if withdrawal != "forget" else (),
                proposal=FormationProposal(
                    kind=RecordKind.RELATION,
                    content="Dara inspected a motor.",
                    subject="Dara",
                    predicate="inspected",
                    value="motor",
                    confidence=0.8,
                ),
            )
        )
        claim_id = operation.created_ids[0]
        if withdrawal == "forget":
            memory.forget((source.id,))
        elif withdrawal == "correct_retired":
            memory.apply(MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(claim_id,)))
        forgotten_at = memory.get(source.id).forgotten_at
        assert forgotten_at is not None
        prior_hits = {hit.id for hit in memory.compile("motor").hits}
        refreshed = next(
            memory.add_stream(
                [StreamInput(content, interpretation="Dara examined the motor carefully.")],
                capture=capture,
            )
        )
        assert refreshed.id == source.id and refreshed.forgotten_at == forgotten_at
        if capture:
            assert memory.settle() == 1
        assert memory.get(source.id).forgotten_at == forgotten_at
        after_hits = {hit.id for hit in memory.compile("motor").hits}
        assert (claim_id in after_hits) is (claim_id in prior_hits)

    with Memory(tmp_path, embedder=TinyEmbedder(), settings=config) as memory:
        assert memory.get(source.id).forgotten_at == forgotten_at
        assert (claim_id in {hit.id for hit in memory.compile("motor").hits}) is (
            claim_id in prior_hits
        )
