"""Oversized native observations retain a clearly limited, source-backed text delivery."""

from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AsyncEventStream,
    AsyncMemory,
    Blob,
    ContextBudget,
    EventChunk,
    EventObservation,
    EventSegment,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    RecordKind,
    StreamInput,
)


class Segmenter:
    event_model = "media-projection-event-fixture"
    event_recipe = "media-projection-event-fixture:v1"

    def segment_events(
        self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
    ) -> tuple[EventSegment, ...]:
        return (
            (EventSegment(2, "Inspect pressure valve V-62 before restarting the compressor."),)
            if len(observations) == 2
            else ()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat_media", [False, True])
async def test_event_search_settlement_reopen_and_default_compile_keep_a_limited_projection(
    tmp_path: Path, repeat_media: bool
) -> None:
    reference = datetime(2026, 11, 16, tzinfo=timezone.utc)

    async def chunks() -> AsyncIterator[EventChunk]:
        for offset in range(2):
            content_offset = 0 if repeat_media else offset
            yield EventChunk(
                (
                    Blob(f"camera window {content_offset}".encode(), "video/mp4"),
                    Blob(f"sound window {content_offset}".encode(), "audio/wav"),
                ),
                reference + timedelta(seconds=offset),
                reference + timedelta(seconds=offset + 1),
            )

    async with AsyncMemory(Memory(tmp_path, embedder=TinyEmbedder())) as async_memory:
        commits = [
            commit
            async for commit in AsyncEventStream(
                async_memory, segmenter=Segmenter(), capture=True
            ).consume(chunks())
        ]
        assert len(commits) == 1
        source = commits[0].record
        assert len(source.assets) == 4
        assert await async_memory.settle() == 1

    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        query = "Inspect pressure valve V-62 before restarting the compressor."
        assert source.id in {hit.id for hit in memory.search(query).hits}
        for budget in (ContextBudget(), ContextBudget(max_media_items=0)):
            bundle = memory.compile(query, budget=budget)
            assert bundle.hits == ()
            assert len(bundle.projections) == 1
            projection = bundle.projections[0]
            assert projection.source_memory_id == source.id
            assert projection.parts == source.parts
            assert projection.coverage == "text_only"
            assert {asset.id for asset in projection.omitted_assets} == {
                asset.id for asset in source.assets
            }
            assert all(asset.path is None for asset in projection.omitted_assets)
            assert any(part.kind == "model_interpretation" for part in projection.parts)
            assert not any(part.kind == "text" for part in projection.parts)
            priced = bundle.render().split("\n\n## Unknowns", 1)[0]
            assert len(priced) <= bundle.chars <= budget.max_chars
            assert bundle.proofs == () and bundle.excerpts == ()
        control = memory.compile(query, budget=ContextBudget(max_chars=48_000))
        assert source.id in {hit.id for hit in control.hits}
        assert control.projections == ()


@pytest.mark.parametrize("selected_proofs", [False, True])
def test_projection_does_not_discharge_a_claims_native_evidence(
    tmp_path: Path, selected_proofs: bool
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = next(
            memory.add_stream(
                [
                    StreamInput(
                        (Blob(b"valve video", "video/mp4"), Blob(b"valve sound", "audio/wav")),
                        interpretation="Inspect valve V-62 before restarting the compressor.",
                    )
                ]
            )
        )
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=RecordKind.RELATION,
                    content="The compressor was restarted safely.",
                    subject="compressor",
                    predicate="restart_outcome",
                    value="safe",
                    confidence=0.8,
                ),
            )
        )
        claim_id = operation.created_ids[0]
        bundle = memory.compile(
            "Inspect valve V-62 before restarting the compressor.",
            budget=ContextBudget(selected_proofs=selected_proofs),
        )
        assert {projection.source_memory_id for projection in bundle.projections} == {source.id}
        assert claim_id not in {hit.id for hit in bundle.hits}
        assert bundle.proofs == ()
        assert any(item.kind.value == "budget_excluded" for item in bundle.unknowns)
        memory.forget((source.id,))
        assert memory.compile("Inspect valve V-62").projections == ()


def test_projection_retains_original_text_media_positions_and_derived_roles(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        source = next(
            memory.add_stream(
                [
                    StreamInput(
                        (
                            "Before opening the valve:",
                            Blob(b"valve video", "video/mp4"),
                            "The operator recorded eight bar.",
                            Blob(b"valve sound", "audio/wav"),
                        ),
                        interpretation="A model infers that the pressure was acceptable.",
                    )
                ]
            )
        )
        bundle = memory.compile("The operator recorded eight bar.")
        projection = bundle.projections[0]
        assert projection.source_memory_id == source.id
        assert projection.parts == source.parts
        rendered = bundle.render().split("## Text-only sources", 1)[1]
        assert rendered.index("Before opening") < rendered.index("omitted video asset")
        assert rendered.index("omitted video asset") < rendered.index("recorded eight bar")
        assert "[original text]" in rendered
        assert "model interpretation: not original evidence" in rendered
        assert "not a complete proof" in rendered
        assert bundle.omitted == 0


def test_repeated_image_positions_compile_whole_or_as_a_text_projection(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        image = Blob(b"one repeated frame", "image/png")
        source = memory.add(("Before the frame.", image, "Again after the frame.", image))
        full = memory.compile("Before the frame.")
        assert source.id in {hit.id for hit in full.hits}
        assert full.projections == ()
        bundle = memory.compile("Before the frame.", budget=ContextBudget(max_media_items=0))
        assert bundle.hits == ()
        projection = bundle.projections[0]
        assert projection.parts == source.parts
        assert len(projection.omitted_assets) == 1
        assert sum(part.kind == "asset" for part in projection.parts) == 2
        assert any("2 native media parts" in item.detail for item in bundle.unknowns)


def test_projection_does_not_truncate_text_or_invent_a_media_description(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        content = (Blob(b"long video", "video/mp4"), Blob(b"long sound", "audio/wav"))
        source = next(
            memory.add_stream(
                [StreamInput(content, interpretation="oversized-interpretation " * 1000)]
            )
        )
        bundle = memory.compile("oversized-interpretation")
        assert bundle.hits == () and bundle.projections == ()
        assert source.id in {hit.id for hit in memory.search("oversized-interpretation").hits}

    with Memory(tmp_path / "media-only", embedder=TinyEmbedder()) as memory:
        source = memory.add(content)
        assert all(part.kind == "asset" for part in source.parts)
        bundle = memory.compile("valve inspection")
        assert bundle.hits == () and bundle.projections == ()
