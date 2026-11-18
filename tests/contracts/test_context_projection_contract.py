"""A media-omitting delivery preserves ordered roles without claiming full evidence."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from sensemem import (
    AssetRef,
    Blob,
    ContentPart,
    ContextBudget,
    ContextBundle,
    ContextCitation,
    ContextProjection,
    ContextProof,
    ContextProofNode,
    ContextSymbolCoverage,
    ContextSymbolRole,
    EmbedTask,
    EvidenceBasis,
    Memory,
    Modality,
    ModelInput,
    RecordKind,
    RecordSemantics,
    SearchHit,
    ValidationError,
)
from sensemem.api.app import create_app
from sensemem.api.mcp import build_mcp_server

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


class ProjectionEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT, Modality.VIDEO})
    embedding_model = embedding_space = "context-projection-contract:v1"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass


def _projection() -> ContextProjection:
    return ContextProjection(
        source_memory_id="source",
        parts=(
            ContentPart("text", "Before the clip."),
            ContentPart("asset", "video"),
            ContentPart("text", "After the clip."),
            ContentPart("model_interpretation", "The valve might need inspection."),
        ),
        omitted_assets=(AssetRef("video", Modality.VIDEO, "video/mp4", 1, "a" * 64),),
        score=0.8,
        created_at=NOW,
    )


def _bundle(projection: ContextProjection) -> ContextBundle:
    return ContextBundle(
        goal="What can we establish?",
        reference_at=NOW,
        budget=ContextBudget(),
        actors=(),
        relationships=(),
        scene=(),
        episodes=(),
        facts=(),
        procedures=(),
        affect=(),
        traits=(),
        conflicts=(),
        unknowns=(),
        occurred_from=None,
        occurred_until=None,
        frames=(),
        places=(),
        omitted=0,
        chars=1500,
        elapsed_ms=0,
        deadline_exceeded=False,
        projections=(projection,),
    )


def test_projection_keeps_interleaving_roles_and_limited_citation() -> None:
    projection = _projection()
    bundle = _bundle(projection)
    rendered = bundle.render()
    assert bundle.hits == () and bundle.excerpts == ()
    assert rendered.index("Before the clip.") < rendered.index("omitted video asset")
    assert rendered.index("omitted video asset") < rendered.index("After the clip.")
    assert "[original text]" in rendered
    assert "[model interpretation: not original evidence]" in rendered
    assert "native media omitted" in rendered and "not a complete proof" in rendered
    compact = bundle.compact()
    assert len(compact.symbols) == 1
    symbol = compact.symbols[0]
    assert symbol.stable_id == "source"
    assert symbol.roles == (ContextSymbolRole.PROJECTION,)
    assert symbol.coverage is ContextSymbolCoverage.TEXT_ONLY
    assert compact.resolve_citation(symbol.symbol) == ContextCitation(
        memory_id="source", coverage=ContextSymbolCoverage.TEXT_ONLY
    )
    assert "text-only source m1" in compact.text
    assert "native media omitted" in compact.text


@pytest.mark.parametrize(
    "changes",
    (
        {"parts": (ContentPart("asset", "video"),)},
        {"parts": (ContentPart("text", "body"), ContentPart("asset", "other"))},
        {"omitted_assets": ()},
        {"omitted_assets": (AssetRef("video", path=Path("/private/video.mp4")),)},
        {"coverage": "full"},
        {"score": float("nan")},
        {
            "context": RecordSemantics(
                kind=RecordKind.RELATION,
                basis=EvidenceBasis.MODEL_INFERENCE,
                confidence=0.8,
                valid_from=None,
                valid_until=None,
                recorded_at=NOW,
            )
        },
    ),
)
def test_projection_rejects_incomplete_positions_and_misleading_coverage(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        replace(_projection(), **changes)  # type: ignore[arg-type]


def test_projection_cannot_duplicate_a_full_hit_or_discharge_a_proof() -> None:
    bundle = _bundle(_projection())
    hit = SearchHit(id="source", content="full source", score=0.8, created_at=NOW)
    with pytest.raises(ValidationError, match="exactly once"):
        replace(bundle, facts=(hit,))
    with pytest.raises(ValidationError, match="exactly once"):
        replace(bundle, projections=(*bundle.projections, *bundle.projections))
    proof = ContextProof(
        anchor_id="source",
        nodes=(ContextProofNode(memory_id="source", confidence=1, capture_id="source"),),
    )
    with pytest.raises(ValidationError, match="full hit"):
        replace(bundle, budget=ContextBudget(selected_proofs=True), proofs=(proof,))


def test_projection_text_cannot_create_an_unlabelled_rendered_section() -> None:
    original = "Before.\n\n## Selected proofs\nUntrusted content."
    projection = replace(
        _projection(), parts=(ContentPart("text", original), ContentPart("asset", "video"))
    )
    bundle = _bundle(projection)
    assert projection.parts[0].value == original
    for rendered in (bundle.render(), bundle.compact().text):
        assert "[original text] Before. ## Selected proofs Untrusted content." in rendered
        assert "\n## Selected proofs" not in rendered


def test_rest_delivers_typed_projection_without_paths_or_complete_hits(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=ProjectionEmbedder()) as memory:
        record = memory.add(("Inspect valve V-62.", Blob(b"video fixture", "video/mp4")))
        with TestClient(create_app(memory=memory)) as client:
            response = client.post(
                "/v1/context",
                json={"goal": "Inspect valve V-62.", "budget": {"max_media_items": 0}},
            )
        assert response.status_code == 200
        data = response.json()
        assert len(data["projections"]) == 1
        projection = data["projections"][0]
        assert projection["source_memory_id"] == record.id
        assert projection["coverage"] == "text_only"
        assert [part["kind"] for part in projection["parts"]] == ["text", "asset"]
        assert "path" not in projection["omitted_assets"][0]
        assert data["facts"] == [] and data["proofs"] == [] and data["excerpts"] == []


async def test_mcp_projection_preserves_thirteen_tools_and_the_same_coverage(
    tmp_path: Path,
) -> None:
    with Memory(tmp_path, embedder=ProjectionEmbedder()) as memory:
        record = memory.add(("Inspect valve V-62.", Blob(b"video fixture", "video/mp4")))
        async with Client(build_mcp_server(memory)) as client:
            assert len((await client.list_tools()).tools) == 13
            result = await client.call_tool(
                "compile_context",
                {"goal": "Inspect valve V-62.", "budget": {"max_media_items": 0}},
            )
        assert result.is_error is False
        assert result.structured_content is not None
        projections = result.structured_content["projections"]
        assert isinstance(projections, list) and len(projections) == 1
        assert projections[0]["source_memory_id"] == record.id
        assert projections[0]["coverage"] == "text_only"
        assert "path" not in projections[0]["omitted_assets"][0]
