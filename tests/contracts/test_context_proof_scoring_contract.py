"""SDK, REST and MCP publish the store's proof scoring rule and source groups."""

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mcp import Client
from test_context_selection_contract import ContextEmbedder

from sensemem import (
    ContextBudget,
    EvidenceBasis,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemorySettings,
    RecordKind,
    ValidationError,
)
from sensemem.api.app import create_app
from sensemem.api.mcp import build_mcp_server


def _reports(memory: Memory) -> str:
    target = ""
    for reporter in ("Lea", "Noor", "Sam"):
        raw = memory.add(f"{reporter} reports Ana's longstanding patience.")
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(raw.id,),
                proposal=FormationProposal(
                    kind=RecordKind.TRAIT,
                    basis=EvidenceBasis.MODEL_INFERENCE,
                    content="Ana shows longstanding patience, according to the reports.",
                    subject="Ana",
                    predicate="disposition",
                    value="patient",
                    confidence=0.6,
                ),
            )
        )
        if operation.created_ids:
            (target,) = operation.created_ids
    return target


@pytest.mark.parametrize("independent", [False, True])
def test_sdk_and_rest_publish_the_same_scoring_and_complete_groups(
    tmp_path: Path, independent: bool
) -> None:
    with Memory(
        tmp_path,
        embedder=ContextEmbedder(),
        settings=MemorySettings(independent_evidence=independent, minimum_relevance=0),
    ) as memory:
        target = _reports(memory)
        bundle = memory.compile(
            "Ana patience", budget=ContextBudget(selected_proofs=True, max_items=12)
        )
        assert target in {hit.id for hit in bundle.hits}
        expected_scoring = "independent_pair" if independent else "capture_groups"
        assert bundle.proof_scoring == expected_scoring
        delivered = next(hit for hit in bundle.hits if hit.id == target)
        assert delivered.context is not None
        assert delivered.context.confidence == pytest.approx(0.84 if independent else 0.936)
        with pytest.raises(ValidationError, match="proof_scoring"):
            replace(bundle, proof_scoring="guess")  # type: ignore[arg-type]
        if not independent:
            with pytest.raises(ValidationError, match="confidence"):
                replace(bundle, proof_scoring="independent_pair")
        with TestClient(create_app(memory=memory)) as client:
            response = client.post(
                "/v1/context",
                json={
                    "goal": "Ana patience",
                    "budget": {"selected_proofs": True, "max_items": 12},
                },
            )
        assert response.status_code == 200
        wire = response.json()
        assert wire["proof_scoring"] == expected_scoring
        proofs = [proof for proof in wire["proofs"] if proof["anchor_id"] == target]
        assert len(proofs) >= (2 if independent else 3)
        for proof in proofs:
            for node in proof["nodes"]:
                assert node["source_group_id"]
                if node["capture_id"] is not None:
                    assert node["source_group_id"] == node["capture_id"]


@pytest.mark.parametrize("independent", [False, True])
async def test_mcp_preserves_scoring_and_delivers_the_supported_trait(
    tmp_path: Path, independent: bool
) -> None:
    with Memory(
        tmp_path,
        embedder=ContextEmbedder(),
        settings=MemorySettings(independent_evidence=independent, minimum_relevance=0),
    ) as memory:
        target = _reports(memory)
        async with Client(build_mcp_server(memory)) as client:
            assert len((await client.list_tools()).tools) == 13
            result = await client.call_tool(
                "compile_context",
                {"goal": "Ana patience", "budget": {"selected_proofs": True, "max_items": 12}},
            )
        assert not result.is_error
        wire = result.structured_content
        assert wire is not None
        assert wire["proof_scoring"] == ("independent_pair" if independent else "capture_groups")
        traits = wire["traits"]
        assert isinstance(traits, list)
        assert any(isinstance(hit, dict) and hit["id"] == target for hit in traits)
