"""Public emotional attribution values survive storage and both transport contracts."""

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

from sensemem import (
    AffectLink,
    AffectObjectRef,
    AffectWitness,
    EmbedTask,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    Modality,
    ModelInput,
    RecordKind,
    RecordSemantics,
)
from sensemem.api.responses import MemoryResponse
from sensemem.control import dump_operation, load_operation, operation_key

QUOTE = "Ana says, 'I feel angry because the launch was canceled.'"


class _ContractEmbedder:
    embedding_capabilities = frozenset({Modality.TEXT})
    embedding_model = "attribution-contract-embedding"
    embedding_space = "attribution-contract-embedding:v1"
    embedding_dimension = 2

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        pass


class _AttributedFormer:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "attribution-contract-fixture"
    formation_space = "attribution-contract-fixture:v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    local_id="feeling",
                    content="Ana feels angry.",
                    subject="Ana",
                    value="angry",
                    cue_modality=Modality.TEXT,
                    valence=-0.6,
                    arousal=0.5,
                ),
                FormationProposal(
                    kind=RecordKind.EVENT,
                    local_id="launch",
                    content="The launch was canceled.",
                ),
                FormationProposal(
                    kind=RecordKind.RELATION,
                    content="Ana reports that the canceled launch elicited her anger.",
                    subject="Ana",
                    predicate="affect_elicitor",
                    value="canceled launch",
                    affect_link=AffectLink(
                        affect_id="proposal:feeling",
                        role="elicitor",
                        object_ref=AffectObjectRef(memory_id="proposal:launch"),
                    ),
                    affect_witnesses=(
                        AffectWitness(source_id=source.memory_id, reporter="Ana", quote=QUOTE),
                    ),
                ),
            )
            for source in inputs
        )

    def close(self) -> None:
        pass


@pytest.fixture
def attributed_record(tmp_path: Path) -> Iterator[MemoryRecord]:
    with Memory(
        tmp_path,
        embedder=_ContractEmbedder(),
        former=_AttributedFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add(QUOTE)
        (relation,) = tuple(
            row
            for row in memory.list(limit=10).items
            if row.context and row.context.kind is RecordKind.RELATION
        )
        assert relation.context is not None and relation.context.affect_link is not None
        link = relation.context.affect_link
        affect_context = memory.get(link.affect_id).context
        assert affect_context is not None and affect_context.kind is RecordKind.AFFECT
        assert link.object_ref.memory_id is not None
        event_context = memory.get(link.object_ref.memory_id).context
        assert event_context is not None and event_context.kind is RecordKind.EVENT
        assert relation.context.affect_witnesses == (
            AffectWitness(source_id=source.id, reporter="Ana", quote=QUOTE),
        )
        yield relation


@pytest.mark.parametrize(
    "link",
    (
        AffectLink(
            affect_id="feeling", role="target", object_ref=AffectObjectRef(identity_id="person")
        ),
        AffectLink(
            affect_id="feeling",
            role="elicitor",
            object_ref=AffectObjectRef(memory_id="event"),
            polarity="denied",
        ),
    ),
)
def test_public_attribution_values_round_trip_as_typed_values(link: AffectLink) -> None:
    adapter = TypeAdapter(AffectLink)
    assert adapter.validate_json(adapter.dump_json(link)) == link


def test_stored_attribution_context_and_operation_preserve_links_and_reporters(
    attributed_record: MemoryRecord,
) -> None:
    context = attributed_record.context
    assert context is not None
    adapter = TypeAdapter(RecordSemantics)
    assert adapter.validate_json(adapter.dump_json(context)) == context
    proposal = FormationProposal(
        kind=RecordKind.RELATION,
        local_id="explanation",
        content=attributed_record.content,
        subject=context.subject,
        predicate=context.predicate,
        value=context.value,
        affect_link=context.affect_link,
        affect_witnesses=context.affect_witnesses,
    )
    operation = MemoryOperation(
        intent=MemoryIntent.CONSOLIDATE,
        evidence_ids=tuple(sorted(context.evidence_ids)),
        proposal=proposal,
    )
    assert load_operation(dump_operation(operation)) == operation
    proposal_adapter = TypeAdapter(FormationProposal)
    assert proposal_adapter.validate_json(proposal_adapter.dump_json(proposal)) == proposal
    assert operation_key(operation, recipe="attribution-v1") != operation_key(
        replace(operation, proposal=replace(proposal, affect_witnesses=())), recipe="attribution-v1"
    )


@pytest.mark.parametrize("response_type", (MemoryResponse,))
def test_rest_and_mcp_preserve_stored_attribution_and_publish_its_schema(
    attributed_record: MemoryRecord,
    response_type: type[BaseModel],
) -> None:
    context = attributed_record.context
    assert context is not None and context.affect_link is not None
    response = response_type.model_validate(attributed_record)
    payload = response.model_dump(mode="json")
    assert payload["context"]["affect_link"] == {
        "affect_id": context.affect_link.affect_id,
        "role": "elicitor",
        "object_ref": {"memory_id": context.affect_link.object_ref.memory_id, "identity_id": None},
        "polarity": "affirmed",
    }
    assert payload["context"]["affect_witnesses"] == [
        {"source_id": context.affect_witnesses[0].source_id, "reporter": "Ana", "quote": QUOTE}
    ]
    assert response_type.model_validate_json(response.model_dump_json()) == response
    definitions = response_type.model_json_schema()["$defs"]
    assert {"affect_link", "affect_witnesses"} <= definitions["RecordSemantics"][
        "properties"
    ].keys()
    assert definitions["AffectLink"]["properties"]["role"]["enum"] == ["target", "elicitor"]
    assert definitions["AffectLink"]["properties"]["polarity"]["enum"] == ["affirmed", "denied"]
    assert {"memory_id", "identity_id"} == definitions["AffectObjectRef"]["properties"].keys()
    assert {"source_id", "reporter", "quote"} == definitions["AffectWitness"]["properties"].keys()
