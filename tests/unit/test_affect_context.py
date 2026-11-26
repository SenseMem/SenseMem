"""Attribution delivery requires scoped peers and pays for their complete evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Literal

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AffectLink,
    AffectObjectRef,
    AffectWitness,
    ContextBudget,
    ContextUnknownKind,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryCategory,
    MemorySettings,
    Modality,
    ObservationContext,
    RecordKind,
    RecordSemantics,
    SearchHit,
    SpatialAnchor,
    SpatialContext,
)
from sensemem.context import EvidenceClosure, compile_context
from sensemem.evidence import answer_evidence_payloads, retain_source_complete_hits
from sensemem.infrastructure.local.store.rows import RecallRead

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


def _hit(identifier: str, context: RecordSemantics | None = None, score: float = 0.5) -> SearchHit:
    return SearchHit(
        id=identifier,
        content=f"evidence {identifier}",
        score=score,
        created_at=NOW,
        memory_type=(
            MemoryCategory.EPISODIC
            if context is not None and context.kind is RecordKind.AFFECT
            else MemoryCategory.SEMANTIC
        ),
        modality=Modality.TEXT,
        context=context,
    )


def _link(identifier: str, polarity: Literal["affirmed", "denied"]) -> SearchHit:
    return _hit(
        identifier,
        RecordSemantics(
            kind=RecordKind.RELATION,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=1.0,
            valid_from=None,
            valid_until=None,
            recorded_at=NOW,
            subject="user",
            predicate="affect_elicitor",
            value="event",
            lineage_id="affect-link-lineage",
            evidence_ids=("source", "cue", "event"),
            affect_link=AffectLink(
                affect_id="cue",
                role="elicitor",
                object_ref=AffectObjectRef(memory_id="event"),
                polarity=polarity,
            ),
        ),
        score=0.9 if polarity == "affirmed" else 0.0,
    )


def _context_fixture() -> tuple[tuple[SearchHit, ...], tuple[EvidenceClosure, ...]]:
    source = _hit("source")
    event = _hit("event")
    cue = _hit(
        "cue",
        RecordSemantics(
            kind=RecordKind.AFFECT,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=1.0,
            valid_from=None,
            valid_until=None,
            recorded_at=NOW,
            evidence_ids=("source",),
            cue_modality=Modality.TEXT,
            valence=0.5,
            arousal=0.2,
        ),
    )
    affirmed, denied = _link("affirmed", "affirmed"), _link("denied", "denied")
    hits = (affirmed, cue, event, source, denied)
    closures = (
        EvidenceClosure(affirmed, (affirmed, cue, event, source)),
        EvidenceClosure(cue, (cue, source)),
        EvidenceClosure(event, (event,)),
        EvidenceClosure(source, (source,)),
        EvidenceClosure(denied, (denied, cue, event, source)),
    )
    return hits, closures


def test_affect_link_peers_are_atomic_and_cues_survive_a_small_budget() -> None:
    hits, closures = _context_fixture()
    full = compile_context(
        "why",
        hits,
        closures=closures,
        budget=ContextBudget(max_items=5),
        reference_at=NOW,
    )
    assert {hit.id for hit in full.relationships} == {"affirmed", "denied"}
    lines = [
        line
        for line in full.render().splitlines()
        if line.startswith(("- [affirmed]", "- [denied]"))
    ]
    assert len(lines) == 2
    assert "affirmed" in lines[0] and "denied" in lines[1]
    assert all("cue" in line and "event" in line and "elicitor" in line for line in lines)
    assert full.chars >= len(full.render())
    small = compile_context(
        "why",
        hits,
        closures=closures,
        budget=ContextBudget(max_items=4),
        reference_at=NOW,
    )
    assert small.relationships == ()
    assert {cue.id for cue in small.affect} == {"cue"}
    assert any(unknown.kind is ContextUnknownKind.BUDGET_EXCLUDED for unknown in small.unknowns)


def test_affect_link_filter_and_deadline_never_leave_one_polarity() -> None:
    hits, closures = _context_fixture()
    denied = hits[-1]
    assert denied.context is not None
    low = replace(denied, context=replace(denied.context, confidence=0.1))
    filtered = compile_context(
        "why",
        (*hits[:-1], low),
        closures=closures,
        budget=ContextBudget(max_items=5, min_confidence=0.5),
        reference_at=NOW,
    )
    assert filtered.relationships == ()
    assert filtered.affect
    expired = compile_context(
        "why",
        hits,
        closures=closures,
        budget=ContextBudget(max_items=5, max_latency_ms=1),
        reference_at=NOW,
        started_at=perf_counter() - 1,
    )
    assert expired.relationships == ()
    assert expired.affect
    assert any("attributions" in unknown.detail for unknown in expired.unknowns)


def test_answer_packing_preserves_attribution_pairs_and_aliases_witnesses() -> None:
    hits, _closures = _context_fixture()
    assert {hit.id for hit in retain_source_complete_hits(hits[:-1], hits)} == {
        "cue",
        "event",
        "source",
    }
    link = hits[0]
    assert link.context is not None
    witnessed = replace(
        link,
        context=replace(
            link.context,
            affect_witnesses=(AffectWitness(source_id="source", reporter="user", quote="happy"),),
        ),
    )
    payload = answer_evidence_payloads((witnessed, *hits[1:]))[0]
    assert payload["context"] == {
        "kind": "relation",
        "basis": "model_inference",
        "confidence": 1.0,
        "recorded_at": NOW.isoformat(),
        "subject": "user",
        "predicate": "affect_elicitor",
        "value": "event",
        "evidence_labels": ["E4", "E2", "E3"],
        "supporting_record_count": 3,
        "affect_link": {
            "affect_label": "E2",
            "role": "elicitor",
            "object_ref": {"memory_label": "E3"},
            "polarity": "affirmed",
        },
        "affect_witnesses": [{"source_label": "E4", "reporter": "user", "quote": "happy"}],
    }


class _Former:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "attribution-fixture"
    formation_space = "attribution-fixture:v1"
    affect_id: str | None = None
    event_id: str | None = None

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        results: list[tuple[FormationProposal, ...]] = []
        for item in inputs:
            if self.affect_id is None or self.event_id is None:
                results.append(
                    (
                        FormationProposal(
                            kind=RecordKind.EVENT,
                            content="the user won a prize",
                            local_id="0",
                        ),
                        FormationProposal(
                            kind=RecordKind.AFFECT,
                            content="the user seems happy",
                            subject="user",
                            value="happy",
                            cue_modality=Modality.TEXT,
                            valence=0.6,
                            arousal=0.3,
                            local_id="1",
                        ),
                        FormationProposal(
                            kind=RecordKind.RELATION,
                            content="the prize may explain the user's happiness",
                            subject="user",
                            predicate="affect_elicitor",
                            value="prize",
                            affect_link=AffectLink(
                                affect_id="proposal:1",
                                role="elicitor",
                                object_ref=AffectObjectRef(memory_id="proposal:0"),
                            ),
                        ),
                    )
                )
            else:
                results.append(
                    (
                        FormationProposal(
                            kind=RecordKind.RELATION,
                            content="the prize did not explain the user's happiness",
                            subject="user",
                            predicate="affect_elicitor",
                            value="prize",
                            evidence_ids=(item.memory_id, self.affect_id, self.event_id),
                            affect_link=AffectLink(
                                affect_id=self.affect_id,
                                role="elicitor",
                                object_ref=AffectObjectRef(memory_id=self.event_id),
                                polarity="denied",
                            ),
                        ),
                    )
                )
        return tuple(results)

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "source_frame,link_frame,accepted",
    [(None, "invented", False), ("observed", "invented", False), ("observed", "observed", True)],
)
def test_affect_link_spatial_grounding_refuses_only_the_invalid_link(
    tmp_path: Path, source_frame: str | None, link_frame: str, accepted: bool
) -> None:
    link_spatial = SpatialContext(frame_id=link_frame, anchor=SpatialAnchor.SUBJECT, x=1, y=2, z=0)

    class SpatialFormer(_Former):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                tuple(
                    replace(proposal, spatial=link_spatial)
                    if proposal.affect_link is not None
                    else proposal
                    for proposal in proposals
                )
                for proposals in super().form(inputs)
            )

    source_spatial = (
        None
        if source_frame is None
        else SpatialContext(frame_id=source_frame, anchor=SpatialAnchor.SUBJECT, x=1, y=2, z=0)
    )
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=SpatialFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add(
            "The user is happy while receiving a prize.",
            context=ObservationContext(spatial=source_spatial),
        )
        records = memory.list(limit=20).items
        ordinary = [
            row
            for row in records
            if row.context and row.context.kind in {RecordKind.AFFECT, RecordKind.EVENT}
        ]
        assert {row.context.kind for row in ordinary if row.context} == {
            RecordKind.AFFECT,
            RecordKind.EVENT,
        }
        links = [row for row in records if row.context and row.context.affect_link is not None]
        assert len(links) == int(accepted)
        if links:
            assert links[0].context is not None and links[0].context.spatial == source_spatial
        assert memory.get(source.id).content == source.content
        assert memory.pending_captures() == ()


@pytest.mark.parametrize(("truncated", "max_items"), ((False, 12), (True, 12), (False, 4)))
def test_compile_reads_unranked_denial_from_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, truncated: bool, max_items: int
) -> None:
    former = _Former()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        memory.add("The user is happy while receiving a prize.")
        records = memory.list(limit=20).items
        former.affect_id = next(
            row.id for row in records if row.context and row.context.kind is RecordKind.AFFECT
        )
        former.event_id = next(
            row.id for row in records if row.context and row.context.kind is RecordKind.EVENT
        )
        memory.add("The user says the happiness was not because of the prize.")
        links = [
            row for row in memory.list(limit=30).items if row.context and row.context.affect_link
        ]
        assert len(links) == 2
        positive = next(
            row
            for row in links
            if row.context
            and row.context.affect_link
            and row.context.affect_link.polarity == "affirmed"
        )
        stored = memory._store.records.read_memory(positive.id)
        assert stored is not None

        def ranked_positive(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                hits=(memory._hydrator.search_hit(stored, 0.9),),
                matched_dense_index_ids={},
            )

        monkeypatch.setattr(memory._retrieval, "search_prepared", ranked_positive)
        if truncated:

            def clipped(*args: object, **kwargs: object) -> RecallRead:
                return RecallRead((stored,), selected=2)

            monkeypatch.setattr(memory._store.semantics, "affect_links", clipped)
        bundle = memory.compile(
            "why happy", budget=ContextBudget(max_items=max_items, max_chars=20_000)
        )
        assert bundle.affect
        if truncated:
            assert bundle.relationships == ()
            assert any("attributions" in unknown.detail for unknown in bundle.unknowns)
        elif max_items == 4:
            assert bundle.relationships == ()
            assert any(
                unknown.kind is ContextUnknownKind.EVIDENCE_UNAVAILABLE
                and "attributions" in unknown.detail
                for unknown in bundle.unknowns
            )
        else:
            assert {hit.id for hit in bundle.relationships} == {row.id for row in links}
