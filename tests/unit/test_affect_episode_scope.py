"""A later explanation is about its original episode, not the reporter's new location."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AffectLink,
    AffectObjectRef,
    ContextBudget,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemorySettings,
    Modality,
    ObservationContext,
    RecordKind,
    RetrievalScope,
    SpatialAnchor,
    SpatialContext,
    ValidationError,
)

_OFFICE = SpatialContext(frame_id="home", anchor=SpatialAnchor.OBSERVER, x=0, y=0, z=0)
_KITCHEN = SpatialContext(frame_id="home", anchor=SpatialAnchor.OBSERVER, x=20, y=0, z=0)


class _Former:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "episode-scope-fixture"
    formation_space = "episode-scope-fixture:v1"

    def __init__(self, *, manual: bool) -> None:
        self.manual = manual
        self.affect_id = "proposal:cue"
        self.event_id = "proposal:event"

    def denial(self) -> FormationProposal:
        return FormationProposal(
            kind=RecordKind.RELATION,
            content="The user denies that cancellation caused the earlier anger.",
            subject="user",
            predicate="affect_elicitor",
            value="cancellation",
            affect_link=AffectLink(
                affect_id=self.affect_id,
                role="elicitor",
                object_ref=AffectObjectRef(memory_id=self.event_id),
                polarity="denied",
            ),
        )

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (() if self.manual else (self.denial(),))
            if source.content.text == "clarification"
            else (
                FormationProposal(
                    kind=RecordKind.AFFECT,
                    local_id="cue",
                    content="The user seems angry.",
                    subject="user",
                    value="angry",
                    cue_modality=Modality.TEXT,
                ),
                FormationProposal(
                    kind=RecordKind.EVENT,
                    local_id="event",
                    content="The activity was canceled.",
                ),
                FormationProposal(
                    kind=RecordKind.RELATION,
                    content="Cancellation may explain the user's anger.",
                    subject="user",
                    predicate="affect_elicitor",
                    value="cancellation",
                    affect_link=AffectLink(
                        affect_id="proposal:cue",
                        role="elicitor",
                        object_ref=AffectObjectRef(memory_id="proposal:event"),
                    ),
                ),
            )
            for source in inputs
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("manual", (False, True), ids=("formation", "consolidation"))
@pytest.mark.parametrize(
    "scope",
    (RetrievalScope(place_id="office"), RetrievalScope(near=_OFFICE, radius_m=1)),
    ids=("place", "near"),
)
def test_later_attributions_keep_the_original_episode_scope(
    tmp_path: Path, manual: bool, scope: RetrievalScope
) -> None:
    former = _Former(manual=manual)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(
            minimum_relevance=0,
            formation_history_max_rows=32,
            formation_history_budget_chars=16_000,
        ),
    ) as memory:
        memory.add(
            "The activity was canceled and I am angry.",
            context=ObservationContext(place_id="office", spatial=_OFFICE),
        )
        records = memory.list(limit=20).items
        former.affect_id = next(
            record.id
            for record in records
            if record.context and record.context.kind is RecordKind.AFFECT
        )
        former.event_id = next(
            record.id
            for record in records
            if record.context and record.context.kind is RecordKind.EVENT
        )
        clarification = memory.add(
            "clarification", context=ObservationContext(place_id="kitchen", spatial=_KITCHEN)
        )
        if manual:
            for omitted in (former.affect_id, former.event_id):
                with pytest.raises(ValidationError, match="invalid_proposal"):
                    memory.apply(
                        MemoryOperation(
                            intent=MemoryIntent.CONSOLIDATE,
                            evidence_ids=tuple(
                                memory_id
                                for memory_id in (
                                    former.affect_id,
                                    former.event_id,
                                    clarification.id,
                                )
                                if memory_id != omitted
                            ),
                            proposal=former.denial(),
                        )
                    )
            memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(former.affect_id, former.event_id, clarification.id),
                    proposal=former.denial(),
                )
            )

        links = tuple(
            hit
            for hit in memory.search("anger", limit=20, scope=scope).hits
            if hit.context and hit.context.affect_link
        )
        assert {
            hit.context.affect_link.polarity
            for hit in links
            if hit.context and hit.context.affect_link
        } == {
            "affirmed",
            "denied",
        }
        assert all(hit.place_id == "office" for hit in links)
        assert all(hit.context and hit.context.spatial == _OFFICE for hit in links)

        bundle = memory.compile(
            "why was the user angry", scope=scope, budget=ContextBudget(max_items=20)
        )
        delivered = {
            hit.context.affect_link.polarity
            for hit in bundle.relationships
            if hit.context and hit.context.affect_link
        }
        assert delivered in (set(), {"affirmed", "denied"})
