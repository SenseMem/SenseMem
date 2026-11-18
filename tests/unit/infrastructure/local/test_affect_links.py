"""Typed affect links retain competing claims and their exact source witnesses."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from sensemem.infrastructure.local import LocalStore, StoredMemory, StoredOperation
from sensemem.semantics import AffectLink, AffectObjectRef, AffectWitness, RecordSemantics
from sensemem.spatial import RetrievalScope
from sensemem.types import EvidenceBasis, Modality, RecordKind

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _memory(memory_id: str, context: RecordSemantics | None = None) -> StoredMemory:
    return StoredMemory(
        memory_id=memory_id,
        content=memory_id,
        metadata_json="{}",
        created_at=_NOW,
        updated_at=_NOW if context is None else context.recorded_at,
        context=context,
        place_id="office",
    )


def _scene(store: LocalStore) -> None:
    store.records.write_memories(tuple(_memory(name) for name in ("scene", "quote-1", "quote-2")))
    store.records.write_memories(
        tuple(
            _memory(
                name,
                RecordSemantics(
                    kind=kind,
                    basis=EvidenceBasis.MODEL_INFERENCE,
                    confidence=0.8,
                    valid_from=_NOW,
                    valid_until=None,
                    recorded_at=_NOW,
                    subject="Ana",
                    value="sad" if kind is RecordKind.AFFECT else None,
                    cue_modality=Modality.TEXT if kind is RecordKind.AFFECT else None,
                    evidence_ids=("scene",),
                    affect_witnesses=(
                        (AffectWitness(source_id="scene", reporter="Ana", quote="scene"),)
                        if kind is RecordKind.AFFECT
                        else ()
                    ),
                ),
            )
            for name, kind in (("affect", RecordKind.AFFECT), ("event", RecordKind.EVENT))
        )
    )


def _link(
    name: str,
    source: str,
    *,
    denied: bool = False,
    recorded_at: datetime = _NOW,
    object_ref: AffectObjectRef | None = None,
) -> StoredMemory:
    role: Literal["elicitor", "target"] = "elicitor" if object_ref is None else "target"
    return _memory(
        name,
        RecordSemantics(
            kind=RecordKind.RELATION,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.7,
            valid_from=_NOW,
            valid_until=None,
            recorded_at=recorded_at,
            subject="Ana",
            predicate=f"affect_{role}",
            value="event",
            evidence_ids=("affect", "event", source),
            affect_link=AffectLink(
                affect_id="affect",
                role=role,
                object_ref=object_ref or AffectObjectRef(memory_id="event"),
                polarity="denied" if denied else "affirmed",
            ),
            affect_witnesses=(AffectWitness(source_id=source, reporter="Ana", quote=source),),
        ),
    )


def _commit(store: LocalStore, memory: StoredMemory) -> None:
    assert memory.context is not None
    store.semantics.apply_formation(
        (memory,),
        (),
        evidence=(),
        evidence_clauses=((memory.memory_id, memory.context.evidence_ids, 0.7),),
        source_memory_ids=(),
        recipe="test-affect-links-v1",
        completed_at=memory.context.recorded_at,
    )


def test_affect_links_round_trip_both_polarities_and_report_a_scoped_bound(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        _scene(store)
        affirmed = _link("affirmed", "quote-1")
        denied = _link("denied", "quote-2", denied=True, recorded_at=_NOW + timedelta(minutes=1))
        _commit(store, affirmed)
        _commit(store, denied)

        links = store.semantics.affect_links(("affect",), limit=1)
        assert len(links) == 1
        assert links.selected == 2
        complete = store.semantics.affect_links(("affect",))
        assert complete.selected == len(complete) == 2
        assert affirmed.context is not None and denied.context is not None
        assert {row.context.affect_link for row in complete if row.context} == {
            affirmed.context.affect_link,
            denied.context.affect_link,
        }
        assert {row.context.affect_witnesses for row in complete if row.context} == {
            affirmed.context.affect_witnesses,
            denied.context.affect_witnesses,
        }
        historical = store.semantics.affect_links(
            ("affect",), scope=RetrievalScope(known_at=_NOW + timedelta(seconds=30))
        )
        assert historical.selected == 1
        assert [row.memory_id for row in historical] == ["affirmed"]
        outside = store.semantics.affect_links(
            ("affect",), scope=RetrievalScope(place_id="somewhere-else")
        )
        assert outside.selected == 0
        assert outside == ()
        affect = store.records.read_memory("affect")
        assert affect is not None and affect.context is not None
        assert affect.context.affect_witnesses == (
            AffectWitness(source_id="scene", reporter="Ana", quote="scene"),
        )


def test_withdrawal_filters_witnesses_and_deleting_last_support_removes_the_link(
    tmp_path: Path,
) -> None:
    with LocalStore(tmp_path) as store:
        _scene(store)
        _commit(store, _link("claim", "quote-1"))
        _commit(store, _link("claim", "quote-2", recorded_at=_NOW + timedelta(minutes=1)))
        (combined,) = store.semantics.affect_links(("affect",))
        assert combined.context is not None
        assert {witness.source_id for witness in combined.context.affect_witnesses} == {
            "quote-1",
            "quote-2",
        }

        store.records.set_forgotten(("quote-1",), forgotten_at=_NOW + timedelta(minutes=2))
        (remaining,) = store.semantics.affect_links(("affect",))
        assert remaining.context is not None
        assert [witness.source_id for witness in remaining.context.affect_witnesses] == ["quote-2"]
        assert store.records.delete_memory("quote-1") is True
        retained = store.records.read_memory("claim")
        assert retained is not None and retained.context is not None
        assert [witness.source_id for witness in retained.context.affect_witnesses] == ["quote-2"]
        with store._connections.connection() as connection:
            raw = connection.execute(
                "SELECT affect_witnesses_json FROM memory_semantics WHERE memory_id = 'claim'"
            ).fetchone()[0]
        assert json.loads(raw) == [{"source_id": "quote-2", "reporter": "Ana", "quote": "quote-2"}]
        assert store.records.delete_memory("quote-2") is True
        assert store.semantics.affect_links(("affect",)) == ()
        assert store.records.read_memory("affect") is not None
        assert store.records.read_memory("event") is not None


def test_correcting_an_event_withdraws_its_link_without_retiring_the_affect(tmp_path: Path) -> None:
    with LocalStore(tmp_path) as store:
        _scene(store)
        _commit(store, _link("claim", "quote-1"))
        assert (
            store.control.apply_control_operation(
                StoredOperation(
                    operation_key="correct-event",
                    intent="correct",
                    trigger="manual",
                    operation_json="{}",
                    applied_at=_NOW + timedelta(minutes=1),
                ),
                correct_ids=("event",),
            )
            is not None
        )

        assert store.semantics.affect_links(("affect",)) == ()
        (affect,) = store.records.read_memories(("affect",), active_only=True)
        assert affect.context is not None
        assert affect.context.retired_at is None


def test_identity_alias_resolves_and_erasure_deletes_links_to_the_whole_cluster(
    tmp_path: Path,
) -> None:
    with LocalStore(tmp_path) as store:
        _scene(store)
        with store._connections.transaction() as connection:
            stamp = _NOW.isoformat()
            connection.execute(
                "INSERT INTO identities(identity_id,name,created_at,updated_at) VALUES(?,?,?,?)",
                ("canonical", "Bob", stamp, stamp),
            )
            connection.execute(
                "INSERT INTO identity_aliases(alias_id,identity_id,created_at) VALUES(?,?,?)",
                ("original", "canonical", stamp),
            )
        claim = _link("target", "quote-1", object_ref=AffectObjectRef(identity_id="original"))
        _commit(store, claim)
        (resolved,) = store.semantics.affect_links(("affect",))
        assert resolved.context is not None and claim.context is not None
        assert claim.context.affect_link is not None
        assert resolved.context.affect_link == replace(
            claim.context.affect_link, object_ref=AffectObjectRef(identity_id="canonical")
        )
        assert store.identities.identity_assertion_memory_ids("canonical") == ("target",)
        assert store.identities.forget_identity("original") is not None
        assert store.records.read_memory("target") is None
        assert store.records.read_memory("affect") is not None
