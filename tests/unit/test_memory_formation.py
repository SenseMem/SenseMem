from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

import httpx2 as httpx
import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sensemem import (
    Blob,
    EmbedTask,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryNotFoundError,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    MemoryTrigger,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    QueryFailure,
    RecordKind,
    RetrievalScope,
    SearchHit,
    SpatialAnchor,
    SpatialContext,
    ValidationError,
)
from sensemem._telemetry import FORMATION_PROPOSALS_REFUSED
from sensemem.models.openai_completion import OpenAICompletion


class PreferenceFormer:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = "preference-test"
    formation_space = "preference-test:v1"

    def __init__(self) -> None:
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        self.calls += 1
        return tuple(
            (
                FormationProposal(
                    kind=RecordKind.ENTITY,
                    content="The user is an entity",
                    subject="user",
                    confidence=0.99,
                ),
                FormationProposal(
                    kind=RecordKind.STATE,
                    content="The user's preferred drink is tea",
                    subject="user",
                    predicate="preferred_drink",
                    value="tea",
                    confidence=0.9,
                    valid_from=value.context.valid_from,
                ),
            )
            for value in inputs
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("withdrawn,cite_name", [(False, False), (False, True), (True, False)])
def test_history_named_events_require_a_standing_cited_name(
    tmp_path: Path,
    withdrawn: bool,
    cite_name: bool,
) -> None:
    class NamedEventFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            self.calls += 1
            if self.calls == 1:
                return (
                    (
                        FormationProposal(
                            kind=RecordKind.ENTITY, subject="Ana", content="The person is Ana"
                        ),
                    ),
                )
            source = inputs[0]
            naming = next(
                record
                for record in source.history
                if record.context is not None and record.context.kind is RecordKind.ENTITY
            )
            witnesses = (source.memory_id, naming.id) if cite_name else (source.memory_id,)
            return (
                (
                    FormationProposal(
                        kind=RecordKind.EVENT,
                        subject="Ana",
                        content="Ana entered the room",
                        evidence_ids=witnesses,
                    ),
                    FormationProposal(kind=RecordKind.EVENT, content="A person entered the room"),
                ),
            )

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=NamedEventFormer(),
        settings=MemorySettings(minimum_relevance=0, formation_history_max_rows=8),
    ) as memory:
        memory.add("A person is introduced as Ana")
        name = next(
            record
            for record in memory.list(limit=20).items
            if record.context is not None and record.context.kind is RecordKind.ENTITY
        )
        if withdrawn:
            memory.apply(MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(name.id,)))
        memory.add("A person enters the room")
        contents = {record.content for record in memory.list(limit=30).items}
        assert "A person entered the room" in contents
        assert ("Ana entered the room" in contents) is (cite_name and not withdrawn)


def test_new_observation_forms_typed_memories_once_without_changing_source_identity(
    tmp_path: Path,
) -> None:
    occurred = datetime(2026, 1, 2, tzinfo=timezone.utc)
    context = ObservationContext(
        basis=EvidenceBasis.USER_STATEMENT,
        source_id="turn-17",
        valid_from=occurred,
    )
    former = PreferenceFormer()

    with Memory(
        tmp_path / "formed",
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add("I prefer tea", occurred_at=occurred, context=context)
        duplicate = memory.add("I prefer tea", occurred_at=occurred, context=context)
        hits = memory.search("preferred drink", limit=10).hits

        assert source == duplicate
        assert former.calls == 1
        assert source.context is not None
        assert source.context.kind is RecordKind.OBSERVATION
        states = [hit for hit in hits if hit.context and hit.context.kind is RecordKind.STATE]
        assert len(states) == 1
        state = states[0].context
        assert state is not None
        assert state.subject == "user"
        assert state.predicate == "preferred_drink"
        assert state.value == "tea"
        assert state.evidence_ids == (source.id,)
        assert state.model_id == "preference-test"

    with Memory(
        tmp_path / "plain", embedder=TinyEmbedder(), settings=MemorySettings(minimum_relevance=0)
    ) as memory:
        plain = memory.add("I prefer tea", occurred_at=occurred, context=context)

    assert plain.id == source.id


def test_distinct_sources_can_form_concurrently(tmp_path: Path) -> None:
    both = Barrier(2, timeout=3)

    class ParallelFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            both.wait()
            return super().form(inputs)

    former = ParallelFormer()
    with Memory(tmp_path, embedder=TinyEmbedder(), former=former) as memory:
        with ThreadPoolExecutor(max_workers=2) as callers:
            records = tuple(callers.map(memory.add, ("first source", "second source")))
        assert len({record.id for record in records}) == 2
        assert former.calls == 2


def test_concurrent_duplicate_source_runs_formation_once(tmp_path: Path) -> None:
    occurred = datetime(2026, 1, 2, tzinfo=timezone.utc)
    context = ObservationContext(source_id="same-turn", valid_from=occurred)
    former = PreferenceFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        with ThreadPoolExecutor(max_workers=2) as executor:
            records = tuple(
                executor.map(
                    lambda _index: memory.add(
                        "I prefer tea",
                        occurred_at=occurred,
                        context=context,
                    ),
                    range(2),
                )
            )

        assert records[0] == records[1]
        assert former.calls == 1


def test_implicit_and_explicit_default_context_have_one_identity(tmp_path: Path) -> None:
    former = PreferenceFormer()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        implicit = memory.add("I prefer tea")
        explicit = memory.add("I prefer tea", context=ObservationContext())

        assert implicit == explicit
        assert former.calls == 1

        implicit_batch = memory.add_many(("batch tea one", "batch tea two"))
        explicit_batch = memory.add_many(
            ("batch tea one", "batch tea two"),
            context=(ObservationContext(), ObservationContext()),
        )
        assert implicit_batch == explicit_batch
        assert former.calls == 2


def test_known_at_only_exposes_evidence_known_at_that_time(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        first = memory.add(
            "first entity witness",
            context=ObservationContext(source_id="turn-1"),
        )
        known_after_first = datetime.now(timezone.utc)
        second = memory.add(
            "second entity witness",
            context=ObservationContext(source_id="turn-2"),
        )

        historical = next(
            hit.context
            for hit in memory.search(
                "user entity",
                limit=10,
                scope=RetrievalScope(known_at=known_after_first),
            ).hits
            if hit.context is not None and hit.context.kind is RecordKind.ENTITY
        )
        current = next(
            hit.context
            for hit in memory.search("user entity", limit=10).hits
            if hit.context is not None and hit.context.kind is RecordKind.ENTITY
        )

        assert historical.evidence_ids == (first.id,)
        assert current.evidence_ids == (first.id, second.id)


def test_deleting_a_source_removes_unsupported_derived_records(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add("I prefer tea")
        assert any(
            hit.context is not None for hit in memory.search("preferred drink", limit=10).hits
        )

        assert memory.delete(source.id) is True
        assert memory.delete(source.id) is False
        assert memory.search("preferred drink", limit=10).hits == ()
        assert memory.compile("preferred drink").hits == ()


def test_unsupported_source_modality_is_kept_without_calling_the_former(tmp_path: Path) -> None:
    former = PreferenceFormer()
    former.formation_capabilities = frozenset({Modality.TEXT})
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        first = memory.add(Blob(b"audio", "audio/wav"))
        duplicate = memory.add(Blob(b"audio", "audio/wav"))

        assert first == duplicate
        assert former.calls == 0
        assert first.context is not None
        assert first.context.kind is RecordKind.OBSERVATION

    former.formation_capabilities = ATOMIC_MODALITIES
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        upgraded = memory.add(Blob(b"audio", "audio/wav"))
        assert upgraded == first
        assert former.calls == 1


def test_a_proposal_the_kernel_refuses_is_dropped_and_not_charged_to_the_write(
    tmp_path: Path,
) -> None:
    """A per-proposal rule costs that proposal, never the observation it was derived from.

    add commits the source before formation runs, so raising failed a write that had in fact
    succeeded -- and the record stayed durable with its formation stuck in the queue, so every
    retry re-ran the model and failed the same way. Text that merely mentions an image ("shared a
    photo described as ...") is enough to make a model call it an image cue, which made this the
    ordinary case rather than a rare one. The model adapter already drops and counts a malformed
    proposal; the kernel's own rules now agree with that policy.
    """

    class MisgroundedAffectFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=RecordKind.STATE,
                        content="The user's preferred drink is tea",
                        subject="user",
                        predicate="preferred_drink",
                        value="tea",
                        confidence=0.9,
                    ),
                    # The source is text; no image was ever observed.
                    FormationProposal(
                        kind=RecordKind.AFFECT,
                        content="The user looked happy",
                        subject="user",
                        value="happy",
                        cue_modality=Modality.IMAGE,
                    ),
                )
                for _value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=MisgroundedAffectFormer(),
        settings=MemorySettings(minimum_relevance=0),
        tracer=provider.get_tracer("test"),
    ) as memory:
        source = memory.add("I prefer tea, and I shared a photo described as a smile")

        assert [item.id for item in memory.list().items] != []
        kinds = {
            hit.context.kind
            for hit in memory.search("preferred drink happy", limit=10).hits
            if hit.context is not None
        }
        assert RecordKind.STATE in kinds
        assert RecordKind.AFFECT not in kinds
        assert memory.pending_captures(memory_ids=(source.id,)) == ()

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.attributes is not None and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]


def test_a_damaged_formation_envelope_still_fails_the_write(tmp_path: Path) -> None:
    """Dropping one bad proposal must not swallow a backend that answered the wrong question."""

    class ShortBatchFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            assert inputs
            return ()

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=ShortBatchFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        with pytest.raises(ModelError) as failure:
            memory.add("I prefer tea")

        assert failure.value.reason == "response_invalid"


def test_a_trait_formed_from_two_observations_in_one_batch_is_one_assessment(
    tmp_path: Path,
) -> None:
    """Formation is an AND writer: one proposal citing two observations is one conjunction.

    The two-independent-groups rule for TRAIT visibility counts assessments, not cited
    observations. One model call that read both observations and inferred the trait once must not
    corroborate itself, so the trait stays hidden until a separate operation supports it, and
    withdrawing either cited observation withdraws the whole conjunction.
    """

    class JointTraitFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            assert len(inputs) == 2
            return (
                (),
                (
                    FormationProposal(
                        kind=RecordKind.TRAIT,
                        content="Lin Yue is patient",
                        subject="Lin Yue",
                        predicate="disposition",
                        value="patient",
                        confidence=0.6,
                        evidence_ids=(inputs[0].memory_id, inputs[1].memory_id),
                    ),
                ),
            )

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=JointTraitFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        first, second = memory.add_many(("Lin Yue waited calmly.", "Lin Yue waited again."))
        with closing(sqlite3.connect(memory._store.database_path)) as connection:
            trait_id, visible = connection.execute(
                """
                SELECT s.memory_id, v.visible
                FROM memory_semantics AS s
                JOIN memory_versions AS v ON v.memory_id = s.memory_id
                WHERE s.kind = 'trait' AND v.retired_at IS NULL
                """
            ).fetchone()
            clauses = connection.execute(
                """
                SELECT c.member_count FROM memory_evidence_clauses AS c
                WHERE c.memory_id = ? AND c.retired_at IS NULL
                """,
                (trait_id,),
            ).fetchall()

        assert clauses == [(2,)]
        assert visible == 0
        assert not [hit for hit in memory.search("patient", limit=10).hits if hit.id == trait_id]
        hidden = memory.get(trait_id)
        assert hidden.context is not None
        assert hidden.context.evidence_ids == (first.id, second.id)

        # Withdrawing one member of the conjunction withdraws the whole assessment.
        assert memory.delete(second.id) is True
        with pytest.raises(MemoryNotFoundError):
            memory.get(trait_id)


@pytest.mark.parametrize("deleted_source", (0, 1))
def test_batch_witnesses_are_persisted_and_compiled_as_one_evidence_closure(
    tmp_path: Path, deleted_source: int
) -> None:
    """A proposal resolved from two observations must cite both durable source IDs."""

    class CrossWitnessFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            assert len(inputs) == 2
            return (
                (),
                (
                    FormationProposal(
                        kind=RecordKind.STATE,
                        content="Lin Yue lives in Suzhou",
                        subject="Lin Yue",
                        predicate="lives_in",
                        value="Suzhou",
                        confidence=0.6,
                        evidence_ids=(inputs[0].memory_id, inputs[1].memory_id),
                    ),
                ),
            )

    with Memory(
        tmp_path / str(deleted_source),
        embedder=TinyEmbedder(),
        former=CrossWitnessFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        first, second = memory.add_many(("My sister is named Lin Yue.", "She lives in Suzhou."))
        formed = next(
            hit
            for hit in memory.search("Lin Yue lives Suzhou", limit=10).hits
            if hit.context is not None and hit.context.kind is RecordKind.STATE
        )

        assert formed.context is not None
        assert formed.context.evidence_ids == (first.id, second.id)
        assert formed.context.confidence == pytest.approx(0.6)
        with closing(sqlite3.connect(memory._store.database_path)) as connection:
            clauses = connection.execute(
                """
                SELECT c.clause_id, m.source_memory_id
                FROM memory_evidence_clauses AS c
                JOIN memory_evidence_clause_members AS m
                  ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
                WHERE c.memory_id = ? AND c.retired_at IS NULL
                ORDER BY c.clause_id, m.position
                """,
                (formed.id,),
            ).fetchall()
        assert len({str(row[0]) for row in clauses}) == 1
        assert [str(row[1]) for row in clauses] == [first.id, second.id]
        bundle = memory.compile("Where does Lin Yue live?")
        assert {first.id, second.id, formed.id} <= {hit.id for hit in bundle.hits}

        assert memory.delete((first, second)[deleted_source].id) is True
        with pytest.raises(MemoryNotFoundError):
            memory.get(formed.id)


def test_singleton_alternative_survives_withdrawal_of_joint_witness(
    tmp_path: Path,
) -> None:
    class AlternativeWitnessFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return (
                (),
                (
                    FormationProposal(
                        kind=RecordKind.ENTITY,
                        content="Lin Yue is a person",
                        subject="Lin Yue",
                        evidence_ids=(inputs[0].memory_id, inputs[1].memory_id),
                    ),
                ),
                (
                    FormationProposal(
                        kind=RecordKind.ENTITY,
                        content="Lin Yue is a person",
                        subject="Lin Yue",
                    ),
                ),
            )

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=AlternativeWitnessFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        first, _second, third = memory.add_many(("A", "B", "C"))
        entity = next(
            hit
            for hit in memory.search("Lin Yue person", limit=10).hits
            if hit.context is not None and hit.context.kind is RecordKind.ENTITY
        )

        assert memory.delete(first.id) is True
        survivor = memory.get(entity.id)
        assert survivor.context is not None
        assert survivor.context.evidence_ids == (third.id,)


def test_missing_index_rebuild_uses_sqlite_clause_withdrawal_without_reembedding(
    tmp_path: Path,
) -> None:
    class CountingEmbedder(TinyEmbedder):
        def __init__(self) -> None:
            self.document_inputs = 0

        def embed(
            self,
            inputs: Sequence[ModelInput],
            task: EmbedTask = EmbedTask.DOCUMENT,
        ) -> tuple[tuple[float, ...], ...]:
            batch = tuple(inputs)
            if task is EmbedTask.DOCUMENT:
                self.document_inputs += len(batch)
            return super().embed(batch, task)

    class CrossWitnessFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return (
                (),
                (
                    FormationProposal(
                        kind=RecordKind.STATE,
                        content="Lin Yue lives in Suzhou",
                        subject="Lin Yue",
                        predicate="lives_in",
                        value="Suzhou",
                        evidence_ids=(inputs[0].memory_id, inputs[1].memory_id),
                    ),
                ),
            )

    data_dir = tmp_path / "rebuild"
    with Memory(
        data_dir,
        embedder=CountingEmbedder(),
        former=CrossWitnessFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        first, _second = memory.add_many(("My sister is Lin Yue", "She lives in Suzhou"))
        derived_id = next(
            hit.id
            for hit in memory.search("Lin Yue Suzhou", limit=10).hits
            if hit.context is not None and hit.context.kind is RecordKind.STATE
        )

    shutil.rmtree(data_dir / "zvec")
    rebuilt = CountingEmbedder()
    with Memory(
        data_dir,
        embedder=rebuilt,
        former=CrossWitnessFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        assert derived_id in {hit.id for hit in memory.search("Lin Yue Suzhou", limit=10).hits}
        assert rebuilt.document_inputs == 0
        assert memory.delete(first.id) is True

    shutil.rmtree(data_dir / "zvec")
    rebuilt_after_withdrawal = CountingEmbedder()
    with Memory(
        data_dir,
        embedder=rebuilt_after_withdrawal,
        former=CrossWitnessFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        assert derived_id not in {hit.id for hit in memory.search("Lin Yue Suzhou", limit=10).hits}
        assert rebuilt_after_withdrawal.document_inputs == 0


@pytest.mark.parametrize("evidence_ids", ((), ("duplicate", "duplicate")))
def test_formation_proposal_rejects_empty_or_duplicate_witnesses(
    evidence_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        FormationProposal(
            kind=RecordKind.EVENT,
            content="bad",
            evidence_ids=evidence_ids,
        )


@pytest.mark.parametrize("mode", ("foreign", "missing_primary"))
def test_invalid_explicit_batch_witnesses_leave_valid_proposals_intact(
    tmp_path: Path, mode: str
) -> None:
    class InvalidWitnessFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            assert len(inputs) == 2
            return tuple(
                (
                    FormationProposal(
                        kind=RecordKind.EVENT,
                        content="bad",
                        evidence_ids=(
                            ("foreign", source.memory_id)
                            if mode == "foreign"
                            else (inputs[1 - index].memory_id,)
                        ),
                    ),
                    FormationProposal(
                        kind=RecordKind.EVENT,
                        content=f"Event from {source.content.text}",
                        evidence_ids=(source.memory_id,),
                    ),
                )
                for index, source in enumerate(inputs)
            )

    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=InvalidWitnessFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        sources = memory.add_many(("first source", "second source"))
        events = {
            item.content: item.context.evidence_ids
            for item in memory.list().items
            if item.context is not None and item.context.kind is RecordKind.EVENT
        }
        assert events == {f"Event from {record.content}": (record.id,) for record in sources}
        assert memory.pending_captures() == ()


def _formed(
    memory: Memory, source_id: str, *, scope: RetrievalScope | None = None
) -> list[SearchHit]:
    hits = memory.search("preferred drink user", limit=10, scope=scope).hits
    return [hit for hit in hits if hit.id != source_id]


def _entities(memory: Memory, *, scope: RetrievalScope | None = None) -> list[SearchHit]:
    hits = memory.search("entity user", limit=10, scope=scope).hits
    return [
        hit for hit in hits if hit.context is not None and hit.context.kind is RecordKind.ENTITY
    ]


def test_formed_records_inherit_the_place_they_were_observed_in(tmp_path: Path) -> None:
    """ "What do we know about the kitchen?" must see the knowledge, not only the raw observation.

    place_id is a hard filter on the record, so a formed record that does not carry the place of
    the observation it was formed from is unreachable from every place-scoped search, ask, and
    compile -- the household question the symbolic axis exists for.
    """
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add("I prefer tea", context=ObservationContext(place_id="kitchen"))

        kitchen = _formed(memory, source.id, scope=RetrievalScope(place_id="kitchen"))
        garage = _formed(memory, source.id, scope=RetrievalScope(place_id="garage"))

        assert {hit.context.kind for hit in kitchen if hit.context} == {
            RecordKind.ENTITY,
            RecordKind.STATE,
        }
        assert {hit.place_id for hit in kitchen} == {"kitchen"}
        assert garage == []
        assert {memory.get(hit.id).place_id for hit in kitchen} == {"kitchen"}


def test_formed_records_inherit_the_metadata_of_their_source(tmp_path: Path) -> None:
    """A host that filters recall by metadata expects formed knowledge to carry the same tag."""
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add("I prefer tea", metadata={"household": "flat-2"})

        formed = _formed(memory, source.id)

        assert formed
        assert all(hit.metadata == {"household": "flat-2"} for hit in formed)


def test_a_formed_record_from_a_placeless_source_has_no_place(tmp_path: Path) -> None:
    """Inheriting a place must not invent one: an unlabelled observation is not "everywhere"."""
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add("I prefer tea")

        formed = _formed(memory, source.id)

        assert formed
        assert all(hit.place_id is None for hit in formed)
        assert _formed(memory, source.id, scope=RetrievalScope(place_id="kitchen")) == []


def test_a_shared_formed_record_keeps_only_what_its_sources_agree_on(tmp_path: Path) -> None:
    """An ENTITY is written once and later sources only add evidence to it.

    Its ID deliberately excludes the source, so the first observation's place and metadata would
    otherwise stand for every later source, including one that disagrees -- filing knowledge in a
    room half its evidence was never observed in. Agreement is the same condition consolidation
    applies to several sources at once.
    """
    with Memory(
        tmp_path / "disagree",
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        memory.add(
            "I prefer tea",
            context=ObservationContext(place_id="kitchen"),
            metadata={"household": "flat-2"},
        )
        memory.add(
            "I still prefer tea",
            context=ObservationContext(place_id="garage"),
            metadata={"household": "flat-9"},
        )

        (entity,) = _entities(memory)
        assert memory.get(entity.id).place_id is None
        assert memory.get(entity.id).metadata == {}
        assert _entities(memory, scope=RetrievalScope(place_id="kitchen")) == []
        assert _entities(memory, scope=RetrievalScope(place_id="garage")) == []

    with Memory(
        tmp_path / "agree",
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        memory.add(
            "I prefer tea",
            context=ObservationContext(place_id="kitchen"),
            metadata={"household": "flat-2"},
        )
        memory.add(
            "I still prefer tea",
            context=ObservationContext(place_id="kitchen"),
            metadata={"household": "flat-2"},
        )

        (entity,) = _entities(memory, scope=RetrievalScope(place_id="kitchen"))
        assert memory.get(entity.id).place_id == "kitchen"
        assert memory.get(entity.id).metadata == {"household": "flat-2"}


def test_deleting_the_disagreeing_source_gives_a_shared_record_its_place_back(
    tmp_path: Path,
) -> None:
    """Cleared inherited columns follow the live evidence, not the moment the record was written.

    Deleting a source retires its evidence, so the survivors may now agree on the place and the
    tag the disagreement cleared. Leaving the record unscoped would keep it out of every
    place-filtered read for evidence that no longer exists.
    """
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PreferenceFormer(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        memory.add(
            "I prefer tea",
            context=ObservationContext(place_id="kitchen"),
            metadata={"household": "flat-2"},
        )
        garage = memory.add(
            "I still prefer tea",
            context=ObservationContext(place_id="garage"),
            metadata={"household": "flat-9"},
        )
        (entity,) = _entities(memory)
        assert memory.get(entity.id).place_id is None

        assert memory.delete(garage.id) is True

        assert memory.get(entity.id).place_id == "kitchen"
        assert memory.get(entity.id).metadata == {"household": "flat-2"}
        assert [hit.id for hit in _entities(memory, scope=RetrievalScope(place_id="kitchen"))] == [
            entity.id
        ]


def test_two_contradictory_states_from_one_source_cost_only_the_later_one(
    tmp_path: Path,
) -> None:
    """Two proposals that contradict each other are one model response's grounding fault.

    Both come from a single source the caller already committed, so failing the write reports an
    observation as unwritten that is in fact stored, and every retry re-runs the model and fails
    the same way. The first claim stands, the contradicting one is refused.
    """

    class ContradictingFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                tuple(
                    FormationProposal(
                        kind=RecordKind.STATE,
                        content=f"The user's preferred drink is {drink}",
                        subject="user",
                        predicate="preferred_drink",
                        value=drink,
                        confidence=0.9,
                    )
                    for drink in ("tea", "coffee")
                )
                for _value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=ContradictingFormer(),
        settings=MemorySettings(minimum_relevance=0),
        tracer=provider.get_tracer("test"),
    ) as memory:
        source = memory.add("I prefer tea")

        states = [
            hit.context.value
            for hit in _formed(memory, source.id)
            if hit.context is not None and hit.context.kind is RecordKind.STATE
        ]
        assert states == ["tea"]
        assert memory.pending_captures(memory_ids=(source.id,)) == ()

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.attributes is not None and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]


def test_refused_proposals_are_totalled_over_the_whole_settle(tmp_path: Path) -> None:
    """settle forms one record at a time, and the operation reports the whole batch.

    Publishing per pass let the last record settled overwrite the count, so a batch whose first
    record lost a proposal reported zero refusals -- the reading an operator would take as proof
    that nothing was lost.
    """

    class PhotoAffectFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=RecordKind.AFFECT,
                        content="The user looked happy",
                        subject="user",
                        value="happy",
                        # The source is text; no image was ever observed.
                        cue_modality=Modality.IMAGE,
                    ),
                )
                if "photo" in (value.content.text or "")
                else ()
                for value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PhotoAffectFormer(),
        settings=MemorySettings(minimum_relevance=0),
        tracer=provider.get_tracer("test"),
    ) as memory:
        memory.capture("I shared a photo described as a smile")
        memory.capture("I prefer tea")

        assert memory.settle() == 2

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.name == "sensemem.settle"
        and span.attributes is not None
        and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]


def test_a_pose_in_another_frame_is_refused_on_the_formation_path(tmp_path: Path) -> None:
    """The spatial rule refuses like the affect rule: the proposal is lost, the write is not.

    A pose the kernel cannot place -- another coordinate frame, another anchor, or a source that
    was never localized -- would otherwise answer a metric scope with a position nothing observed.
    """

    class MisplacedPoseFormer(PreferenceFormer):
        def form(
            self, inputs: Sequence[FormationInput]
        ) -> tuple[tuple[FormationProposal, ...], ...]:
            return tuple(
                (
                    FormationProposal(
                        kind=RecordKind.EVENT,
                        content="The user poured tea in the study",
                        subject="user",
                        spatial=SpatialContext(
                            frame_id="study",
                            anchor=SpatialAnchor.OBSERVER,
                            x=1.0,
                            y=2.0,
                        ),
                        confidence=0.8,
                    ),
                )
                for _value in inputs
            )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=MisplacedPoseFormer(),
        settings=MemorySettings(minimum_relevance=0),
        tracer=provider.get_tracer("test"),
    ) as memory:
        source = memory.add(
            "I poured tea",
            context=ObservationContext(
                spatial=SpatialContext(
                    frame_id="kitchen", anchor=SpatialAnchor.OBSERVER, x=0.0, y=0.0
                )
            ),
        )

        assert memory.get(source.id).content == "I poured tea"
        assert _formed(memory, source.id) == []
        assert memory.pending_captures(memory_ids=(source.id,)) == ()

    assert [
        span.attributes[FORMATION_PROPOSALS_REFUSED]
        for span in exporter.get_finished_spans()
        if span.attributes is not None and FORMATION_PROPOSALS_REFUSED in span.attributes
    ] == [1]


class ClaimBackend:
    formation_capabilities = ATOMIC_MODALITIES
    formation_model = consolidation_model = "authority-fixture"
    formation_space = consolidation_recipe = "authority-fixture:v1"

    def __init__(self, kind: RecordKind, basis: EvidenceBasis) -> None:
        self.kind = kind
        self.basis = basis
        self.value = "patient"

    def proposal(self) -> FormationProposal:
        return FormationProposal(
            kind=self.kind,
            basis=self.basis,
            content=f"Rava is {self.value}.",
            subject="Rava",
            predicate="disposition",
            value=self.value,
            confidence=0.9,
        )

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple((self.proposal(),) for _source in inputs)

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
        query_failure: QueryFailure | None = None,
    ) -> tuple[MemoryOperation, ...]:
        return (
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=tuple(source.id for source in evidence),
                proposal=self.proposal(),
            ),
        )

    def close(self) -> None:
        pass


@pytest.mark.parametrize("kind", (RecordKind.STATE, RecordKind.TRAIT))
@pytest.mark.parametrize("basis", (EvidenceBasis.USER_STATEMENT, EvidenceBasis.RESPONSE_FEEDBACK))
def test_custom_backend_cannot_grant_authority_to_automatic_subclaims(
    tmp_path: Path, kind: RecordKind, basis: EvidenceBasis
) -> None:
    backend = ClaimBackend(kind, basis)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=backend,
        consolidator=backend,
    ) as memory:
        source = memory.add(
            "Lea reports that Rava is not patient.",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )
        report = memory.consolidate(evidence_ids=(source.id,))
        assert report.operations == ()
        assert len(report.rejected) == 1
        assert [row.id for row in memory.list().items] == [source.id]
        assert source.context is not None
        assert source.context.basis is EvidenceBasis.USER_STATEMENT


@pytest.mark.parametrize("kind", (RecordKind.STATE, RecordKind.TRAIT))
def test_host_confirms_typed_claim_without_delegating_authority_to_later_inference(
    tmp_path: Path, kind: RecordKind
) -> None:
    backend = ClaimBackend(kind, EvidenceBasis.MODEL_INFERENCE)
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=backend,
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        source = memory.add(
            "Rava has been patient for years.",
            context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
        )
        (inferred,) = [
            row for row in memory.list().items if row.context and row.context.kind is kind
        ]
        assert inferred.context is not None
        assert inferred.context.basis is EvidenceBasis.MODEL_INFERENCE
        assert inferred.context.visible is (kind is RecordKind.STATE)

        confirmation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=kind,
                    basis=EvidenceBasis.USER_STATEMENT,
                    content="Rava is patient.",
                    subject="Rava",
                    predicate="disposition",
                    value="patient",
                    confidence=0.9,
                ),
            )
        )
        (confirmed_id,) = confirmation.created_ids
        confirmed = memory.get(confirmed_id)
        assert confirmed.context is not None
        assert confirmed.context.basis is EvidenceBasis.USER_STATEMENT
        assert confirmed.context.visible

        backend.value = "impatient"
        memory.add("An observer reports that Rava is impatient.")
        standing = memory.get(confirmed_id)
        assert standing.context is not None
        assert standing.context.visible and standing.context.retired_at is None
        assert standing.context.valid_until is None

        assert memory.rollback(confirmation.operation_id)
        with pytest.raises(MemoryNotFoundError):
            memory.get(confirmed_id)
        assert confirmed_id not in {hit.id for hit in memory.search("Rava").hits}


class RestatingFormer:
    formation_capabilities = TinyEmbedder.embedding_capabilities
    formation_model = "self-evidence-test"
    formation_space = "self-evidence-test:v1"

    def __init__(self, *, valid_restatement: bool) -> None:
        self.valid_restatement = valid_restatement
        self.calls = 0

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        assert len(inputs) == 1
        source = inputs[0]
        self.calls += 1
        atlas = FormationProposal(
            kind=RecordKind.RELATION,
            content="Riley owns an atlas.",
            subject="Riley",
            predicate="owns",
            value="atlas",
            confidence=0.8,
        )
        if self.calls == 1:
            return ((atlas,),)
        previous = next(
            record
            for record in source.history
            if record.context is not None and record.context.kind is RecordKind.RELATION
        )
        # This declaration is conjunctive; discarding only the self member would falsely
        # authorize a different, independent assessment by the new raw source.
        invalid = replace(atlas, evidence_ids=(source.memory_id, previous.id))
        telescope = replace(atlas, content="Riley owns a telescope.", value="telescope")
        return ((invalid, *((atlas,) if self.valid_restatement else ()), telescope),)

    def close(self) -> None:
        pass


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("valid_restatement", [False, True])
def test_automatic_self_evidence_refuses_whole_proposal_without_losing_valid_facts(
    tmp_path: Path, independent: bool, valid_restatement: bool
) -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    former = RestatingFormer(valid_restatement=valid_restatement)
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    try:
        with Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            former=former,
            settings=MemorySettings(
                formation_history_max_rows=12, independent_evidence=independent
            ),
            tracer=provider.get_tracer("self-evidence-test"),
        ) as memory:
            original = memory.add("Riley owns an atlas.", occurred_at=at)
            atlas = next(
                record
                for record in memory.list().items
                if record.context is not None and record.context.kind is RecordKind.RELATION
            )
            current = memory.add(
                "Riley also owns a telescope and still has the atlas.",
                occurred_at=at + timedelta(days=1),
            )
            assert memory.get(current.id).content == current.content
            updated = memory.get(atlas.id)
            assert updated.context is not None
            expected_sources = {original.id, current.id} if valid_restatement else {original.id}
            assert set(updated.context.evidence_ids) == expected_sources
            telescope = next(
                record
                for record in memory.list().items
                if record.context is not None and record.context.value == "telescope"
            )
            assert telescope.context is not None
            assert telescope.context.evidence_ids == (current.id,)
            with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
                clauses = connection.execute(
                    "SELECT c.member_count,m.source_memory_id FROM memory_evidence_clauses c "
                    "JOIN memory_evidence_clause_members m USING(memory_id,clause_id) "
                    "WHERE c.memory_id=? AND c.retired_at IS NULL",
                    (atlas.id,),
                ).fetchall()
                assert set(clauses) == {(1, source) for source in expected_sources}
                assert connection.execute(
                    "SELECT COUNT(*) FROM memory_evidence WHERE memory_id=source_memory_id"
                ).fetchone() == (0,)
                assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
                assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
            assert memory.delete(original.id)
            if valid_restatement:
                surviving = memory.get(atlas.id).context
                assert surviving is not None and surviving.visible
            else:
                with pytest.raises(MemoryNotFoundError):
                    memory.get(atlas.id)
            surviving_telescope = memory.get(telescope.id).context
            assert surviving_telescope is not None and surviving_telescope.visible
            assert memory.get(current.id).content == current.content
    finally:
        provider.shutdown()
    adds = [span for span in exporter.get_finished_spans() if span.name == "sensemem.add"]
    assert [
        (span.attributes or {}).get("sensemem.formation.refused_proposals") for span in adds
    ] == [0, 1]


def test_inferred_current_preference_is_immediate_and_a_change_preserves_history(
    tmp_path: Path,
) -> None:
    clock = datetime(2026, 3, 1, tzinfo=timezone.utc)

    def respond(request: httpx.Request) -> httpx.Response:
        source = json.loads(json.loads(request.content)["messages"][1]["content"])["observations"][
            0
        ]
        statement = source["content"]
        value = "coffee" if "coffee" in statement else "tea"
        proposal = {
            "kind": "state",
            "content": statement,
            "subject": "Ana",
            "predicate": "current_preferred_drink"
            if value == "coffee"
            else "only_current_preferred_drink",
            "value": value,
            "confidence": 0.9,
            "evidence_observation_ids": ["observation_0"],
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {"observation_id": "observation_0", "proposals": [proposal]}
                                    ]
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
        Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            former=OpenAICompletion(client, generation_model="fixture-snapshot"),
            settings=MemorySettings(minimum_relevance=0),
        ) as memory,
    ):
        for value, when in (("tea", clock), ("coffee", clock + timedelta(days=1))):
            memory.add(
                f"Ana's current preferred drink is {value}.",
                occurred_at=when,
                context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
            )
            traits = [
                hit
                for hit in memory.search("Ana's current preference").hits
                if hit.context and hit.context.kind is RecordKind.STATE
            ]
            assert len(traits) == 1
            assert traits[0].context is not None
            assert traits[0].context.basis is EvidenceBasis.MODEL_INFERENCE
            assert traits[0].context.value == value
        historical = [
            hit
            for hit in memory.search("Ana's preference", scope=RetrievalScope(valid_at=clock)).hits
            if hit.context and hit.context.kind is RecordKind.STATE
        ]
        assert len(historical) == 1
        assert historical[0].context is not None and historical[0].context.value == "tea"
