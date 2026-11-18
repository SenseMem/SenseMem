"""Naming premises bind people and withdraw claims without becoming independent reports."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    AssetRef,
    Blob,
    ContentPart,
    EvidenceBasis,
    FaceAnalysis,
    FaceEmbedding,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    Modality,
    ModelInput,
    ObservationContext,
    RecordKind,
    RecordSemantics,
    ValidationError,
)
from sensemem.kernel.formation_identity import claim_identity, history_identity_refusal

NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


class Faces:
    face_capabilities = frozenset({Modality.IMAGE})
    face_model = "naming-premise-faces"
    face_space = "naming-premise-faces:2"
    face_analysis_space = "naming-premise-detector:v1"

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[FaceAnalysis, ...]:
        return tuple(
            FaceAnalysis(
                (
                    FaceEmbedding(
                        "face",
                        (1.0, 0.0) if asset.name == "first.png" else (0.0, 1.0),
                        (0.1, 0.1, 0.5, 0.5),
                    ),
                )
            )
            for asset in assets
        )

    def close(self) -> None:
        pass


class Former:
    formation_capabilities = TinyEmbedder.embedding_capabilities
    formation_model = "naming-premise-former"
    formation_space = "naming-premise-former:v1"

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.kind = RecordKind.TRAIT

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        result: list[tuple[FormationProposal, ...]] = []
        for source in inputs:
            if source.content.text == "A person is introduced as Rava.":
                result.append(
                    (
                        FormationProposal(
                            kind=RecordKind.ENTITY, subject="Rava", content="The person is Rava."
                        ),
                    )
                )
                continue
            if not source.content.text.startswith("Rava "):
                result.append(())
                continue
            name = self.names.get(source.context.source_id or "")
            if name is not None:
                assert any(record.id == name for record in source.history)
            result.append(
                (
                    FormationProposal(
                        kind=self.kind,
                        subject="Rava",
                        predicate="temperament",
                        value="calm",
                        content="Rava stays calm under pressure.",
                        confidence=0.6,
                        evidence_ids=(source.memory_id,)
                        if name is None
                        else (source.memory_id, name),
                    ),
                )
            )
        return tuple(result)

    def close(self) -> None:
        pass


def _register(memory: Memory, filename: str) -> tuple[str, str, int]:
    photo = memory.add(Blob(filename.encode(), "image/png", filename))
    person = memory.perceive(photo.id, kinds=frozenset({"faces"})).faces[0].identity_id
    memory.register_identity(person, "Rava")
    naming = next(
        operation
        for operation in memory.operations()
        if operation.operation.intent is MemoryIntent.IDENTIFY
        and operation.operation.claim is not None
        and operation.operation.claim.identity_id == person
    )
    return person, naming.created_ids[0], naming.operation_id


def _traits(memory: Memory) -> tuple[MemoryRecord, ...]:
    return tuple(
        record
        for record in memory.list(limit=50).items
        if record.context is not None and record.context.kind is RecordKind.TRAIT
    )


def test_a_global_name_cannot_create_an_uncited_identity_dependency(tmp_path: Path) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=Former(),
        settings=MemorySettings(formation_history_max_rows=0, independent_evidence=True),
    ) as memory:
        person, _name, operation = _register(memory, "first.png")
        first = memory.add(
            "Rava stayed calm during one delay.", context=ObservationContext(source_id="first")
        )
        second = memory.add(
            "Rava remained calm during another delay.",
            context=ObservationContext(source_id="second"),
        )
        (trait,) = _traits(memory)
        assert trait.context is not None and trait.context.identity_id is None
        assert trait.context.visible and trait.context.confidence == pytest.approx(0.84)
        assert set(trait.context.evidence_ids) == {first.id, second.id}
        assert memory.rollback(operation)
        profile = memory.identity(person)
        assert profile is not None and profile.name is None
        assert memory.get(trait.id).context == trait.context


def test_original_named_reports_keep_unbound_and_same_name_identity_claims_separate(
    tmp_path: Path,
) -> None:
    former = Former()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=former,
        settings=MemorySettings(independent_evidence=True),
    ) as memory:
        first_person, first_name, _first_operation = _register(memory, "first.png")
        second_person, second_name, _second_operation = _register(memory, "second.png")
        assert first_person != second_person
        source = memory.add(
            "Rava stayed calm during one delay.",
            context=ObservationContext(source_id="explicit-original-name"),
        )
        (unbound,) = _traits(memory)
        assert unbound.context is not None
        assert unbound.context.identity_id is None
        assert not unbound.context.visible
        assert unbound.context.evidence_ids == (source.id,)

        former.names = {"first-person": first_name, "second-person": second_name}
        first = memory.add(
            "Rava stayed calm during another delay.",
            context=ObservationContext(source_id="first-person"),
        )
        second = memory.add(
            "Rava stayed calm during a later delay.",
            context=ObservationContext(source_id="second-person"),
        )
        claims = _traits(memory)
        assert len(claims) == 3 and len({claim.id for claim in claims}) == 3
        assert len({claim.context.lineage_id for claim in claims if claim.context is not None}) == 3
        by_person = {
            claim.context.identity_id: claim for claim in claims if claim.context is not None
        }
        assert set(by_person) == {None, first_person, second_person}
        assert by_person[None].context == unbound.context
        for person, current, naming in (
            (first_person, first, first_name),
            (second_person, second, second_name),
        ):
            context = by_person[person].context
            assert context is not None and not context.visible
            assert set(context.evidence_ids) == {current.id, naming}
        first_profile = memory.identity(first_person)
        second_profile = memory.identity(second_person)
        assert first_profile is not None and first_profile.name == "Rava"
        assert second_profile is not None and second_profile.name == "Rava"


def test_same_display_name_does_not_combine_different_people_into_trait_support(
    tmp_path: Path,
) -> None:
    former = Former()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=former,
        settings=MemorySettings(formation_history_max_rows=32, independent_evidence=True),
    ) as memory:
        first_person, first_name, _first_operation = _register(memory, "first.png")
        second_person, second_name, _second_operation = _register(memory, "second.png")
        assert first_person != second_person
        former.names = {"first": first_name, "second": second_name}
        first = memory.add(
            "Rava stayed calm during one delay.", context=ObservationContext(source_id="first")
        )
        second = memory.add(
            "Rava remained calm during another delay.",
            context=ObservationContext(source_id="second"),
        )
        traits = _traits(memory)
        assert len(traits) == 2
        by_person = {
            record.context.identity_id: record.context
            for record in traits
            if record.context is not None
        }
        assert set(by_person) == {first_person, second_person}
        for person, source, name in (
            (first_person, first, first_name),
            (second_person, second, second_name),
        ):
            assert not by_person[person].visible
            assert by_person[person].confidence == pytest.approx(0.6)
            assert set(by_person[person].evidence_ids) == {source.id, name}


@pytest.mark.parametrize("with_name", [False, True])
def test_consolidation_binds_only_its_cited_naming_premise(tmp_path: Path, with_name: bool) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), face_analyzer=Faces()) as memory:
        person, name, _operation = _register(memory, "first.png")
        if with_name:
            _register(memory, "second.png")
        source = memory.add("Rava repaired a motor.")
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id, name) if with_name else (source.id,),
                proposal=FormationProposal(
                    kind=RecordKind.RELATION,
                    subject="Rava",
                    predicate="repaired",
                    value="motor",
                    content="Rava repaired a motor.",
                    confidence=0.8,
                ),
            )
        )
        context = memory.get(operation.created_ids[0]).context
        assert context is not None
        assert context.identity_id == (person if with_name else None)


def test_consolidation_rejects_competing_cited_people_instead_of_creating_an_unbound_claim(
    tmp_path: Path,
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), face_analyzer=Faces()) as memory:
        _first_person, first_name, _first_operation = _register(memory, "first.png")
        _second_person, second_name, _second_operation = _register(memory, "second.png")
        source = memory.add("Rava repaired a motor.")
        with pytest.raises(ValidationError):
            memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(source.id, first_name, second_name),
                    proposal=FormationProposal(
                        kind=RecordKind.RELATION,
                        subject="Rava",
                        predicate="repaired",
                        value="motor",
                        content="Rava repaired a motor.",
                    ),
                )
            )


@pytest.mark.parametrize("cite_name", [False, True])
def test_unbound_name_history_preserves_declared_premises_and_standalone_reports(
    tmp_path: Path, cite_name: bool
) -> None:
    former = Former()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=former,
        settings=MemorySettings(formation_history_max_rows=16, independent_evidence=True),
    ) as memory:
        memory.add("A person is introduced as Rava.")
        name = next(
            record
            for record in memory.list(limit=30).items
            if record.context is not None and record.context.kind is RecordKind.ENTITY
        )
        if cite_name:
            former.names = {"first": name.id, "second": name.id}
        first = memory.add(
            "Rava stayed calm during one delay.", context=ObservationContext(source_id="first")
        )
        second = memory.add(
            "Rava remained calm during another delay.",
            context=ObservationContext(source_id="second"),
        )
        (trait,) = _traits(memory)
        assert trait.context is not None and trait.context.identity_id is None
        assert trait.context.visible is not cite_name
        assert trait.context.confidence == pytest.approx(0.6 if cite_name else 0.84)
        assert set(trait.context.evidence_ids) == (
            {first.id, second.id, name.id} if cite_name else {first.id, second.id}
        )
        correction = memory.apply(
            MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=(name.id,))
        )
        withdrawn = memory.get(trait.id).context
        assert withdrawn is not None
        assert withdrawn.confidence == pytest.approx(0 if cite_name else 0.84)
        assert withdrawn.visible is not cite_name
        assert memory.rollback(correction.operation_id)
        restored = memory.get(trait.id).context
        assert restored is not None
        assert restored.confidence == trait.context.confidence
        assert restored.visible == trait.context.visible
        assert restored.identity_id == trait.context.identity_id
        assert set(restored.evidence_ids) == set(trait.context.evidence_ids)


@pytest.mark.parametrize("independent", [False, True])
def test_a_cited_stable_name_is_a_withdrawable_identity_premise(
    tmp_path: Path, independent: bool
) -> None:
    former = Former()
    former.kind = RecordKind.RELATION
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=former,
        settings=MemorySettings(formation_history_max_rows=16, independent_evidence=independent),
    ) as memory:
        person, name, naming_operation = _register(memory, "first.png")
        former.names["first"] = name
        source = memory.add(
            "Rava stayed calm during one delay.", context=ObservationContext(source_id="first")
        )
        (claim,) = tuple(
            record
            for record in memory.list(limit=30).items
            if record.context is not None and record.context.kind is RecordKind.RELATION
        )
        assert claim.context is not None
        assert claim.context.identity_id == person and claim.context.visible
        assert set(claim.context.evidence_ids) == {source.id, name}
        assert memory.rollback(naming_operation)
        withdrawn = memory.get(claim.id).context
        assert withdrawn is not None and not withdrawn.visible
        assert withdrawn.confidence == 0 and withdrawn.evidence_ids == ()
    with Memory(
        tmp_path, embedder=TinyEmbedder(), settings=MemorySettings(independent_evidence=independent)
    ) as reopened:
        withdrawn = reopened.get(claim.id).context
        assert withdrawn is not None and withdrawn.identity_id == person
        assert not withdrawn.visible and withdrawn.confidence == 0
        assert withdrawn.evidence_ids == ()
        profile = reopened.identity(person)
        assert profile is not None and profile.name is None


def _name(*, identity: str | None = None) -> MemoryRecord:
    return MemoryRecord(
        "name",
        "The person is Rava.",
        NOW,
        context=RecordSemantics(
            kind=RecordKind.ENTITY,
            basis=EvidenceBasis.MODEL_INFERENCE,
            confidence=0.8,
            valid_from=None,
            valid_until=None,
            recorded_at=NOW,
            subject="Rava",
            identity_id=identity,
        ),
    )


@pytest.mark.parametrize("identity", [None, "person"])
@pytest.mark.parametrize("kind", ["original", "derived", "pronoun", "substring"])
def test_only_original_explicit_subject_text_can_replace_an_unbound_label_reference(
    kind: str, identity: str | None
) -> None:
    proposal = FormationProposal(
        kind=RecordKind.TRAIT,
        subject="Rava",
        predicate="temperament",
        value="calm",
        content="Rava stays calm.",
        evidence_ids=("current",),
    )
    text = (
        "She stays calm."
        if kind == "pronoun"
        else "Bravado stays calm."
        if kind == "substring"
        else "Rava stays calm."
    )
    source = FormationInput(
        "current",
        ModelInput(
            text=text,
            parts=(ContentPart("model_interpretation", text),) if kind == "derived" else (),
        ),
        ObservationContext(),
        history=(_name(identity=identity),),
    )
    assert history_identity_refusal(proposal, source, ()) is (kind != "original")


def test_an_original_explicit_label_does_not_restore_a_stable_or_withdrawn_binding() -> None:
    proposal = FormationProposal(
        kind=RecordKind.TRAIT,
        subject="Rava",
        predicate="temperament",
        value="calm",
        content="Rava stays calm.",
    )
    name = _name(identity="person")
    assert name.context is not None
    for known in (
        name,
        replace(name, context=replace(name.context, retired_at=NOW + timedelta(seconds=1))),
    ):
        source = FormationInput(
            "current", ModelInput(text="Rava stays calm."), ObservationContext(), history=(known,)
        )
        assert not history_identity_refusal(proposal, source, ())
        assert claim_identity(proposal, ()) is None
    source = FormationInput(
        "current", ModelInput(text="Rava stays calm."), ObservationContext(), history=(name,)
    )
    assert not history_identity_refusal(proposal, source, (name,))
    assert claim_identity(proposal, (name,)) == "person"


def test_an_original_explicit_label_cannot_resolve_competing_cited_identities() -> None:
    proposal = FormationProposal(
        kind=RecordKind.TRAIT,
        subject="Rava",
        predicate="temperament",
        value="calm",
        content="Rava stays calm.",
    )
    first = _name(identity="first-person")
    second = replace(_name(identity="second-person"), id="second-name")
    source = FormationInput(
        "current",
        ModelInput(text="Rava stays calm."),
        ObservationContext(),
        history=(first, second),
    )
    assert history_identity_refusal(proposal, source, (first, second))
    with pytest.raises(ValidationError):
        claim_identity(proposal, (first, second))
