"""Naming is an AND condition, separate from observations that corroborate a claim."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from fixtures.formation import Faces
from fixtures.formation import register_identity_fixture as _register

from sensemem import (
    Blob,
    ContextBudget,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    IdentityClaim,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemoryRecord,
    MemorySettings,
    ObservationContext,
    RecordKind,
    StorageError,
)
from sensemem.infrastructure.local import LocalStore
from sensemem.infrastructure.local.store._lineage import add_evidence_clause
from sensemem.kernel.corroboration import Assessment, EvidenceNode, independent_support


class TraitReports:
    formation_capabilities = TinyEmbedder.embedding_capabilities
    formation_model = "naming-condition-report-fixture"
    formation_space = "naming-condition-report-fixture:v1"

    def __init__(self) -> None:
        self.naming: str | None = None
        self.extra: tuple[str, ...] = ()

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (
                FormationProposal(
                    kind=RecordKind.TRAIT,
                    subject="Rava",
                    predicate="temperament",
                    value="calm",
                    content="Rava stays calm under pressure.",
                    confidence=0.6,
                    evidence_ids=(source.memory_id, self.naming, *self.extra),
                ),
            )
            if self.naming is not None
            and source.content.text.startswith(("Lea reports", "Noor reports"))
            else ()
            for source in inputs
        )

    def close(self) -> None:
        pass


def _trait(memory: Memory) -> MemoryRecord:
    (record,) = tuple(
        record
        for record in memory.list(limit=50).items
        if record.context is not None and record.context.kind is RecordKind.TRAIT
    )
    return record


def _reports(memory: Memory, *, shared_capture: bool = False) -> tuple[str, str]:
    first, second = (
        memory.add(
            text,
            context=ObservationContext(source_id="shared" if shared_capture else capture),
        ).id
        for capture, text in (
            (
                "lea-long-term-report",
                "Lea reports from years of working with Rava that Rava consistently stays "
                "calm under pressure.",
            ),
            (
                "noor-long-term-report",
                "Noor reports from many years of knowing Rava that Rava has a lasting "
                "tendency to remain calm under pressure.",
            ),
        )
    )
    return first, second


@pytest.mark.parametrize("independent", [False, True])
def test_host_naming_preserves_two_independent_observations_and_complete_certificates(
    tmp_path: Path, independent: bool
) -> None:
    former = TraitReports()
    config = MemorySettings(
        independent_evidence=independent, formation_history_max_rows=32, minimum_relevance=0
    )
    with Memory(
        tmp_path, embedder=TinyEmbedder(), face_analyzer=Faces(), former=former, settings=config
    ) as memory:
        person, naming, naming_operation = _register(memory, "first.png")
        former.naming = naming
        first, second = _reports(memory)
        trait = _trait(memory)
        assert trait.context is not None
        assert trait.context.identity_id == person
        assert set(trait.context.evidence_ids) == {first, second, naming}
        assert trait.context.visible
        assert trait.context.confidence == pytest.approx(0.84)
        assert any(
            trait.id in candidate.memory_ids
            for candidate in memory.consolidation_candidates(limit=30)
        )

        bundle = memory.compile(
            "Rava calm temperament",
            budget=ContextBudget(selected_proofs=True, max_items=30, max_chars=40_000),
        )
        assert trait.id in {hit.id for hit in bundle.hits}
        witnesses = tuple(proof for proof in bundle.proofs if proof.anchor_id == trait.id)
        assert len(witnesses) >= 2
        assert all(naming in {node.memory_id for node in proof.nodes} for proof in witnesses)
        # This method is structural and lacks the bundle's typed naming-condition context.
        assert not witnesses[0].structurally_independent_of(witnesses[1])

        assert memory.rollback(naming_operation)
        withdrawn = memory.get(trait.id).context
        assert withdrawn is not None and not withdrawn.visible
        assert withdrawn.confidence == 0 and withdrawn.evidence_ids == ()
    with Memory(tmp_path, embedder=TinyEmbedder(), settings=config) as reopened:
        withdrawn = reopened.get(trait.id).context
        assert withdrawn is not None and not withdrawn.visible
        assert withdrawn.confidence == 0 and withdrawn.evidence_ids == ()


@pytest.mark.parametrize("independent", [False, True])
def test_host_naming_does_not_split_one_capture_into_independent_observations(
    tmp_path: Path, independent: bool
) -> None:
    former = TraitReports()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=former,
        settings=MemorySettings(independent_evidence=independent, formation_history_max_rows=32),
    ) as memory:
        _person, former.naming, _operation = _register(memory, "first.png")
        _reports(memory, shared_capture=True)
        trait = _trait(memory)
        assert trait.context is not None and not trait.context.visible
        assert trait.context.confidence == pytest.approx(0.6)


@pytest.mark.parametrize("independent", [False, True])
def test_a_name_alone_has_no_trait_observation_or_confidence(
    tmp_path: Path, independent: bool
) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        settings=MemorySettings(independent_evidence=independent),
    ) as memory:
        person, naming, _operation = _register(memory, "first.png")
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(naming,),
                proposal=FormationProposal(
                    kind=RecordKind.TRAIT,
                    subject="Rava",
                    predicate="temperament",
                    value="calm",
                    content="Rava stays calm under pressure.",
                    confidence=0.6,
                ),
            )
        )
        context = memory.get(operation.created_ids[0]).context
        assert context is not None and context.identity_id == person
        assert context.evidence_ids == (naming,)
        assert not context.visible and context.confidence == 0


@pytest.mark.parametrize("kind", [RecordKind.STATE, RecordKind.TRAIT])
@pytest.mark.parametrize("basis", [EvidenceBasis.USER_STATEMENT, EvidenceBasis.RESPONSE_FEEDBACK])
def test_explicit_host_confirmation_retains_its_authority_with_only_naming_evidence(
    tmp_path: Path, kind: RecordKind, basis: EvidenceBasis
) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        settings=MemorySettings(minimum_relevance=0),
    ) as memory:
        _person, naming, _operation = _register(memory, "first.png")
        created = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(naming,),
                proposal=FormationProposal(
                    kind=kind,
                    basis=basis,
                    subject="Rava",
                    predicate="temperament",
                    value="calm",
                    content="Rava stays calm under pressure.",
                    confidence=0.9,
                ),
            )
        ).created_ids[0]
        context = memory.get(created).context
        assert context is not None and context.visible and context.basis is basis
        assert context.confidence == pytest.approx(0.9)
        bundle = memory.compile(
            "Rava calm", budget=ContextBudget(selected_proofs=True, max_chars=40_000)
        )
        assert created in {hit.id for hit in bundle.hits}


@pytest.mark.parametrize("independent", [False, True])
def test_naming_only_state_cannot_retire_a_state_supported_by_an_observation(
    tmp_path: Path, independent: bool
) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        settings=MemorySettings(independent_evidence=independent),
    ) as memory:
        _person, naming, _operation = _register(memory, "first.png")
        source = memory.add("Rava currently prefers tea.")

        def preference(value: str, evidence: tuple[str, ...]) -> str:
            return memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=evidence,
                    proposal=FormationProposal(
                        kind=RecordKind.STATE,
                        subject="Rava",
                        predicate="preferred_drink",
                        value=value,
                        content=f"Rava currently prefers {value}.",
                        confidence=0.9,
                    ),
                )
            ).created_ids[0]

        tea = preference("tea", (source.id, naming))
        coffee = preference("coffee", (naming,))
        unsupported = memory.get(coffee).context
        standing = memory.get(tea).context
        assert unsupported is not None and not unsupported.visible and unsupported.confidence == 0
        assert standing is not None and standing.visible and standing.valid_until is None
        assert standing.retired_at is None


@pytest.mark.parametrize("independent", [False, True])
def test_three_named_reports_preserve_the_default_all_group_noisy_or(
    tmp_path: Path, independent: bool
) -> None:
    former = TraitReports()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=former,
        settings=MemorySettings(independent_evidence=independent, formation_history_max_rows=32),
    ) as memory:
        _person, former.naming, _operation = _register(memory, "first.png")
        _reports(memory)
        memory.add(
            "Lea reports from a separate long-running project that Rava consistently stays calm.",
            context=ObservationContext(source_id="another-project"),
        )
        context = _trait(memory).context
        assert context is not None and context.visible
        assert context.confidence == pytest.approx(0.84 if independent else 0.936)
        bundle = memory.compile(
            "Rava calm temperament",
            budget=ContextBudget(selected_proofs=True, max_items=30, max_chars=40_000),
        )
        trait = _trait(memory)
        assert trait.id in {hit.id for hit in bundle.hits}
        witnesses = tuple(proof for proof in bundle.proofs if proof.anchor_id == trait.id)
        assert len(witnesses) == (2 if independent else 3)
        assert all(former.naming in {node.memory_id for node in proof.nodes} for proof in witnesses)


@pytest.mark.parametrize("independent", [False, True])
def test_a_naming_clause_does_not_change_the_default_treatment_of_other_joint_clauses(
    tmp_path: Path, independent: bool
) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        settings=MemorySettings(independent_evidence=independent),
    ) as memory:
        _person, naming, _operation = _register(memory, "first.png")
        roots = tuple(memory.add(f"Independent source {index}.").id for index in range(4))
        target = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(naming,),
                proposal=FormationProposal(
                    kind=RecordKind.TRAIT,
                    subject="Rava",
                    predicate="temperament",
                    value="calm",
                    content="Rava stays calm under pressure.",
                    confidence=0.6,
                ),
            )
        ).created_ids[0]
        # Exercise the store's existing mixed-clause statistics, without adding an SDK bypass.
        with memory._store._connections.transaction() as connection:
            for members in (roots[:2], roots[2:]):
                add_evidence_clause(
                    connection,
                    target,
                    members,
                    confidence=0.6,
                    recorded_at=datetime.now(timezone.utc),
                )
        context = memory.get(target).context
        assert context is not None and context.visible is independent
        assert context.confidence == pytest.approx(0.84 if independent else 0.6)


@pytest.mark.parametrize("independent", [False, True])
def test_previous_projection_recipe_requires_fresh_replay(
    tmp_path: Path, independent: bool
) -> None:
    config = MemorySettings(independent_evidence=independent)
    with Memory(tmp_path, embedder=TinyEmbedder(), settings=config) as memory:
        memory.add("A retained original source.")
    with LocalStore(tmp_path) as store:
        store.update_metadata(
            {"evidence.projection_recipe": "independent-v1" if independent else "complete-v1"}
        )
    with pytest.raises(StorageError, match=r"evidence\.projection_recipe"):
        Memory(tmp_path, embedder=TinyEmbedder(), settings=config)


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("shared_kind", ["summary", "other_identity"])
def test_shared_nonmatching_conditions_still_prevent_independent_support(
    tmp_path: Path, independent: bool, shared_kind: str
) -> None:
    former = TraitReports()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=former,
        settings=MemorySettings(independent_evidence=independent, formation_history_max_rows=32),
    ) as memory:
        person, former.naming, _operation = _register(memory, "first.png")
        if shared_kind == "other_identity":
            photo = memory.add(Blob(b"second person", "image/png", "second.png"))
            other = memory.perceive(photo.id, kinds=frozenset({"faces"})).faces[0].identity_id
            assert other != person
            memory.register_identity(other, "Lina")
            extra = next(
                record.id
                for record in memory.list(limit=50).items
                if record.context is not None
                and record.context.kind is RecordKind.ENTITY
                and record.context.identity_id == other
            )
        else:
            source = memory.add("A shared account of the delay.")
            extra = memory.apply(
                MemoryOperation(
                    intent=MemoryIntent.CONSOLIDATE,
                    evidence_ids=(source.id,),
                    proposal=FormationProposal(
                        kind=RecordKind.RELATION,
                        subject="delay",
                        predicate="reported",
                        value="shared account",
                        content="A shared account of the delay.",
                        confidence=0.6,
                    ),
                )
            ).created_ids[0]
        former.extra = (extra,)
        _reports(memory)
        trait = _trait(memory)
        assert trait.context is not None and trait.context.identity_id == person
        assert not trait.context.visible and trait.context.confidence == pytest.approx(0.6)


@pytest.mark.parametrize("independent", [False, True])
def test_a_model_inferred_name_remains_a_shared_inference_premise(
    tmp_path: Path, independent: bool
) -> None:
    former = TraitReports()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        face_analyzer=Faces(),
        former=former,
        settings=MemorySettings(independent_evidence=independent, formation_history_max_rows=32),
    ) as memory:
        photos = tuple(
            memory.add(
                Blob(f"photo {index}".encode(), "image/png", "first.png"),
                context=ObservationContext(source_id=f"photo-{index}"),
            )
            for index in range(2)
        )
        people = tuple(
            memory.perceive(photo.id, kinds=frozenset({"faces"})).faces[0].identity_id
            for photo in photos
        )
        assert people[0] == people[1]
        operation = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.IDENTIFY,
                evidence_ids=tuple(photo.id for photo in photos),
                claim=IdentityClaim(identity_id=people[0], name="Rava"),
            )
        )
        former.naming = operation.created_ids[0]
        naming = memory.get(former.naming).context
        assert naming is not None and naming.basis is EvidenceBasis.MODEL_INFERENCE
        assert naming.visible and set(naming.evidence_ids) == {photo.id for photo in photos}
        _reports(memory)
        trait = _trait(memory)
        assert trait.context is not None and trait.context.identity_id == people[0]
        assert not trait.context.visible and trait.context.confidence == pytest.approx(0.6)


def _graph(*, ancestry: bool) -> dict[str, EvidenceNode]:
    """Typed ancestry shape for a host assertion; not presently constructible through IDENTIFY."""
    return {
        "target": EvidenceNode(
            root_group=None,
            kind="trait",
            basis="model_inference",
            identity_id="person",
            assessments=(Assessment(("a", "name"), 0.6), Assessment(("b", "name"), 0.6)),
        ),
        "name": EvidenceNode(
            root_group=None if ancestry else "name-capture",
            kind="entity",
            predicate="identity",
            basis="user_statement",
            identity_id="person",
            assessments=(Assessment(("c",), 1.0),) if ancestry else (),
        ),
        "a": EvidenceNode(root_group="a-capture"),
        "b": EvidenceNode(root_group="b-capture"),
        "c": EvidenceNode(root_group="c-capture"),
    }


@pytest.mark.parametrize("ancestry", [False, True])
def test_host_naming_ancestry_is_a_complete_condition_not_an_observation(ancestry: bool) -> None:
    support = independent_support(_graph(ancestry=ancestry), "target")
    assert support.count == 2 and support.confidence == pytest.approx(0.84)
    assert not support.truncated


@pytest.mark.parametrize("via_summaries", [False, True])
def test_a_naming_ancestor_still_counts_when_shared_through_an_observation_path(
    via_summaries: bool,
) -> None:
    nodes = _graph(ancestry=True)
    if via_summaries:
        nodes["left"] = EvidenceNode(None, assessments=(Assessment(("a", "c"), 0.6),))
        nodes["right"] = EvidenceNode(None, assessments=(Assessment(("b", "c"), 0.6),))
        clauses = (Assessment(("left", "name"), 0.6), Assessment(("right", "name"), 0.6))
    else:
        clauses = (Assessment(("a", "c", "name"), 0.6), Assessment(("b", "c", "name"), 0.6))
    nodes["target"] = EvidenceNode(
        None, kind="trait", basis="model_inference", identity_id="person", assessments=clauses
    )
    support = independent_support(nodes, "target")
    assert support.count == 1 and support.confidence == pytest.approx(0.6)
    assert not support.truncated


@pytest.mark.parametrize("ancestry", [False, True])
def test_only_a_naming_condition_cannot_supply_experience(ancestry: bool) -> None:
    nodes = _graph(ancestry=ancestry)
    nodes["target"] = EvidenceNode(
        None,
        kind="trait",
        basis="model_inference",
        identity_id="person",
        assessments=(Assessment(("name",), 0.6),),
    )
    support = independent_support(nodes, "target")
    assert support.count == 0 and support.confidence == 0


@pytest.mark.parametrize("variant", ["other_identity", "model_name", "entity_target"])
@pytest.mark.parametrize("ancestry", [False, True])
def test_condition_role_requires_host_naming_for_the_same_non_entity_target(
    variant: str, ancestry: bool
) -> None:
    nodes = _graph(ancestry=ancestry)
    if variant == "other_identity":
        nodes["name"] = replace(nodes["name"], identity_id="another-person")
    elif variant == "model_name":
        nodes["name"] = replace(nodes["name"], basis="model_inference")
    else:
        nodes["target"] = replace(nodes["target"], kind="entity")
    support = independent_support(nodes, "target")
    assert support.count == 1 and support.confidence == pytest.approx(0.6)


@pytest.mark.parametrize("ancestry", [False, True])
def test_one_capture_stays_one_observation_even_with_host_naming(ancestry: bool) -> None:
    nodes = _graph(ancestry=ancestry)
    nodes["a"] = EvidenceNode(root_group="same-capture")
    nodes["b"] = EvidenceNode(root_group="same-capture")
    support = independent_support(nodes, "target")
    assert support.count == 1 and support.confidence == pytest.approx(0.6)


@pytest.mark.parametrize("failure", ["missing_ancestor", "cyclic_ancestor"])
def test_naming_conditions_do_not_bypass_complete_proof_requirements(failure: str) -> None:
    nodes = _graph(ancestry=True)
    if failure == "missing_ancestor":
        nodes.pop("c")
    else:
        nodes["name"] = EvidenceNode(
            None,
            kind="entity",
            predicate="identity",
            basis="user_statement",
            identity_id="person",
            assessments=(Assessment(("target",), 1.0),),
        )
    support = independent_support(nodes, "target")
    assert support.count == 0 and support.confidence == 0
