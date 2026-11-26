"""Delivery certificates preserve provenance, corroboration and actual budget costs."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fixtures.control import add_root_memory as _root
from fixtures.control import derive_memory as _derive
from fixtures.control import open_independent_memory as _open

from sensemem import (
    ContextBudget,
    ContextProof,
    ContextProofNode,
    EvidenceBasis,
    ObservationContext,
    ValidationError,
)
from sensemem.api.content import ContextBudgetInput, context_budget
from sensemem.api.responses import ContextBundleResponse
from sensemem.kernel.corroboration import (
    Assessment,
    EvidenceNode,
    SupportSummary,
    independent_support,
)
from sensemem.kernel.proofs import certificates, proof_support, select_certificates


def test_default_three_group_score_uses_only_the_delivered_assessments() -> None:
    nodes = {key: EvidenceNode(key, source_group_id=key) for key in ("a", "b", "c")}
    nodes["target"] = EvidenceNode(
        None,
        assessments=tuple(Assessment((key,), 0.6) for key in nodes),
        source_group_id="target",
    )
    proofs, clipped = certificates(nodes, "target")
    assert not clipped
    assert proof_support(proofs, scoring="capture_groups") == pytest.approx((3, 0.936))
    assert proof_support(proofs, scoring="independent_pair") == pytest.approx((2, 0.84))
    selected = select_certificates(
        proofs,
        SupportSummary(2, 0.936, False),
        dict.fromkeys(nodes, 1),
        scoring="capture_groups",
    )
    assert len(selected) == 3
    assert proof_support(selected[:-1], scoring="capture_groups")[1] == pytest.approx(0.84)


def test_default_group_score_keeps_assessment_confidence_and_joint_alternatives() -> None:
    nodes = {
        "a": EvidenceNode("shared", confidence=0.4, source_group_id="shared"),
        "b": EvidenceNode("shared", confidence=0.4, source_group_id="shared"),
        "c": EvidenceNode("other", source_group_id="other"),
        "target": EvidenceNode(
            None,
            assessments=(
                Assessment(("a",), 0.6),
                Assessment(("b",), 0.6),
                Assessment(("a", "c"), 0.9),
            ),
            source_group_id="target",
        ),
    }
    proofs, _ = certificates(nodes, "target")
    assert {proof.confidence for proof in proofs} == {0.4}
    assert proof_support(proofs, scoring="capture_groups") == pytest.approx((1, 0.9))
    assert proof_support(proofs, scoring="independent_pair") == pytest.approx((1, 0.4))


@pytest.mark.parametrize("direct", [False, True])
def test_default_removes_only_a_top_level_naming_condition(direct: bool) -> None:
    nodes = {
        "name": EvidenceNode("registration", source_group_id="registration"),
        "summary": EvidenceNode(
            None,
            assessments=(Assessment(("name",), 0.8),),
            source_group_id="registration",
        ),
        "target": EvidenceNode(
            None,
            assessments=(Assessment(("name" if direct else "summary",), 0.6),),
            source_group_id="registration",
        ),
    }
    proofs, _ = certificates(nodes, "target")
    conditions = frozenset({"name"})
    assert proof_support(proofs, conditions, scoring="capture_groups") == pytest.approx(
        (0, 0.0) if direct else (1, 0.6)
    )
    assert proof_support(proofs, conditions, scoring="independent_pair") == (0, 0.0)


@pytest.mark.parametrize("independent", [False, True])
def test_sdk_three_default_groups_can_deliver_their_projected_confidence(
    tmp_path: Path, independent: bool
) -> None:
    with _open(tmp_path, independent) as memory:
        roots = tuple(_root(memory, f"Independent long-term report {index}") for index in range(3))
        target = _derive(memory, (roots[0],), "calm", trait=True).created_ids[0]
        latest = None
        for root in roots[1:]:
            latest = _derive(memory, (root,), "calm", trait=True)
        context = memory.get(target).context
        assert context is not None and context.visible
        assert context.confidence == pytest.approx(0.84 if independent else 0.936)
        bundle = memory.compile(
            "calm", budget=ContextBudget(max_items=20, max_chars=40_000, selected_proofs=True)
        )
        assert target in {hit.id for hit in bundle.hits}
        assert bundle.proof_scoring == ("independent_pair" if independent else "capture_groups")
        witnesses = tuple(proof for proof in bundle.proofs if proof.anchor_id == target)
        assert len(witnesses) == (2 if independent else 3)
        capture = next(node for node in witnesses[0].nodes if node.capture_id is not None)
        with pytest.raises(ValidationError, match="source_group_id must match"):
            replace(capture, source_group_id="different-capture")
        assert proof_support(witnesses, scoring=bundle.proof_scoring)[1] == pytest.approx(
            context.confidence
        )
        if not independent:
            with pytest.raises(ValidationError, match="confidence"):
                replace(
                    bundle, proofs=tuple(proof for proof in bundle.proofs if proof != witnesses[0])
                )
            with pytest.raises(ValidationError, match="confidence"):
                replace(bundle, proof_scoring="independent_pair")
            with pytest.raises(ValidationError, match="source_group_id"):
                replace(
                    bundle,
                    proofs=(
                        replace(
                            witnesses[0],
                            nodes=tuple(
                                replace(node, source_group_id=None)
                                if node.memory_id == target
                                else node
                                for node in witnesses[0].nodes
                            ),
                        ),
                        *witnesses[1:],
                    ),
                )
            with pytest.raises(ValidationError, match="consistent source_group_id"):
                replace(
                    bundle,
                    proofs=(
                        replace(
                            witnesses[0],
                            nodes=tuple(
                                replace(node, source_group_id="conflicting-group")
                                if node.memory_id == target
                                else node
                                for node in witnesses[0].nodes
                            ),
                        ),
                        *witnesses[1:],
                    ),
                )
            too_small = memory.compile(
                "calm", budget=ContextBudget(max_items=3, max_chars=40_000, selected_proofs=True)
            )
            assert target not in {hit.id for hit in too_small.hits}
            assert too_small.unknowns
        assert latest is not None and memory.rollback(latest.operation_id)
        restored = memory.compile(
            "calm", budget=ContextBudget(max_items=20, max_chars=40_000, selected_proofs=True)
        )
        standing = next(hit for hit in restored.hits if hit.id == target)
        assert standing.context is not None
        assert standing.context.confidence == pytest.approx(0.84)
    with _open(tmp_path, independent) as reopened:
        restored = reopened.compile(
            "calm", budget=ContextBudget(max_items=20, max_chars=40_000, selected_proofs=True)
        )
        assert target in {hit.id for hit in restored.hits}


def test_sdk_default_preserves_low_confidence_raw_capture_and_derived_or_group(
    tmp_path: Path,
) -> None:
    with _open(tmp_path, independent=False) as memory:
        roots = tuple(
            memory.add(
                f"Independent report {index}",
                context=ObservationContext(basis=EvidenceBasis.OBSERVATION, confidence=0.4),
            ).id
            for index in range(3)
        )
        summary = _derive(memory, (roots[0],), "combined report").created_ids[0]
        _derive(memory, (roots[1],), "combined report")
        target = _derive(memory, (summary,), "calm", trait=True).created_ids[0]
        _derive(memory, (roots[2],), "calm", trait=True)
        bundle = memory.compile(
            "calm", budget=ContextBudget(max_items=20, max_chars=40_000, selected_proofs=True)
        )
        assert target in {hit.id for hit in bundle.hits}
        assert bundle.proof_scoring == "capture_groups"
        delivered = next(hit for hit in bundle.hits if hit.id == target)
        assert delivered.context is not None and delivered.context.confidence == pytest.approx(0.84)
        assert {
            node.source_group_id
            for proof in bundle.proofs
            for node in proof.nodes
            if node.memory_id == summary
        } == {summary}
        assert max(proof.confidence for proof in bundle.proofs if proof.anchor_id == target) == 0.4


def test_count_and_confidence_can_require_three_separate_witnesses() -> None:
    nodes = {
        "a": EvidenceNode("capture-a"),
        "b": EvidenceNode("capture-b"),
        "target": EvidenceNode(
            None,
            assessments=(
                Assessment(("a",), 0.2),
                Assessment(("b",), 0.2),
                Assessment(("a", "b"), 0.99),
            ),
        ),
    }
    proofs, truncated = certificates(nodes, "target")
    required = independent_support(nodes, "target")
    bought = select_certificates(proofs, required, dict.fromkeys(nodes, 1))
    assert not truncated
    assert len(bought) == 3
    assert proof_support(bought) == (2, 0.99)
    assert {step.memory_id for proof in bought for step in proof.nodes} == set(nodes)


@pytest.mark.parametrize("steps", range(1, 14))
def test_work_exhaustion_cannot_certify_an_unfinished_and(steps: int) -> None:
    nodes = {
        "a": EvidenceNode("shared"),
        "b": EvidenceNode("shared"),
        "target": EvidenceNode(None, assessments=(Assessment(("a", "b"), 0.7),)),
    }
    proofs, _ = certificates(nodes, "target", max_steps=steps)
    for proof in proofs:
        assert {step.memory_id for step in proof.nodes} == set(nodes)
    count, confidence = proof_support(proofs)
    assert count <= 1 and confidence <= 0.7


def test_cycles_and_missing_sources_never_produce_certificates() -> None:
    nodes = {
        "a": EvidenceNode(None, assessments=(Assessment(("b",), 1.0),)),
        "b": EvidenceNode(
            None, assessments=(Assessment(("a",), 1.0), Assessment(("missing",), 1.0))
        ),
    }
    assert certificates(nodes, "a")[0] == ()


def test_repeated_capture_cannot_satisfy_two_witness_obligation() -> None:
    nodes = {
        "a": EvidenceNode("shared"),
        "b": EvidenceNode("shared"),
        "target": EvidenceNode(
            None, assessments=(Assessment(("a",), 0.6), Assessment(("b",), 0.6))
        ),
    }
    proofs, _ = certificates(nodes, "target")
    assert select_certificates(proofs, SupportSummary(2, 0.6, False), dict.fromkeys(nodes, 1)) == ()


@pytest.mark.parametrize("sources", [("missing",), ("target",)])
def test_public_proof_rejects_incomplete_or_cyclic_certificate(sources: tuple[str, ...]) -> None:
    with pytest.raises(ValidationError):
        ContextProof(
            anchor_id="target",
            nodes=(ContextProofNode(memory_id="target", sources=sources, confidence=0.6),),
        )


def test_sdk_delivers_certificates_without_rewriting_the_audit_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _open(tmp_path) as memory:
        roots = tuple(_root(memory, f"evidence {index}") for index in range(8))
        target = _derive(memory, (roots[0],), "patient", trait=True).created_ids[0]
        for root in roots[1:]:
            _derive(memory, (root,), "patient", trait=True)
        stored = memory._store.records.read_memory(target)
        assert stored is not None and stored.context is not None
        audit_ids = stored.context.evidence_ids
        monkeypatch.setattr(
            memory._retrieval,
            "search_prepared",
            lambda *args, **kwargs: SimpleNamespace(
                hits=(memory._hydrator.search_hit(stored, 0.9),), matched_dense_index_ids={}
            ),
        )
        full = memory.compile("patient", budget=ContextBudget(max_items=3))
        selected = memory.compile(
            "patient", budget=ContextBudget(max_items=3, selected_proofs=True)
        )
        assert not full.hits
        assert len(selected.hits) == 3
        assert {hit.id for hit in selected.hits} >= {target}
        delivered = next(hit for hit in selected.hits if hit.id == target)
        assert delivered.context is not None and delivered.context.evidence_ids == audit_ids
        assert memory.get(target).context == stored.context
        assert proof_support(
            tuple(proof for proof in selected.proofs if proof.anchor_id == target)
        ) == pytest.approx((2, 0.84))
        # The fixed header reserves the maximum digit width, as in ordinary compilation.
        assert 0 <= selected.chars - len(selected.render().split("\n\n## Unknowns")[0]) <= 4
        assert selected.chars <= selected.budget.max_chars
        assert "## Selected proofs" in selected.compact().text
        assert target not in selected.compact().text
        for model in (ContextBundleResponse,):
            wire = model.model_validate(selected.document()).model_dump(mode="json")
            assert wire["budget"]["selected_proofs"] is True
            assert wire["proofs"][0]["anchor_id"] == target
        too_small = memory.compile(
            "patient",
            budget=ContextBudget(
                max_items=3,
                max_chars=selected.chars - 5,
                selected_proofs=True,
            ),
        )
        assert not too_small.hits
        with pytest.raises(ValidationError, match="confidence"):
            replace(selected, proofs=selected.proofs[:1])
        with pytest.raises(ValidationError, match="full hit"):
            replace(selected, facts=())


def test_budget_input_preserves_opt_in_and_rejects_non_boolean() -> None:
    budget = context_budget(ContextBudgetInput(selected_proofs=True))
    assert budget is not None and budget.selected_proofs
    with pytest.raises(ValidationError):
        ContextBudget(selected_proofs=1)  # type: ignore[arg-type]
