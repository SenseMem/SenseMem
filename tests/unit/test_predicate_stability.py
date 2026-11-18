from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder

from sensemem import (
    EmbedTask,
    EvidenceBasis,
    FormationInput,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemorySettings,
    Modality,
    ModelInput,
    RecordKind,
)
from sensemem.infrastructure.local import LocalStore
from sensemem.kernel.formation_rules import formation_context, formation_memory_id, naming_proposal


class PredicateFormer:
    formation_capabilities = frozenset({Modality.TEXT})
    formation_model = "predicate-fixture-v1"
    formation_space = "predicate-fixture-v1"

    def form(self, inputs: Sequence[FormationInput]) -> tuple[tuple[FormationProposal, ...], ...]:
        return tuple(
            (FormationProposal(content=value.content.text, **json.loads(value.content.text)),)
            for value in inputs
        )

    def close(self) -> None:
        pass


def _add_claim(memory: Memory, predicate: str, value: str, **fields: object) -> str:
    source = memory.add(
        json.dumps(
            {"kind": "state", "subject": "Ana", "predicate": predicate, "value": value, **fields},
            ensure_ascii=False,
        )
    )
    return next(
        record.id
        for record in memory.list().items
        if record.context is not None and source.id in record.context.evidence_ids
    )


def _confirm_trait(memory: Memory, predicate: str, value: str) -> str:
    content = json.dumps(
        {"kind": "trait", "subject": "Ana", "predicate": predicate, "value": value},
        ensure_ascii=False,
    )
    source = memory.add(content)
    operation = memory.apply(
        MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(source.id,),
            proposal=FormationProposal(
                kind=RecordKind.TRAIT,
                content=content,
                subject="Ana",
                predicate=predicate,
                value=value,
                basis=EvidenceBasis.USER_STATEMENT,
            ),
        )
    )
    (claim_id,) = operation.created_ids
    return claim_id


_FULLWIDTH_ALIAS = " " + "".join(chr(ord(c) + 0xFEE0) for c in "FAVORITE_BEVERAGE") + " "


@pytest.mark.parametrize(
    "first_subject,first_identity,second_subject,second_identity,same_claim",
    [
        ("Ana", "person-1", "Ann", "person-1", True),
        ("Ana", "person-1", "Ana", "person-2", False),
        ("Ana", None, "Ann", None, False),
    ],
)
def test_stable_claim_identity_survives_a_rename_without_merging_distinct_people(
    first_subject: str,
    first_identity: str | None,
    second_subject: str,
    second_identity: str | None,
    same_claim: bool,
) -> None:
    claims = []
    lineages = []
    for index, (subject, identity_id) in enumerate(
        ((first_subject, first_identity), (second_subject, second_identity))
    ):
        proposal = FormationProposal(
            kind=RecordKind.TRAIT,
            content=f"{subject} is patient.",
            subject=subject,
            predicate="disposition",
            value="patient",
        )
        context = formation_context(
            None,
            proposal,
            model_id="identity-fixture",
            recipe="identity-fixture:v1",
            recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            identity_id=identity_id,
        )
        lineages.append(context.lineage_id)
        claims.append(
            formation_memory_id(
                f"source-{index}", proposal, recipe="identity-fixture:v1", context=context
            )
        )
    assert (lineages[0] == lineages[1]) is same_claim
    assert (claims[0] == claims[1]) is same_claim


def test_naming_assertions_keep_distinct_claims_within_one_identity() -> None:
    claims = []
    lineages = []
    for name in ("Ana", "Ann"):
        proposal = naming_proposal(name, None, basis=EvidenceBasis.USER_STATEMENT)
        context = formation_context(
            None,
            proposal,
            model_id=None,
            recipe="identity-fixture:v1",
            recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            identity_id="person-1",
        )
        lineages.append(context.lineage_id)
        claims.append(
            formation_memory_id("person-1", proposal, recipe="identity-fixture:v1", context=context)
        )
    assert lineages[0] == lineages[1]
    assert claims[0] != claims[1]


@pytest.mark.parametrize(
    "alias",
    [
        "favorite_beverage",
        "current_preferred_drink",
        "only_current_preferred_drink",
        "喜欢的饮料",
        _FULLWIDTH_ALIAS,
    ],
)
def test_alias_correction_reaches_the_original_claim(tmp_path: Path, alias: str) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first_id = _add_claim(memory, "preferred_drink", "tea")
        second_id = _add_claim(memory, alias, "coffee")
        first, second = memory.get(first_id).context, memory.get(second_id).context
        assert first is not None and second is not None
        assert first.lineage_id == second.lineage_id
        assert first.valid_until == second.valid_from
        assert second.supersedes_id == first_id
        assert (first.predicate, second.predicate) == ("preferred_drink", alias.strip())


@pytest.mark.parametrize(
    "first_predicate,second_predicate,fields",
    [
        ("home_city", "current_location", {}),
        ("preferred_drink", "likes", {}),
        ("preferred_drink", "favorite_beverage", {"subject": "Mei"}),
        ("preferred_drink", "favorite_beverage", {"kind": "relation"}),
    ],
)
def test_distinct_slots_do_not_merge(
    tmp_path: Path, first_predicate: str, second_predicate: str, fields: dict[str, str]
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first_id = _add_claim(memory, first_predicate, "tea")
        second_id = _add_claim(memory, second_predicate, "coffee", **fields)
        first, second = memory.get(first_id).context, memory.get(second_id).context
        assert first is not None and second is not None
        assert first.lineage_id != second.lineage_id
        assert first.valid_until is None


def test_relation_aliases_keep_multiple_values(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first_id = _add_claim(memory, "preferred_drink", "tea", kind="relation")
        second_id = _add_claim(memory, "favorite_beverage", "coffee", kind="relation")
        first, second = memory.get(first_id).context, memory.get(second_id).context
        assert first is not None and second is not None
        assert first.lineage_id == second.lineage_id
        assert first.kind is second.kind is RecordKind.RELATION
        assert first.retired_at is second.retired_at is None


def _history(path: Path) -> dict[str, list[tuple[object, ...]]]:
    with closing(sqlite3.connect(path / "state.sqlite3")) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in (
                "memory_versions",
                "memory_evidence_clauses",
                "memory_evidence_clause_versions",
                "embeddings",
                "search_index_queue",
                "memory_operations",
            )
        }


def test_current_predicate_provenance_and_history_survive_reopening(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first = _add_claim(memory, "preferred_drink", "tea")
        second = _add_claim(memory, "favorite_beverage", "coffee")
    before = _history(tmp_path)
    with LocalStore(tmp_path):
        pass
    assert _history(tmp_path) == before
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        provenance = connection.execute(
            "SELECT predicate, predicate_key, predicate_registry, predicate_match "
            "FROM memory_semantics WHERE memory_id = ?",
            (second,),
        ).fetchone()
        assert provenance == (
            "favorite_beverage",
            "preferred_drink",
            "sensemem-predicates-v2",
            "alias",
        )
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        original, other = memory.get(first).context, memory.get(second).context
        assert original is not None and other is not None
        assert original.lineage_id == other.lineage_id
        assert original.valid_until == other.valid_from
        third = _add_claim(memory, "喜欢的饮料", "water")
        corrected = memory.get(third).context
        original, other = memory.get(first).context, memory.get(second).context
        assert corrected is not None and original is not None and other is not None
        assert other.valid_until == corrected.valid_from
        assert original.valid_until == other.valid_from


def test_alias_index_rebuild_reuses_authoritative_vectors(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        first = _add_claim(memory, "preferred_drink", "tea", kind="relation")
        second = _add_claim(memory, "favorite_beverage", "coffee", kind="relation")
    shutil.rmtree(tmp_path / "zvec")

    class StoredVectorsOnly(TinyEmbedder):
        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            assert task is EmbedTask.QUERY, "rebuild must reuse authoritative vectors"
            return super().embed(inputs, task=task)

    with Memory(
        tmp_path, embedder=StoredVectorsOnly(), settings=MemorySettings(minimum_relevance=0)
    ) as memory:
        left, right = memory.get(first).context, memory.get(second).context
        assert left is not None and right is not None and left.lineage_id == right.lineage_id
        assert {first, second}.issubset({hit.id for hit in memory.search("drink", limit=20).hits})


def test_user_stated_alias_overrides_an_inferred_trait(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        inferred = _add_claim(memory, "preferred_drink", "tea", kind="trait", confidence=0.8)
        _add_claim(memory, "preferred_drink", "tea", kind="trait", confidence=0.9)
        stated = _confirm_trait(memory, "favorite_beverage", "coffee")
        old, new = memory.get(inferred).context, memory.get(stated).context
        assert old is not None and new is not None
        assert old.lineage_id == new.lineage_id
        assert old.valid_until == new.valid_from


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("kind", [RecordKind.TRAIT, RecordKind.RELATION])
@pytest.mark.parametrize("alias", ["favorite_beverage", "only_current_preferred_drink"])
def test_reviewed_aliases_share_one_claim_and_independent_support(
    tmp_path: Path, independent: bool, kind: RecordKind, alias: str
) -> None:
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        former=PredicateFormer(),
        settings=MemorySettings(independent_evidence=independent),
    ) as memory:
        first = _add_claim(memory, "preferred_drink", "tea", kind=kind.value, confidence=0.6)
        first_context = memory.get(first).context
        assert first_context is not None
        assert first_context.visible is (kind is RecordKind.RELATION)
        second = _add_claim(memory, alias, "tea", kind=kind.value, confidence=0.6)
        assert second == first
        context = memory.get(first).context
        assert context is not None and context.visible
        assert len(context.evidence_ids) == 2
        assert context.confidence == pytest.approx(0.84)
        assert context.predicate == "preferred_drink"


@pytest.mark.parametrize(
    "kind,delete_source", [("state", False), ("state", True), ("trait", False)]
)
def test_removing_a_correction_restores_the_previous_alias_claim(
    tmp_path: Path, kind: str, delete_source: bool
) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), former=PredicateFormer()) as memory:
        add_claim = _confirm_trait if kind == "trait" else _add_claim
        first = add_claim(memory, "preferred_drink", "tea")
        second = add_claim(memory, "favorite_beverage", "coffee")
        correction = add_claim(memory, "喜欢的饮料", "water")
        context = memory.get(correction).context
        assert context is not None
        assert memory.delete(context.evidence_ids[0] if delete_source else correction)
        left, right = memory.get(first).context, memory.get(second).context
        assert left is not None and right is not None
        assert left.valid_until == right.valid_from
        assert right.valid_until is None
