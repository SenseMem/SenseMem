from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sensemem import EvidenceBasis, FormationProposal, MemoryKind, SpatialAnchor, SpatialContext
from sensemem.kernel.formation import formation_context
from sensemem.kernel.predicates import PREDICATES, PredicateDefinition, resolve_predicate


def _lineage(
    predicate: str,
    *,
    value: str = "tea",
    kind: MemoryKind = MemoryKind.STATE,
    basis: EvidenceBasis = EvidenceBasis.MODEL_INFERENCE,
    identity_id: str | None = None,
    spatial: SpatialContext | None = None,
) -> str | None:
    return formation_context(
        None,
        FormationProposal(
            kind=kind,
            basis=basis,
            content="fixture",
            subject="Ana",
            predicate=predicate,
            value=value,
            spatial=spatial,
        ),
        model_id="predicate-fixture-v1",
        recipe="predicate-fixture-v1",
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        identity_id=identity_id,
    ).lineage_id


def test_registry_aliases_are_unique_and_policy_predicates_are_not_registered() -> None:
    aliases = [alias for definition in PREDICATES for alias in definition.aliases]
    assert len(set(aliases)) == len(aliases)
    assert not {"consent", "name", "relationship"}.intersection(aliases)


@pytest.mark.parametrize(
    "value", ["tea, coffee", '["tea"]', "tea/coffee", "茶、咖啡", "tea\ncoffee"]
)
def test_compound_objects_do_not_acquire_scalar_alias_keys(value: str) -> None:
    assert _lineage("preferred_drink", value=value) != _lineage("favorite_beverage", value=value)


def test_aliases_preserve_bound_identity_and_spatial_scope() -> None:
    at_home = SpatialContext(frame_id="home", anchor=SpatialAnchor.SUBJECT, x=0, y=0, z=0)
    at_office = SpatialContext(frame_id="office", anchor=SpatialAnchor.SUBJECT, x=0, y=0, z=0)
    first = _lineage("preferred_drink", identity_id="person-1", spatial=at_home)
    assert first == _lineage("favorite_beverage", identity_id="person-1", spatial=at_home)
    assert first != _lineage("favorite_beverage", identity_id="person-2", spatial=at_home)
    assert first != _lineage("favorite_beverage", identity_id="person-1", spatial=at_office)


def test_asserted_traits_can_still_override_inferred_traits_in_the_same_slot() -> None:
    assert _lineage("preferred_drink", kind=MemoryKind.TRAIT) == _lineage(
        "favorite_beverage", kind=MemoryKind.TRAIT, basis=EvidenceBasis.USER_STATEMENT
    )


@pytest.mark.parametrize("kind", [MemoryKind.ENTITY, MemoryKind.EVENT, MemoryKind.RESPONSE_POLICY])
def test_aliases_do_not_change_other_semantic_types(kind: MemoryKind) -> None:
    assert _lineage("preferred_drink", kind=kind) != _lineage("favorite_beverage", kind=kind)


@pytest.mark.parametrize("definition", PREDICATES, ids=lambda definition: definition.identifier)
def test_every_declared_alias_resolves_to_one_stable_key(definition: PredicateDefinition) -> None:
    # Definitions are data, while the assertion checks the actual formation key used by writers.
    keys = {_lineage(alias) for alias in definition.aliases}
    assert len(keys) == 1


def test_unknown_predicates_keep_literal_resolution() -> None:
    result = resolve_predicate("  Likes  ", kind="state", basis="model_inference", value="tea")
    assert (result.key, result.match, result.value_type) == ("likes", "literal", None)


def test_durable_predicate_keys_do_not_change_with_registry_maintenance() -> None:
    assert _lineage("favorite_beverage") == (
        "188229f8e2109d6904d9f594989541150b78e9c2945364a38d4e54b1116994b5"
    )
    assert _lineage("unregistered") == (
        "453b5277a2b0628664f7504493f8143dbc67f99b6452f7f1cb7f633844e16f04"
    )
