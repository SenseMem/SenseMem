"""Pure naming-condition roles and path-sensitive observational footprints."""

from collections.abc import Container, Mapping


def is_naming_condition(
    kind: str | None,
    predicate: str | None,
    basis: str | None,
    identity_id: str | None,
    target_kind: str | None,
    target_identity_id: str | None,
    target_basis: str | None,
) -> bool:
    return (
        target_kind != "entity"
        and target_basis == "model_inference"
        and target_identity_id is not None
        and (kind, predicate, basis, identity_id)
        == ("entity", "identity", "user_statement", target_identity_id)
    )


def experience_footprint(
    anchor_id: str,
    nodes: Mapping[str, tuple[tuple[str, ...], str | None]],
    conditions: Container[str],
) -> frozenset[tuple[str, str]]:
    """Walk a complete certificate, stopping only the naming branch, not shared ancestors."""
    footprint: set[tuple[str, str]] = set()
    pending = [anchor_id]
    seen: set[str] = set()
    while pending:
        memory_id = pending.pop()
        if memory_id in seen or memory_id in conditions:
            continue
        seen.add(memory_id)
        sources, capture = nodes[memory_id]
        if capture is not None:
            footprint.add(("capture", capture))
        elif memory_id != anchor_id:
            footprint.add(("node", memory_id))
        pending.extend(sources)
    return frozenset(footprint) if any(kind == "capture" for kind, _ in footprint) else frozenset()
