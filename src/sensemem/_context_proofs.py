"""Pure scoring of delivered certificates under the store's evidence policy."""

from __future__ import annotations

from itertools import combinations

from sensemem.context_values import ContextProof, ContextProofScoring


def singleton_group(proof: ContextProof, conditions: frozenset[str]) -> str | None:
    """Return the stored group of one effective assessment member, never a path's leaf."""
    nodes = {node.memory_id: node for node in proof.nodes}
    members = tuple(source for source in nodes[proof.anchor_id].sources if source not in conditions)
    return nodes[members[0]].source_group_id if len(members) == 1 else None


def proof_support(
    proofs: tuple[ContextProof, ...],
    conditions: frozenset[str] = frozenset(),
    *,
    scoring: ContextProofScoring = "independent_pair",
) -> tuple[int, float]:
    """Score only complete delivered witnesses; a shared condition never creates a vote."""
    if len({proof.anchor_id for proof in proofs}) > 1:
        raise ValueError("support witnesses must have the same anchor")
    if scoring == "capture_groups":
        groups: dict[str, float] = {}
        joint = 0.0
        for proof in proofs:
            anchor = next(node for node in proof.nodes if node.memory_id == proof.anchor_id)
            members = tuple(source for source in anchor.sources if source not in conditions)
            group = singleton_group(proof, conditions)
            if group is not None:
                groups[group] = max(groups.get(group, 0.0), anchor.confidence)
            elif len(members) > 1 or anchor.capture_id is not None:
                joint = max(joint, anchor.confidence)
        confidence = 0.0
        for score in groups.values():
            confidence = 1.0 - (1.0 - confidence) * (1.0 - score)
        return max(len(groups), int(joint > 0.0)), max(confidence, joint)
    experiences = {proof: proof._experience_footprint(conditions) for proof in proofs}
    count = int(any(experiences.values()))
    confidence = max((proof.confidence for proof in proofs if experiences[proof]), default=0.0)
    for left, right in combinations(proofs, 2):
        if (
            experiences[left]
            and experiences[right]
            and next(node.sources for node in left.nodes if node.memory_id == left.anchor_id)
            != next(node.sources for node in right.nodes if node.memory_id == right.anchor_id)
            and experiences[left].isdisjoint(experiences[right])
        ):
            count = 2
            confidence = max(confidence, 1.0 - (1.0 - left.confidence) * (1.0 - right.confidence))
    return count, confidence
