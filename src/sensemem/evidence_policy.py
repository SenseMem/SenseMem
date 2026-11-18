"""Pure evidence authority and support requirements shared by every trust boundary."""

import math


def required_support(kind: str, basis: str, identity_id: str | None) -> int:
    """Require corroboration for inferred traits and identity-bound naming assertions."""
    return (
        2
        if basis not in {"user_statement", "response_feedback"}
        and (kind == "trait" or (kind == "entity" and identity_id is not None))
        else 1
    )


def state_can_supersede(
    basis: str,
    confidence: float,
    previous_basis: str,
    previous_confidence: float,
) -> bool:
    """Prefer explicit authority, then equally or better supported assertions in that tier."""
    assertions = {"user_statement", "response_feedback"}
    explicit = basis in assertions
    previous_explicit = previous_basis in assertions
    if explicit != previous_explicit:
        return explicit
    # Noisy-OR can round an otherwise equal support score, e.g. 0.3, by one ULP.
    return confidence >= previous_confidence or math.isclose(
        confidence, previous_confidence, rel_tol=1e-15, abs_tol=0.0
    )
