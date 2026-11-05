"""The public correction value preserves counterevidence through durable serialization."""

import pytest

from sensemem import MemoryIntent, MemoryOperation, ValidationError
from sensemem.control import dump_operation, load_operation, operation_key


def test_correction_counterevidence_round_trips_and_changes_operation_identity() -> None:
    correction = MemoryOperation(
        intent=MemoryIntent.CORRECT, target_ids=("inference",), evidence_ids=("counterexample",)
    )
    assert load_operation(dump_operation(correction)) == correction
    host_correction = MemoryOperation(intent=MemoryIntent.CORRECT, target_ids=("inference",))
    assert host_correction.evidence_ids == ()
    assert operation_key(correction, recipe="v5") != operation_key(host_correction, recipe="v5")


def test_forgetting_does_not_accept_evidence_as_authorization() -> None:
    with pytest.raises(ValidationError):
        MemoryOperation(
            intent=MemoryIntent.FORGET, target_ids=("memory",), evidence_ids=("permission",)
        )
