"""The public correction value preserves counterevidence through durable serialization."""

import inspect

import pytest

from sensemem import (
    AsyncMemory,
    IdentityChange,
    Memory,
    MemoryIntent,
    MemoryOperation,
    ValidationError,
)
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


def test_host_identity_binding_keeps_the_chosen_survivor_in_its_public_payload() -> None:
    binding = MemoryOperation(
        intent=MemoryIntent.MERGE,
        identity=IdentityChange(identity_id="face", moved_ids=("voice",)),
        rationale="The host confirmed the speaking person.",
    )
    assert load_operation(dump_operation(binding)) == binding
    assert binding.target_ids == binding.evidence_ids == ()


@pytest.mark.parametrize("owner", (Memory, AsyncMemory))
def test_answer_contract_has_no_implicit_identity_binding_switch(
    owner: type[Memory] | type[AsyncMemory],
) -> None:
    assert "link_identities" not in inspect.signature(owner.ask).parameters
    assert "link_identities" not in inspect.signature(owner.ask_stream).parameters
