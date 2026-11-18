"""Ingestion computations propose immutable interpretation without source authority."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from sensemem import StageResult, ValidationError


@pytest.mark.parametrize("interpretation", [None, 1, False, (), {}])
def test_stage_result_requires_text(interpretation: object) -> None:
    with pytest.raises(ValidationError, match="stage interpretation must be text"):
        StageResult(cast(str, interpretation))


def test_stage_result_normalizes_derived_text_and_is_immutable() -> None:
    result = StageResult("  A possible interpretation  ")
    assert result.interpretation == "A possible interpretation"
    assert StageResult().interpretation == StageResult(" \n ").interpretation == ""
    with pytest.raises(FrozenInstanceError):
        result.interpretation = "Original source"  # type: ignore[misc]
