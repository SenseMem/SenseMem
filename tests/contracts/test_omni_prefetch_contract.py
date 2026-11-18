"""Revision and compiled-evidence values on the supported streaming surface."""

from typing import cast

import pytest

from sensemem import (
    AsyncMemory,
    AsyncOmniPrefetch,
    ContextBudget,
    MemoryCategory,
    PrefetchResult,
    PrefetchUpdate,
    ValidationError,
)


def test_prefetch_updates_carry_valid_request_local_revisions() -> None:
    result = PrefetchResult(hits=())
    assert result.bundle is None
    assert PrefetchUpdate(1, result).result is result
    for invalid in (0, -1, True):
        with pytest.raises(ValidationError):
            PrefetchUpdate(invalid, result)


def test_compiled_prefetch_does_not_silently_ignore_search_filters() -> None:
    with pytest.raises(ValidationError, match="budget filters"):
        AsyncOmniPrefetch(
            cast(AsyncMemory, object()),
            budget=ContextBudget(),
            memory_type=MemoryCategory.EPISODIC,
        )
