"""The bounded native read port keeps existing index failure mapping."""

from collections.abc import Callable, Sequence
from typing import cast

import pytest

from sensemem import (
    CandidateQuery,
    CandidateRequest,
    IndexUnavailableError,
    RetrievalMode,
    ValidationError,
)
from sensemem.infrastructure.local.zvec_index import IndexHit
from sensemem.kernel.candidates import NativeCandidateReader
from sensemem.kernel.execution import Execution
from sensemem.kernel.runtime import Index


class BrokenIndex:
    def search(self, vector: Sequence[float], **options: object) -> tuple[IndexHit, ...]:
        raise OSError("native backend body")


class InlineRoutes:
    def index_routes(
        self, calls: Sequence[Callable[[], tuple[IndexHit, ...]]]
    ) -> tuple[tuple[IndexHit, ...], ...]:
        return tuple(call() for call in calls)


def _reader() -> NativeCandidateReader:
    return NativeCandidateReader(
        index=cast(Index, BrokenIndex()),
        execution=cast(Execution, InlineRoutes()),
        request=CandidateRequest(
            query_text="token",
            vectors=((1.0, 0.0),),
            lexical_queries=("token",),
            limit=10,
            max_routes=2,
            mode=RetrievalMode.HYBRID,
            memory_types=None,
            occurred_from=None,
            occurred_until=None,
            preferred_range=None,
        ),
        space_id="fixture-space",
        task="document",
        dimension=2,
        place_id=None,
        identity_id=None,
    )


def test_native_failure_retains_index_unavailable_error_mapping() -> None:
    with pytest.raises(IndexUnavailableError, match="failed to search memories"):
        _reader().read((CandidateQuery(kind="dense", vector=(1.0, 0.0)),))


def test_native_read_budget_is_enforced_before_starting_routes() -> None:
    query = CandidateQuery(kind="dense", vector=(1.0, 0.0))
    with pytest.raises(ValidationError, match="native route budget"):
        _reader().read((query, query, query))


def test_invalid_native_vector_is_refused_before_index_execution() -> None:
    with pytest.raises(ValidationError, match="invalid vector"):
        _reader().read((CandidateQuery(kind="dense", vector=(1.0,)),))
