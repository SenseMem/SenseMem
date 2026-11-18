"""REST request and response contracts, independent of application routing."""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from sensemem.api.content import (
    Content,
    ContextBudgetInput,
    ContextSupplementInput,
    Limit,
    StrictModel,
)
from sensemem.identity import (
    ConsentState,
    PerceptionKind,
)
from sensemem.spatial import ObservationContext, RetrievalScope
from sensemem.types import AnswerPolicy, MemoryCategory

# The same identifier inside a request body, where `PathParameter` does not apply. Constrained
# here rather than only in the SDK so a malformed ID fails validation on every route alike.
_BodyMemoryId = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
# A bounded plain string for person-facing text -- a name or relationship label -- the same shape
# the MCP `register_identity` tool's `name`/`relationship` arguments use. Named separately from
# `_BodyMemoryId` so a future ID-specific tightening of that type cannot silently start rejecting
# names too.
_PersonText = _BodyMemoryId


class MemoryCreate(StrictModel):
    content: Content
    occurred_at: AwareDatetime | None = None
    occurred_end: AwareDatetime | None = None
    metadata: dict[str, JsonValue] | None = None
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC
    context: ObservationContext | None = None


class MemoryBatchCreate(StrictModel):
    contents: Annotated[list[Content], Field(min_length=1, max_length=100)]
    occurred_at: list[AwareDatetime | None] | None = None
    occurred_end: list[AwareDatetime | None] | None = None
    metadata: list[dict[str, JsonValue] | None] | None = None
    memory_type: MemoryCategory = MemoryCategory.SEMANTIC
    context: list[ObservationContext | None] | None = None


class QueryRequest(StrictModel):
    query: Content
    limit: Limit = 10
    memory_type: MemoryCategory | None = None
    reference_at: AwareDatetime | None = None
    occurred_from: AwareDatetime | None = None
    occurred_until: AwareDatetime | None = None
    scope: RetrievalScope | None = None
    explain: bool = False

    @model_validator(mode="after")
    def validate_occurrence_range(self) -> QueryRequest:
        if (
            self.occurred_from is not None
            and self.occurred_until is not None
            and self.occurred_until <= self.occurred_from
        ):
            raise ValueError("occurred_until must be later than occurred_from")
        return self


class ReinforceRequest(StrictModel):
    memory_ids: Annotated[list[_BodyMemoryId], Field(min_length=1, max_length=100)]


class AnswerRequest(StrictModel):
    question: Content
    limit: Limit = 5
    memory_type: MemoryCategory | None = None
    reference_at: AwareDatetime | None = None
    scope: RetrievalScope | None = None
    # Abstention is the caller's policy, not the server's: `best_effort` commits to the most
    # likely answer and still reports `abstained`. The default keeps the refusal behaviour.
    answer_policy: AnswerPolicy = "strict"


# The shared bounds under the name FastAPI publishes as the OpenAPI component; no fields and no
# docstring of its own, so the schema stays what it was.
class ContextBudgetRequest(ContextBudgetInput):
    pass


class ContextRequest(StrictModel):
    goal: Content
    budget: ContextBudgetRequest | None = None
    reference_at: AwareDatetime | None = None
    scope: RetrievalScope | None = None
    allow_partial_sources: Annotated[bool, Field(strict=True)] = False
    supplements: tuple[ContextSupplementInput, ...] = ()


class SettleRequest(StrictModel):
    limit: Limit = 100
    max_attempts: Annotated[int, Field(strict=True, ge=1)] = 3
    memory_ids: Annotated[list[_BodyMemoryId], Field(min_length=1, max_length=100)] | None = None


class PerceptionRequest(StrictModel):
    memory_id: _BodyMemoryId
    kinds: Annotated[frozenset[PerceptionKind], Field(min_length=1)] | None = None


class IdentityRegisterRequest(StrictModel):
    identity_id: _BodyMemoryId
    name: _PersonText
    relationship: _PersonText | None = None


class ConsentRequest(StrictModel):
    state: ConsentState
    note: _PersonText | None = None


class RetentionRequest(StrictModel):
    dry_run: bool = False
