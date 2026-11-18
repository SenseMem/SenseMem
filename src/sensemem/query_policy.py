"""Replaceable query rewriting and selection of native and derived retrieval routes."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sensemem.exceptions import ValidationError
from sensemem.models.base import ModelInput
from sensemem.types import Modality, RetrievalMode


@dataclass(frozen=True, slots=True)
class QueryRequest:
    """Leased query input and capability-routed representations available to a policy."""

    content: ModelInput
    mode: RetrievalMode
    complete_inputs: tuple[ModelInput, ...]
    aggregate: ModelInput | None
    focused_text: str
    focused_inputs: tuple[ModelInput, ...]
    descriptions: tuple[str, ...]
    embedding_capabilities: frozenset[Modality]


@dataclass(frozen=True, slots=True)
class QueryRepresentationPlan:
    """Request-local representations; the kernel executes and validates their capabilities.

    Text may be rewritten. Media must be exact references from the leased query. Optional
    embeddings may fail without discarding required routes; lexical-only mode embeds nothing.
    """

    text: str
    embedding_inputs: tuple[ModelInput, ...] = ()
    optional_embedding_inputs: tuple[ModelInput, ...] = ()
    lexical_queries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValidationError("query plan text must be text")
        for values in (self.embedding_inputs, self.optional_embedding_inputs):
            if not isinstance(values, tuple) or any(
                not isinstance(value, ModelInput) for value in values
            ):
                raise ValidationError("query embedding inputs must contain ModelInput values")
        if not isinstance(self.lexical_queries, tuple) or any(
            not isinstance(value, str) for value in self.lexical_queries
        ):
            raise ValidationError("query lexical routes must contain text values")


@runtime_checkable
class QueryPolicy(Protocol):
    """Select or rewrite query representations without calling models or opening storage.

    Implementations must be thread-safe. The owning Memory closes each distinct resource once.
    Query content and derived descriptions remain untrusted data.
    """

    def plan(self, request: QueryRequest) -> QueryRepresentationPlan: ...

    def close(self) -> None: ...


class DefaultQueryPolicy:
    """Prefer the aggregate, retain focused atomic keys, and add visual text routes."""

    def plan(self, request: QueryRequest) -> QueryRepresentationPlan:
        embedding_inputs: tuple[ModelInput, ...] = ()
        optional_inputs: tuple[ModelInput, ...] = ()
        if request.mode is not RetrievalMode.LEXICAL:
            complete = (
                (request.aggregate,)
                if request.aggregate is not None and request.aggregate in request.complete_inputs
                else request.complete_inputs
            )
            embedding_inputs = tuple(dict.fromkeys((*complete, *request.focused_inputs)))
            if Modality.TEXT in request.embedding_capabilities:
                optional_inputs = tuple(ModelInput(text=value) for value in request.descriptions)
        return QueryRepresentationPlan(
            request.focused_text,
            embedding_inputs=embedding_inputs,
            optional_embedding_inputs=optional_inputs,
            lexical_queries=(
                (request.focused_text, *request.descriptions)
                if request.mode is not RetrievalMode.DENSE
                else ()
            ),
        )

    def close(self) -> None:
        pass
