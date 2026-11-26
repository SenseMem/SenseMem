"""Query plans are immutable typed proposals with provider-neutral default routing."""

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from sensemem import (
    DefaultQueryPolicy,
    Modality,
    ModelInput,
    QueryRepresentationPlan,
    QueryRequest,
    RetrievalMode,
    ValidationError,
)


@pytest.mark.parametrize("text", [None, 1, False, ()])
def test_query_plan_requires_text(text: object) -> None:
    with pytest.raises(ValidationError, match="query plan text"):
        QueryRepresentationPlan(cast(str, text))


@pytest.mark.parametrize("field", ["embedding_inputs", "optional_embedding_inputs"])
@pytest.mark.parametrize("value", [None, "query", ("query",), [ModelInput(text="query")]])
def test_query_plan_requires_immutable_model_input_routes(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="query embedding inputs"):
        if field == "embedding_inputs":
            QueryRepresentationPlan("query", embedding_inputs=cast(tuple[ModelInput, ...], value))
        else:
            QueryRepresentationPlan(
                "query", optional_embedding_inputs=cast(tuple[ModelInput, ...], value)
            )


@pytest.mark.parametrize("value", [None, "query", (1,), ["query"]])
def test_query_plan_requires_immutable_lexical_text_routes(value: object) -> None:
    with pytest.raises(ValidationError, match="query lexical routes"):
        QueryRepresentationPlan("query", lexical_queries=cast(tuple[str, ...], value))


@pytest.mark.parametrize("mode", list(RetrievalMode))
@pytest.mark.parametrize("aggregate_supported", [False, True])
def test_default_query_policy_preserves_aggregate_focus_and_description_routes(
    mode: RetrievalMode, aggregate_supported: bool
) -> None:
    aggregate, atomic, focused = (
        ModelInput(text=value) for value in ("complete", "atomic", "focus")
    )
    complete = (aggregate, atomic) if aggregate_supported else (atomic,)
    request = QueryRequest(
        content=aggregate,
        mode=mode,
        complete_inputs=complete,
        aggregate=aggregate,
        focused_text="focus",
        focused_inputs=(focused, aggregate) if aggregate_supported else (focused,),
        descriptions=("visual description",),
        embedding_capabilities=frozenset({Modality.TEXT}),
    )
    plan = DefaultQueryPolicy().plan(request)
    assert plan.text == "focus"
    if mode is RetrievalMode.LEXICAL:
        assert plan.embedding_inputs == plan.optional_embedding_inputs == ()
    else:
        assert plan.embedding_inputs == (
            (aggregate, atomic, focused) if aggregate_supported else (atomic, focused)
        )
        assert plan.optional_embedding_inputs == (ModelInput(text="visual description"),)
    assert plan.lexical_queries == (
        () if mode is RetrievalMode.DENSE else ("focus", "complete", "visual description")
    )
    with pytest.raises(FrozenInstanceError):
        plan.text = "changed"  # type: ignore[misc]
