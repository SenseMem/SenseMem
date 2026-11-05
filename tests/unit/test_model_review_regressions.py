"""Completion, budget, and autonomous mutation boundaries from the model review."""

import json
from dataclasses import replace
from pathlib import Path

import httpx2 as httpx
import pytest
from _feature_support import TinyEmbedder
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_memory_control_plane import ScriptedConsolidator
from test_openai_sdk import _consolidation_evidence, _sdk_client, _search_hit

from sensemem import (
    FormationProposal,
    Memory,
    MemoryConfig,
    MemoryIntent,
    MemoryKind,
    MemoryOperation,
    MemoryTrigger,
    ModelError,
    ModelInput,
    OpenAIModels,
    ValidationError,
)
from sensemem._telemetry import model_span


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish", [None, "tool_calls"])
def test_incomplete_answer_never_succeeds_or_reinforces(
    tmp_path: Path, stream: bool, finish: str | None
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        choice: dict[str, object] = {"index": 0, "finish_reason": finish}
        if stream:
            choice["delta"] = {"content": "An unfinished answer"}
            body = f"data: {json.dumps({'choices': [choice]})}\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        choice["message"] = {"content": "An unfinished answer"}
        return httpx.Response(200, json={"choices": [choice]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAIModels(_sdk_client(transport))
        if not stream:
            with pytest.raises(ModelError) as failure:
                adapter.answer(ModelInput(text="What happened?"), (_search_hit(),))
            assert failure.value.reason == "response_invalid"
            return
        with Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            answerer=adapter,
            config=MemoryConfig(minimum_relevance=0),
        ) as memory:
            source = memory.add("A meeting happened.")
            with pytest.raises(ModelError) as failure:
                memory.ask("What happened?")
            assert failure.value.reason == "response_invalid"
            stored = memory._store.records.read_memory(source.id)
            assert stored is not None and stored.access_count == 0


@pytest.mark.parametrize(
    "field", ["max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "seed"]
)
def test_extra_body_cannot_replace_owned_generation_controls(field: str) -> None:
    with pytest.raises(ValidationError):
        OpenAIModels(generation_model="compatible-model", generation_extra_body={field: 1})


@pytest.mark.parametrize("limit", [True, 0, -1])
def test_media_diagnosis_rejects_invalid_budgets(limit: int) -> None:
    with pytest.raises(ValidationError, match="generation_media_max_tokens"):
        OpenAIModels(generation_media_max_tokens=limit)


@pytest.mark.parametrize(
    "global_limit,media_limit,expected",
    [(None, 8192, 8192), (None, 12000, 12000), (2000, 8192, 2000)],
)
def test_reasoning_media_diagnosis_has_an_independent_honored_budget(
    global_limit: int | None, media_limit: int, expected: int
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["max_completion_tokens"] == expected
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"content": '{"requests":[]}'}, "finish_reason": "stop"}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAIModels(
            _sdk_client(transport),
            generation_max_tokens=global_limit,
            generation_media_max_tokens=media_limit,
        )
        assert adapter.select_media({}, count=1, max_items=1) == ()


def test_truncated_diagnosis_is_observable_without_losing_the_text_fallback() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"content": ""}, "finish_reason": "length"}]},
        )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with (
            httpx.Client(transport=httpx.MockTransport(respond)) as transport,
            model_span(provider.get_tracer(__name__), "diagnosis", attributes={}),
        ):
            adapter = OpenAIModels(_sdk_client(transport))
            assert adapter.select_media({}, count=1, max_items=1) == ()
        diagnosis = next(
            span for span in exporter.get_finished_spans() if span.name.endswith("media_diagnosis")
        )
        assert diagnosis.attributes is not None
        assert diagnosis.attributes["sensemem.grounding.media_diagnosis.truncated"] is True
        assert diagnosis.attributes["sensemem.grounding.media_diagnosis.valid"] is False
    finally:
        provider.shutdown()


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("missing", "counterevidence_required"),
        ("self", "target_is_evidence"),
        ("unshown", "evidence_not_shown"),
        ("valid", None),
    ],
)
def test_model_correction_requires_shown_counterevidence_and_keeps_it_in_the_log(
    tmp_path: Path, mode: str, reason: str | None
) -> None:
    consolidator = ScriptedConsolidator()
    with Memory(tmp_path, embedder=TinyEmbedder(), consolidator=consolidator) as memory:
        original = memory.add("The toolbox is blue.")
        contrary = memory.add("The toolbox is red; the earlier blue description was wrong.")
        created = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(original.id,),
                proposal=FormationProposal(
                    kind=MemoryKind.EVENT, content="The toolbox is blue.", confidence=0.9
                ),
            )
        )
        target = created.created_ids[0]
        evidence = () if mode == "missing" else (target,) if mode == "self" else (contrary.id,)
        correction = MemoryOperation(
            intent=MemoryIntent.CORRECT, target_ids=(target,), evidence_ids=evidence
        )
        consolidator._scripts.append((correction,))
        report = memory.consolidate(
            evidence_ids=(target,) if mode == "unshown" else (target, contrary.id)
        )
        if reason is not None:
            assert report.rejected == ((correction, reason),)
            assert not report.operations
        else:
            assert report.operations[0].operation.evidence_ids == (contrary.id,)
            assert memory.operations()[0].operation.evidence_ids == (contrary.id,)
            assert memory.rollback(report.operations[0].operation_id)
        context = memory.get(target).context
        assert context is not None and context.retired_at is None


@pytest.mark.parametrize(
    "budget,trigger,reason",
    [
        (None, MemoryTrigger.PRESSURE, "unauthorized"),
        (1, MemoryTrigger.MANUAL, "unauthorized"),
        (2, MemoryTrigger.PRESSURE, "stale"),
        (1, MemoryTrigger.PRESSURE, None),
    ],
)
def test_model_forgetting_requires_host_pressure_and_cannot_go_below_budget(
    tmp_path: Path, budget: int | None, trigger: MemoryTrigger, reason: str | None
) -> None:
    consolidator = ScriptedConsolidator()
    with Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=consolidator,
        config=MemoryConfig(memory_budget_records=budget),
    ) as memory:
        first, second = memory.add_many(("Keep the original note.", "Another note."))
        operation = MemoryOperation(intent=MemoryIntent.FORGET, target_ids=(first.id,))
        consolidator._scripts.append((operation,))
        report = memory.consolidate(evidence_ids=(first.id, second.id), trigger=trigger)
        assert (report.rejected[0][1] if report.rejected else None) == reason
        assert report.applied == int(reason is None)
        if reason is None:
            consolidator._scripts.append((replace(operation, target_ids=(second.id,)),))
            next_report = memory.consolidate(evidence_ids=(second.id,), trigger=trigger)
            assert next_report.rejected[0][1] == "stale"
            assert memory.get(second.id).forgotten_at is None


def test_derived_claim_and_its_own_source_are_not_independent_duplicates(tmp_path: Path) -> None:
    consolidator = ScriptedConsolidator()
    with Memory(tmp_path, embedder=TinyEmbedder(), consolidator=consolidator) as memory:
        source = memory.add("The toolbox is blue.")
        original = memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE,
                evidence_ids=(source.id,),
                proposal=FormationProposal(
                    kind=MemoryKind.EVENT, content="The toolbox is blue.", confidence=0.9
                ),
            )
        )
        target = original.created_ids[0]
        operation = MemoryOperation(
            intent=MemoryIntent.CONSOLIDATE,
            evidence_ids=(source.id, target),
            target_ids=(source.id, target),
            proposal=FormationProposal(
                kind=MemoryKind.EVENT, content="The blue toolbox was observed.", confidence=0.9
            ),
        )
        consolidator._scripts.append((operation,))
        report = memory.consolidate(evidence_ids=(source.id, target))
        assert report.rejected == ((operation, "dependent_evidence"),)
        assert memory.get(source.id).forgotten_at is None
        assert memory.get(target).forgotten_at is None
        assert memory.apply(operation).created_ids


def test_consolidation_shows_source_relationships_without_inventing_missing_indices() -> None:
    source, derived = _consolidation_evidence()
    assert derived.context is not None
    derived = replace(
        derived, context=replace(derived.context, evidence_ids=(source.id, "omitted"))
    )

    def respond(request: httpx.Request) -> httpx.Response:
        evidence = json.loads(json.loads(request.content)["messages"][1]["content"])["evidence"]
        assert evidence[0]["created_at"] == source.created_at.isoformat()
        assert evidence[1]["context"]["evidence_indices"] == [0]
        assert evidence[1]["context"]["sources_not_shown_count"] == 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": '{"operations":[]}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        model = OpenAIModels(_sdk_client(transport))
        assert model.consolidate((source, derived), trigger=MemoryTrigger.MANUAL) == ()
