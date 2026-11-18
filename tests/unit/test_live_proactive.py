"""Native evidence gates proactive proposals and their attributed dialogue records."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
from _feature_support import ATOMIC_MODALITIES, TinyEmbedder
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    Blob,
    EventObservation,
    EventSegment,
    EvidenceBasis,
    FormationProposal,
    Memory,
    MemoryIntent,
    MemoryOperation,
    MemorySettings,
    MemorySpec,
    ModelInput,
    ObservationContext,
    RecordKind,
    SearchHit,
)


@pytest.fixture(autouse=True)
def example_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2]))


class Segmenter:
    event_model = "proactive-fixture"
    event_recipe = "proactive-fixture:v1"

    def segment_events(
        self, observations: Sequence[EventObservation], *, end_of_stream: bool = False
    ) -> tuple[EventSegment, ...]:
        return ()


class Reviewer:
    generation_capabilities = ATOMIC_MODALITIES

    def __init__(self) -> None:
        self.inputs: list[ModelInput] = []
        self.result = '{"speak":true,"utterance":"要我记住钥匙的位置吗?","reason":"follow_up"}'

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        self.inputs.append(question)
        return AnswerResult(self.result, hits=tuple(hits))

    def close(self) -> None:
        pass


@pytest.mark.parametrize(
    "value",
    [
        '{"speak":false,"utterance":"still speaking","reason":"none"}',
        '{"speak":true,"utterance":"","reason":"follow_up"}',
        '{"speak":true,"utterance":"hello","reason":"none"}',
        '{"speak":"true","utterance":"hello","reason":"follow_up"}',
        '{"speak":false,"utterance":"","reason":"none","extra":1}',
    ],
)
def test_inconsistent_or_non_structured_decisions_cannot_speak(value: str) -> None:
    from examples.live.app import ProactiveDecision

    with pytest.raises(ValidationError):
        ProactiveDecision.model_validate_json(value)


def test_source_review_and_proactive_dialogue_preserve_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from examples.live import app as demo
    from examples.live.updates import LiveEvent

    reviewer = Reviewer()
    runtimes: list[demo.LiveRuntime] = []

    class ObservedRuntime(demo.LiveRuntime):
        def __init__(
            self,
            config: MemorySpec,
            memory_factory: demo.MemoryFactory,
            realtime_url: str,
        ) -> None:
            super().__init__(config, memory_factory, realtime_url)
            runtimes.append(self)

    monkeypatch.setattr(demo, "LiveRuntime", ObservedRuntime)

    def factory(config: MemorySpec) -> tuple[AsyncMemory, Segmenter]:
        return AsyncMemory(
            Memory(
                config.data_dir,
                embedder=TinyEmbedder(),
                answerer=reviewer,
                settings=MemorySettings(minimum_relevance=0, ambiguity_margin=0),
            )
        ), Segmenter()

    config = MemorySpec.model_validate(
        {
            "data_dir": tmp_path,
            "embedding": {"provider": "openai"},
            "generation": {"provider": "openai"},
        }
    )
    with TestClient(demo.create_app(config, memory_factory=factory)) as client:
        source_id = client.post("/api/session").json()["source_id"]
        runtime = runtimes[0]

        async def review() -> dict[str, object]:
            memory = await runtime.open_memory()
            record = await memory.add(
                (
                    "Keys placed in a bowl",
                    *(Blob(f"original-image-{index}".encode(), "image/png") for index in range(6)),
                ),
                context=ObservationContext(source_id=source_id),
            )
            now = datetime.now(timezone.utc)
            event = LiveEvent(
                "event-1", source_id, (record.id,), "Unverified summary", now, now, ("image",)
            )
            proposal = await runtime.assess_event(event, runtime.memory_revision, "visual")
            assert proposal is not None
            assert len((await memory.list()).items) == 1  # A proposal is never a memory.
            assert len(reviewer.inputs[0].assets) == 6  # The completing tail is not dropped.
            assert reviewer.inputs[0].assets[0].path is not None
            assert reviewer.inputs[0].assets[0].path.read_bytes() == b"original-image-0"
            assert '"summary_verified": false' in reviewer.inputs[0].text
            assert '"occurred_at":' in reviewer.inputs[0].text
            assert '"assessed_at":' in reviewer.inputs[0].text
            assert proposal["reviewed_memory_ids"] == [record.id]
            assert '"observation_sources":' in reviewer.inputs[0].text
            assert '"asset_order":' in reviewer.inputs[0].text
            # A declined or invalidated proposal is not a previously completed contribution.
            assert await runtime.assess_event(event, runtime.memory_revision, "memory") is not None
            assert '"previous_suggestions": []' in reviewer.inputs[-1].text
            reviewer.result = "Not JSON: do not speak"
            assert await runtime.assess_event(event, runtime.memory_revision, "memory") is None
            reviewer.result = '{"speak":false,"utterance":"","reason":"none"}'
            assert await runtime.assess_event(event, runtime.memory_revision, "audio") is None
            await memory.forget((record.id,))
            assert await runtime._revalidate_review((record,)) is None
            assert await runtime.assess_event(event, runtime.memory_revision, "memory") is None
            return proposal

        assert client.portal is not None
        proposal = client.portal.call(review)
        payload = {
            "origin": "proactive",
            "source_id": source_id,
            "response_id": "response-1",
            "started_at": "2026-11-16T10:00:00Z",
            "completed_at": "2026-11-16T10:00:03Z",
            "candidate_id": proposal["candidate_id"],
            "event_id": proposal["event_id"],
            "evidence_ids": proposal["evidence_ids"],
            "assistant": {"item_ids": ["assistant-1"], "text": proposal["utterance"]},
        }
        assert (
            client.post("/api/dialogue", json={**payload, "candidate_id": "invented"}).status_code
            == 422
        )
        assert (
            client.post(
                "/api/dialogue", json={**payload, "user": {"item_id": "u", "text": "invented"}}
            ).status_code
            == 422
        )
        accepted = client.post("/api/dialogue", json=payload)
        assert accepted.status_code == 202

        async def inspect() -> None:
            record = await (await runtime.open_memory()).get(accepted.json()["id"])
            dialogue = record.metadata["dialogue"]
            assert isinstance(dialogue, dict)
            assert dialogue["origin"] == "proactive" and dialogue["user"] is None
            assert dialogue["evidence_ids"] == proposal["evidence_ids"]
            assert dialogue["reviewed_memory_ids"] == proposal["reviewed_memory_ids"]
            assert proposal["candidate_id"] in runtime.completed_initiatives
            assert dialogue["event_id"] == "event-1"
            assert record.content == f"SenseMem\uff1a\n{proposal['utterance']}"

        client.portal.call(inspect)


async def test_unrelated_write_does_not_withdraw_intact_reviewed_evidence(tmp_path: Path) -> None:
    from examples.live.app import LiveRuntime

    config = MemorySpec.model_validate(
        {
            "data_dir": tmp_path,
            "embedding": {"provider": "openai"},
        }
    )
    runtime = LiveRuntime(
        config,
        lambda config: (AsyncMemory(Memory(config.data_dir, embedder=TinyEmbedder())), Segmenter()),
        "wss://realtime.example.test/v1/realtime",
    )
    try:
        memory = await runtime.open_memory()
        source = await memory.add("The keys were moved into the white bowl")
        revision = runtime.memory_revision
        await memory.add("An unrelated background observation")
        runtime.memory_changed(schedule=False)
        assert runtime.memory_revision > revision
        assert await runtime._revalidate_review((source,)) == runtime.memory_revision
        await memory.forget((source.id,))
        runtime.memory_changed(schedule=False)
        assert await runtime._revalidate_review((source,)) is None
    finally:
        await runtime.close()


async def test_withdrawn_support_rejects_a_reviewed_hit_that_becomes_hidden(tmp_path: Path) -> None:
    from examples.live.app import LiveRuntime

    config = MemorySpec.model_validate({"data_dir": tmp_path, "embedding": {"provider": "openai"}})
    runtime = LiveRuntime(
        config,
        lambda config: (
            AsyncMemory(
                Memory(
                    config.data_dir,
                    embedder=TinyEmbedder(),
                    settings=MemorySettings(independent_evidence=True, minimum_relevance=0),
                )
            ),
            Segmenter(),
        ),
        "wss://realtime.example.test/v1/realtime",
    )
    try:
        memory = await runtime.open_memory()
        roots = (await memory.add("First independent event"), await memory.add("Second event"))
        proposal = FormationProposal(
            kind=RecordKind.TRAIT,
            basis=EvidenceBasis.MODEL_INFERENCE,
            content="Ada is patient",
            subject="Ada",
            predicate="disposition",
            value="patient",
            confidence=0.6,
        )
        first = await memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE, evidence_ids=(roots[0].id,), proposal=proposal
            )
        )
        second = await memory.apply(
            MemoryOperation(
                intent=MemoryIntent.CONSOLIDATE, evidence_ids=(roots[1].id,), proposal=proposal
            )
        )
        reviewed_hit = next(
            hit
            for hit in (await memory.search("patient", limit=10)).hits
            if hit.id == first.created_ids[0]
        )
        assert await runtime._revalidate_review((reviewed_hit,)) == runtime.memory_revision
        assert await memory.rollback(second.operation_id)
        runtime.memory_changed(schedule=False)
        hidden = await memory.get(reviewed_hit.id)
        assert hidden.forgotten_at is None
        assert hidden.context is not None and not hidden.context.visible
        assert hidden.context.retired_at is None
        assert reviewed_hit.id not in {hit.id for hit in (await memory.search("patient")).hits}
        assert await runtime._revalidate_review((reviewed_hit,)) is None
    finally:
        await runtime.close()
