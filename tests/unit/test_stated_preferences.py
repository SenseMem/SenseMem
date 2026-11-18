"""Automatic preferences retain inference authority and current/historical values."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx2 as httpx
from _feature_support import TinyEmbedder
from openai import OpenAI

from sensemem import (
    EvidenceBasis,
    Memory,
    MemorySettings,
    ObservationContext,
    RecordKind,
    RetrievalScope,
)
from sensemem.models.openai_completion import OpenAICompletion


def test_inferred_current_preference_is_immediate_and_a_change_preserves_history(
    tmp_path: Path,
) -> None:
    clock = datetime(2026, 3, 1, tzinfo=timezone.utc)

    def respond(request: httpx.Request) -> httpx.Response:
        source = json.loads(json.loads(request.content)["messages"][1]["content"])["observations"][
            0
        ]
        statement = source["content"]
        value = "coffee" if "coffee" in statement else "tea"
        proposal = {
            "kind": "state",
            "content": statement,
            "subject": "Ana",
            "predicate": "current_preferred_drink"
            if value == "coffee"
            else "only_current_preferred_drink",
            "value": value,
            "confidence": 0.9,
            "evidence_observation_ids": ["observation_0"],
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {"observation_id": "observation_0", "proposals": [proposal]}
                                    ]
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
        Memory(
            tmp_path,
            embedder=TinyEmbedder(),
            former=OpenAICompletion(client, generation_model="fixture-snapshot"),
            settings=MemorySettings(minimum_relevance=0),
        ) as memory,
    ):
        for value, when in (("tea", clock), ("coffee", clock + timedelta(days=1))):
            memory.add(
                f"Ana's current preferred drink is {value}.",
                occurred_at=when,
                context=ObservationContext(basis=EvidenceBasis.USER_STATEMENT),
            )
            traits = [
                hit
                for hit in memory.search("Ana's current preference").hits
                if hit.context and hit.context.kind is RecordKind.STATE
            ]
            assert len(traits) == 1
            assert traits[0].context is not None
            assert traits[0].context.basis is EvidenceBasis.MODEL_INFERENCE
            assert traits[0].context.value == value
        historical = [
            hit
            for hit in memory.search("Ana's preference", scope=RetrievalScope(valid_at=clock)).hits
            if hit.context and hit.context.kind is RecordKind.STATE
        ]
        assert len(historical) == 1
        assert historical[0].context is not None and historical[0].context.value == "tea"
