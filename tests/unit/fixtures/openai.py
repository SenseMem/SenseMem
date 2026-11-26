"""Shared OpenAI transport and value helpers for unit tests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import httpx2 as httpx
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.completion_usage import CompletionUsage

from sensemem import (
    AssetRef,
    EvidenceBasis,
    MemoryRecord,
    Modality,
    RecordKind,
    RecordSemantics,
    SearchHit,
)

NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


def stream_chunk(
    choices: list[ChunkChoice], *, usage: CompletionUsage | None = None
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chatcmpl-fixture",
        created=0,
        model="caption-model",
        object="chat.completion.chunk",
        choices=choices,
        usage=usage,
    )


def consolidation_evidence() -> tuple[MemoryRecord, ...]:
    return (
        MemoryRecord(id="raw-1", content="The lamp looks on.", created_at=NOW),
        MemoryRecord(
            id="derived-1",
            content="Ana is patient",
            created_at=NOW,
            context=RecordSemantics(
                kind=RecordKind.TRAIT,
                basis=EvidenceBasis.MODEL_INFERENCE,
                confidence=0.6,
                valid_from=NOW,
                valid_until=None,
                recorded_at=NOW,
                subject="Ana",
                predicate="disposition",
                value="patient",
                evidence_ids=("raw-0",),
            ),
        ),
    )


def search_hit() -> SearchHit:
    return SearchHit(id="memory_1", content="The toolbox is blue.", score=0.9, created_at=NOW)


def sdk_client(client: httpx.Client, *, api_key: str = "sdk-secret") -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://sdk.example.test/v1",
        http_client=client,
        max_retries=0,
    )


def make_asset(
    directory: Path,
    asset_id: str,
    modality: Modality,
    media_type: str,
    data: bytes,
) -> AssetRef:
    path = directory / f"{asset_id}.bin"
    path.write_bytes(data)
    return AssetRef(
        id=asset_id,
        modality=modality,
        media_type=media_type,
        size_bytes=len(data),
        sha256="a" * 64,
        name=f"{asset_id}.wav" if modality is Modality.AUDIO else f"{asset_id}.bin",
        path=path,
    )


def answer_policy_transport(
    requests: list[dict[str, object]],
    reply: str,
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": reply}, "finish_reason": "stop"}]
            },
        )

    return httpx.MockTransport(respond)
