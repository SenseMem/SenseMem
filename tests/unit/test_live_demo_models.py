"""The checked-in live recipe preserves native multimodal embedding inputs."""

from __future__ import annotations

import base64
import hashlib
import json
from contextlib import closing
from pathlib import Path
from typing import cast

import httpx2
import pytest
from openai import OpenAI

import sensemem.recipes as recipes
from sensemem import (
    AssetRef,
    EmbedTask,
    MemorySpec,
    Modality,
    ModelInput,
    build_memory_composition,
)


@pytest.mark.parametrize(
    ("task", "prefix"),
    [
        (EmbedTask.DOCUMENT, "title: none | text: "),
        (EmbedTask.QUERY, "task: search result | query: "),
    ],
)
def test_live_embedding_recipe_uses_existing_native_embeddinggemma2_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: EmbedTask, prefix: str
) -> None:
    config = MemorySpec.model_validate_json(
        (Path(__file__).parents[2] / "examples/live/config.example.json").read_text()
    )
    # Only the embedding slot is exercised, so no hosted-model credential is needed.
    config = config.model_copy(
        update={"data_dir": tmp_path, "generation": None, "formation": None, "consolidation": None}
    )
    assets: list[AssetRef] = []
    parts: list[dict[str, object]] = [{"type": "text", "text": prefix + "Remember this."}]
    for modality, media_type in (
        (Modality.AUDIO, "audio/wav"),
        (Modality.VIDEO, "video/webm"),
    ):
        data = f"original {modality.value} bytes".encode()
        path = tmp_path / modality.value
        path.write_bytes(data)
        assets.append(
            AssetRef(
                id=modality.value,
                modality=modality,
                media_type=media_type,
                size_bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                path=path,
            )
        )
        kind = f"{modality.value}_url"
        parts.append(
            {
                "type": kind,
                kind: {"url": f"data:{media_type};base64,{base64.b64encode(data).decode()}"},
            }
        )

    requests: list[dict[str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "http://127.0.0.1:8000/v1/embeddings"
        assert request.headers["authorization"] == "Bearer EMPTY"
        payload = cast(dict[str, object], json.loads(request.content))
        requests.append(payload)
        assert payload["model"] == "google/embeddinggemma-2"
        assert payload["add_special_tokens"] is True
        assert "dimensions" not in payload
        assert "input" not in payload
        assert payload["messages"] == [{"role": "user", "content": parts}]
        return httpx2.Response(
            200, json={"data": [{"index": 0, "embedding": [1.0, *([0.0] * 767)]}]}
        )

    with httpx2.Client(transport=httpx2.MockTransport(respond)) as http_client:

        def client(**values: object) -> OpenAI:
            assert values["base_url"] == "http://127.0.0.1:8000/v1"
            assert values["api_key"] == "EMPTY"
            return OpenAI(
                base_url="http://127.0.0.1:8000/v1",
                api_key="EMPTY",
                http_client=http_client,
                max_retries=0,
            )

        monkeypatch.setattr(recipes, "_openai_client", client)
        with closing(build_memory_composition(config)) as composition:
            embedder = composition.plugins.embedder
            assert embedder.embedding_capabilities == frozenset(
                {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
            )
            assert embedder.embedding_dimension == 768
            vectors = embedder.embed(
                (ModelInput(text="Remember this.", assets=tuple(assets)),), task=task
            )

    assert len(requests) == 1
    assert len(vectors[0]) == 768
