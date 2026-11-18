"""Remote readers receive verified original media through REST and MCP resources."""

from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from fastapi.testclient import TestClient
from mcp.types import InputRequiredResult

from sensemem import Blob, Memory, MemoryNotFoundError, StorageError, ValidationError
from sensemem.api.app import create_app
from sensemem.api.mcp import build_mcp_server


async def test_source_media_can_be_read_by_sdk_http_and_mcp(tmp_path: Path) -> None:
    source = Blob(b"original source bytes", "audio/wav", "recording.wav")
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add(source)
        asset_id = record.assets[0].id
        assert memory.read_asset(record.id, asset_id) == source
        with TestClient(create_app(memory=memory)) as client:
            fetched = client.get(f"/v1/memories/{record.id}")
            assert fetched.json()["parts"] == [{"kind": "asset", "value": asset_id}]
            response = client.get(f"/v1/memories/{record.id}/assets/{asset_id}")
            assert response.status_code == 200
            assert response.content == source.data
            assert response.headers["content-type"] == source.media_type
        server = build_mcp_server(memory)
        templates = await server.list_resource_templates()
        assert any(str(item.uri_template).startswith("sensemem://memories/") for item in templates)
        contents = await server.read_resource(f"sensemem://memories/{record.id}/assets/{asset_id}")
        assert not isinstance(contents, InputRequiredResult)
        assert next(iter(contents)).content == source.data
        with pytest.raises(ValidationError, match="does not belong"):
            memory.read_asset(record.id, "other-asset")
        memory.delete(record.id)
        with pytest.raises(MemoryNotFoundError):
            memory.read_asset(record.id, asset_id)


def test_source_media_read_rejects_changed_original_bytes(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        record = memory.add(Blob(b"original bytes", "audio/wav"))
        assert record.assets[0].path is not None
        Path(record.assets[0].path).write_bytes(b"tampered bytes")
        with pytest.raises(StorageError) as raised:
            memory.read_asset(record.id, record.assets[0].id)
        assert raised.value.reason == "asset_changed"
