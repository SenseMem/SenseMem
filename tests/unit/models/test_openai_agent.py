"""Real SDK request serialization against a local transport, without model services."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from traceback import format_exception
from typing import cast

import httpx2
import pytest
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from sensemem.content import AssetRef, ContentPart
from sensemem.exceptions import ModelError, ModelOutputTruncatedError
from sensemem.kernel.agent import AgentRunner
from sensemem.models.base import ModelInput
from sensemem.models.openai_agent import OpenAIAgent
from sensemem.prompts.agent import DEFAULT_AGENT_INSTRUCTIONS
from sensemem.tools import Tool, ToolRegistry
from sensemem.types import Modality


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    asset_id: str
    metadata: dict[str, object] = Field(default_factory=dict)


class _Output(BaseModel):
    text: str


def _response(message: dict[str, object], *, finish: str = "stop") -> dict[str, object]:
    return {
        "id": "chatcmpl-local-agent",
        "object": "chat.completion",
        "created": 0,
        "model": "local-vlm-revision",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }


def _client(handler: Callable[[httpx2.Request], httpx2.Response]) -> OpenAI:
    return OpenAI(
        api_key="local-test-key",
        base_url="https://agent.example.test/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )


def test_vlm_tool_call_results_and_media_survive_the_sdk_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"local-image-fixture")
    asset = AssetRef(
        id="image-1",
        modality=Modality.IMAGE,
        media_type="image/png",
        path=path,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    input = ModelInput(
        text="Read this image.",
        assets=(asset,),
        parts=(ContentPart("text", "Read this image."), ContentPart("asset", asset.id)),
    )
    requests: list[dict[str, object]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(cast(dict[str, object], json.loads(request.content)))
        if len(requests) == 1:
            return httpx2.Response(
                200,
                json=_response(
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-ocr",
                                "type": "function",
                                "function": {
                                    "name": "read_image",
                                    "arguments": '{"asset_id":"image-1","metadata":{"sections":[1,2]}}',
                                },
                            }
                        ],
                    },
                    finish="tool_calls",
                ),
            )
        return httpx2.Response(200, json=_response({"content": "The image says seven."}))

    seen: list[str] = []

    def read_image(value: _Input) -> _Output:
        seen.append(value.asset_id)
        return _Output(text="seven")

    registry = ToolRegistry(
        (Tool("read_image", "Read the supplied image.", _Input, _Output, read_image),)
    )
    with _client(handle) as client:
        backend = OpenAIAgent(
            client,
            model="local-vlm-revision",
            capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            max_tokens=500,
        )
        result = AgentRunner(backend, registry).run(input)
        backend.close()
        assert not client.is_closed()

    assert result.content == "The image says seven."
    assert seen == ["image-1"]
    assert all(request["store"] is False for request in requests)
    assert requests[0]["max_completion_tokens"] == 500
    first = cast(list[dict[str, object]], requests[0]["messages"])
    assert first[0] == {"role": "system", "content": DEFAULT_AGENT_INSTRUCTIONS}
    content = cast(list[dict[str, object]], first[1]["content"])
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert cast(dict[str, str], content[1]["image_url"])["url"].startswith("data:image/png;base64,")
    second = cast(list[dict[str, object]], requests[1]["messages"])
    assert second[1] == first[1]
    assert second[2]["role"] == "assistant"
    calls = cast(list[dict[str, object]], second[2]["tool_calls"])
    function = cast(dict[str, str], calls[0]["function"])
    assert json.loads(function["arguments"])["metadata"] == {"sections": [1, 2]}
    assert second[3]["role"] == "tool"
    assert second[3]["tool_call_id"] == "call-ocr"
    assert json.loads(cast(str, second[3]["content"]))["output"] == {"text": "seven"}


def test_sdk_tool_error_is_returned_before_a_corrected_call() -> None:
    requests: list[dict[str, object]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(cast(dict[str, object], json.loads(request.content)))
        if len(requests) < 3:
            arguments = '{"asset_id":7}' if len(requests) == 1 else '{"asset_id":"sample"}'
            return httpx2.Response(
                200,
                json=_response(
                    {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{len(requests)}",
                                "type": "function",
                                "function": {"name": "read", "arguments": arguments},
                            }
                        ],
                    },
                    finish="tool_calls",
                ),
            )
        return httpx2.Response(200, json=_response({"content": "recovered"}))

    executed: list[str] = []

    def read(value: _Input) -> _Output:
        executed.append(value.asset_id)
        return _Output(text="recovered")

    registry = ToolRegistry((Tool("read", "Read the source.", _Input, _Output, read),))
    with _client(handle) as client:
        result = AgentRunner(OpenAIAgent(client, model="revision"), registry).run(
            ModelInput(text="Read the source.")
        )
    assert result.content == "recovered"
    assert executed == ["sample"]
    messages = cast(list[dict[str, object]], requests[1]["messages"])
    error = json.loads(cast(str, messages[-1]["content"]))["error"]
    assert error["code"] == "invalid_arguments"


@pytest.mark.parametrize(
    "message,finish,reason",
    [
        ({"content": None, "refusal": "private refusal"}, "stop", "model_refused"),
        ({"content": "partial"}, "length", "output_truncated"),
        ({"content": "blocked"}, "content_filter", "model_refused"),
        (
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "[]"},
                    }
                ],
            },
            "tool_calls",
            "response_invalid",
        ),
        ({"content": ""}, "stop", "response_invalid"),
    ],
)
def test_native_failure_classification(
    message: dict[str, object], finish: str, reason: str
) -> None:
    with (
        _client(
            lambda request: httpx2.Response(200, json=_response(message, finish=finish))
        ) as client,
        pytest.raises(ModelError) as failure,
    ):
        OpenAIAgent(client, model="model").step(ModelInput(text="Question"), tools=(), history=())
    assert failure.value.reason == reason
    assert "private refusal" not in str(failure.value)
    assert isinstance(failure.value, ModelOutputTruncatedError) == (reason == "output_truncated")


@pytest.mark.parametrize(
    "code,reason",
    [("insufficient_quota", "quota_exhausted"), ("rate_limit_exceeded", "rate_limited")],
)
def test_provider_errors_use_the_sdk_taxonomy(code: str, reason: str) -> None:
    def fail(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429, json={"error": {"message": "private provider body", "type": code, "code": code}}
        )

    with _client(fail) as client, pytest.raises(ModelError) as failure:
        OpenAIAgent(client, model="model").step(ModelInput(text="Question"), tools=(), history=())
    assert failure.value.reason == reason
    assert "private provider body" not in "".join(format_exception(failure.value))
