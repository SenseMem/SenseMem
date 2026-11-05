"""Caller omission applies to both planning and final answering requests."""

import json

import httpx2 as httpx
import pytest
from test_openai_sdk import NOW, _sdk_client, _search_hit

from sensemem import ModelInput, OpenAIModels


@pytest.mark.parametrize("maximum_effort", [False, True])
def test_deepseek_default_budget_is_omitted_for_planning_and_answering(
    maximum_effort: bool,
) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        payload = json.loads(request.content)
        assert "max_tokens" not in payload
        assert "max_completion_tokens" not in payload
        if maximum_effort:
            assert payload["thinking"] == {"type": "enabled"}
            assert payload["reasoning_effort"] == "max"
        content = (
            '{"shape":"point","steps":[{"op":"similar","query":"toolbox","k":1}]}'
            if calls == 0
            else "Blue."
        )
        calls += 1
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4.1-flash",
                "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_model="deepseek-v4.1-flash",
            generation_extra_body=(
                {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}
                if maximum_effort
                else None
            ),
        )
        try:
            assert (
                model.plan_recall(
                    "What color is the toolbox?", reference_at=NOW, corpus_digest="1 text record"
                )
                is not None
            )
            assert (
                model.answer(ModelInput(text="What color is the toolbox?"), (_search_hit(),)).answer
                == "Blue."
            )
        finally:
            model.close()
    assert calls == 2
