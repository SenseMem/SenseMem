"""A typed answer's citations do not measure the evidence actually supplied to its model."""

import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import httpx2 as httpx
import pytest
import yaml
from openai import OpenAI

from sensemem import OpenAIModels
from sensemem.configuration import OpenAIGenerationConfig


def test_reader_probe_measures_three_inputs_when_answer_cites_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "embedding": {"provider": "openai"},
                "generation": {
                    "provider": "openai",
                    "model": "deepseek-v4.1-flash",
                    "api_key": "fixture-key",
                    "answer_format": "structured",
                    "answer_style": "concise",
                },
                "benchmark": {"run": {"tasks": "atm-bench-main-sgm", "download": False}},
            }
        )
    )
    functions = runpy.run_path(
        str(Path(__file__).parents[3] / "benchmarks/atm_reader_role_probe.py")
    )
    run = cast(Callable[[Path, Path], int], functions["run_probe"])
    make_case = cast(
        Callable[[str, tuple[tuple[str, str], ...], str, str], object], functions["Case"]
    )
    case = make_case(
        "subset_citations",
        (
            ("first", "The first box is blue."),
            ("second", "The second box is red."),
            ("third", "The third box is blue."),
        ),
        "List the source IDs that mention blue boxes. Return only comma-separated IDs.",
        r"(?:first\s*,\s*third|third\s*,\s*first)",
    )
    monkeypatch.setitem(run.__globals__, "CASES", (case,))
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert body["temperature"] == 0 and body["seed"] == 0
        assert "max_tokens" not in body and "max_completion_tokens" not in body
        payload = json.loads(body["messages"][1]["content"])
        assert len(payload["hits"]) == 3
        labels = {hit["metadata"]["source_id"]: hit["evidence_label"] for hit in payload["hits"]}
        draft = {
            "draft": {"kind": "source_ids", "evidence_labels": [labels["first"], labels["third"]]}
        }
        if body.get("stream"):
            chunk = {
                "model": "deepseek-v4.1-flash",
                "choices": [
                    {"index": 0, "delta": {"content": json.dumps(draft)}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4.1-flash",
                "choices": [
                    {"index": 0, "message": {"content": json.dumps(draft)}, "finish_reason": "stop"}
                ],
            },
        )

    directory = tmp_path / "probe"
    directory.mkdir()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        sdk = OpenAI(
            api_key="fixture-key",
            base_url="https://models.invalid/v1",
            http_client=client,
            max_retries=0,
        )

        def generation(config: OpenAIGenerationConfig) -> OpenAIModels:
            return OpenAIModels(
                sdk,
                generation_model=config.model,
                generation_answer_format=config.answer_format,
                generation_answer_style=config.answer_style,
                generation_temperature=config.temperature,
                generation_seed=config.seed,
            )

        monkeypatch.setitem(run.__globals__, "_build_generation", generation)
        # This one-case regression intentionally omits the suite's separate planning fixture.
        assert run(config, directory) == 1
    assert len(requests) == 1
    report = json.loads((directory / "reader-probe.json").read_text())
    result = report["results"][0]
    assert result["passed"] and result["all_sources_supplied"] and result["request_observed"]
    assert result["supplied_record_count"] == 3 and result["returned_record_count"] == 2
