"""Opt-in real planner and local embedding qualification of goal-directed context."""

import json
import os

import pytest
from openai import OpenAI

from sensemem.benchmarks.context_quality import run_context_quality_eval
from sensemem.models.openai_completion import OpenAICompletion
from sensemem.models.openai_embedding import OpenAIEmbedder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SENSEMEM_CONTEXT_EVAL") != "1" or not os.environ.get("OPENAI_API_KEY"),
        reason="requires SENSEMEM_CONTEXT_EVAL=1 and OPENAI_API_KEY",
    ),
]


def test_goal_context_quality() -> None:
    model = os.environ.get("SENSEMEM_CONTEXT_EVAL_MODEL", "qwen3.8-omni-flash")
    reasoning_controls = (
        {"reasoning_effort": "none"} if model.rsplit("/", 1)[-1] == "qwen3.8-omni-flash" else None
    )
    with (
        OpenAI(timeout=120, max_retries=0) as generation,
        OpenAI(
            api_key="local",
            base_url=os.environ.get("SENSEMEM_EMBEDDING_BASE_URL", "http://127.0.0.1:8000/v1"),
            timeout=120,
            max_retries=0,
        ) as embedding,
    ):
        planner = OpenAICompletion(
            client=generation,
            generation_model=model,
            generation_max_tokens=None,
            generation_response_format="json_schema",
            generation_extra_body=reasoning_controls,
        )
        embedder = OpenAIEmbedder(
            client=embedding,
            embedding_model="google/embeddinggemma-2",
            embedding_dimension=768,
            embedding_request_format="messages",
        )
        report = run_context_quality_eval(
            planner, embedder, repeats=3, model_revision=planner.generation_model
        )
    report["generation_controls"] = {
        "reasoning_effort": "none" if reasoning_controls else "provider_default",
        "response_format": "json_schema",
        "output_limits_omitted": True,
    }
    report["client_timeout_seconds"] = 120
    print(json.dumps(report, ensure_ascii=False))
    assert report["passed"], "goal context quality fixture failed; inspect safe case checks"
