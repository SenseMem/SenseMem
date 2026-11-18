"""Opt-in affect, attribution, and preference gate against a caller-identified real model."""

import json
import os

import pytest
from openai import OpenAI

from sensemem import Modality
from sensemem.benchmarks.affective_quality import DEFAULT_MODEL, run_affective_quality_eval
from sensemem.models.openai_completion import OpenAICompletion

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SENSEMEM_AFFECTIVE_EVAL") != "1" or not os.environ.get("OPENAI_API_KEY"),
        reason="requires SENSEMEM_AFFECTIVE_EVAL=1 and OPENAI_API_KEY",
    ),
]


def test_affective_memory_quality() -> None:
    with OpenAI(timeout=120, max_retries=0) as client:
        models = OpenAICompletion(
            client,
            generation_model=os.environ.get("SENSEMEM_AFFECTIVE_EVAL_MODEL", DEFAULT_MODEL),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_max_tokens=None,
            generation_extra_body={"reasoning_effort": "none"},
            generation_response_format="json_schema",
        )
        report = run_affective_quality_eval(
            models,
            repeats=3,
            model_revision=os.environ.get(
                "SENSEMEM_AFFECTIVE_MODEL_REVISION", "unknown: provider model alias"
            ),
        )
    print(json.dumps(report, ensure_ascii=False))
    assert report["passed"], "synthetic affective quality fixture failed; inspect safe case checks"
