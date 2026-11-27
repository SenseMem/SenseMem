from __future__ import annotations

import pytest
from pydantic import ValidationError

from sensemem.benchmarks.model_config import HarnessOverrides, ModelConfig


@pytest.mark.parametrize(
    "url",
    (
        "embed.example.test",
        "//embed.example.test/v1",
        "ftp://embed.example.test/v1",
        "http://",
        "https://model host/v1",
    ),
)
def test_model_config_rejects_non_absolute_http_generation_url(url: str) -> None:
    with pytest.raises(ValueError, match="absolute http"):
        ModelConfig(generation_base_url=url)


@pytest.mark.parametrize(
    "url",
    ("http://models.example.test/v1", "https://127.0.0.1:8000/v1"),
)
def test_model_config_accepts_absolute_http_generation_url(url: str) -> None:
    assert ModelConfig(generation_base_url=url).generation_base_url == url


def test_harness_rejects_invalid_metrics_urls_and_unknown_performance_budgets() -> None:
    with pytest.raises(ValidationError, match="absolute HTTP"):
        HarnessOverrides.model_validate(
            {"server_metrics": {"generation_url": "models.example.test/metrics"}}
        )
    with pytest.raises(ValidationError, match="unknown performance budget"):
        HarnessOverrides.model_validate({"performance_budgets": {"mystery": 0.1}})


def test_harness_accepts_named_nonnegative_performance_budgets() -> None:
    overrides = HarnessOverrides.model_validate(
        {
            "performance_budgets": {
                "answer_e2e_ttft_p95": 0.1,
                "retrieval_e2e_latency_p95": 0.2,
            },
            "run": {"repeat_index": 2},
        }
    )

    assert overrides.performance_budgets == {
        "answer_e2e_ttft_p95": 0.1,
        "retrieval_e2e_latency_p95": 0.2,
    }
    assert overrides.run.repeat_index == 2
