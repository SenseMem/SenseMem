"""Deterministic PersonaMem-v3 formula regressions from the pinned evaluator."""

from __future__ import annotations

import hashlib

import pytest

from sensemem.benchmarks._official import personamem_v3_scoring as scoring
from sensemem.benchmarks._official.personamem_v3_scoring import compute_geo_shift_metrics


def test_personamem_relationship_penalty_preserves_prompt_and_score() -> None:
    task_type = "agentic_auto_reply"
    prompt = scoring.build_unified_rubric_prompt(
        task_type,
        "{}",
        "A reply.",
        scoring.positive_dims(task_type),
        scoring.hard_rule_dims(task_type),
        scoring.penalty_dims(task_type),
        query_text="Write a reply.",
    )
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "4ad0c0b3f5ee1c83a8604c86aa21fda41c44bcfcd3793c446711efe048110ee9"
    )
    scores = scoring.score_unified_rubric(
        task_type, {"voice_match": 8, "telegraph_avoidance": 10, "relationship_aware": 6}
    )
    assert scores["pr_relationship_aware_score"] == 6.0
    assert scores["pr_penalty_points"] == 2.0
    assert scores["pr_combined_personalization_score"] == 6.0


@pytest.mark.parametrize(
    ("location", "current_city", "prior_city", "current_region", "grounded", "stale", "score"),
    [
        ("denver", "Denver", "Dallas", "CO", 1.0, 0.0, 1.0),
        ("CO", "Denver", "Dallas", "CO", 1.0, 0.0, 1.0),
        ("Denver and Dallas", "Denver", "Dallas", "CO", 1.0, 1.0, 0.0),
        ("Dallas", "Denver", "Dallas", "CO", 0.0, 1.0, 0.0),
        ("the coast", "Denver", "Dallas", "CO", 0.0, 0.0, 0.5),
        ("St. John's", " St. John's ", "Dallas", "", 1.0, 0.0, 1.0),
        ("nearby", "", "", "", 0.0, 0.0, 0.5),
    ],
)
def test_geo_shift_scores_each_leg_with_stale_city_as_hard_fail(
    location: str,
    current_city: str,
    prior_city: str,
    current_region: str,
    grounded: float,
    stale: float,
    score: float,
) -> None:
    response = (
        f"Around {location}, try climbing, outdoor running, swimming, cycling, tennis, "
        "basketball, or yoga. These activities offer different settings, equipment, "
        "intensity, and schedules depending on your mood."
    )
    assert compute_geo_shift_metrics(response, current_city, prior_city, current_region) == {
        "current_city_grounded": grounded,
        "stale_geo_anchor": stale,
        "geo_neutral_response": float(not grounded and not stale),
        "geo_shift_correctness": score,
    }


@pytest.mark.parametrize(
    "response",
    [
        "",
        "Try Denver.",
        "Denver " * 20,
        "one two six ten red tan dog cat bat rat fox hen cow pig A AB 中文",
    ],
)
def test_geo_shift_rejects_fewer_than_fifteen_distinct_ascii_tokens(response: str) -> None:
    assert compute_geo_shift_metrics(response, "Denver", "Dallas") == {
        "current_city_grounded": 0.0,
        "stale_geo_anchor": 0.0,
        "geo_neutral_response": 0.0,
        "geo_shift_correctness": 0.0,
        "non_substantive_response": 1.0,
    }


def test_geo_shift_accepts_exactly_fifteen_distinct_ascii_tokens() -> None:
    response = "one two six ten red tan dog cat bat rat fox hen cow pig #new"
    assert compute_geo_shift_metrics(response, "Denver", "Dallas")["geo_shift_correctness"] == 0.5
