"""Proactive protocol fixtures from PersonaMem-v3's pinned runner and judge."""

from __future__ import annotations

import json

import pytest

from sensemem.benchmarks._official import personamem_v3_proactive as proactive


def _response(*, should_act: bool | str, content: str = "Want to check Ada's message?") -> str:
    return json.dumps(
        {
            "should_act": should_act,
            "action_class": "friend_alert" if should_act is True else "stay_silent",
            "content": content,
            "evidence_cited": "Ada's unread message",
            "reasoning": "Ada is a close friend, and the message just arrived.",
        }
    )


def _judge(**overrides: object) -> dict[str, object]:
    return {
        "trigger_detection_correctness": 10,
        "justification_quality": 10,
        "preference_alignment": 10,
        "avoid_overpersonalization": 10,
        "voice_match": 10,
        "restraint_justification": 10,
        "negative_leakage": False,
        "stale_preference_use": False,
        **overrides,
    }


def test_proactive_prompt_does_not_disclose_instance_ground_truth() -> None:
    evidence: dict[str, object] = {
        "expected_behavior": "HIDDEN_POLARITY",
        "trigger_evidence": {"friend_id": "HIDDEN_FRIEND"},
        "jitai_card": {"reasoning": "HIDDEN_REASON"},
        "profile": {"name": "HIDDEN_PROFILE"},
        "user_query": "HIDDEN_BROWSING_LABEL",
    }

    prompt = proactive.build_prompt(evidence)

    assert "HIDDEN_" not in prompt
    assert "The user has NOT just sent" in prompt
    assert '"should_act": <bool>' in prompt
    assert "Do NOT attempt to call any tools" in prompt


def test_proactive_uses_instance_polarity_instead_of_the_broad_row_label() -> None:
    # Real released rows say personalize/proactive_assist on the outside; the
    # official runner reads act/restrain from instance_full instead.
    instance: dict[str, object] = {
        "expected_behavior": "restrain",
        "trigger_evidence": {"friend_id": "friend_5"},
        "jitai_card": {"recommended_action_class": "stay_silent"},
    }
    prediction = _response(should_act=False, content="")

    with pytest.raises(ValueError, match="act/restrain"):
        proactive.score(prediction, {"expected_behavior": "personalize"})
    assert proactive.score(prediction, instance)["decision_score"] == 1
    prompt = proactive.build_judge_prompt(prediction, instance)
    assert "expected_behavior = `restrain`" in prompt
    assert '"friend_id": "friend_5"' in prompt
    assert '"recommended_action_class": "stay_silent"' in prompt


@pytest.mark.parametrize("prediction", ["no action", "{}", "[]", "{invalid"])
def test_invalid_proactive_json_keeps_raw_output_and_defaults_to_no_action(prediction: str) -> None:
    parsed = proactive.parse_response(prediction)

    assert parsed == {
        "should_act": False,
        "content": prediction,
        "reasoning": "(failed_to_parse)",
    }
    assert proactive.score(prediction, {"expected_behavior": "act"})["decision_score"] == 0
    assert proactive.score(prediction, {"expected_behavior": "restrain"})["decision_score"] == 1


def test_proactive_reads_fenced_json_and_does_not_treat_string_false_as_true() -> None:
    prediction = f"```json\n{_response(should_act='false')}\n```"

    result = proactive.score(prediction, {"expected_behavior": "restrain"}, _judge())

    assert result["content_word_count"] > 0
    assert result["decision_score"] == 1
    assert result["proactive_action_score"] == 1


@pytest.mark.parametrize(
    ("behavior", "should_act", "judge", "expected"),
    [
        ("act", True, _judge(), 1.0),
        ("act", False, _judge(), 0.3),
        ("restrain", True, _judge(), 0.3),
        (
            "restrain",
            False,
            _judge(
                justification_quality=0,
                preference_alignment=0,
                avoid_overpersonalization=0,
                voice_match=0,
                restraint_justification=8,
            ),
            0.94,
        ),
        ("restrain", False, _judge(restraint_justification=0), 0.0),
        ("act", True, _judge(negative_leakage=True), 0.0),
        ("restrain", False, _judge(stale_preference_use="true"), 0.0),
    ],
)
def test_proactive_combines_decision_rationale_and_hard_rules(
    behavior: str, should_act: bool, judge: dict[str, object], expected: float
) -> None:
    result = proactive.score(
        _response(should_act=should_act), {"expected_behavior": behavior}, judge
    )

    assert result["proactive_action_score"] == pytest.approx(expected)


def test_proactive_no_judge_fallback_and_length_penalty_match_runner() -> None:
    prediction = _response(should_act=True, content=" ".join(["word"] * 31))

    fallback = proactive.score(prediction, {"expected_behavior": "act"})
    judged = proactive.score(prediction, {"expected_behavior": "act"}, _judge())

    assert fallback["content_length_ok"] == 0
    assert fallback["proactive_action_score"] == pytest.approx(0.805)
    assert judged["proactive_action_score"] == pytest.approx(0.91)
    no_rationale = '{"should_act": false, "content": ""}'
    restrained = proactive.score(no_rationale, {"expected_behavior": "restrain"})
    assert restrained["proactive_action_score"] == pytest.approx(0.7)


def test_proactive_partial_judge_keeps_gates_before_falling_back() -> None:
    prediction = _response(should_act=True)

    incomplete = proactive.score(prediction, {"expected_behavior": "act"}, {})
    gated = proactive.score(prediction, {"expected_behavior": "act"}, {"negative_leakage": True})

    assert incomplete["proactive_action_score"] == 1.0
    assert "justification_score" not in incomplete
    assert gated["proactive_action_score"] == 0.0
