"""Keep extracted benchmark instructions byte-identical to their existing pinned protocols."""

import hashlib
from typing import Literal

import pytest

from sensemem.benchmarks._official import judge_prompts, query_prompts
from sensemem.benchmarks.baseline_prompts import baseline_system_prompt
from sensemem.benchmarks.longmemeval_v2 import LongMemEvalV2Question
from sensemem.benchmarks.videozerobench import (
    SpatialEvidence,
    TemporalEvidence,
    VideoZeroQuestion,
)
from sensemem.types import AnswerPolicy


@pytest.mark.parametrize("has_context", [False, True])
@pytest.mark.parametrize("answer_policy", ["strict", "best_effort"])
@pytest.mark.parametrize("answer_style", ["default", "concise"])
def test_baseline_system_builder_composes_policy_style_and_context_instructions(
    has_context: bool,
    answer_policy: AnswerPolicy,
    answer_style: Literal["default", "concise"],
) -> None:
    prompt = baseline_system_prompt(
        has_context=has_context,
        override=None,
        answer_policy=answer_policy,
        answer_style=answer_style,
    )
    assert prompt
    assert ("untrusted data" in prompt) is has_context
    assert ("no access to the user's memories" in prompt) is (not has_context)
    if answer_policy == "strict":
        assert "do not decline" not in prompt.casefold()
    else:
        assert "do not decline" in prompt.casefold() or "best" in prompt.casefold()
    concise_prompt = baseline_system_prompt(
        has_context=has_context,
        override=None,
        answer_policy=answer_policy,
        answer_style="concise",
    )
    default_prompt = baseline_system_prompt(
        has_context=has_context,
        override=None,
        answer_policy=answer_policy,
        answer_style="default",
    )
    assert concise_prompt != default_prompt


def test_baseline_system_builder_respects_literal_override() -> None:
    assert (
        baseline_system_prompt(
            has_context=False,
            override="",
            answer_policy="strict",
            answer_style="default",
        )
        == ""
    )
    overridden = baseline_system_prompt(
        has_context=True,
        override="OVERRIDE {literal}",
        answer_policy="best_effort",
        answer_style="default",
    )
    assert overridden.startswith("OVERRIDE {literal}")
    assert "untrusted data" in overridden


@pytest.mark.parametrize(
    ("prompt", "digest"),
    [
        (
            judge_prompts._LOCOMO_PROMPT,
            "899d035bd61813bc718812b2997f5908111c63ccd2ed76533e65f5dc3ca337a4",
        ),
        (
            judge_prompts._M3_PROMPT,
            "88543b6433f3d46c3cd8621b6e0837b8a8c406310510279d87021c283a13acaa",
        ),
        (
            judge_prompts._EGOTEMPO_PROMPT,
            "99da958bd7467147196f781982eb5b0e14f17864df8fc1bcadf620d3f8310fd8",
        ),
        (
            judge_prompts._MM_LIFELONG_SYSTEM,
            "effeeba5b3200fbae47699b5652e04afa2451337e61f9b5dc1e850ff37d3c231",
        ),
        (
            judge_prompts._ATM_PROMPT,
            "d5669b9effea50bad2abbed3e141a4710a1731d1cccc93c211a06560448a2210",
        ),
        (
            judge_prompts._GALLERY_PROMPT,
            "d2efdbd896f38e77644522d1c773e52d6921c24b72f17df376a5dc7d918de734",
        ),
    ],
)
def test_judge_components_preserve_pinned_protocol_bytes(prompt: str, digest: str) -> None:
    assert hashlib.sha256(prompt.encode()).hexdigest() == digest


@pytest.mark.parametrize(
    ("domain", "digest"),
    [
        ("web", "d7dc1e6ba33ab8972c2076309efeed0a82e236b4f353ab075122d4cd4b151dad"),
        ("enterprise", "f4b83f8340c34f559877cb3943e4d89088db68b575fcd2e809229970f1ea5ff3"),
    ],
)
def test_longmemeval_query_keeps_the_domain_and_literal_question(
    domain: Literal["web", "enterprise"], digest: str
) -> None:
    question = LongMemEvalV2Question(
        id="q1",
        domain=domain,
        environment="fixture",
        question_type="static-environment",
        question="Where is the {blue} chair?",
        image=None,
        answer="Hidden gold must not enter the prompt",
        eval_function="norm_phrase_set_match",
    )
    prompt = query_prompts.longmemeval_v2_question_prompt(question)
    assert hashlib.sha256(prompt.encode()).hexdigest() == digest


@pytest.mark.parametrize(
    ("language", "digests"),
    [
        (
            "en",
            (
                "c62073b3dd0c51de04d9f2e00e83da1ce5f9e94ede73e9ba147c10efa42df8d9",
                "84f5c3eace478089f65bdc85685eb74a553dac379faee826fa8abb97a395a1da",
                "a98d851bd2a1ed6c68d6003ccd9404cfc992baa432c46de78093b0231eb159e2",
                "bcc3684d7b41279824cef1760943027c99a27b881dd863cee611d264fb512511",
                "8c2ff227452e9cf0b245b487fbdb1cb34ebb9a99d522747a1c647695b6efc072",
            ),
        ),
        (
            "cn",
            (
                "4c557411015a5d16de720037dfa9b74e031068a7ead59b2a6d81c7f7c1e5e6e0",
                "a321a73dc68c82f387388ed4f729f5224ec85b73aabd4af4c4ecd8f1c8bfa0c8",
                "4405860bf3917457f8a544f0fd7d58a5f5934ccb809c5c778208b44806cbfc8c",
                "bcc3684d7b41279824cef1760943027c99a27b881dd863cee611d264fb512511",
                "8c2ff227452e9cf0b245b487fbdb1cb34ebb9a99d522747a1c647695b6efc072",
            ),
        ),
    ],
)
def test_videozero_queries_preserve_each_level_and_language(
    language: Literal["en", "cn"], digests: tuple[str, ...]
) -> None:
    question = VideoZeroQuestion(
        question_id=1,
        question="Where is the {blue} chair?",
        answer="Hidden gold must not enter the prompt",
        evidence_windows=(TemporalEvidence(start=2.0, end=4.0),),
        evidence_boxes=(SpatialEvidence(time=2.5, box=(0.1, 0.2, 0.5, 0.6)),),
        video="clip.mp4",
        video_id="clip",
        category="Daily Vlogs",
        language=language,
        duration=10.0,
        evidence_span="single-frame",
        annotation_capabilities=("counting",),
    )
    queries = query_prompts.videozero_level_queries(question)
    assert tuple(level for level, _ in queries) == (1, 2, 3, 4, 5)
    assert tuple(hashlib.sha256(prompt.encode()).hexdigest() for _, prompt in queries) == digests
