"""The affect fixture scores SDK behavior and does not award credit for a model call alone."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import httpx2 as httpx
import pytest
from openai import OpenAI

from sensemem import Modality
from sensemem.benchmarks import affective_quality
from sensemem.models.openai_completion import OpenAICompletion


def _responder(  # noqa: C901 - one fixture branch per independent quality case
    broken: str | None,
) -> Callable[[httpx.Request], httpx.Response]:
    trait_calls = 0

    def respond(  # noqa: C901 - mirror the fixture's explicitly different source inputs
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal trait_calls
        payload = json.loads(request.content)
        assert "max_tokens" not in payload and "max_completion_tokens" not in payload
        content = payload["messages"][1]["content"]
        source = (
            json.loads(content)["observations"][0]
            if isinstance(content, str)
            else json.loads(content[0]["text"])["observation"]
        )
        text = source["content"]
        case = next(case for case in affective_quality.CASES if text in case.texts)

        def affect(subject: str, valence: float, modality: str = "text") -> dict[str, object]:
            return {
                "kind": "affect",
                "content": f"{subject}'s reported affect in {modality}",
                "subject": subject,
                "value": "anxious" if valence < 0 else "calm",
                "cue_modality": modality,
                "valence": valence,
                "arousal": 0.5,
                "confidence": 0.9,
                "evidence_observation_ids": ["observation_0"],
            }

        def trait(
            value: str, *, statement: bool = False, subject: str | None = None
        ) -> dict[str, object]:
            subject = case.subjects[0] if subject is None else subject
            return {
                "kind": "trait",
                "content": text if statement else f"{subject} has a {value} disposition.",
                "subject": subject,
                "predicate": "preferred_drink" if statement else "disposition",
                "value": value,
                "confidence": 0.9,
                "evidence_observation_ids": ["observation_0"],
            }

        if case.family in {
            "trait_separate",
            "trait_negative",
            "trait_instruction",
            "trait_withdrawn",
        }:
            second = text == case.texts[1]
            if not second:
                proposals = [trait(case.values[0])]
            elif case.family == "trait_separate":
                proposals = [
                    trait(
                        case.values[0] if broken else case.values[-1],
                        subject=case.subjects[0] if broken else case.subjects[-1],
                    )
                ]
            elif broken:
                proposals = [
                    trait(
                        "invented permanent disposition"
                        if case.family == "trait_withdrawn"
                        else case.values[0]
                    )
                ]
            else:
                proposals = []
        elif case.family == "affect_link_correction" and text == case.texts[1]:
            history = source["read_only_history"]
            feeling = next(
                row["observation_id"]
                for row in history
                if row.get("context") and row["context"]["kind"] == "affect"
            )
            launch = next(
                row["observation_id"]
                for row in history
                if row.get("context") and row["context"]["kind"] == "event"
            )
            witnesses = ["observation_0", *(row["observation_id"] for row in history)]
            proposals = (
                []
                if broken
                else [
                    {
                        "kind": "event",
                        "local_id": "hiding",
                        "subject": "Ana",
                        "content": "Bob hiding the decision from Ana.",
                        "confidence": 0.9,
                        "evidence_observation_ids": ["observation_0"],
                    },
                    *(
                        {
                            "kind": "relation",
                            "subject": "Ana",
                            "predicate": "affect_elicitor",
                            "value": word,
                            "content": text,
                            "confidence": 0.9,
                            "evidence_observation_ids": witnesses,
                            "affect_link": {
                                "affect_id": feeling,
                                "role": "elicitor",
                                "object_ref": {"memory_id": reference, "identity_id": None},
                                "polarity": polarity,
                            },
                            "affect_witnesses": [
                                {"source_id": "observation_0", "reporter": "Ana", "quote": text}
                            ],
                        }
                        for polarity, reference, word in (
                            ("denied", launch, "launch"),
                            ("affirmed", "proposal:hiding", "hiding"),
                        )
                    ),
                ]
            )
        elif case.family.startswith("affect_link_"):
            cue = affect(case.subjects[0], case.signs[0] * 0.5)
            cue["local_id"] = "feeling"
            proposals = [cue]
            no_link = case.family in {"affect_link_unknown", "affect_link_injection"}
            if (not no_link and (not broken or case.family == "affect_link_correction")) or (
                no_link and broken
            ):
                roles = (
                    [("target", "affirmed", "Bob"), ("elicitor", "affirmed", "launch")]
                    if case.family == "affect_link_self"
                    else [("elicitor", "affirmed", "launch"), ("elicitor", "denied", "criticized")]
                    if case.family == "affect_link_denied"
                    else [("elicitor", "affirmed", case.values[0] if case.values else "invented")]
                )
                reporter = (
                    case.subjects[1] if case.family == "affect_link_report" else case.subjects[0]
                )
                for index, (role, polarity, word) in enumerate(roles):
                    local_id = f"object{index}"
                    proposals.append(
                        {
                            "kind": "entity" if role == "target" else "event",
                            "local_id": local_id,
                            "subject": word if role == "target" else case.subjects[0],
                            "content": word,
                            "confidence": 0.9,
                            "evidence_observation_ids": ["observation_0"],
                        }
                    )
                    proposals.append(
                        {
                            "kind": "relation",
                            "subject": case.subjects[0],
                            "predicate": f"affect_{role}",
                            "value": word,
                            "content": text,
                            "confidence": 0.9,
                            "evidence_observation_ids": ["observation_0"],
                            "affect_link": {
                                "affect_id": "proposal:feeling",
                                "role": role,
                                "object_ref": {
                                    "memory_id": f"proposal:{local_id}",
                                    "identity_id": None,
                                },
                                "polarity": polarity,
                            },
                            "affect_witnesses": [
                                {"source_id": "observation_0", "reporter": reporter, "quote": text}
                            ],
                        }
                    )
        elif case.family == "attribution":
            proposals = [
                affect(subject, sign * (-0.5 if broken else 0.5))
                for subject, sign in zip(case.subjects, case.signs, strict=True)
            ]
        elif case.family == "disagreement":
            proposals = [affect(case.subjects[0], case.signs[0] * 0.5)]
            if not broken:
                proposals.append(affect(case.subjects[0], case.signs[1] * 0.5, "image"))
        elif case.family in {"transient", "correction"}:
            proposals = [affect(case.subjects[0], -0.5)]
            if broken:
                proposals.append(trait("permanent sadness"))
        elif case.family in {"trait", "same_capture"}:
            trait_calls += 1
            proposals = [trait("different" if broken and trait_calls > 1 else case.values[0])]
            if broken and case.family == "same_capture":
                proposals[0]["kind"] = "state"
        elif case.family == "preference":
            second = text == case.texts[1]
            proposals = [trait(case.values[int(second)], statement=True)]
            proposals[0]["kind"] = "state"
            proposals[0]["predicate"] = "different_property" if broken and second else "preference"
        elif case.family == "negation":
            proposals = [trait(case.values[0], statement=True)] if broken else []
        else:
            proposals = [affect(case.subjects[0], -0.5)]
            if broken:
                # No attack marker: a paraphrased invented trait must fail the gate too.
                proposals.append(trait("permanently volatile"))
        return httpx.Response(
            200,
            json={
                "model": "returned-affective-fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {"observation_id": "observation_0", "proposals": proposals}
                                    ]
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    return respond


def test_affective_gate_uses_fresh_stores_and_records_observed_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories: list[Path] = []

    def tracked_directory(*, prefix: str) -> TemporaryDirectory[str]:
        directory = TemporaryDirectory(prefix=prefix)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(affective_quality, "TemporaryDirectory", tracked_directory)
    with (
        httpx.Client(transport=httpx.MockTransport(_responder(None))) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        models = OpenAICompletion(
            client,
            generation_model="affective-fixture",
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_max_tokens=None,
        )
        report = affective_quality.run_affective_quality_eval(models, repeats=1)
    assert report["passed"]
    assert report["returned_models"] == ["returned-affective-fixture"]
    assert len(directories) == len(set(directories)) == len(affective_quality.CASE_IDS)
    assert all(not path.exists() for path in directories)


@pytest.mark.parametrize("case_id", affective_quality.CASE_IDS)
def test_affective_gate_refuses_each_incorrect_outcome(
    monkeypatch: pytest.MonkeyPatch, case_id: str
) -> None:
    monkeypatch.setattr(affective_quality, "CASE_IDS", (case_id,))
    with (
        httpx.Client(transport=httpx.MockTransport(_responder(case_id))) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        models = OpenAICompletion(
            client,
            generation_model="affective-fixture",
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_max_tokens=None,
        )
        report = affective_quality.run_affective_quality_eval(models, repeats=1)
    assert not report["passed"]


@pytest.mark.parametrize(
    ("field", "value", "dropped", "refused"),
    [("valence", 2, 1, 0), ("cue_modality", "audio", 0, 1)],
)
def test_affective_gate_distinguishes_parser_loss_from_kernel_refusal(
    field: str, value: object, dropped: int, refused: int
) -> None:
    valid = _responder(None)

    def respond(request: httpx.Request) -> httpx.Response:
        envelope = valid(request).json()
        payload = json.loads(envelope["choices"][0]["message"]["content"])
        payload["items"][0]["proposals"][0][field] = value
        envelope["choices"][0]["message"]["content"] = json.dumps(payload)
        return httpx.Response(200, json=envelope)

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        report = affective_quality.run_affective_quality_eval(
            OpenAICompletion(client, generation_model="fixture", generation_max_tokens=None),
            repeats=1,
            case_ids=("transient_affect",),
        )
    assert not report["passed"]
    cases = cast(list[dict[str, object]], report["cases"])
    assert cases[0]["parser_dropped"] == dropped
    assert cases[0]["kernel_refused"] == refused


def test_missing_second_candidate_reports_absence_after_source_withdrawal() -> None:
    valid = _responder(None)
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        envelope = valid(request).json()
        if calls == 2:
            payload = json.loads(envelope["choices"][0]["message"]["content"])
            payload["items"][0]["proposals"] = []
            envelope["choices"][0]["message"]["content"] = json.dumps(payload)
        return httpx.Response(200, json=envelope)

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        report = affective_quality.run_affective_quality_eval(
            OpenAICompletion(client, generation_model="fixture", generation_max_tokens=None),
            repeats=1,
            case_ids=("independent_trait",),
        )
    (case,) = cast(list[dict[str, object]], report["cases"])
    checks = cast(dict[str, bool], case["checks"])
    assert not report["passed"]
    assert not checks["independent_capture_promotes_same_claim"]
    assert not checks["withdrawn_source_removes_promotion"]
    diagnostics = cast(list[dict[str, object]], case["diagnostics"])
    withdrawn = next(row for row in diagnostics if row["phase"] == "after_first_source_deleted")
    assert withdrawn["record_counts"] == {"observation": 1}
    assert withdrawn["claims"] == []


@pytest.mark.parametrize(
    "mutation", ("target_instead_of_elicitor", "new_episode", "duplicate_event")
)
def test_attribution_revision_gate_rejects_wrong_roles_and_reference_identity(
    mutation: str,
) -> None:
    valid = _responder(None)
    case = next(case for case in affective_quality.CASES if case.family == "affect_link_correction")

    def respond(request: httpx.Request) -> httpx.Response:
        envelope = valid(request).json()
        source = json.loads(json.loads(request.content)["messages"][1]["content"])["observations"][
            0
        ]
        if source["content"] == case.texts[1]:
            payload = json.loads(envelope["choices"][0]["message"]["content"])
            proposals = payload["items"][0]["proposals"]
            if mutation == "new_episode":
                proposals.append(
                    {
                        "kind": "affect",
                        "local_id": "different_episode",
                        "subject": "Ana",
                        "content": "Ana feels angry in a different episode.",
                        "value": "angry",
                        "cue_modality": "text",
                        "valence": -0.5,
                        "arousal": 0.5,
                        "confidence": 0.9,
                        "evidence_observation_ids": ["observation_0"],
                    }
                )
            elif mutation == "duplicate_event":
                proposals.append(
                    {
                        "kind": "event",
                        "local_id": "different_launch",
                        "subject": "Ana",
                        "content": "The canceled launch.",
                        "confidence": 0.9,
                        "evidence_observation_ids": ["observation_0"],
                    }
                )
            for proposal in proposals:
                link = proposal.get("affect_link")
                if link is None:
                    continue
                if mutation == "target_instead_of_elicitor":
                    link["role"] = "target"
                    proposal["predicate"] = "affect_target"
                elif mutation == "new_episode":
                    link["affect_id"] = "proposal:different_episode"
                elif link["polarity"] == "denied":
                    link["object_ref"]["memory_id"] = "proposal:different_launch"
            envelope["choices"][0]["message"]["content"] = json.dumps(payload)
        return httpx.Response(200, json=envelope)

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        OpenAI(api_key="test", http_client=transport, max_retries=0) as client,
    ):
        report = affective_quality.run_affective_quality_eval(
            OpenAICompletion(client, generation_model="fixture", generation_max_tokens=None),
            repeats=1,
            case_ids=(case.case_id,),
        )
    assert not report["passed"]
    results = cast(list[dict[str, object]], report["cases"])
    assert results[0]["parser_dropped"] == results[0]["kernel_refused"] == 0
