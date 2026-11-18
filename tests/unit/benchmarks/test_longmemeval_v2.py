"""LongMemEval-V2 release shapes, evidence boundaries, media and scorer parity."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from sensemem.benchmarks import download, prepare_media
from sensemem.benchmarks._official.longmemeval_v2_scoring import parse_eval_spec
from sensemem.benchmarks.eval_adapters import load_task
from sensemem.benchmarks.longmemeval_v2 import (
    LONGMEMEVAL_V2_ADAPTER_VERSION,
    load_longmemeval_v2_questions,
)
from sensemem.benchmarks.official_scorers import (
    judge_plan,
    local_scores,
    official_judge_model,
    parse_judge_response,
    scorer_protocol,
    task_family,
)
from sensemem.benchmarks.task_catalog import TASKS, expand

_PHRASES = (
    "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true"
    "|separators=,;|require_non_empty=true"
)


def _question(question_id: str = "q1", **overrides: object) -> dict[str, object]:
    return {
        "id": question_id,
        "domain": "web",
        "environment": "webarena-reddit",
        "question_type": "static-environment",
        "question": "Which menu? Mark your answer in \\boxed{}.",
        "image": None,
        "answer": "scorer-only gold",
        "eval_function": _PHRASES,
        **overrides,
    }


def _trajectory(trajectory_id: str, **overrides: object) -> dict[str, object]:
    return {
        "id": trajectory_id,
        "domain": "web",
        "environment": "webarena-reddit",
        "goal": f"Find the menu in {trajectory_id}",
        "outcome": "failure",
        "start_url": "http://forum.test/start",
        "states": [
            {
                "state_index": 0,
                "step": 0,
                "url": "http://forum.test/start",
                "action": None,
                "thought": "Inspect navigation",
                "accessibility_tree": "Menu Settings",
                "screenshot": f"screenshots/{trajectory_id}/0.png",
            },
            {
                "state_index": 1,
                "step": 2,
                "url": "http://forum.test/settings",
                "action": "click [7]",
                "thought": None,
                "accessibility_tree": "Settings page",
                "screenshot": None,
            },
        ],
        **overrides,
    }


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> Path:
    release = tmp_path / "longmemeval-v2"
    _jsonl(
        release / "questions.jsonl",
        [_question(image="question_screenshots/q1.png"), _question("q2"), _question("q3")],
    )
    _jsonl(release / "trajectories.jsonl", [_trajectory("t1"), _trajectory("t2")])
    (release / "haystacks").mkdir()
    for tier in ("small", "medium"):
        (release / f"haystacks/lme_v2_{tier}.json").write_text(
            json.dumps({"q1": ["t2", "t1"], "q2": ["t2", "t1"], "q3": ["t1", "t2"]}),
            encoding="utf-8",
        )
    for name in ("question_screenshots/q1.png", "screenshots/t1/0.png", "screenshots/t2/0.png"):
        image = release / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(name.encode())
    return release


@pytest.mark.parametrize("tier", ["small", "medium"])
def test_catalog_loads_ordered_haystacks_and_keeps_labels_private(
    tmp_path: Path, tier: str
) -> None:
    release = _fixture(tmp_path)
    spec = TASKS[f"longmemeval-v2-{tier}"]
    task = load_task(spec, root=tmp_path, verify_digest=False)
    # Streaming the fingerprint must preserve the existing serialized identity.
    assert (
        task.input_sha256["memory"]
        == "3cf3b453d4f129f6b12db555b1074f7656ec63e12dc015e22109dbeef4a1c07f"
    )

    assert spec.adapter_version == LONGMEMEVAL_V2_ADAPTER_VERSION
    assert spec.repository == "xiaowu0162/longmemeval-v2"
    assert spec.variant == tier
    assert len(task.units) == 2
    first, reordered = task.units
    assert [question.question_id for question in first.questions] == ["q1", "q2"]
    assert first.unit_id != reordered.unit_id
    assert "Find the menu in t2" in str(first.memories[0].content[0])
    assert "Find the menu in t1" in str(reordered.memories[0].content[0])
    assert [memory.source_id for memory in first.memories] == [
        "Tt2_S0000",
        "Tt2_S0001",
        "Tt1_S0000",
        "Tt1_S0001",
    ]
    assert first.memories[0] is reordered.memories[2]
    assert first.memories[0].content[1] == release / "screenshots/t2/0.png"
    assert "Outcome: failure" in str(first.memories[0].content[0])
    assert "Action: click [7]" in str(first.memories[1].content[0])
    question = first.questions[0]
    assert question.content[1] == release / "question_screenshots/q1.png"
    assert question.refusal == r"\boxed{UNKNOWN}"
    assert question.references == ("scorer-only gold",)
    assert question.metadata["haystack_size"] == 2
    assert "Do not guess" in str(question.content[0])
    assert all(memory.occurred_at is None for memory in first.memories)
    visible = repr(tuple(memory.content for memory in first.memories)) + repr(question.content)
    assert "scorer-only gold" not in visible
    assert "norm_phrase_set_match" not in visible

    original_digest = task.evaluation_sha256
    (release / "question_screenshots/q1.png").write_bytes(b"changed question image")
    changed = load_task(spec, root=tmp_path, verify_digest=False)
    assert changed.evaluation_sha256 != original_digest


def test_question_selection_happens_before_haystack_grouping(tmp_path: Path) -> None:
    _fixture(tmp_path)
    task = load_task(
        TASKS["longmemeval-v2-small"], root=tmp_path, limit=1, offset=1, verify_digest=False
    )
    assert len(task.units) == 1
    assert [question.question_id for question in task.units[0].questions] == ["q2"]
    assert expand(("longmemeval-v2",)) == ("longmemeval-v2-small", "longmemeval-v2-medium")
    assert task_family("longmemeval-v2-small") == "longmemeval-v2"
    assert official_judge_model("longmemeval-v2-small") == "gpt-5.2"


@pytest.mark.parametrize(
    "failure", ["missing_id", "duplicate_id", "cross_domain", "state_order", "path"]
)
def test_invalid_haystacks_and_trajectories_fail(tmp_path: Path, failure: str) -> None:
    release = _fixture(tmp_path)
    if failure == "missing_id":
        _jsonl(release / "trajectories.jsonl", [_trajectory("t1")])
        message = "unknown trajectory"
    elif failure == "duplicate_id":
        _jsonl(release / "trajectories.jsonl", [_trajectory("t1"), _trajectory("t1")])
        message = "duplicate IDs"
    elif failure == "cross_domain":
        _jsonl(
            release / "trajectories.jsonl",
            [_trajectory("t1"), _trajectory("t2", domain="enterprise")],
        )
        message = "cross-domain"
    else:
        trajectory = _trajectory("t2")
        states = trajectory["states"]
        assert isinstance(states, list)
        states[0]["state_index" if failure == "state_order" else "screenshot"] = (
            9 if failure == "state_order" else "screenshots/t2/../../escape.png"
        )
        _jsonl(release / "trajectories.jsonl", [_trajectory("t1"), trajectory])
        message = "state indices" if failure == "state_order" else "screenshot must be under"
    with pytest.raises(ValueError, match=message):
        load_task(TASKS["longmemeval-v2-small"], root=tmp_path, verify_digest=False)


def test_missing_screenshot_fails_instead_of_silently_scoring_text_only(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    (release / "screenshots/t2/0.png").unlink()
    with pytest.raises(FileNotFoundError):
        load_task(TASKS["longmemeval-v2-small"], root=tmp_path, verify_digest=False)


def test_long_observations_are_split_without_dropping_the_screenshot(tmp_path: Path) -> None:
    release = _fixture(tmp_path)
    trajectory = _trajectory("t2")
    states = trajectory["states"]
    assert isinstance(states, list)
    states[0]["accessibility_tree"] = "x" * 60_000
    _jsonl(release / "trajectories.jsonl", [_trajectory("t1"), trajectory])
    task = load_task(TASKS["longmemeval-v2-small"], root=tmp_path, limit=1, verify_digest=False)
    blocks = [
        memory for memory in task.units[0].memories if memory.source_id.startswith("Tt2_S0000")
    ]
    assert len(blocks) > 1
    assert sum(str(memory.content[0]).count("x") for memory in blocks) == 60_000
    assert sum(isinstance(part, Path) for memory in blocks for part in memory.content) == 1


@pytest.mark.parametrize(
    ("spec", "answer", "prediction", "score"),
    [
        (_PHRASES, "nav-bar; settings", r"Reasoning. \boxed{Settings and NAV_bar}", 1.0),
        (_PHRASES, "app", r"\boxed{application}", 0.0),
        (_PHRASES, "app", "UNKNOWN", 0.0),
        (_PHRASES.replace("match|", "match_ordered|"), "Reports;Problems", "Problems;Reports", 0.0),
        (
            _PHRASES.replace("match|", "match_ordered|"),
            "Reports;Problems",
            r"\boxed{Reports;Problems}",
            1.0,
        ),
        (
            "mc_choice_match|require_non_empty=true",
            "B",
            r"First \boxed{A} then \boxed{Option B.}",
            1.0,
        ),
        ("mc_choice_match|require_non_empty=true", "B", "B because it is blue", 0.0),
        (
            "mc_choice_set_match|require_non_empty=true",
            "A, C",
            r"\boxed{Final answers: C and A}",
            1.0,
        ),
        ("mc_choice_set_match|require_non_empty=true", "A, C", "A,B,C", 0.0),
        (_PHRASES, "{Settings}", r"\boxed{{Settings}}", 1.0),
    ],
)
def test_official_deterministic_scores(
    spec: str, answer: str, prediction: str, score: float
) -> None:
    metadata: dict[str, object] = {"eval_function": spec}
    assert local_scores(
        "longmemeval-v2-small",
        score_kind="text",
        prediction=prediction,
        parsed_choice=None,
        expected_choice=None,
        references=(answer,),
        question="Which menu?",
        metadata=metadata,
        evidence_source_ids=(),
    ) == {"accuracy": score}
    assert (
        judge_plan(
            "longmemeval-v2-small",
            question="Which menu?",
            references=(answer,),
            prediction=prediction,
            metadata=metadata,
        )
        is None
    )


@pytest.mark.parametrize("name", ["llm_abstention_checker", "llm_gotchas_checker"])
def test_judges_keep_full_response_and_extracted_answer(name: str) -> None:
    spec = f"{name}|require_non_empty=true"
    plan = judge_plan(
        "longmemeval-v2-small",
        question="Why?",
        references=("Expected flaw",),
        prediction=r"Contradictory reasoning. \boxed{Final explanation}",
        metadata={"eval_function": spec},
    )
    assert plan is not None
    assert plan.protocol == scorer_protocol("longmemeval-v2-small")
    assert plan.max_tokens == 4096
    assert plan.calls[0][0].role == "system"
    assert "Contradictory reasoning" in plan.calls[0][1].content
    assert "Model extracted final answer:\nFinal explanation\n" in plan.calls[0][1].content
    assert parse_judge_response(plan, '```json\n{"label": 1, "reason": "ok"}\n```') == {
        "accuracy": 1.0
    }
    assert parse_judge_response(plan, "label = 0") == {"accuracy": 0.0}
    with pytest.raises(ValueError, match="binary label"):
        parse_judge_response(plan, "yes")
    assert (
        judge_plan(
            "longmemeval-v2-small",
            question="Why?",
            references=("Expected flaw",),
            prediction=r"\boxed{UNKNOWN}",
            metadata={"eval_function": spec},
        )
        is None
    )


@pytest.mark.parametrize(
    "spec",
    [
        "__import__('os')",
        "mc_choice_match|evaluator_base_url=http://bad",
        "mc_choice_match|lower=yes",
    ],
)
def test_unsupported_evaluation_specs_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_eval_spec(spec)


def test_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    _jsonl(path, [_question(), _question()])
    with pytest.raises(ValueError, match="unique"):
        load_longmemeval_v2_questions(path)


def test_download_pins_inputs_and_prepares_only_selected_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = TASKS["longmemeval-v2-small"]
    snapshots: list[tuple[str, ...]] = []

    def snapshot(
        repository: str, revision: str, patterns: tuple[str, ...], destination: Path
    ) -> None:
        assert repository == spec.repository
        assert revision == spec.revision
        snapshots.append(patterns)
        for pattern in patterns:
            target = destination / pattern
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"placeholder")

    monkeypatch.setattr(download, "_snapshot", snapshot)
    download.acquire_inputs(spec, tmp_path)
    assert snapshots == [("questions.jsonl", "trajectories.jsonl", "haystacks/lme_v2_small.json")]
    release = tmp_path / "longmemeval-v2"
    _jsonl(release / "questions.jsonl", [_question(image="question_screenshots/q1.png")])
    archive = release / "trajectory_screenshots/web_screenshots.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as volume:
        data = b"screenshot"
        member = tarfile.TarInfo("t1/0.png")
        member.size = len(data)
        volume.addfile(member, io.BytesIO(data))
    image = release / "question_screenshots/q1.png"
    image.parent.mkdir()
    image.write_bytes(b"question")
    assert prepare_media._selected_patterns(spec, release / "questions.jsonl", 1, 0) == (
        "question_screenshots/q1.png",
        "trajectory_screenshots/web_screenshots.tar.gz",
    )
    prepare_media.prepare_task_media(
        spec,
        root=tmp_path,
        dataset_path=release / "questions.jsonl",
        media_root=None,
        manifest=None,
        limit=1,
        offset=0,
        download=False,
    )
    assert (release / "screenshots/t1/0.png").read_bytes() == b"screenshot"
    archive.unlink()
    # An operator may retain only the extracted tree for an offline run.
    prepare_media.prepare_task_media(
        spec,
        root=tmp_path,
        dataset_path=release / "questions.jsonl",
        media_root=None,
        manifest=None,
        limit=1,
        offset=0,
        download=False,
    )
