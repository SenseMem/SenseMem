"""MOSAIC native inputs, scorer settings, denominators and SDK isolation."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx2 as httpx
import pytest
from openai import AsyncOpenAI
from opentelemetry import trace

from sensemem import (
    AsyncMemory,
    EmbedTask,
    IndexQuantization,
    Memory,
    MemoryConfig,
    Modality,
    ModelInput,
)
from sensemem.benchmarks import eval_judge
from sensemem.benchmarks._official.mosaic_scoring import (
    JUDGE_SYSTEM_PROMPT,
    answer_system_prompt,
    parse_judge_score,
    text_metrics,
)
from sensemem.benchmarks.download import acquire_inputs
from sensemem.benchmarks.eval import _score, run_loaded_task
from sensemem.benchmarks.eval_adapters import (
    load_task,
)
from sensemem.benchmarks.eval_arms import _Arm
from sensemem.benchmarks.eval_config import _Arguments, _JudgeConfig
from sensemem.benchmarks.eval_metrics import _comparisons, _in_run_blind_rows, _metrics
from sensemem.benchmarks.eval_results import SampleResult
from sensemem.benchmarks.isolation import BenchmarkRun
from sensemem.benchmarks.mosaic import (
    MOSAIC_DATA_SHA256,
    load_mosaic,
    mosaic_media_path,
    mosaic_refusal,
)
from sensemem.benchmarks.official_scorers import (
    judge_failure_scores,
    judge_plan,
    local_scores,
    metric_is_official,
    parse_judge_response,
    scorer_protocol,
)
from sensemem.benchmarks.prepare_media import _selected_patterns, prepare_task_media
from sensemem.benchmarks.task_catalog import TASKS, expand


def _fixture(root: Path) -> Path:
    dataset = TASKS["mosaic"].dataset_path(root)
    dataset.mkdir(parents=True)
    messages: list[dict[str, object]] = [
        {"role": "user", "content": "Visible text", "fact_key": "fact-1", "has_answer": True},
    ]
    for modality, filename in (("image", "a.png"), ("audio", "a.mp3"), ("video", "a.mp4")):
        path = root / "mosaic/dataset/media" / modality / filename
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture media")
        messages.append(
            {
                "role": "user",
                "content": f"Visible {modality}",
                "fact_key": f"{modality}-fact",
                "multimodal_content": {
                    "modality": modality,
                    "media_path": f"/original/clean_output/media/{modality}/{filename}",
                    "generation_prompt": "SECRET GENERATION PROMPT",
                    "intended_answer": "SECRET INTENDED ANSWER",
                },
            }
        )
    messages.append({"role": "assistant", "content": "Visible reply"})
    first: dict[str, object] = {
        "question_id": "q1",
        "question_type": "knowledge_update",
        "answerability": "answerable",
        "question": "What changed?",
        "answer": "SECRET ANSWER",
        "answer_aliases": ["SECRET ALIAS"],
        "question_date": "2026/04/01",
        "evidence_plan": [{"fact_key": "image-fact", "required_for_answer": True}],
        "haystack_sessions": [
            {
                "session_id": "s1",
                "timestamp": "2026/04/05 (Sun) 16:44",
                "messages": messages,
            },
            {
                "session_id": "empty",
                "timestamp": "2026/04/05 (Sun) 16:45",
                "messages": [],
            },
        ],
    }
    second = {
        **first,
        "question_id": "q2",
        "question_type": "answer_refusal",
        "answerability": "unanswerable",
        "answer": "I don't know.",
        "answer_aliases": [],
        "haystack_sessions": [
            {
                "session_id": "s1",
                "timestamp": "2026/04/06 (Mon) 16:44",
                "messages": [{"role": "user", "content": "SHOULD NOT REPLACE FIRST SESSION"}],
            },
            {
                "session_id": "s2",
                "timestamp": "2026/04/07 (Tue) 16:44",
                "messages": [
                    {
                        "role": "user",
                        "content": "Other question's visible history",
                        "fact_key": "image-fact",
                    }
                ],
            },
        ],
    }
    third = {
        **first,
        "question_id": "q3",
        "haystack_sessions": [
            {
                "session_id": "s1",
                "timestamp": "2026/04/05 (Sun) 16:44",
                "messages": [{"role": "user", "content": "Second persona"}],
            }
        ],
    }
    for name, items in (("p1", [first, second]), ("p2", [third])):
        (dataset / f"stage2_with_media_{name}.json").write_text(
            json.dumps(
                {
                    "persona": {"uuid": name, "profile": "SECRET PROFILE"},
                    "items": items,
                }
            )
        )
    return dataset


def _arguments(**overrides: object) -> _Arguments:
    return cast(
        _Arguments,
        SimpleNamespace(
            **{
                "seed": 42,
                "bootstrap_samples": 32,
                "recall_limit": 20,
                "blind": False,
                "offset": 0,
                "limit": None,
                "predict_only": False,
                **overrides,
            }
        ),
    )


def _sample() -> SampleResult:
    return SampleResult(
        task="mosaic",
        benchmark="MOSAIC",
        dataset_sha256=MOSAIC_DATA_SHA256,
        evaluation_sha256="evaluation",
        unit_id="persona",
        question_id="q",
        prediction="answer",
        parsed_choice=None,
        score=0.7,
        exact_match=None,
        latency_ms=1.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={"answerability": "answerable"},
        metrics={"judge_score": 0.7, "false_refusal_rate": 0.0},
        judge_model="qwen/qwen3.7-plus",
        scorer_protocol=scorer_protocol("mosaic"),
    )


class _TextEmbedder:
    embedding_model = "mosaic-batch-fixture"
    embedding_space = "mosaic-batch-fixture:2"
    embedding_dimension = 2
    embedding_capabilities = frozenset({Modality.TEXT})

    def embed(
        self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in inputs)

    def close(self) -> None:
        return None


async def test_serial_and_batched_runs_preserve_persona_isolation_and_question_order(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    original = load_task(TASKS["mosaic"], root=tmp_path, verify_digest=False)
    # Exercise the public SDK without decoding synthetic placeholder media or calling a model.
    task = replace(
        original,
        units=tuple(
            replace(
                unit,
                memories=tuple(replace(item, content=(item.source_id,)) for item in unit.memories),
            )
            for unit in original.units
        ),
    )

    def factory(data_dir: Path) -> AsyncMemory:
        return AsyncMemory(
            Memory(
                data_dir,
                embedder=_TextEmbedder(),
                config=MemoryConfig(index_quantization=IndexQuantization.NONE),
            )
        )

    orders = []
    for label, batch_size, concurrency in (("serial", 1, 1), ("batch", 2, 2)):
        run = BenchmarkRun(tmp_path / "runs", "mosaic", label)
        results = await run_loaded_task(
            task,
            run=run,
            memory_factory=factory,
            batch_size=batch_size,
            unit_concurrency=concurrency,
            request_concurrency=4,
            recall_limit=10,
            predict_only=True,
            arms=(_Arm("random"),),
            ingest_concurrency=concurrency,
            on_search_replay_ready=lambda _replay: None,
        )
        assert all(
            result.error_code is None and result.ingest_failure_count == 0 for result in results
        )
        orders.append([(result.unit_id, result.question_id) for result in results])
        assert run.unit_path("p1") != run.unit_path("p2")
        for unit in task.units:
            with Memory(
                run.unit_path(unit.unit_id),
                embedder=_TextEmbedder(),
                config=MemoryConfig(index_quantization=IndexQuantization.NONE),
            ) as memory:
                assert {
                    record.metadata["source_id"] for record in memory.list(limit=100).items
                } == {item.source_id for item in unit.memories}
    assert orders == [[("p1", "q1"), ("p1", "q2"), ("p2", "q3")]] * 2


@pytest.mark.parametrize(
    "unsupported",
    [
        "Unsupported parameter: enable_thinking",
        "Unsupported value: temperature does not support 0.0",
    ],
)
async def test_mosaic_never_silently_drops_binding_judge_controls(unsupported: str) -> None:
    requests = []

    def reject(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": unsupported}})

    plan = judge_plan(
        "mosaic",
        question="Question?",
        references=("reference",),
        prediction="answer",
        metadata={"answerability": "answerable"},
    )
    assert plan is not None
    async with AsyncOpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reject)),
    ) as client:
        result = await eval_judge._judge_sample(
            _sample(),
            plan,
            client=client,
            cache=None,
            semaphore=asyncio.Semaphore(1),
            config=_JudgeConfig("qwen/qwen3.7-plus", "http://judge.invalid/v1"),
            log_samples=False,
            tracer=trace.get_tracer(__name__),
        )
    assert len(requests) == 1 and result.error_code == "JudgeError"
    assert requests[0]["temperature"] == 0.0 and requests[0]["enable_thinking"] is False


@pytest.mark.parametrize(("limit", "offset"), [(1, 0), (1, 2), (0.5, 1), (None, 1)])
def test_bounded_loading_skips_unselected_histories_but_keeps_complete_persona_corpora(
    tmp_path: Path,
    limit: int | float | None,
    offset: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sensemem.benchmarks import mosaic
    from sensemem.benchmarks.eval_adapters import _selected

    dataset = _fixture(tmp_path)
    complete = load_mosaic(dataset)
    selected = _selected(
        tuple((p.persona_id, q.question_id) for p in complete for q in p.questions), limit, offset
    )
    parsed: list[bytes] = []
    original = mosaic._Release.model_validate_json

    def validate(data: bytes) -> mosaic._Release:
        parsed.append(data)
        return original(data)

    monkeypatch.setattr(mosaic._Release, "model_validate_json", validate)
    limited = load_mosaic(dataset, limit=limit, offset=offset)
    assert [(p.persona_id, q.question_id) for p in limited for q in p.questions] == list(selected)
    for persona in limited:
        assert persona.sessions == next(
            p.sessions for p in complete if p.persona_id == persona.persona_id
        )
    assert len(parsed) == len({persona for persona, _ in selected})


def test_official_persona_corpus_chunking_selection_and_label_boundaries(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    task = load_task(TASKS["mosaic"], root=tmp_path, verify_digest=False)
    assert expand(("mosaic",)) == ("mosaic",)
    assert [unit.unit_id for unit in task.units] == ["p1", "p2"]
    first, second = task.units
    assert [q.question_id for q in first.questions] == ["q1", "q2"]
    assert [m.source_id for m in first.memories] == ["s1_mm1", "s1_mm2", "s1_mm3", "s1", "s2"]
    assert first.memories[3].content == ("[user] Visible text\n[assistant] Visible reply",)
    assert first.memories[0].content[0] == "Visible image"
    assert second.memories[0].content == ("[user] Second persona",)
    inputs = repr([m.content for u in task.units for m in u.memories])
    inputs += repr([q.content for u in task.units for q in u.questions])
    assert "SECRET" not in inputs and "SHOULD NOT REPLACE" not in inputs
    assert first.questions[0].references == ("SECRET ANSWER", "SECRET ALIAS")
    assert first.questions[0].metadata["evidence_groups"] == (("s1_mm1",),)
    assert "from <old> to <new>" in str(first.questions[0].content)
    assert "from <old> to <new>" not in str(first.questions[1].content)
    limited = load_task(TASKS["mosaic"], root=tmp_path, limit=1, verify_digest=False)
    assert limited.units[0].memories == first.memories
    assert len(limited.units[0].questions) == 1
    offset = load_task(TASKS["mosaic"], root=tmp_path, limit=1, offset=2, verify_digest=False)
    assert offset.units[0].unit_id == "p2"
    assert _selected_patterns(TASKS["mosaic"], dataset, 1, 0) == (
        "dataset/media/image/a.png",
        "dataset/media/audio/a.mp3",
        "dataset/media/video/a.mp4",
    )
    (tmp_path / "mosaic/dataset/media/audio/a.mp3").unlink()
    with pytest.raises(FileNotFoundError, match=r"audio/a\.mp3"):
        load_task(TASKS["mosaic"], root=tmp_path, limit=1, verify_digest=False)


def test_acquisition_uses_pinned_annotations_and_selected_personas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sensemem.benchmarks import download

    calls: list[tuple[str, str, tuple[str, ...], Path]] = []

    def snapshot(repo: str, revision: str, patterns: tuple[str, ...], destination: Path) -> None:
        calls.append((repo, revision, patterns, destination))
        if patterns == ("dataset/stage2_human_filtered/*.json",):
            _fixture(tmp_path)

    monkeypatch.setattr(download, "_snapshot", snapshot)
    acquire_inputs(TASKS["mosaic"], tmp_path)
    spec = TASKS["mosaic"]
    assert calls[0] == (
        spec.repository,
        spec.revision,
        ("dataset/stage2_human_filtered/*.json",),
        tmp_path / "mosaic",
    )
    assert (
        prepare_task_media(
            spec,
            root=tmp_path,
            dataset_path=spec.dataset_path(tmp_path),
            media_root=None,
            manifest=None,
            limit=1,
            offset=0,
            download=True,
        )
        is None
    )
    assert calls[1][2] == _selected_patterns(spec, spec.dataset_path(tmp_path), 1, 0)


@pytest.mark.parametrize("path", ["../escape.png", "/outside.png", "clean_output/media/../a.png"])
def test_invalid_media_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="invalid MOSAIC media path"):
        mosaic_media_path(path)


def test_duplicate_question_ids_and_dataset_digest_fail(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    with pytest.raises(ValueError, match="digest mismatch"):
        load_task(TASKS["mosaic"], root=tmp_path)
    source = dataset / "stage2_with_media_p1.json"
    (dataset / "stage2_with_media_duplicate.json").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="duplicate question IDs"):
        load_mosaic(dataset)


@pytest.mark.parametrize(
    ("response", "score"),
    [
        ('{"score": 0.7}', 0.7),
        ('```json\n{"score": "0.5"}\n```', 0.5),
        ("{score: 0.3,}", 0.3),
        ('{"score": 8}', 1.0),
        ('{"score": -2}', 0.0),
        ("{}", 0.0),
        ('{"score": null}', 0.0),
        ('{"score": "oops"}', 0.0),
        ('{"score": NaN}', 0.0),
        ("[]", 0.0),
    ],
)
def test_official_json_repair_and_numeric_scoring(response: str, score: float) -> None:
    assert parse_judge_score(response)[0] == score


def test_official_judge_messages_and_request_settings() -> None:
    plan = judge_plan(
        "mosaic",
        question="When?",
        references=("8:15 AM", "8:15"),
        prediction="8:15",
        metadata={"answerability": "answerable"},
    )
    assert plan is not None
    assert plan.calls[0][0].content == JUDGE_SYSTEM_PROMPT
    assert plan.calls[0][1].content == (
        "Question: When?\nAnswerability: answerable\n\nExpected answer: 8:15 AM\n"
        'Accepted aliases: "8:15"\n\nPredicted answer: 8:15\n\nScore this answer.'
    )
    assert plan.calls[0][0].role == "system"
    assert plan.max_tokens == 4096 and plan.extra_body == {
        "thinking": {"type": "disabled"},
        "enable_thinking": False,
    }
    assert parse_judge_response(plan, '{"score": 0.7}') == {"judge_score": 0.7}
    assert judge_failure_scores("mosaic", {"em": 1.0}) == {"em": 1.0, "judge_score": 0.0}
    assert metric_is_official("mosaic", "judge_score", "qwen/qwen3.7-plus", uses_judge=True)
    assert not metric_is_official("mosaic", "judge_score", "other-model", uses_judge=True)
    # Full-prompt hashes below are taken from the pinned upstream AST, not from this module.
    assert hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest() == (
        "fa3fd8177a5ce20e1e0f2cd08d73f9f350d1c09c90aec5c7173afd792f7a695d"
    )
    assert hashlib.sha256(answer_system_prompt("knowledge_update").encode()).hexdigest() == (
        "1f8473f22674fc034d643c7994b103f64b336d8c984f92af482108e4be73da60"
    )
    assert hashlib.sha256(answer_system_prompt("information_extraction").encode()).hexdigest() == (
        "acb01b2ca7eb0e64a5bba3e82d8c33cf0b0ebe40f8d5e27bbbb3ec7ac19096e1"
    )
    assert answer_system_prompt("knowledge_update") != answer_system_prompt(
        "information_extraction"
    )


def test_official_text_metrics_and_refusal_phrase_matching() -> None:
    assert text_metrics("Answer: 8:15", ("8:15 AM", "8:15")) == {
        "em": 1.0,
        "f1": 1.0,
        "bleu1": 1.0,
    }
    assert text_metrics("a a", ("a a a",)) == {"em": 0.0, "f1": 0.8, "bleu1": 0.6065}
    assert text_metrics("", ("",)) == {"em": 1.0, "f1": 1.0, "bleu1": 1.0}
    assert mosaic_refusal(" I do not\n know. ")
    assert mosaic_refusal("There is insufficient information about it")
    assert not mosaic_refusal("unknown") and not mosaic_refusal("none")


def test_headline_refusal_and_comparison_denominators(tmp_path: Path) -> None:
    _fixture(tmp_path)
    task = load_task(TASKS["mosaic"], root=tmp_path, verify_digest=False)
    samples = []
    for unit in task.units:
        for question in unit.questions:
            prediction = "I don't know." if question.question_id != "q1" else "8:15 AM"
            score = 0.0 if question.question_id == "q3" else 1.0
            metrics = local_scores(
                "mosaic",
                score_kind="text",
                prediction=prediction,
                references=question.references,
                parsed_choice=None,
                expected_choice=None,
                question=question.source_question,
                metadata=question.metadata,
                evidence_source_ids=(),
            )
            samples.append(
                SampleResult(
                    task="mosaic",
                    benchmark="MOSAIC",
                    dataset_sha256=task.dataset_sha256,
                    evaluation_sha256=task.evaluation_sha256,
                    unit_id=unit.unit_id,
                    question_id=question.question_id,
                    prediction=prediction,
                    parsed_choice=None,
                    score=score,
                    exact_match=None,
                    latency_ms=1.0,
                    confidence=1.0,
                    memory_ids=(),
                    ingest_failure_count=0,
                    error_code="ModelError" if question.question_id == "q3" else None,
                    metadata=question.metadata,
                    metrics={**metrics, "judge_score": score},
                    judge_model="qwen/qwen3.7-plus",
                    scorer_protocol=scorer_protocol("mosaic"),
                )
            )
    result = _metrics(task, samples, _arguments())
    assert cast(Mapping[str, object], result["score"])["mean"] == 0.5
    assert result["question_count"] == 3 and result["scored_question_count"] == 2
    metrics_by_name = cast(Mapping[str, Mapping[str, object]], result["metrics"])
    assert metrics_by_name["judge_score"]["mean"] == 0.5
    assert metrics_by_name["correct_refusal_rate"]["mean"] == 1.0
    assert metrics_by_name["false_refusal_rate"]["mean"] == 0.5
    blind = [replace(sample, arm="blind") for sample in samples]
    rows = _in_run_blind_rows(_arguments(arms=("blind",), run_id="fixture"), (task,), blind)
    assert rows["mosaic"]["mean"] == 0.5 and rows["mosaic"]["question_count"] == 2
    baseline = tmp_path / "samples.jsonl"
    baseline.write_text("\n".join(json.dumps(sample.json()) for sample in samples))
    comparisons = _comparisons(
        _arguments(compare=baseline, regression_threshold=0.0), (task,), samples
    )
    assert comparisons[0]["metric"] == "judge_score"
    failed_question = task.units[1].questions[0]
    _, failed_score, _ = _score(
        "mosaic",
        failed_question,
        "",
        None,
        (),
        predict_only=False,
        answer_failed=True,
    )
    assert failed_score == 0.0


def test_long_official_chunks_retain_one_source_and_all_visible_text(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    visible = "x" * 60_001
    (dataset / "stage2_with_media_p2.json").write_text(
        json.dumps(
            {
                "persona": {"uuid": "p2"},
                "items": [
                    {
                        "question_id": "q3",
                        "question_type": "information_extraction",
                        "answerability": "answerable",
                        "question": "What?",
                        "answer": "x",
                        "question_date": "2026/04/01",
                        "haystack_sessions": [
                            {
                                "session_id": "long",
                                "timestamp": "2026/04/05 (Sun) 16:44",
                                "messages": [{"role": "user", "content": visible}],
                            }
                        ],
                    }
                ],
            }
        )
    )
    unit = load_task(
        TASKS["mosaic"],
        root=tmp_path,
        verify_digest=False,
        offset=2,
        limit=1,
    ).units[0]
    assert len(unit.memories) == 4 and all(item.source_id == "long" for item in unit.memories)
    parts = tuple(part for item in unit.memories for part in item.content if isinstance(part, str))
    assert "".join(parts) == f"[user] {visible}" and max(map(len, parts)) <= 16_000
    with Memory(tmp_path / "long-store", embedder=_TextEmbedder()) as memory:
        records = memory.add_many(
            tuple(item.content for item in unit.memories),
            metadata=tuple({"source_id": item.source_id} for item in unit.memories),
        )
        assert "".join(record.content for record in records) == f"[user] {visible}"
        assert all(record.metadata["source_id"] == "long" for record in records)


async def test_official_judge_transport_keeps_both_thinking_controls() -> None:
    requests: list[dict[str, object]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "judge-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen/qwen3.7-plus",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": '{"score":0.7}'}}
                ],
            },
        )

    sample = SampleResult(
        task="mosaic",
        benchmark="MOSAIC",
        dataset_sha256="dataset",
        evaluation_sha256="evaluation",
        unit_id="persona",
        question_id="q",
        prediction="8:15",
        parsed_choice=None,
        score=None,
        exact_match=None,
        latency_ms=0.0,
        confidence=1.0,
        memory_ids=(),
        ingest_failure_count=0,
        error_code=None,
        metadata={"answerability": "answerable"},
    )
    plan = judge_plan(
        "mosaic",
        question="When?",
        references=("8:15 AM",),
        prediction=sample.prediction,
        metadata=sample.metadata,
    )
    assert plan is not None
    async with AsyncOpenAI(
        api_key="fixture",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reply)),
    ) as client:
        result = await eval_judge._judge_sample(
            sample,
            plan,
            client=client,
            cache=None,
            semaphore=asyncio.Semaphore(1),
            config=_JudgeConfig("qwen/qwen3.7-plus", "http://judge.invalid/v1"),
            log_samples=False,
            tracer=trace.get_tracer(__name__),
        )
    assert result.score == 0.7 and result.error_code is None
    assert len(requests) == 1
    assert requests[0]["temperature"] == 0.0 and requests[0]["max_tokens"] == 4096
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert requests[0]["enable_thinking"] is False
    assert requests[0]["messages"] == [
        {"role": message.role, "content": message.content} for message in plan.calls[0]
    ]
