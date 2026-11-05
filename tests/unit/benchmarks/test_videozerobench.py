"""VideoZeroBench release boundaries, deterministic scores, and the public SDK runner."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from test_eval_metrics import _arguments

from sensemem import (
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    EmbedTask,
    Memory,
    MemoryConfig,
    Modality,
    ModelInput,
    SearchHit,
)
from sensemem.benchmarks import prepare_media
from sensemem.benchmarks.eval import _regressed, run_loaded_task
from sensemem.benchmarks.eval_adapters import LoadedTask, load_task
from sensemem.benchmarks.eval_console import _table
from sensemem.benchmarks.eval_metrics import _comparisons, _metrics, _videozerobench_metrics
from sensemem.benchmarks.eval_results import SampleResult
from sensemem.benchmarks.isolation import BenchmarkRun
from sensemem.benchmarks.official_scorers import judge_plan, metric_is_official
from sensemem.benchmarks.task_catalog import TASKS, expand
from sensemem.benchmarks.videozerobench import (
    VIDEOZERO_METRICS,
    SpatialEvidence,
    TemporalEvidence,
    answer_correct,
    load_videozerobench,
    spatial_iou,
    temporal_iou,
)


def _release(tmp_path: Path) -> LoadedTask:
    dataset = tmp_path / "questions.json"
    question = {
        "question_id": 0,
        "question": "How many blue chairs?",
        "answer": "8",
        "evidence_windows": [{"start": 2.0, "end": 4.0}],
        "evidence_boxes": [{"time": 2.5, "box": [0.1, 0.2, 0.5, 0.6]}],
        "video": "clip.mp4",
        "video_id": "clip",
        "category": "Daily Vlogs",
        "language": "en",
        "duration": 10.0,
        "evidence_span": "single-frame",
        "annotation_capabilities": ["counting"],
    }
    dataset.write_text(
        json.dumps(
            [
                question,
                {**question, "question_id": 1, "evidence_windows": [], "evidence_boxes": []},
            ]
        )
    )
    (tmp_path / "clip.mp4").write_bytes(b"video fixture")
    return load_task(
        replace(TASKS["videozerobench"], digest=None),
        root=tmp_path,
        dataset_path=dataset,
        media_root=tmp_path,
    )


def test_release_groups_videos_and_exposes_only_protocol_allowed_hints(tmp_path: Path) -> None:
    task = _release(tmp_path)
    assert expand(("videozerobench",)) == ("videozerobench",)
    assert len(task.units) == len(task.units[0].memories) == 1
    questions = task.units[0].questions
    assert len(questions) == 8
    assert len({question.question_id for question in questions}) == 8
    for question in questions[:5]:
        prompt = "\n".join(str(part) for part in question.content)
        level = question.metadata["level"]
        assert "8" not in prompt
        assert ("From <2.00 seconds>" in prompt) == (level in (1, 2))
        assert ("[100,200,500,600]" in prompt) == (level == 1)
        assert ("<2.50 seconds>" in prompt) == (level in (1, 5))
    assert task.units[0].memories[0].content == (tmp_path / "clip.mp4",)
    subset = load_task(
        task.spec,
        root=tmp_path,
        dataset_path=task.dataset_path,
        media_root=tmp_path,
        limit=1,
        offset=1,
    )
    assert len(subset.units[0].questions) == 3
    assert all(question.question_id.startswith("1/") for question in subset.units[0].questions)
    rows = json.loads(task.dataset_path.read_text())
    task.dataset_path.write_text(json.dumps([rows[0], rows[0]]))
    with pytest.raises(ValueError, match="unique"):
        load_videozerobench(task.dataset_path)


@pytest.mark.parametrize(
    ("reference", "prediction", "correct"),
    [
        ("8", "08", False),
        ("8", "<think>wrong</think><answer>8</answer>", True),
        ("Red Door", '```text\n"red door."\n```', True),
        ("red door", "The red door", False),
        ("蓝色", "蓝", True),
        ("车", "一辆车", True),
    ],
)
def test_answer_rules_match_the_bilingual_release(
    reference: str,
    prediction: str,
    correct: bool,
) -> None:
    assert answer_correct(reference, prediction) is correct


def test_iou_uses_unions_and_scores_missing_or_malformed_grounding_as_zero() -> None:
    windows = (
        TemporalEvidence(start=0, end=10),
        TemporalEvidence(start=5, end=15),
        TemporalEvidence(start=20, end=20),
    )
    assert temporal_iou(windows, "From <0 seconds> to <15 seconds>.") == 1.0
    assert temporal_iou(windows, "00:00-00:10 00:05~00:15") == 1.0
    assert temporal_iou(windows, "From <0 seconds> to <5 seconds>.") == pytest.approx(1 / 3)
    boxes = (
        SpatialEvidence(time=2.5, box=(0, 0, 0.5, 1)),
        SpatialEvidence(time=2.5, box=(0.25, 0, 0.75, 1)),
    )
    assert spatial_iou(boxes, '[{"time":2.5,"bbox_2d":[[0,0,500,1000],[250,0,750,1000]]}]') == 1
    assert spatial_iou(boxes, '{"time":2.5,"bbox_2d":[0,0,1000,1000]}') == pytest.approx(0.75)
    for prediction in ("not JSON", "[]", '[{"time":2.5,"bbox_2d":[]}]'):
        assert spatial_iou(boxes, prediction) == 0


def test_joint_scores_use_l3_correctness_strict_thresholds_and_all_question_denominators(
    tmp_path: Path,
) -> None:
    task = _release(tmp_path)
    samples = []
    for question in task.units[0].questions:
        level = question.metadata["level"]
        samples.append(
            SampleResult(
                task="videozerobench",
                benchmark="VideoZeroBench",
                dataset_sha256=task.dataset_sha256,
                evaluation_sha256=task.evaluation_sha256,
                unit_id="clip",
                question_id=question.question_id,
                prediction="",
                parsed_choice=None,
                score=1.0,
                exact_match=None,
                latency_ms=1,
                confidence=0,
                memory_ids=(),
                ingest_failure_count=0,
                error_code=None,
                metadata=question.metadata,
                metrics={"temporal_iou": 0.3}
                if level == 4
                else {"spatial_iou": 1.0}
                if level == 5
                else {"accuracy": 1.0},
            )
        )
    summaries = _videozerobench_metrics(task, samples[::-1], seed=0, bootstrap_samples=8)
    assert summaries["Level-3_acc"]["mean"] == 100
    assert summaries["Level-4_mean_tIoU"]["sample_count"] == 1
    assert summaries["Level-5_mean_vIoU"]["sample_count"] == 1
    assert summaries["Level-4_score"]["mean"] == summaries["Level-5_score"]["mean"] == 0
    samples[3] = replace(samples[3], metrics={"temporal_iou": 1})
    summaries = _videozerobench_metrics(task, samples, seed=0, bootstrap_samples=8)
    assert summaries["Level-4_score"]["mean"] == summaries["Level-5_score"]["mean"] == 50
    samples[2] = replace(samples[2], metrics={"accuracy": 0}, score=0)
    summaries = _videozerobench_metrics(task, samples, seed=0, bootstrap_samples=8)
    assert summaries["Level-5_score"]["mean"] == 0
    assert (
        judge_plan("videozerobench", question=None, references=(), prediction="", metadata={})
        is None
    )
    assert metric_is_official("videozerobench", "Level-5_score", "", uses_judge=False)


def test_media_preparation_reuses_video_segments_for_all_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _release(tmp_path)
    calls: list[str] = []

    def parts(unit: str, source: Path, cache: Path, announce: object) -> list[dict[str, object]]:
        calls.append(unit)
        return [{"source_id": unit, "paths": [str(source)], "start_seconds": 0, "end_seconds": 10}]

    monkeypatch.setattr(prepare_media, "_video_parts", parts)
    manifest = prepare_media.prepare_task_media(
        task.spec,
        root=tmp_path,
        dataset_path=task.dataset_path,
        media_root=tmp_path,
        manifest=None,
        limit=None,
        offset=0,
        download=False,
    )
    assert calls == ["clip"]
    assert manifest is not None and "clip" in cast(dict[str, object], manifest["units"])


async def test_eval_runs_all_levels_through_asyncmemory_without_writing_gold_hints(
    tmp_path: Path,
) -> None:
    task = _release(tmp_path)
    documents: list[tuple[ModelInput, ...]] = []

    class Reader:
        generation_capabilities = frozenset({Modality.TEXT, Modality.VIDEO})
        embedding_capabilities = generation_capabilities
        embedding_model = "videozero-fixture"
        embedding_space = "videozero-fixture:2"
        embedding_dimension = 2

        def embed(
            self, inputs: Sequence[ModelInput], task: EmbedTask = EmbedTask.DOCUMENT
        ) -> tuple[tuple[float, ...], ...]:
            if task is EmbedTask.DOCUMENT:
                documents.append(tuple(inputs))
            return ((1.0, 0.0),) * len(inputs)

        def answer(
            self,
            question: ModelInput,
            hits: Sequence[SearchHit],
            *,
            answer_policy: AnswerPolicy = "strict",
            exhaustive: bool = False,
        ) -> AnswerResult:
            answer = (
                "From <2 seconds> to <4 seconds>."
                if "Find one or more" in question.text
                else '[{"time":2.5,"bbox_2d":[[100,200,500,600]]}]'
                if "bbox_2d" in question.text
                else "8"
            )
            return AnswerResult(answer, hits=tuple(hits))

        def close(self) -> None:
            pass

    def factory(path: Path) -> AsyncMemory:
        backend = Reader()
        return AsyncMemory(
            Memory(
                path, embedder=backend, answerer=backend, config=MemoryConfig(minimum_relevance=0)
            )
        )

    samples = await run_loaded_task(
        task,
        run=BenchmarkRun(tmp_path / "stores", "videozerobench", "test"),
        memory_factory=factory,
        batch_size=4,
        unit_concurrency=1,
        request_concurrency=2,
        recall_limit=5,
    )
    assert len(samples) == 8
    assert all(sample.error_code is None for sample in samples)
    assert all(
        "blue chairs" not in value.text and "100,200" not in value.text
        for batch in documents
        for value in batch
    )
    result = _metrics(task, samples, _arguments())
    assert cast(dict[str, object], result["score"])["mean"] == 1
    metrics = cast(dict[str, dict[str, object]], result["metrics"])
    assert metrics["accuracy"]["sample_count"] == 2
    assert metrics["Level-3_acc"]["mean"] == 100
    assert metrics["Level-5_score"]["mean"] == 50
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(
        "\n".join(
            json.dumps(
                replace(sample, score=0, metrics={"accuracy": 0}).json()
                if sample.metadata["level"] in (1, 2)
                else replace(sample, metrics={"temporal_iou": 0}).json()
                if sample.metadata["level"] == 4
                else replace(sample, metrics={"spatial_iou": 0}).json()
                if sample.metadata["level"] == 5
                else sample.json()
            )
            for sample in samples
        )
    )
    comparisons = _comparisons(_arguments(compare=baseline), (task,), samples)
    assert {row["metric"]: row["mean"] for row in comparisons} == {
        "Level-1_acc": 100,
        "Level-2_acc": 100,
        "Level-3_acc": 0,
        "Level-4_mean_tIoU": 100,
        "Level-4_score": 50,
        "Level-5_mean_vIoU": 100,
        "Level-5_score": 50,
    }
    assert all(row["score_scale"] == 100 and row["noise_floor"] == 3 for row in comparisons)
    assert [row["paired_sample_count"] for row in comparisons] == [2, 2, 2, 1, 2, 1, 2]
    performance = {
        "duration_seconds": {"total": 1.0, "average": 0.125},
        "token_usage": {"total_tokens": 0, "average_tokens": 0.0},
    }
    table = _table({"tasks": [{**result, "task": "videozerobench", "performance": performance}]})
    assert all(name in table for name in VIDEOZERO_METRICS)
    baseline.write_text("\n".join(baseline.read_text().splitlines()[1:]))
    with pytest.raises(ValueError, match="identical complete level requests"):
        _comparisons(_arguments(compare=baseline), (task,), samples)
    baseline.write_text(
        "\n".join(json.dumps(replace(sample, score=None, metrics={}).json()) for sample in samples)
    )
    with pytest.raises(ValueError, match="baseline has no scored samples"):
        _comparisons(_arguments(compare=baseline), (task,), samples)


def test_regression_threshold_respects_percentage_and_fraction_scales() -> None:
    assert not _regressed(
        ({"confidence_interval_95": [-2, -1], "score_scale": 100.0},), threshold=0.03
    )
    assert _regressed(({"confidence_interval_95": [-5, -4], "score_scale": 100.0},), threshold=0.03)
    assert _regressed(
        ({"confidence_interval_95": [-0.05, -0.04], "score_scale": 1.0},), threshold=0.03
    )
