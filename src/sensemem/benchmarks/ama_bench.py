"""Pinned AMA-Bench input contract and official leaderboard result export."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sensemem.benchmarks._contracts import ContractModel, Identifier
from sensemem.benchmarks._official.ama_scoring import SCORER_VERSION, extract_answer

if TYPE_CHECKING:
    from sensemem.benchmarks.eval_results import SampleResult

AMA_ADAPTER_VERSION = "ama_bench_official_v1"
AMA_DATA_REVISION = "a5777378066f53229a94557a7b192435cd027909"
AMA_DATA_SHA256 = "45c36052e1520d87ad9de4114f71c9df42d4aac9cf158c0c353e800b653d65ff"
CAPABILITIES = ("A", "B", "C", "D")
# Published six-decimal weights, including their rounding, from the pinned
# leaderboard's data/qa_distribution.json. Recomputing these from counts changes scores.
_WEIGHTS = {
    "TEXT2SQL": (
        0.245192,
        (0.364379, 0.25, 0.218954, 0.166667),
        (0.089344, 0.061298, 0.053686, 0.040865),
    ),
    "SOFTWARE": (
        0.173077,
        (0.490741, 0.173611, 0.168981, 0.166667),
        (0.084936, 0.030048, 0.029247, 0.028846),
    ),
    "WEB": (0.149038, (0.336022, 0.25, 0.25, 0.163978), (0.050080, 0.037260, 0.037260, 0.024439)),
    "GAME": (0.144231, (0.333333, 0.25, 0.25, 0.166667), (0.048077, 0.036058, 0.036058, 0.024038)),
    "EMBODIED_AI": (
        0.144231,
        (0.169444, 0.25, 0.416667, 0.163889),
        (0.024439, 0.036058, 0.060096, 0.023638),
    ),
    "OPENWORLD_QA": (
        0.144231,
        (0.272222, 0.263889, 0.297222, 0.166667),
        (0.039263, 0.038062, 0.042868, 0.024038),
    ),
}
_CAP_WEIGHTS = (0.336058, 0.238782, 0.259296, 0.165865)


class AMAStep(ContractModel):
    turn_idx: Annotated[int, Field(strict=True, ge=0)]
    action: str | None
    observation: str | None

    def text(self) -> str:
        # The release contains null actions/observations. The official f-string
        # renders them as None; retain that representation and every source step.
        return f"Step {self.turn_idx}:\nAction: {self.action}\nObservation: {self.observation}\n"


class AMAQuestion(ContractModel):
    question: Annotated[str, Field(min_length=1)]
    answer: str
    type: Literal["A", "B", "C", "D"]
    question_uuid: Identifier


class AMAEpisode(ContractModel):
    episode_id: Annotated[int, Field(strict=True, ge=0)]
    task: str
    task_type: Identifier
    domain: Literal["TEXT2SQL", "SOFTWARE", "WEB", "Game", "EMBODIED_AI", "OPENWORLD_QA"]
    success: bool
    num_turns: Annotated[int, Field(strict=True, ge=0)]
    total_tokens: Annotated[int, Field(strict=True, ge=0)]
    trajectory: tuple[AMAStep, ...] = Field(min_length=1)
    qa_pairs: tuple[AMAQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_questions(self) -> AMAEpisode:
        if len({q.question_uuid for q in self.qa_pairs}) != len(self.qa_pairs):
            raise ValueError("AMA-Bench question UUIDs must be unique")
        return self


def load_ama_bench(path: Path) -> tuple[AMAEpisode, ...]:
    with path.open(encoding="utf-8") as stream:
        episodes = tuple(AMAEpisode.model_validate_json(line) for line in stream if line.strip())
    if not episodes or len({row.episode_id for row in episodes}) != len(episodes):
        raise ValueError("AMA-Bench episode IDs must be non-empty and unique")
    uuids = [q.question_uuid for row in episodes for q in row.qa_pairs]
    if len(set(uuids)) != len(uuids):
        raise ValueError("AMA-Bench question UUIDs must be globally unique")
    return episodes


def leaderboard_scores(rows: Sequence[tuple[str, str, float]]) -> dict[str, object]:
    """Match submission.py cells and app.py's two weighted leaderboard tables.

    Leaderboard F1 is an upstream alias of judge accuracy, not token F1.
    A fractional F1 fallback is incorrect under evaluate.py's score == 1 rule.
    """
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for domain, capability, score in rows:
        cells[domain.upper(), capability].append(float(score == 1.0))
    scores = {
        domain: [
            {cap: round(sum(values) / len(values), 4) if (values := cells[domain, cap]) else 0.0}
            for cap in CAPABILITIES
        ]
        for domain in sorted(_WEIGHTS)
    }
    domains = {}
    capability_sums = dict.fromkeys(CAPABILITIES, 0.0)
    capability_weights = dict.fromkeys(CAPABILITIES, 0.0)
    # app.py iterates the sorted Score cells and accumulates with +=. Retain
    # that order as well as the published weights, including floating-point rounding.
    for domain in scores:
        weighted_sum = weight_total = 0.0
        for i, cap in enumerate(CAPABILITIES):
            value = scores[domain][i][cap]
            within, overall = _WEIGHTS[domain][1][i], _WEIGHTS[domain][2][i]
            weighted_sum += value * within
            weight_total += within
            capability_sums[cap] += value * overall
            capability_weights[cap] += overall
        domains[domain] = weighted_sum / weight_total
    capabilities = {cap: capability_sums[cap] / capability_weights[cap] for cap in CAPABILITIES}
    capability_avg = sum(
        capabilities[c] * w for c, w in zip(CAPABILITIES, _CAP_WEIGHTS, strict=True)
    ) / sum(_CAP_WEIGHTS)
    summary_sum = summary_weight = 0.0
    for cap, weight in zip(CAPABILITIES, _CAP_WEIGHTS, strict=True):
        summary_sum += capabilities[cap] * weight
        summary_weight += weight
    return {
        "Score": scores,
        "scored_questions": len(rows),
        "domain_accuracy": domains,
        "capability_accuracy": capabilities,
        "domain_avg_score": sum(domains[d] * w[0] for d, w in _WEIGHTS.items())
        / sum(w[0] for w in _WEIGHTS.values()),
        "capability_avg_score": capability_avg,
        "avg_accuracy": summary_sum / summary_weight,
        "avg_f1": summary_sum / summary_weight,
        "domain_f1": domains.copy(),
        "capability_f1": capabilities.copy(),
    }


def evaluation_summary(samples: Sequence[SampleResult]) -> dict[str, object]:
    """Match evaluate.py's averages and exact-1 accuracy at all three groupings."""
    scored = [
        s
        for s in samples
        if "llm_judge_score" in s.metrics and s.error_code is None and not s.ingest_failure_count
    ]

    def stats(values: Sequence[float]) -> dict[str, int | float]:
        return {
            "count": len(values),
            "avg_score": sum(values) / len(values) if values else 0.0,
            "accuracy": sum(value == 1.0 for value in values) / len(values) if values else 0.0,
        }

    overall = stats([s.metrics["llm_judge_score"] for s in scored])
    overall["total_questions"] = overall.pop("count")
    result: dict[str, object] = {"overall": overall}
    for field, key in (
        ("task_type", "by_task_type"),
        ("domain", "by_domain"),
        ("qa_type", "by_qa_type"),
    ):
        groups: dict[str, list[float]] = defaultdict(list)
        for sample in scored:
            groups[str(sample.metadata[field])].append(sample.metrics["llm_judge_score"])
        result[key] = {label: stats(values) for label, values in groups.items()}
    return result


def result_summary(
    samples: Sequence[SampleResult], *, full_dataset_selected: bool
) -> dict[str, object]:
    """Keep the evaluator's unrounded score distinct from the public board projection."""
    judged = [
        s
        for s in samples
        if "llm_judge_score" in s.metrics and s.error_code is None and not s.ingest_failure_count
    ]
    return {
        "official_evaluation": evaluation_summary(samples),
        "leaderboard": {
            **leaderboard_scores(
                tuple(
                    (
                        str(s.metadata["domain"]),
                        str(s.metadata["qa_type"]),
                        s.metrics["llm_judge_score"],
                    )
                    for s in judged
                )
            ),
            "complete": full_dataset_selected
            and len(judged) == len(samples) == 2496
            and len({s.unit_id for s in judged}) == 208
            and all(s.dataset_sha256 == AMA_DATA_SHA256 for s in judged),
            "verified": False,
        },
    }


class _ExportSample(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)
    task: str
    arm: str
    unit_id: str
    question_id: str
    prediction: str
    dataset_sha256: str
    metrics: dict[str, float]
    error_code: str | None
    ingest_failure_count: Annotated[int, Field(ge=0)]
    judge_model: str | None
    scorer_protocol: str | None


def export_submission(dataset: Path, samples: Path, output: Path) -> None:
    """Write the current Space schema, requiring complete, genuinely judged output."""
    from sensemem.benchmarks.official_scorers import judge_model_is_official

    if output.suffix != ".jsonl":
        raise ValueError("AMA-Bench submissions must use the .jsonl extension")
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != AMA_DATA_SHA256:
        raise ValueError("AMA-Bench submission requires the pinned official dataset")
    episodes = load_ama_bench(dataset)
    selected: dict[tuple[str, str], _ExportSample] = {}
    with samples.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = _ExportSample.model_validate_json(line)
            if row.task != "ama-bench" or row.arm != "sensemem":
                continue
            key = (row.unit_id, row.question_id)
            if key in selected:
                raise ValueError("AMA-Bench submission contains duplicate predictions")
            if (
                row.dataset_sha256 != AMA_DATA_SHA256
                or row.error_code is not None
                or row.ingest_failure_count
                or "llm_judge_score" not in row.metrics
            ):
                raise ValueError("AMA-Bench submission requires successful, judged predictions")
            if row.scorer_protocol != SCORER_VERSION or not judge_model_is_official(
                "ama-bench", row.judge_model or ""
            ):
                raise ValueError(
                    "AMA-Bench submission requires the pinned scorer and Qwen/Qwen3-32B judge"
                )
            selected[key] = row
    expected = {(str(ep.episode_id), q.question_uuid) for ep in episodes for q in ep.qa_pairs}
    if selected.keys() != expected:
        raise ValueError(
            "AMA-Bench submission must cover exactly all 208 episodes and 2,496 questions"
        )
    rows = [
        {
            "episode_id": str(ep.episode_id),
            "question_uuid_list": [q.question_uuid for q in ep.qa_pairs],
            "answer_list": [
                extract_answer(selected[str(ep.episode_id), q.question_uuid].prediction)
                for q in ep.qa_pairs
            ],
            "llm_as_judge_score_list": [
                selected[str(ep.episode_id), q.question_uuid].metrics["llm_judge_score"] == 1.0
                for q in ep.qa_pairs
            ],
        }
        for ep in episodes
    ]
    from sensemem.benchmarks.eval_artifacts import _atomic_replace, _jsonl_bytes

    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_replace(((output, _jsonl_bytes(rows)),))


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("command", choices=("export",))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    export_submission(args.dataset, args.samples, args.output)
    print(json.dumps({"submission": str(args.output), "verified": False}))
    return 0
