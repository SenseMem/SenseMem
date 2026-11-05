"""Pinned VideoZeroBench questions, five diagnostic requests, and deterministic scoring.

Adapted from marinero4972/VideoZeroBench@21a46d686302825e85e330c3fa4b64f01df3c9da,
eval/VLMEvalKit-lite/vlmeval/dataset/VideoBench/videozerobench.py (Apache-2.0).
Copyright 2023 VLMEvalKit Authors. See _official/NOTICE.md for attribution and data terms.
The memory harness uses its configured media transport rather than upstream's vLLM frame sampler.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic import ValidationError as SchemaError

from sensemem.benchmarks._contracts import ContractModel, Identifier, NonEmptyString

VIDEOZERO_ADAPTER_VERSION = "videozerobench_memory_21a46d686302_v1"
VIDEOZERO_SCORER_VERSION = "videozerobench_five_level_21a46d686302_v1"
VIDEOZERO_METRICS = (
    "Level-1_acc",
    "Level-2_acc",
    "Level-3_acc",
    "Level-4_mean_tIoU",
    "Level-4_score",
    "Level-5_mean_vIoU",
    "Level-5_score",
)
Interval: TypeAlias = tuple[float, float]
Box: TypeAlias = tuple[float, float, float, float]
_Finite = Annotated[float, Field(allow_inf_nan=False)]
_Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class TemporalEvidence(ContractModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(ge=0, allow_inf_nan=False)


class SpatialEvidence(ContractModel):
    time: float = Field(ge=0, allow_inf_nan=False)
    box: tuple[_Coordinate, _Coordinate, _Coordinate, _Coordinate]


class VideoZeroQuestion(ContractModel):
    question_id: int = Field(ge=0, strict=True)
    question: NonEmptyString
    answer: NonEmptyString
    evidence_windows: tuple[TemporalEvidence, ...]
    evidence_boxes: tuple[SpatialEvidence, ...]
    video: Identifier
    video_id: Identifier
    category: NonEmptyString
    language: Literal["en", "cn"]
    duration: float = Field(gt=0, allow_inf_nan=False)
    evidence_span: NonEmptyString
    annotation_capabilities: tuple[NonEmptyString, ...]


def load_videozerobench(path: Path) -> tuple[VideoZeroQuestion, ...]:
    """Read the released JSON; answers and evidence remain evaluator-owned values."""
    questions = TypeAdapter(tuple[VideoZeroQuestion, ...]).validate_json(path.read_bytes())
    ids = tuple(question.question_id for question in questions)
    if not questions or len(set(ids)) != len(ids):
        raise ValueError("VideoZeroBench question IDs must be non-empty and unique")
    for question in questions:
        if (
            Path(question.video).name != question.video
            or question.video != f"{question.video_id}.mp4"
        ):
            raise ValueError("VideoZeroBench video must be the video_id MP4 filename")
    return questions


def level_queries(question: VideoZeroQuestion) -> tuple[tuple[int, str], ...]:
    """Preserve the hint boundary: L1 time/boxes, L2 time, L3/L4 none, L5 key times."""
    header = f"Question: {question.question}"
    temporal = "; ".join(
        f"From <{window.start:.2f} seconds> to <{window.end:.2f} seconds>"
        for window in question.evidence_windows
    )
    spatial = "; ".join(
        f"Time=<{item.time:.2f} seconds>, Normalized Box="
        f"[{','.join(str(int(value * 1000)) for value in item.box)}]"
        for item in question.evidence_boxes
    )
    answer_instruction = (
        "请直接输出问题的最终答案。"
        if question.language == "cn"
        else "Please directly output the final answer."
    )
    queries = []
    for level in (1, 2, 3):
        lines = [header]
        if level <= 2 and temporal:
            lines.append(f"The temporal evidence for answering the question is: {temporal}.")
        if level == 1 and spatial:
            lines.append(f"The spatial evidence for answering the question is: {spatial}.")
        lines.append(answer_instruction)
        queries.append((level, "\n".join(lines)))
    if question.evidence_windows:
        queries.append(
            (
                4,
                f"{header}\n"
                "Task: Find one or more of the most important key time ranges (no more than "
                "20 segments) in the video, that provide sufficient evidence to answer the "
                "question.\nOutput format: (Example)\n"
                "From <start_timestamp_1 seconds> to <end_timestamp_1 seconds>. "
                "From <start_timestamp_2 seconds> to <end_timestamp_2 seconds>.\nRules:\n"
                "- Output ONLY the time ranges in the specified sentence format. "
                "No explanation, no extra words. Do not answer the original question.\n"
                "- Each segment must follow exactly: 'From <X seconds> to <Y seconds>.'\n"
                "- Use absolute time in seconds (NOT MM:SS format).\n"
                "- Use decimal numbers if necessary (e.g., 12.35).\n"
                "- Separate segments by a single space.\n"
                "- Ensure each start_timestamp < end_timestamp.\n",
            )
        )
    if question.evidence_boxes:
        times = ", ".join(
            f"<{time:.2f} seconds>"
            for time in dict.fromkeys(b.time for b in question.evidence_boxes)
        )
        queries.append(
            (
                5,
                f"{header}\nGiven key time points (absolute seconds): {times}\n"
                "Task: For each provided time point, output 1 or more 2D bounding boxes "
                "that are relevant evidence for answering the question.\n"
                "Output format: a JSON array of objects.\n"
                '[{"time": ..., "bbox_2d":[[...],[...],...]}, ...]\nRules:\n'
                "- Output ONLY valid JSON. No markdown fences, no explanation text. "
                "Do not need to answer the original question.\n"
                "- The length of the json data should be consistent with the number of key "
                "time points provided.\n"
                "- Each object's 'time' MUST be one of the provided time points (in seconds).\n"
                "- 'bbox_2d' MUST be a list of one or more boxes.\n"
                "- Each box is normalized coordinates in [0,1000]: "
                "[x_min, y_min, x_max, y_max].\n",
            )
        )
    return tuple(queries)


def _unfence(value: str) -> str:
    match = re.search(r"```(?:json|python|bash|text)?\s*\n(.*?)\n```", value, re.I | re.S)
    return (match[1] if match else value).strip()


def answer_correct(reference: str, prediction: str) -> bool:
    """Match upstream's bilingual exact-answer rules, including its Chinese special cases."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", prediction, re.I | re.S)
    prediction = match[1] if match else prediction
    reference, prediction = (
        re.sub(r'^[\s"\'“”\u2018\u2019]+|[\s"\'“”\u2018\u2019\.\。]+$', "", _unfence(text))
        for text in (reference, prediction)
    )
    if not reference:
        return False
    if re.fullmatch(r"\d+", reference):
        return prediction == reference
    if re.search(r"[A-Za-z]", reference):
        return prediction.lower() == reference.lower()
    if "色" in reference:
        return prediction in reference
    if reference == "车":
        return reference in prediction
    return reference == prediction


def _merged(intervals: Sequence[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _length(intervals: Sequence[Interval]) -> float:
    return sum(end - start for start, end in _merged(intervals))


def temporal_iou(evidence: Sequence[TemporalEvidence], prediction: str) -> float:
    """Union IoU in absolute seconds; zero-length release windows do not enter its denominator."""
    gold = [(item.start, item.end) for item in evidence if item.end > item.start]
    text = re.sub(r"[<>]", "", _unfence(prediction))
    token = r"(?:\d+:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?)"
    predicted = []
    for pattern in (
        rf"(?is)\b(?:from\s+)?({token})\s*(?:[^\d:]+)?\s*to\s*({token})\b",
        rf"(?is)\b({token})\s*[-\u2013\u2014~]\s*({token})\b",
    ):
        for start, end in re.findall(pattern, text):
            pair = tuple(_seconds(value) for value in (start, end))
            if pair[0] is not None and pair[1] is not None and pair[1] > pair[0]:
                predicted.append((pair[0], pair[1]))
    intersection = _length(
        [(max(a, c), min(b, d)) for a, b in gold for c, d in predicted if min(b, d) > max(a, c)]
    )
    union = _length(gold) + _length(predicted) - intersection
    return intersection / union if union > 0 else 0.0


def _seconds(value: str) -> float | None:
    if ":" not in value:
        absolute_seconds = float(value)
        return absolute_seconds if math.isfinite(absolute_seconds) else None
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + float(seconds) if float(seconds) < 60 else None


class _SpatialPrediction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    time: _Finite
    bbox_2d: (
        tuple[tuple[_Finite, _Finite, _Finite, _Finite], ...]
        | tuple[_Finite, _Finite, _Finite, _Finite]
    )


def _rectangle(box: Sequence[float]) -> Box | None:
    x1, y1, x2, y2 = (max(0.0, min(1.0, value)) for value in box)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _area(rectangles: Sequence[Box]) -> float:
    edges = sorted({value for box in rectangles for value in (box[0], box[2])})
    return sum(
        (right - left)
        * _length([(y1, y2) for x1, y1, x2, y2 in rectangles if x1 < right and x2 > left])
        for left, right in pairwise(edges)
    )


def spatial_iou(evidence: Sequence[SpatialEvidence], prediction: str) -> float:
    """Mean union-box IoU across gold key times, matching upstream's 0-1000 coordinate mode."""
    try:
        parsed = TypeAdapter(tuple[_SpatialPrediction, ...] | _SpatialPrediction).validate_json(
            _unfence(prediction)
        )
    except SchemaError:
        return 0.0
    entries = (parsed,) if isinstance(parsed, _SpatialPrediction) else parsed
    predicted: dict[float, list[Box]] = {}
    for entry in entries:
        boxes = entry.bbox_2d
        if not boxes:
            return 0.0
        raw_boxes = (boxes,) if isinstance(boxes[0], float) else boxes
        rectangles = []
        for box in raw_boxes:
            rectangle = _rectangle(tuple(value / 1000 for value in box))
            if rectangle is not None:
                rectangles.append(rectangle)
        predicted[round(entry.time, 2)] = rectangles
    gold: dict[float, list[Box]] = {}
    for item in evidence:
        rectangles = gold.setdefault(round(item.time, 2), [])
        rectangle = _rectangle(item.box)
        if rectangle is not None:
            rectangles.append(rectangle)
    scores = []
    for time, gold_boxes in gold.items():
        predicted_boxes = predicted.get(time, [])
        intersections = [
            overlap
            for a in gold_boxes
            for b in predicted_boxes
            if (
                overlap := _rectangle(
                    (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
                )
            )
            is not None
        ]
        intersection = _area(intersections)
        union = _area(gold_boxes) + _area(predicted_boxes) - intersection
        scores.append(intersection / union if union else float(not gold_boxes))
    return sum(scores) / len(scores) if scores else 1.0


def local_metrics(
    level: int, reference: str, prediction: str, metadata: Mapping[str, object]
) -> dict[str, float]:
    """Score one independent request; task aggregation joins L3 correctness to L4/L5 IoU."""
    if level <= 3:
        return {"accuracy": float(answer_correct(reference, prediction))}
    if level == 4:
        windows = TypeAdapter(tuple[TemporalEvidence, ...]).validate_python(
            metadata["evidence_windows"]
        )
        return {"temporal_iou": temporal_iou(windows, prediction)}
    boxes = TypeAdapter(tuple[SpatialEvidence, ...]).validate_python(metadata["evidence_boxes"])
    return {"spatial_iou": spatial_iou(boxes, prediction)}
