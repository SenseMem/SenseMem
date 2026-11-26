"""Remaining PersonaMem-v3 formulas at bowen-upenn/PersonaMem-v3@ad80a3b1b322.

Adapted from evaluation/tasks/{new_suggestions,e6_active_mistake_prevention}.py
and evaluation/metrics.py (MIT, copyright 2026 The PersonaMem-v3 Authors).
Transport, typed input validation, and numeric diagnostic serialization are local.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import cast

from sensemem.benchmarks._official.personamem_v3_prompts import ClusterResponse
from sensemem.benchmarks._official.personamem_v3_scoring import mapping, string

CLUSTER_TASK_TYPES = frozenset(
    {"over_personalization_repetition_chatbot", "over_personalization_repetition_recsys"}
)

_WARN_PHRASE_RE = re.compile(
    r"\b("
    r"warn(?:ing)?|caution|heads?\s*up|be\s+aware|be\s+careful|careful|"
    r"alert|attention|flag|concern(?:ed)?|worried|worry|note\s+that|"
    r"keep\s+in\s+mind|bear\s+in\s+mind|make\s+sure|might\s+want\s+to|"
    r"you\s+may\s+want\s+to|you\s+might\s+want\s+to|"
    r"wanted\s+to\s+let\s+you\s+know|just\s+so\s+you\s+know|"
    r"double[-\s]?check|before\s+you|contradic|conflict|mismatch|"
    r"don'?t\s+forget|heads[-\s]?up|actually[,\s]+"
    r")\b",
    re.IGNORECASE,
)


def response_object(raw: str) -> Mapping[str, object]:
    """Read the released fenced-JSON response format, retaining raw-text fallback."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    candidate = match.group(1).strip() if match else raw.strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed: object = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def strings(source: Mapping[str, object], key: str) -> list[str]:
    value = source.get(key) or ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError(f"PersonaMem-v3 {key} must be a list of strings")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"PersonaMem-v3 {key} must contain strings")
    return list(cast(Sequence[str], value))


def substantive(response: str) -> bool:
    """The pinned evaluator's fifteen-distinct-token engagement gate."""
    return (
        len({token.lower() for token in re.findall(r"[A-Za-z0-9#]+", response) if len(token) >= 3})
        >= 15
    )


def cluster_response(
    raw: str, *, user_query: str, timestamp: int, is_target: bool
) -> ClusterResponse:
    parsed = response_object(raw)
    return ClusterResponse(
        title=string(parsed, "title"),
        caption=string(parsed, "caption"),
        hashtags=strings(parsed, "hashtags"),
        response=string(parsed, "response") or raw,
        user_query=user_query,
        ts=timestamp,
        is_target=is_target,
    )


def cluster_responses(prediction: str) -> tuple[ClusterResponse, ...]:
    rows = response_object(prediction).get("responses", ())
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        raise ValueError("PersonaMem-v3 cluster responses must be a sequence")
    responses = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PersonaMem-v3 cluster responses must contain objects")
        timestamp, target = row.get("ts"), row.get("is_target")
        if not isinstance(timestamp, int) or not isinstance(target, bool):
            raise ValueError("PersonaMem-v3 cluster response needs its timestamp and target flag")
        responses.append(
            ClusterResponse(
                title=string(row, "title"),
                caption=string(row, "caption"),
                hashtags=strings(row, "hashtags"),
                response=string(row, "response"),
                user_query=string(row, "user_query"),
                ts=timestamp,
                is_target=target,
            )
        )
    return tuple(responses)


def repetition_metrics(
    scores: Sequence[float | None], n_allowed_repetitions: int
) -> dict[str, float]:
    """Keep the original head/tail positions when a judge fails on an individual turn."""
    warmup = min(n_allowed_repetitions + 1, len(scores))
    head = [value for value in scores[:warmup] if value is not None]
    tail = [value for value in scores[warmup:] if value is not None]
    result = {
        "n_total": float(len(scores)),
        "n_warmup": float(warmup),
        "n_monitored": float(len(tail)),
        "n_judge_failed": float(sum(value is None for value in scores)),
        "warmup_invocations": float(sum(value < 5 for value in head)),
        "fatigue_violations": float(sum(value < 5 for value in tail)),
        "fatigue_overuse_rate": round(sum((10.0 - value) / 10.0 for value in tail) / len(tail), 3)
        if tail
        else 0.0,
    }
    if tail:
        result.update(
            fatigue_restraint_score=round(sum(tail) / len(tail), 2),
            query_score_0_10=round(sum(tail) / len(tail), 2),
            fatigue_passed=float(all(value >= 7 for value in tail)),
        )
    return result


def suggestion_metrics(prediction: str, evidence: Mapping[str, object]) -> dict[str, float]:
    """Reject saturated topics and suggestions dominated by recently used hashtags."""
    parsed = response_object(prediction)
    tags = parsed.get("hashtags") or []
    proposed = (
        {tag.lstrip("#").lower() for tag in tags if isinstance(tag, str) and tag.strip()}
        if isinstance(tags, list)
        else {match.group(1).lower() for match in re.finditer(r"#([a-z0-9_]+)", prediction.lower())}
    )
    fatigued = {tag.lstrip("#").lower() for tag in strings(evidence, "fatigued_hashtags")}
    leaked = {tag.lstrip("#").lower() for tag in strings(evidence, "leak_set_hashtags")}
    fatigue_overlap = proposed & fatigued
    leak_overlap = proposed & leaked
    hard_fail = bool(fatigue_overlap) or len(leak_overlap) >= max(
        2, (max(1, len(proposed)) + 1) // 2
    )
    result = {
        "valid": float(bool(parsed.get("recommendation") or prediction)),
        "hard_fail": float(hard_fail),
        "fatigue_overlap_count": float(len(fatigue_overlap)),
        "leak_overlap_count": float(len(leak_overlap)),
    }
    if hard_fail:
        result.update(alignment_score=0.0, passed=0.0)
    return result


def mistake_metrics(prediction: str, evidence: Mapping[str, object]) -> dict[str, float]:
    """Score the released warning detector, coverage threshold, and foil engagement gate."""
    parsed = response_object(prediction)
    response = string(parsed, "response") or prediction
    frame = mapping(evidence, "expected_warning_frame")
    must_mention = strings(frame, "must_mention")
    must_not_mention = strings(frame, "must_not_mention")
    regex_warn = bool(_WARN_PHRASE_RE.search(response))
    self_warn = bool(parsed.get("warned"))
    warned = regex_warn or self_warn
    coverage = (
        sum(
            bool(term.strip()) and term.strip().lower() in response.lower() for term in must_mention
        )
        / len(must_mention)
        if must_mention
        else 1.0
    )
    leak = any(
        term.strip() and term.strip().lower() in response.lower() for term in must_not_mention
    )
    polarity = string(evidence, "polarity").lower()
    engaged = substantive(response)
    correct = (
        warned and coverage >= 0.5 and not leak
        if polarity == "warn"
        else engaged and not warned and not leak
        if polarity == "foil"
        else False
    )
    return {
        "warning_issued": float(warned),
        "regex_warn": float(regex_warn),
        "self_warn": float(self_warn),
        "must_mention_coverage": round(coverage, 3),
        "leak": float(leak),
        "correct_warn": float(correct and polarity == "warn"),
        "correct_foil": float(correct and polarity == "foil"),
        "correct": float(correct),
        "is_persona_safety": float(bool(evidence.get("is_persona_safety"))),
        "response_is_substantive": float(engaged),
    }
