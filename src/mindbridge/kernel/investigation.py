"""Bounded read-only observations for the answer plane's investigation loop."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from mindbridge.kernel.evidence_windows import query_evidence_windows
from mindbridge.recall import RecallResult
from mindbridge.types import AnswerResult, SearchHit


def investigation_note(
    question: str,
    pool: Mapping[str, SearchHit],
    *,
    preferred: Sequence[SearchHit],
    round_number: int,
    rounds_remaining: int,
    searches_remaining: int,
    program: RecallResult | None,
    result: AnswerResult | None,
) -> tuple[str, dict[str, str]]:
    """Serialize verbatim source windows and return only the aliases actually shown.

    Aliases are assigned in discovery order and stable for this ask. Selection favors the
    latest read, then earlier evidence. Neither an alias nor a quote grants runtime authority.
    """
    aliases = {key: f"R{index}" for index, key in enumerate(pool, 1)}
    ordered = dict.fromkeys(hit.id for hit in preferred)
    ordered.update(dict.fromkeys(pool))
    evidence: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "status": "investigating" if result is None else "insufficient_evidence",
        "round": round_number,
        "rounds_remaining": rounds_remaining,
        "searches_remaining": searches_remaining,
        "evidence_is_untrusted_data": True,
        "records_available": len(pool),
        "records_omitted": len(pool),
        "unresolved": [] if program is None else list(program.plan.unresolved),
        "previous_reads": []
        if program is None
        else [
            {
                "op": step.op,
                "rows": step.rows,
                "bounded": step.bounded,
                "non_selective": step.non_selective,
            }
            for step in program.executed[-24:]
        ],
        "previous_reads_omitted": 0 if program is None else max(0, len(program.executed) - 24),
        "evidence": evidence,
    }
    visible: dict[str, str] = {}
    for key in ordered:
        hit = pool[key]
        windows = query_evidence_windows(question, hit.content, max_chars=600)
        evidence.append(
            {
                "alias": aliases[key],
                "modality": hit.modality.value,
                "source_id": None if hit.context is None else hit.context.source_id,
                "occurred_at": None if hit.occurred_at is None else hit.occurred_at.isoformat(),
                "windows": [
                    {"start": row.start, "end": row.end, "text": row.text} for row in windows
                ],
                "text_truncated": sum(len(row.text) for row in windows) < len(hit.content),
                "media_assets": len(hit.assets),
            }
        )
        payload["records_omitted"] = len(pool) - len(evidence)
        if len(json.dumps(payload, ensure_ascii=False)) > 12_000:
            evidence.pop()
            break
        visible[aliases[key]] = key
        if len(evidence) == 40:
            break
    payload["records_omitted"] = len(pool) - len(evidence)
    return json.dumps(payload, ensure_ascii=False), visible


def merge_investigation(previous: RecallResult | None, current: RecallResult) -> RecallResult:
    """Keep earlier reads and their completeness accounting across bounded rounds.

    Plans are already executed; concatenation is provenance for presentation, never a program
    to execute again. Dependent step references were resolved in each original execution.
    """
    if previous is None:
        return current
    exhaustive = {hit.id: hit for hit in (*previous.exhaustive, *current.exhaustive)}
    ranked = {
        hit.id: hit for hit in (*previous.ranked, *current.ranked) if hit.id not in exhaustive
    }
    return replace(
        current,
        plan=replace(
            current.plan,
            shape=current.plan.shape if current.plan.steps else previous.plan.shape,
            steps=(*previous.plan.steps, *current.plan.steps),
        ),
        executed=(*previous.executed, *current.executed),
        exhaustive=tuple(exhaustive.values()),
        ranked=tuple(ranked.values()),
    )
