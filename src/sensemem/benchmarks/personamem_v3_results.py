"""PersonaMem-v3 paired lifecycle results and warning/foil summary statistics.

Formulas follow tasks/e5_horizon_lifecycle.py and scripts/aggregate_eval.py at
bowen-upenn/PersonaMem-v3@ad80a3b1b322. The official micro uses per-row E6
correct; complete warn/foil pairs additionally report paired_correct and F1.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from sensemem.benchmarks.eval_results import SampleResult


def complete_lifecycle(samples: Sequence[SampleResult]) -> tuple[SampleResult, ...]:
    """Attach each canonical's pre/post delta while retaining both original answer rows."""
    pairs: dict[tuple[str, str, str], dict[str, SampleResult]] = defaultdict(dict)
    for sample in samples:
        if (
            sample.task != "personamem-v3"
            or sample.metadata.get("task_type") != "short_vs_long_term_lifecycle"
        ):
            continue
        evidence = cast(Mapping[str, object], sample.metadata["judge_evidence"])
        phase, canonical = str(evidence.get("phase", "")), str(evidence.get("canonical_id", ""))
        if phase in {"pre", "post"} and canonical:
            pairs[(sample.arm, sample.unit_id, canonical)][phase] = sample
    updates = {}
    for pair in pairs.values():
        complete = set(pair) == {"pre", "post"}
        for sample in pair.values():
            updates[sample.sample_id] = replace(
                sample,
                metadata={**sample.metadata, "lifecycle_pair_complete": complete},
            )
        if not complete:
            continue
        pre, post = pair["pre"], pair["post"]
        if "match_rate_at_3" not in pre.metrics or "match_rate_at_3" not in post.metrics:
            continue
        delta = pre.metrics["match_rate_at_3"] - post.metrics["match_rate_at_3"]
        score = (1.0 + max(-1.0, min(1.0, delta))) / 2.0
        if any(row.error_code is not None or row.ingest_failure_count for row in pair.values()):
            score = 0.0
        for sample in pair.values():
            updates[sample.sample_id] = replace(
                updates[sample.sample_id],
                score=score,
                metrics={**sample.metrics, "lifecycle_score": delta, "personamem_score": score},
            )
    return tuple(updates.get(sample.sample_id, sample) for sample in samples)


def mistake_summary(samples: Sequence[SampleResult]) -> dict[str, object]:
    pairs: dict[tuple[str, str, str], dict[str, SampleResult]] = defaultdict(dict)
    for sample in samples:
        if sample.metadata.get("task_type") != "active_mistake_prevention":
            continue
        evidence = cast(Mapping[str, object], sample.metadata["judge_evidence"])
        polarity, pair_id = str(evidence.get("polarity", "")), str(evidence.get("pair_id", ""))
        if polarity in {"warn", "foil"} and pair_id:
            pairs[(sample.arm, sample.unit_id, pair_id)][polarity] = sample
    warn = [
        pair["warn"].metrics.get("correct_warn", 0.0) for pair in pairs.values() if "warn" in pair
    ]
    foil = [
        pair["foil"].metrics.get("correct_foil", 0.0) for pair in pairs.values() if "foil" in pair
    ]
    paired = [
        float(
            pair["warn"].metrics.get("correct_warn", 0.0) >= 1
            and pair["foil"].metrics.get("correct_foil", 0.0) >= 1
        )
        for pair in pairs.values()
        if set(pair) == {"warn", "foil"}
    ]
    warn_rate = sum(warn) / len(warn) if warn else 0.0
    foil_rate = sum(foil) / len(foil) if foil else 0.0
    return {
        "n_pairs": len(pairs),
        "complete_pairs": len(paired),
        "incomplete_pairs": len(pairs) - len(paired),
        "warn_recall": round(warn_rate, 4),
        "foil_precision": round(foil_rate, 4),
        "paired_correct": round(sum(paired) / len(paired), 4) if paired else None,
        "paired_f1": round(2 * warn_rate * foil_rate / (warn_rate + foil_rate), 4)
        if warn_rate + foil_rate
        else 0.0,
    }
