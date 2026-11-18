"""Official answer scorers used by the unified benchmark runner."""

from __future__ import annotations

import ast
import json
import math
import re
import string
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol, cast

from sensemem.benchmarks._official import ama_scoring, mosaic_scoring
from sensemem.benchmarks._official import personamem_v3_proactive as pm3_proactive
from sensemem.benchmarks._official import personamem_v3_scoring as pm3
from sensemem.benchmarks._official import personamem_v3_tasks as pm3_tasks
from sensemem.benchmarks._official.atm_score import (
    deterministic_accuracy as atm_deterministic_accuracy,
)
from sensemem.benchmarks._official.atm_score import list_jaccard_score
from sensemem.benchmarks._official.icm_prompt import SEMANTIC_EQUIVALENCE_PROMPT
from sensemem.benchmarks._official.judge_prompts import (
    _ATM_PROMPT,
    _EGOTEMPO_PROMPT,
    _GALLERY_PROMPT,
    _LOCOMO_PROMPT,
    _M3_PROMPT,
    _M3_SYSTEM_PROMPT,
    _MM_LIFELONG_SYSTEM,
    _MM_LIFELONG_USER_PROMPT,
)
from sensemem.benchmarks._official.longmemeval_v2_scoring import (
    SCORER_VERSION as LONGMEMEVAL_V2_SCORER_VERSION,
)
from sensemem.benchmarks._official.longmemeval_v2_scoring import (
    deterministic_score as longmemeval_v2_score,
)
from sensemem.benchmarks._official.longmemeval_v2_scoring import (
    judge_messages as longmemeval_v2_messages,
)
from sensemem.benchmarks._official.longmemeval_v2_scoring import (
    parse_eval_spec as longmemeval_v2_rule,
)
from sensemem.benchmarks._official.longmemeval_v2_scoring import (
    parse_judgement as parse_longmemeval_v2,
)
from sensemem.benchmarks._official.memlens_prompts import (
    build_judge_prompt as build_memlens_prompt,
)
from sensemem.benchmarks._official.memlens_prompts import get_task_key
from sensemem.benchmarks._official.personamem_v3_prompts import (
    _C1D_PREF_INVOCATION_JUDGE_PROMPT,
    judge_new_suggestions_chatbot_prompt,
)
from sensemem.benchmarks.mosaic import (
    MOSAIC_SCORER_VERSION,
    mosaic_refusal,
)
from sensemem.benchmarks.personamem_v3 import RANKING_TASK_TYPES
from sensemem.benchmarks.videozerobench import (
    VIDEOZERO_METRICS,
    VIDEOZERO_SCORER_VERSION,
    local_metrics,
)

SCORER_VERSION = "official_scorers_v8"


class _Stemmer(Protocol):
    def stem(self, word: str) -> str: ...


@dataclass(frozen=True, slots=True)
class JudgeMessage:
    """One portable OpenAI-compatible judge message."""

    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True, slots=True)
class JudgePlan:
    """The exact upstream judge calls and parser for one answer."""

    protocol: str
    parser: Literal[
        "locomo",
        "m3",
        "icm",
        "egotempo",
        "memlens",
        "mm_lifelong",
        "atm",
        "gallery",
        "longmemeval_v2",
        "personamem_v3",
        "ama_bench",
        "mosaic",
    ]
    calls: tuple[tuple[JudgeMessage, ...], ...]
    max_tokens: int | None = None
    extra_body: Mapping[str, object] | None = None
    details: Mapping[str, str] = field(default_factory=dict)
    candidate_metrics: tuple[Mapping[str, float], ...] = ()


_PROTOCOLS = {
    "mosaic": MOSAIC_SCORER_VERSION,
    "ama-bench": ama_scoring.SCORER_VERSION,
    "icm-bench": "icm_semantic_equivalence_10f02babe3c7_user_only_v2",
    "locomo-refined": "locomo_refined_judge_887091190789",
    "m3-bench": "m3_agent_judge_0e3e41939bd8_system_v1",
    "videozerobench": VIDEOZERO_SCORER_VERSION,
    "egotempo": "egotempo_gemini_judge_7022ba77b4d8",
    "memlens": "memlens_judge_77f3ab9a52fa",
    "mm-lifelong": "mm_lifelong_judge_248aa82039a5",
    "supermemory-vqa": "supermemory_vqa_metrics_1d228e0f1004",
    "atm-bench": "atm_bench_scorer_ef4e5dff1a47_minimal_v3",
    "mem-gallery": "mem_gallery_scorer_a93959e1e978_v2",
    "longmemeval-v2": LONGMEMEVAL_V2_SCORER_VERSION,
    "personamem-v3": "personamem_v3_ad80a3b1b322_v3",
}

_OFFICIAL_JUDGE_MODELS = {
    "mosaic": "qwen3.7-plus",
    "ama-bench": "qwen3-32b",
    "icm-bench": "gemini-3-flash-preview",
    "locomo-refined": "qwen3-14b",
    "m3-bench": "gpt-4o-2024-11-20",
    "egotempo": "gemini-1.5-flash",
    "memlens": "qwen3-235b-judge",
    "mm-lifelong": "gpt-5",
    "atm-bench": "gpt-5-mini",
    "mem-gallery": "qwen2.5-72b-instruct",
    "longmemeval-v2": "gpt-5.2",
    # `EVAL.md`: "the judge is a QueryLLM client on Azure gpt-5.5".
    "personamem-v3": "gpt-5.5",
    # `common_configurations.py` supplies this ID to every released QA script.
}

# Metrics the pinned upstream protocol reports itself. `metric_is_official` reads this registry
# positively: a metric absent from its family's entry is a SenseMem diagnostic, never an
# upstream number. The families that publish no deterministic metric beyond their judge output
# are still listed, so an empty entry means "judge metrics only" rather than "not yet audited".
_OFFICIAL_METRICS: dict[str, frozenset[str]] = {
    "mosaic": frozenset(
        {"judge_score", "correct_refusal_rate", "false_refusal_rate", "em", "f1", "bleu1"}
    ),
    "ama-bench": frozenset({"accuracy", "llm_judge_score", "exact_match", "token_f1"}),
    "icm-bench": frozenset({"accuracy"}),
    # `evaluate.py` keeps the F1 and BLEU-1 of the accepted reference beside its judge label.
    "locomo-refined": frozenset({"llm_judge", "token_f1", "bleu_1"}),
    "m3-bench": frozenset({"accuracy"}),
    "videozerobench": frozenset({"accuracy", *VIDEOZERO_METRICS}),
    "egotempo": frozenset({"accuracy", "judge_score_0_5"}),
    "memlens": frozenset({"accuracy"}),
    # `Ref@300` is MM-Lifelong's own quantized temporal IoU.
    "mm-lifelong": frozenset({"answer_accuracy", "judge_score_0_5", "ref_at_300"}),
    "supermemory-vqa": frozenset({"qa_accuracy", "answerability"}),
    # `_official/atm_score.py` publishes `deterministic_accuracy` and `list_jaccard_score`, both
    # reported here as `accuracy`, plus the open-ended judge. Its retrieval and joint metrics are
    # SenseMem diagnostics and are deliberately absent.
    "atm-bench": frozenset({"accuracy"}),
    "mem-gallery": frozenset(
        {
            "llm_judge",
            "f1",
            "bleu",
            "bleu_1",
            "bleu_2",
            "exact_match",
            "retrieval_precision@10",
            "retrieval_recall@10",
            "retrieval_hit@10",
        }
    ),
    "longmemeval-v2": frozenset({"accuracy"}),
    # Everything `_official/personamem_v3_scoring.py` emits: the unified rubric's `pr_` dims
    # (see `_OFFICIAL_METRIC_PREFIXES`), the four task-specific judges' coercions, the ranking
    # metrics the task runners report, and the 0-100 micro aggregate headline.
    "personamem-v3": frozenset(
        {
            "accuracy_pct_micro",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "hit_at_1",
            "hit_at_3",
            # The @ai directive runner retains these historical spellings.
            "recall@3",
            "recall@5",
            "hit@1",
            "mrr",
            "ndcg_at_5",
            "sycophancy_caved",
            "sycophancy_resistance_0_10",
            "used_outdated_stance",
            "preference_shift_consistency",
            "deep_motivation_alignment",
            "surface_query_satisfaction",
            "preference_alignment",
            "telegraph_avoidance_fail",
            "privacy_leak_fail",
            "abstention_quality_0_10",
            "current_city_grounded",
            "stale_geo_anchor",
            "geo_neutral_response",
            "geo_shift_correctness",
            "non_substantive_response",
            "decision_correct",
            "decision_score",
            "content_word_count",
            "content_length_ok",
            "evidence_cited",
            "trigger_detection_correctness",
            "justification_quality",
            "avoid_overpersonalization",
            "voice_match",
            "restraint_justification",
            "negative_leakage",
            "stale_preference_use",
            "justification_score",
            "proactive_action_score",
            "alignment_score",
            "passed",
            "hard_fail",
            "valid",
            "query_score_0_10",
            "fatigue_restraint_score",
            "fatigue_overuse_rate",
            "fatigue_passed",
            "fatigue_violations",
            "warmup_invocations",
            "n_total",
            "n_warmup",
            "n_monitored",
            "n_judge_failed",
            "match_rate_at_3",
            "matching_in_top1",
            "mrr_matching",
            "lifecycle_score",
            "warning_issued",
            "regex_warn",
            "self_warn",
            "must_mention_coverage",
            "leak",
            "correct_warn",
            "correct_foil",
            "correct",
            "is_persona_safety",
            "response_is_substantive",
            "paired_correct",
        }
    ),
}

# One family scores an open dimension vocabulary upstream: `score_unified_rubric` names every
# rubric dim it was given, so the prefix it stamps them with is the registry entry.
_OFFICIAL_METRIC_PREFIXES: dict[str, tuple[str, ...]] = {"personamem-v3": ("pr_",)}

_JUDGE_METRICS = {
    "mosaic": frozenset({"judge_score"}),
    "ama-bench": frozenset({"accuracy", "llm_judge_score"}),
    "icm-bench": frozenset({"accuracy"}),
    "locomo-refined": frozenset({"llm_judge"}),
    "m3-bench": frozenset({"accuracy"}),
    "egotempo": frozenset({"accuracy", "judge_score_0_5"}),
    "memlens": frozenset({"accuracy"}),
    "mm-lifelong": frozenset({"answer_accuracy", "judge_score_0_5"}),
    "atm-bench": frozenset({"accuracy"}),
    "mem-gallery": frozenset({"llm_judge"}),
    "longmemeval-v2": frozenset({"accuracy"}),
    "personamem-v3": frozenset(
        {
            "accuracy_pct_micro",
            "sycophancy_caved",
            "sycophancy_resistance_0_10",
            "used_outdated_stance",
            "preference_shift_consistency",
            "deep_motivation_alignment",
            "surface_query_satisfaction",
            "preference_alignment",
            "telegraph_avoidance_fail",
            "privacy_leak_fail",
            "abstention_quality_0_10",
            "trigger_detection_correctness",
            "justification_quality",
            "avoid_overpersonalization",
            "voice_match",
            "restraint_justification",
            "negative_leakage",
            "stale_preference_use",
            "justification_score",
            "proactive_action_score",
            "alignment_score",
            "passed",
            "query_score_0_10",
            "fatigue_restraint_score",
            "fatigue_overuse_rate",
            "fatigue_passed",
            "fatigue_violations",
            "warmup_invocations",
        }
    ),
}


# PersonaMem-v3 headline per task type, and the divisor that maps it onto the
# internal 0-1 row value from which the official micro percentage is computed
# (`evaluation/task_registry.py: PRIMARY_METRIC`).
_PERSONAMEM_HEADLINES: dict[str, tuple[str, float]] = {
    "chatbot_personalized_response": ("pr_preference_alignment_score", 10.0),
    "over_personalization_sycophancy": ("sycophancy_resistance_0_10", 10.0),
    "preference_shift_followthrough": ("preference_shift_consistency", 10.0),
    "personal_qa_hallucination": ("abstention_quality_0_10", 10.0),
    "hidden_persona_implicit_qa": ("deep_motivation_alignment", 3.0),
    "local_recommendation_geo_shift": ("geo_shift_correctness", 1.0),
    "new_suggestions_chatbot": ("passed", 1.0),
    "active_mistake_prevention": ("correct", 1.0),
    "over_personalization_repetition_chatbot": ("query_score_0_10", 10.0),
    "over_personalization_repetition_recsys": ("query_score_0_10", 10.0),
}
_PERSONAMEM_RUBRIC_HEADLINE = ("pr_combined_personalization_score", 10.0)
# `task_registry.PRIMARY_METRIC` -- the column upstream's aggregator reads --
# gives all three single-target ranking tasks a graded nDCG@5, chosen over a
# binary top-1 because the gold is "subtle by design" and nDCG rewards
# surfacing it high with a smooth position discount. `compute_ranking_metrics`
# also reports top-k diagnostics, but those are not the aggregator headline.
# The shared graded metric uses positive +2, filler +1, and negative -2.
_PERSONAMEM_RANKING_HEADLINE = ("ndcg_at_5", 1.0)

# Ranks a slate like the other three, but its upstream headline is
# `lifecycle_score` -- the delta between a paired pre-row's and post-row's
# match rate -- which no single row can carry. It also has 2 to 5 targets per
# row, so a top-1 recall, which divides by the target count, could never exceed
# 0.5 there. Phase rows acquire their headline only after lifecycle pairing.
_PERSONAMEM_PAIRED_RANKING = frozenset({"short_vs_long_term_lifecycle"})
_RANKED_INDEXES = re.compile(r"ranked[ _]index(?:es|ices)?\s*[:=]?\s*\[([^\]]*)\]", re.IGNORECASE)
_RANKED_JSON = re.compile(r'"ranked_indices"\s*:\s*\[([^\]]*)\]')

_LOCOMO_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def task_family(task: str) -> str | None:
    """Return the benchmark family one task name belongs to, if it has one."""
    return _family_or_none(task)


def scorer_protocol(task: str) -> str:
    """Return the pinned scorer protocol recorded in result artifacts."""
    family = _family_or_none(task)
    return "custom_diagnostic" if family is None else _PROTOCOLS[family]


def task_primary_metric(task: str) -> str:
    """Return the benchmark's leaderboard-facing primary metric."""
    family = _family_or_none(task)
    if family is None:
        return "accuracy"
    return {
        "mosaic": "judge_score",
        "ama-bench": "accuracy",
        "icm-bench": "accuracy",
        "locomo-refined": "llm_judge",
        "m3-bench": "accuracy",
        "videozerobench": "accuracy",
        "egotempo": "accuracy",
        "memlens": "accuracy",
        "mm-lifelong": "answer_accuracy",
        "supermemory-vqa": "qa_accuracy",
        "atm-bench": "accuracy",
        "mem-gallery": "f1",
        "longmemeval-v2": "accuracy",
        "personamem-v3": "accuracy_pct_micro",
    }[family]


def sample_primary_metric(task: str) -> str:
    """Return the per-question metric used for comparisons and grouped scoring."""
    family = _family_or_none(task)
    if family == "personamem-v3":
        return "personamem_score"
    return task_primary_metric(task)


def official_judge_model(task: str) -> str | None:
    """Return the model that makes an LLM-judge metric publication-comparable."""
    family = _family_or_none(task)
    return None if family is None else _OFFICIAL_JUDGE_MODELS.get(family)


def judge_model_is_official(task: str, model: str) -> bool:
    """Match provider-qualified model IDs without accepting nearby model sizes."""
    expected = official_judge_model(task)
    if expected is None:
        return True
    actual_key = _model_key(model)
    expected_key = _model_key(expected)
    if _family_or_none(task) == "memlens":
        return actual_key.endswith(expected_key) or "qwen3vl235b" in actual_key
    return actual_key.endswith(expected_key)


def metric_is_upstream(task: str, metric: str) -> bool:
    """Report whether the pinned upstream protocol publishes this metric at all."""
    family = _family_or_none(task)
    if family is None:
        return False
    if metric in _OFFICIAL_METRICS.get(family, frozenset()):
        return True
    return metric.startswith(_OFFICIAL_METRIC_PREFIXES.get(family, ()))


def metric_is_official(task: str, metric: str, judge_model: str, *, uses_judge: bool) -> bool:
    """Report protocol fidelity per metric instead of blessing a whole task at once.

    A metric is official only when upstream publishes it and, for judged metrics, the required
    judge produced it. Metrics this harness invents -- the retrieval and joint diagnostics above
    all -- are never official, however faithful the rest of the protocol was.
    """
    if not metric_is_upstream(task, metric):
        return False
    family = _family_or_none(task) or ""
    requires_judge = metric in _JUDGE_METRICS.get(family, ()) or (
        family == "personamem-v3" and metric.startswith("pr_")
    )
    if not requires_judge or not uses_judge:
        return True
    return judge_model_is_official(task, judge_model)


# Gold retrieval labels, in the order one question's label is looked up. `evidence_ids` and
# `clue_ids` name gold sources one by one; `evidence_groups` nests one sequence of source IDs per
# gold group, any one of which retrieved counts -- a release that labels the answering session
# while one memory is one turn (MEMLENS) publishes exactly that.
GOLD_SOURCE_KEYS = ("evidence_ids", "clue_ids", "evidence_groups")


def gold_source_groups(metadata: Mapping[str, object], key: str) -> tuple[tuple[str, ...], ...]:
    """Return one label as groups of source IDs, any one of which is a hit, in label order.

    A flat label is one singleton group per ID, so exact set recall is the special case of
    group recall where every group has one member and the two formulas coincide. Blank members
    are not IDs and are dropped, so a label of blanks is no label.
    """
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    groups: list[tuple[str, ...]] = []
    for item in value:
        members = (
            item if isinstance(item, Sequence) and not isinstance(item, str | bytes) else (item,)
        )
        group = tuple(dict.fromkeys(str(member) for member in members if str(member).strip()))
        if group:
            groups.append(group)
    return tuple(dict.fromkeys(groups))


def retrieval_gold_ids(task: str, metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Return the gold source IDs one question's retrieval diagnostics score against.

    Any adapter that labels a question with stored source IDs gets the ranked retrieval query and
    a measured recall, rather than a recall number computed from the generator's citations.
    Groups are flattened: every member is a source the ranked query has to be able to reach.
    """
    del task
    for key in GOLD_SOURCE_KEYS:
        sources = tuple(
            dict.fromkeys(member for group in gold_source_groups(metadata, key) for member in group)
        )
        if sources:
            return sources
    return ()


def local_scores(  # noqa: C901 - direct task dispatch mirrors official scorer families
    task: str,
    *,
    score_kind: str,
    prediction: str,
    parsed_choice: str | None,
    expected_choice: str | None,
    references: Sequence[str],
    question: str | None,
    metadata: Mapping[str, object],
    evidence_source_ids: Sequence[str],
) -> dict[str, float]:
    """Calculate every official deterministic per-question metric."""
    if score_kind == "submission":
        return {}
    family = _family_or_none(task)
    if family is None:
        if score_kind == "choice":
            return {"accuracy": float(parsed_choice == expected_choice)}
        return {"token_f1": _locomo_f1(prediction, references[0])}
    if family == "mosaic":
        refusal_metric = (
            "correct_refusal_rate"
            if metadata["answerability"] == "unanswerable"
            else "false_refusal_rate"
        )
        return {
            **mosaic_scoring.text_metrics(prediction, references),
            refusal_metric: float(mosaic_refusal(prediction)),
        }
    if score_kind == "choice":
        metric = "qa_accuracy" if family == "supermemory-vqa" else "accuracy"
        score = float(parsed_choice == expected_choice)
        return {metric: score}
    if family == "ama-bench":
        normalized = ama_scoring.extract_answer(prediction)
        return {
            "exact_match": ama_scoring.exact_match(normalized, references[0]),
            "token_f1": ama_scoring.f1_score(normalized, references[0]),
        }
    if family == "videozerobench":
        return local_metrics(cast(int, metadata["level"]), references[0], prediction, metadata)
    if family == "longmemeval-v2":
        v2_score = longmemeval_v2_score(
            longmemeval_v2_rule(str(metadata["eval_function"])), prediction, references[0]
        )
        return {} if v2_score is None else {"accuracy": v2_score}
    if family == "locomo-refined":
        normalized = _locomo_prediction(prediction)
        if any(normalized == reference for reference in references):
            return {"llm_judge": 1.0, "token_f1": 1.0, "bleu_1": 1.0}
        if not normalized and any(references):
            return {"llm_judge": 0.0, "token_f1": 0.0, "bleu_1": 0.0}
        return {
            "token_f1": max(_locomo_f1(normalized, reference) for reference in references),
            "bleu_1": max(
                _bleu_1(normalized, reference, _locomo_tokens) for reference in references
            ),
        }
    if family in {"m3-bench", "icm-bench"} and not prediction:
        return {"accuracy": 0.0}
    if family == "memlens" and len(_memlens_prediction(prediction).split()) > 500:
        return {"accuracy": 0.0}
    if family == "atm-bench":
        qtype = str(metadata.get("qtype", ""))
        reference = references[0]
        if qtype == "number":
            scores = {
                "accuracy": float(atm_deterministic_accuracy(reference, prediction, question))
            }
        elif qtype == "list_recall":
            scores = {"accuracy": float(list_jaccard_score(reference, prediction))}
        else:
            scores = {}
        scores.update(_atm_retrieval(metadata, evidence_source_ids))
        return finalize_scores(task, scores)
    if family == "personamem-v3":
        return _personamem_local(prediction, metadata)
    if family == "mem-gallery":
        reference = references[0]
        scores = {
            "f1": _gallery_f1(prediction, reference),
            "bleu": _gallery_bleu(prediction, reference, (0.25, 0.25, 0.25, 0.25)),
            "bleu_1": _gallery_bleu(prediction, reference, (1.0, 0.0, 0.0, 0.0)),
            "bleu_2": _gallery_bleu(prediction, reference, (0.5, 0.5, 0.0, 0.0)),
            "exact_match": float(
                _mem_gallery_normalize(prediction) == _mem_gallery_normalize(reference)
            ),
        }
        scores.update(_gallery_retrieval(metadata, evidence_source_ids))
        return scores
    return {}


def finalize_scores(task: str, scores: Mapping[str, float]) -> dict[str, float]:
    """Add official metrics that combine answer and retrieval correctness."""
    result = dict(scores)
    if _family_or_none(task) == "atm-bench" and "accuracy" in result:
        for cutoff in (5, 10):
            recall = result.get(f"retrieval_recall@{cutoff}")
            if recall is not None:
                result[f"joint_partial@{cutoff}"] = result["accuracy"] * recall
                result[f"joint_strict@{cutoff}"] = result["accuracy"] * float(recall == 1.0)
    return result


def judge_failure_scores(task: str, scores: Mapping[str, float]) -> dict[str, float]:
    """Zero unavailable judge verdicts while retaining deterministic answer metrics."""
    family = _family_or_none(task)
    result = dict(scores)
    for metric in _JUDGE_METRICS.get(family or "", ()):
        result[metric] = 0.0
    if family == "personamem-v3":
        result.update((metric, 0.0) for metric in result if metric.startswith("pr_"))
    primary = sample_primary_metric(task)
    if family == "personamem-v3" or primary not in result:
        result[primary] = 0.0
    return finalize_scores(task, result)


def judge_plan(  # noqa: C901 - direct task dispatch keeps official protocols auditable
    task: str,
    *,
    question: str | None,
    references: Sequence[str],
    prediction: str,
    metadata: Mapping[str, object],
) -> JudgePlan | None:
    """Build the pinned upstream judge prompt for one generated answer."""
    family = _family_or_none(task)
    if family is None:
        return None
    if family not in _JUDGE_METRICS:
        return None
    if family in {"m3-bench", "icm-bench"} and not prediction:
        return None
    if family == "atm-bench" and metadata.get("qtype") != "open_end":
        return None
    if family == "personamem-v3" and not _personamem_judged(metadata):
        return None
    if family == "memlens" and len(_memlens_prediction(prediction).split()) > 500:
        return None
    if question is None:
        raise ValueError(f"{task} official judge needs the original question")
    protocol = scorer_protocol(task)
    if family == "mosaic":
        return JudgePlan(
            protocol,
            "mosaic",
            (
                (
                    JudgeMessage("system", mosaic_scoring.JUDGE_SYSTEM_PROMPT),
                    JudgeMessage(
                        "user",
                        mosaic_scoring.judge_user_prompt(
                            question, references, prediction, str(metadata["answerability"])
                        ),
                    ),
                ),
            ),
            4096,
            {"thinking": {"type": "disabled"}, "enable_thinking": False},
        )
    if family == "locomo-refined":
        normalized = _locomo_prediction(prediction)
        if not normalized or any(normalized == reference for reference in references):
            return None
        calls = tuple(
            (
                JudgeMessage(
                    "user",
                    _LOCOMO_PROMPT.format(
                        question=question,
                        gold_answer=reference,
                        generated_answer=normalized,
                    ),
                ),
            )
            for reference in references
        )
        candidate_metrics = tuple(
            {
                "token_f1": _locomo_f1(normalized, reference),
                "bleu_1": _bleu_1(normalized, reference, _locomo_tokens),
            }
            for reference in references
        )
        return JudgePlan(
            protocol,
            "locomo",
            calls,
            extra_body={"enable_thinking": False},
            candidate_metrics=candidate_metrics,
        )
    reference = references[0]
    if family == "ama-bench":
        normalized = ama_scoring.extract_answer(prediction)
        return JudgePlan(
            protocol,
            "ama_bench",
            (
                (
                    JudgeMessage(
                        "user",
                        ama_scoring.judge_prompt(
                            question,
                            reference,
                            normalized,
                            task=str(metadata["task_description"]),
                            task_type=str(metadata["task_type"]),
                            episode_id=str(metadata["episode_id"]),
                        ),
                    ),
                ),
            ),
            max_tokens=2048,
            details={"prediction": normalized, "reference": reference},
        )
    if family == "icm-bench":
        prompt = SEMANTIC_EQUIVALENCE_PROMPT.format(
            question=question,
            ground_truth_answer=reference,
            agent_answer=prediction,
        )
        return JudgePlan(protocol, "icm", ((JudgeMessage("user", prompt),),), 8192)
    if family == "m3-bench":
        prompt = _M3_PROMPT.format(
            question=question,
            ground_truth_answer=reference,
            agent_answer=prediction,
        )
        return JudgePlan(
            protocol,
            "m3",
            (
                (
                    JudgeMessage("system", _M3_SYSTEM_PROMPT),
                    JudgeMessage("user", prompt),
                ),
            ),
            2048,
        )
    if family == "egotempo":
        prompt = _EGOTEMPO_PROMPT.format(question=question, answer=reference, prediction=prediction)
        return JudgePlan(protocol, "egotempo", ((JudgeMessage("user", prompt),),))
    if family == "memlens":
        question_type = str(metadata.get("question_type", ""))
        subtype = str(metadata.get("question_subtype") or "")
        task_key = get_task_key(question_type, subtype, reference)
        old_answer_value = metadata.get("old_answer")
        old_answer = old_answer_value if isinstance(old_answer_value, str) else None
        prompt = build_memlens_prompt(
            task_key,
            question,
            reference,
            _memlens_prediction(prediction),
            old_answer,
        )
        return JudgePlan(
            protocol,
            "memlens",
            ((JudgeMessage("user", prompt),),),
            2048,
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"task_key": task_key},
        )
    if family == "mm-lifelong":
        user = _MM_LIFELONG_USER_PROMPT.format(
            question=question, reference=reference, prediction=prediction
        )
        return JudgePlan(
            protocol,
            "mm_lifelong",
            ((JudgeMessage("system", _MM_LIFELONG_SYSTEM), JudgeMessage("user", user)),),
        )
    if family == "atm-bench":
        prompt = (
            _ATM_PROMPT.replace("{{question}}", question)
            .replace("{{answer}}", reference)
            .replace("{{prediction}}", prediction)
        )
        return JudgePlan(protocol, "atm", ((JudgeMessage("user", prompt),),), 600)
    if family == "longmemeval-v2":
        rule = longmemeval_v2_rule(str(metadata["eval_function"]))
        if longmemeval_v2_score(rule, prediction, reference) is not None:
            return None
        judge_system, judge_user = longmemeval_v2_messages(rule, question, reference, prediction)
        return JudgePlan(
            protocol,
            "longmemeval_v2",
            ((JudgeMessage("system", judge_system), JudgeMessage("user", judge_user)),),
            4096,
        )
    if family == "personamem-v3":
        return _personamem_plan(protocol, question, metadata, prediction)
    if family == "mem-gallery":
        prompt = (
            _GALLERY_PROMPT.replace("{{question}}", question)
            .replace("{{ground_truth}}", reference)
            .replace("{{model_output}}", prediction)
        )
        return JudgePlan(protocol, "gallery", ((JudgeMessage("user", prompt),),))
    raise AssertionError(f"unhandled judge family: {family}")


def parse_judge_response(  # noqa: C901 - mirrors seven incompatible upstream parsers
    plan: JudgePlan, response: str
) -> dict[str, float]:
    """Parse one response exactly as the pinned upstream evaluator does."""
    if plan.parser == "ama_bench":
        score = ama_scoring.parse_judgement(
            response, plan.details["prediction"], plan.details["reference"]
        )
        return {"llm_judge_score": score, "accuracy": float(score == 1.0)}
    if plan.parser == "locomo":
        payload = _json_object(response)
        label = str(payload.get("label", "")).strip().upper()
        if label not in {"CORRECT", "WRONG"}:
            raise ValueError("LoCoMo judge did not return CORRECT or WRONG")
        return {"llm_judge": float(label == "CORRECT")}
    if plan.parser == "m3":
        return {"accuracy": float("yes" in response.lower())}
    if plan.parser == "icm":
        return {"accuracy": float(response.strip().lower().rstrip(".") == "yes")}
    if plan.parser == "egotempo":
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if match is None:
            raise ValueError("EgoTempo judge did not return a dictionary")
        payload = ast.literal_eval(match.group())
        if not isinstance(payload, dict):
            raise ValueError("EgoTempo judge response is not a dictionary")
        raw_score = int(payload.get("score", 0))
        if not 0 <= raw_score <= 5:
            raise ValueError("EgoTempo judge score is outside 0-5")
        return {
            "accuracy": float(str(payload.get("pred", "")).strip().lower() == "correct"),
            "judge_score_0_5": float(raw_score),
        }
    if plan.parser == "memlens":
        match = re.search(r'"answer_score"\s*:\s*(\d+)', response)
        if match is None:
            match = re.search(r"deserves\s+(\d+)\s+point", response, re.IGNORECASE)
        if match is None:
            match = re.search(r"\[Score\]\s*:\s*(\d+)", response)
        return {"accuracy": float(min(int(match.group(1)), 1)) if match else 0.0}
    if plan.parser == "mm_lifelong":
        match = re.search(r"Final Score:\s*([0-5])", response, re.IGNORECASE)
        if match is None:
            match = re.search(r"\b([0-5])\b", response)
        if match is None:
            raise ValueError("MM-Lifelong judge did not return a 0-5 score")
        raw = int(match.group(1))
        mapped = 1.0 if raw in (4, 5) else 0.5 if raw == 3 else 0.0
        return {"answer_accuracy": mapped, "judge_score_0_5": float(raw)}
    if plan.parser == "atm":
        return {"accuracy": float(_atm_judge_accuracy(response))}
    if plan.parser == "longmemeval_v2":
        return {"accuracy": parse_longmemeval_v2(response)}
    if plan.parser == "personamem_v3":
        return _personamem_judge_scores(plan, response)
    if plan.parser == "gallery":
        score = _gallery_judge_score(response)
        return {"llm_judge": 0.0 if score < 0.25 else 0.5 if score < 0.75 else 1.0}
    if plan.parser == "mosaic":
        mosaic_score, valid = mosaic_scoring.parse_judge_score(response)
        return {"judge_score": mosaic_score, **({} if valid else {"judge_parse_error": 1.0})}
    raise AssertionError(f"unhandled judge parser: {plan.parser}")


def combine_judge_scores(
    plan: JudgePlan, scores: Sequence[Mapping[str, float]]
) -> dict[str, float]:
    """Use the best accepted gold answer, matching LoCoMo's multi-reference behavior."""
    if not scores:
        raise ValueError("judge plan produced no scores")
    if plan.parser == "locomo":
        if len(scores) != len(plan.candidate_metrics):
            raise ValueError("LoCoMo judge results do not match its answer candidates")
        selected = max(
            range(len(scores)),
            key=lambda index: (
                scores[index]["llm_judge"],
                plan.candidate_metrics[index]["token_f1"],
                plan.candidate_metrics[index]["bleu_1"],
            ),
        )
        return {**plan.candidate_metrics[selected], **scores[selected]}
    if (
        plan.parser == "personamem_v3"
        and plan.details.get("task_type") in pm3_tasks.CLUSTER_TASK_TYPES
    ):
        slots = cast(list[float | None], json.loads(plan.details["restraint_slots"]))
        indexes = cast(list[int], json.loads(plan.details["call_slots"]))
        for index, result in zip(indexes, scores, strict=True):
            slots[index] = result.get("restraint_score")
        return _personamem_headline(
            plan.details["task_type"],
            pm3_tasks.repetition_metrics(slots, int(plan.details["n_allowed_repetitions"])),
        )
    if len(scores) == 1:
        return dict(scores[0])
    names = set().union(*(score.keys() for score in scores))
    return {name: max(score[name] for score in scores if name in score) for name in names}


def _personamem_metric(task_type: str) -> tuple[str, float] | None:
    """Return one task type's headline metric and the divisor onto 0-1, if any.

    Keyed on EVAL.md's headline table rather than on the rubric's applicability
    map: upstream runs the rubric on several more task types but keeps its
    output as a diagnostic beside a headline it computes a different way (a
    leak-set composite, fatigue counters, a paired-row delta). A task type this
    returns `None` for is answered and reported without a `personamem_score`,
    so it stays out of the aggregate rather than entering it under a number
    upstream does not publish.
    """
    if task_type in _PERSONAMEM_HEADLINES:
        return _PERSONAMEM_HEADLINES[task_type]
    if task_type in pm3_proactive.TASK_TYPES:
        return "proactive_action_score", 1.0
    if task_type in pm3.RUBRIC_HEADLINE_TASK_TYPES:
        return _PERSONAMEM_RUBRIC_HEADLINE
    if task_type in RANKING_TASK_TYPES and task_type not in _PERSONAMEM_PAIRED_RANKING:
        return _PERSONAMEM_RANKING_HEADLINE
    return None


def _personamem_judged(metadata: Mapping[str, object]) -> bool:
    """Report whether one row's headline needs -- and can have -- a judge call.

    Ranking and geographic-shift rows are scored deterministically, so they
    are excluded here even though they do have a headline.
    """
    task_type = str(metadata.get("task_type", ""))
    return (
        task_type not in RANKING_TASK_TYPES
        and task_type not in {"local_recommendation_geo_shift", "active_mistake_prevention"}
        and _personamem_metric(task_type) is not None
    )


def _personamem_plan(
    protocol: str,
    question: str,
    metadata: Mapping[str, object],
    prediction: str,
) -> JudgePlan | None:
    task_type = str(metadata.get("task_type", ""))
    evidence = metadata.get("judge_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if task_type in pm3_tasks.CLUSTER_TASK_TYPES:
        return _personamem_cluster_plan(protocol, task_type, prediction, evidence, metadata)
    if task_type == "new_suggestions_chatbot":
        if pm3_tasks.suggestion_metrics(prediction, evidence)["hard_fail"]:
            return None
        return JudgePlan(
            protocol,
            "personamem_v3",
            (
                (
                    JudgeMessage(
                        "user",
                        judge_new_suggestions_chatbot_prompt(
                            prediction,
                            pm3._string(evidence, "gold_topic"),
                            pm3_tasks.strings(evidence, "gold_hashtags"),
                            pm3_tasks.strings(evidence, "fatigued_hashtags"),
                            pm3_tasks.strings(evidence, "leak_set_hashtags"),
                            pm3._string(evidence, "trigger_kind"),
                        ),
                    ),
                ),
            ),
            details={"task_type": task_type},
        )
    if task_type in pm3_proactive.TASK_TYPES:
        return JudgePlan(
            protocol,
            "personamem_v3",
            ((JudgeMessage("user", pm3_proactive.build_judge_prompt(prediction, evidence)),),),
            details={
                "task_type": task_type,
                "expected_behavior": pm3_proactive.expected_behavior(evidence),
                "prediction": prediction,
            },
        )
    if task_type in pm3.OWN_JUDGE_TASK_TYPES:
        prompt = pm3.build_own_judge_prompt(task_type, evidence, prediction)
    else:
        prompt = pm3.build_unified_rubric_prompt(
            task_type,
            _personamem_evidence(question, metadata),
            prediction,
            pm3.positive_dims(task_type),
            pm3.hard_rule_dims(task_type),
            pm3.penalty_dims(task_type),
            query_text=question,
        )
    return JudgePlan(
        protocol,
        "personamem_v3",
        ((JudgeMessage("user", prompt),),),
        details={"task_type": task_type},
    )


def _personamem_cluster_plan(
    protocol: str,
    task_type: str,
    prediction: str,
    evidence: Mapping[str, object],
    metadata: Mapping[str, object],
) -> JudgePlan | None:
    calls: list[tuple[JudgeMessage, ...]] = []
    slots: list[float | None] = []
    indexes = []
    tags = pm3_tasks.strings(
        evidence,
        "cluster_hashtags" if task_type.endswith("_recsys") else "target_hashtags",
    )
    for response in pm3_tasks.cluster_responses(prediction):
        if not response["is_target"]:
            continue
        text = (
            f"{response['title']} {response['caption']}".strip()
            if task_type.endswith("_recsys")
            else response["response"]
        )
        if not pm3_tasks.substantive(text):
            slots.append(0.0)
            continue
        indexes.append(len(slots))
        slots.append(None)
        prompt = _C1D_PREF_INVOCATION_JUDGE_PROMPT.format(
            target_pref=pm3._string(evidence, "target_pref")[:200],
            primary_category=pm3._string(evidence, "primary_category") or "(none)",
            target_hashtags=", ".join(f"#{tag.lstrip('#')}" for tag in tags[:8]) or "(none)",
            user_query=response["user_query"][:240],
            response=text[:800],
        )
        calls.append((JudgeMessage("user", prompt),))
    if not calls:
        return None
    return JudgePlan(
        protocol,
        "personamem_v3",
        tuple(calls),
        details={
            "task_type": task_type,
            "restraint_slots": json.dumps(slots),
            "call_slots": json.dumps(indexes),
            "n_allowed_repetitions": str(metadata.get("n_allowed_repetitions", 2)),
        },
    )


def _personamem_evidence(question: str, metadata: Mapping[str, object]) -> str:
    """Serialise the evidence block the unified rubric judge reads.

    The dataset's frozen instance payload contains the same source-A slices
    consumed by the released judge: relevant preferences, same-day avoids,
    privacy flags, contradictions, voice references, and friend records.
    """
    released = metadata.get("judge_evidence")
    released = released if isinstance(released, Mapping) else {}
    gt_slice = released.get("gt_slice")
    gt_slice = gt_slice if isinstance(gt_slice, Mapping) else {}
    top_preferences = released.get("top_k_relevant_prefs")
    if not isinstance(top_preferences, Sequence) or isinstance(top_preferences, str | bytes):
        top_preferences = ()
    held_out = released.get("held_out_preference") or metadata.get("groundtruth_preference")
    if not top_preferences and held_out:
        top_preferences = (held_out,)
    payload = {
        "query_text": question,
        "user_top_preferences": list(top_preferences),
        "user_negatives_nearby": list(
            gt_slice.get("avoid", ()) if isinstance(gt_slice.get("avoid", ()), Sequence) else ()
        ),
        "privacy_flagged_prefs": list(
            released.get("privacy_flagged_prefs", ())
            if isinstance(released.get("privacy_flagged_prefs", ()), Sequence)
            else ()
        ),
        "update_history_contradictions": list(
            released.get("update_history_contradictions", ())
            if isinstance(released.get("update_history_contradictions", ()), Sequence)
            else ()
        ),
        "user_style_refs": list(
            released.get("user_style_refs", ())
            if isinstance(released.get("user_style_refs", ()), Sequence)
            else ()
        ),
        "user_friends": list(
            released.get("user_friends", ())
            if isinstance(released.get("user_friends", ()), Sequence)
            else ()
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _personamem_judge_scores(plan: JudgePlan, response: str) -> dict[str, float]:
    task_type = plan.details.get("task_type", "")
    payload = pm3_json(response)
    if task_type in pm3_tasks.CLUSTER_TASK_TYPES:
        value = payload.get("restraint_score")
        if value is None and "invoked" in payload:
            value = 0.0 if payload["invoked"] else 10.0
        score = pm3._bounded(value, 0.0, 10.0)
        return {} if score is None else {"restraint_score": score}
    if task_type == "new_suggestions_chatbot":
        value = pm3._bounded(payload.get("alignment_score"), 0.0, 3.0)
        hard_fail = payload.get("hard_fail") is True
        scores = {
            "passed": float(value is not None and value >= 2.0 and not hard_fail),
            "hard_fail": float(hard_fail),
        }
        if value is not None:
            scores["alignment_score"] = value
    elif task_type in pm3_proactive.TASK_TYPES:
        scores = pm3_proactive.score(
            plan.details["prediction"],
            {"expected_behavior": plan.details["expected_behavior"]},
            payload,
        )
    elif task_type in pm3.OWN_JUDGE_TASK_TYPES:
        scores = pm3.parse_own_judge(task_type, payload)
    else:
        scores = pm3.score_unified_rubric(task_type, payload)
    return _personamem_headline(task_type, scores)


def _personamem_headline(task_type: str, scores: Mapping[str, float]) -> dict[str, float]:
    """Add the 0-1 value the upstream aggregator compares across task types."""
    result = dict(scores)
    metric = _personamem_metric(task_type)
    if metric is None:
        return result
    name, scale = metric
    if name in result:
        result["personamem_score"] = min(1.0, max(0.0, result[name] / scale))
    return result


def pm3_json(response: str) -> Mapping[str, object]:
    """Extract the judge's JSON object the way upstream's helper does."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
    candidate = match.group(1).strip() if match else response.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("PersonaMem-v3 judge did not return a JSON object") from None
        payload = json.loads(candidate[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("PersonaMem-v3 judge did not return a JSON object")
    return payload


def _personamem_local(  # noqa: C901 - direct dispatch keeps pinned task formulas auditable
    prediction: str, metadata: Mapping[str, object]
) -> dict[str, float]:
    """Score deterministic PersonaMem rows without a model call."""
    task_type = str(metadata.get("task_type", ""))
    if task_type in pm3_tasks.CLUSTER_TASK_TYPES:
        slots = [
            None
            if pm3_tasks.substantive(
                f"{row['title']} {row['caption']}".strip()
                if task_type.endswith("_recsys")
                else row["response"]
            )
            else 0.0
            for row in pm3_tasks.cluster_responses(prediction)
            if row["is_target"]
        ]
        return _personamem_headline(
            task_type,
            pm3_tasks.repetition_metrics(
                slots, cast(int, metadata.get("n_allowed_repetitions", 2))
            ),
        )
    if task_type in {"new_suggestions_chatbot", "active_mistake_prevention"}:
        evidence = metadata.get("judge_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("PersonaMem-v3 task scoring requires instance evidence")
        scores = (
            pm3_tasks.suggestion_metrics(prediction, evidence)
            if task_type == "new_suggestions_chatbot"
            else pm3_tasks.mistake_metrics(prediction, evidence)
        )
        return _personamem_headline(task_type, scores)
    if task_type in pm3_proactive.TASK_TYPES:
        evidence = metadata.get("judge_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("PersonaMem-v3 proactive scoring requires instance evidence")
        scores = pm3_proactive.score(prediction, evidence)
        # The upstream runner's judge-off fallback is a diagnostic; the official
        # headline needs the judge's justification and hard-rule verdicts.
        del scores["proactive_action_score"]
        return scores
    if task_type == "local_recommendation_geo_shift":
        evidence = metadata.get("judge_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("PersonaMem-v3 geo-shift scoring requires instance evidence")
        try:
            parsed = pm3_json(prediction)
        except ValueError:
            parsed = {}
        response = parsed.get("response") or prediction
        if not isinstance(response, str):
            raise ValueError("PersonaMem-v3 geo-shift response must be text")
        scores = pm3.compute_geo_shift_metrics(
            response,
            pm3._string(evidence, "current_city"),
            pm3._string(evidence, "prior_city"),
            pm3._string(evidence, "current_region"),
        )
        return _personamem_headline(task_type, scores)
    if task_type not in RANKING_TASK_TYPES:
        return {}
    return _personamem_ranking(prediction, metadata)


def _personamem_ranking(prediction: str, metadata: Mapping[str, object]) -> dict[str, float]:
    """Score the frozen candidate slate using the task's released gain rules."""
    task_type = str(metadata.get("task_type", ""))
    count = metadata.get("candidate_count")
    if not isinstance(count, int) or count <= 0:
        return {}
    positives = {
        index for index in _index_values(metadata, "positive_indexes") if 0 <= index < count
    }
    if not positives:
        return {}
    ranked = _parse_ranking(
        prediction, count, identity_fallback=task_type == "at_ai_directive_followup"
    )
    if task_type == "short_vs_long_term_lifecycle":
        return {
            "match_rate_at_3": len(positives & set(ranked[:3])) / 3.0,
            "matching_in_top1": float(bool(ranked) and ranked[0] in positives),
            "mrr_matching": pm3.mrr(ranked, positives),
        }
    negatives = {
        index for index in _index_values(metadata, "negative_indexes") if 0 <= index < count
    }
    # `_graded_ndcg_at_k` defaults the filler grade to +1; only the hidden-persona runner passes
    # 0 when its slate carries no hard negatives, and the result is clamped to [0, 1].
    filler_gain = 0.0 if task_type == "hidden_persona_recommendation" and not negatives else 1.0
    gains = [
        2.0 if index in positives else -2.0 if index in negatives else filler_gain
        for index in ranked
    ]
    ideal = sorted(gains, reverse=True)
    denominator = pm3.dcg(ideal[:5])
    ndcg_at_5 = min(1.0, max(0.0, pm3.dcg(gains[:5]) / denominator)) if denominator > 0 else 0.0
    scores = {
        "recall_at_1": pm3.recall_at_k(ranked, positives, 1),
        "recall_at_3": pm3.recall_at_k(ranked, positives, 3),
        "recall_at_5": pm3.recall_at_k(ranked, positives, 5),
        "hit_at_1": pm3.hit_at_k(ranked, positives, 1),
        "hit_at_3": pm3.hit_at_k(ranked, positives, 3),
        "mrr": pm3.mrr(ranked, positives),
        "ndcg_at_5": round(ndcg_at_5, 4),
        "negative_in_top1": float(bool(ranked) and ranked[0] in negatives),
        "negative_in_top3": float(any(index in negatives for index in ranked[:3])),
    }
    if task_type == "at_ai_directive_followup":
        scores.update(
            {
                "hit@1": scores["hit_at_1"],
                "recall@3": scores["recall_at_3"],
                "recall@5": scores["recall_at_5"],
            }
        )
    return _personamem_headline(task_type, scores)


def _parse_ranking(prediction: str, count: int, *, identity_fallback: bool) -> tuple[int, ...]:
    """Read a ranking out of an answer, in either published shape.

    The pinned release's own `example_response` is `Ranked indexes: [...]`; the
    evaluation repository's current prompt asks for a fenced-JSON
    `ranked_indices`. Both are accepted. The @ai directive runner replaces an
    invalid permutation with identity order; the other ranking runners score
    an absent or partial list as returned.
    """
    match = _RANKED_JSON.search(prediction) or _RANKED_INDEXES.search(prediction)
    if match is not None:
        try:
            ranked = [int(part) for part in match.group(1).replace(",", " ").split()]
        except ValueError:
            ranked = []
        if len(ranked) == len(set(ranked)) and all(0 <= index < count for index in ranked):
            return tuple(ranked)
    return tuple(range(count)) if identity_fallback else ()


def _index_values(metadata: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))


def _family_or_none(task: str) -> str | None:
    for family in sorted(_PROTOCOLS, key=len, reverse=True):
        if task == family or task.startswith(f"{family}-"):
            return family
    return None


def _model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _locomo_tokens(value: str) -> list[str]:
    return [token.lower() for token in _LOCOMO_TOKEN.findall(value)]


def _locomo_prediction(value: str) -> str:
    cleaned = str(value or "").strip()
    boxed = _last_boxed_value(cleaned)
    if boxed is not None:
        return boxed
    lowered = cleaned.lower()
    if "final answer:" in lowered:
        position = lowered.index("final answer:")
        cleaned = cleaned[position + len("final answer:") :].strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    return cleaned


def _last_boxed_value(value: str) -> str | None:
    result: str | None = None
    for match in re.finditer(r"\\box(?:ed)?\{", value):
        depth = 0
        start = match.end()
        for index in range(match.end() - 1, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    result = value[start:index].strip()
                    break
        else:
            result = value[start:].strip()
    return result


def _locomo_f1(prediction: str, reference: str) -> float:
    predicted, expected = _locomo_tokens(prediction), _locomo_tokens(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _bleu_1(
    prediction: str,
    reference: str,
    tokenizer: Callable[[str], list[str]],
) -> float:
    predicted, expected = tokenizer(prediction), tokenizer(reference)
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    if not precision:
        return 0.0
    penalty = (
        math.exp(1 - len(expected) / len(predicted)) if len(predicted) < len(expected) else 1.0
    )
    return penalty * precision


@lru_cache(maxsize=1)
def _porter() -> _Stemmer:
    try:
        from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("Mem-Gallery scoring requires sensemem[benchmarks]") from None
    return cast(_Stemmer, PorterStemmer())


def _mem_gallery_normalize(value: str) -> str:
    # a93959e1e978/evaluation.py uses regex, whose word boundaries include combining marks.
    import regex  # type: ignore[import-untyped]

    normalized = str(value).lower()
    normalized = regex.sub(r"(?<=\d)\.(?=\d)", "DOTPLACEHOLDER", normalized)
    normalized = normalized.replace("_", "UNDERSCOREPLACEHOLDER")
    normalized = regex.sub(r"\b(a|an|the|and)\b", " ", normalized)
    normalized = "".join(
        character if character not in string.punctuation else " " for character in normalized
    )
    return " ".join(
        normalized.replace("DOTPLACEHOLDER", ".").replace("UNDERSCOREPLACEHOLDER", "_").split()
    )


def _mem_gallery_tokens(value: str) -> list[str]:
    return _mem_gallery_normalize(value).split()


def _gallery_f1(
    prediction: str,
    reference: str,
) -> float:
    predicted = [_porter().stem(word) for word in _mem_gallery_tokens(prediction)]
    expected = [_porter().stem(word) for word in _mem_gallery_tokens(reference)]
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _gallery_bleu(
    prediction: str,
    reference: str,
    weights: tuple[float, float, float, float],
) -> float:
    try:
        from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
            SmoothingFunction,
            sentence_bleu,
        )
    except ImportError:
        raise RuntimeError("Mem-Gallery scoring requires sensemem[benchmarks]") from None
    predicted, expected = _mem_gallery_tokens(prediction), _mem_gallery_tokens(reference)
    if not predicted or not expected:
        return 0.0
    return float(
        sentence_bleu(
            [expected],
            predicted,
            weights=weights,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def _deduplicated_sources(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _metadata_values(metadata: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(str(item) for item in value)


def _atm_retrieval(
    metadata: Mapping[str, object], evidence_source_ids: Sequence[str]
) -> dict[str, float]:
    expected = set(_metadata_values(metadata, "evidence_ids"))
    if not expected:
        return {}
    retrieved = _deduplicated_sources(evidence_source_ids)
    scores: dict[str, float] = {}
    for cutoff in (1, 5, 10):
        selected = set(retrieved[:cutoff])
        scores[f"retrieval_recall@{cutoff}"] = len(selected & expected) / len(expected)
    at_gt = set(retrieved[: min(len(expected), len(retrieved))])
    scores["retrieval_recall@gt"] = len(at_gt & expected) / len(expected)
    scores["retrieval_hit@1"] = float(bool(retrieved) and retrieved[0] in expected)
    return scores


def _gallery_retrieval(
    metadata: Mapping[str, object], evidence_source_ids: Sequence[str]
) -> dict[str, float]:
    expected = set(_metadata_values(metadata, "clue_ids"))
    # The evaluator truncates the raw ranking before deduplication, and empty clues still
    # contribute zeros to its overall means (not an omitted row).
    retrieved = tuple(dict.fromkeys(evidence_source_ids[:10]))
    hits = sum(source in expected for source in retrieved)
    return {
        "retrieval_precision@10": hits / max(1, len(retrieved)),
        "retrieval_recall@10": hits / len(expected) if expected else 0.0,
        "retrieval_hit@10": float(hits > 0),
    }


def _memlens_prediction(value: str) -> str:
    text = value
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = re.sub(
        r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>",
        r"\1",
        text,
        flags=re.DOTALL,
    )
    text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    text = text.replace("[CONTENT_FILTER_REJECTED]", "").strip()
    if len(text) > 6_000 and len(text) > 3_000:
        text = "[...truncated earlier reasoning...]\n" + text[int(len(text) * 0.7) :]
    return text


def _json_object(value: str) -> Mapping[str, object]:
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response does not contain a JSON object")
    payload = json.loads(value[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge response JSON must be an object")
    return payload


def _atm_judge_accuracy(value: str) -> float:
    try:
        payload = _json_object(value)
    except (ValueError, json.JSONDecodeError):
        payload = {}
    if "accuracy" in payload:
        # run_llm_judge converts the verdict before summarize_atm coerces scores.
        # Numeric partial credit belongs to deterministic list scores, not judge verdicts.
        return float(str(payload["accuracy"]).lower() == "true")
    lowered = value.lower()
    if '"accuracy": true' in lowered or '"accuracy":true' in lowered:
        return 1.0
    if '"accuracy": false' in lowered or '"accuracy":false' in lowered:
        return 0.0
    return float(any(word in lowered for word in ("accurate", "correct", "matches", "true", "yes")))


def _gallery_judge_score(value: str) -> float:
    try:
        payload = _json_object(value)
        raw_score = payload.get("score")
        if isinstance(raw_score, bool):
            raise ValueError
        return float(cast(str | int | float, raw_score))
    except (TypeError, ValueError, json.JSONDecodeError):
        for pattern in (
            r'"score"\s*:\s*([0-9.]+)',
            r"score\s*:\s*([0-9.]+)",
        ):
            match = re.search(pattern, value, re.IGNORECASE)
            if match is not None:
                return float(match.group(1))
    raise ValueError("Mem-Gallery judge did not return a numeric score")
