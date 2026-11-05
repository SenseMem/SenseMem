"""LongMemEval-V2 scoring adapted from the Apache-2.0 official evaluator.

Source: xiaowu0162/LongMemEval-V2@2cc8c540bdb87fe6761629b585e727e1c4704520,
evaluation/qa_eval_metrics.py. See NOTICE.md for attribution. Transport is owned
by the benchmark runner; only the six released evaluation functions are accepted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

SCORER_VERSION = "longmemeval_v2_2cc8c540bdb8_v1"
JUDGE_FUNCTIONS = frozenset({"llm_abstention_checker", "llm_gotchas_checker"})
_FUNCTIONS = JUDGE_FUNCTIONS | {
    "norm_phrase_set_match",
    "norm_phrase_set_match_ordered",
    "mc_choice_match",
    "mc_choice_set_match",
}
_FILLER_WORDS = frozenset(
    {
        "AND",
        "ANSWER",
        "ANSWERS",
        "CHOICE",
        "CHOICES",
        "FINAL",
        "LETTER",
        "LETTERS",
        "OPTION",
        "OPTIONS",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationRule:
    name: str
    lower: bool = True
    normalize_hyphen: bool = True
    strip_punct: bool = True
    require_non_empty: bool = True
    separators: tuple[str, ...] = (",", ";")


def parse_eval_spec(spec: str) -> EvaluationRule:
    """Validate the release's declarative spec without executing dataset code."""
    name, *options = (part.strip() for part in spec.split("|"))
    if name not in _FUNCTIONS:
        raise ValueError(f"unknown LongMemEval-V2 evaluation function: {name}")
    flags: dict[str, bool] = {}
    separators: tuple[str, ...] = (",", ";")
    seen: set[str] = set()
    for option in options:
        key, separator, value = option.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key in seen:
            raise ValueError("invalid or duplicate LongMemEval-V2 evaluation option")
        seen.add(key)
        if key == "separators":
            separators = tuple(char for char in value if not char.isspace())
        elif key in {"lower", "normalize_hyphen", "strip_punct", "require_non_empty"}:
            if value.lower() not in {"true", "false"}:
                raise ValueError("LongMemEval-V2 evaluation flags must be boolean")
            flags[key] = value.lower() == "true"
        else:
            raise ValueError(f"unknown LongMemEval-V2 evaluation option: {key}")
    return EvaluationRule(name, separators=separators, **flags)


def extract_boxed_answer(text: str) -> str:
    """Read the last boxed answer, including balanced nested braces, as upstream does."""
    marker = "\\boxed{"
    position = text.rfind(marker)
    if position == -1:
        return text.strip()
    depth = 1
    result = []
    for char in text[position + len(marker) :]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        result.append(char)
    return "".join(result).strip() or text.strip()


def deterministic_score(rule: EvaluationRule, prediction: str, answer: str) -> float | None:
    """Return a binary score, or None when the question needs an LLM judge."""
    prediction = extract_boxed_answer(prediction)
    if prediction.lower() == "unknown" or (
        rule.require_non_empty and (not prediction or not answer.strip())
    ):
        return 0.0
    if rule.name in JUDGE_FUNCTIONS:
        return None
    if rule.name == "mc_choice_match":
        boxed = re.search(r"\\boxed\{([^}]*)\}", prediction.lower())
        candidate = boxed.group(1) if boxed else prediction
        cleaned = re.sub(r"\b(choice|option)\b", "", candidate, flags=re.IGNORECASE)
        return float(cleaned.replace(".", "").strip().upper() == answer.strip().upper())
    if rule.name == "mc_choice_set_match":

        def letters(text: str) -> set[str]:
            return {
                letter
                for chunk in re.findall(r"[A-Z]+", text.upper())
                if chunk not in _FILLER_WORDS
                for letter in chunk
            }

        predicted, expected = letters(prediction), letters(answer)
        return float(
            predicted == expected and (not rule.require_non_empty or bool(predicted and expected))
        )

    return _phrase_score(rule, prediction, answer)


def _phrase_score(rule: EvaluationRule, prediction: str, answer: str) -> float:
    def normalize(text: str) -> str:
        if rule.lower:
            text = text.lower()
        if rule.normalize_hyphen:
            text = text.replace("-", " ").replace("_", " ")
        text = re.sub(r"[,;]", " ", text)
        if rule.strip_punct:
            text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    normalized = normalize(prediction)
    parts = (
        re.split("|".join(re.escape(sep) for sep in rule.separators), answer)
        if rule.separators
        else [answer]
    )
    phrases = [phrase for part in parts if (phrase := normalize(part))]
    if rule.require_non_empty and (not normalized or not phrases):
        return 0.0
    start = 0
    for phrase in phrases:
        match = re.search(rf"\b{re.escape(phrase)}\b", normalized[start:])
        if match is None:
            return 0.0
        if rule.name == "norm_phrase_set_match_ordered":
            start += match.end()
    return 1.0


def judge_messages(
    rule: EvaluationRule, question: str, answer: str, prediction: str
) -> tuple[str, str]:
    """Reproduce the two released judge rubrics, keeping full and final answers."""
    final = extract_boxed_answer(prediction)
    if rule.name == "llm_abstention_checker":
        system = (
            "You are a strict grader for flawed-premise (abstention) questions. "
            "Judge whether a model answer correctly identifies that the question premise is wrong, "
            "consistent with the reference answer. "
            "If the model follows the flawed premise and gives a concrete answer under that premise, "
            "it must be graded 0. "
            "If the model's final answer is just UNKNOWN / cannot determine without identifying the flaw, grade 0. "
            "If the model is contradictory (both rejects premise and also gives a concrete premise-following answer), grade 0. "
            "Paraphrases are allowed when they preserve the same core flaw described by the reference answer."
        )
        user = (
            "Evaluate whether the model correctly handled a flawed-premise question.\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer (the expected flaw-aware conclusion):\n{answer}\n\n"
            f"Model full response:\n{prediction.strip()}\n\n"
            f"Model extracted final answer:\n{final}\n\n"
            "Scoring rubric:\n"
            "- Label 1 only if the model clearly identifies the flawed premise and reaches a conclusion "
            "equivalent to the reference answer.\n"
            "- Label 1 also if the model explicitly says it lacks access to the user's specific "
            "live environment/instance/configuration and therefore cannot verify, provided it does not "
            "give a concrete premise-following answer.\n"
            "- Label 0 if the model follows the flawed premise and gives a concrete answer under that premise.\n"
            "- Label 0 for generic UNKNOWN/insufficient-info replies that do not identify a flaw and do not "
            "make the explicit environment-access limitation clear.\n"
            "- Label 0 if contradictory.\n\n"
            'Output JSON only:\n{"label": 0 or 1, "reason": "short rationale"}'
        )
    elif rule.name == "llm_gotchas_checker":
        system = (
            "You are a strict grader for gotchas-style insight questions. "
            "The reference answer describes the key insight(s). "
            "Grade 1 if the model response includes at least one correct insight point from the reference answer "
            "(paraphrase allowed), and does not contradict any reference point. "
            "If the model's direction is wrong, or it contains contradictions against any reference point, grade 0. "
            "If the model gives multiple points, partial coverage is enough for 1 as long as no contradictions appear."
        )
        user = (
            "Evaluate whether the model answer captures the gotcha insight.\n\n"
            f"Question:\n{question}\n\n"
            f"Reference answer (insight points):\n{answer}\n\n"
            f"Model full response:\n{prediction.strip()}\n\n"
            f"Model extracted final answer:\n{final}\n\n"
            "Scoring rubric:\n"
            "- Label 1 if the model includes at least one correct insight point from the reference answer "
            "(paraphrase acceptable), and does not contradict any reference point.\n"
            "- Label 1 even if only part of a multi-point reference answer is covered, as long as there is "
            "no contradiction.\n"
            "- Label 0 if direction is wrong (suggests opposite action/cause), even if some wording overlaps.\n"
            "- Label 0 if any point in the model response contradicts any reference point.\n"
            "- Label 0 if the response is irrelevant or generic without insight.\n\n"
            'Output JSON only:\n{"label": 0 or 1, "reason": "short rationale"}'
        )
    else:
        raise ValueError("LongMemEval-V2 deterministic questions do not use a judge")
    return system, user


def parse_judgement(response: str) -> float:
    """Parse the official binary JSON label and its fallback spellings."""
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            pass
        else:
            if not isinstance(payload, dict):
                raise ValueError("LongMemEval-V2 judge must return an object")
            label = payload.get("label")
            if label in (0, 1, "0", "1"):
                return float(label)
    for pattern in (r'"label"\s*:\s*([01])', r"'label'\s*:\s*([01])", r"\blabel\b\s*[:=]\s*([01])"):
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError("LongMemEval-V2 judge did not return a binary label")
