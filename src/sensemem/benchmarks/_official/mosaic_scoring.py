"""MOSAIC prompts and scoring from HKUST-KnowComp/MOSAIC@82dc5ccf8063.

Sources: evaluation/judge/eval_judge.py and evaluation/rag/eval_rag_mm.py.
Upstream declares no license at this revision; see NOTICE.md. No dataset is redistributed.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence

SCORE_BANDS = """Score bands (how well does the predicted answer match the expected answer):

1.0 — PERFECT. Semantically identical to the expected answer or an alias.
     Only differs in formatting, capitalization, or obvious paraphrase.
     Example: expected="8:15 AM", predicted="8:15 in the morning" → 1.0
     Example: expected="mental health", predicted="mental health awareness" → 1.0

0.7 — MOSTLY CORRECT. Right answer present but with extra context, minor
     redundancy, or slightly imprecise wording that doesn't change meaning.
     Example: expected="a snowboard", predicted="A snowboard with multicolored
     graphic design" → 0.7
     Example: expected="from $12 to $85", predicted="from $12.00 to $85.00" → 1.0

0.5 — PARTIAL. Gets the general category or one part of a multi-part answer
     right, but misses specifics or a sub-part.
     Example: expected="from $12 to $85", predicted="$85" → 0.5
     Example: expected="from pale yellow to soft sage green",
              predicted="sage green" → 0.5

0.3 — ATTEMPTED. Shows an attempt to answer but the core fact is wrong or
     mostly incorrect. May have a faint connection to the question.
     Example: expected="8:15 AM", predicted="around noon" → 0.3

0.0 — WRONG. Completely incorrect, irrelevant, refused to answer
     ("I don't know" on an answerable question), empty, or error.
     Example: expected="8:15 AM", predicted="I don't know" → 0.0"""

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for a memory benchmark. Your task is to "
    "score how well a predicted answer matches the expected answer.\n\n"
    f"{SCORE_BANDS}\n\n"
    "Important:\n"
    "- Judge semantic equivalence, not verbatim string match.\n"
    "- Accepted aliases count as correct answers.\n"
    '- For unanswerable questions, "I don\'t know" or similar refusal '
    "should score 1.0; any other answer should score 0.0.\n"
    "- If the predicted answer is empty, error, or nonsensical → 0.0.\n\n"
    "Respond with a JSON object:\n"
    '{"reasoning": "brief explanation of why this score", "score": <float>}'
)


def judge_user_prompt(
    question: str, references: Sequence[str], prediction: str, answerability: str
) -> str:
    aliases = ", ".join(f'"{alias}"' for alias in references[1:]) if references[1:] else "(none)"
    return (
        f"Question: {question}\n"
        f"Answerability: {answerability}\n\n"
        f"Expected answer: {references[0]}\n"
        f"Accepted aliases: {aliases}\n\n"
        f"Predicted answer: {prediction}\n\n"
        "Score this answer."
    )


def answer_system_prompt(question_type: str) -> str:
    update_rule = (
        "If this is a knowledge_update question (what changed / was updated), "
        'answer with the full transition as "from <old> to <new>". '
        "Do not answer with only the final/current state when both the earlier "
        "and later values appear in the evidence."
        if question_type.strip().lower() == "knowledge_update"
        else "If the evidence contains updates or contradictions about a user's "
        "state, preferences, or attributes, the most recent information takes "
        "precedence."
    )
    return (
        "You are an evaluation assistant. Your task is to answer questions "
        "about a user based on retrieved conversation history and attached media.\n\n"
        "Answer rules:\n"
        "1. Answer using ONLY the retrieved evidence provided. Do not use "
        "outside knowledge.\n"
        "2. The evidence may include native multimodal artefacts (images, "
        "audio recordings, video clips). Examine each artefact carefully — "
        "they may contain visual or auditory details that the surrounding "
        "text does not describe.\n"
        "3. Your answer should be as short as possible — typically a single "
        "phrase, number, name, or date.\n"
        "4. Do not include reasoning, explanations, or hedging in your final "
        "answer. Do not describe the multimodal artefacts in your answer; "
        "extract the requested fact directly.\n"
        "5. If the retrieved evidence does not contain enough information to "
        "answer the question, respond with exactly: I don't know.\n"
        f"6. {update_rule}\n"
        '7. For temporal references in the question ("currently", "now", '
        '"recently"), interpret them relative to the question\'s date.\n'
        "8. Examples of correct answer format:\n"
        '   - Question: "What was the work start time?" → Answer: "8:15 AM"\n'
        '   - Question: "What did the charity race raise awareness for?" → Answer: "mental health"\n'
        '   - Question: "How many books did I read this year?" → Answer: "12"\n'
    )


def parse_judge_score(response: str) -> tuple[float, bool]:
    """Return the official parsed score and whether parsing/coercion succeeded."""
    from json_repair import loads

    text = response.strip()
    match = re.match(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        try:
            value: object = json.loads(text)
        except ValueError:
            value = loads(text)
        if not isinstance(value, dict):
            return 0.0, False
        score = float(value.get("score", 0))
    except (ValueError, TypeError):
        return 0.0, False
    # NaN is not a valid benchmark score even though Python's JSON parser accepts it.
    return (min(1.0, max(0.0, score)), True) if math.isfinite(score) else (0.0, False)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    for prefix in ("answer:", "the answer is:", "response:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def text_metrics(prediction: str, references: Sequence[str]) -> dict[str, float]:
    """Official best-reference EM, count-weighted F1 and smoothed BLEU-1."""
    from nltk.translate.bleu_score import (  # type: ignore[import-untyped]
        SmoothingFunction,
        sentence_bleu,
    )

    predicted = _normalize(prediction)
    tokens = predicted.split()
    best_f1 = best_bleu = 0.0
    for reference in references:
        expected = _normalize(reference).split()
        common = sum((Counter(tokens) & Counter(expected)).values())
        if not tokens or not expected:
            f1 = float(tokens == expected)
        elif not common:
            f1 = 0.0
        else:
            precision = common / len(tokens)
            recall = common / len(expected)
            f1 = 2 * precision * recall / (precision + recall)
        bleu = (
            float(
                sentence_bleu(
                    [expected],
                    tokens,
                    weights=(1.0,),
                    smoothing_function=SmoothingFunction().method1,
                )
            )
            if tokens and expected
            else float(tokens == expected)
        )
        best_f1 = max(best_f1, f1)
        best_bleu = max(best_bleu, bleu)
    return {
        "em": float(any(predicted == _normalize(reference) for reference in references)),
        "f1": round(best_f1, 4),
        "bleu1": round(best_bleu, 4),
    }
